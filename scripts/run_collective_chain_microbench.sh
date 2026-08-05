#!/bin/bash
# Protected no-model 32-chip discriminator for the 75-stage MoE collective
# dependency chain.  This script never creates compute; it uses only the
# existing db-v4-64-od pod and archives only to the approved same-region
# bucket.
set -euo pipefail

ZONE=us-central2-b
POD=db-v4-64-od
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIN=$(git -C "$ROOT" rev-parse HEAD)
TAG="collective_chain_mb_$(date -u +%Y%m%dT%H%M%S%NZ)"
RUN_DIR="$HOME/glm-run/$TAG"
GCS_RUN="gs://driftbench-dsv4-uc/results/$TAG"
COORDINATOR="$(hostname -i | awk '{print $1}'):8476"
TIMEOUT_S="${GLM_MB_TIMEOUT_S:-1800}"
mkdir -p "$RUN_DIR"

say() {
  echo "[collective-chain $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_8_unique_markers() {
  local file="$1" marker="$2" lines unique
  lines=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l)
  unique=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" |
    sort -u | wc -l)
  [ "$lines" -eq 8 ] && [ "$unique" -eq 8 ]
}

strict_census() {
  local label="$1" out="$RUN_DIR/census_${label}.txt"
  # Bracketed patterns keep the remote census shell from matching itself.
  # shellcheck disable=SC2016
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; work=$(pgrep -af "[c]ollective_chain_microbench[.]py|VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[r]un_bench[.]py" 2>/dev/null || true); ray=$(pgrep -x raylet 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$work" ] || [ -n "$ray" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$work" ] && echo "$work"; [ -n "$ray" ] && echo "raylet: $ray"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" CENSUS_OK
}

pin_census() {
  local out="$RUN_DIR/pin_census.txt"
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='cd ~/glm-tpu || exit 91; head=$(git rev-parse HEAD); dirty=$(git status --porcelain --untracked-files=no | wc -l); if [ "$head" = '"$PIN"' ] && [ "$dirty" -eq 0 ]; then echo "PIN_OK $(hostname) $head"; else echo "PIN_BAD $(hostname) head=$head dirty=$dirty"; fi' \
    > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" PIN_OK
}

cleanup_tagged() {
  # Kill only benchmark processes that carry this run's nonce.  This is a
  # recovery path for an SSH/timeout failure, never a broad pod reset.
  # shellcheck disable=SC2016
  local cmd='for p in $(pgrep -f "[c]ollective_chain_microbench[.]py" 2>/dev/null || true); do if tr "\0" "\n" < "/proc/$p/environ" 2>/dev/null | grep -qx "GLM_MB_TAG='"$TAG"'"; then kill -TERM "$p" 2>/dev/null || true; fi; done; sleep 2; for p in $(pgrep -f "[c]ollective_chain_microbench[.]py" 2>/dev/null || true); do if tr "\0" "\n" < "/proc/$p/environ" 2>/dev/null | grep -qx "GLM_MB_TAG='"$TAG"'"; then kill -KILL "$p" 2>/dev/null || true; fi; done'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$RUN_DIR/recovery_cleanup.txt" 2>&1 || true
}

validate_fleet_log() {
  "$HOME/vllm-env/bin/python" - "$RUN_DIR/fleet.log" \
    "$RUN_DIR/fleet_validation.json" <<'PY'
import json
import re
import statistics
import sys

path, output = sys.argv[1:]
variants = {}
hosts = {}
for line in open(path, encoding="utf-8", errors="replace"):
    if "GLM_MB_VARIANT " in line:
        payload = json.loads(line.split("GLM_MB_VARIANT ", 1)[1])
        key = (payload["process_id"], payload["kind"])
        assert key not in variants, key
        variants[key] = payload
    match = re.search(r"GLM_MB_HOST_OK\s+(\S+)\s+process_id=(\d+)", line)
    if match:
        host, process_id = match.group(1), int(match.group(2))
        assert process_id not in hosts, process_id
        hosts[process_id] = host

expected_kinds = {"full32", "dcp8", "model4", "sequential8x4"}
assert set(hosts) == set(range(8)), hosts
assert set(variants) == {
    (process_id, kind)
    for process_id in range(8)
    for kind in expected_kinds
}, sorted(variants)
for kind in expected_kinds:
    counts = {variants[(process_id, kind)]["logical_all_reduce"]
              for process_id in range(8)}
    assert counts == ({150} if kind == "sequential8x4" else {75}), (kind, counts)
    outputs = {variants[(process_id, kind)]["output_first"]
               for process_id in range(8)}
    assert len(outputs) == 1, (kind, outputs)

fleet = {}
for kind in sorted(expected_kinds):
    medians = [variants[(process_id, kind)]["median_ms"]
               for process_id in range(8)]
    assert min(medians) > 0, (kind, medians)
    # Every call is a global SPMD program, so host-side blocking latencies
    # should agree closely. A wide spread makes the timing non-promotable.
    assert max(medians) / min(medians) < 1.25, (kind, medians)
    fleet[kind] = {
        "host_medians_ms": medians,
        "median_of_host_medians_ms": statistics.median(medians),
        "min_host_median_ms": min(medians),
        "max_host_median_ms": max(medians),
        "max_over_min": max(medians) / min(medians),
        "output_first": next(iter({
            variants[(process_id, kind)]["output_first"]
            for process_id in range(8)
        })),
    }
with open(output, "w") as handle:
    json.dump({
        "hosts_by_process_id": hosts,
        "variants": fleet,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("FLEET_LOG_VALID hosts=8 variants=32")
PY
}

validate_result() {
  "$HOME/vllm-env/bin/python" - "$RUN_DIR/result.json" \
    "$RUN_DIR/fleet_validation.json" "$PIN" <<'PY'
import json
import sys

path, fleet_path, pin = sys.argv[1:]
data = json.load(open(path))
fleet = json.load(open(fleet_path))
assert data["process_id"] == 0
assert data["process_count"] == 8
assert data["local_device_count"] == 4
assert data["global_device_count"] == 32
assert data["mesh_shape"] == [1, 1, 1, 1, 4, 8]
assert len(data["mesh_devices"]) == 32
variants = {row["kind"]: row for row in data["variants"]}
assert set(variants) == {"full32", "dcp8", "model4", "sequential8x4"}
for name, row in variants.items():
    expected = 150 if name == "sequential8x4" else 75
    assert row["expected_collectives"] == expected
    assert row["hlo_counts"]["logical_all_reduce"] == expected
    assert row["latency_ms"]["median"] > 0
    assert row["samples"] == 40 and row["warmup"] == 5
    assert row["output_first"] == row["output_first"]
data["harness_git"] = pin
data["fleet_validation"] = fleet
fleet_medians = {
    name: row["median_of_host_medians_ms"]
    for name, row in fleet["variants"].items()
}
data["fleet_speedup_vs_full32"] = {
    name: fleet_medians["full32"] / latency
    for name, latency in fleet_medians.items()
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("RESULT_VALID variants=4 physical_mesh=32")
PY
}

exec 9>"$HOME/glm-run/.glm_pod_workload.lock"
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the lock"
  exit 1
}

local_dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l)
if [ "$local_dirty" -ne 0 ]; then
  say "ABORT: local harness checkout has tracked changes"
  exit 1
fi

say "RUN_DIR=$RUN_DIR pin=$PIN coordinator=$COORDINATOR"
strict_census pre || {
  say "ABORT: preflight census is not clean on all eight hosts"
  exit 1
}
pin_census || {
  say "ABORT: harness pin/cleanliness mismatch across the fleet"
  exit 1
}

REMOTE_CMD="cd ~/glm-tpu || exit 91; process_id=\${HOSTNAME##*-}; mkdir -p '$RUN_DIR'; export GLM_MB_TAG='$TAG' GLM_MB_COORDINATOR='$COORDINATOR' GLM_MB_PROCESS_ID=\$process_id GLM_MB_RESULT_PATH='$RUN_DIR/result.json' JAX_PLATFORMS=tpu JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false'; timeout --signal=TERM --kill-after=30s '$TIMEOUT_S' ~/vllm-env/bin/python -u scripts/collective_chain_microbench.py"

say "launching eight-process no-model benchmark"
if ! gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$REMOTE_CMD" > "$RUN_DIR/fleet.log" 2>&1; then
  say "ABORT: fleet benchmark command failed"
  cleanup_tagged
  strict_census failure_cleanup || true
  exit 1
fi

validate_fleet_log | tee -a "$RUN_DIR/orchestrator.log"
validate_result | tee -a "$RUN_DIR/orchestrator.log"
strict_census post || {
  say "ABORT: post-run census is not clean"
  cleanup_tagged
  strict_census failure_cleanup || true
  exit 1
}

{
  echo "tag=$TAG"
  echo "harness_git=$PIN"
  echo "pod=$POD"
  echo "zone=$ZONE"
  echo "coordinator=$COORDINATOR"
  echo "gcs=$GCS_RUN"
} > "$RUN_DIR/config.txt"
touch "$RUN_DIR/SUCCESS"
(
  cd "$RUN_DIR"
  manifest_tmp=$(mktemp)
  trap 'rm -f "$manifest_tmp"' EXIT
  find . -type f ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum > "$manifest_tmp"
  mv "$manifest_tmp" SHA256SUMS
  trap - EXIT
)

ARCHIVE_TMP="/tmp/${TAG}_archive.log"
if ! gcloud storage rsync -r "$RUN_DIR" "$GCS_RUN" > "$ARCHIVE_TMP" 2>&1; then
  say "ABORT: durable archive failed; local evidence retained"
  exit 1
fi
cp "$ARCHIVE_TMP" "$RUN_DIR/archive.log"
gcloud storage cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/archive.log" \
  "$RUN_DIR/SUCCESS" "$GCS_RUN/" >> "$ARCHIVE_TMP" 2>&1 || {
    say "ABORT: final archive confirmation failed"
    exit 1
  }
say "MICROBENCH VALID: $RUN_DIR archive=$GCS_RUN"
