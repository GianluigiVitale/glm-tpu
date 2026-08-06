#!/usr/bin/env bash
# Protected production-shaped TPU-v4 proof for greenfield FP8 projection kernels.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
KERNEL=${GLM_GREENFIELD_FP8_MATMUL_KERNEL:-single_up}
TAG=${GLM_GREENFIELD_FP8_MATMUL_TAG:-greenfield_fp8_${KERNEL}_$(date -u +%Y%m%dT%H%M%S%NZ)}
WARMUP=${GLM_GREENFIELD_FP8_MATMUL_WARMUP:-200}
ITERATIONS=${GLM_GREENFIELD_FP8_MATMUL_ITERATIONS:-1000}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing FP8 kernel run outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing FP8 kernel run from a dirty worktree" >&2
  exit 2
}
[[ $KERNEL == single_up || $KERNEL == up_gate ]] || {
  echo "FP8 kernel must be single_up or up_gate" >&2
  exit 2
}
[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || {
  echo "protected FP8 kernel run requires warmup>=200" >&2
  exit 2
}
[[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || {
  echo "protected FP8 kernel run requires iterations>=1000" >&2
  exit 2
}
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[fp8-matmul $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[m]icrobench_fp8_matmul[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN kernel=$KERNEL warmup=$WARMUP iterations=$ITERATIONS"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "compiling and timing GLM expert $KERNEL projection on TPU v4"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/microbench_fp8_matmul.py \
      --expected-code-hash "$PIN" \
      --output "$RUN_DIR/runner.json" \
      --hlo-output "$RUN_DIR/hlo/fp8_matmul.optimized_hlo.txt" \
      --kernel "$KERNEL" \
      --warmup "$WARMUP" \
      --iterations "$ITERATIONS"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("runner status/code identity failed")
if not runner["hlo"]["contract"]["passed"]:
    raise SystemExit("Pallas custom-call/full-overlay HLO contract failed")
if not runner["comparison"]["passed"]:
    raise SystemExit("Pallas/reference correctness failed")
if not runner["profiler_free_timing"]:
    raise SystemExit("kernel wall distribution is not profiler-free")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model=f"zai-org/GLM-5.2-FP8:greenfield-fp8-{runner['kernel']}-kernel",
    revision="runtime-u8-e4m3fn-block128",
    env={
        "GLM_ENGINE": "greenfield_fp8_matmul",
        "greenfield_code_hash": pin,
        "hlo_sha256": runner["hlo"]["sha256"],
        "device_kind": runner["device_kind"],
        "kernel": runner["kernel"],
    },
    note=(
        "Protected production-shaped Pallas FP8 projection microbenchmark: "
        + runner["kernel"]
    ),
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    conn,
    run_id,
    benchmark=f"greenfield_fp8_{runner['kernel']}",
    item_id="m8_k6144_n2048",
    prompt=(
        "Raw-U8 E4M3FN 128x128 block-scaled expert projection: "
        + runner["kernel"]
    ),
    gold="Bounded exact-fallback output and one compact Pallas custom call.",
    raw_output=json.dumps(runner, sort_keys=True),
    extracted=str(runner["checksum"]),
    correct=True,
    score=1.0,
    latency_ms=runner["latency"]["p50_ms"],
)
pv.finalize(
    conn,
    run_id,
    benchmark=f"greenfield_fp8_{runner['kernel']}",
    metric="contract_valid",
    value=1.0,
    note="Standalone kernel microbenchmark; not layer latency or token throughput.",
)
conn.close()

summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "runner": runner,
    "claim_scope": "standalone profiler-free kernel wall; no token-rate claim",
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
check = sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0]
if check != "ok":
    raise SystemExit(f"results DB snapshot integrity failed: {check}")
print(f"FP8_MATMUL_PROOF_VALID db_run={run_id}")
PY

(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db census_pre.txt
) >"$RUN_DIR/evidence.sha256"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
sha256sum "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"
touch "$RUN_DIR/SUCCESS"

gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber \
  "$RUN_DIR/runner.json" "$RUN_DIR/runner.log" "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post.txt" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS DB=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
