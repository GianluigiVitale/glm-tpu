#!/usr/bin/env bash
# Protected model-free replay of DB550 dense partials through one M32 StrategyND reduction.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z
readonly SOURCE_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/dense_partials/8k/$SOURCE_TAG
readonly SOURCE_NPZ=$SOURCE_DIR/dense_partials_capture/dense_partials.npz
readonly SOURCE_CAPTURE=$SOURCE_DIR/dense_partials_capture/capture.json
readonly SOURCE_COMPARISON=$SOURCE_DIR/dense_partials_capture/comparison.json
readonly SOURCE_SUCCESS=$SOURCE_DIR/SUCCESS
readonly SOURCE_REMOTE_OBJECTS=$SOURCE_DIR/remote_objects.json
readonly SOURCE_NPZ_SHA=e5977248acbe7582db351178b3fc823c6f87db46f8b6143b763319a6b299582c
readonly SOURCE_CAPTURE_SHA=9b6a4a6d608a0e88f9fbda3bc68be77e2ee43a0dda35a361a32c76f22fda01e6
readonly SOURCE_COMPARISON_SHA=92707ccac80a337bcae0c527148fc9133383067d25add36eb0b36f7e7c9198de
readonly SOURCE_SUCCESS_SHA=9605aa5c0f76fd9a5ec9b8e78b1aa720111cbc0633d7b962834004537f321b23
readonly SOURCE_REMOTE_OBJECTS_SHA=663adbf1a11c32bbfc28b1030a0c329080d29fcf96fe4a2d77169e64e8858a05
readonly DB533_TAG=greenfield_collective_association_20260811T213152133863450Z
readonly DB533_DIR=/home/gianl/glm-run/$DB533_TAG
readonly DB533_REMOTE=$APPROVED_BUCKET/results/$DB533_TAG
readonly DB533_ANALYSIS=$DB533_DIR/association/analysis.json
readonly DB533_SUMMARY=$DB533_DIR/summary.json
readonly DB533_SUCCESS=$DB533_DIR/SUCCESS
readonly DB533_HLO_CONTRACT=$DB533_DIR/hlo/strategy_nd_association_bfloat16_32x6144.hlo_contract.json
readonly DB533_ANALYSIS_SHA=e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108
readonly DB533_SUMMARY_SHA=3ca82073f69fbe56526e1765594c7e8e9a738c73eb2a62b2df6d9a0c4d3136b7
readonly DB533_SUCCESS_SHA=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
readonly DB533_HLO_CONTRACT_SHA=966a5dd8be19c409ac616dc194fe8e7dc78ae6c253132c7a73dcdeb8e0c040bc

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_STRATEGY_ND_REPLAY_TAG:-greenfield_strategy_nd_dense_replay_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]] || {
  echo "refusing replay outside exact greenfield worktree" >&2
  exit 2
}
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing replay outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) && ! -e $RUN_DIR ]] || {
  echo "dirty worktree or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/hlo" "$RUN_DIR/replay" \
  "$RUN_DIR/source" "$RUN_DIR/source_db533"

say() {
  echo "[strategy-nd-replay $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

set +e
remote_prefix_listing=$(gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
remote_prefix_rc=$?
set -e
printf '%s\n' "$remote_prefix_listing" >"$RUN_DIR/remote_prefix_preflight.txt"
if [[ $remote_prefix_rc -eq 0 ]]; then
  say "ABORT: append-only remote prefix already contains objects"
  exit 1
elif [[ $remote_prefix_rc -ne 1 ]] || ! grep -q "matched no objects" \
  "$RUN_DIR/remote_prefix_preflight.txt"; then
  say "ABORT: remote-prefix vacancy check failed"
  exit 1
fi

require_sha() {
  local path=$1 expected=$2 label=$3
  [[ -r $path && $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: $label prerequisite hash drifted"
    exit 1
  }
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[m]icrobench_collectives[.]py|[c]ompile_short_decoder[.]py|[p]robe_layer0" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    say "FAILED status=$status; partial diagnostic evidence preserved"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN"
require_sha "$SOURCE_NPZ" "$SOURCE_NPZ_SHA" "DB550 dense partials"
require_sha "$SOURCE_CAPTURE" "$SOURCE_CAPTURE_SHA" "DB550 capture manifest"
require_sha "$SOURCE_COMPARISON" "$SOURCE_COMPARISON_SHA" "DB550 comparison"
require_sha "$SOURCE_SUCCESS" "$SOURCE_SUCCESS_SHA" "DB550 SUCCESS"
require_sha "$SOURCE_REMOTE_OBJECTS" "$SOURCE_REMOTE_OBJECTS_SHA" "DB550 remote ledger"
require_sha "$DB533_ANALYSIS" "$DB533_ANALYSIS_SHA" "DB533 analysis"
require_sha "$DB533_SUMMARY" "$DB533_SUMMARY_SHA" "DB533 summary"
require_sha "$DB533_SUCCESS" "$DB533_SUCCESS_SHA" "DB533 SUCCESS"
require_sha "$DB533_HLO_CONTRACT" "$DB533_HLO_CONTRACT_SHA" "DB533 HLO contract"
[[ $(gcloud storage cat "$SOURCE_REMOTE/dense_partials_capture/dense_partials.npz" |
  sha256sum | awk '{print $1}') == "$SOURCE_NPZ_SHA" ]] || {
  say "ABORT: remote DB550 dense partials drifted"
  exit 1
}
[[ $(gcloud storage cat "$SOURCE_REMOTE/dense_partials_capture/capture.json" |
  sha256sum | awk '{print $1}') == "$SOURCE_CAPTURE_SHA" ]] || {
  say "ABORT: remote DB550 capture manifest drifted"
  exit 1
}
[[ $(gcloud storage cat "$SOURCE_REMOTE/dense_partials_capture/comparison.json" |
  sha256sum | awk '{print $1}') == "$SOURCE_COMPARISON_SHA" ]] || {
  say "ABORT: remote DB550 comparison drifted"
  exit 1
}
[[ $(gcloud storage cat "$SOURCE_REMOTE/SUCCESS" |
  sha256sum | awk '{print $1}') == "$SOURCE_SUCCESS_SHA" ]] || {
  say "ABORT: remote DB550 SUCCESS drifted"
  exit 1
}
[[ $(gcloud storage cat "$SOURCE_REMOTE/remote_objects.json" |
  sha256sum | awk '{print $1}') == "$SOURCE_REMOTE_OBJECTS_SHA" ]] || {
  say "ABORT: remote DB550 object ledger drifted"
  exit 1
}
[[ $(gcloud storage cat "$DB533_REMOTE/association/analysis.json" |
  sha256sum | awk '{print $1}') == "$DB533_ANALYSIS_SHA" ]] || {
  say "ABORT: remote DB533 analysis drifted"
  exit 1
}
[[ $(gcloud storage cat "$DB533_REMOTE/summary.json" |
  sha256sum | awk '{print $1}') == "$DB533_SUMMARY_SHA" ]] || {
  say "ABORT: remote DB533 summary drifted"
  exit 1
}
[[ $(gcloud storage cat "$DB533_REMOTE/SUCCESS" |
  sha256sum | awk '{print $1}') == "$DB533_SUCCESS_SHA" ]] || {
  say "ABORT: remote DB533 SUCCESS drifted"
  exit 1
}
[[ $(gcloud storage cat "$DB533_REMOTE/hlo/strategy_nd_association_bfloat16_32x6144.hlo_contract.json" |
  sha256sum | awk '{print $1}') == "$DB533_HLO_CONTRACT_SHA" ]] || {
  say "ABORT: remote DB533 HLO contract drifted"
  exit 1
}
cp "$SOURCE_NPZ" "$RUN_DIR/source/dense_partials.npz"
cp "$SOURCE_CAPTURE" "$RUN_DIR/source/capture.json"
cp "$SOURCE_COMPARISON" "$RUN_DIR/source/comparison.json"
cp "$SOURCE_SUCCESS" "$RUN_DIR/source/SUCCESS"
cp "$SOURCE_REMOTE_OBJECTS" "$RUN_DIR/source/remote_objects.json"
cp "$DB533_ANALYSIS" "$RUN_DIR/source_db533/analysis.json"
cp "$DB533_SUMMARY" "$RUN_DIR/source_db533/summary.json"
cp "$DB533_SUCCESS" "$RUN_DIR/source_db533/SUCCESS"
cp "$DB533_HLO_CONTRACT" "$RUN_DIR/source_db533/hlo_contract.json"

strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact replay pin to eight isolated worker worktrees"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code sync failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n $coordinator ]] || {
  say "ABORT: could not resolve worker-0 coordinator address"
  exit 1
}
coordinator="$coordinator:8476"
say "launching one model-free M32 reduction (plus one deterministic repeat)"
started=$(date +%s)

# shellcheck disable=SC2016
capture_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; pin='"$PIN"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; source_remote='"$SOURCE_REMOTE"'; source_sha='"$SOURCE_NPZ_SHA"'; coordinator='"$coordinator"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/host_records" "$run/hlo" "$run/replay" "$run/source"; upload_diagnostics() { if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/diagnostic_hlo/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/replay/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/replay/* "$remote/diagnostic_replay/" >/dev/null 2>&1 || true; fi; }; trap upload_diagnostics EXIT; gcloud storage cp "$source_remote/dense_partials_capture/dense_partials.npz" "$run/source/dense_partials.npz" >/dev/null; [[ $(sha256sum "$run/source/dense_partials.npz" | awk '\''{print $1}'\'') == "$source_sha" ]]; cd "$wt"; GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python scripts/greenfield/microbench_collectives.py --mode strategy_nd_dense_replay --coordinator-address "$coordinator" --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --expected-code-hash "$pin" --output "$run/collective.rank${idx}.json" --groups 32 --operations all_reduce --shape 32,6144 --dtype bfloat16 --association-trials 1 --association-replay-input "$run/source/dense_partials.npz"; gcloud storage cp --no-clobber "$run/collective.rank${idx}.json" "$remote/host_records/" >/dev/null; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null; fi; if compgen -G "$run/replay/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/replay/* "$remote/replay/" >/dev/null; fi; trap - EXIT; echo "REPLAY_UPLOAD_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$capture_command" >"$RUN_DIR/capture.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/capture.txt" REPLAY_UPLOAD_OK || {
  say "ABORT: dense replay did not complete on all eight hosts"
  exit 1
}
elapsed=$(( $(date +%s) - started ))
say "device replay workflow completed in ${elapsed}s"

gcloud storage cp "$REMOTE_PREFIX/host_records/collective.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/replay/*" "$RUN_DIR/replay/" >/dev/null
# Worker zero shares the orchestrator filesystem. Its root output is only the
# producer location that keeps HLO/replay siblings at their declared paths;
# the authenticated fleet copy lives under host_records.
rm -f "$RUN_DIR/collective.rank0.json"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$elapsed" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import validate_strategy_nd_dense_replay

run_dir, pin, run_tag, elapsed = sys.argv[1:]
run_dir = Path(run_dir)
summary = validate_strategy_nd_dense_replay(
    run_dir,
    expected_code_hash=pin,
    expected_run_tag=run_tag,
)
summary["elapsed_seconds"] = int(elapsed)
summary["results_db_run_id"] = None
summary["results_db_role"] = "diagnostic_only_no_performance_row"
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(
    "STRATEGY_ND_DENSE_REPLAY_VALID "
    f"classification={summary['classification']} "
    f"mismatches={summary['row0_mismatch_count']}"
)
PY

say "freezing and archiving model-free replay evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find host_records hlo replay source source_db533 -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum capture.txt census_pre.txt census_post.txt orchestrator.sealed.log \
    remote_prefix_preflight.txt summary.json sync.txt
) >"$RUN_DIR/evidence.sha256"

gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" \
  >"$RUN_DIR/remote_objects.json" <<'PY'
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file()
    and path.relative_to(root).as_posix() not in {"SUCCESS", "remote_objects.json"}
]

def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or crc32c != local_crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {
        "crc32c": crc32c,
        "generation": value["generation"],
        "path": relative,
        "size": int(value["size"]),
    }

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null
[[ $(sha256sum "$RUN_DIR/remote_objects.json" | awk '{print $1}') == \
  $(gcloud storage cat "$REMOTE_PREFIX/remote_objects.json" | sha256sum | awk '{print $1}') ]] || {
  say "ABORT: remote object ledger checksum mismatch"
  exit 1
}

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2].rstrip("/")
expected = {
    f"{prefix}/{path.relative_to(root).as_posix()}"
    for path in root.rglob("*")
    if path.is_file() and path.relative_to(root).as_posix() != "SUCCESS"
}
observed = set(subprocess.run(
    ["gcloud", "storage", "ls", f"{prefix}/**"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines())
if observed != expected:
    raise SystemExit(
        "remote nonterminal object set drifted: "
        f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
    )
PY

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
values = {
    "artifact_kind": summary["artifact_kind"],
    "classification": summary["classification"],
    "code_hash": summary["code_hash"],
    "diagnostic_only": "true",
    "hardware_hidden_2795_bfloat16_bits": summary["hardware_hidden_2795_bfloat16_bits"],
    "hardware_row0_raw_sha256": summary["hardware_row0_raw_sha256"],
    "row0_exact": str(summary["row0_exact"]).lower(),
    "row0_mismatch_count": summary["row0_mismatch_count"],
    "software_hidden_2795_bfloat16_bits": summary["software_hidden_2795_bfloat16_bits"],
    "software_row0_raw_sha256": summary["software_row0_raw_sha256"],
    "source_db_run_id": summary["source"]["db_run_id"],
    "source_npz_sha256": summary["source"]["npz_sha256"],
    "source_success_sha256": summary["source"]["success_sha256"],
    "evidence_sha256": sha256((root / "evidence.sha256").read_bytes()).hexdigest(),
    "post_census_sha256": sha256((root / "census_post.txt").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in values.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
[[ $(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}') == \
  $(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}') ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}
terminal_success_done=1

trap - EXIT
classification=$(sed -n 's/^classification=//p' "$RUN_DIR/SUCCESS")
mismatches=$(sed -n 's/^row0_mismatch_count=//p' "$RUN_DIR/SUCCESS")
echo "SUCCESS classification=$classification mismatches=$mismatches elapsed=${elapsed}s"
echo "ARCHIVE=$REMOTE_PREFIX"
