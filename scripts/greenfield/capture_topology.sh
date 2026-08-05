#!/usr/bin/env bash
# Protected Gate-A physical-topology capture. Model-free; existing v4-64 only.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing capture outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing capture from a dirty greenfield worktree" >&2
  exit 2
}
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_TOPOLOGY_TAG:-greenfield_topology_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
mkdir -p "$RUN_DIR/host_records"

say() {
  echo "[topology $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  # Literal remote program; expansions occur on the TPU VM.
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[r]un_bench[.]py|[g]ate_sparse128k[.]sh|[s]tage256k[.]sh|[b]ench_run[.]sh|[e]0_capture_arm[.]sh|[i]nspect_topology[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR and $REMOTE_PREFIX"
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact greenfield pin to isolated worker worktrees"
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
[[ -n "$coordinator" ]] || {
  say "ABORT: could not resolve worker-0 coordinator address"
  exit 1
}
coordinator="$coordinator:8476"
say "launching eight-host topology capture coordinator=$coordinator"

# shellcheck disable=SC2016
capture_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; pin='"$PIN"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run"; cd "$wt"; GLM_GREENFIELD_RUN_TAG="$tag" /home/gianl/vllm-env/bin/python scripts/greenfield/inspect_topology.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --expected-code-hash "$pin" --output "$run/topology.rank${idx}.json"; sha256sum "$run/topology.rank${idx}.json" >"$run/topology.rank${idx}.sha256"; gcloud storage cp --no-clobber "$run/topology.rank${idx}.json" "$run/topology.rank${idx}.sha256" "$remote/host_records/" >/dev/null; echo "CAPTURE_UPLOAD_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$capture_command" >"$RUN_DIR/capture.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/capture.txt" CAPTURE_UPLOAD_OK || {
  say "ABORT: topology capture/upload did not pass on all eight hosts"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/topology.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
say "validating fleet agreement and appending provenance DB row"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, oracle_pin, db_path, repo, oracle_repo = sys.argv[1:]
run_dir = Path(run_dir)
records = [json.loads(path.read_text()) for path in sorted((run_dir / "host_records").glob("*.json"))]
if len(records) != 8:
    raise SystemExit(f"expected 8 host records, got {len(records)}")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover process indices 0..7 exactly")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover launch process ids 0..7 exactly")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records do not contain eight distinct hostnames")
contract_hashes = {record["contract_hash"] for record in records}
if len(contract_hashes) != 1:
    raise SystemExit(f"fleet topology contracts differ: {sorted(contract_hashes)}")
contract_hash = contract_hashes.pop()
for record in records:
    if record["contract"]["code_hash"] != pin:
        raise SystemExit("host record carries stale code hash")
    if record["fleet_contract_hashes"] != [contract_hash] * 8:
        raise SystemExit("on-device fleet contract agreement is incomplete")
    if record["jax_device_count"] != 32 or record["jax_local_device_count"] != 4:
        raise SystemExit("runtime JAX chip counts do not match v4-64")
    if record["jax_process_count"] != 8 or len(record["local_device_ids"]) != 4:
        raise SystemExit("runtime process/local inventory is incomplete")
contract = records[0]["contract"]
summary = {
    "code_hash": pin,
    "contract_hash": contract_hash,
    "hostnames": sorted(record["hostname"] for record in records),
    "launch_to_jax_process": {
        str(record["launch_process_id"]): record["jax_process_index"]
        for record in sorted(records, key=lambda item: item["launch_process_id"])
    },
    "oracle_code_hash": oracle_pin,
    "pp16_lp2_hash": contract["pp16_lp2_hash"],
    "pp8_lp4_hash": contract["pp8_lp4_hash"],
    "process_indices": list(range(8)),
    "topology_hash": contract["topology_hash"],
    "topology_shape": contract["topology"]["topology_shape"],
}

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-topology-only",
    revision=None,
    env={
        "GLM_ENGINE": "greenfield_topology_capture",
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "topology_contract_hash": contract_hash,
        "topology_hash": contract["topology_hash"],
        "pp8_lp4_hash": contract["pp8_lp4_hash"],
        "pp16_lp2_hash": contract["pp16_lp2_hash"],
    },
    note="Gate-A runtime physical topology and explicit local groups",
    harness_repo=repo,
    fork_repo=oracle_repo,
)
pv.record_item(
    conn,
    run_id,
    benchmark="greenfield_topology",
    item_id=contract_hash,
    prompt="Capture v4-64 physical topology and derive PP8_LP4/PP16_LP2 local groups.",
    gold="8 hosts, 32 chips, 2x4x4 permutation, host-local PP8, adjacent PP16",
    raw_output=json.dumps(summary, sort_keys=True),
    extracted=contract_hash,
    correct=True,
    score=1.0,
)
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_topology",
    metric="contract_valid",
    value=1.0,
    note="Synthetic/device metadata proof only; no model throughput claim.",
)
conn.close()
summary["results_db_run_id"] = run_id
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
check = sqlite3.connect(run_dir / "results_ckpt.db").execute("PRAGMA integrity_check").fetchone()[0]
if check != "ok":
    raise SystemExit(f"results DB snapshot integrity failed: {check}")
print(f"TOPOLOGY_PROOF_VALID contract={contract_hash} db_run={run_id}")
PY

(cd "$RUN_DIR" && sha256sum host_records/*.json summary.json results_ckpt.db \
  >evidence.sha256)
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  "$RUN_DIR/evidence.sha256" "$RUN_DIR/orchestrator.log" "$RUN_DIR/sync.txt" \
  "$RUN_DIR/capture.txt" "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ "$remote_success" == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS contract=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract_hash"])' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
