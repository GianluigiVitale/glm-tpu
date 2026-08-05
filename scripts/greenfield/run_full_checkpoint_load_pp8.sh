#!/usr/bin/env bash
# Protected Gate-B complete PP8 raw-FP8 load and device-byte round-trip.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly CHECKPOINT_DESTINATION=gs://driftbench-dsv4-uc/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly TOPOLOGY_RUN=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z
readonly PACKED_MANIFEST_SHA=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly LAYOUT_MANIFEST_SHA=aca0eb6d4c498271baf3260dacbe55c9bf935539f42a4b7eaa889947627506a0
readonly SOURCE_INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PLAN_GROUP_SHA=d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14
readonly PLAN_MANIFEST_SHA=c5bfacc3eeaecce59fb92d10d1a03087b70751a96d13ce6dbd7673270995c4ed
readonly EXECUTION_PLAN_SHA=23fd23f8513bea9e5f0c53fafd0b978534a75f60010be43d6a105e0d261f5c89
readonly LAYOUT_CODE_HASH=61c68727b8b57f31533c0d24f1eb1f812cdad357
readonly PACK_CODE_HASH=61475d30a708252bc64aaf50fec2c2ea1ea03c63

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_FULL_LOAD_TAG:-greenfield_full_checkpoint_load_pp8_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing full load outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing full load from a dirty greenfield worktree" >&2
  exit 2
}
[[ -f $CHECKPOINT_ROOT/SUCCESS && -f $CHECKPOINT_ROOT/packed_manifest.json ]] || {
  echo "complete checkpoint SUCCESS/manifest is unavailable" >&2
  exit 2
}
[[ -r $RESULTS_DB ]] || {
  echo "results database is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory already exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/probe" "$RUN_DIR/host_records"

say() {
  echo "[full-load-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[l]oad_full_checkpoint_stage[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; preserving partial evidence at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN"
say "CHECKPOINT=$PACKED_MANIFEST_SHA LAYOUT=$LAYOUT_MANIFEST_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact loader pin and checking immutable prerequisites"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; checkpoint='"$CHECKPOINT_ROOT"'; topology='"$TOPOLOGY_RUN"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin '"$BRANCH"'; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch '"$BRANCH"' "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; topo="$topology/topology.rank${idx}.json"; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$checkpoint/SUCCESS" ]] && [[ -r "$topo" ]] && findmnt -T "$checkpoint" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code/artifact sync failed"
  exit 1
}

readonly COMMON_ARGS="--checkpoint-root $CHECKPOINT_ROOT --expected-code-hash $PIN --packed-manifest-sha256 $PACKED_MANIFEST_SHA --layout-manifest-sha256 $LAYOUT_MANIFEST_SHA --source-inventory-sha256 $SOURCE_INVENTORY_SHA --source-revision $SOURCE_REVISION --topology-sha256 $TOPOLOGY_SHA --plan-group-sha256 $PLAN_GROUP_SHA --plan-manifest-sha256 $PLAN_MANIFEST_SHA --execution-plan-sha256 $EXECUTION_PLAN_SHA --layout-code-hash $LAYOUT_CODE_HASH --pack-code-hash $PACK_CODE_HASH --destination $CHECKPOINT_DESTINATION --plan-id PP8_LP4 --verify-device-roundtrip"

say "running one-host complete-stage probe before fleet allocation"
# shellcheck disable=SC2016
probe_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; common='"'$COMMON_ARGS'"'; topology='"$TOPOLOGY_RUN"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag/probe; mkdir -p "$run"; out="$run/stage.worker${idx}.json"; state="$run/state.worker${idx}.json"; log="$run/stage.worker${idx}.log"; trap '\''gcloud storage cp --no-clobber "$log" "$out" "$state" "$remote/probe_diagnostic/" >/dev/null 2>&1 || true'\'' EXIT; cd "$wt"; env JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" timeout --signal=TERM --kill-after=60 3600 /home/gianl/vllm-env/bin/python scripts/greenfield/load_full_checkpoint_stage.py $common --topology-capture "$topology/topology.rank${idx}.json" --output "$out" --state-manifest-output "$state" >"$log" 2>&1; sha256sum "$out" "$state" >"$run/evidence.worker${idx}.sha256"; gcloud storage cp --no-clobber "$out" "$state" "$log" "$run/evidence.worker${idx}.sha256" "$remote/probe/" >/dev/null; trap - EXIT; echo "PROBE_UPLOAD_OK $(hostname)"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="$probe_command" >"$RUN_DIR/probe.txt" 2>&1
grep -q '^PROBE_UPLOAD_OK ' "$RUN_DIR/probe.txt" || {
  say "ABORT: one-host complete-stage probe failed"
  exit 1
}
gcloud storage cp "$REMOTE_PREFIX/probe/*" "$RUN_DIR/probe/" >/dev/null
/home/gianl/vllm-env/bin/python - "$RUN_DIR/probe" "$PIN" <<'PY'
import json
from pathlib import Path
import sys
root, pin = Path(sys.argv[1]), sys.argv[2]
records = [json.loads(path.read_text()) for path in root.glob("stage.worker*.json")]
if len(records) != 1 or records[0].get("status") != "SUCCESS":
    raise SystemExit("complete-stage probe record failed")
record = records[0]
if (
    record.get("code_hash") != pin
    or not record["load"]["device_roundtrip_verified"]
    or record["load"]["device_roundtrip_bytes"]
    != record["load"]["loaded_payload_bytes"]
    or record["load"]["fp8_host_dequantizations"] != 0
    or record["load"]["runtime_checkpoint_reshards"] != 0
):
    raise SystemExit("complete-stage probe load contract failed")
print(f"PROBE_VALID stage={record['stage_id']} payload={record['load']['loaded_payload_bytes']}")
PY
strict_census post_probe || {
  say "ABORT: probe did not release the fleet cleanly"
  exit 1
}

say "probe passed; launching eight standalone topology-resolved stage loads"
# shellcheck disable=SC2016
load_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; common='"'$COMMON_ARGS'"'; topology='"$TOPOLOGY_RUN"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag/host_records; mkdir -p "$run"; out="$run/stage.worker${idx}.json"; state="$run/state.worker${idx}.json"; log="$run/stage.worker${idx}.log"; trap '\''gcloud storage cp --no-clobber "$log" "$out" "$state" "$remote/fleet_diagnostic/" >/dev/null 2>&1 || true'\'' EXIT; cd "$wt"; env JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" timeout --signal=TERM --kill-after=60 3600 /home/gianl/vllm-env/bin/python scripts/greenfield/load_full_checkpoint_stage.py $common --topology-capture "$topology/topology.rank${idx}.json" --output "$out" --state-manifest-output "$state" >"$log" 2>&1; sha256sum "$out" "$state" >"$run/evidence.worker${idx}.sha256"; gcloud storage cp --no-clobber "$out" "$state" "$log" "$run/evidence.worker${idx}.sha256" "$remote/host_records/" >/dev/null; trap - EXIT; echo "LOAD_UPLOAD_OK $(hostname)"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$load_command" >"$RUN_DIR/load.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/load.txt" LOAD_UPLOAD_OK || {
  say "ABORT: eight-host load/upload did not pass"
  exit 1
}
gcloud storage cp "$REMOTE_PREFIX/host_records/*" \
  "$RUN_DIR/host_records/" >/dev/null

strict_census post || {
  say "ABORT: post-load census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "validating all 32 owners and linking protected evidence to results DB"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$ORACLE_PIN" "$RESULTS_DB" "$WORKTREE" \
  "$ORACLE_REPO" "$CHECKPOINT_ROOT" "$PACKED_MANIFEST_SHA" \
  "$LAYOUT_MANIFEST_SHA" "$SOURCE_INVENTORY_SHA" "$SOURCE_REVISION" \
  "$TOPOLOGY_SHA" "$PLAN_GROUP_SHA" "$PLAN_MANIFEST_SHA" \
  "$EXECUTION_PLAN_SHA" "$LAYOUT_CODE_HASH" "$PACK_CODE_HASH" \
  "$CHECKPOINT_DESTINATION" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

(
    run_dir, pin, oracle_pin, db_path, repo, oracle_repo, checkpoint_root,
    packed_sha, layout_sha, inventory_sha, source_revision, topology_sha,
    group_sha, plan_manifest_sha, execution_sha, layout_code, pack_code,
    destination,
) = sys.argv[1:]
run_dir = Path(run_dir)
records = [
    json.loads(path.read_text())
    for path in sorted((run_dir / "host_records").glob("stage.worker*.json"))
]
states_by_file = {
    path.name.replace("state.", "stage."): json.loads(path.read_text())
    for path in sorted((run_dir / "host_records").glob("state.worker*.json"))
}
if len(records) != 8 or len(states_by_file) != 8:
    raise SystemExit(
        f"expected eight runner/state records, got {len(records)}/{len(states_by_file)}"
    )
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("stage-load records do not contain eight distinct hosts")
if {record["stage_id"] for record in records} != set(range(8)):
    raise SystemExit("stage-load records do not cover stages 0..7")
if {record["captured_process_index"] for record in records} != set(range(8)):
    raise SystemExit("stage-load records do not cover captured processes 0..7")
if {
    device_id
    for record in records
    for device_id in record["captured_device_ids_in_slot_order"]
} != set(range(32)):
    raise SystemExit("stage-load records do not cover physical device ids 0..31")

sys.path.insert(0, str(Path(repo)))
from glm_tpu.greenfield.checkpoint import (
    FullCheckpointLoadExpectation,
    verify_full_packed_checkpoint,
)

expectation = FullCheckpointLoadExpectation(
    packed_manifest_sha256=packed_sha,
    layout_manifest_sha256=layout_sha,
    source_inventory_sha256=inventory_sha,
    source_revision=source_revision,
    topology_hash=topology_sha,
    plan_group_hash=group_sha,
    plan_manifest_sha256=plan_manifest_sha,
    execution_plan_sha256=execution_sha,
    layout_code_hash=layout_code,
    pack_code_hash=pack_code,
    destination=destination,
)
checkpoint = verify_full_packed_checkpoint(Path(checkpoint_root), expectation)
base_plans = [plan for plan in checkpoint.plans if plan.load_set == "base_decoder"]
expected_by_stage = {
    stage: sorted(
        (plan for plan in base_plans if plan.stage_id == stage),
        key=lambda item: item.device_slot,
    )
    for stage in range(8)
}
loaded_files = set()
loaded_payload = 0
loaded_tensors = 0
max_peak_hbm = 0
stage_summaries = []
for record in records:
    if record.get("status") != "SUCCESS" or record.get("code_hash") != pin:
        raise SystemExit("stage-load status/code identity failed")
    if (
        record.get("plan_id") != "PP8_LP4"
        or record.get("topology_sha256") != topology_sha
        or record.get("plan_group_sha256") != group_sha
    ):
        raise SystemExit("stage-load plan/topology identity failed")
    load = record["load"]
    if (
        not load["device_roundtrip_verified"]
        or load["device_roundtrip_bytes"] != load["loaded_payload_bytes"]
        or load["fp8_device_dequantizations"] != 0
        or load["fp8_host_dequantizations"] != 0
        or load["host_global_concatenations"] != 0
        or load["runtime_checkpoint_reshards"] != 0
    ):
        raise SystemExit("stage-load direct raw-FP8 contract failed")
    runner_name = next(
        path.name
        for path in (run_dir / "host_records").glob("stage.worker*.json")
        if json.loads(path.read_text())["hostname"] == record["hostname"]
    )
    state = states_by_file[runner_name]
    unhashed = dict(state)
    state_hash = unhashed.pop("manifest_sha256", None)
    computed = sha256(
        json.dumps(
            unhashed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if state_hash != computed or state_hash != record["state_manifest"]["manifest_sha256"]:
        raise SystemExit("loaded state semantic hash failed")
    state_path = run_dir / "host_records" / runner_name.replace("stage.", "state.")
    if sha256(state_path.read_bytes()).hexdigest() != record["state_manifest"]["file_sha256"]:
        raise SystemExit("loaded state file hash failed")
    plans = expected_by_stage[record["stage_id"]]
    state_files = sorted(state["files"], key=lambda item: item["device_slot"])
    if len(plans) != 4 or len(state_files) != 4:
        raise SystemExit("stage does not contain exactly four owner files")
    for plan, state_file in zip(plans, state_files, strict=True):
        expected_evidence = checkpoint.evidence_by_filename[plan.filename]
        if (
            state_file["filename"] != plan.filename
            or state_file["device_id"] != plan.device_id
            or state_file["device_slot"] != plan.device_slot
            or state_file["file_sha256"] != expected_evidence["sha256"]
            or state_file["payload_bytes"] != plan.payload_bytes
            or state_file["tensor_count"] != len(plan.tensors)
        ):
            raise SystemExit(f"loaded owner file drifted: {plan.filename}")
        expected_tensors = {
            tensor.name: (
                tensor.dtype,
                list(tensor.shape),
                tensor.byte_count,
            )
            for tensor in plan.tensors
        }
        observed_tensors = {
            tensor["name"]: (
                tensor["logical_dtype"],
                tensor["shape"],
                tensor["byte_count"],
            )
            for tensor in state_file["tensors"]
        }
        if observed_tensors != expected_tensors:
            raise SystemExit(f"loaded leaf contract drifted: {plan.filename}")
        if any(len(tensor["sha256"]) != 64 for tensor in state_file["tensors"]):
            raise SystemExit(f"loaded leaf checksum missing: {plan.filename}")
        loaded_files.add(plan.filename)
    if state["payload_bytes"] != load["loaded_payload_bytes"]:
        raise SystemExit("state/runner loaded byte totals disagree")
    if state["tensor_count"] != load["loaded_tensor_count"]:
        raise SystemExit("state/runner tensor totals disagree")
    loaded_payload += state["payload_bytes"]
    loaded_tensors += state["tensor_count"]
    peaks = [item["peak_bytes_in_use"] for item in record["device_memory_after_load"]]
    max_peak_hbm = max(max_peak_hbm, *peaks)
    stage_summaries.append(
        {
            "captured_process_index": record["captured_process_index"],
            "device_ids": record["captured_device_ids_in_slot_order"],
            "hostname": record["hostname"],
            "load_seconds": record["payload_load_and_roundtrip_seconds"],
            "peak_hbm_bytes": max(peaks),
            "payload_bytes": state["payload_bytes"],
            "stage_id": record["stage_id"],
            "state_manifest_sha256": state_hash,
            "tensor_count": state["tensor_count"],
        }
    )
if loaded_files != {plan.filename for plan in base_plans}:
    raise SystemExit("fleet load did not cover the exact base file set")
if loaded_payload != sum(plan.payload_bytes for plan in base_plans):
    raise SystemExit("fleet loaded payload bytes do not reconcile")
if loaded_tensors != sum(len(plan.tensors) for plan in base_plans):
    raise SystemExit("fleet loaded tensor count does not reconcile")

summary = {
    "artifact_kind": "greenfield_full_checkpoint_load_proof",
    "base_file_count": len(base_plans),
    "checkpoint_destination": destination,
    "code_hash": pin,
    "device_roundtrip_verified": True,
    "host_count": len(records),
    "loaded_payload_bytes": loaded_payload,
    "loaded_tensor_count": loaded_tensors,
    "maximum_peak_hbm_bytes": max_peak_hbm,
    "oracle_code_hash": oracle_pin,
    "packed_manifest_sha256": packed_sha,
    "plan_id": "PP8_LP4",
    "stage_summaries": sorted(stage_summaries, key=lambda item: item["stage_id"]),
    "status": "SUCCESS",
}

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-raw-fp8-final-layout-load",
    revision=source_revision,
    env={
        "GLM_ENGINE": "greenfield",
        "GLM_EXECUTION_PLAN": "PP8_LP4",
        "greenfield_code_hash": pin,
        "layout_manifest_sha256": layout_sha,
        "packed_manifest_sha256": packed_sha,
        "topology_hash": topology_sha,
    },
    note="Gate-B full checkpoint direct-load integrity proof; not throughput.",
    harness_repo=repo,
    fork_repo=oracle_repo,
)
for stage in summary["stage_summaries"]:
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_full_checkpoint_load_pp8",
        item_id=f"stage_{stage['stage_id']:02d}",
        prompt="Directly load one final-owner PP8 stage and round-trip every device byte.",
        gold="Exact physical owners, raw FP8 resident, full SHA/state/HBM contract.",
        raw_output=json.dumps(stage, sort_keys=True),
        extracted=stage["state_manifest_sha256"],
        correct=True,
        score=1.0,
    )
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_full_checkpoint_load_pp8",
    metric="contract_valid",
    value=1.0,
    note="Load/integrity evidence only; no token latency or throughput claim.",
)
conn.close()
summary["results_db_run_id"] = run_id
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
print(
    f"FULL_LOAD_PROOF_VALID stages=8 devices=32 payload={loaded_payload} "
    f"db_run={run_id} max_peak_hbm={max_peak_hbm}"
)
PY

(cd "$RUN_DIR" && find probe host_records -type f -print0 | sort -z | \
  xargs -0 sha256sum >evidence.sha256)
sha256sum "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/sync.txt" "$RUN_DIR/probe.txt" \
  "$RUN_DIR/load.txt" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post_probe.txt" "$RUN_DIR/census_post.txt" \
  >>"$RUN_DIR/evidence.sha256"

touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/sync.txt" "$RUN_DIR/probe.txt" \
  "$RUN_DIR/load.txt" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post_probe.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS DB=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")"
