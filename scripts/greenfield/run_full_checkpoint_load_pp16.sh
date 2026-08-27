#!/usr/bin/env bash
# Protected Gate-B complete PP16 raw-FP8 load and device-byte round-trip.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly CHECKPOINT_TAG=greenfield_full_pack_pp16_20260827T032310295108546Z
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP16_LP2/$CHECKPOINT_TAG
readonly CHECKPOINT_DESTINATION=$APPROVED_BUCKET/checkpoints/greenfield/glm52/packed/PP16_LP2/$CHECKPOINT_TAG
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records
readonly PACKED_MANIFEST_SHA=13ad2e926b44bee86e620d877d4c266dbacc4f9a59b352fc4f2c8ef534efedb5
readonly LAYOUT_MANIFEST_SHA=f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9
readonly SOURCE_INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PLAN_GROUP_SHA=6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21
readonly PLAN_MANIFEST_SHA=3c3ea07b0f97a84702e7b2ad370b014ea22c9e69bc7d35ad88e23a4ad82ded16
readonly EXECUTION_PLAN_SHA=079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3
readonly LAYOUT_CODE_HASH=6dc73048fec3b15892e061bbaf1ec5886f22d29c
readonly PACK_CODE_HASH=3685ee40960f53874e18ad54ec1e8a0c65e53a62

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_FULL_LOAD:-0} == 1 ]] || {
  echo "PP16 complete checkpoint load is default-off" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_FULL_LOAD_TAG:-greenfield_full_checkpoint_load_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_full_checkpoint_load_pp16_[0-9]{8}T[0-9]{15}Z$ ]]
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
[[ -r $CHECKPOINT_ROOT/SUCCESS && -r $CHECKPOINT_ROOT/packed_manifest.json ]]
[[ -r $RESULTS_DB ]]
mkdir -p "$RUN_DIR/probe" "$RUN_DIR/host_records"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || { echo "another protected TPU workflow holds the lease" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[full-load-pp16 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  local command
  command='generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[l]oad_full_checkpoint_stage[.]py|[l]oad_checkpoint_probe[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}}" 2>/dev/null); if [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; echo "$generic"; echo "$holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || true; fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving nonterminal diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TAG=$TAG checkpoint=$PACKED_MANIFEST_SHA"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state,health)') == $'READY\tHEALTHY' ]]
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]
strict_census pre || { say "ABORT: pre-run census is not 8/8 clean"; exit 1; }

say "syncing exact loader pin and checkpoint/topology prerequisites on all hosts"
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; wt='"$WORKTREE"'; pin='"$PIN"'; branch='"$BRANCH"'; checkpoint='"$CHECKPOINT_ROOT"'; topology='"$TOPOLOGY_ROOT"'; if [[ "$idx" == 0 ]]; then [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else [[ -e "$wt/.git" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$checkpoint/SUCCESS" ]] && [[ -r "$topology/topology.rank${idx}.json" ]] && findmnt -T "$checkpoint" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host sync failed"
  exit 1
}

readonly COMMON_ARGS="--checkpoint-root $CHECKPOINT_ROOT --expected-code-hash $PIN --packed-manifest-sha256 $PACKED_MANIFEST_SHA --layout-manifest-sha256 $LAYOUT_MANIFEST_SHA --source-inventory-sha256 $SOURCE_INVENTORY_SHA --source-revision $SOURCE_REVISION --topology-sha256 $TOPOLOGY_SHA --plan-group-sha256 $PLAN_GROUP_SHA --plan-manifest-sha256 $PLAN_MANIFEST_SHA --execution-plan-sha256 $EXECUTION_PLAN_SHA --layout-code-hash $LAYOUT_CODE_HASH --pack-code-hash $PACK_CODE_HASH --destination $CHECKPOINT_DESTINATION --plan-id PP16_LP2 --verify-device-roundtrip"

say "running worker-4/stage-0 complete two-owner probe before fleet allocation"
probe_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; stage=0; tag='"$TAG"'; wt='"$WORKTREE"'; common='"'$COMMON_ARGS'"'; topology='"$TOPOLOGY_ROOT"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag/probe; mkdir -p "$run"; out="$run/stage_00.worker${idx}.json"; state="$run/state_00.worker${idx}.json"; log="$run/stage_00.worker${idx}.log"; cd "$wt"; status=0; env JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" timeout --signal=TERM --kill-after=60 3600 /home/gianl/vllm-env/bin/python scripts/greenfield/load_full_checkpoint_stage.py $common --stage-id "$stage" --topology-capture "$topology/topology.rank${idx}.json" --output "$out" --state-manifest-output "$state" >"$log" 2>&1 || status=$?; if [[ $status -ne 0 ]]; then gcloud storage cp --no-clobber "$log" "$out" "$state" "$remote/probe_diagnostic/" >/dev/null 2>&1 || true; exit "$status"; fi; sha256sum "$out" "$state" >"$run/evidence.stage00.worker${idx}.sha256"; gcloud storage cp --no-clobber "$out" "$state" "$log" "$run/evidence.stage00.worker${idx}.sha256" "$remote/probe/" >/dev/null; echo "PROBE_OK $(hostname) stage=0"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=4 \
  --command="$probe_command" >"$RUN_DIR/probe_dispatch.txt" 2>&1
grep -q '^PROBE_OK .* stage=0$' "$RUN_DIR/probe_dispatch.txt"
gcloud storage cp "$REMOTE_PREFIX/probe/*" "$RUN_DIR/probe/" >/dev/null
/home/gianl/vllm-env/bin/python - "$RUN_DIR/probe" "$PIN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); pin=sys.argv[2]
records=[json.loads(p.read_text()) for p in root.glob('stage_*.json')]
if len(records)!=1 or records[0]['status']!='SUCCESS' or records[0]['code_hash']!=pin:
 raise SystemExit('PP16 full-stage probe failed')
r=records[0]; load=r['load']
if (r['stage_id']!=0 or r['captured_device_ids_in_slot_order']!=[0,1] or
    not load['device_roundtrip_verified'] or load['device_roundtrip_bytes']!=load['loaded_payload_bytes'] or
    any(load[k] for k in ('fp8_device_dequantizations','fp8_host_dequantizations',
                          'host_global_concatenations','runtime_checkpoint_reshards'))):
 raise SystemExit('PP16 full-stage probe contract failed')
print(f"PROBE_VALID payload={load['loaded_payload_bytes']}")
PY
strict_census post_probe || { say "ABORT: probe did not release fleet cleanly"; exit 1; }

say "probe passed; loading both authenticated PP16 stages per host"
load_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; case "$idx" in 0) stages="9 14";; 1) stages="8 13";; 2) stages="10 15";; 3) stages="1 6";; 4) stages="0 5";; 5) stages="3 4";; 6) stages="11 12";; 7) stages="2 7";; *) exit 2;; esac; tag='"$TAG"'; wt='"$WORKTREE"'; common='"'$COMMON_ARGS'"'; topology='"$TOPOLOGY_ROOT"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag/host_records; mkdir -p "$run"; cd "$wt"; for stage in $stages; do label=$(printf "%02d" "$stage"); out="$run/stage_${label}.worker${idx}.json"; state="$run/state_${label}.worker${idx}.json"; log="$run/stage_${label}.worker${idx}.log"; status=0; env JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" timeout --signal=TERM --kill-after=60 3600 /home/gianl/vllm-env/bin/python scripts/greenfield/load_full_checkpoint_stage.py $common --stage-id "$stage" --topology-capture "$topology/topology.rank${idx}.json" --output "$out" --state-manifest-output "$state" >"$log" 2>&1 || status=$?; if [[ $status -ne 0 ]]; then gcloud storage cp --no-clobber "$log" "$out" "$state" "$remote/fleet_diagnostic/" >/dev/null 2>&1 || true; exit "$status"; fi; evidence="$run/evidence.stage${label}.worker${idx}.sha256"; sha256sum "$out" "$state" >"$evidence"; gcloud storage cp --no-clobber "$out" "$state" "$log" "$evidence" "$remote/host_records/" >/dev/null; done; echo "LOAD_HOST_OK $(hostname) stages=$stages"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$load_command" >"$RUN_DIR/load_dispatch.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/load_dispatch.txt" LOAD_HOST_OK || {
  say "ABORT: sixteen-stage fleet load failed"
  exit 1
}
gcloud storage cp "$REMOTE_PREFIX/host_records/*" "$RUN_DIR/host_records/" >/dev/null
strict_census post || { say "ABORT: post-run census is not 8/8 clean"; exit 1; }
post_census_done=1

say "validating all 32 base owners, state/HBM, DB linkage and archive"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  scripts/greenfield/validate_full_checkpoint_load_pp16.py \
  --run-dir "$RUN_DIR" --code-hash "$PIN" --results-db "$RESULTS_DB" \
  --checkpoint-root "$CHECKPOINT_ROOT" --destination "$CHECKPOINT_DESTINATION"

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(cd "$RUN_DIR" && find probe host_records -type f -print0 | sort -z | xargs -0 sha256sum >evidence.sha256)
sha256sum "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" "$RUN_DIR/sync.txt" \
  "$RUN_DIR/probe_dispatch.txt" "$RUN_DIR/load_dispatch.txt" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post_probe.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/orchestrator.sealed.log" >>"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing nonterminal fleet proof"
for name in summary.json results_ckpt.db sync.txt probe_dispatch.txt load_dispatch.txt \
  census_pre.txt census_post_probe.txt census_post.txt orchestrator.sealed.log evidence.sha256; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/$name" >/dev/null
done
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  scripts/greenfield/seal_full_checkpoint_load_pp16.py \
  --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" --code-hash "$PIN" --run-tag "$TAG"
trap - EXIT
db_id=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")
echo "[full-load-pp16 $(date -u +%H:%M:%S)] COMPLETE DB=$db_id"
