#!/usr/bin/env bash
# Protected PP16 feature-runtime -> fused-QKV-A final-layout derivative.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP16_LP2/greenfield_full_pack_pp16_20260827T032310295108546Z
readonly SOURCE_MANIFEST_SHA=13ad2e926b44bee86e620d877d4c266dbacc4f9a59b352fc4f2c8ef534efedb5
readonly PARENT_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP16_LP2/greenfield_runtime_pack_pp16_20260827T091323450875229Z
readonly PARENT_RUNTIME_MANIFEST_SHA=b0f624666e921a93e92c625e0e1248aaa9162c45a9be974f4491755c8604d2e5
readonly SOURCE_FEATURE_TAG=greenfield_runtime_feature_pack_pp16_20260827T095428043535926Z
readonly SOURCE_FEATURE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$SOURCE_FEATURE_TAG
readonly SOURCE_FEATURE_MANIFEST_SHA=0f1bb2718a700fb2eee23dc9f172cd9e5cbd1639396d8c8fa8421e1e3b52b6f1
readonly TOPOLOGY_RUN=/home/gianl/glm-run/greenfield_topology_20260826T194116460015528Z

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_FEATURE_QKV_PACK:-0} == 1 ]] || {
  echo "PP16 direct fused-QKV-A pack is default-off" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_FEATURE_QKV_TAG:-greenfield_runtime_feature_qkv_direct_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_runtime_feature_qkv_direct_pp16_[0-9]{8}T[0-9]{15}Z$ ]]
RUN_DIR=/home/gianl/glm-run/$TAG
DESTINATION=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$TAG
CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
RESUME=0
RESUME_FLAG=
if [[ ${GLM_GREENFIELD_PP16_FEATURE_QKV_RESUME:-0} == 1 ]]; then
  RESUME=1
  RESUME_FLAG=--resume
fi
readonly PIN TAG RUN_DIR DESTINATION CHECKPOINT_ROOT REMOTE_PREFIX RESUME RESUME_FLAG

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
for path in \
  "$SOURCE_ROOT/packed_manifest.json" "$SOURCE_ROOT/SUCCESS" \
  "$PARENT_RUNTIME_ROOT/runtime_manifest.json" "$PARENT_RUNTIME_ROOT/SUCCESS" \
  "$SOURCE_FEATURE_ROOT/runtime_manifest.json" "$SOURCE_FEATURE_ROOT/SUCCESS"; do
  [[ -r $path ]] || { echo "missing pack prerequisite: $path" >&2; exit 2; }
done
if [[ $RESUME == 0 && -e $RUN_DIR ]]; then
  echo "append-only run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/host_records"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || { echo "another protected TPU workflow holds the lease" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[pp16-feature-qkv $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1 out="$RUN_DIR/census_${1}.txt" carrier="${TAG}_${1}"
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[r]un_short_decoder_ws32[.]py|[l]oad_full_checkpoint_stage[.]py|[p]ack_feature_runtime_checkpoint[.]py" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; echo "$generic"; echo "$holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || true; fi
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    say "FAILED status=$status; preserving append-only diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TAG=$TAG source_feature=$SOURCE_FEATURE_MANIFEST_SHA"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state,health)') == $'READY\tHEALTHY' ]]
if [[ $RESUME == 0 ]]; then
  gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$RUN_DIR/remote_vacancy.txt"
  [[ ! -s $RUN_DIR/remote_vacancy.txt ]]
fi
strict_census pre || { say "ABORT: pre-pack fleet is not 8/8 clean"; exit 1; }

say "syncing exact code and same-region mounted prerequisites"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; source='"$SOURCE_ROOT"'; parent='"$PARENT_RUNTIME_ROOT"'; feature='"$SOURCE_FEATURE_ROOT"'; topology='"$TOPOLOGY_RUN"'; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin '"$BRANCH"'; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch '"$BRANCH"' "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; capture="$topology/topology.rank${idx}.json"; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source/SUCCESS" ]] && [[ -r "$parent/SUCCESS" ]] && [[ -r "$feature/SUCCESS" ]] && [[ -r "$capture" ]] && findmnt -T "$source" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$parent" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$feature" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname)"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || { say "ABORT: sync failed"; exit 1; }

COMMON_ARGS=(
  --source-checkpoint-root "$SOURCE_ROOT"
  --source-packed-manifest-sha256 "$SOURCE_MANIFEST_SHA"
  --source-runtime-root "$PARENT_RUNTIME_ROOT"
  --source-runtime-manifest-sha256 "$PARENT_RUNTIME_MANIFEST_SHA"
  --source-feature-runtime-root "$SOURCE_FEATURE_ROOT"
  --source-feature-runtime-manifest-sha256 "$SOURCE_FEATURE_MANIFEST_SHA"
  --source-metadata-only --fused-qkv-a
  --destination "$DESTINATION" --expected-code-hash "$PIN"
)
say "preparing direct derivative metadata"
/home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py \
  prepare "${COMMON_ARGS[@]}" --run-dir "$RUN_DIR/pack_control" $RESUME_FLAG \
  >"$RUN_DIR/prepare.txt" 2>&1

# The probe executes only on the launcher whose captured JAX process index is 0.
# shellcheck disable=SC2016
probe_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; capture='"$TOPOLOGY_RUN"'/topology.rank${idx}.json; jax=$(/home/gianl/vllm-env/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))[\"jax_process_index\"])" "$capture"); if [[ "$jax" != 0 ]]; then echo "PROBE_SKIP $(hostname)"; exit 0; fi; cd '"$WORKTREE"'; run=/home/gianl/glm-run/'"$TAG"'/probe; mkdir -p "$run"; /home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py pack-stage --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --source-runtime-root '"$PARENT_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$PARENT_RUNTIME_MANIFEST_SHA"' --source-feature-runtime-root '"$SOURCE_FEATURE_ROOT"' --source-feature-runtime-manifest-sha256 '"$SOURCE_FEATURE_MANIFEST_SHA"' --source-metadata-only --fused-qkv-a --destination '"$DESTINATION"' --run-dir "$run" --expected-code-hash '"$PIN"' --topology-capture "$capture" --stage-id 0 '"$RESUME_FLAG"' >"$run/pack.log" 2>&1; gcloud storage cp --recursive --no-clobber "$run" '"$REMOTE_PREFIX"'/probe/ >/dev/null; echo "PROBE_OK $(hostname)"'
say "running one exact stage-0 probe before fleet fanout"
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$probe_command" >"$RUN_DIR/probe.txt" 2>&1
[[ $(awk '$1=="PROBE_OK" {print $2}' "$RUN_DIR/probe.txt" | sort -u | wc -l) -eq 1 ]]
[[ $(awk '$1=="PROBE_SKIP" {print $2}' "$RUN_DIR/probe.txt" | sort -u | wc -l) -eq 7 ]]
strict_census probe_post || { say "ABORT: probe left fleet work"; exit 1; }

# Each host packs its two PP16 stages. Resume authenticates the sealed probe.
# shellcheck disable=SC2016
pack_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; capture='"$TOPOLOGY_RUN"'/topology.rank${idx}.json; jax=$(/home/gianl/vllm-env/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))[\"jax_process_index\"])" "$capture"); run=/home/gianl/glm-run/'"$TAG"'/host_pack; mkdir -p "$run"; cd '"$WORKTREE"'; for stage in $((2*jax)) $((2*jax+1)); do timeout --signal=TERM --kill-after=60 21600 /home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py pack-stage --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --source-runtime-root '"$PARENT_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$PARENT_RUNTIME_MANIFEST_SHA"' --source-feature-runtime-root '"$SOURCE_FEATURE_ROOT"' --source-feature-runtime-manifest-sha256 '"$SOURCE_FEATURE_MANIFEST_SHA"' --source-metadata-only --fused-qkv-a --destination '"$DESTINATION"' --run-dir "$run" --expected-code-hash '"$PIN"' --topology-capture "$capture" --stage-id "$stage" --resume >"$run/stage_${stage}.log" 2>&1; done; gcloud storage cp --recursive --no-clobber "$run" '"$REMOTE_PREFIX"'/host_records/worker${idx}/ >/dev/null; echo "PACK_HOST_OK $(hostname)"'
say "streaming all 16 stages from the retained feature runtime"
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$pack_command" >"$RUN_DIR/pack.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/pack.txt" PACK_HOST_OK || { say "ABORT: fanout failed"; exit 1; }

say "finalizing only after all 32 files and tensor ledgers reconcile"
/home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py \
  finalize "${COMMON_ARGS[@]}" --run-dir "$RUN_DIR/final" \
  >"$RUN_DIR/finalize.txt" 2>&1
RUNTIME_MANIFEST_SHA=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$RUN_DIR/final/runtime_manifest.json")
readonly RUNTIME_MANIFEST_SHA

say "independently verifying the mounted final artifact"
/home/gianl/vllm-env/bin/python scripts/greenfield/inspect_feature_runtime_checkpoint.py \
  --runtime-root "$CHECKPOINT_ROOT" \
  --source-checkpoint-root "$SOURCE_ROOT" \
  --source-packed-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --source-runtime-root "$PARENT_RUNTIME_ROOT" \
  --source-runtime-manifest-sha256 "$PARENT_RUNTIME_MANIFEST_SHA" \
  --source-feature-runtime-root "$SOURCE_FEATURE_ROOT" \
  --source-feature-runtime-manifest-sha256 "$SOURCE_FEATURE_MANIFEST_SHA" \
  --source-metadata-only --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA" \
  --output "$RUN_DIR/verification.json" >"$RUN_DIR/inspect.txt" 2>&1

strict_census post || { say "ABORT: post-pack fleet is not 8/8 clean"; exit 1; }
post_census_done=1
say "sealing compact evidence in the approved same-region bucket"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
sha256sum "$RUN_DIR"/*.txt "$RUN_DIR/verification.json" \
  "$RUN_DIR/orchestrator.sealed.log" >"$RUN_DIR/evidence.sha256"
printf '%s  runtime_manifest.json\n' "$RUNTIME_MANIFEST_SHA" >"$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" --no-clobber >/dev/null
terminal_success_done=1
trap - EXIT
echo "PP16_FEATURE_QKV_DIRECT_OK $TAG $RUNTIME_MANIFEST_SHA"
