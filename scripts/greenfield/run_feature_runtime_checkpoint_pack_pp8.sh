#!/usr/bin/env bash
# Protected distributed derivative from complete-expert runtime to feature owners.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly SOURCE_PACKED_MANIFEST_SHA=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly SOURCE_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP8_LP4/greenfield_runtime_pack_pp8_20260806T002756318310857Z
readonly SOURCE_RUNTIME_MANIFEST_SHA=fdedaae31fb3c094266272ed48dfe62bb098257a78272b93c14eafbd57e31dec
readonly TOPOLOGY_RUN=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
readonly FUSED_QKV_A=${GLM_GREENFIELD_FEATURE_RUNTIME_FUSED_QKV_A:-0}
readonly DENSE_CONVOLUTION=${GLM_GREENFIELD_FEATURE_RUNTIME_DENSE_CONVOLUTION:-0}
[[ $FUSED_QKV_A == 0 || $FUSED_QKV_A == 1 ]] || {
  echo "feature-runtime fused qkv-a flag must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_CONVOLUTION == 0 || $DENSE_CONVOLUTION == 1 ]] || {
  echo "feature-runtime dense convolution flag must be 0 or 1" >&2
  exit 2
}
if [[ $DENSE_CONVOLUTION == 1 && $FUSED_QKV_A != 1 ]]; then
  echo "dense convolution runtime requires fused qkv-a" >&2
  exit 2
fi
FUSED_QKV_A_FLAG=
DENSE_CONVOLUTION_FLAG=
DEFAULT_TAG=greenfield_runtime_feature_pack_pp8_$(date -u +%Y%m%dT%H%M%S%NZ)
if [[ $FUSED_QKV_A == 1 ]]; then
  FUSED_QKV_A_FLAG=--fused-qkv-a
  DEFAULT_TAG=greenfield_runtime_feature_qkv_pack_pp8_$(date -u +%Y%m%dT%H%M%S%NZ)
fi
if [[ $DENSE_CONVOLUTION == 1 ]]; then
  DENSE_CONVOLUTION_FLAG=--dense-convolution
  DEFAULT_TAG=greenfield_runtime_feature_qkv_dense_pack_pp8_$(date -u +%Y%m%dT%H%M%S%NZ)
fi
TAG=${GLM_GREENFIELD_FEATURE_RUNTIME_PACK_TAG:-$DEFAULT_TAG}
RUN_DIR=/home/gianl/glm-run/$TAG
CHECKPOINT_DESTINATION=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$TAG
CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
RESUME_FLAG=
if [[ ${GLM_GREENFIELD_FEATURE_RUNTIME_PACK_RESUME:-0} == 1 ]]; then
  RESUME_FLAG=--resume
fi

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing feature runtime pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing feature runtime pack from a dirty worktree" >&2
  exit 2
}
[[ -f $SOURCE_CHECKPOINT_ROOT/SUCCESS && -f $SOURCE_CHECKPOINT_ROOT/packed_manifest.json ]] || {
  echo "protected final-owner checkpoint is unavailable" >&2
  exit 2
}
[[ -f $SOURCE_RUNTIME_ROOT/SUCCESS && -f $SOURCE_RUNTIME_ROOT/runtime_manifest.json ]] || {
  echo "protected executable runtime checkpoint is unavailable" >&2
  exit 2
}
if [[ -z $RESUME_FLAG && -e $RUN_DIR ]]; then
  echo "append-only run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

say() {
  echo "[feature-runtime-pack-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[l]oad_full_checkpoint_stage[.]py|[p]ack_runtime_checkpoint[.]py|[p]ack_feature_runtime_checkpoint[.]py|[r]un_short_decode[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
say "PIN=$PIN"
say "SOURCE_RUNTIME=$SOURCE_RUNTIME_MANIFEST_SHA"
say "DESTINATION=$CHECKPOINT_DESTINATION"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact code and immutable runtime/topology prerequisites"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; source_checkpoint='"$SOURCE_CHECKPOINT_ROOT"'; source_runtime='"$SOURCE_RUNTIME_ROOT"'; topology='"$TOPOLOGY_RUN"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin '"$BRANCH"'; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch '"$BRANCH"' "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; capture="$topology/topology.rank${idx}.json"; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source_checkpoint/SUCCESS" ]] && [[ -r "$source_runtime/SUCCESS" ]] && [[ -r "$capture" ]] && findmnt -T "$source_checkpoint" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$source_runtime" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code/artifact sync failed"
  exit 1
}

COMMON_ARGS=(
  --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT"
  --source-packed-manifest-sha256 "$SOURCE_PACKED_MANIFEST_SHA"
  --source-runtime-root "$SOURCE_RUNTIME_ROOT"
  --source-runtime-manifest-sha256 "$SOURCE_RUNTIME_MANIFEST_SHA"
  --destination "$CHECKPOINT_DESTINATION"
  --expected-code-hash "$PIN"
)
RESUME_ARGS=()
if [[ -n $RESUME_FLAG ]]; then
  RESUME_ARGS+=("$RESUME_FLAG")
fi
FUSED_ARGS=()
if [[ -n $FUSED_QKV_A_FLAG ]]; then
  FUSED_ARGS+=("$FUSED_QKV_A_FLAG")
fi
DENSE_ARGS=()
if [[ -n $DENSE_CONVOLUTION_FLAG ]]; then
  DENSE_ARGS+=("$DENSE_CONVOLUTION_FLAG")
fi
say "preparing content-addressed feature-runtime layout and empty destination"
cd "$WORKTREE"
/home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py \
  prepare "${COMMON_ARGS[@]}" --run-dir "$RUN_DIR/pack_control" \
  "${RESUME_ARGS[@]}" "${FUSED_ARGS[@]}" \
  "${DENSE_ARGS[@]}" \
  >"$RUN_DIR/prepare.txt" 2>&1

say "authenticating and streaming eight host-local stages"
# shellcheck disable=SC2016
pack_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; source_checkpoint='"$SOURCE_CHECKPOINT_ROOT"'; source_packed_sha='"$SOURCE_PACKED_MANIFEST_SHA"'; source_runtime='"$SOURCE_RUNTIME_ROOT"'; source_runtime_sha='"$SOURCE_RUNTIME_MANIFEST_SHA"'; topology='"$TOPOLOGY_RUN"'; destination='"$CHECKPOINT_DESTINATION"'; pin='"$PIN"'; resume='"$RESUME_FLAG"'; fused='"$FUSED_QKV_A_FLAG"'; dense='"$DENSE_CONVOLUTION_FLAG"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag/host_pack; capture="$topology/topology.rank${idx}.json"; mkdir -p "$run"; cd "$wt"; set +e; timeout --signal=TERM --kill-after=60 21600 /home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py pack-stage --source-checkpoint-root "$source_checkpoint" --source-packed-manifest-sha256 "$source_packed_sha" --source-runtime-root "$source_runtime" --source-runtime-manifest-sha256 "$source_runtime_sha" --destination "$destination" --run-dir "$run" --expected-code-hash "$pin" --topology-capture "$capture" $resume $fused $dense >"$run/pack.log" 2>&1; rc=$?; set -e; gcloud storage cp --recursive --no-clobber "$run" "$remote/host_records/worker${idx}/" >/dev/null 2>&1 || true; if [[ $rc -eq 0 ]]; then echo "PACK_HOST_OK $(hostname)"; else tail -100 "$run/pack.log" >&2 || true; echo "PACK_HOST_FAILED $(hostname) rc=$rc" >&2; exit "$rc"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$pack_command" >"$RUN_DIR/pack.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/pack.txt" PACK_HOST_OK || {
  say "ABORT: distributed feature-runtime stream did not complete 8/8"
  exit 1
}

say "finalizing only after all 32 payloads and transform sidecars reconcile"
/home/gianl/vllm-env/bin/python scripts/greenfield/pack_feature_runtime_checkpoint.py \
  finalize "${COMMON_ARGS[@]}" --run-dir "$RUN_DIR/final" \
  "${FUSED_ARGS[@]}" "${DENSE_ARGS[@]}" \
  >"$RUN_DIR/finalize.txt" 2>&1
RUNTIME_MANIFEST_SHA=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$RUN_DIR/final/runtime_manifest.json")

say "verifying the complete mounted feature-runtime artifact"
/home/gianl/vllm-env/bin/python scripts/greenfield/inspect_feature_runtime_checkpoint.py \
  --runtime-root "$CHECKPOINT_ROOT" \
  --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
  --source-packed-manifest-sha256 "$SOURCE_PACKED_MANIFEST_SHA" \
  --source-runtime-root "$SOURCE_RUNTIME_ROOT" \
  --source-runtime-manifest-sha256 "$SOURCE_RUNTIME_MANIFEST_SHA" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA" \
  --output "$RUN_DIR/verification.json" >"$RUN_DIR/inspect.txt" 2>&1

strict_census post || {
  say "ABORT: post-pack census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "sealing local evidence and approved-bucket result archive"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
sha256sum \
  "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/prepare.txt" "$RUN_DIR/pack.txt" \
  "$RUN_DIR/finalize.txt" "$RUN_DIR/inspect.txt" "$RUN_DIR/verification.json" \
  "$RUN_DIR/orchestrator.sealed.log" >"$RUN_DIR/evidence.sha256"
printf '%s  runtime_manifest.json\n' "$RUNTIME_MANIFEST_SHA" >"$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" --no-clobber >/dev/null
trap - EXIT
echo "FEATURE_RUNTIME_PACK_OK $TAG $RUNTIME_MANIFEST_SHA"
