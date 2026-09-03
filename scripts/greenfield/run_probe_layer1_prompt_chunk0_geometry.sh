#!/usr/bin/env bash
# Bounded one-host probe: layer-1 prompt keys of chunk 0 in the legacy prefill
# geometry, compared with the sealed legacy layer-1 prompt cache. Serialized
# under the pod lease with eight-host zero-work censuses; append-only results.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly LEGACY_LAYER1_CACHE_DIR=/home/gianl/glm-run/greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z/prompt_index_cache
readonly LEGACY_LAYER1_MANIFEST_SHA=d9058cc6584aca784212706789e72b4981754cb9880e553122b5e854bc3961ac
readonly DB518_RESULT=/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/result.npz
readonly DB518_RESULT_SHA=534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-greenfield_layer1_prompt_chunk0_geometry_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $TAG =~ ^[A-Za-z0-9_]+$ ]] || { echo "unsafe tag" >&2; exit 2; }
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing probe from a dirty worktree" >&2
  exit 2
}
[[ -r $INPUT_DIR/manifest.json && -r $CHECKPOINT_ROOT/model.safetensors.index.json &&
   -r $LEGACY_LAYER1_CACHE_DIR/manifest.json && -r $DB518_RESULT && ! -e $RUN_DIR ]] || {
  echo "inputs missing or append-only run path already exists" >&2
  exit 2
}
CHECKPOINT_INDEX_SHA=$(sha256sum "$CHECKPOINT_ROOT/model.safetensors.index.json" | awk '{print $1}')
mkdir -p "$RUN_DIR/hlo" "$RUN_DIR/probe"

say() {
  echo "[chunk0-geometry $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; ray_pids=$(pgrep -f "ray::|raylet|gcs_server|EngineCore" | wc -l); tpu_users=$(sudo -n fuser /dev/accel* 2>/dev/null | wc -w); if [ "$tools_ok" = 1 ] && [ "$ray_pids" = 0 ] && [ "$tpu_users" = 0 ]; then echo "CENSUS_OK $(hostname)"; else echo "CENSUS_BAD $(hostname) ray=$ray_pids tpu=$tpu_users tools=$tools_ok"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
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

say "RUN_DIR=$RUN_DIR PIN=$PIN input=$INPUT_MANIFEST_SHA legacy_layer1=$LEGACY_LAYER1_MANIFEST_SHA db518=$DB518_RESULT_SHA checkpoint_index=$CHECKPOINT_INDEX_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running bounded chunk-0 legacy-geometry probe on one TPU-v4 host"
started=$(date +%s)
set +e
(
  cd "$WORKTREE"
  /usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /usr/bin/timeout --signal=TERM --kill-after=60 3600 \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py \
      --expected-code-hash "$PIN" \
      --run-tag "$TAG" \
      --input-dir "$INPUT_DIR" \
      --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --checkpoint-index-sha256 "$CHECKPOINT_INDEX_SHA" \
      --legacy-layer1-cache-dir "$LEGACY_LAYER1_CACHE_DIR" \
      --legacy-layer1-manifest-sha256 "$LEGACY_LAYER1_MANIFEST_SHA" \
      --db518-result "$DB518_RESULT" \
      --db518-result-sha256 "$DB518_RESULT_SHA" \
      --output "$RUN_DIR/probe" \
      --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
probe_status=$?
set -e
elapsed=$(( $(date +%s) - started ))
say "probe exited status=$probe_status in ${elapsed}s"

(
  cd "$RUN_DIR"
  find hlo probe -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.log census_pre.txt
) >"$RUN_DIR/evidence.sha256"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
sha256sum "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"
[[ $probe_status -eq 0 ]] || {
  say "probe FAILED (status $probe_status); diagnostics preserved, no SUCCESS"
  exit 1
}
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
say "SUCCESS tag=$TAG remote=$REMOTE_PREFIX"
