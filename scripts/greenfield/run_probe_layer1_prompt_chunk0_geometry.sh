#!/usr/bin/env bash
# Bounded one-host probe: layer-1 prompt keys of chunk 0 in the legacy prefill
# geometry, compared with the sealed legacy layer-1 prompt cache. Serialized
# under the pod lease with eight-host zero-work censuses; executes committed
# bytes from a run-owned detached worktree of the origin-authenticated pin;
# append-only results with three-surface vacancy, generation-bound ledger and
# exact remote-set validation. Invoke sanitized:
#   /usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin:/snap/bin LANG=C LC_ALL=C \
#     GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG=<approved tag> \
#     /bin/bash /home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf
readonly WEIGHT_DIGESTS_RELATIVE=docs/artifacts/gate-d-chunk0-probe-weight-digests.json
readonly WEIGHT_DIGESTS_SHA=5a49ab9a8c6a6dc7ee41cf104856e4709c849b0c42cb68d00e52c33241552463
readonly LEGACY_LAYER1_CACHE_DIR=/home/gianl/glm-run/greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z/prompt_index_cache
readonly LEGACY_LAYER1_MANIFEST_SHA=d9058cc6584aca784212706789e72b4981754cb9880e553122b5e854bc3961ac
readonly DB518_RESULT=/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/result.npz
readonly DB518_RESULT_SHA=534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0
readonly PYTHON=/home/gianl/vllm-env/bin/python
readonly GCLOUD=/snap/bin/gcloud

TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-}
[[ -n $TAG && $TAG =~ ^greenfield_layer1_prompt_chunk0_geometry_[0-9TZ]+$ ]] || {
  echo "GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG must be an approved composed tag" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing probe outside $BRANCH" >&2
  exit 2
}
PIN=$(git -C "$WORKTREE" rev-parse HEAD)
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing probe from a dirty canonical worktree" >&2
  exit 2
}
# Origin authentication: the pin must be the exact tip of the branch on origin.
[[ $(git -C "$WORKTREE" remote get-url origin) == "$GREENFIELD_ORIGIN" ]] || {
  echo "canonical worktree origin drifted" >&2
  exit 2
}
git -C "$WORKTREE" fetch -q origin "$BRANCH"
[[ $(git -C "$WORKTREE" rev-parse "origin/$BRANCH") == "$PIN" ]] || {
  echo "canonical HEAD is not the origin tip of $BRANCH" >&2
  exit 2
}
[[ -r $INPUT_DIR/manifest.json && -r $CHECKPOINT_ROOT/model.safetensors.index.json &&
   -r $LEGACY_LAYER1_CACHE_DIR/manifest.json && -r $DB518_RESULT && ! -e $RUN_DIR ]] || {
  echo "inputs missing or append-only run path already exists" >&2
  exit 2
}
[[ $(sha256sum "$CHECKPOINT_ROOT/model.safetensors.index.json" | awk '{print $1}') == "$CHECKPOINT_INDEX_SHA" ]] || {
  echo "checkpoint index digest drifted" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo" "$RUN_DIR/probe"

say() {
  echo "[chunk0-geometry $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}

# Three-surface remote vacancy: no object under the prefix (results, diagnostic
# or any other suffix) may exist before this run.
"$GCLOUD" storage ls "$REMOTE_PREFIX/**" >"$RUN_DIR/remote_vacancy.txt" 2>"$RUN_DIR/remote_vacancy.stderr" || true
if [[ -s $RUN_DIR/remote_vacancy.txt ]] || ! grep -q "matched no objects" "$RUN_DIR/remote_vacancy.stderr"; then
  say "ABORT: remote prefix is not vacant or vacancy could not be proven"
  exit 1
fi

# Execute committed bytes from a run-owned detached worktree of the exact pin.
SOURCE=$RUN_DIR/source
git -C "$WORKTREE" worktree add -q --detach "$SOURCE" "$PIN"
[[ $(git -C "$SOURCE" rev-parse HEAD) == "$PIN" && -z $(git -C "$SOURCE" status --porcelain --ignored) ]] || {
  say "ABORT: detached source worktree does not match the pin"
  exit 1
}
[[ $(sha256sum "$SOURCE/$WEIGHT_DIGESTS_RELATIVE" | awk '{print $1}') == "$WEIGHT_DIGESTS_SHA" ]] || {
  say "ABORT: pinned weight digest record drifted in the committed source"
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
  "$GCLOUD" compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    if ! strict_census failure_exit; then
      say "CENSUS_UNVERIFIED: failure-exit census is not an authenticated 8/8 zero-work state"
      touch "$RUN_DIR/CENSUS_UNVERIFIED"
    fi
  fi
  git -C "$WORKTREE" worktree remove --force "$SOURCE" >/dev/null 2>&1 || true
  git -C "$WORKTREE" worktree prune >/dev/null 2>&1 || true
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    "$GCLOUD" storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN input=$INPUT_MANIFEST_SHA legacy_layer1=$LEGACY_LAYER1_MANIFEST_SHA db518=$DB518_RESULT_SHA checkpoint_index=$CHECKPOINT_INDEX_SHA weight_digests=$WEIGHT_DIGESTS_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running bounded chunk-0 legacy-geometry probe on one TPU-v4 host from the detached source"
started=$(date +%s)
set +e
/usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$SOURCE" \
  /usr/bin/timeout --signal=TERM --kill-after=60 3600 \
  "$PYTHON" "$SOURCE/scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py" \
    --expected-code-hash "$PIN" \
    --run-tag "$TAG" \
    --input-dir "$INPUT_DIR" \
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --checkpoint-index-sha256 "$CHECKPOINT_INDEX_SHA" \
    --weight-digests "$SOURCE/$WEIGHT_DIGESTS_RELATIVE" \
    --weight-digests-sha256 "$WEIGHT_DIGESTS_SHA" \
    --legacy-layer1-cache-dir "$LEGACY_LAYER1_CACHE_DIR" \
    --legacy-layer1-manifest-sha256 "$LEGACY_LAYER1_MANIFEST_SHA" \
    --db518-result "$DB518_RESULT" \
    --db518-result-sha256 "$DB518_RESULT_SHA" \
    --output "$RUN_DIR/probe" \
    --hlo-dir "$RUN_DIR/hlo" \
  >"$RUN_DIR/runner.log" 2>&1
probe_status=$?
set -e
elapsed=$(( $(date +%s) - started ))
say "probe exited status=$probe_status in ${elapsed}s"

# The detached source is transport, not evidence: remove it before sealing.
git -C "$WORKTREE" worktree remove --force "$SOURCE"
git -C "$WORKTREE" worktree prune

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
[[ $probe_status -eq 0 ]] || {
  say "probe FAILED (status $probe_status); diagnostics preserved, no SUCCESS"
  exit 1
}
(
  cd "$RUN_DIR"
  find hlo probe -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.log orchestrator.log census_pre.txt census_post.txt remote_vacancy.txt remote_vacancy.stderr
) >"$RUN_DIR/evidence.sha256"
touch "$RUN_DIR/SUCCESS"

say "uploading append-only results and building the generation-bound ledger"
"$GCLOUD" storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
# Describe every uploaded object (size, CRC32C, generation) and seal the
# ledger as the terminal-last object, then validate the exact remote set.
PYTHONPATH="$WORKTREE" "$PYTHON" - "$RUN_DIR" "$REMOTE_PREFIX" "$GCLOUD" <<'PY' >"$RUN_DIR/remote_objects.json"
import base64, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import google_crc32c
root, prefix, gcloud = Path(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
paths = sorted((p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file() and p.name != "remote_objects.json")
def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()
def describe(item):
    path, relative = item
    remote = json.loads(subprocess.run([gcloud, "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"], check=True, capture_output=True, text=True).stdout)
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if int(remote["size"]) != path.stat().st_size or crc32c != local_crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {"crc32c": crc32c, "generation": remote["generation"], "path": relative, "size": int(remote["size"])}
with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
"$GCLOUD" storage cp --no-clobber "$RUN_DIR/remote_objects.json" "$REMOTE_PREFIX/remote_objects.json" >/dev/null
[[ $(sha256sum "$RUN_DIR/remote_objects.json" | awk '{print $1}') == \
   $("$GCLOUD" storage cat "$REMOTE_PREFIX/remote_objects.json" | sha256sum | awk '{print $1}') ]] || {
  say "ABORT: terminal ledger checksum mismatch"
  exit 1
}
PYTHONPATH="$WORKTREE" "$PYTHON" - "$RUN_DIR" "$REMOTE_PREFIX" "$GCLOUD" <<'PY'
import subprocess, sys
from pathlib import Path
from glm_tpu.greenfield.validation.attention_update import validate_exact_remote_object_set
root, prefix, gcloud = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
listing = subprocess.run([gcloud, "storage", "ls", f"{prefix}/**"], check=True, capture_output=True, text=True).stdout.splitlines()
validate_exact_remote_object_set(root, prefix, listing)
print("REMOTE_SET_EXACT", len(listing))
PY
say "SUCCESS tag=$TAG remote=$REMOTE_PREFIX"
