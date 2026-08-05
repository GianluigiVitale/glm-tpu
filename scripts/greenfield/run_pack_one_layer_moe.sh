#!/usr/bin/env bash
# Append-only real layer-3 PP8 pack. This opens only source shards 38-40.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PP8_GROUP_HASH=d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ONE_LAYER_PACK_TAG:-greenfield_one_layer_pack_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
PACK_DIR=$RUN_DIR/packed
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/PP8_LP4/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing pack from a dirty greenfield worktree" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/model.safetensors.index.json ]] || {
  echo "source checkpoint index is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory already exists: $RUN_DIR" >&2
  exit 2
}

mkdir -p "$RUN_DIR"
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

say() {
  echo "[one-layer-pack $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN"
say "REMOTE_PREFIX=$REMOTE_PREFIX"
say "capturing immutable GCS object provenance"

objects=(
  config.json
  model.safetensors.index.json
  model-00038-of-00141.safetensors
  model-00039-of-00141.safetensors
  model-00040-of-00141.safetensors
)
for object in "${objects[@]}"; do
  gcloud storage objects describe "$SOURCE_URI/$object" --format=json
done | jq -s '{objects: sort_by(.name)}' >"$RUN_DIR/source_objects.json"

[[ $(jq '.objects | length' "$RUN_DIR/source_objects.json") -eq 5 ]] || {
  say "ABORT: source object provenance is incomplete"
  exit 1
}
source_revision=gcs-object-set-$(sha256sum "$RUN_DIR/source_objects.json" | awk '{print $1}')
say "SOURCE_REVISION=$source_revision"

say "packing exact layer-3 ownership; complete model construction is forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  /home/gianl/vllm-env/bin/python scripts/greenfield/pack_one_layer_moe.py \
    --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_URI" \
    --source-revision "$source_revision" \
    --output "$PACK_DIR" \
    --layer 3 \
    --expected-code-hash "$PIN" \
    --topology-hash "$TOPOLOGY_HASH" \
    --plan-group-hash "$PP8_GROUP_HASH"
) >"$RUN_DIR/pack.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "pack completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$PACK_DIR" <<'PY' >"$RUN_DIR/inspection.json"
from pathlib import Path
import json
import sys

from glm_tpu.greenfield.checkpoint.one_layer import inspect_one_layer_artifact

manifest = inspect_one_layer_artifact(Path(sys.argv[1]))
print(json.dumps({
    "files": len(manifest["files"]),
    "manifest_sha256": manifest["manifest_sha256"],
    "packed_payload_byte_count": manifest["packed_payload_byte_count"],
    "source_index_sha256": manifest["source_index_sha256"],
    "source_leaves": len(manifest["source_leaves"]),
    "source_payload_byte_count": manifest["source_payload_byte_count"],
    "source_revision": manifest["source_revision"],
}, indent=2, sort_keys=True))
PY

say "writing evidence checksums without rereading packed payloads"
(
  cd "$RUN_DIR"
  sha256sum source_objects.json pack.log inspection.json packed/manifest.json
  jq -r '.files[] | "\(.sha256)  packed/\(.filename)"' packed/manifest.json
) >"$RUN_DIR/evidence.sha256"

say "uploading append-only artifact to approved bucket"
gcloud storage cp --no-clobber \
  "$RUN_DIR/source_objects.json" \
  "$RUN_DIR/pack.log" \
  "$RUN_DIR/inspection.json" \
  "$RUN_DIR/evidence.sha256" \
  "$PACK_DIR/manifest.json" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$PACK_DIR"/device_slot_*.safetensors \
  "$REMOTE_PREFIX/packed/" >/dev/null

remote_files=$(gcloud storage ls "$REMOTE_PREFIX/packed/device_slot_*.safetensors" | wc -l)
[[ $remote_files -eq 4 ]] || {
  say "ABORT: remote artifact has $remote_files/4 device files"
  exit 1
}

say "SUCCESS manifest=$(jq -r .manifest_sha256 "$PACK_DIR/manifest.json")"
say "No TPU/model-performance claim: this is a checkpoint-layout artifact only."
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/orchestrator.log" \
  "$REMOTE_PREFIX/orchestrator.log" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
