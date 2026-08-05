#!/usr/bin/env bash
# Append-only real layer-3 PP8/PP16 pack. Opens only source shards 38-40.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PP8_GROUP_HASH=d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14
readonly PP16_GROUP_HASH=6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21

PLAN_ID=${GLM_GREENFIELD_ONE_LAYER_PLAN:-PP8_LP4}
case "$PLAN_ID" in
  PP8_LP4)
    PLAN_SLUG=pp8
    STAGE_SIZE=4
    PLAN_GROUP_HASH=$PP8_GROUP_HASH
    ;;
  PP16_LP2)
    PLAN_SLUG=pp16
    STAGE_SIZE=2
    PLAN_GROUP_HASH=$PP16_GROUP_HASH
    ;;
  *)
    echo "unsupported bounded pack plan: $PLAN_ID" >&2
    exit 2
    ;;
esac
readonly PLAN_ID PLAN_SLUG STAGE_SIZE PLAN_GROUP_HASH

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ONE_LAYER_PACK_TAG:-greenfield_one_layer_pack_${PLAN_SLUG}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
PACK_DIR=$RUN_DIR/packed
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/$PLAN_ID/$TAG

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
  echo "[one-layer-pack-$PLAN_SLUG $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_URI" "$RUN_DIR/source_objects.json" "${objects[@]}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

source_uri, output, *objects = sys.argv[1:]
records = []
for object_name in objects:
    completed = subprocess.run(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            f"{source_uri}/{object_name}",
            "--format=json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)
    if record.get("name") != f"models/GLM-5.2-FP8/{object_name}":
        raise SystemExit(f"wrong source object identity for {object_name}")
    if not record.get("generation") or not record.get("size"):
        raise SystemExit(f"incomplete source object metadata for {object_name}")
    records.append(record)
Path(output).write_text(
    json.dumps({"objects": sorted(records, key=lambda item: item["name"])},
               indent=2,
               sort_keys=True) + "\n"
)
PY
source_revision=gcs-object-set-$(sha256sum "$RUN_DIR/source_objects.json" | awk '{print $1}')
say "SOURCE_REVISION=$source_revision"

say "packing exact layer-3 ownership; complete model construction is forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python scripts/greenfield/pack_one_layer_moe.py \
    --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_URI" \
    --source-revision "$source_revision" \
    --output "$PACK_DIR" \
    --layer 3 \
    --expected-code-hash "$PIN" \
    --topology-hash "$TOPOLOGY_HASH" \
    --plan-group-hash "$PLAN_GROUP_HASH" \
    --plan-id "$PLAN_ID" \
    --stage-size "$STAGE_SIZE"
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
  /home/gianl/vllm-env/bin/python - <<'PY'
import json

manifest = json.load(open("packed/manifest.json"))
for record in manifest["files"]:
    print(f"{record['sha256']}  packed/{record['filename']}")
PY
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
[[ $remote_files -eq $STAGE_SIZE ]] || {
  say "ABORT: remote artifact has $remote_files/$STAGE_SIZE device files"
  exit 1
}

manifest_hash=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$PACK_DIR/manifest.json")
say "SUCCESS manifest=$manifest_hash"
say "No TPU/model-performance claim: this is a checkpoint-layout artifact only."
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/orchestrator.log" \
  "$REMOTE_PREFIX/orchestrator.log" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
