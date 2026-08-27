#!/usr/bin/env bash
# Append-only PP8/PP16 layer-3 expert-feature derivative; no TPU execution.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc

PLAN_ID=${GLM_GREENFIELD_ONE_LAYER_PALLAS_FEATURE_PLAN:-PP8_LP4}
case "$PLAN_ID" in
  PP8_LP4)
    SOURCE_TAG=greenfield_one_layer_pallas_pack_20260806T041854316280053Z
    SOURCE_RUN=/home/gianl/glm-run/$SOURCE_TAG
    SOURCE_MANIFEST_SHA256=3da63bd9c2332dd67fc29a1d158e5468e0fdf1db1a9a0e8b7977277b0812e427
    PLAN_SLUG=pp8
    ;;
  PP16_LP2)
    SOURCE_RUN=${GLM_GREENFIELD_ONE_LAYER_PALLAS_SOURCE_RUN:-}
    SOURCE_MANIFEST_SHA256=${GLM_GREENFIELD_ONE_LAYER_PALLAS_SOURCE_MANIFEST_SHA256:-}
    [[ -n $SOURCE_RUN && -n $SOURCE_MANIFEST_SHA256 ]] || {
      echo "PP16 feature pack requires its local Pallas source run and manifest SHA" >&2
      exit 2
    }
    SOURCE_TAG=$(basename "$SOURCE_RUN")
    PLAN_SLUG=pp16
    ;;
  *)
    echo "unsupported one-layer feature plan: $PLAN_ID" >&2
    exit 2
    ;;
esac
SOURCE_ARTIFACT=$SOURCE_RUN/packed
SOURCE_ARTIFACT_URI=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/$PLAN_ID/pallas_final/$SOURCE_TAG
readonly PLAN_ID SOURCE_TAG SOURCE_RUN SOURCE_ARTIFACT
readonly SOURCE_MANIFEST_SHA256 SOURCE_ARTIFACT_URI PLAN_SLUG

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ONE_LAYER_PALLAS_FEATURE_TAG:-greenfield_one_layer_pallas_feature_pack_${PLAN_SLUG}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
PACK_DIR=$RUN_DIR/packed
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/$PLAN_ID/pallas_feature/$TAG
readonly PIN TAG RUN_DIR PACK_DIR REMOTE_PREFIX

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing feature pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing feature pack from a dirty worktree" >&2
  exit 2
}
bucket_location=$(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)')
[[ $bucket_location == US-CENTRAL2 ]] || {
  echo "approved bucket location drifted: $bucket_location" >&2
  exit 2
}
[[ -r $SOURCE_ARTIFACT/manifest.json ]] || {
  echo "source Pallas artifact is unavailable" >&2
  exit 2
}
observed_source_manifest=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$SOURCE_ARTIFACT/manifest.json")
[[ $observed_source_manifest == "$SOURCE_MANIFEST_SHA256" ]] || {
  echo "source Pallas manifest identity drifted" >&2
  exit 2
}
active_account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)')
[[ -n $active_account ]] || {
  echo "gcloud has no active authenticated account" >&2
  exit 2
}
source_payload_bytes=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["packed_payload_byte_count"])' \
  "$SOURCE_ARTIFACT/manifest.json")
available_bytes=$(df -B1 --output=avail /home/gianl/glm-run | tail -n 1)
required_bytes=$(( source_payload_bytes + 2147483648 ))
[[ $available_bytes -ge $required_bytes ]] || {
  echo "insufficient disk for append-only feature derivative: available=$available_bytes required=$required_bytes" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only feature run already exists: $RUN_DIR" >&2
  exit 2
}

mkdir -p "$RUN_DIR"
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

say() {
  echo "[one-layer-pallas-feature $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN SOURCE_MANIFEST_SHA256=$SOURCE_MANIFEST_SHA256"
say "REMOTE_PREFIX=$REMOTE_PREFIX"
say "offline redistributing expert identities over exact $PLAN_ID feature owners; TPU forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    scripts/greenfield/pack_one_layer_pallas_feature.py \
    --source-artifact "$SOURCE_ARTIFACT" \
    --source-artifact-uri "$SOURCE_ARTIFACT_URI" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --output "$PACK_DIR" \
    --expected-code-hash "$PIN"
) >"$RUN_DIR/pack.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "pack and source-transform inspection completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PACK_DIR/manifest.json" <<'PY' >"$RUN_DIR/inspection.json"
from pathlib import Path
import json
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({
    "files": len(manifest["files"]),
    "layout_sha256": manifest["layout"]["layout_sha256"],
    "manifest_sha256": manifest["manifest_sha256"],
    "packed_payload_byte_count": manifest["packed_payload_byte_count"],
    "source_layout_sha256": manifest["source_layout_sha256"],
    "source_manifest_sha256": manifest["source_manifest_sha256"],
}, indent=2, sort_keys=True))
PY

say "writing sealed evidence ledger"
(
  cd "$RUN_DIR"
  sha256sum pack.log inspection.json packed/manifest.json
  /home/gianl/vllm-env/bin/python - <<'PY'
import json
manifest = json.load(open("packed/manifest.json"))
for record in manifest["files"]:
    print(f"{record['sha256']}  packed/{record['filename']}")
PY
) >"$RUN_DIR/evidence.sha256"

say "uploading immutable feature derivative"
gcloud storage cp --no-clobber \
  "$RUN_DIR/pack.log" "$RUN_DIR/inspection.json" \
  "$RUN_DIR/evidence.sha256" "$PACK_DIR/manifest.json" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$PACK_DIR"/device_slot_*.safetensors \
  "$REMOTE_PREFIX/packed/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PACK_DIR/manifest.json" "$REMOTE_PREFIX" <<'PY' >"$RUN_DIR/remote_objects.json"
from pathlib import Path
import json
import subprocess
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
remote = sys.argv[2]
objects = [("manifest.json", Path(sys.argv[1]).stat().st_size)] + [
    (f"packed/{item['filename']}", item["file_byte_count"])
    for item in manifest["files"]
]
records = []
for name, expected_size in objects:
    completed = subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{remote}/{name}", "--format=json"],
        check=True, capture_output=True, text=True,
    )
    value = json.loads(completed.stdout)
    if int(value["size"]) != int(expected_size) or not value.get("generation") or not value.get("crc32c"):
        raise SystemExit(f"remote feature identity drift for {name}")
    records.append({
        "crc32c": value["crc32c"], "generation": value["generation"],
        "name": name, "size": int(value["size"]),
    })
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY

manifest_hash=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$PACK_DIR/manifest.json")
say "SUCCESS manifest=$manifest_hash"
say "No TPU/layer/token-speed claim: this is feature-layout checkpoint evidence."
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/remote_objects.json" \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/" >/dev/null
