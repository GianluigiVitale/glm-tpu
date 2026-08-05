#!/usr/bin/env bash
# Append-only bounded PP8 Gate C pack. This workflow never initializes TPU.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly PARENT_LAYOUT=/home/gianl/glm-run/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/layout_manifest.json
readonly PARENT_LAYOUT_HASH=aca0eb6d4c498271baf3260dacbe55c9bf935539f42a4b7eaa889947627506a0
readonly ORACLE_DIR=/home/gianl/glm-run/greenfield_gate_c_oracle_20260805T212801776974822Z/oracle
readonly ORACLE_HASH=54262529bd561c57f0833a993e0ac6cdbec20b9e270a0f6f1726d0049d9c4a9f
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_GATE_C_PACK_TAG:-greenfield_gate_c_pack_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
PACK_DIR=$RUN_DIR/packed
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/gate_c/PP8_LP4/$TAG
readonly PIN TAG RUN_DIR PACK_DIR REMOTE_PREFIX

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing Gate C pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing Gate C pack from a dirty greenfield worktree" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/model-00020-of-00141.safetensors ]] || {
  echo "Gate C source shard 20 is unavailable" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/model-00038-of-00141.safetensors ]] || {
  echo "Gate C source shard 38 is unavailable" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/model-00040-of-00141.safetensors ]] || {
  echo "Gate C source shard 40 is unavailable" >&2
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
  echo "[gate-c-pack $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN"
say "REMOTE_PREFIX=$REMOTE_PREFIX"
say "validating protected parent layout and independent Gate C oracle"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PARENT_LAYOUT" "$PARENT_LAYOUT_HASH" "$ORACLE_DIR/manifest.json" \
  "$ORACLE_HASH" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from glm_tpu.greenfield.partitioning import inspect_layout_manifest

parent_path, parent_hash, oracle_path, oracle_hash = sys.argv[1:]
parent = inspect_layout_manifest(Path(parent_path))
oracle = json.loads(Path(oracle_path).read_text())
if parent["manifest_sha256"] != parent_hash:
    raise SystemExit("protected parent layout hash drifted")
if oracle["manifest_sha256"] != oracle_hash:
    raise SystemExit("independent Gate C oracle hash drifted")
PY

cp "$ORACLE_DIR/../source_objects.json" "$RUN_DIR/source_objects.json"

say "streaming 31 raw leaves once into four exact stage-0 owners"
started=$(date +%s)
(
  cd "$WORKTREE"
  env JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python scripts/greenfield/pack_gate_c_checkpoint.py \
      --parent-layout "$PARENT_LAYOUT" \
      --oracle-dir "$ORACLE_DIR" \
      --source-root "$SOURCE_ROOT" \
      --output "$PACK_DIR" \
      --expected-code-hash "$PIN"
) >"$RUN_DIR/pack.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "bounded pack completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PACK_DIR" "$PARENT_LAYOUT" "$ORACLE_DIR" <<'PY' \
  >"$RUN_DIR/inspection.json"
from __future__ import annotations

import json
from pathlib import Path
import sys

from glm_tpu.greenfield.checkpoint import inspect_gate_c_checkpoint
from glm_tpu.greenfield.partitioning import inspect_layout_manifest
from glm_tpu.greenfield.validation import inspect_gate_c_oracle

pack_dir, parent_path, oracle_dir = map(Path, sys.argv[1:])
manifest = inspect_gate_c_checkpoint(
    pack_dir,
    parent_layout=inspect_layout_manifest(parent_path),
    oracle_manifest=inspect_gate_c_oracle(oracle_dir),
)
print(json.dumps({
    "file_count": manifest["file_count"],
    "layout_manifest_sha256": manifest["layout_manifest_sha256"],
    "manifest_sha256": manifest["manifest_sha256"],
    "oracle_manifest_sha256": manifest["oracle_manifest_sha256"],
    "packed_file_bytes": manifest["packed_file_bytes"],
    "packed_payload_bytes": manifest["packed_payload_bytes"],
    "parent_layout_manifest_sha256": manifest["parent_layout_manifest_sha256"],
    "source_payload_bytes": manifest["source_payload_bytes"],
    "source_revision": manifest["source_revision"],
    "source_tensor_count": manifest["source_tensor_count"],
}, indent=2, sort_keys=True))
PY

say "uploading append-only checkpoint to the approved bucket"
gcloud storage cp --no-clobber \
  "$PACK_DIR/layout_manifest.json" \
  "$PACK_DIR/manifest.json" \
  "$REMOTE_PREFIX/packed/" >/dev/null
gcloud storage cp --no-clobber \
  "$PACK_DIR"/base_decoder/stage_00/device_slot_*.safetensors \
  "$REMOTE_PREFIX/packed/base_decoder/stage_00/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PACK_DIR/manifest.json" "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" \
  <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

manifest_path, remote_prefix, output = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
objects = []
for record in manifest["files"]:
    uri = f"{remote_prefix}/packed/{record['filename']}"
    completed = subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = json.loads(completed.stdout)
    if int(remote.get("size", -1)) != record["file_bytes"]:
        raise SystemExit(f"remote size mismatch for {uri}")
    if not remote.get("generation") or not remote.get("crc32c_hash"):
        raise SystemExit(f"remote generation/CRC32C missing for {uri}")
    objects.append({
        "crc32c_hash": remote["crc32c_hash"],
        "file_bytes": record["file_bytes"],
        "generation": remote["generation"],
        "sha256": record["sha256"],
        "uri": uri,
    })
Path(output).write_text(json.dumps({"objects": objects}, indent=2, sort_keys=True) + "\n")
PY

manifest_hash=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$PACK_DIR/manifest.json")
say "remote size/generation/CRC32C verified; sealing evidence for manifest=$manifest_hash"

(
  cd "$RUN_DIR"
  sha256sum orchestrator.log source_objects.json pack.log inspection.json \
    remote_objects.json packed/layout_manifest.json packed/manifest.json
  while read -r digest filename; do
    printf '%s  packed/%s\n' "$digest" "$filename"
  done < <(/home/gianl/vllm-env/bin/python - <<'PY'
import json

manifest = json.load(open("packed/manifest.json"))
for record in manifest["files"]:
    print(record["sha256"], record["filename"])
PY
  )
) >"$RUN_DIR/evidence.sha256"

gcloud storage cp --no-clobber \
  "$RUN_DIR/orchestrator.log" \
  "$RUN_DIR/source_objects.json" \
  "$RUN_DIR/pack.log" \
  "$RUN_DIR/inspection.json" \
  "$RUN_DIR/remote_objects.json" \
  "$RUN_DIR/evidence.sha256" \
  "$REMOTE_PREFIX/" >/dev/null
printf '%s  packed/manifest.json\n' "$manifest_hash" >"$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
