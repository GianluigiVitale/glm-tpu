#!/usr/bin/env bash
# Standalone raw-source layer-3 oracle capture. No model or TPU is constructed.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly PACK_RUN=/home/gianl/glm-run/greenfield_one_layer_pack_20260805T151828912346032Z
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly VLLM_REPO=/home/gianl/vllm-build
readonly VLLM_PIN=a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ONE_LAYER_ORACLE_TAG:-greenfield_one_layer_oracle_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/layer3/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing oracle capture outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing oracle capture from a dirty greenfield worktree" >&2
  exit 2
}
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "legacy oracle pin changed" >&2
  exit 2
}
[[ $(git -C "$VLLM_REPO" rev-parse HEAD) == "$VLLM_PIN" ]] || {
  echo "vLLM source pin changed" >&2
  exit 2
}
[[ -f $PACK_RUN/SUCCESS && -r $PACK_RUN/packed/manifest.json ]] || {
  echo "successful real layer pack is unavailable" >&2
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
  echo "[one-layer-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN LEGACY_PIN=$LEGACY_PIN VLLM_PIN=$VLLM_PIN"
say "REMOTE_PREFIX=$REMOTE_PREFIX"

objects=(
  config.json
  model.safetensors.index.json
  model-00038-of-00141.safetensors
  model-00039-of-00141.safetensors
  model-00040-of-00141.safetensors
)
/home/gianl/vllm-env/bin/python - \
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
    json.dumps(
        {"objects": sorted(records, key=lambda item: item["name"])},
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
cmp "$RUN_DIR/source_objects.json" "$PACK_RUN/source_objects.json"
source_revision=gcs-object-set-$(sha256sum "$RUN_DIR/source_objects.json" | awk '{print $1}')
pack_revision=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_revision"])' \
  "$PACK_RUN/packed/manifest.json")
[[ $source_revision == "$pack_revision" ]] || {
  say "ABORT: live source revision differs from the real pack"
  exit 1
}
say "SOURCE_REVISION=$source_revision matches real pack"

legacy_files=(
  tpu_inference/layers/vllm/interface/moe.py
  tpu_inference/layers/common/fused_moe_gmm.py
  tpu_inference/layers/common/moe_psum_fusion.py
  tpu_inference/layers/jax/quantization/fp8.py
  tpu_inference/models/jax/deepseek_v3.py
)
vllm_files=(
  vllm/models/deepseek_v4/amd/model.py
  vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py
  vllm/model_executor/layers/activation.py
)
legacy_args=()
vllm_args=()
for relative in "${legacy_files[@]}"; do
  digest=$(sha256sum "$LEGACY_REPO/$relative" | awk '{print $1}')
  legacy_args+=(--legacy-source-hash "$relative=$digest")
done
for relative in "${vllm_files[@]}"; do
  digest=$(sha256sum "$VLLM_REPO/$relative" | awk '{print $1}')
  vllm_args+=(--vllm-source-hash "$relative=$digest")
done
git -C "$LEGACY_REPO" status --short --branch >"$RUN_DIR/legacy_status.txt"
git -C "$VLLM_REPO" status --short --branch >"$RUN_DIR/vllm_status.txt"

say "capturing raw-source CPU oracle; model construction and packed inputs are forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    scripts/greenfield/capture_one_layer_oracle.py \
    --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_URI" \
    --source-revision "$source_revision" \
    --output "$ORACLE_DIR" \
    --layer 3 \
    --expected-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    "${legacy_args[@]}" \
    --vllm-code-hash "$VLLM_PIN" \
    "${vllm_args[@]}" \
    --allowed-source-shard model-00038-of-00141.safetensors \
    --allowed-source-shard model-00039-of-00141.safetensors \
    --allowed-source-shard model-00040-of-00141.safetensors
) >"$RUN_DIR/capture.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "oracle capture completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$ORACLE_DIR" "$source_revision" <<'PY' >"$RUN_DIR/inspection.json"
from pathlib import Path
import json
import sys

from glm_tpu.greenfield.validation.one_layer_oracle import inspect_one_layer_oracle

manifest = inspect_one_layer_oracle(Path(sys.argv[1]))
if manifest["source_revision"] != sys.argv[2]:
    raise SystemExit("oracle source revision changed")
normal_slots = manifest["cases"]["normal"]["stage_slots"]
if len(normal_slots) < 2:
    raise SystemExit(f"normal routes are insufficiently distributed: {normal_slots}")
case = manifest["cases"]["concentrated"]
if case["stage_slot"] != 2 or not all(128 <= value < 136 for value in case["route_indices"]):
    raise SystemExit("concentrated routes escaped experts 128..135")
allowed = {
    "model-00038-of-00141.safetensors",
    "model-00039-of-00141.safetensors",
    "model-00040-of-00141.safetensors",
}
if not {item["source_shard"] for item in manifest["source_tensors"]} <= allowed:
    raise SystemExit("oracle source tensor escaped bounded shard set")
print(json.dumps({
    "concentrated_routes": case["route_indices"],
    "file_sha256": manifest["file"]["sha256"],
    "manifest_sha256": manifest["manifest_sha256"],
    "normal_routes": manifest["cases"]["normal"]["route_indices"],
    "normal_stage_slots": normal_slots,
    "source_tensors": len(manifest["source_tensors"]),
}, indent=2, sort_keys=True))
PY

(
  cd "$RUN_DIR"
  sha256sum source_objects.json legacy_status.txt vllm_status.txt capture.log \
    inspection.json oracle/manifest.json oracle/oracle.safetensors
) >"$RUN_DIR/evidence.sha256"

say "uploading append-only oracle to approved bucket"
gcloud storage cp --no-clobber \
  "$RUN_DIR/source_objects.json" \
  "$RUN_DIR/legacy_status.txt" \
  "$RUN_DIR/vllm_status.txt" \
  "$RUN_DIR/capture.log" \
  "$RUN_DIR/inspection.json" \
  "$RUN_DIR/evidence.sha256" \
  "$ORACLE_DIR/manifest.json" \
  "$ORACLE_DIR/oracle.safetensors" \
  "$REMOTE_PREFIX/" >/dev/null

remote_oracle=$(gcloud storage ls "$REMOTE_PREFIX/oracle.safetensors" 2>/dev/null || true)
[[ $remote_oracle == "$REMOTE_PREFIX/oracle.safetensors" ]] || {
  say "ABORT: remote oracle did not verify"
  exit 1
}
manifest_hash=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$ORACLE_DIR/manifest.json")
say "SUCCESS manifest=$manifest_hash"
say "No TPU/model-performance claim: this is an independent correctness artifact only."
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/orchestrator.log" \
  "$REMOTE_PREFIX/orchestrator.log" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
