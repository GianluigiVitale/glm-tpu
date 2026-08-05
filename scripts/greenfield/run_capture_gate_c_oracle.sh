#!/usr/bin/env bash
# Standalone raw-source Gate C oracle capture. No model runtime or TPU is used.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly FULL_PACK=/home/gianl/glm-run/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly FULL_PACK_MANIFEST=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly VLLM_REPO=/home/gianl/vllm-build
readonly VLLM_PIN=a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c
readonly TRANSFORMERS_ROOT=/home/gianl/vllm-env/lib/python3.12/site-packages/transformers

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_GATE_C_ORACLE_TAG:-greenfield_gate_c_oracle_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/gate_c/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing Gate C oracle capture outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing Gate C oracle capture from a dirty worktree" >&2
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
[[ -f $FULL_PACK/SUCCESS && -r $FULL_PACK/control.json ]] || {
  echo "protected full checkpoint pack is unavailable" >&2
  exit 2
}
observed_pack_manifest=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$FULL_PACK/summary.json")
[[ $observed_pack_manifest == "$FULL_PACK_MANIFEST" ]] || {
  echo "protected full checkpoint manifest changed" >&2
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
  echo "[gate-c-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN LEGACY_PIN=$LEGACY_PIN VLLM_PIN=$VLLM_PIN"
say "REMOTE_PREFIX=$REMOTE_PREFIX"

source_revision=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_revision"])' \
  "$FULL_PACK/control.json")
say "SOURCE_REVISION=$source_revision from protected full pack $FULL_PACK_MANIFEST"

objects=(
  config.json
  model.safetensors.index.json
  model-00020-of-00141.safetensors
  model-00038-of-00141.safetensors
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

legacy_files=(
  tpu_inference/kernels/dsa/sparse_mla_kernel.py
  tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py
  tpu_inference/layers/vllm/custom_ops/mla_attention.py
  tpu_inference/layers/common/quantization/fp8.py
  tpu_inference/models/jax/deepseek_v3.py
)
vllm_files=(
  vllm/model_executor/layers/attention/mla_attention.py
  vllm/models/deepseek_v4/attention.py
)
transformers_files=(
  models/glm_moe_dsa/configuration_glm_moe_dsa.py
  models/glm_moe_dsa/modeling_glm_moe_dsa.py
)
reference_args=()
for relative in "${legacy_files[@]}"; do
  digest=$(sha256sum "$LEGACY_REPO/$relative" | awk '{print $1}')
  reference_args+=(--reference-source-hash "legacy/$relative=$digest")
done
for relative in "${vllm_files[@]}"; do
  digest=$(sha256sum "$VLLM_REPO/$relative" | awk '{print $1}')
  reference_args+=(--reference-source-hash "vllm/$relative=$digest")
done
for relative in "${transformers_files[@]}"; do
  digest=$(sha256sum "$TRANSFORMERS_ROOT/$relative" | awk '{print $1}')
  reference_args+=(--reference-source-hash "transformers/$relative=$digest")
done
config_digest=$(sha256sum "$WORKTREE/configs/glm-5.2-fp8-config.json" | awk '{print $1}')
reference_args+=(--reference-source-hash "config/glm-5.2-fp8-config.json=$config_digest")

git -C "$LEGACY_REPO" status --short --branch >"$RUN_DIR/legacy_status.txt"
git -C "$VLLM_REPO" status --short --branch >"$RUN_DIR/vllm_status.txt"
/home/gianl/vllm-env/bin/python -c \
  'import transformers; print(transformers.__version__)' \
  >"$RUN_DIR/transformers_version.txt"
[[ $(cat "$RUN_DIR/transformers_version.txt") == 5.12.0 ]] || {
  say "ABORT: Transformers reference version changed"
  exit 1
}

say "capturing raw-source CPU oracle; JAX/model construction/TPU are forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  env JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
    scripts/greenfield/capture_gate_c_oracle.py \
    --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_URI" \
    --source-revision "$source_revision" \
    --output "$ORACLE_DIR" \
    --producer-layer 2 \
    --consumer-layer 3 \
    --expected-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --vllm-code-hash "$VLLM_PIN" \
    "${reference_args[@]}" \
    --allowed-source-shard model-00020-of-00141.safetensors \
    --allowed-source-shard model-00038-of-00141.safetensors \
    --allowed-source-shard model-00040-of-00141.safetensors
) >"$RUN_DIR/capture.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "oracle capture completed in ${elapsed}s"

env JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" \
  /home/gianl/vllm-env/bin/python - \
  "$ORACLE_DIR" "$source_revision" "$PIN" <<'PY' \
  >"$RUN_DIR/inspection.json"
from pathlib import Path
import json
import sys

import numpy as np
from safetensors import safe_open

from glm_tpu.greenfield.validation import inspect_gate_c_oracle

oracle_dir = Path(sys.argv[1])
manifest = inspect_gate_c_oracle(oracle_dir)
if manifest["source_revision"] != sys.argv[2]:
    raise SystemExit("Gate C oracle source revision changed")
if manifest["code_hash"] != sys.argv[3]:
    raise SystemExit("Gate C oracle code hash changed")
expected_geometry = {
    "context_length": 2304,
    "hidden_size": 6144,
    "top_k": 2048,
}
for name, expected in expected_geometry.items():
    if manifest["geometry"][name] != expected:
        raise SystemExit(f"Gate C geometry {name} changed")
if len(manifest["source_tensors"]) != 31:
    raise SystemExit("Gate C source tensor census changed")
source_shards = sorted(
    {record["source_shard"] for record in manifest["source_tensors"]}
)
if source_shards != [
    "model-00020-of-00141.safetensors",
    "model-00038-of-00141.safetensors",
    "model-00040-of-00141.safetensors",
]:
    raise SystemExit(f"Gate C source shard set changed: {source_shards}")
full = manifest["cases"]["full_dsa"]
share = manifest["cases"]["index_share"]
if not full["current_selected"] or full["selected_count"] != 2048:
    raise SystemExit("full DSA selected-set contract failed")
if share["payload_byte_count"] != 8192:
    raise SystemExit("IndexShare payload is not exactly 8192 bytes")
path = oracle_dir / manifest["file"]["filename"]
with safe_open(path, framework="pt", device="cpu") as handle:
    producer = handle.get_tensor("producer_selected_positions").numpy()
    consumer = handle.get_tensor("consumer_selected_positions").numpy()
    canonical = handle.get_tensor("consumer_attention_positions").numpy()
    scores = handle.get_tensor("producer_selected_scores").numpy()
if not np.array_equal(producer, consumer):
    raise SystemExit("IndexShare consumer positions differ")
if np.unique(producer).size != 2048 or not np.array_equal(
    canonical, np.sort(producer, axis=-1)
):
    raise SystemExit("selected positions are duplicate or noncanonical")
if np.any(scores[:, 1:] > scores[:, :-1]):
    raise SystemExit("DSA selected scores are not descending")
print(
    json.dumps(
        {
            "current_position": full["current_position"],
            "file_bytes": manifest["file"]["byte_count"],
            "file_sha256": manifest["file"]["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "payload_byte_count": share["payload_byte_count"],
            "selected_count": full["selected_count"],
            "source_shards": source_shards,
            "source_tensors": len(manifest["source_tensors"]),
        },
        indent=2,
        sort_keys=True,
    )
)
PY

(
  cd "$RUN_DIR"
  sha256sum source_objects.json legacy_status.txt vllm_status.txt \
    transformers_version.txt capture.log inspection.json \
    oracle/manifest.json oracle/oracle.safetensors
) >"$RUN_DIR/evidence.sha256"

say "uploading append-only oracle to approved bucket"
gcloud storage cp --no-clobber \
  "$RUN_DIR/source_objects.json" \
  "$RUN_DIR/legacy_status.txt" \
  "$RUN_DIR/vllm_status.txt" \
  "$RUN_DIR/transformers_version.txt" \
  "$RUN_DIR/capture.log" \
  "$RUN_DIR/inspection.json" \
  "$RUN_DIR/evidence.sha256" \
  "$ORACLE_DIR/manifest.json" \
  "$ORACLE_DIR/oracle.safetensors" \
  "$REMOTE_PREFIX/" >/dev/null

remote_oracle=$(gcloud storage ls "$REMOTE_PREFIX/oracle.safetensors" 2>/dev/null || true)
[[ $remote_oracle == "$REMOTE_PREFIX/oracle.safetensors" ]] || {
  say "ABORT: remote Gate C oracle did not verify"
  exit 1
}
manifest_hash=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$ORACLE_DIR/manifest.json")
say "SUCCESS manifest=$manifest_hash"
say "No TPU/model-performance claim: independent correctness artifact only."
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/orchestrator.log" \
  "$REMOTE_PREFIX/orchestrator.log" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
