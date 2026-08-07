#!/usr/bin/env python3
"""Run the bounded real layer-0 8K DSA association matrix on TPU v4."""

from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
# Accepted sealed state-hash log values for layer-0 fused_qkv_a_proj.
SEALED_LEGACY_FUSED_QKV_WEIGHT_BYTE_SUM = 2_448_103_424
SEALED_LEGACY_FUSED_QKV_SCALE_BYTE_SUM = 53_100_864


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _inspect_distributed_q_a_norm_artifact(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_code_hash: str,
    expected_input_manifest_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    from safetensors import safe_open

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != "greenfield_distributed_q_a_norm" or (
        manifest.get("format_version") != 1
        or manifest.get("diagnostic_only") is not True
    ):
        raise ValueError("unsupported distributed q-a norm artifact")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest) or (
        manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("distributed q-a norm manifest checksum mismatch")
    if manifest.get("code_hash") != expected_code_hash or (
        manifest.get("input_manifest_sha256")
        != expected_input_manifest_sha256
    ):
        raise ValueError("distributed q-a norm provenance drifted")
    file_record = manifest.get("file", {})
    if file_record.get("filename") != "distributed_q_a_norm.safetensors":
        raise ValueError("distributed q-a norm filename drifted")
    tensor_path = artifact_dir / file_record["filename"]
    if tensor_path.stat().st_size != file_record.get("byte_count") or (
        _sha256_file(tensor_path) != file_record.get("sha256")
    ):
        raise ValueError("distributed q-a norm tensor integrity failed")
    with safe_open(tensor_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": "greenfield_distributed_q_a_norm",
            "format_version": "1",
        } or list(handle.keys()) != ["q_residual_bfloat16_bits"]:
            raise ValueError("distributed q-a norm tensor metadata drifted")
        q_bits = handle.get_tensor("q_residual_bfloat16_bits").copy()
    q_record = manifest.get("q_residual", {})
    if q_bits.shape != (32, 2048) or q_bits.dtype != np.uint16 or (
        q_record.get("shape") != [32, 2048]
        or q_record.get("dtype") != "bfloat16"
        or q_record.get("byte_count") != q_bits.nbytes
        or q_record.get("sha256")
        != sha256(q_bits.view(np.uint8)).hexdigest()
    ):
        raise ValueError("distributed q-a norm tensor contract drifted")
    return manifest, q_bits


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _tensor_delta(left: Any, right: Any) -> dict[str, Any]:
    left_host = np.asarray(left, dtype=np.float32)
    right_host = np.asarray(right, dtype=np.float32)
    difference = np.abs(left_host - right_host)
    return {
        "elementwise_exact": bool(np.array_equal(left_host, right_host)),
        "mismatch_count": int(np.count_nonzero(left_host != right_host)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.percentile(difference, 99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--distributed-q-a-norm-dir", type=Path, required=True)
    parser.add_argument(
        "--distributed-q-a-norm-manifest-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.dsa import validate_dsa_score_hlo
    from glm_tpu.greenfield.benchmarking.dsa_association import (
        validate_dsa_association_hlo,
    )
    from glm_tpu.greenfield.kernels.pallas.dsa import dsa_scores_pallas
    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        layer0_dsa_state_from_q_residual,
        layer0_dsa_state,
        legacy_pagewise_dcp_scores,
        one_row_pagewise_scores,
        pack_legacy_fused_qkv_runtime_weights,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        compare_dsa_association_scores,
        inspect_layer0_dsa_association_input,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"layer-0 DSA probe requires TPU, got {jax.default_backend()}"
        )
    if jax.local_device_count() != 4:
        raise RuntimeError(
            "layer-0 DSA probe requires one complete four-chip TPU-v4 host"
        )
    device = jax.local_devices()[0]
    manifest, arrays = inspect_layer0_dsa_association_input(
        args.input_dir,
        expected_manifest_sha256=args.input_manifest_sha256,
    )
    distributed_q_a_manifest, distributed_q_a_bits = (
        _inspect_distributed_q_a_norm_artifact(
            args.distributed_q_a_norm_dir,
            expected_manifest_sha256=(
                args.distributed_q_a_norm_manifest_sha256
            ),
            expected_code_hash=code_hash,
            expected_input_manifest_sha256=manifest["manifest_sha256"],
        )
    )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)

    unique_ids = arrays["unique_token_ids"]
    prompt_ids = arrays["prompt_token_ids"]
    current_id = arrays["current_token_id"]
    prompt_rows_host = np.searchsorted(unique_ids, prompt_ids).astype(np.int32)
    current_row_host = np.searchsorted(unique_ids, current_id).astype(np.int32)
    if not np.array_equal(unique_ids[prompt_rows_host], prompt_ids) or (
        not np.array_equal(unique_ids[current_row_host], current_id)
    ):
        raise RuntimeError("layer-0 DSA unique embedding map drifted")

    def bf16_host(name: str) -> np.ndarray:
        return arrays[name].view(ml_dtypes.bfloat16)

    def put_host(value: np.ndarray) -> Any:
        return jax.device_put(value, device)

    unique_embeddings = put_host(bf16_host("unique_embedding_bfloat16_bits"))
    prompt_rows = put_host(prompt_rows_host)
    current_row = put_host(current_row_host)
    input_norm_weight = put_host(bf16_host("input_layernorm__weight"))
    q_a_norm_weight = put_host(bf16_host("self_attn__q_a_layernorm__weight"))
    distributed_q_a_residual = put_host(
        distributed_q_a_bits.view(ml_dtypes.bfloat16)
    )
    key_norm_weight = put_host(
        bf16_host("self_attn__indexer__k_norm__weight")
    )
    key_norm_bias = put_host(bf16_host("self_attn__indexer__k_norm__bias"))
    head_weight = put_host(
        bf16_host("self_attn__indexer__weights_proj__weight")
    )
    q_a_bits = put_host(arrays["self_attn__q_a_proj__weight"])
    q_a_scale = put_host(arrays["self_attn__q_a_proj__weight_scale_inv"])
    kv_a_bits = put_host(arrays["self_attn__kv_a_proj_with_mqa__weight"])
    kv_a_scale = put_host(
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"]
    )
    wq_b_bits = put_host(arrays["self_attn__indexer__wq_b__weight"])
    wq_b_scale = put_host(
        arrays["self_attn__indexer__wq_b__weight_scale_inv"]
    )
    wk_bits = put_host(arrays["self_attn__indexer__wk__weight"])
    wk_scale = put_host(arrays["self_attn__indexer__wk__weight_scale_inv"])

    compile_seconds: dict[str, float] = {}

    def dequantize(name: str, bits: Any, scale: Any, dtype: Any) -> Any:
        function = partial(
            dequantize_fp8_bits_block_weight,
            output_dtype=dtype,
        )
        started = time.monotonic()
        compiled = jax.jit(function).lower(bits, scale).compile()
        compile_seconds[f"dequant_{name}"] = time.monotonic() - started
        value = compiled(bits, scale)
        jax.block_until_ready(value)
        return value

    q_a_bf16 = dequantize("q_a_bf16", q_a_bits, q_a_scale, jnp.bfloat16)
    kv_a_bf16 = dequantize(
        "kv_a_bf16", kv_a_bits, kv_a_scale, jnp.bfloat16
    )
    fused_qkv_a_bf16 = jnp.concatenate((q_a_bf16, kv_a_bf16), axis=0)
    jax.block_until_ready(fused_qkv_a_bf16)
    geometry = Layer0DsaProbeGeometry()
    runtime_pack_function = partial(
        pack_legacy_fused_qkv_runtime_weights,
        geometry=geometry,
    )
    started = time.monotonic()
    runtime_pack_compiled = jax.jit(runtime_pack_function).lower(
        q_a_bits,
        q_a_scale,
        kv_a_bits,
        kv_a_scale,
    ).compile()
    compile_seconds["legacy_fused_qkv_runtime_pack"] = (
        time.monotonic() - started
    )
    runtime_weights = runtime_pack_compiled(
        q_a_bits,
        q_a_scale,
        kv_a_bits,
        kv_a_scale,
    )
    jax.block_until_ready(runtime_weights)
    runtime_weight_host = np.asarray(runtime_weights.global_weight)
    runtime_scale_host = np.ascontiguousarray(
        np.asarray(runtime_weights.global_scale, dtype=np.float32)
    )
    runtime_layout_identity = {
        "global_weight_shape": list(runtime_weight_host.shape),
        "global_weight_dtype": str(runtime_weight_host.dtype),
        "global_weight_byte_sum": int(
            runtime_weight_host.view(np.uint8).sum(dtype=np.uint64)
        ),
        "global_weight_sha256": sha256(
            runtime_weight_host.view(np.uint8).tobytes(order="C")
        ).hexdigest(),
        "global_scale_shape": list(runtime_scale_host.shape),
        "global_scale_dtype": str(runtime_scale_host.dtype),
        "global_scale_byte_sum": int(
            runtime_scale_host.view(np.uint8).sum(dtype=np.uint64)
        ),
        "global_scale_sha256": sha256(
            runtime_scale_host.view(np.uint8).tobytes(order="C")
        ).hexdigest(),
        "sealed_legacy_state_hash_shape_and_byte_sum_match": False,
    }
    if runtime_layout_identity["global_weight_byte_sum"] != (
        SEALED_LEGACY_FUSED_QKV_WEIGHT_BYTE_SUM
    ) or runtime_layout_identity["global_scale_byte_sum"] != (
        SEALED_LEGACY_FUSED_QKV_SCALE_BYTE_SUM
    ):
        raise RuntimeError(
            "legacy fused qkv runtime layout disagrees with the sealed "
            f"state-hash log: {runtime_layout_identity}"
        )
    runtime_layout_identity[
        "sealed_legacy_state_hash_shape_and_byte_sum_match"
    ] = True
    runtime_pack_hlo = runtime_pack_compiled.as_text()
    runtime_pack_path = (
        args.hlo_dir / "legacy_fused_qkv_runtime_pack.optimized_hlo.txt"
    )
    runtime_pack_path.write_text(runtime_pack_hlo)
    runtime_pack_contract = validate_dsa_association_hlo(
        runtime_pack_hlo,
        phase="legacy_fused_qkv_runtime_pack",
    )
    if not runtime_pack_contract["passed"]:
        raise RuntimeError(
            "legacy fused qkv runtime-pack HLO contract failed: "
            f"{runtime_pack_contract}"
        )
    wq_b_fp32 = dequantize("wq_b_fp32", wq_b_bits, wq_b_scale, jnp.float32)
    wk_fp32 = dequantize("wk_fp32", wk_bits, wk_scale, jnp.float32)
    wq_b_bf16 = dequantize("wq_b_bf16", wq_b_bits, wq_b_scale, jnp.bfloat16)
    wk_bf16 = dequantize("wk_bf16", wk_bits, wk_scale, jnp.bfloat16)

    state_prefix = (
        unique_embeddings,
        prompt_rows,
        current_row,
        input_norm_weight,
    )
    state_tail = (key_norm_weight, key_norm_bias, head_weight)
    state_definitions = {
        "legacy_fp32_divsqrt": (
            q_a_bf16,
            None,
            wq_b_fp32,
            wk_fp32,
            "divide_sqrt",
            "separate_q_a",
        ),
        "legacy_fp32_rsqrt": (
            q_a_bf16,
            None,
            wq_b_fp32,
            wk_fp32,
            "multiply_rsqrt",
            "separate_q_a",
        ),
        "legacy_bf16_divsqrt": (
            q_a_bf16,
            None,
            wq_b_bf16,
            wk_bf16,
            "divide_sqrt",
            "separate_q_a",
        ),
        "greenfield_bf16_rsqrt_legacy_geometry": (
            q_a_bf16,
            None,
            wq_b_bf16,
            wk_bf16,
            "multiply_rsqrt",
            "separate_q_a",
        ),
        "legacy_fused_qkv_fp32_divsqrt": (
            fused_qkv_a_bf16,
            None,
            wq_b_fp32,
            wk_fp32,
            "divide_sqrt",
            "legacy_fused_qkv_a",
        ),
        "legacy_runtime_fused_qkv_global_fp32_divsqrt": (
            runtime_weights.global_weight,
            runtime_weights.global_scale,
            wq_b_fp32,
            wk_fp32,
            "divide_sqrt",
            "legacy_runtime_fused_qkv_a_global",
        ),
        "legacy_runtime_fused_qkv_sharded_fp32_divsqrt": (
            runtime_weights.sharded_weight,
            runtime_weights.sharded_scale,
            wq_b_fp32,
            wk_fp32,
            "divide_sqrt",
            "legacy_runtime_fused_qkv_a_sharded",
        ),
    }
    states: dict[str, Any] = {}
    state_hlo_records: dict[str, Any] = {}
    for name, (
        q_a_projection_weight,
        q_a_projection_scale,
        wq_weight,
        wk_weight,
        norm_mode,
        q_a_projection_mode,
    ) in state_definitions.items():
        function = partial(
            layer0_dsa_state,
            geometry=geometry,
            key_norm_mode=norm_mode,
            q_a_projection_mode=q_a_projection_mode,
        )
        function_arguments = (
            *state_prefix,
            q_a_projection_weight,
            q_a_norm_weight,
            wq_weight,
            wk_weight,
            *state_tail,
            q_a_projection_scale,
        )
        started = time.monotonic()
        compiled = jax.jit(function).lower(*function_arguments).compile()
        compile_seconds[f"state_{name}"] = time.monotonic() - started
        state = compiled(*function_arguments)
        jax.block_until_ready(state)
        if state.query.shape != (32, 32, 128) or (
            state.index_keys.shape != (8156, 128)
        ) or state.head_weights.shape != (32, 32) or (
            state.qkv_a_companion.shape != (32, 576)
        ):
            raise RuntimeError(f"layer-0 DSA state geometry drifted: {name}")
        if state.query.dtype != jnp.float32 or (
            state.index_keys.dtype != jnp.bfloat16
        ) or state.head_weights.dtype != jnp.float32 or (
            state.qkv_a_companion.dtype != jnp.bfloat16
        ):
            raise RuntimeError(f"layer-0 DSA state dtype drifted: {name}")
        states[name] = state
        if name in (
            "legacy_fp32_divsqrt",
            "legacy_fused_qkv_fp32_divsqrt",
            "legacy_runtime_fused_qkv_global_fp32_divsqrt",
            "legacy_runtime_fused_qkv_sharded_fp32_divsqrt",
            "greenfield_bf16_rsqrt_legacy_geometry",
        ):
            hlo = compiled.as_text()
            hlo_path = args.hlo_dir / f"state_{name}.optimized_hlo.txt"
            hlo_path.write_text(hlo)
            contract = validate_dsa_association_hlo(
                hlo,
                phase={
                    "legacy_fused_qkv_a": "legacy_fused_qkv_state",
                    "legacy_runtime_fused_qkv_a_global": (
                        "legacy_runtime_fused_qkv_global_state"
                    ),
                    "legacy_runtime_fused_qkv_a_sharded": (
                        "legacy_runtime_fused_qkv_sharded_state"
                    ),
                }.get(q_a_projection_mode, "legacy_state"),
            )
            if not contract["passed"]:
                raise RuntimeError(
                    f"layer-0 DSA state HLO contract failed: {name}: {contract}"
                )
            state_hlo_records[name] = {
                "filename": hlo_path.name,
                "sha256": sha256(hlo.encode()).hexdigest(),
                "contract": contract,
            }

    sharded_state = states[
        "legacy_runtime_fused_qkv_sharded_fp32_divsqrt"
    ]
    replacement_arguments = (
        distributed_q_a_residual,
        sharded_state.index_keys,
        sharded_state.head_weights,
        sharded_state.qkv_a_companion,
        wq_b_fp32,
    )
    replacement_function = partial(
        layer0_dsa_state_from_q_residual,
        geometry=geometry,
    )
    started = time.monotonic()
    replacement_compiled = jax.jit(replacement_function).lower(
        *replacement_arguments
    ).compile()
    compile_seconds["state_legacy_tp32_distributed_q_a_norm"] = (
        time.monotonic() - started
    )
    distributed_state = replacement_compiled(*replacement_arguments)
    jax.block_until_ready(distributed_state)
    distributed_state_name = (
        "legacy_tp32_distributed_q_a_norm_fp32_divsqrt"
    )
    states[distributed_state_name] = distributed_state
    replacement_hlo = replacement_compiled.as_text()
    replacement_hlo_path = (
        args.hlo_dir / "state_legacy_tp32_distributed_q_a_norm.optimized_hlo.txt"
    )
    replacement_hlo_path.write_text(replacement_hlo)
    replacement_contract = validate_dsa_association_hlo(
        replacement_hlo,
        phase="legacy_tp32_distributed_q_a_norm_state",
    )
    if not replacement_contract["passed"]:
        raise RuntimeError(
            "distributed q-a norm replacement-state HLO failed: "
            f"{replacement_contract}"
        )
    state_hlo_records[distributed_state_name] = {
        "filename": replacement_hlo_path.name,
        "sha256": sha256(replacement_hlo.encode()).hexdigest(),
        "contract": replacement_contract,
    }

    legacy_state = states["legacy_fp32_divsqrt"]
    started = time.monotonic()
    legacy_score_compiled = jax.jit(legacy_pagewise_dcp_scores).lower(
        legacy_state.query,
        legacy_state.index_keys,
        legacy_state.head_weights,
    ).compile()
    compile_seconds["legacy_score"] = time.monotonic() - started
    legacy_score_hlo = legacy_score_compiled.as_text()
    legacy_score_path = args.hlo_dir / "legacy_score.optimized_hlo.txt"
    legacy_score_path.write_text(legacy_score_hlo)
    legacy_score_contract = validate_dsa_association_hlo(
        legacy_score_hlo,
        phase="legacy_score",
    )
    if not legacy_score_contract["passed"]:
        raise RuntimeError(
            f"legacy DSA score HLO contract failed: {legacy_score_contract}"
        )

    expected_positions = arrays["expected_selected_positions"]
    expected_scores = arrays["expected_selected_scores"]
    comparisons: dict[str, Any] = {}
    full_scores: dict[str, np.ndarray] = {}
    for name, state in states.items():
        score = legacy_score_compiled(
            state.query,
            state.index_keys,
            state.head_weights,
        )
        jax.block_until_ready(score)
        score_host = np.asarray(score, dtype=np.float32)
        full_scores[name] = score_host
        comparisons[name] = compare_dsa_association_scores(
            score_host,
            expected_positions,
            expected_scores,
        )

    query_one = legacy_state.query[:1]
    head_one = legacy_state.head_weights[:1]
    started = time.monotonic()
    one_row_compiled = jax.jit(one_row_pagewise_scores).lower(
        query_one,
        legacy_state.index_keys,
        head_one,
    ).compile()
    compile_seconds["one_row_pagewise_score"] = time.monotonic() - started
    one_row_hlo = one_row_compiled.as_text()
    one_row_path = args.hlo_dir / "one_row_pagewise_score.optimized_hlo.txt"
    one_row_path.write_text(one_row_hlo)
    one_row_contract = validate_dsa_association_hlo(
        one_row_hlo,
        phase="one_row_score",
    )
    if not one_row_contract["passed"]:
        raise RuntimeError(
            f"one-row pagewise DSA HLO contract failed: {one_row_contract}"
        )
    one_row_score = one_row_compiled(
        query_one,
        legacy_state.index_keys,
        head_one,
    )
    jax.block_until_ready(one_row_score)
    one_row_host = np.asarray(one_row_score, dtype=np.float32)
    comparisons["one_row_pagewise_on_legacy_state"] = (
        compare_dsa_association_scores(
            one_row_host,
            expected_positions,
            expected_scores,
        )
    )

    started = time.monotonic()
    pallas_compiled = jax.jit(dsa_scores_pallas).lower(
        query_one,
        legacy_state.index_keys,
        head_one,
    ).compile()
    compile_seconds["one_row_pallas_score"] = time.monotonic() - started
    pallas_hlo = pallas_compiled.as_text()
    pallas_path = args.hlo_dir / "one_row_pallas_score.optimized_hlo.txt"
    pallas_path.write_text(pallas_hlo)
    pallas_contract = validate_dsa_score_hlo(
        pallas_hlo,
        local_context=8156,
    )
    if not pallas_contract["passed"]:
        raise RuntimeError(
            f"one-row Pallas DSA HLO contract failed: {pallas_contract}"
        )
    pallas_score = pallas_compiled(
        query_one,
        legacy_state.index_keys,
        head_one,
    )
    jax.block_until_ready(pallas_score)
    pallas_host = np.asarray(pallas_score[0], dtype=np.float32)
    comparisons["one_row_pallas_on_legacy_state"] = (
        compare_dsa_association_scores(
            pallas_host,
            expected_positions,
            expected_scores,
        )
    )

    fused_state = states["legacy_fused_qkv_fp32_divsqrt"]
    fused_query_one = fused_state.query[:1]
    fused_head_one = fused_state.head_weights[:1]
    fused_one_row_score = one_row_compiled(
        fused_query_one,
        fused_state.index_keys,
        fused_head_one,
    )
    jax.block_until_ready(fused_one_row_score)
    fused_one_row_host = np.asarray(fused_one_row_score, dtype=np.float32)
    comparisons["one_row_pagewise_on_fused_qkv_legacy_state"] = (
        compare_dsa_association_scores(
            fused_one_row_host,
            expected_positions,
            expected_scores,
        )
    )
    fused_pallas_score = pallas_compiled(
        fused_query_one,
        fused_state.index_keys,
        fused_head_one,
    )
    jax.block_until_ready(fused_pallas_score)
    fused_pallas_host = np.asarray(fused_pallas_score[0], dtype=np.float32)
    comparisons["one_row_pallas_on_fused_qkv_legacy_state"] = (
        compare_dsa_association_scores(
            fused_pallas_host,
            expected_positions,
            expected_scores,
        )
    )

    runtime_one_row_scores: dict[str, np.ndarray] = {}
    runtime_pallas_scores: dict[str, np.ndarray] = {}
    for label, state_name in (
        (
            "runtime_fused_qkv_global",
            "legacy_runtime_fused_qkv_global_fp32_divsqrt",
        ),
        (
            "runtime_fused_qkv_sharded",
            "legacy_runtime_fused_qkv_sharded_fp32_divsqrt",
        ),
        (
            "runtime_fused_qkv_distributed_norm",
            distributed_state_name,
        ),
    ):
        runtime_state = states[state_name]
        runtime_one_row = one_row_compiled(
            runtime_state.query[:1],
            runtime_state.index_keys,
            runtime_state.head_weights[:1],
        )
        jax.block_until_ready(runtime_one_row)
        runtime_one_row_host = np.asarray(runtime_one_row, dtype=np.float32)
        runtime_one_row_scores[label] = runtime_one_row_host
        comparisons[f"one_row_pagewise_on_{label}_legacy_state"] = (
            compare_dsa_association_scores(
                runtime_one_row_host,
                expected_positions,
                expected_scores,
            )
        )
        runtime_pallas = pallas_compiled(
            runtime_state.query[:1],
            runtime_state.index_keys,
            runtime_state.head_weights[:1],
        )
        jax.block_until_ready(runtime_pallas)
        runtime_pallas_host = np.asarray(runtime_pallas[0], dtype=np.float32)
        runtime_pallas_scores[label] = runtime_pallas_host
        comparisons[f"one_row_pallas_on_{label}_legacy_state"] = (
            compare_dsa_association_scores(
                runtime_pallas_host,
                expected_positions,
                expected_scores,
            )
        )

    state_deltas = {
        name: {
            "query_vs_legacy": _tensor_delta(state.query[0], legacy_state.query[0]),
            "keys_vs_legacy": _tensor_delta(
                state.index_keys,
                legacy_state.index_keys,
            ),
            "head_weights_vs_legacy": _tensor_delta(
                state.head_weights[0],
                legacy_state.head_weights[0],
            ),
            "qkv_a_companion_vs_legacy": _tensor_delta(
                state.qkv_a_companion,
                legacy_state.qkv_a_companion,
            ),
        }
        for name, state in states.items()
        if name != "legacy_fp32_divsqrt"
    }
    score_deltas = {
        "one_row_pagewise_vs_legacy_pagewise": _tensor_delta(
            one_row_host,
            full_scores["legacy_fp32_divsqrt"],
        ),
        "one_row_pallas_vs_legacy_pagewise": _tensor_delta(
            pallas_host,
            full_scores["legacy_fp32_divsqrt"],
        ),
        "fused_qkv_one_row_pagewise_vs_legacy_pagewise": _tensor_delta(
            fused_one_row_host,
            full_scores["legacy_fused_qkv_fp32_divsqrt"],
        ),
        "fused_qkv_one_row_pallas_vs_legacy_pagewise": _tensor_delta(
            fused_pallas_host,
            full_scores["legacy_fused_qkv_fp32_divsqrt"],
        ),
    }
    for label, state_name in (
        (
            "runtime_fused_qkv_global",
            "legacy_runtime_fused_qkv_global_fp32_divsqrt",
        ),
        (
            "runtime_fused_qkv_sharded",
            "legacy_runtime_fused_qkv_sharded_fp32_divsqrt",
        ),
        (
            "runtime_fused_qkv_distributed_norm",
            distributed_state_name,
        ),
    ):
        score_deltas[f"{label}_one_row_pagewise_vs_legacy_pagewise"] = (
            _tensor_delta(
                runtime_one_row_scores[label],
                full_scores[state_name],
            )
        )
        score_deltas[f"{label}_one_row_pallas_vs_legacy_pagewise"] = (
            _tensor_delta(
                runtime_pallas_scores[label],
                full_scores[state_name],
            )
        )
    record = {
        "status": "SUCCESS",
        "claim_scope": (
            "bounded diagnostic layer-0 DSA association; no decoder, Gate-D, "
            "latency, or token-rate claim"
        ),
        "code_hash": code_hash,
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_builder_code_hash": manifest["code_hash"],
        "distributed_q_a_norm_artifact": distributed_q_a_manifest,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "compile_seconds": compile_seconds,
        "comparisons": comparisons,
        "state_deltas": state_deltas,
        "score_deltas": score_deltas,
        "runtime_fused_qkv_layout_identity": runtime_layout_identity,
        "hlo": {
            "legacy_fused_qkv_runtime_pack": {
                "filename": runtime_pack_path.name,
                "sha256": sha256(runtime_pack_hlo.encode()).hexdigest(),
                "contract": runtime_pack_contract,
            },
            "state": state_hlo_records,
            "legacy_score": {
                "filename": legacy_score_path.name,
                "sha256": sha256(legacy_score_hlo.encode()).hexdigest(),
                "contract": legacy_score_contract,
            },
            "one_row_pagewise_score": {
                "filename": one_row_path.name,
                "sha256": sha256(one_row_hlo.encode()).hexdigest(),
                "contract": one_row_contract,
            },
            "one_row_pallas_score": {
                "filename": pallas_path.name,
                "sha256": sha256(pallas_hlo.encode()).hexdigest(),
                "contract": pallas_contract,
            },
        },
        "standalone_legacy_association_restored": comparisons[
            "legacy_fp32_divsqrt"
        ]["passed"],
        "legacy_association_restored": comparisons[
            distributed_state_name
        ]["passed"],
        "predecoded_fused_qkv_association_restored": comparisons[
            "legacy_fused_qkv_fp32_divsqrt"
        ]["passed"],
        "runtime_fused_qkv_global_association_restored": comparisons[
            "legacy_runtime_fused_qkv_global_fp32_divsqrt"
        ]["passed"],
        "runtime_fused_qkv_sharded_association_restored": comparisons[
            "legacy_runtime_fused_qkv_sharded_fp32_divsqrt"
        ]["passed"],
        "runtime_fused_qkv_distributed_norm_association_restored": comparisons[
            distributed_state_name
        ]["passed"],
        "one_row_pagewise_restored": comparisons[
            "one_row_pagewise_on_legacy_state"
        ]["passed"],
        "one_row_pallas_restored": comparisons[
            "one_row_pallas_on_legacy_state"
        ]["passed"],
        "fused_qkv_one_row_pagewise_restored": comparisons[
            "one_row_pagewise_on_fused_qkv_legacy_state"
        ]["passed"],
        "fused_qkv_one_row_pallas_restored": comparisons[
            "one_row_pallas_on_fused_qkv_legacy_state"
        ]["passed"],
        "runtime_fused_qkv_global_one_row_pagewise_restored": comparisons[
            "one_row_pagewise_on_runtime_fused_qkv_global_legacy_state"
        ]["passed"],
        "runtime_fused_qkv_global_one_row_pallas_restored": comparisons[
            "one_row_pallas_on_runtime_fused_qkv_global_legacy_state"
        ]["passed"],
        "runtime_fused_qkv_sharded_one_row_pagewise_restored": comparisons[
            "one_row_pagewise_on_runtime_fused_qkv_sharded_legacy_state"
        ]["passed"],
        "runtime_fused_qkv_sharded_one_row_pallas_restored": comparisons[
            "one_row_pallas_on_runtime_fused_qkv_sharded_legacy_state"
        ]["passed"],
        "runtime_fused_qkv_distributed_norm_one_row_pagewise_restored": (
            comparisons[
                "one_row_pagewise_on_runtime_fused_qkv_distributed_norm_legacy_state"
            ]["passed"]
        ),
        "runtime_fused_qkv_distributed_norm_one_row_pallas_restored": (
            comparisons[
                "one_row_pallas_on_runtime_fused_qkv_distributed_norm_legacy_state"
            ]["passed"]
        ),
        "profiler_free_timing": False,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
