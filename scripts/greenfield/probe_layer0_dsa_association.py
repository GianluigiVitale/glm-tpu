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


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        layer0_dsa_state,
        legacy_pagewise_dcp_scores,
        one_row_pagewise_scores,
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
    key_norm_weight = put_host(
        bf16_host("self_attn__indexer__k_norm__weight")
    )
    key_norm_bias = put_host(bf16_host("self_attn__indexer__k_norm__bias"))
    head_weight = put_host(
        bf16_host("self_attn__indexer__weights_proj__weight")
    )
    q_a_bits = put_host(arrays["self_attn__q_a_proj__weight"])
    q_a_scale = put_host(arrays["self_attn__q_a_proj__weight_scale_inv"])
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
    wq_b_fp32 = dequantize("wq_b_fp32", wq_b_bits, wq_b_scale, jnp.float32)
    wk_fp32 = dequantize("wk_fp32", wk_bits, wk_scale, jnp.float32)
    wq_b_bf16 = dequantize("wq_b_bf16", wq_b_bits, wq_b_scale, jnp.bfloat16)
    wk_bf16 = dequantize("wk_bf16", wk_bits, wk_scale, jnp.bfloat16)

    state_arguments = (
        unique_embeddings,
        prompt_rows,
        current_row,
        input_norm_weight,
        q_a_bf16,
        q_a_norm_weight,
    )
    state_tail = (key_norm_weight, key_norm_bias, head_weight)
    geometry = Layer0DsaProbeGeometry()
    state_definitions = {
        "legacy_fp32_divsqrt": (wq_b_fp32, wk_fp32, "divide_sqrt"),
        "legacy_fp32_rsqrt": (wq_b_fp32, wk_fp32, "multiply_rsqrt"),
        "legacy_bf16_divsqrt": (wq_b_bf16, wk_bf16, "divide_sqrt"),
        "greenfield_bf16_rsqrt_legacy_geometry": (
            wq_b_bf16,
            wk_bf16,
            "multiply_rsqrt",
        ),
    }
    states: dict[str, Any] = {}
    state_hlo_records: dict[str, Any] = {}
    for name, (wq_weight, wk_weight, norm_mode) in state_definitions.items():
        function = partial(
            layer0_dsa_state,
            geometry=geometry,
            key_norm_mode=norm_mode,
        )
        function_arguments = (
            *state_arguments,
            wq_weight,
            wk_weight,
            *state_tail,
        )
        started = time.monotonic()
        compiled = jax.jit(function).lower(*function_arguments).compile()
        compile_seconds[f"state_{name}"] = time.monotonic() - started
        state = compiled(*function_arguments)
        jax.block_until_ready(state)
        if state.query.shape != (32, 32, 128) or (
            state.index_keys.shape != (8156, 128)
        ) or state.head_weights.shape != (32, 32):
            raise RuntimeError(f"layer-0 DSA state geometry drifted: {name}")
        if state.query.dtype != jnp.float32 or (
            state.index_keys.dtype != jnp.bfloat16
        ) or state.head_weights.dtype != jnp.float32:
            raise RuntimeError(f"layer-0 DSA state dtype drifted: {name}")
        states[name] = state
        if name in (
            "legacy_fp32_divsqrt",
            "greenfield_bf16_rsqrt_legacy_geometry",
        ):
            hlo = compiled.as_text()
            hlo_path = args.hlo_dir / f"state_{name}.optimized_hlo.txt"
            hlo_path.write_text(hlo)
            contract = validate_dsa_association_hlo(
                hlo,
                phase="legacy_state",
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
    }
    record = {
        "status": "SUCCESS",
        "claim_scope": (
            "bounded diagnostic layer-0 DSA association; no decoder, Gate-D, "
            "latency, or token-rate claim"
        ),
        "code_hash": code_hash,
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_builder_code_hash": manifest["code_hash"],
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "compile_seconds": compile_seconds,
        "comparisons": comparisons,
        "state_deltas": state_deltas,
        "score_deltas": score_deltas,
        "hlo": {
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
        "legacy_association_restored": comparisons[
            "legacy_fp32_divsqrt"
        ]["passed"],
        "one_row_pagewise_restored": comparisons[
            "one_row_pagewise_on_legacy_state"
        ]["passed"],
        "one_row_pallas_restored": comparisons[
            "one_row_pallas_on_legacy_state"
        ]["passed"],
        "profiler_free_timing": False,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
