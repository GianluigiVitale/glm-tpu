#!/usr/bin/env python3
"""Bounded real-input discriminator for the WS32 layer-0 DSA path.

This program intentionally loads only the sealed layer-0 DSA leaves and the
8,155-row index cache.  It never loads or executes the 753B decoder.  The two
query-owner arms are compiled separately so TPU-v4 cannot fuse across the
hypotheses being adjudicated.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.ascontiguousarray(expected)
    actual = np.ascontiguousarray(actual)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(
            "WS32 DSA comparison geometry drifted: "
            f"expected={expected.shape}/{expected.dtype} "
            f"actual={actual.shape}/{actual.dtype}"
        )
    exact = bool(np.array_equal(expected, actual))
    if np.issubdtype(expected.dtype, np.inexact):
        difference = np.abs(
            actual.astype(np.float64) - expected.astype(np.float64)
        )
        mismatch_count = int(np.count_nonzero(actual != expected))
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean())
    else:
        mismatch_count = int(np.count_nonzero(actual != expected))
        max_abs = float(mismatch_count != 0)
        mean_abs = float(mismatch_count / max(1, expected.size))
    return {
        "actual_sha256": _sha256_array(actual),
        "elementwise_exact": exact,
        "expected_sha256": _sha256_array(expected),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mismatch_count": mismatch_count,
        "shape": list(expected.shape),
    }


def _record_hlo(
    hlo_dir: Path,
    name: str,
    lowered: Any,
    compiled: Any,
) -> tuple[str, str, dict[str, Any]]:
    stablehlo = lowered.as_text()
    optimized_hlo = compiled.as_text()
    stable_path = hlo_dir / f"{name}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{name}.optimized_hlo.txt"
    _atomic_text(stable_path, stablehlo)
    _atomic_text(optimized_path, optimized_hlo)
    return stablehlo, optimized_hlo, {
        "optimized_hlo": {
            "filename": optimized_path.name,
            "sha256": sha256(optimized_hlo.encode()).hexdigest(),
        },
        "stablehlo": {
            "filename": stable_path.name,
            "sha256": sha256(stablehlo.encode()).hexdigest(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--association-input-dir", type=Path, required=True)
    parser.add_argument("--association-input-manifest-sha256", required=True)
    parser.add_argument("--prompt-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache-manifest-sha256", required=True)
    parser.add_argument("--internal-observer-dir", type=Path, required=True)
    parser.add_argument("--internal-contract-sha256", required=True)
    parser.add_argument("--internal-tensor-sha256", required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--source-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.dsa_association import (
        validate_dsa_association_hlo,
    )
    from glm_tpu.greenfield.benchmarking.ws32_dsa_association import (
        classify_ws32_dsa_query_head_contract,
        validate_ws32_dsa_component_hlo,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import (
        DsaNumericalContract,
        dsa_scores,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )
    from glm_tpu.greenfield.kernels.ws32 import (
        ws32_fp8_feature_linear_pallas_mapped,
    )
    from glm_tpu.greenfield.kernels.ws32_layer import (
        ws32_exact_dsa_current_key,
        ws32_grouped_dsa_query_and_head,
    )
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        compare_dsa_association_scores,
        inspect_greenfield_layer0_dsa_internal_observation,
        inspect_layer0_dsa_association_input,
        pack_stage_local_index_keys,
        stitch_stage_local_scores,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        inspect_legacy_prompt_index_cache,
    )
    from glm_tpu.greenfield.validation.ws32_dsa_association import (
        SOURCE_CODE_HASH,
        SOURCE_DB_RUN_ID,
        inspect_ws32_dsa_source_summary,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "WS32 DSA discriminator requires one four-chip TPU host, got "
            f"backend={jax.default_backend()} local={jax.local_device_count()}"
        )
    devices = np.asarray(jax.local_devices(), dtype=object)
    mesh = Mesh(devices, ("feature",))
    device = jax.local_devices()[0]
    args.hlo_dir.mkdir(parents=True, exist_ok=True)

    association_manifest, association = inspect_layer0_dsa_association_input(
        args.association_input_dir,
        expected_manifest_sha256=args.association_input_manifest_sha256,
    )
    cache_manifest, prompt_bits = inspect_legacy_prompt_index_cache(
        args.prompt_cache_dir,
        expected_manifest_sha256=args.prompt_cache_manifest_sha256,
    )
    internal_contract, internals = inspect_greenfield_layer0_dsa_internal_observation(
        args.internal_observer_dir,
        expected_contract_sha256=args.internal_contract_sha256,
        expected_tensor_sha256=args.internal_tensor_sha256,
    )
    inspect_ws32_dsa_source_summary(
        args.source_summary,
        expected_sha256=args.source_summary_sha256,
        association_manifest_sha256=args.association_input_manifest_sha256,
        prompt_cache_manifest_sha256=args.prompt_cache_manifest_sha256,
        internal_tensor_sha256=args.internal_tensor_sha256,
    )
    contract = DsaNumericalContract()
    position_host = np.asarray([8155], dtype=np.int32)
    normalized_bits = np.ascontiguousarray(
        internals["normalized_hidden_bfloat16_bits"][0:1]
    )
    q_expected_bits = np.ascontiguousarray(
        internals["q_a_state_bfloat16_bits"][0:1]
    )
    normalized_host = normalized_bits.view(ml_dtypes.bfloat16)
    q_expected_host = q_expected_bits.view(ml_dtypes.bfloat16)
    query_expected = np.ascontiguousarray(internals["query"][0:1])
    head_expected = np.ascontiguousarray(internals["head_weights"][0:1])
    key_expected = np.ascontiguousarray(internals["current_key"][0:1])

    normalized_sharded = jax.device_put(
        normalized_host, NamedSharding(mesh, P(None, "feature"))
    )
    q_bits_sharded = jax.device_put(
        association["self_attn__q_a_proj__weight"],
        NamedSharding(mesh, P(None, "feature")),
    )
    q_scale_sharded = jax.device_put(
        association["self_attn__q_a_proj__weight_scale_inv"],
        NamedSharding(mesh, P(None, "feature")),
    )
    q_norm_weight = jax.device_put(
        association["self_attn__q_a_layernorm__weight"].view(
            ml_dtypes.bfloat16
        ),
        NamedSharding(mesh, P()),
    )

    def q_a_local(
        normalized_local: Any,
        bits_local: Any,
        scale_local: Any,
        norm_weight: Any,
    ) -> Any:
        projected = ws32_fp8_feature_linear_pallas_mapped(
            normalized_local,
            bits_local,
            scale_local,
            feature_axis="feature",
            block_shape=(128, 128),
            interpret=False,
        )
        return rms_norm(projected, norm_weight, epsilon=1e-5)

    q_a_mapped = jax.shard_map(
        q_a_local,
        mesh=mesh,
        in_specs=(P(None, "feature"), P(None, "feature"), P(None, "feature"), P()),
        out_specs=P(),
        check_vma=False,
    )
    q_started = time.monotonic()
    q_lowered = jax.jit(q_a_mapped).lower(
        normalized_sharded, q_bits_sharded, q_scale_sharded, q_norm_weight
    )
    q_compiled = q_lowered.compile()
    q_compile_seconds = time.monotonic() - q_started
    q_stablehlo, q_hlo, q_hlo_record = _record_hlo(
        args.hlo_dir, "ws32_current_q_a", q_lowered, q_compiled
    )
    q_hlo_contract = validate_ws32_dsa_component_hlo(
        q_hlo,
        stablehlo=q_stablehlo,
        expected_entry_parameters={
            ("bf16", (1, 1536)): 1,
            ("u8", (2048, 1536)): 1,
            ("f32", (16, 12)): 1,
            ("bf16", (2048,)): 1,
        },
        expected_collective_groups=(4,),
        required_live_markers=(
            "greenfield_fp8_block_matmul_f32_m8_k1536_n2048",
            "greenfield_ws32_linear/feature_reduce",
        ),
    )
    if not q_hlo_contract["passed"]:
        raise RuntimeError(f"WS32 q-a HLO contract failed: {q_hlo_contract}")
    q_output = q_compiled(
        normalized_sharded, q_bits_sharded, q_scale_sharded, q_norm_weight
    )
    jax.block_until_ready(q_output)
    q_actual_host = np.ascontiguousarray(np.asarray(q_output))
    q_actual_bits = q_actual_host.view(np.uint16)

    normalized_device = jax.device_put(normalized_host, device)
    q_current_device = jax.device_put(q_actual_host, device)
    q_accepted_device = jax.device_put(q_expected_host, device)
    position_device = jax.device_put(position_host, device)
    full_wq_bits = association["self_attn__indexer__wq_b__weight"]
    full_wq_scale = association[
        "self_attn__indexer__wq_b__weight_scale_inv"
    ]
    full_head_weight = association[
        "self_attn__indexer__weights_proj__weight"
    ].view(ml_dtypes.bfloat16)

    tensor_values: dict[str, np.ndarray] = {
        "accepted_normalized_bfloat16_bits": normalized_bits,
        "accepted_q_a_bfloat16_bits": q_expected_bits,
        "accepted_query": query_expected,
        "accepted_head_weights": head_expected,
        "accepted_current_key": key_expected,
        "ws32_current_q_a_bfloat16_bits": q_actual_bits,
    }
    candidate_records: dict[str, Any] = {}

    def compile_materializer(
        *, alias_count: int
    ) -> tuple[Any, list[Any], dict[str, Any], float]:
        local_width = contract.num_heads * contract.head_dim // alias_count
        scale_rows = local_width // 128
        materializer = jax.jit(
            lambda bits, scale: dequantize_fp8_bits_block_weight(
                bits, scale, output_dtype=jnp.float32
            )
        )
        sample_bits = jax.device_put(full_wq_bits[:local_width], device)
        sample_scale = jax.device_put(full_wq_scale[:scale_rows], device)
        started = time.monotonic()
        lowered = materializer.lower(sample_bits, sample_scale)
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - started
        stablehlo, hlo, hlo_record = _record_hlo(
            args.hlo_dir,
            f"tuple{alias_count}_query_weight_materializer",
            lowered,
            compiled,
        )
        hlo_contract = validate_ws32_dsa_component_hlo(
            hlo,
            stablehlo=stablehlo,
            expected_stablehlo_component=(
                f"tuple{alias_count}_materializer"
            ),
            expected_entry_parameters={
                ("u8", (local_width, contract.q_lora_rank)): 1,
                ("f32", (scale_rows, contract.q_lora_rank // 128)): 1,
            },
        )
        if not hlo_contract["passed"]:
            raise RuntimeError(
                f"tuple{alias_count} materializer HLO failed: {hlo_contract}"
            )
        materialized: list[Any] = []
        for owner in range(alias_count):
            row_start = owner * local_width
            scale_start = owner * scale_rows
            bits = jax.device_put(
                full_wq_bits[row_start : row_start + local_width], device
            )
            scale = jax.device_put(
                full_wq_scale[scale_start : scale_start + scale_rows], device
            )
            value = compiled(bits, scale)
            jax.block_until_ready(value)
            materialized.append(value)
        hlo_record["contract"] = hlo_contract
        return compiled, materialized, hlo_record, compile_seconds

    for alias_count in (8, 4):
        arm = f"tuple{alias_count}"
        local_heads = contract.num_heads // alias_count
        local_width = local_heads * contract.head_dim
        _, owners, materializer_record, materializer_seconds = (
            compile_materializer(alias_count=alias_count)
        )
        sample_head = jax.device_put(full_head_weight[:local_heads], device)

        def query_candidate(
            q_state: Any,
            normalized: Any,
            aliases: tuple[Any, ...],
            head_weight: Any,
            decode_position: Any,
        ) -> tuple[Any, Any]:
            return ws32_grouped_dsa_query_and_head(
                q_state,
                normalized,
                aliases,
                head_weight,
                decode_position,
                contract=contract,
            )

        sample_aliases = tuple(owners[0] for _ in range(alias_count))
        query_jit = jax.jit(query_candidate)
        started = time.monotonic()
        query_lowered = query_jit.lower(
            q_accepted_device,
            normalized_device,
            sample_aliases,
            sample_head,
            position_device,
        )
        query_compiled = query_lowered.compile()
        query_compile_seconds = time.monotonic() - started
        query_stablehlo, query_hlo, query_hlo_record = _record_hlo(
            args.hlo_dir,
            f"{arm}_query_head",
            query_lowered,
            query_compiled,
        )
        query_hlo_contract = validate_ws32_dsa_component_hlo(
            query_hlo,
            stablehlo=query_stablehlo,
            expected_stablehlo_component=f"{arm}_query_head",
            expected_entry_parameters={
                ("bf16", (1, contract.q_lora_rank)): 1,
                ("bf16", (1, contract.hidden_size)): 1,
                ("f32", (local_width, contract.q_lora_rank)): alias_count,
                ("bf16", (local_heads, contract.hidden_size)): 1,
                ("s32", (1,)): 1,
            },
            required_16k_fusion_shape=(alias_count, local_width),
        )
        query_hlo_record["contract"] = query_hlo_contract
        query_contract_classification = classify_ws32_dsa_query_head_contract(
            query_hlo_contract
        )
        if query_contract_classification == "INVALID":
            raise RuntimeError(
                f"{arm} query HLO contract invalid: {query_hlo_contract}"
            )
        if query_contract_classification == "HYPOTHESIS_REJECTED":
            candidate_records[arm] = {
                "alias_count": alias_count,
                "compile_seconds": {
                    "materializer": materializer_seconds,
                    "query_head": query_compile_seconds,
                },
                "hlo": {
                    "materializer": materializer_record,
                    "query_head": query_hlo_record,
                },
                "local_heads": local_heads,
                "rejection": {
                    "component": "query_head",
                    "reason": "16k_fusion_hypothesis_rejected",
                },
                "status": "HLO_REJECTED",
            }
            continue
        accepted_q_query_parts = []
        current_q_query_parts = []
        head_parts = []
        for owner, query_weight in enumerate(owners):
            head_start = owner * local_heads
            head = jax.device_put(
                full_head_weight[head_start : head_start + local_heads], device
            )
            aliases = tuple(query_weight for _ in range(alias_count))
            accepted_q_query_part, head_part = query_compiled(
                q_accepted_device,
                normalized_device,
                aliases,
                head,
                position_device,
            )
            current_q_query_part, current_head_part = query_compiled(
                q_current_device,
                normalized_device,
                aliases,
                head,
                position_device,
            )
            jax.block_until_ready(
                (accepted_q_query_part, current_q_query_part, head_part)
            )
            if not np.array_equal(
                np.asarray(head_part), np.asarray(current_head_part)
            ):
                raise RuntimeError(f"{arm} head result depends on q-a input")
            accepted_q_query_parts.append(
                np.asarray(accepted_q_query_part, dtype=np.float32)
            )
            current_q_query_parts.append(
                np.asarray(current_q_query_part, dtype=np.float32)
            )
            head_parts.append(np.asarray(head_part, dtype=np.float32))
        accepted_q_query_host = np.ascontiguousarray(
            np.concatenate(accepted_q_query_parts, axis=1)
        )
        current_q_query_host = np.ascontiguousarray(
            np.concatenate(current_q_query_parts, axis=1)
        )
        head_host = np.ascontiguousarray(np.concatenate(head_parts, axis=1))
        tensor_values[f"{arm}_accepted_q_query"] = accepted_q_query_host
        tensor_values[f"{arm}_current_q_query"] = current_q_query_host
        tensor_values[f"{arm}_head_weights"] = head_host
        candidate_records[arm] = {
            "alias_count": alias_count,
            "compile_seconds": {
                "materializer": materializer_seconds,
                "query_head": query_compile_seconds,
            },
            "head_comparison": _comparison(head_expected, head_host),
            "hlo": {
                "materializer": materializer_record,
                "query_head": query_hlo_record,
            },
            "local_heads": local_heads,
            "status": "SUCCESS",
            "accepted_q_query_comparison": _comparison(
                query_expected, accepted_q_query_host
            ),
            "current_q_query_comparison": _comparison(
                query_expected, current_q_query_host
            ),
        }

    wk_bits = jax.device_put(
        association["self_attn__indexer__wk__weight"], device
    )
    wk_scale = jax.device_put(
        association["self_attn__indexer__wk__weight_scale_inv"], device
    )
    wk_decode = jax.jit(
        lambda bits, scale: decode_stage_local_prefill_index_wk_bf16(
            bits, scale, contract=contract
        )
    )
    wk_decode_lowered = wk_decode.lower(wk_bits, wk_scale)
    wk_decode_compiled = wk_decode_lowered.compile()
    wk_decode_stablehlo, wk_decode_hlo, wk_decode_record = _record_hlo(
        args.hlo_dir, "wk_decode_bfloat16", wk_decode_lowered, wk_decode_compiled
    )
    wk_decode_contract = validate_ws32_dsa_component_hlo(
        wk_decode_hlo,
        stablehlo=wk_decode_stablehlo,
        expected_stablehlo_component="wk_decode",
        expected_entry_parameters={
            ("u8", (contract.head_dim, contract.hidden_size)): 1,
            ("f32", (1, contract.hidden_size // 128)): 1,
        },
    )
    wk_bf16 = wk_decode_compiled(wk_bits, wk_scale)
    jax.block_until_ready(wk_bf16)
    wk_promote = jax.jit(
        lambda value: promote_stage_local_prefill_index_wk(
            value, contract=contract
        )
    )
    wk_promote_lowered = wk_promote.lower(wk_bf16)
    wk_promote_compiled = wk_promote_lowered.compile()
    wk_promote_stablehlo, wk_promote_hlo, wk_promote_record = _record_hlo(
        args.hlo_dir, "wk_promote_float32", wk_promote_lowered, wk_promote_compiled
    )
    wk_promote_contract = validate_ws32_dsa_component_hlo(
        wk_promote_hlo,
        stablehlo=wk_promote_stablehlo,
        expected_stablehlo_component="wk_promote",
        expected_entry_parameters={
            ("bf16", (contract.head_dim, contract.hidden_size)): 1,
        },
    )
    if not wk_decode_contract["passed"] or not wk_promote_contract["passed"]:
        raise RuntimeError("WS32 exact wk materializer HLO contract failed")
    wk_f32 = wk_promote_compiled(wk_bf16)
    jax.block_until_ready(wk_f32)
    norm_weight = jax.device_put(
        association["self_attn__indexer__k_norm__weight"].view(
            ml_dtypes.bfloat16
        ),
        device,
    )
    norm_bias = jax.device_put(
        association["self_attn__indexer__k_norm__bias"].view(
            ml_dtypes.bfloat16
        ),
        device,
    )
    key_jit = jax.jit(
        lambda normalized, wk, weight, bias, position: ws32_exact_dsa_current_key(
            normalized,
            wk,
            weight,
            bias,
            position,
            contract=contract,
        )
    )
    key_lowered = key_jit.lower(
        normalized_device, wk_f32, norm_weight, norm_bias, position_device
    )
    key_compiled = key_lowered.compile()
    key_stablehlo, key_hlo, key_hlo_record = _record_hlo(
        args.hlo_dir, "exact_current_key", key_lowered, key_compiled
    )
    key_hlo_contract = validate_ws32_dsa_component_hlo(
        key_hlo,
        stablehlo=key_stablehlo,
        expected_stablehlo_component="exact_current_key",
        expected_entry_parameters={
            ("bf16", (1, contract.hidden_size)): 1,
            ("f32", (contract.head_dim, contract.hidden_size)): 1,
            ("bf16", (contract.head_dim,)): 2,
            ("s32", (1,)): 1,
        },
    )
    if not key_hlo_contract["passed"]:
        raise RuntimeError(f"WS32 exact key HLO failed: {key_hlo_contract}")
    key_output = key_compiled(
        normalized_device, wk_f32, norm_weight, norm_bias, position_device
    )
    jax.block_until_ready(key_output)
    key_host = np.ascontiguousarray(np.asarray(key_output, dtype=np.float32))
    tensor_values["exact_current_key"] = key_host
    key_record = {
        "comparison": _comparison(key_expected, key_host),
        "hlo": {
            **key_hlo_record,
            "contract": key_hlo_contract,
            "wk_decode": {
                **wk_decode_record,
                "contract": wk_decode_contract,
            },
            "wk_promote": {
                **wk_promote_record,
                "contract": wk_promote_contract,
            },
        },
        "key_norm_mode": "divide_sqrt",
    }

    expected_positions = association["expected_selected_positions"]
    expected_scores = association["expected_selected_scores"]
    global_bits, lane_bits, lane_positions = pack_stage_local_index_keys(
        prompt_bits,
        key_host[0],
        logical_page_size=512,
        local_parallel_size=8,
    )
    if lane_bits.shape != (8, 1024, 128):
        raise RuntimeError("WS32 exact scorer lane geometry drifted")
    first_lane = jax.device_put(lane_bits[0].view(ml_dtypes.bfloat16), device)
    score_records: dict[str, Any] = {}
    for arm in ("tuple8", "tuple4"):
        if candidate_records[arm]["status"] != "SUCCESS":
            continue
        query_device = jax.device_put(
            tensor_values[f"{arm}_accepted_q_query"], device
        )
        head_device = jax.device_put(
            tensor_values[f"{arm}_head_weights"], device
        )
        score_jit: Callable[[Any, Any, Any], Any] = jax.jit(
            lambda query, keys, head: dsa_scores(
                query, keys, head, precision="default"
            )
        )
        started = time.monotonic()
        score_lowered = score_jit.lower(query_device, first_lane, head_device)
        score_compiled = score_lowered.compile()
        score_compile_seconds = time.monotonic() - started
        score_stablehlo, score_hlo, score_hlo_record = _record_hlo(
            args.hlo_dir,
            f"{arm}_default_score",
            score_lowered,
            score_compiled,
        )
        score_contract = validate_dsa_association_hlo(
            score_hlo, phase="local_wide_default_score", context=1024
        )
        score_body_contract = validate_ws32_dsa_component_hlo(
            score_hlo,
            stablehlo=score_stablehlo,
            expected_stablehlo_component="default_score",
            expected_entry_parameters={
                ("f32", (1, contract.num_heads, contract.head_dim)): 1,
                ("bf16", (1024, contract.head_dim)): 1,
                ("f32", (1, contract.num_heads)): 1,
            },
        )
        if not score_contract["passed"] or not score_body_contract["passed"]:
            raise RuntimeError(f"{arm} scorer HLO failed: {score_contract}")
        lane_scores = []
        for bits in lane_bits:
            keys = jax.device_put(bits.view(ml_dtypes.bfloat16), device)
            score = score_compiled(query_device, keys, head_device)
            jax.block_until_ready(score)
            lane_scores.append(np.asarray(score[0], dtype=np.float32))
        logical_scores = stitch_stage_local_scores(
            np.stack(lane_scores), lane_positions, context=8156
        )
        aligned_scores = np.ascontiguousarray(logical_scores[expected_positions])
        tensor_values[f"{arm}_expected_position_scores"] = aligned_scores
        tensor_values[f"{arm}_logical_scores"] = logical_scores
        score_hlo_record["contract"] = {
            "association": score_contract,
            "component": score_body_contract,
            "passed": True,
        }
        comparison = compare_dsa_association_scores(
            logical_scores, expected_positions, expected_scores
        )
        score_records[arm] = {
            "aligned_score_comparison": _comparison(
                expected_scores, aligned_scores
            ),
            "compile_seconds": score_compile_seconds,
            "comparison": comparison,
            "hlo": score_hlo_record,
            "logical_score_sha256": _sha256_array(logical_scores),
        }
        candidate_records[arm]["score"] = score_records[arm]

    q_comparison = _comparison(q_expected_bits, q_actual_bits)
    exact_arms = sorted(
        arm
        for arm, record in candidate_records.items()
        if record["status"] == "SUCCESS"
        and record["accepted_q_query_comparison"]["elementwise_exact"]
        and record["head_comparison"]["elementwise_exact"]
        and record["score"]["comparison"]["passed"]
        and record["score"]["aligned_score_comparison"]["elementwise_exact"]
    )
    association_restored = bool(
        q_comparison["elementwise_exact"]
        and key_record["comparison"]["elementwise_exact"]
        and exact_arms
    )
    candidate_mechanisms_proven = bool(
        key_record["comparison"]["elementwise_exact"] and exact_arms
    )
    tensor_path = args.output.parent / "ws32_layer0_dsa_association.npz"
    _atomic_npz(tensor_path, tensor_values)
    record = {
        "artifact_kind": "greenfield_ws32_layer0_dsa_association",
        "association_restored": association_restored,
        "backend": jax.default_backend(),
        "candidates": candidate_records,
        "candidate_mechanisms_proven": candidate_mechanisms_proven,
        "claim_scope": (
            "bounded one-layer position-8155 WS32 DSA association only; "
            "no decoder, Gate-D, latency, or token-rate claim"
        ),
        "code_hash": code_hash,
        "device": str(device),
        "device_kind": device.device_kind,
        "diagnostic_only": True,
        "exact_arms": exact_arms,
        "format_version": 2,
        "inputs": {
            "association_manifest_sha256": association_manifest[
                "manifest_sha256"
            ],
            "internal_contract_sha256": args.internal_contract_sha256,
            "internal_tensor_sha256": args.internal_tensor_sha256,
            "layer0_reference": internal_contract["layer0_reference"],
            "prompt_cache_manifest_sha256": cache_manifest[
                "manifest_sha256"
            ],
            "prompt_cache_sha256": cache_manifest[
                "prompt_index_key_bfloat16_sha256"
            ],
            "source_code_hash": SOURCE_CODE_HASH,
            "source_db_run_id": SOURCE_DB_RUN_ID,
            "source_summary_sha256": args.source_summary_sha256,
        },
        "key": key_record,
        "one_live_row": True,
        "performance_claim": False,
        "q_a": {
            "comparison": q_comparison,
            "compile_seconds": q_compile_seconds,
            "hlo": {**q_hlo_record, "contract": q_hlo_contract},
        },
        "score_packing": {
            "global_bfloat16_sha256": _sha256_array(global_bits),
            "lane_bfloat16_sha256": _sha256_array(lane_bits),
            "lane_positions_sha256": _sha256_array(lane_positions),
            "lane_width": 1024,
            "local_parallel_size": 8,
            "logical_context": 8156,
            "logical_page_size": 512,
        },
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
