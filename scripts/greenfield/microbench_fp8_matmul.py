#!/usr/bin/env python3
"""Compile, verify, and time one production-shaped greenfield FP8 kernel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _is_bounded_route_restore_index_call(
    line: str,
    *,
    route_count: int,
    kernel_name: str = "fp8_selected_up_gate",
) -> bool:
    """Recognize only TPU's compact top-k output-gather index annotation."""

    return (
        f" = s32[{route_count},2]" in line
        and 'custom_call_target="GatherScatterIndicesBitpacked"' in line
        and (
            f'metadata={{op_name="jit({kernel_name})/concatenate"'
            in line
        )
    )


def _custom_call_result_name(line: str) -> str:
    return line.split(" = ", 1)[0].strip()


def _is_bounded_compact_input_scatter(
    line: str,
    *,
    index_result: str,
    route_count: int,
    row_tile: int,
    width: int,
    kernel_name: str,
) -> bool:
    """Recognize only one selected-down BF16 route-compaction scatter."""

    return (
        f" = bf16[{route_count},{row_tile},{width}]" in line
        and " scatter(" in line
        and f", {index_result}," in line
        and "update_window_dims={1}" in line
        and "inserted_window_dims={0,1}" in line
        and "scatter_dims_to_operand_dims={0,1}" in line
        and "index_vector_dim=1" in line
        and f'metadata={{op_name="jit({kernel_name})/scatter"' in line
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--contraction", type=int, default=6144)
    parser.add_argument("--output-width", type=int, default=2048)
    parser.add_argument(
        "--kernel",
        choices=(
            "single_up",
            "attention_output",
            "rmsnorm_linear",
            "up_gate",
            "selected_up_gate",
            "selected_swiglu_down",
            "dsa_wq_b",
            "dsa_wk",
        ),
        default="single_up",
    )
    parser.add_argument(
        "--output-tile",
        type=int,
        choices=(128, 256),
        default=128,
    )
    parser.add_argument(
        "--selected-route-case",
        choices=("normal_two", "concentrated_eight"),
        default="concentrated_eight",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
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
    production_shapes = {
        "attention_output": (1, 4096, 6144),
        "dsa_wq_b": (1, 2048, 1024),
        "dsa_wk": (1, 6144, 128),
    }
    expected_shape = production_shapes.get(
        args.kernel,
        (1 if args.kernel == "rmsnorm_linear" else 8, 6144, 2048),
    )
    if (args.rows, args.contraction, args.output_width) != expected_shape:
        raise ValueError(
            "protected metal proof is fixed to its production decode shape: "
            f"rows={expected_shape[0]}, hidden={expected_shape[1]}, "
            f"output={expected_shape[2]}"
        )
    if args.kernel != "attention_output" and args.output_tile != 128:
        raise ValueError(
            "a non-default output tile requires the attention-output kernel"
        )
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected FP8 microbenchmark requires 200/1000 samples")

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from jax import lax

    from glm_tpu.greenfield.kernels.pallas import (
        Fp8BlockMatmulConfig,
        fp8_block_matmul,
        fp8_block_matmul_f32,
        fp8_block_up_gate,
        fp8_rmsnorm_block_matmul,
        fp8_selected_swiglu_down,
        fp8_selected_up_gate,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"FP8 metal proof requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4:
        raise RuntimeError(
            f"FP8 metal proof requires one four-chip host, got {jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    rows = args.rows
    contraction = args.contraction
    output = args.output_width
    selected_kernel = args.kernel in (
        "selected_up_gate",
        "selected_swiglu_down",
    )

    lhs_rows = (
        1
        if args.kernel
        in ("selected_up_gate", "rmsnorm_linear", "dsa_wq_b", "dsa_wk")
        else rows
    )
    lhs_host = (
        np.sin(np.arange(lhs_rows * contraction, dtype=np.float32) * 0.0037)
        .reshape(lhs_rows, contraction)
        .astype(ml_dtypes.bfloat16)
    )
    linear = np.arange(output * contraction, dtype=np.uint64)
    # Codes 0..119 and 128..247 are finite E4M3FN values.  Avoid the two NaN
    # encodings 127/255 while exercising both signs and every finite exponent.
    weight_host = (
        (linear % np.uint64(120))
        + ((linear // np.uint64(120)) % np.uint64(2)) * np.uint64(128)
    ).astype(np.uint8).reshape(output, contraction)
    scale_host = np.linspace(
        0.0005,
        0.0015,
        (output // 128) * (contraction // 128),
        dtype=np.float32,
    ).reshape(output // 128, contraction // 128)
    gate_weight_host = (
        ((linear * np.uint64(37) + np.uint64(11)) % np.uint64(120))
        + ((linear // np.uint64(73)) % np.uint64(2)) * np.uint64(128)
    ).astype(np.uint8).reshape(output, contraction)
    gate_scale_host = np.linspace(
        0.000375,
        0.001625,
        (output // 128) * (contraction // 128),
        dtype=np.float32,
    ).reshape(output // 128, contraction // 128)
    local_experts = None
    route_indices_host = None
    gate_activation_host = None
    up_activation_host = None
    if selected_kernel:
        local_experts = 64
        if args.kernel == "selected_up_gate":
            single_weight = weight_host
            single_gate_weight = gate_weight_host
            # Materialize final selected-expert [G,K,N] checkpoint layout.
            weight_host = np.empty(
                (local_experts, contraction, output), dtype=np.uint8
            )
            gate_weight_host = np.empty_like(weight_host)
            for expert in range(local_experts):
                weight_host[expert] = np.bitwise_xor(
                    single_weight, np.uint8(expert & 1) * np.uint8(128)
                ).T
                gate_weight_host[expert] = np.bitwise_xor(
                    single_gate_weight,
                    np.uint8((expert // 2) & 1) * np.uint8(128),
                ).T
            scale_host = np.linspace(
                0.0005,
                0.0015,
                local_experts * (output // 128) * (contraction // 128),
                dtype=np.float32,
            ).reshape(local_experts, output // 128, contraction // 128)
            gate_scale_host = np.linspace(
                0.000375,
                0.001625,
                local_experts * (output // 128) * (contraction // 128),
                dtype=np.float32,
            ).reshape(local_experts, output // 128, contraction // 128)
        else:
            # Down final layout is [G,intermediate,hidden] = [G,2048,6144].
            single_down = weight_host
            weight_host = np.empty(
                (local_experts, output, contraction), dtype=np.uint8
            )
            for expert in range(local_experts):
                weight_host[expert] = np.bitwise_xor(
                    single_down, np.uint8(expert & 1) * np.uint8(128)
                )
            scale_host = np.linspace(
                0.0005,
                0.0015,
                local_experts * (contraction // 128) * (output // 128),
                dtype=np.float32,
            ).reshape(local_experts, contraction // 128, output // 128)
            activation_linear = np.arange(rows * output, dtype=np.float32).reshape(
                rows, output
            )
            gate_activation_host = (
                np.sin(activation_linear * 0.0031) * 0.75
            ).astype(ml_dtypes.bfloat16)
            up_activation_host = (
                np.cos(activation_linear * 0.0047) * 0.625
            ).astype(ml_dtypes.bfloat16)
        if args.selected_route_case == "normal_two":
            # Two routes owned by this 0:64 chip, interleaved with the six
            # routes owned by the other PP8 stage chips.
            route_indices_host = np.asarray(
                [0, 64, 128, 192, 1, 65, 129, 193], dtype=np.int32
            )
        else:
            route_indices_host = np.asarray(
                [0, 1, 7, 15, 31, 47, 55, 63], dtype=np.int32
            )

    with jax.default_device(device):
        lhs = jax.device_put(lhs_host, device)
        weight_bits = jax.device_put(weight_host, device)
        scale = jax.device_put(scale_host, device)
        reference_lhs = lhs
        if args.kernel in ("single_up", "attention_output"):
            def kernel(*values: Any) -> Any:
                return fp8_block_matmul(
                    *values,
                    config=Fp8BlockMatmulConfig(
                        output_tile=args.output_tile
                    ),
                )

            kernel_inputs = (lhs, weight_bits, scale)
            reference_inputs = ((args.kernel, weight_bits, scale),)
            if args.kernel == "attention_output":
                kernel_hlo_name = (
                    "greenfield_fp8_block_matmul_m8_k4096_n6144"
                    + (
                        f"_ot{args.output_tile}"
                        if args.output_tile != 128
                        else ""
                    )
                )
            else:
                kernel_hlo_name = "greenfield_fp8_block_matmul"
        elif args.kernel in ("dsa_wq_b", "dsa_wk"):
            kernel = fp8_block_matmul_f32
            kernel_inputs = (lhs, weight_bits, scale)
            reference_inputs = ((args.kernel, weight_bits, scale),)
            kernel_hlo_name = "greenfield_fp8_block_matmul_f32"
        elif args.kernel == "rmsnorm_linear":
            norm_weight_host = np.linspace(
                0.5, 1.5, contraction, dtype=np.float32
            ).astype(ml_dtypes.bfloat16)
            norm_weight = jax.device_put(norm_weight_host, device)
            kernel_inputs = (
                lhs,
                norm_weight,
                weight_bits,
                scale,
            )

            def kernel(*values: Any) -> Any:
                return fp8_rmsnorm_block_matmul(
                    *values, epsilon=1e-5
                )

            reference_lhs = rms_norm(lhs, norm_weight, epsilon=1e-5)
            reference_inputs = (("rmsnorm_linear", weight_bits, scale),)
            kernel_hlo_name = "greenfield_fp8_rmsnorm_block_matmul"
        elif args.kernel == "up_gate":
            gate_bits = jax.device_put(gate_weight_host, device)
            gate_scale = jax.device_put(gate_scale_host, device)
            kernel = fp8_block_up_gate
            kernel_inputs = (
                lhs,
                gate_bits,
                gate_scale,
                weight_bits,
                scale,
            )
            reference_inputs = (
                ("gate", gate_bits, gate_scale),
                ("up", weight_bits, scale),
            )
            kernel_hlo_name = "greenfield_fp8_block_up_gate"
        elif args.kernel == "selected_up_gate":
            assert local_experts is not None and route_indices_host is not None
            gate_bits = jax.device_put(gate_weight_host, device)
            gate_scale = jax.device_put(gate_scale_host, device)
            route_indices = jax.device_put(route_indices_host, device)
            expert_start = jax.device_put(np.asarray(0, dtype=np.int32), device)
            kernel = fp8_selected_up_gate
            kernel_inputs = (
                lhs,
                route_indices,
                expert_start,
                gate_bits,
                gate_scale,
                weight_bits,
                scale,
            )
            reference_inputs = (
                ("gate", gate_bits, gate_scale),
                ("up", weight_bits, scale),
            )
            kernel_hlo_name = "greenfield_fp8_selected_up_gate"
        else:
            assert local_experts is not None and route_indices_host is not None
            assert gate_activation_host is not None
            assert up_activation_host is not None
            gate_activation = jax.device_put(gate_activation_host, device)
            up_activation = jax.device_put(up_activation_host, device)
            route_indices = jax.device_put(route_indices_host, device)
            expert_start = jax.device_put(np.asarray(0, dtype=np.int32), device)
            kernel = fp8_selected_swiglu_down
            kernel_inputs = (
                gate_activation,
                up_activation,
                route_indices,
                expert_start,
                weight_bits,
                scale,
            )
            reference_inputs = (("down", weight_bits, scale),)
            kernel_hlo_name = "greenfield_fp8_selected_swiglu_down"
        lower_started = time.monotonic()
        compiled = jax.jit(kernel).lower(*kernel_inputs).compile()
        compile_seconds = time.monotonic() - lower_started
        hlo = compiled.as_text()
        hlo_sha256 = sha256(hlo.encode()).hexdigest()
        # Preserve compiler output even when the fail-closed HLO contract
        # rejects a diagnostic before correctness or timing.
        args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
        args.hlo_output.write_text(hlo)

        custom_calls = [
            line.strip()
            for line in hlo.splitlines()
            if "custom-call(" in line
        ]
        kernel_calls = [
            line
            for line in custom_calls
            if kernel_hlo_name in line or "tpu_custom_call" in line
        ]
        bounded_metadata_calls = [
            line
            for line in custom_calls
            if 'custom_call_target="AssumeGatherIndicesInBound"' in line
            and " = s32[" in line
        ]
        route_restore_kernel_name = (
            "fp8_selected_swiglu_down"
            if args.kernel == "selected_swiglu_down"
            else "fp8_selected_up_gate"
        )
        bitpacked_index_calls = [
            line
            for line in custom_calls
            if _is_bounded_route_restore_index_call(
                line,
                route_count=rows,
                kernel_name=route_restore_kernel_name,
            )
        ]
        hlo_lines = [line.strip() for line in hlo.splitlines()]
        bounded_compact_input_scatter_calls = [
            call
            for call in bitpacked_index_calls
            if any(
                _is_bounded_compact_input_scatter(
                    line,
                    index_result=_custom_call_result_name(call),
                    route_count=rows,
                    row_tile=8,
                    width=output,
                    kernel_name="fp8_selected_swiglu_down",
                )
                for line in hlo_lines
            )
        ]
        bounded_compact_input_scatters = [
            line
            for line in hlo_lines
            if any(
                _is_bounded_compact_input_scatter(
                    line,
                    index_result=_custom_call_result_name(call),
                    route_count=rows,
                    row_tile=8,
                    width=output,
                    kernel_name="fp8_selected_swiglu_down",
                )
                for call in bounded_compact_input_scatter_calls
            )
        ]
        bounded_route_restore_calls = [
            call
            for call in bitpacked_index_calls
            if call not in bounded_compact_input_scatter_calls
        ]
        route_restore_width = (
            contraction
            if args.kernel == "selected_swiglu_down"
            else 2 * output
        )
        bounded_route_restore_gathers = [
            line.strip()
            for line in hlo_lines
            if " gather(" in line
            and f" = bf16[{rows},{route_restore_width}]" in line
            and "collapsed_slice_dims={0,1}" in line
            and "start_index_map={0,1}" in line
            and f"slice_sizes={{1,1,{route_restore_width}}}" in line
            and any(
                f", {_custom_call_result_name(call)})" in line
                for call in bounded_route_restore_calls
            )
        ]
        selected_down_scale_index_calls = [
            line
            for line in bounded_metadata_calls
            if " = s32[1024]" in line
            and (
                'metadata={op_name="jit(fp8_selected_swiglu_down)/'
                'jit(_take)/gather"'
            ) in line
        ]
        selected_down_restore_index_calls = [
            line
            for line in bounded_metadata_calls
            if " = s32[1024]" in line
            and (
                'metadata={op_name="jit(fp8_selected_swiglu_down)/gather"'
            ) in line
        ]
        selected_down_scale_gathers = [
            line
            for line in hlo_lines
            if f" = f32[{rows},{contraction // 128},{output // 128}]" in line
            and " gather(" in line
            and "offset_dims={1,2}" in line
            and "collapsed_slice_dims={0}" in line
            and "start_index_map={0}" in line
            and (
                f"slice_sizes={{1,{contraction // 128},{output // 128}}}"
                in line
            )
            and (
                'metadata={op_name="jit(fp8_selected_swiglu_down)/'
                'jit(_take)/gather"'
            ) in line
        ]
        selected_down_restore_gathers = [
            line
            for line in hlo_lines
            if f" = bf16[{rows},8,{contraction}]" in line
            and " gather(" in line
            and "offset_dims={1,2}" in line
            and "collapsed_slice_dims={0}" in line
            and "start_index_map={0}" in line
            and f"slice_sizes={{1,8,{contraction}}}" in line
            and (
                'metadata={op_name="jit(fp8_selected_swiglu_down)/gather"'
            ) in line
        ]
        allowed_auxiliary_calls = (
            bounded_metadata_calls
            + bounded_compact_input_scatter_calls
            + bounded_route_restore_calls
        )
        unexpected_auxiliary_calls = [
            line
            for line in custom_calls
            if line not in kernel_calls and line not in allowed_auxiliary_calls
        ]
        forbidden_full_overlays = [
            shape
            for shape in tuple(
                dict.fromkeys((
                f"bf16[{output},{contraction}]",
                f"f32[{output},{contraction}]",
                f"bf16[{contraction},{output}]",
                f"f32[{contraction},{output}]",
                *(() if local_experts is None else (
                    f"bf16[{local_experts},{output},{contraction}]",
                    f"f32[{local_experts},{output},{contraction}]",
                    f"bf16[{local_experts},{contraction},{output}]",
                    f"f32[{local_experts},{contraction},{output}]",
                    f"bf16[{local_experts},{contraction},{2 * output}]",
                    f"f32[{local_experts},{contraction},{2 * output}]",
                    f"bf16[{local_experts},{2 * output},{contraction}]",
                    f"f32[{local_experts},{2 * output},{contraction}]",
                )),
                ))
            )
            if shape in hlo
        ]
        raw_weight_operand = f"u8[{output},{contraction}]"
        formatted_weight_operand = f"f8e4m3fn[{output},{contraction}]"
        required_raw_weight_count = 2 if args.kernel == "up_gate" else 1
        raw_weight_operand_count = (
            kernel_calls[0].count(raw_weight_operand) if kernel_calls else 0
        )
        forbidden_formatted_weight = (
            formatted_weight_operand in hlo and not selected_kernel
        )
        hlo_contract = {
            "custom_call_count": len(custom_calls),
            "kernel_custom_call_count": len(kernel_calls),
            "kernel_custom_calls": kernel_calls,
            "bounded_metadata_custom_call_count": len(
                allowed_auxiliary_calls
            ),
            "bounded_metadata_custom_calls": allowed_auxiliary_calls,
            "bounded_gather_index_custom_call_count": len(
                bounded_metadata_calls
            ),
            "bounded_gather_index_custom_calls": bounded_metadata_calls,
            "bounded_route_restore_custom_call_count": len(
                bounded_route_restore_calls
            ),
            "bounded_route_restore_custom_calls": bounded_route_restore_calls,
            "bounded_route_restore_gathers": bounded_route_restore_gathers,
            "bounded_compact_input_scatter_custom_call_count": len(
                bounded_compact_input_scatter_calls
            ),
            "bounded_compact_input_scatter_custom_calls": (
                bounded_compact_input_scatter_calls
            ),
            "bounded_compact_input_scatters": bounded_compact_input_scatters,
            "selected_down_scale_index_custom_calls": (
                selected_down_scale_index_calls
            ),
            "selected_down_scale_gathers": selected_down_scale_gathers,
            "selected_down_restore_index_custom_calls": (
                selected_down_restore_index_calls
            ),
            "selected_down_restore_gathers": selected_down_restore_gathers,
            "unexpected_auxiliary_custom_calls": unexpected_auxiliary_calls,
            "forbidden_full_weight_overlays": forbidden_full_overlays,
            "raw_weight_operand": raw_weight_operand,
            "raw_weight_operand_count": raw_weight_operand_count,
            "required_raw_weight_operand_count": required_raw_weight_count,
            "forbidden_formatted_weight_operand": (
                formatted_weight_operand if forbidden_formatted_weight else None
            ),
            "passed": (
                len(kernel_calls) == 1
                and not forbidden_full_overlays
                and not unexpected_auxiliary_calls
                and (
                    selected_kernel
                    or (
                        raw_weight_operand_count
                        >= required_raw_weight_count
                        and not forbidden_formatted_weight
                    )
                )
                and (
                    not selected_kernel
                    or (
                        2 <= len(allowed_auxiliary_calls) <= 8
                        and (
                            (
                                len(bounded_metadata_calls) == 2
                                and len(selected_down_scale_index_calls) == 1
                                and len(selected_down_scale_gathers) == 1
                                and len(selected_down_restore_index_calls) == 1
                                and len(selected_down_restore_gathers) == 1
                                and len(
                                    bounded_compact_input_scatter_calls
                                ) == 2
                                and len(bounded_compact_input_scatters) == 2
                                and not bounded_route_restore_calls
                            )
                            if args.kernel == "selected_swiglu_down"
                            else (
                                len(bounded_route_restore_calls) <= 1
                                and not bounded_compact_input_scatter_calls
                            )
                        )
                        and len(bounded_route_restore_gathers)
                        == len(bounded_route_restore_calls)
                        and (
                            f"u8[{local_experts},{output},{contraction}]"
                            if args.kernel == "selected_swiglu_down"
                            else f"u8[{local_experts},{contraction},{output}]"
                        )
                        in kernel_calls[0]
                        and (
                            f"f8e4m3fn[{local_experts},{output},{contraction}]"
                            if args.kernel == "selected_swiglu_down"
                            else f"f8e4m3fn[{local_experts},{contraction},{output}]"
                        )
                        not in hlo
                    )
                )
            ),
        }
        if not hlo_contract["passed"]:
            raise RuntimeError(f"FP8 Pallas HLO contract failed: {hlo_contract}")

        actual_raw = compiled(*kernel_inputs)
        single_output_kernel = args.kernel in (
            "single_up",
            "attention_output",
            "rmsnorm_linear",
            "selected_swiglu_down",
            "dsa_wq_b",
            "dsa_wk",
        )
        actual_values = (actual_raw,) if single_output_kernel else tuple(actual_raw)
        jax.block_until_ready(actual_values)
        expected_values = []
        selected_output_width = (
            contraction
            if args.kernel == "selected_swiglu_down"
            else output
        )
        for _, bits, projection_scale in reference_inputs:
            if not selected_kernel:
                decoded = dequantize_fp8_bits_block_weight(
                    bits, projection_scale
                )
                projection = lax.dot_general(
                    (
                        reference_lhs.astype(jnp.float32)
                        if args.kernel in ("dsa_wq_b", "dsa_wk")
                        else reference_lhs
                    ),
                    (
                        decoded.astype(jnp.float32)
                        if args.kernel in ("dsa_wq_b", "dsa_wk")
                        else decoded
                    ),
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                expected_values.append(
                    projection
                    if args.kernel in ("dsa_wq_b", "dsa_wk")
                    else projection.astype(jnp.bfloat16)
                )
                continue
            assert route_indices_host is not None
            assert local_experts is not None
            route_values = []
            for route_slot, global_expert in enumerate(
                route_indices_host.tolist()
            ):
                local_expert = global_expert
                if not 0 <= local_expert < local_experts:
                    route_values.append(
                        jnp.zeros(
                            (selected_output_width,),
                            dtype=jnp.bfloat16,
                        )
                    )
                    continue
                if args.kernel == "selected_swiglu_down":
                    assert gate_activation_host is not None
                    assert up_activation_host is not None
                    decoded = dequantize_fp8_bits_block_weight(
                        jnp.transpose(bits[local_expert]),
                        projection_scale[local_expert],
                    )
                    activated = (
                        gate_activation[route_slot]
                        * jax.nn.sigmoid(gate_activation[route_slot])
                        * up_activation[route_slot]
                    ).astype(jnp.bfloat16)
                    route_values.append(
                        lax.dot_general(
                            activated[None, :],
                            decoded,
                            dimension_numbers=(((1,), (1,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        ).astype(jnp.bfloat16)[0]
                    )
                    continue
                decoded = dequantize_fp8_bits_block_weight(
                    jnp.transpose(bits[local_expert]),
                    projection_scale[local_expert],
                )
                route_values.append(
                    lax.dot_general(
                        lhs,
                        decoded,
                        dimension_numbers=(((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)[0]
                )
            expected_values.append(jnp.stack(route_values, axis=0))
        jax.block_until_ready(tuple(expected_values))
        output_comparisons = []
        differences = []
        for (name, _, _), actual, expected in zip(
            reference_inputs, actual_values, expected_values, strict=True
        ):
            actual_host = np.asarray(actual, dtype=np.float32)
            expected_host = np.asarray(expected, dtype=np.float32)
            difference = np.abs(actual_host - expected_host)
            differences.append(difference.reshape(-1))
            output_comparisons.append(
                {
                    "name": name,
                    "all_finite": bool(np.isfinite(actual_host).all()),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "p99_abs": float(np.percentile(difference, 99)),
                    "passed": bool(
                        np.isfinite(actual_host).all()
                        and np.allclose(
                            actual_host,
                            expected_host,
                            rtol=0.02,
                            atol=0.0625,
                        )
                    ),
                }
            )
        combined_difference = np.concatenate(differences)
        comparison = {
            "outputs": output_comparisons,
            "all_finite": all(value["all_finite"] for value in output_comparisons),
            "max_abs": float(combined_difference.max()),
            "mean_abs": float(combined_difference.mean()),
            "p99_abs": float(np.percentile(combined_difference, 99)),
            "passed": all(value["passed"] for value in output_comparisons),
        }
        if not comparison["passed"]:
            raise RuntimeError(f"FP8 Pallas/reference comparison failed: {comparison}")

        for _ in range(args.warmup):
            jax.block_until_ready(compiled(*kernel_inputs))
        samples = []
        checksum = 0.0
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            value_raw = compiled(*kernel_inputs)
            jax.block_until_ready(value_raw)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
            values = (value_raw,) if single_output_kernel else tuple(value_raw)
            checksum += sum(
                float(np.asarray(value[0, 0], dtype=np.float32))
                for value in values
            )

    if args.kernel == "selected_swiglu_down":
        shape_record: dict[str, Any] = {
            "gate": [rows, output],
            "up": [rows, output],
            "weight_bits": list(weight_host.shape),
            "scale": list(scale_host.shape),
            "output": [rows, contraction],
        }
    else:
        shape_record = {
            "lhs": [lhs_rows, contraction],
            "weight_bits": list(weight_host.shape),
            "scale": list(scale_host.shape),
            "output": [rows, output],
        }
    if route_indices_host is not None:
        shape_record["route_indices"] = list(route_indices_host.shape)
        shape_record["local_experts"] = local_experts
    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "kernel": args.kernel,
        "output_tile": args.output_tile,
        "selected_route_case": (
            args.selected_route_case
            if selected_kernel
            else None
        ),
        "selected_local_route_count": (
            int(np.count_nonzero(route_indices_host < local_experts))
            if route_indices_host is not None and local_experts is not None
            else None
        ),
        "shape": shape_record,
        "dtype_contract": {
            "lhs": "bfloat16",
            "rmsnorm": (
                "fp32-square-mean-rsqrt;bf16-round;bf16-weight"
                if args.kernel == "rmsnorm_linear"
                else None
            ),
            "activation": (
                "bfloat16-swiglu"
                if args.kernel == "selected_swiglu_down"
                else None
            ),
            "weight_storage": "uint8:e4m3fn-bits",
            "scale": "float32",
            "tile_dequant": "float32-product-to-bfloat16",
            "accumulator": "float32",
            "output": (
                "float32"
                if args.kernel in ("dsa_wq_b", "dsa_wk")
                else "bfloat16"
            ),
        },
        "compile_seconds": compile_seconds,
        "hlo": {
            "path": str(args.hlo_output),
            "sha256": hlo_sha256,
            "contract": hlo_contract,
        },
        "comparison": comparison,
        "profiler_free_timing": True,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency": _percentiles(samples),
        "checksum": checksum,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
