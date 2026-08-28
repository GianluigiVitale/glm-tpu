"""Exactness and HLO gates for the bounded PP16 dense boundary probe."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any

import ml_dtypes
import numpy as np

from ..sharding.hlo_contract import parse_hlo_module
from .association_fingerprint import replay_db533_strategy_nd_row0_bits


_HOST_MARKERS = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)
_FORBIDDEN_STABLE_COLLECTIVES = (
    "stablehlo.all_gather",
    "stablehlo.all_to_all",
    "stablehlo.collective_permute",
    "stablehlo.reduce_scatter",
)
PP16_DENSE_BOUNDARY_ACCEPTED_DENSE_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
PP16_DENSE_BOUNDARY_ACCEPTED_RESIDUAL_SHA256 = (
    "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
)


def derive_expected_dense_boundary_bits(
    dense_partial_bits: np.ndarray,
    post_attention_residual_bits: np.ndarray,
    model_axis_device_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive accepted dense update and carried residual from sealed leaves."""

    dense_partial_bits = np.ascontiguousarray(dense_partial_bits)
    post_attention_residual_bits = np.ascontiguousarray(
        post_attention_residual_bits
    )
    if (
        dense_partial_bits.dtype != np.uint16
        or dense_partial_bits.shape != (4, 8, 1, 6144)
    ):
        raise ValueError("dense boundary partial bits geometry drifted")
    if (
        post_attention_residual_bits.dtype != np.uint16
        or post_attention_residual_bits.shape != (1, 6144)
    ):
        raise ValueError("dense boundary carried-input bits geometry drifted")
    dense_bits = replay_db533_strategy_nd_row0_bits(
        dense_partial_bits.reshape(32, 6144), model_axis_device_ids
    ).reshape(1, 6144)
    carried_bits = np.ascontiguousarray(
        np.asarray(
            dense_bits.view(ml_dtypes.bfloat16).astype(np.float32)
            + post_attention_residual_bits.view(ml_dtypes.bfloat16).astype(
                np.float32
            ),
            dtype=ml_dtypes.bfloat16,
        )
    ).view(np.uint16)
    return dense_bits, carried_bits


def exact_bfloat16_bits(
    expected: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    """Compare two BF16 bit arrays without silently converting arithmetic."""

    expected = np.ascontiguousarray(expected)
    observed = np.ascontiguousarray(observed)
    if expected.dtype != np.uint16 or observed.dtype != np.uint16:
        raise ValueError("exact BF16 comparison requires uint16 bit arrays")
    if expected.shape != observed.shape:
        raise ValueError(
            "exact BF16 comparison shape drifted: "
            f"expected={expected.shape} observed={observed.shape}"
        )
    mismatches = np.flatnonzero(expected.reshape(-1) != observed.reshape(-1))
    return {
        "elementwise_exact": bool(mismatches.size == 0),
        "expected_sha256": sha256(expected.tobytes(order="C")).hexdigest(),
        "first_mismatch_flat_index": (
            None if mismatches.size == 0 else int(mismatches[0])
        ),
        "mismatch_count": int(mismatches.size),
        "observed_sha256": sha256(observed.tobytes(order="C")).hexdigest(),
        "shape": list(expected.shape),
    }


def validate_pp16_dense_boundary_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    hidden_size: int = 6144,
) -> dict[str, Any]:
    """Require one LP2 combine, one dense kernel, and no dead/global rows."""

    module = parse_hlo_module(optimized_hlo)
    violations: list[str] = []
    collectives = module.collectives
    expected_groups = ((0, 1),)
    if len(collectives) != 1:
        violations.append(
            f"expected exactly one optimized collective, found {len(collectives)}"
        )
    for collective in collectives:
        if collective.opcode != "all-reduce":
            violations.append(
                f"expected LP2 all-reduce, found {collective.opcode}"
            )
        if collective.replica_groups != expected_groups:
            violations.append(
                "LP2 collective group drifted: "
                f"expected={expected_groups} observed={collective.replica_groups}"
            )
        payloads = {
            (shape.dtype.lower(), shape.dimensions)
            for shape in collective.result_shapes
        }
        if not any(
            dtype in {"bf16", "bfloat16"}
            and dimensions == (1, hidden_size)
            for dtype, dimensions in payloads
        ):
            violations.append(
                "LP2 combine lost the exact one-row hidden payload: "
                f"{sorted(payloads)}"
            )
    if module.num_partitions != 2:
        violations.append(
            f"expected exactly two partitions, found {module.num_partitions}"
        )

    kernel_name = (
        "greenfield_fp8_fused_block_swiglu_"
        f"m8_h{hidden_size}_i{hidden_size}_o{hidden_size}"
    )
    custom_calls = tuple(
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    )
    kernel_count = sum(kernel_name in line for line in custom_calls)
    if len(custom_calls) != 1 or kernel_count != 1:
        violations.append(
            "PP16 dense kernel set drifted: "
            f"custom_calls={len(custom_calls)} expected_kernel={kernel_count}"
        )

    forbidden_shapes: list[dict[str, Any]] = []
    for instruction in module.instructions:
        for shape in instruction.result_shapes:
            if shape.dimensions in {
                (32, hidden_size),
                (32, 1, hidden_size),
                (1, 32, hidden_size),
            }:
                forbidden_shapes.append(
                    {
                        "dimensions": list(shape.dimensions),
                        "instruction": instruction.name,
                    }
                )
    if forbidden_shapes:
        violations.append("optimized HLO reconstructs a 32-row hidden tensor")

    host_markers = sorted(
        marker
        for marker in _HOST_MARKERS
        if marker in optimized_hlo.lower() or marker in stablehlo.lower()
    )
    if host_markers:
        violations.append(f"host execution markers present: {host_markers}")

    stable_collective_count = stablehlo.count("stablehlo.all_reduce")
    if stable_collective_count != 1:
        violations.append(
            "StableHLO expected exactly one all_reduce, found "
            f"{stable_collective_count}"
        )
    stable_forbidden = sorted(
        marker for marker in _FORBIDDEN_STABLE_COLLECTIVES if marker in stablehlo
    )
    if stable_forbidden:
        violations.append(
            f"StableHLO contains forbidden collectives: {stable_forbidden}"
        )
    compact_stable = "".join(stablehlo.split())
    stable_group_exact = (
        "replica_groups=dense<[[0,1]]>:tensor<1x2xi64>" in compact_stable
    )
    if not stable_group_exact:
        violations.append("StableHLO lost explicit replica group [[0,1]]")

    entry_roots = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )
    root_shapes = Counter(
        (shape.dtype.lower(), shape.dimensions)
        for root in entry_roots
        for shape in root.result_shapes
    )
    expected_root_shapes = Counter({("bf16", (1, hidden_size)): 3})
    if len(entry_roots) != 1 or root_shapes != expected_root_shapes:
        violations.append(
            "optimized HLO lost the three one-row boundary outputs: "
            f"{dict(root_shapes)}"
        )

    return {
        "collectives": [item.to_dict() for item in collectives],
        "custom_call_count": len(custom_calls),
        "expected_kernel_name": kernel_name,
        "forbidden_shapes": forbidden_shapes,
        "host_markers": host_markers,
        "kernel_count": kernel_count,
        "num_partitions": module.num_partitions,
        "optimized_hlo_sha256": sha256(optimized_hlo.encode()).hexdigest(),
        "passed": not violations,
        "stable_group_exact": stable_group_exact,
        "stablehlo_all_reduce_count": stable_collective_count,
        "stablehlo_forbidden_collectives": stable_forbidden,
        "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
        "violations": violations,
    }
