"""Correctness and optimized-HLO gates for the protected real MoE layer."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..sharding.hlo_contract import parse_hlo_module


@dataclass(frozen=True, slots=True)
class TensorTolerance:
    max_abs: float
    p99_abs: float
    mean_abs: float

    def __post_init__(self) -> None:
        if min(self.max_abs, self.p99_abs, self.mean_abs) < 0:
            raise ValueError("tensor tolerances must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p99_abs": self.p99_abs,
        }


REAL_LAYER_OUTPUT_TOLERANCE = TensorTolerance(
    max_abs=0.125,
    p99_abs=0.0625,
    mean_abs=0.02,
)
ROUTE_WEIGHT_TOLERANCE = TensorTolerance(
    max_abs=5e-4,
    p99_abs=5e-4,
    mean_abs=2e-4,
)


def compare_bounded_tensor(
    observed: Any,
    reference: Any,
    tolerance: TensorTolerance,
) -> dict[str, Any]:
    """Return a complete bounded-error record and fail only at the caller."""

    observed_array = np.asarray(observed, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    if observed_array.shape != reference_array.shape:
        raise ValueError(
            "bounded tensor shape mismatch: "
            f"observed={observed_array.shape} reference={reference_array.shape}"
        )
    if not np.isfinite(observed_array).all() or not np.isfinite(
        reference_array
    ).all():
        raise ValueError("bounded tensor comparison contains non-finite values")
    error = np.abs(observed_array - reference_array)
    maximum = float(np.max(error, initial=0.0))
    mean = float(np.mean(error)) if error.size else 0.0
    p99 = float(np.quantile(error, 0.99)) if error.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
    maximum_flat_index = int(np.argmax(error)) if error.size else 0
    maximum_index = (
        [
            int(value)
            for value in np.unravel_index(maximum_flat_index, error.shape)
        ]
        if error.size
        else []
    )
    passed = (
        maximum <= tolerance.max_abs
        and p99 <= tolerance.p99_abs
        and mean <= tolerance.mean_abs
    )
    return {
        "error": {
            "max_abs": maximum,
            "max_abs_index": maximum_index,
            "mean_abs": mean,
            "p99_abs": p99,
            "rms": rms,
        },
        "observed_range": [
            float(np.min(observed_array, initial=0.0)),
            float(np.max(observed_array, initial=0.0)),
        ],
        "passed": passed,
        "reference_range": [
            float(np.min(reference_array, initial=0.0)),
            float(np.max(reference_array, initial=0.0)),
        ],
        "shape": list(error.shape),
        "tolerance": tolerance.to_dict(),
    }


def validate_real_layer_hlo(
    optimized_hlo: str,
    *,
    hidden_size: int = 6144,
    stage_size: int = 4,
    reconstruct_down_fp32: bool = False,
) -> dict[str, Any]:
    """Require only the declared local reconstruction/combine collectives."""

    module = parse_hlo_module(optimized_hlo)
    violations = []
    collectives = module.collectives
    expected_collective_count = 2 if reconstruct_down_fp32 else 1
    if len(collectives) != expected_collective_count:
        expected_label = (
            "one"
            if expected_collective_count == 1
            else str(expected_collective_count)
        )
        violations.append(
            f"expected exactly {expected_label} collectives, "
            f"found {len(collectives)}"
        )
    expected_groups = (tuple(range(stage_size)),)
    for collective in collectives:
        if collective.opcode != "all-reduce":
            violations.append(
                f"expected all-reduce combine, found {collective.opcode}"
            )
        if collective.replica_groups != expected_groups:
            violations.append(
                "expected exact local replica group "
                f"{expected_groups}, found {collective.replica_groups}"
            )
    expected_payloads = [("bf16", (2, 1, hidden_size))]
    if reconstruct_down_fp32:
        expected_payloads.append(("f32", (8, hidden_size)))
    observed_payloads = [
        (shape.dtype.lower(), shape.dimensions)
        for collective in collectives
        for shape in collective.result_shapes
    ]
    for expected_dtype, expected_shape in expected_payloads:
        accepted_dtypes = (
            {"bf16", "bfloat16"}
            if expected_dtype == "bf16"
            else {expected_dtype}
        )
        if not any(
            dtype in accepted_dtypes and shape == expected_shape
            for dtype, shape in observed_payloads
        ):
            violations.append(
                "local collective payload is missing exact "
                f"{expected_dtype}{expected_shape}: {observed_payloads}"
            )
    forbidden = []
    for instruction in module.instructions:
        for shape in instruction.result_shapes:
            if shape.dimensions == (32, hidden_size):
                forbidden.append(instruction.name)
    if forbidden:
        violations.append(
            f"found forbidden dead-row/full-pod hidden shapes: {forbidden}"
        )
    if module.num_partitions not in (None, stage_size):
        violations.append(
            f"expected {stage_size} partitions, found {module.num_partitions}"
        )
    return {
        "collective_count": len(collectives),
        "expected_collective_count": expected_collective_count,
        "collectives": [item.to_dict() for item in collectives],
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": not violations,
        "violations": violations,
    }


def validate_pallas_real_layer_hlo(
    optimized_hlo: str,
    *,
    hidden_size: int = 6144,
    intermediate_size: int = 2048,
    local_experts: int = 64,
    stage_size: int = 4,
    routed_intermediate_size: int | None = None,
    feature_sharded_routed: bool = False,
    routed_output_tile: int = 128,
    fuse_route_weighting: bool = False,
    reconstruct_down_fp32: bool = False,
) -> dict[str, Any]:
    """Require raw-FP8 kernels, bounded metadata, and local combines."""

    routed_intermediate = (
        intermediate_size
        if routed_intermediate_size is None
        else routed_intermediate_size
    )
    if routed_intermediate <= 0:
        raise ValueError("routed intermediate size must be positive")
    if routed_output_tile not in (128, 256):
        raise ValueError("routed output tile must be 128 or 256")
    if not isinstance(fuse_route_weighting, bool):
        raise ValueError("route-weight fusion flag must be boolean")
    if fuse_route_weighting and not feature_sharded_routed:
        raise ValueError(
            "route-weight fusion requires the feature-sharded routed layout"
        )
    if not isinstance(reconstruct_down_fp32, bool):
        raise ValueError("FP32 reconstruction flag must be boolean")
    if reconstruct_down_fp32 and not feature_sharded_routed:
        raise ValueError(
            "FP32 reconstruction requires the feature-sharded routed layout"
        )
    if reconstruct_down_fp32 and fuse_route_weighting:
        raise ValueError(
            "FP32 reconstruction is incompatible with fused route weighting"
        )

    base = validate_real_layer_hlo(
        optimized_hlo,
        hidden_size=hidden_size,
        stage_size=stage_size,
        reconstruct_down_fp32=reconstruct_down_fp32,
    )
    violations = list(base["violations"])
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
    ]
    kernel_prefixes = (
        "greenfield_fp8_fused_selected_moe_",
        "greenfield_fp8_block_up_gate_",
        "greenfield_fp8_block_matmul_",
    ) + (
        ("greenfield_fp32_to_bf16_",)
        if reconstruct_down_fp32
        else ()
    )
    kernel_calls = {
        prefix: [
            line
            for line in custom_calls
            if prefix in line
            and 'custom_call_target="tpu_custom_call"' in line
        ]
        for prefix in kernel_prefixes
    }
    for prefix, lines in kernel_calls.items():
        if len(lines) != 1:
            violations.append(
                f"expected one {prefix} Pallas call, found {len(lines)}"
            )

    target_pattern = re.compile(r'custom_call_target="([^"]+)"')
    targets = []
    for line in custom_calls:
        match = target_pattern.search(line)
        targets.append(match.group(1) if match else "<missing>")
    allowed_targets = {
        "tpu_custom_call",
        "AssumeGatherIndicesInBound",
        # The TPU layout pass lowers each already-local shared FP8 table to
        # four asynchronous VMEM slices followed by this non-collective
        # bitwise reassembly.  Exact output shapes are checked below.
        "ConcatBitcast",
    }
    unexpected_targets = sorted(
        target for target in targets if target not in allowed_targets
    )
    if unexpected_targets:
        violations.append(
            f"unexpected Pallas-layer custom-call targets: {unexpected_targets}"
        )
    target_counts = {
        target: targets.count(target) for target in sorted(set(targets))
    }
    expected_target_counts = {
        # Three selected scale-table gathers, one final routed-output restore,
        # and the exact correction-bias lookup from the 256-entry vector. The
        # fused route-sum path has no final routed-output restore.
        "AssumeGatherIndicesInBound": (
            4 if fuse_route_weighting else 5
        ),
        "ConcatBitcast": 3,
        "tpu_custom_call": 4 if reconstruct_down_fp32 else 3,
    }
    if target_counts != expected_target_counts:
        violations.append(
            "Pallas-layer custom-call counts drifted: "
            f"expected={expected_target_counts} observed={target_counts}"
        )

    concat_calls = [
        line
        for line in custom_calls
        if 'custom_call_target="ConcatBitcast"' in line
    ]
    # The current raw-U8 path reconstructs only the already-local shared
    # gate/up and down shards at the four-device layout boundary.  The older
    # F8-input path could instead expose a BF16 router layout marker; reject
    # that historical variant so the one-layer guard matches the decoder's
    # protected direct-U8 contract.
    expected_concat_shapes = {
        f"u8[{intermediate_size // stage_size},{hidden_size}]": 2,
        f"u8[{hidden_size},{intermediate_size // stage_size}]": 1,
    }
    observed_concat_shapes = {
        shape: sum(f"= {shape}" in line for line in concat_calls)
        for shape in expected_concat_shapes
    }
    if observed_concat_shapes != expected_concat_shapes:
        violations.append(
            "local shared-FP8 ConcatBitcast shapes drifted: "
            f"expected={expected_concat_shapes} "
            f"observed={observed_concat_shapes}"
        )
    concat_arity = [
        len(line.split("custom-call(", 1)[1].split(")", 1)[0].split(","))
        for line in concat_calls
    ]
    # ConcatBitcast arity is TPU v4's four-way on-device layout tiling, not
    # the physical stage width. LP2 therefore retains four local slices while
    # its sole real collective remains the independently checked two-rank sum.
    expected_concat_arity = 4
    if concat_arity != [expected_concat_arity] * len(concat_calls):
        violations.append(
            "local shared-FP8 ConcatBitcast arity drifted: "
            f"expected={expected_concat_arity} observed={concat_arity}"
        )

    selected_line = kernel_calls["greenfield_fp8_fused_selected_moe_"]
    expected_selected_name = (
        "greenfield_fp8_fused_selected_moe_"
        f"r8_g{local_experts}_h{hidden_size}_i{routed_intermediate}"
    )
    if routed_output_tile != 128:
        expected_selected_name += f"_ot{routed_output_tile}"
    if reconstruct_down_fp32:
        expected_selected_name += "_downf32"
    if fuse_route_weighting:
        expected_selected_name += "_wsum"
    expected_selected_pattern = re.compile(
        re.escape(expected_selected_name) + r"(?=[^A-Za-z0-9_]|$)"
    )
    if selected_line and not expected_selected_pattern.search(selected_line[0]):
        violations.append(
            "fused selected kernel fingerprint drifted: expected "
            f"{expected_selected_name}"
        )
    expected_selected_result = (
        f"bf16[8,{hidden_size}]"
        if fuse_route_weighting
        else (
            f"f32[8,8,{hidden_size}]"
            if reconstruct_down_fp32
            else f"bf16[8,8,{hidden_size}]"
        )
    )
    if selected_line and expected_selected_result not in selected_line[0]:
        violations.append(
            "fused selected result shape drifted: expected "
            f"{expected_selected_result}"
        )
    if reconstruct_down_fp32:
        boundary_lines = kernel_calls["greenfield_fp32_to_bf16_"]
        if boundary_lines and (
            f"bf16[8,{hidden_size}]" not in boundary_lines[0]
            or f"f32[8,{hidden_size}]" not in boundary_lines[0]
        ):
            violations.append(
                "FP32 reconstruction boundary lacks exact FP32 operand and "
                "BF16 result"
            )
    if fuse_route_weighting and (
        f"bf16[8,8,{hidden_size}]" in optimized_hlo
    ):
        violations.append(
            "fused route weighting retains the routed [8,8,H] HBM result"
        )
    if selected_line and selected_line[0].count(
        f"u8[{local_experts},{hidden_size},{routed_intermediate}]"
    ) < 2:
        violations.append(
            "fused selected call lacks two exact raw-U8 gate/up tables"
        )
    if selected_line and (
        f"u8[{local_experts},{routed_intermediate},{hidden_size}]"
        not in selected_line[0]
    ):
        violations.append(
            "fused selected call lacks its exact raw-U8 down table"
        )
    full_decoded_shapes = tuple(
        dict.fromkeys(
            (
                f"bf16[{local_experts},{hidden_size},{routed_intermediate}]",
                f"f32[{local_experts},{hidden_size},{routed_intermediate}]",
                f"bf16[{local_experts},{routed_intermediate},{hidden_size}]",
                f"f32[{local_experts},{routed_intermediate},{hidden_size}]",
                f"bf16[{hidden_size},{intermediate_size // stage_size}]",
                f"f32[{hidden_size},{intermediate_size // stage_size}]",
                f"bf16[{intermediate_size // stage_size},{hidden_size}]",
                f"f32[{intermediate_size // stage_size},{hidden_size}]",
            )
        )
    )
    forbidden_overlays = [
        shape for shape in full_decoded_shapes if shape in optimized_hlo
    ]
    if forbidden_overlays:
        violations.append(
            f"found forbidden complete decoded weight overlays: {forbidden_overlays}"
        )
    return {
        **base,
        "custom_call_count": len(custom_calls),
        "custom_call_target_counts": target_counts,
        "expected_custom_call_target_counts": expected_target_counts,
        "forbidden_decoded_overlays": forbidden_overlays,
        "kernel_custom_call_count": sum(
            len(lines) for lines in kernel_calls.values()
        ),
        "kernel_custom_calls": kernel_calls,
        "local_layout_custom_call_count": len(concat_calls),
        "local_layout_custom_calls": concat_calls,
        "routed_output_tile": routed_output_tile,
        "fuse_route_weighting": fuse_route_weighting,
        "reconstruct_down_fp32": reconstruct_down_fp32,
        "routed_layout": (
            "expert_intermediate_shard"
            if feature_sharded_routed
            else "complete_expert_identity_shard"
        ),
        "passed": not violations,
        "violations": violations,
    }
