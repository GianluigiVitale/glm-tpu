"""Fail-closed HLO checks for the exact TPU-v4 DSA selector."""

from __future__ import annotations

from typing import Any, Literal


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def validate_dsa_topk_hlo(
    optimized_hlo: str,
    *,
    phase: Literal["local", "merge"],
    top_k: int = 2048,
    local_context: int = 65_536,
    local_block_size: int = 2048,
    local_group_size: int = 4,
) -> dict[str, Any]:
    """Require the exact bitonic tree and reject XLA sort/top-k fallbacks."""

    for name, value in (
        ("top_k", top_k),
        ("local_context", local_context),
        ("local_block_size", local_block_size),
        ("local_group_size", local_group_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if phase not in ("local", "merge"):
        raise ValueError(f"unsupported DSA top-k HLO phase {phase!r}")

    if phase == "local":
        groups = _next_power_of_two(_ceil_div(local_context, local_block_size))
        padded_context = groups * local_block_size
        expected_names = [
            "greenfield_dsa_topk_local_bitonic_select_"
            f"n{padded_context}_k{top_k}_b{local_block_size}_g{groups}"
        ]
        level = 0
        while groups > 1:
            expected_names.append(
                "greenfield_dsa_topk_local_bitonic_merge_"
                f"l{level}_g{groups}_k{top_k}"
            )
            groups //= 2
            level += 1
        required_shapes = (
            f"f32[1,{local_context}]",
            f"s32[{local_context}]",
            f"f32[1,{top_k}]",
            f"s32[1,{top_k}]",
        )
    else:
        groups = _next_power_of_two(local_group_size)
        expected_names = []
        level = 0
        while groups > 1:
            expected_names.append(
                "greenfield_dsa_topk_global_bitonic_merge_"
                f"l{level}_g{groups}_k{top_k}"
            )
            groups //= 2
            level += 1
        required_shapes = (
            f"f32[{local_group_size},1,{top_k}]",
            f"s32[{local_group_size},1,{top_k}]",
            f"s32[1,{top_k}]",
        )

    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
    ]
    kernel_calls = {
        name: [
            line
            for line in custom_calls
            if name in line and 'custom_call_target="tpu_custom_call"' in line
        ]
        for name in expected_names
    }
    expected_lines = {
        line for lines in kernel_calls.values() for line in lines
    }
    unexpected_custom_calls = [
        line for line in custom_calls if line not in expected_lines
    ]
    forbidden_operations = [
        token
        for token in (
            " sort(",
            " topk(",
            " top-k(",
            "approx_max_k",
            " all-gather(",
            " all-reduce(",
            " collective-permute(",
            " reduce-scatter(",
        )
        if token in optimized_hlo
    ]
    forbidden_dead_rows = [
        shape
        for shape in (
            f"f32[32,{local_context}]",
            f"s32[32,{local_context}]",
            f"f32[32,{top_k}]",
            f"s32[32,{top_k}]",
        )
        if shape in optimized_hlo
    ]
    missing_shapes = [shape for shape in required_shapes if shape not in optimized_hlo]
    violations: list[str] = []
    for name, lines in kernel_calls.items():
        if len(lines) != 1:
            violations.append(f"expected one {name} call, found {len(lines)}")
    if unexpected_custom_calls:
        violations.append(
            f"unexpected DSA top-k custom calls: {unexpected_custom_calls}"
        )
    if forbidden_operations:
        violations.append(
            f"DSA top-k contains forbidden operations: {forbidden_operations}"
        )
    if forbidden_dead_rows:
        violations.append(
            f"DSA top-k contains batch-32 dead rows: {forbidden_dead_rows}"
        )
    if missing_shapes:
        violations.append(f"DSA top-k lacks required shapes: {missing_shapes}")
    return {
        "phase": phase,
        "expected_kernel_names": expected_names,
        "expected_kernel_count": len(expected_names),
        "kernel_custom_call_counts": {
            name: len(lines) for name, lines in kernel_calls.items()
        },
        "custom_call_count": len(custom_calls),
        "unexpected_custom_calls": unexpected_custom_calls,
        "forbidden_operations": forbidden_operations,
        "forbidden_dead_rows": forbidden_dead_rows,
        "missing_required_shapes": missing_shapes,
        "passed": not violations,
        "violations": violations,
    }
