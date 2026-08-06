"""Fail-closed HLO and numerical checks for the greenfield DSA scorer."""

from __future__ import annotations

from typing import Any


def validate_dsa_score_hlo(
    optimized_hlo: str,
    *,
    heads: int = 32,
    head_dim: int = 128,
    local_context: int = 65_536,
) -> dict[str, Any]:
    """Require one compact scorer kernel and no per-head score overlay."""

    if heads <= 0 or head_dim <= 0 or local_context <= 0:
        raise ValueError("DSA HLO dimensions must be positive")
    padded_context = ((local_context + 127) // 128) * 128
    expected_name = (
        f"greenfield_dsa_score_r1_h{heads}_d{head_dim}_s{padded_context}"
    )
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
    ]
    kernel_calls = [
        line
        for line in custom_calls
        if expected_name in line
        and 'custom_call_target="tpu_custom_call"' in line
    ]
    unexpected_custom_calls = [
        line for line in custom_calls if line not in kernel_calls
    ]
    forbidden_overlays = [
        shape
        for shape in (
            f"f32[{heads},{padded_context}]",
            f"f32[1,{heads},{padded_context}]",
            f"bf16[{heads},{padded_context}]",
            f"bf16[1,{heads},{padded_context}]",
            f"f32[32,{padded_context}]",
            f"bf16[32,{padded_context}]",
        )
        if shape in optimized_hlo
    ]
    forbidden_collectives = [
        token
        for token in (
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
            f"f32[32,{heads},{head_dim}]",
            f"bf16[32,{heads},{head_dim}]",
        )
        if shape in optimized_hlo
    ]
    violations: list[str] = []
    if len(kernel_calls) != 1:
        violations.append(
            f"expected one {expected_name} call, found {len(kernel_calls)}"
        )
    if unexpected_custom_calls:
        violations.append(
            f"unexpected DSA scorer custom calls: {unexpected_custom_calls}"
        )
    if forbidden_overlays:
        violations.append(
            f"forbidden per-head DSA score overlays: {forbidden_overlays}"
        )
    if forbidden_collectives:
        violations.append(
            f"standalone DSA scorer contains collectives: {forbidden_collectives}"
        )
    if forbidden_dead_rows:
        violations.append(
            f"DSA scorer contains batch-32 dead-row shapes: {forbidden_dead_rows}"
        )
    if kernel_calls:
        call = kernel_calls[0]
        for required in (
            f"f32[1,{heads},{head_dim}]",
            f"bf16[{padded_context},{head_dim}]",
            f"f32[1,{heads},128]",
            f"f32[1,{padded_context}]",
        ):
            if required not in call:
                violations.append(
                    f"DSA scorer call lacks required local shape {required}"
                )
    return {
        "expected_kernel_name": expected_name,
        "kernel_custom_call_count": len(kernel_calls),
        "kernel_custom_calls": kernel_calls,
        "unexpected_custom_calls": unexpected_custom_calls,
        "forbidden_per_head_overlays": forbidden_overlays,
        "forbidden_collectives": forbidden_collectives,
        "forbidden_dead_rows": forbidden_dead_rows,
        "passed": not violations,
        "violations": violations,
    }
