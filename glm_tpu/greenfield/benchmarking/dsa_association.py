"""Fail-closed HLO checks for the bounded layer-0 DSA association probe."""

from __future__ import annotations

from typing import Any, Literal


AssociationPhase = Literal[
    "legacy_state",
    "legacy_score",
    "one_row_score",
]


def validate_dsa_association_hlo(
    optimized_hlo: str,
    *,
    phase: AssociationPhase,
    context: int = 8156,
    decode_rows: int = 32,
    heads: int = 32,
    head_dim: int = 128,
    page_size: int = 512,
) -> dict[str, Any]:
    """Pin legacy diagnostic geometry separately from the one-row challenger."""

    dimensions = (context, decode_rows, heads, head_dim, page_size)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError("DSA association HLO dimensions must be positive")
    required_by_phase = {
        "legacy_state": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            "bf16[2048,6144]",
        ),
        "legacy_score": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            f"f32[{decode_rows},{heads},{page_size}]",
            f"f32[{context}]",
        ),
        "one_row_score": (
            f"f32[1,{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[1,{heads}]",
            f"f32[{heads},{page_size}]",
            f"f32[{context}]",
        ),
    }
    if phase not in required_by_phase:
        raise ValueError(f"unsupported DSA association HLO phase {phase!r}")
    required_shapes = required_by_phase[phase]
    missing_shapes = [
        shape for shape in required_shapes if shape not in optimized_hlo
    ]
    forbidden_operations = [
        token
        for token in (
            " all-gather(",
            " all-reduce(",
            " collective-permute(",
            " reduce-scatter(",
            "xla_python_cpu_callback",
            "host_callback",
            "outside_compilation",
        )
        if token in optimized_hlo
    ]
    forbidden_dead_rows = []
    if phase == "one_row_score":
        forbidden_dead_rows = [
            shape
            for shape in (
                f"f32[{decode_rows},{heads},{head_dim}]",
                f"bf16[{decode_rows},{heads},{head_dim}]",
            )
            if shape in optimized_hlo
        ]
    violations: list[str] = []
    if missing_shapes:
        violations.append(
            f"DSA association {phase} lacks exact shapes: {missing_shapes}"
        )
    if forbidden_operations:
        violations.append(
            f"DSA association {phase} contains forbidden operations: "
            f"{forbidden_operations}"
        )
    if forbidden_dead_rows:
        violations.append(
            f"one-row DSA challenger contains legacy dead rows: "
            f"{forbidden_dead_rows}"
        )
    return {
        "phase": phase,
        "required_shapes": list(required_shapes),
        "missing_shapes": missing_shapes,
        "forbidden_operations": forbidden_operations,
        "forbidden_dead_rows": forbidden_dead_rows,
        "diagnostic_batch32_allowed": phase != "one_row_score",
        "passed": not violations,
        "violations": violations,
    }
