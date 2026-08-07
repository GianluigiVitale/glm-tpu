"""Fail-closed HLO checks for the bounded layer-0 DSA association probe."""

from __future__ import annotations

from typing import Any, Literal


AssociationPhase = Literal[
    "legacy_state",
    "legacy_fused_qkv_runtime_pack",
    "legacy_fused_qkv_state",
    "legacy_runtime_fused_qkv_global_state",
    "legacy_runtime_fused_qkv_sharded_state",
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
        "legacy_fused_qkv_state": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            "bf16[2624,6144]",
            f"bf16[{decode_rows},576]",
            "legacy_fused_qkv_a_m32_n2624",
        ),
        "legacy_fused_qkv_runtime_pack": (
            "u8[2048,6144]",
            "f32[16,48]",
            "u8[576,6144]",
            "f32[5,48]",
            "f8e4m3fn[6144,2624]",
            "f32[48,2624]",
            "f8e4m3fn[32,6144,82]",
            "f32[32,48,82]",
            "legacy_fused_qkv_runtime_pack_tp32_n82",
        ),
        "legacy_runtime_fused_qkv_global_state": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            "f8e4m3fn[6144,2624]",
            "f32[48,2624]",
            f"bf16[{decode_rows},576]",
            "legacy_runtime_fused_qkv_a_m32_global_n2624",
        ),
        "legacy_runtime_fused_qkv_sharded_state": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            "f8e4m3fn[32,6144,82]",
            "f32[32,48,82]",
            f"bf16[{decode_rows},576]",
            "legacy_runtime_fused_qkv_a_m32_tp32_n82",
        ),
        "legacy_score": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            f"f32[{context}]",
        ),
        "one_row_score": (
            f"f32[1,{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[1,{heads}]",
            f"f32[{context}]",
        ),
    }
    if phase not in required_by_phase:
        raise ValueError(f"unsupported DSA association HLO phase {phase!r}")
    required_shapes = required_by_phase[phase]
    missing_shapes = [
        shape for shape in required_shapes if shape not in optimized_hlo
    ]
    score_intermediate_candidates: tuple[str, ...] = ()
    score_source_markers: tuple[str, ...] = ()
    if phase == "legacy_score":
        score_intermediate_candidates = (
            f"f32[{decode_rows},{heads},{page_size}]",
            f"f32[{decode_rows},{page_size},{heads}]",
        )
        score_source_markers = (
            "thd,tpd->thp/dot_general",
            "th,thp->tp/dot_general",
        )
    elif phase == "one_row_score":
        score_intermediate_candidates = (
            f"f32[{heads},{page_size}]",
            f"f32[{page_size},{heads}]",
        )
        score_source_markers = (
            "hd,pd->hp/dot_general",
            "h,hp->p/dot_general",
        )
    score_intermediate_shapes = [
        shape
        for shape in score_intermediate_candidates
        if shape in optimized_hlo
    ]
    missing_score_markers = [
        marker for marker in score_source_markers if marker not in optimized_hlo
    ]
    fused_qkv_candidates: tuple[str, ...] = ()
    if phase == "legacy_fused_qkv_state":
        fused_qkv_candidates = (
            f"bf16[{decode_rows},2624]",
            f"bf16[2624,{decode_rows}]",
        )
    elif phase == "legacy_runtime_fused_qkv_global_state":
        fused_qkv_candidates = (
            f"bf16[{decode_rows},2624]",
            f"bf16[2624,{decode_rows}]",
        )
    elif phase == "legacy_runtime_fused_qkv_sharded_state":
        fused_qkv_candidates = tuple(
            dict.fromkeys(
                (
                    f"bf16[32,{decode_rows},82]",
                    f"bf16[32,82,{decode_rows}]",
                    f"bf16[{decode_rows},32,82]",
                    f"bf16[{decode_rows},82,32]",
                )
            )
        )
    fused_qkv_intermediate_shapes = [
        shape for shape in fused_qkv_candidates if shape in optimized_hlo
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
    if score_intermediate_candidates and not score_intermediate_shapes:
        violations.append(
            f"DSA association {phase} lacks an exact logical/physical score tile: "
            f"{list(score_intermediate_candidates)}"
        )
    if missing_score_markers:
        violations.append(
            f"DSA association {phase} lost score source markers: "
            f"{missing_score_markers}"
        )
    if fused_qkv_candidates and not fused_qkv_intermediate_shapes:
        violations.append(
            "DSA association fused qkv_a state lacks its exact "
            f"logical/physical output: {list(fused_qkv_candidates)}"
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
        "score_intermediate_shapes": score_intermediate_shapes,
        "missing_score_markers": missing_score_markers,
        "fused_qkv_intermediate_shapes": fused_qkv_intermediate_shapes,
        "forbidden_operations": forbidden_operations,
        "forbidden_dead_rows": forbidden_dead_rows,
        "diagnostic_batch32_allowed": phase != "one_row_score",
        "passed": not violations,
        "violations": violations,
    }
