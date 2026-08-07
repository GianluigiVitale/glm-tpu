"""Fail-closed HLO checks for the bounded layer-0 DSA association probe."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from ..sharding.hlo_contract import parse_hlo_module


AssociationPhase = Literal[
    "legacy_state",
    "legacy_fused_qkv_runtime_pack",
    "legacy_fused_qkv_state",
    "legacy_runtime_fused_qkv_global_state",
    "legacy_runtime_fused_qkv_sharded_state",
    "legacy_tp32_distributed_q_a_norm",
    "legacy_tp32_gspmd_q_a_norm",
    "legacy_tp32_distributed_q_a_norm_state",
    "legacy_score",
    "legacy_local_dcp_xla_score",
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
    hidden_size: int = 6144,
    q_lora_rank: int = 2048,
    qkv_a_companion_rank: int = 576,
    tensor_shards: int = 32,
    dcp_size: int = 8,
    local_cache_pages: int = 24,
    max_model_len: int = 8704,
    allow_cpu_bf16_collective_promotion: bool = False,
) -> dict[str, Any]:
    """Pin legacy diagnostic geometry separately from the one-row challenger."""

    dimensions = (
        context,
        decode_rows,
        heads,
        head_dim,
        page_size,
        hidden_size,
        q_lora_rank,
        qkv_a_companion_rank,
        tensor_shards,
        dcp_size,
        local_cache_pages,
        max_model_len,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError("DSA association HLO dimensions must be positive")
    if q_lora_rank % tensor_shards or (
        qkv_a_companion_rank % tensor_shards
    ) or hidden_size % 128:
        raise ValueError("DSA association distributed geometry must divide")
    if not isinstance(allow_cpu_bf16_collective_promotion, bool):
        raise ValueError("CPU collective-promotion flag must be boolean")
    q_local = q_lora_rank // tensor_shards
    companion_local = qkv_a_companion_rank // tensor_shards
    local_output = q_local + companion_local
    global_page_size = page_size * dcp_size
    owned_blocks = (
        max_model_len + global_page_size - 1
    ) // global_page_size
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
        "legacy_tp32_distributed_q_a_norm": (
            f"f8e4m3fn[{hidden_size},{local_output}]",
            f"f32[{hidden_size // 128},{local_output}]",
            f"bf16[{decode_rows},{q_lora_rank}]",
            f"bf16[{decode_rows},{companion_local}]",
            "legacy_runtime_fused_qkv_a_m32_tp32_distributed_norm",
            "legacy_tp32_q_a_rms_norm_variance_psum",
            "legacy_tp32_q_a_rms_norm_bf16_all_gather",
        ),
        "legacy_tp32_gspmd_q_a_norm": (
            f"f8e4m3fn[{hidden_size},{local_output}]",
            f"f32[{hidden_size // 128},{local_output}]",
            f"bf16[{decode_rows},{q_lora_rank}]",
            f"bf16[{decode_rows},{companion_local}]",
            "legacy_runtime_fused_qkv_a_m32_tp32_gspmd_norm",
            "legacy_tp32_q_a_rms_norm_source_mean",
        ),
        "legacy_tp32_distributed_q_a_norm_state": (
            f"bf16[{decode_rows},{q_lora_rank}]",
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            f"bf16[{decode_rows},{qkv_a_companion_rank}]",
        ),
        "legacy_score": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{context},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            f"f32[{context}]",
        ),
        "legacy_local_dcp_xla_score": (
            f"f32[{decode_rows},{heads},{head_dim}]",
            f"bf16[{local_cache_pages},{page_size},{head_dim}]",
            f"f32[{decode_rows},{heads}]",
            f"s32[{decode_rows},{owned_blocks}]",
            f"s32[{decode_rows}]",
            f"f32[{decode_rows},{owned_blocks * page_size}]",
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
    if phase in ("legacy_score", "legacy_local_dcp_xla_score"):
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
    allowed_distributed_collectives = phase in (
        "legacy_tp32_distributed_q_a_norm",
        "legacy_tp32_gspmd_q_a_norm",
    )
    forbidden_tokens = (
        " collective-permute(",
        " reduce-scatter(",
        " all-to-all(",
        "xla_python_cpu_callback",
        "host_callback",
        "outside_compilation",
    )
    if not allowed_distributed_collectives:
        forbidden_tokens = (
            " all-gather(",
            " all-reduce(",
            *forbidden_tokens,
        )
    forbidden_operations = [
        token
        for token in forbidden_tokens
        if token in optimized_hlo
    ]
    distributed_collective_contract: dict[str, Any] | None = None
    distributed_collective_violations: list[str] = []
    if allowed_distributed_collectives:
        try:
            module = parse_hlo_module(optimized_hlo)
        except ValueError as error:
            distributed_collective_violations.append(
                f"distributed q-a norm HLO parse failed: {error}"
            )
        else:
            collectives = module.collectives
            counts = Counter(item.opcode for item in collectives)
            expected_counts = Counter({"all-reduce": 1, "all-gather": 1})
            if counts != expected_counts:
                distributed_collective_violations.append(
                    "distributed q-a norm requires exactly one all-reduce and "
                    f"one all-gather, found {dict(sorted(counts.items()))}"
                )
            expected_groups = (tuple(range(tensor_shards)),)
            for item in collectives:
                if item.replica_groups != expected_groups:
                    distributed_collective_violations.append(
                        f"{item.name} replica groups drifted: "
                        f"expected={expected_groups} found={item.replica_groups}"
                    )
                if not item.use_global_device_ids:
                    distributed_collective_violations.append(
                        f"{item.name} does not use global device ids"
                    )
            reductions = [
                item for item in collectives if item.opcode == "all-reduce"
            ]
            gathers = [
                item for item in collectives if item.opcode == "all-gather"
            ]
            reduction_shapes = [
                (shape.dtype, shape.dimensions)
                for item in reductions
                for shape in item.result_shapes
            ]
            gather_shapes = [
                (shape.dtype, shape.dimensions)
                for item in gathers
                for shape in item.result_shapes
            ]
            expected_reduction_shapes = {
                ("f32", (decode_rows,)),
                ("f32", (decode_rows, 1)),
            }
            expected_gather_shapes = {
                ("bf16", (decode_rows, q_local, tensor_shards))
            }
            if phase == "legacy_tp32_gspmd_q_a_norm":
                expected_gather_shapes.add(
                    ("bf16", (decode_rows, q_lora_rank))
                )
            if allow_cpu_bf16_collective_promotion:
                expected_gather_shapes.add(
                    ("f32", (decode_rows, q_local, tensor_shards))
                )
                if phase == "legacy_tp32_gspmd_q_a_norm":
                    expected_gather_shapes.add(
                        ("f32", (decode_rows, q_lora_rank))
                    )
            if reductions and not (
                set(reduction_shapes) & expected_reduction_shapes
            ):
                distributed_collective_violations.append(
                    "distributed q-a norm all-reduce payload drifted: "
                    f"{reduction_shapes}"
                )
            if gathers and not (set(gather_shapes) & expected_gather_shapes):
                distributed_collective_violations.append(
                    "distributed q-a norm all-gather payload drifted: "
                    f"expected={sorted(expected_gather_shapes)} "
                    f"found={gather_shapes}"
                )
            reduction_operand_names = [
                name for item in reductions for name in item.operand_names
            ]
            named_multiply_reduce_operand = any(
                "multiply_reduce_fusion" in name
                for name in reduction_operand_names
            )
            distributed_collective_contract = {
                "collective_count": len(collectives),
                "collective_counts": dict(sorted(counts.items())),
                "collectives": [item.to_dict() for item in collectives],
                "expected_replica_groups": [list(expected_groups[0])],
                "expected_all_reduce_shapes": [
                    {"dtype": dtype, "dimensions": list(dimensions)}
                    for dtype, dimensions in sorted(expected_reduction_shapes)
                ],
                "expected_all_gather_shapes": [
                    {"dtype": dtype, "dimensions": list(dimensions)}
                    for dtype, dimensions in sorted(expected_gather_shapes)
                ],
                "cpu_bf16_collective_promotion_allowed": (
                    allow_cpu_bf16_collective_promotion
                ),
                "all_reduce_operand_names": reduction_operand_names,
                "named_multiply_reduce_operand": (
                    named_multiply_reduce_operand
                ),
            }
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
    violations.extend(distributed_collective_violations)
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
        "distributed_collective_contract": distributed_collective_contract,
        "distributed_collective_violations": (
            distributed_collective_violations
        ),
        "forbidden_dead_rows": forbidden_dead_rows,
        "diagnostic_batch32_allowed": phase != "one_row_score",
        "passed": not violations,
        "violations": violations,
    }
