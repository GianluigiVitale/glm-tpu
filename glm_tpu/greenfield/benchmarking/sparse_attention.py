"""Fail-closed HLO checks for fused selected-KV sparse MLA."""

from __future__ import annotations

from typing import Any


def validate_sparse_attention_hlo(
    optimized_hlo: str,
    *,
    heads: int = 64,
    latent: int = 512,
    rope_width: int = 64,
    top_k: int = 2048,
    segment_block: int = 128,
    dma_rows: int = 8,
    cache_rows: int = 65_536,
    cache_width: int = 640,
    dtype: str = "bf16",
    expected_metadata_gather_count: int = 1,
) -> dict[str, Any]:
    """Require the two intended kernels and reject selected-KV HBM materialization."""

    for name, value in (
        ("heads", heads),
        ("latent", latent),
        ("rope_width", rope_width),
        ("top_k", top_k),
        ("segment_block", segment_block),
        ("dma_rows", dma_rows),
        ("cache_rows", cache_rows),
        ("cache_width", cache_width),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(expected_metadata_gather_count, int)
        or isinstance(expected_metadata_gather_count, bool)
        or expected_metadata_gather_count < 0
    ):
        raise ValueError("expected metadata-gather count must be a nonnegative integer")
    if dtype not in ("bf16", "f32"):
        raise ValueError("sparse-attention HLO dtype must be 'bf16' or 'f32'")

    expected_names = (
        f"greenfield_owner_position_order_k{top_k}",
        "greenfield_fused_selected_kv_sparse_mla_"
        f"h{heads}_k{top_k}_b{segment_block}_w{cache_width}_d{dma_rows}",
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
    metadata_gather_calls = [
        line
        for line in custom_calls
        if 'custom_call_target="AssumeGatherIndicesInBound"' in line
        and f"s32[{top_k}]" in line
        and "take_along_axis)/gather" in line
    ]
    expected_lines = {
        line for lines in kernel_calls.values() for line in lines
    } | set(metadata_gather_calls)
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
    forbidden_selected_kv = [
        shape
        for shape in (
            f"bf16[1,{top_k},{cache_width}]",
            f"bf16[{top_k},{cache_width}]",
            f"f32[1,{top_k},{cache_width}]",
            f"f32[{top_k},{cache_width}]",
            f"bf16[1,{top_k * dma_rows},{cache_width}]",
            f"bf16[{top_k * dma_rows},{cache_width}]",
            f"f32[1,{top_k * dma_rows},{cache_width}]",
            f"f32[{top_k * dma_rows},{cache_width}]",
        )
        if shape in optimized_hlo
    ]
    forbidden_dead_rows = [
        shape
        for shape in (
            f"s32[32,{top_k}]",
            f"{dtype}[32,{heads},{latent}]",
            f"{dtype}[32,{top_k},{cache_width}]",
        )
        if shape in optimized_hlo
    ]
    required_shapes = (
        f"s32[1,{top_k}]",
        f"{dtype}[1,{heads},{latent}]",
        f"{dtype}[1,{heads},{rope_width}]",
        f"{dtype}[{cache_rows},{cache_width}]",
        f"f32[1,{heads}]",
    )
    missing_shapes = [
        shape for shape in required_shapes if shape not in optimized_hlo
    ]

    violations: list[str] = []
    for name, lines in kernel_calls.items():
        if len(lines) != 1:
            violations.append(f"expected one {name} call, found {len(lines)}")
    if len(metadata_gather_calls) != expected_metadata_gather_count:
        violations.append(
            "expected compact block-table metadata gather count "
            f"{expected_metadata_gather_count}, found {len(metadata_gather_calls)}"
        )
    if unexpected_custom_calls:
        violations.append(
            "unexpected sparse-attention custom calls: "
            f"{unexpected_custom_calls}"
        )
    if forbidden_operations:
        violations.append(
            "sparse-attention contains forbidden operations: "
            f"{forbidden_operations}"
        )
    if forbidden_selected_kv:
        violations.append(
            "selected KV was materialized outside the fused kernel: "
            f"{forbidden_selected_kv}"
        )
    if forbidden_dead_rows:
        violations.append(
            f"sparse-attention contains batch-32 dead rows: {forbidden_dead_rows}"
        )
    if missing_shapes:
        violations.append(
            f"sparse-attention lacks required production shapes: {missing_shapes}"
        )
    return {
        "expected_kernel_names": list(expected_names),
        "kernel_custom_call_counts": {
            name: len(lines) for name, lines in kernel_calls.items()
        },
        "custom_call_count": len(custom_calls),
        "metadata_gather_custom_call_count": len(metadata_gather_calls),
        "expected_metadata_gather_custom_call_count": (
            expected_metadata_gather_count
        ),
        "metadata_gather_custom_calls": metadata_gather_calls,
        "unexpected_custom_calls": unexpected_custom_calls,
        "forbidden_operations": forbidden_operations,
        "forbidden_selected_kv_materializations": forbidden_selected_kv,
        "forbidden_dead_rows": forbidden_dead_rows,
        "missing_required_shapes": missing_shapes,
        "passed": not violations,
        "violations": violations,
    }


def validate_sparse_attention_integration_hlo(
    optimized_hlo: str,
    *,
    heads: int = 64,
    top_k: int = 2048,
    segment_block: int = 128,
    cache_width: int = 640,
    dma_rows: int = 8,
) -> dict[str, Any]:
    """Prove that an enclosing layer executable retained both Pallas calls."""

    for name, value in (
        ("heads", heads),
        ("top_k", top_k),
        ("segment_block", segment_block),
        ("cache_width", cache_width),
        ("dma_rows", dma_rows),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    names = (
        f"greenfield_owner_position_order_k{top_k}",
        "greenfield_fused_selected_kv_sparse_mla_"
        f"h{heads}_k{top_k}_b{segment_block}_w{cache_width}_d{dma_rows}",
    )
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
    ]
    counts = {
        name: sum(
            name in line and 'custom_call_target="tpu_custom_call"' in line
            for line in custom_calls
        )
        for name in names
    }
    forbidden_dead_rows = [
        shape
        for shape in (
            f"s32[32,{top_k}]",
            f"bf16[32,{top_k},{cache_width}]",
            f"f32[32,{top_k},{cache_width}]",
        )
        if shape in optimized_hlo
    ]
    violations = [
        f"expected one integrated {name} call, found {count}"
        for name, count in counts.items()
        if count != 1
    ]
    if forbidden_dead_rows:
        violations.append(
            f"integrated sparse attention contains dead rows: {forbidden_dead_rows}"
        )
    return {
        "expected_kernel_names": list(names),
        "kernel_custom_call_counts": counts,
        "forbidden_dead_rows": forbidden_dead_rows,
        "passed": not violations,
        "violations": violations,
    }
