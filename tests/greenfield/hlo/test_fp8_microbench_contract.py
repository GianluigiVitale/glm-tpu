from __future__ import annotations

from pathlib import Path

import pytest

from scripts.greenfield.microbench_fp8_matmul import (
    _is_bounded_compact_input_scatter,
    _is_bounded_route_restore_index_call,
    _reference_overlay_contract,
    _validate_sampling_contract,
)


REPO = Path(__file__).resolve().parents[3]


def test_route_restore_index_call_requires_exact_bounded_contract() -> None:
    accepted = (
        "%custom-call.1 = s32[8,2]{1,0:T(8,128)} "
        "custom-call(%indices), "
        'custom_call_target="GatherScatterIndicesBitpacked", '
        'metadata={op_name="jit(fp8_selected_up_gate)/concatenate"}'
    )
    assert _is_bounded_route_restore_index_call(accepted, route_count=8)

    rejected = (
        accepted.replace("s32[8,2]", "s32[8,3]"),
        accepted.replace("s32[8,2]", "s32[32,2]"),
        accepted.replace("GatherScatterIndicesBitpacked", "tpu_custom_call"),
        accepted.replace(
            "jit(fp8_selected_up_gate)/concatenate",
            "jit(fp8_selected_up_gate)/weight_gather",
        ),
    )
    assert not any(
        _is_bounded_route_restore_index_call(line, route_count=8)
        for line in rejected
    )


def test_route_restore_index_call_binds_selected_down_kernel_name() -> None:
    down = (
        "%custom-call.1 = s32[8,2]{1,0:T(8,128)} "
        "custom-call(%indices), "
        'custom_call_target="GatherScatterIndicesBitpacked", '
        'metadata={op_name="jit(fp8_selected_swiglu_down)/concatenate"}'
    )
    assert _is_bounded_route_restore_index_call(
        down,
        route_count=8,
        kernel_name="fp8_selected_swiglu_down",
    )
    assert not _is_bounded_route_restore_index_call(down, route_count=8)


def test_compact_input_scatter_requires_exact_selected_down_contract() -> None:
    accepted = (
        "%scatter.4 = bf16[8,8,2048]{2,1,0} "
        "scatter(%zeros, %custom-call.1, %gate), "
        "update_window_dims={1}, inserted_window_dims={0,1}, "
        "scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, "
        'metadata={op_name="jit(fp8_selected_swiglu_down)/scatter"}'
    )
    kwargs = {
        "index_result": "%custom-call.1",
        "route_count": 8,
        "row_tile": 8,
        "width": 2048,
        "kernel_name": "fp8_selected_swiglu_down",
    }
    assert _is_bounded_compact_input_scatter(accepted, **kwargs)
    assert not _is_bounded_compact_input_scatter(
        accepted.replace("bf16[8,8,2048]", "bf16[32,8,2048]"),
        **kwargs,
    )
    assert not _is_bounded_compact_input_scatter(
        accepted.replace("scatter_dims_to_operand_dims={0,1}", ""),
        **kwargs,
    )


def test_reference_diagnostic_is_small_and_shape_bounded() -> None:
    _validate_sampling_contract(
        kernel="single_up_m1",
        diagnostic_reference_timing=True,
        warmup=1,
        iterations=3,
    )
    with pytest.raises(ValueError, match="M=1 projection"):
        _validate_sampling_contract(
            kernel="single_up",
            diagnostic_reference_timing=True,
            warmup=1,
            iterations=3,
        )
    with pytest.raises(ValueError, match="warmup>=1"):
        _validate_sampling_contract(
            kernel="single_up_m1",
            diagnostic_reference_timing=True,
            warmup=0,
            iterations=3,
        )


def test_reference_diagnostic_requires_full_weight_overlay() -> None:
    accepted = _reference_overlay_contract(
        "ROOT %x = bf16[2048,6144] copy(%p)",
        contraction=6144,
        output_width=2048,
    )
    rejected = _reference_overlay_contract(
        "ROOT %x = bf16[8,2048] copy(%p)",
        contraction=6144,
        output_width=2048,
    )

    assert accepted["passed"] is True
    assert rejected["passed"] is False
