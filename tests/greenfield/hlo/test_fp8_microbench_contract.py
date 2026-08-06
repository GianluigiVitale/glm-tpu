from __future__ import annotations

from scripts.greenfield.microbench_fp8_matmul import (
    _is_bounded_route_restore_index_call,
)


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
