from __future__ import annotations

import pytest

from glm_tpu.greenfield.benchmarking.sparse_attention import (
    validate_sparse_attention_hlo,
)


def _valid_hlo() -> str:
    order = "greenfield_owner_position_order_k2048"
    attention = (
        "greenfield_fused_selected_kv_sparse_mla_"
        "h64_k2048_b128_w640"
    )
    return f'''HloModule sparse_attention
%positions = s32[1,2048] parameter(0)
%query = bf16[1,64,512] parameter(1)
%rope = bf16[1,64,64] parameter(2)
%cache = bf16[65536,640] parameter(3)
%lse = f32[1,64] constant(0)
%ordered = s32[1,2048] custom-call(%positions), custom_call_target="tpu_custom_call", metadata={{op_name="{order}"}}
%result = (bf16[1,64,512], f32[1,64,128]) custom-call(%query, %rope, %cache, %ordered), custom_call_target="tpu_custom_call", metadata={{op_name="{attention}"}}
'''


def test_validate_sparse_attention_hlo_accepts_two_fused_kernels() -> None:
    record = validate_sparse_attention_hlo(_valid_hlo())
    assert record["passed"]
    assert record["custom_call_count"] == 2
    assert not record["forbidden_selected_kv_materializations"]


@pytest.mark.parametrize(
    "drift",
    [
        " sort(",
        " all-gather(",
        "bf16[1,2048,640]",
        "f32[2048,640]",
        "s32[32,2048]",
    ],
)
def test_validate_sparse_attention_hlo_rejects_mechanism_drift(
    drift: str,
) -> None:
    record = validate_sparse_attention_hlo(_valid_hlo() + "\n" + drift)
    assert not record["passed"]


def test_validate_sparse_attention_hlo_rejects_missing_and_extra_calls() -> None:
    missing = validate_sparse_attention_hlo(
        _valid_hlo().replace("greenfield_owner_position_order_k2048", "wrong")
    )
    assert not missing["passed"]
    extra = validate_sparse_attention_hlo(
        _valid_hlo()
        + '\n%x = f32[1] custom-call(), custom_call_target="mystery"\n'
    )
    assert not extra["passed"]
    assert extra["unexpected_custom_calls"]


def test_validate_sparse_attention_hlo_rejects_shape_and_contract_drift() -> None:
    missing_shape = validate_sparse_attention_hlo(
        _valid_hlo().replace("bf16[65536,640]", "bf16[32768,640]")
    )
    assert not missing_shape["passed"]
    with pytest.raises(ValueError, match="positive integer"):
        validate_sparse_attention_hlo(_valid_hlo(), top_k=0)
    with pytest.raises(ValueError, match="dtype"):
        validate_sparse_attention_hlo(_valid_hlo(), dtype="f16")
