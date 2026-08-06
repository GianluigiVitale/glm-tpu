from __future__ import annotations

import pytest

from glm_tpu.greenfield.benchmarking.topk import validate_dsa_topk_hlo


def _local_hlo() -> str:
    names = [
        "greenfield_dsa_topk_local_select_n65536_k2048_b2048_g32",
        "greenfield_dsa_topk_local_merge_l0_g32_k2048",
        "greenfield_dsa_topk_local_merge_l1_g16_k2048",
        "greenfield_dsa_topk_local_merge_l2_g8_k2048",
        "greenfield_dsa_topk_local_merge_l3_g4_k2048",
        "greenfield_dsa_topk_local_merge_l4_g2_k2048",
    ]
    calls = "\n".join(
        f'%{name} = (f32[1,2048], s32[1,2048]) custom-call(%x), '
        f'custom_call_target="tpu_custom_call", metadata={{op_name="{name}"}}'
        for name in names
    )
    return f"""
HloModule local
%x = f32[1,65536] parameter(0)
%p = s32[65536] parameter(1)
%out = f32[1,2048] copy(%x)
%positions = s32[1,2048] copy(%p)
{calls}
"""


def _merge_hlo() -> str:
    names = [
        "greenfield_dsa_topk_global_merge_l0_g4_k2048",
        "greenfield_dsa_topk_global_merge_l1_g2_k2048",
    ]
    calls = "\n".join(
        f'%{name} = (f32[1,2048], s32[1,2048]) custom-call(%x), '
        f'custom_call_target="tpu_custom_call", metadata={{op_name="{name}"}}'
        for name in names
    )
    return f"""
HloModule merge
%x = f32[4,1,2048] parameter(0)
%p = s32[4,1,2048] parameter(1)
%positions = s32[1,2048] copy(%p)
{calls}
"""


def test_validate_topk_hlo_accepts_exact_local_tree() -> None:
    record = validate_dsa_topk_hlo(_local_hlo(), phase="local")
    assert record["passed"]
    assert record["expected_kernel_count"] == 6
    assert record["custom_call_count"] == 6


def test_validate_topk_hlo_accepts_exact_four_owner_merge() -> None:
    record = validate_dsa_topk_hlo(_merge_hlo(), phase="merge")
    assert record["passed"]
    assert record["expected_kernel_count"] == 2


@pytest.mark.parametrize(
    "drift",
    [
        " sort(",
        " topk(",
        " all-gather(",
        "f32[32,65536]",
    ],
)
def test_validate_topk_hlo_rejects_sort_collective_and_dead_rows(
    drift: str,
) -> None:
    record = validate_dsa_topk_hlo(_local_hlo() + "\n" + drift, phase="local")
    assert not record["passed"]


def test_validate_topk_hlo_rejects_missing_and_unexpected_calls() -> None:
    missing = validate_dsa_topk_hlo(
        _local_hlo().replace(
            "greenfield_dsa_topk_local_merge_l4_g2_k2048", "wrong"
        ),
        phase="local",
    )
    assert not missing["passed"]
    extra = validate_dsa_topk_hlo(
        _merge_hlo()
        + '\n%x = f32[1] custom-call(), custom_call_target="mystery"\n',
        phase="merge",
    )
    assert not extra["passed"]
    assert extra["unexpected_custom_calls"]


def test_validate_topk_hlo_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_dsa_topk_hlo(_local_hlo(), phase="local", top_k=0)
    with pytest.raises(ValueError, match="unsupported"):
        validate_dsa_topk_hlo(_local_hlo(), phase="other")  # type: ignore[arg-type]
