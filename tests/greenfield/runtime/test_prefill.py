from __future__ import annotations

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime import validate_teacher_forced_prefill_loops


def _loop_hlo(*op_names: str) -> str:
    instructions = "\n".join(
        f'  %loop.{index} = s32[] while(%seed), condition=%cond, body=%body, '
        f'metadata={{op_name="{op_name}"}}'
        for index, op_name in enumerate(op_names)
    )
    return (
        "HloModule prefill, num_partitions=32\n\n"
        "ENTRY %main {\n"
        "  %seed = s32[] parameter(0)\n"
        f"{instructions}\n"
        "}\n"
    )


def test_prefill_loop_contract_accepts_exact_outer_and_fused_qkv_loops() -> None:
    fused = (
        "jit(execute)/while/body/closed_call/shard_map/cond/branch_1_fun/"
        "one_row_fused_qkv_a_n82_convolution/while"
    )
    contract = validate_teacher_forced_prefill_loops(
        _loop_hlo("jit(execute)/while", fused, fused),
        expected_fused_qkv_internal_loops=2,
    )

    assert contract["passed"]
    assert contract["loop_count"] == 3
    assert contract["outer_loop_count"] == 1
    assert contract["fused_qkv_internal_loop_count"] == 2
    assert contract["expected_total_loop_count"] == 3
    assert contract["unclassified_loops"] == []


@pytest.mark.parametrize(
    ("op_names", "expected_fused", "violation"),
    (
        (
            ("jit(execute)/while",),
            1,
            "fused qkv-a internal loop count drifted",
        ),
        (
            ("jit(execute)/while", "jit(execute)/while/body/other/while"),
            0,
            "unclassified physical loops",
        ),
        (
            (
                "jit(execute)/while",
                "jit(other)/one_row_fused_qkv_a_n82_convolution/while",
            ),
            1,
            "unclassified physical loops",
        ),
        (
            ("jit(execute)/while/body/other/while",),
            0,
            "exactly one outer device loop",
        ),
    ),
)
def test_prefill_loop_contract_rejects_count_or_identity_drift(
    op_names: tuple[str, ...],
    expected_fused: int,
    violation: str,
) -> None:
    contract = validate_teacher_forced_prefill_loops(
        _loop_hlo(*op_names),
        expected_fused_qkv_internal_loops=expected_fused,
    )

    assert not contract["passed"]
    assert any(violation in item for item in contract["violations"])


@pytest.mark.parametrize("value", (-1, True, 1.5))
def test_prefill_loop_contract_rejects_invalid_expected_count(
    value: object,
) -> None:
    with pytest.raises(
        PlanValidationError,
        match="internal loop count must be non-negative",
    ):
        validate_teacher_forced_prefill_loops(
            _loop_hlo("jit(execute)/while"),
            expected_fused_qkv_internal_loops=value,  # type: ignore[arg-type]
        )
