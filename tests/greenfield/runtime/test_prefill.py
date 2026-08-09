from __future__ import annotations

from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime import validate_teacher_forced_prefill_loops
from glm_tpu.greenfield.runtime.prefill import (
    PrefillBackendContract,
    _prefill_index_repair_chunk_count,
    _validate_physical_m64_prefill_index_repair_hlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


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


def test_prefill_loop_contract_classifies_physical_m64_repair() -> None:
    repair = "jit(execute)/shard_map/cond/branch_0_fun/while"
    contract = validate_teacher_forced_prefill_loops(
        _loop_hlo("jit(execute)/while", repair),
        expected_fused_qkv_internal_loops=0,
        expected_prefill_index_repair_loops=1,
    )

    assert contract["passed"]
    assert contract["loop_count"] == 2
    assert contract["expected_total_loop_count"] == 2
    assert contract["prefill_index_repair_loop_count"] == 1
    assert contract["unclassified_loops"] == []


def test_protected_8k_prefill_repair_has_four_chunks_per_layer() -> None:
    assert _prefill_index_repair_chunk_count(8155) == 4
    assert 21 * _prefill_index_repair_chunk_count(8155) == 84


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
                "jit(execute)/while/body/shard_map/cond/"
                "branch_0_fun/while",
            ),
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


def _repair_hlo() -> str:
    scope = "jit(execute)/shard_map/cond/branch_0_fun"
    return f'''HloModule repair, num_partitions=32

ENTRY %main {{
  %history = bf16[2048,1,6144] parameter(0)
  %lhs = bf16[64,6144] parameter(1)
  %wk = f32[128,6144] parameter(2)
  %cache = bf16[16,128] parameter(3)
  %slots = s32[2048] parameter(4)
  %projection = f32[64,128] convolution(%lhs, %wk), dim_labels=bf_oi->bf, metadata={{op_name="{scope}/while/body/dot_general"}}
  %sqrt.0 = f32[64] sqrt(%projection), metadata={{op_name="{scope}/while/body/sqrt"}}
  %sqrt.1 = f32[64] fusion(%sqrt.0), metadata={{op_name="{scope}/while/body/sqrt"}}
  %affine = f32[64,128] add(%projection, %projection), metadata={{op_name="{scope}/while/body/add"}}
  ROOT %written = bf16[16,128] scatter(%cache, %slots, %affine), metadata={{op_name="{scope}/scatter"}}
}}
'''


def _repair_contract(
    hlo: str,
    *,
    backend_contract: PrefillBackendContract = (
        "tpu_v4_pp8_pallas_feature_linear"
    ),
) -> dict[str, object]:
    config = SimpleNamespace(
        hidden_size=6144,
        index_key_width=128,
        maximum_full_indexer_slots=1,
        total_devices=32,
    )
    program = SimpleNamespace(
        decoder=SimpleNamespace(config=config),
        prompt_length=2048,
    )
    schedule = SimpleNamespace(
        stages=(
            SimpleNamespace(
                layers=(SimpleNamespace(indexer_kind="full"),)
            ),
        )
    )
    return _validate_physical_m64_prefill_index_repair_hlo(
        parse_hlo_module(hlo),
        program=program,
        schedule=schedule,
        backend_contract=backend_contract,
    )


def test_physical_m64_prefill_repair_hlo_is_exact_and_fail_closed() -> None:
    accepted = _repair_contract(_repair_hlo())
    assert accepted["passed"] is True
    assert accepted["projection_count"] == 1
    assert accepted["exact_projection_operand_count"] == 1
    assert accepted["physical_sqrt_count"] == 2
    assert accepted["physical_affine_count"] == 1
    assert accepted["cache_write_count"] == 1
    assert accepted["grouped_sqrt_count"] == 0
    assert accepted["repair_collectives"] == []

    bf16_rhs = _repair_contract(
        _repair_hlo().replace(
            "%wk = f32[128,6144] parameter(2)",
            "%wk = bf16[128,6144] parameter(2)",
        )
    )
    assert bf16_rhs["passed"] is False
    assert "BF16-M64/FP32-wk operands" in bf16_rhs["violations"][0]

    grouped = _repair_contract(
        _repair_hlo().replace(
            "%sqrt.0 = f32[64]",
            "%sqrt.0 = f32[32,64]",
        )
    )
    assert grouped["passed"] is False
    assert grouped["grouped_sqrt_count"] == 1
    assert any("grouped [32,64]" in item for item in grouped["violations"])

    collective = _repair_contract(
        _repair_hlo().replace(
            "ROOT %written = bf16[16,128] scatter",
            "ROOT %written = bf16[16,128] all-reduce",
        )
    )
    assert collective["passed"] is False
    assert collective["repair_collectives"]

    full_pod_history = _repair_contract(
        _repair_hlo().replace(
            "%history = bf16[2048,1,6144] parameter(0)",
            "%history = bf16[2048,32,6144] parameter(0)",
        )
    )
    assert full_pod_history["passed"] is False
    assert full_pod_history["full_pod_history_shapes"]
    assert any(
        "full-pod prompt history" in item
        for item in full_pod_history["violations"]
    )

    cpu_lhs = _repair_hlo().replace(
        "%lhs = bf16[64,6144] parameter(1)",
        "%lhs = f32[64,6144] parameter(1)",
    )
    cpu = _repair_contract(
        cpu_lhs,
        backend_contract="cpu_reference",
    )
    assert cpu["passed"] is True
    assert cpu["exact_projection_operand_count"] == 1
