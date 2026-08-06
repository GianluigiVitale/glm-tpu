from __future__ import annotations

import numpy as np

from glm_tpu.greenfield.benchmarking.one_layer import (
    TensorTolerance,
    compare_bounded_tensor,
    validate_pallas_real_layer_hlo,
    validate_real_layer_hlo,
)


GOOD_HLO = """
HloModule real_layer, replica_count=1, num_partitions=4

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT z = bf16[] add(x, y)
}

ENTRY main {
  input = bf16[2,1,6144]{2,1,0} parameter(0)
  ROOT combine = bf16[2,1,6144]{2,1,0} all-reduce(input), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
}
"""


def test_bounded_tensor_records_all_three_bars() -> None:
    reference = np.zeros((1, 100), dtype=np.float32)
    observed = reference.copy()
    observed[0, -1] = 0.125
    record = compare_bounded_tensor(
        observed,
        reference,
        TensorTolerance(max_abs=0.125, p99_abs=0.01, mean_abs=0.002),
    )
    assert record["passed"]
    assert record["error"]["max_abs"] == 0.125
    assert record["error"]["max_abs_index"] == [0, 99]
    failed = compare_bounded_tensor(
        observed,
        reference,
        TensorTolerance(max_abs=0.1, p99_abs=0.01, mean_abs=0.002),
    )
    assert not failed["passed"]


def test_real_layer_hlo_requires_one_exact_local_bf16_combine() -> None:
    record = validate_real_layer_hlo(GOOD_HLO)
    assert record["passed"], record
    assert record["collective_count"] == 1


def test_real_layer_hlo_accepts_exact_two_rank_pp16_combine() -> None:
    hlo = GOOD_HLO.replace(
        "num_partitions=4", "num_partitions=2"
    ).replace("{{0,1,2,3}}", "{{0,1}}")
    record = validate_real_layer_hlo(hlo, stage_size=2)
    assert record["passed"], record
    assert record["collectives"][0]["replica_groups"] == [[0, 1]]


def test_real_layer_hlo_rejects_promoted_or_extra_collectives() -> None:
    promoted = GOOD_HLO.replace("bf16[2,1,6144]", "f32[2,1,6144]")
    record = validate_real_layer_hlo(promoted)
    assert not record["passed"]
    assert any("payload" in item for item in record["violations"])

    extra = GOOD_HLO.replace(
        "ROOT combine =",
        "gather = bf16[2,1,6144]{2,1,0} all-gather(input), dimensions={0}, replica_groups={{0,1,2,3}}, use_global_device_ids=true\n  ROOT combine =",
    )
    record = validate_real_layer_hlo(extra)
    assert not record["passed"]
    assert any("exactly one" in item for item in record["violations"])


def test_real_layer_hlo_rejects_wrong_group_and_dead_rows() -> None:
    wrong_group = GOOD_HLO.replace("{{0,1,2,3}}", "{{0,1},{2,3}}")
    record = validate_real_layer_hlo(wrong_group)
    assert not record["passed"]
    assert any("replica group" in item for item in record["violations"])

    dead_rows = GOOD_HLO.replace(
        "input = bf16[2,1,6144]{2,1,0} parameter(0)",
        "dead = bf16[32,6144]{1,0} parameter(0)\n  input = bf16[2,1,6144]{2,1,0} parameter(1)",
    )
    record = validate_real_layer_hlo(dead_rows)
    assert not record["passed"]
    assert any("dead-row" in item for item in record["violations"])


def _pallas_hlo() -> str:
    calls = [
        "gate = (bf16[8,8,2048], bf16[8,8,2048]) custom-call(hidden, gate_bits, up_bits), custom_call_target=\"tpu_custom_call\", metadata={op_name=\"greenfield_fp8_selected_up_gate_r8_g64_k6144_n2048\"}, operand_layout_constraints={u8[64,6144,2048],u8[64,6144,2048]}",
        "down = bf16[8,8,6144] custom-call(gate, down_bits), custom_call_target=\"tpu_custom_call\", metadata={op_name=\"greenfield_fp8_selected_swiglu_down_r8_g64_k2048_n6144\"}, operand_layout_constraints={u8[64,2048,6144]}",
        "shared_gate = (bf16[1,512], bf16[1,512]) custom-call(hidden), custom_call_target=\"tpu_custom_call\", metadata={op_name=\"greenfield_fp8_block_up_gate_m8_k6144_n512\"}",
        "shared_down = bf16[1,6144] custom-call(shared_gate), custom_call_target=\"tpu_custom_call\", metadata={op_name=\"greenfield_fp8_block_matmul_m8_k512_n6144\"}",
    ]
    calls.extend(
        f"gather{i} = s32[1024] custom-call(index), custom_call_target=\"AssumeGatherIndicesInBound\""
        for i in range(6)
    )
    calls.extend(
        f"scatter{i} = s32[8,2] custom-call(index), custom_call_target=\"GatherScatterIndicesBitpacked\""
        for i in range(2)
    )
    return GOOD_HLO.replace(
        "  ROOT combine =",
        "  " + "\n  ".join(calls) + "\n  ROOT combine =",
    )


def test_pallas_real_layer_hlo_requires_exact_kernel_and_metadata_calls() -> None:
    record = validate_pallas_real_layer_hlo(_pallas_hlo())
    assert record["passed"], record
    assert record["kernel_custom_call_count"] == 4
    assert record["custom_call_count"] == 12

    drifted = _pallas_hlo().replace(
        'custom_call_target="AssumeGatherIndicesInBound"',
        'custom_call_target="unexpected_call"',
        1,
    )
    record = validate_pallas_real_layer_hlo(drifted)
    assert not record["passed"]
    assert any("unexpected" in item for item in record["violations"])


def test_pallas_real_layer_hlo_rejects_complete_decoded_overlay() -> None:
    hlo = _pallas_hlo().replace(
        "  ROOT combine =",
        "  overlay = bf16[64,6144,2048] parameter(9)\n  ROOT combine =",
    )
    record = validate_pallas_real_layer_hlo(hlo)
    assert not record["passed"]
    assert record["forbidden_decoded_overlays"]
