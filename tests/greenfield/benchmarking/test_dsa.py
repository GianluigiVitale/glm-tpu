from __future__ import annotations

import pytest

from glm_tpu.greenfield.benchmarking.dsa import validate_dsa_score_hlo


def _valid_hlo() -> str:
    return """
HloModule jit_dsa, num_partitions=1
ENTRY %main (
  %query: f32[1,32,128],
  %keys: bf16[65536,128],
  %weights: f32[1,32,128]
) -> f32[1,65536] {
  ROOT %greenfield_dsa_score_r1_h32_d128_s65536 = f32[1,65536] custom-call(%query, %keys, %weights), custom_call_target="tpu_custom_call", operand_layout_constraints={f32[1,32,128], bf16[65536,128], f32[1,32,128]}, metadata={op_name="jit(dsa_scores_pallas)/greenfield_dsa_score_r1_h32_d128_s65536/pallas_call"}
}
"""


def test_validate_dsa_score_hlo_accepts_one_compact_kernel() -> None:
    record = validate_dsa_score_hlo(_valid_hlo())
    assert record["passed"]
    assert record["kernel_custom_call_count"] == 1
    assert not record["forbidden_per_head_overlays"]


@pytest.mark.parametrize(
    "drift",
    [
        "f32[1,32,65536]",
        "f32[32,65536]",
        " all-reduce(",
        "f32[32,32,128]",
    ],
)
def test_validate_dsa_score_hlo_rejects_overlay_collective_and_dead_rows(
    drift: str,
) -> None:
    record = validate_dsa_score_hlo(_valid_hlo() + "\n" + drift)
    assert not record["passed"]


def test_validate_dsa_score_hlo_rejects_call_count_and_shape_drift() -> None:
    missing = validate_dsa_score_hlo(
        _valid_hlo().replace("greenfield_dsa_score_r1_h32_d128_s65536", "wrong")
    )
    assert not missing["passed"]

    extra = validate_dsa_score_hlo(
        _valid_hlo()
        + '\n%x = f32[1] custom-call(), custom_call_target="mystery"\n'
    )
    assert not extra["passed"]
    assert extra["unexpected_custom_calls"]


def test_validate_dsa_score_hlo_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        validate_dsa_score_hlo(_valid_hlo(), local_context=0)
