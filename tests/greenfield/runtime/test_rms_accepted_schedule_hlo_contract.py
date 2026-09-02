"""Hostile tests of the accepted-schedule RMS HLO contract on real TPU bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from glm_tpu.greenfield.runtime.decoder import (
    _validate_rms_accepted_schedule_hlo,
    _validate_rms_accepted_schedule_stablehlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

RUN = Path("/home/gianl/gate-d-runs/gate_d_layer1_rms_schedule_20260902T061305905714981Z/hlo")
SCHEDULE = RUN / "layer1_rms_schedule_schedule.optimized_hlo.txt"
CONTROL = RUN / "layer1_rms_schedule_control.optimized_hlo.txt"
SCHEDULE_STABLE = RUN / "layer1_rms_schedule_schedule.stablehlo.mlir"
CONTROL_STABLE = RUN / "layer1_rms_schedule_control.stablehlo.mlir"
pytestmark = pytest.mark.skipif(not SCHEDULE.exists(), reason="archived TPU HLO unavailable")


CENSUS = dict(expected_accepted_count=1, expected_sharded_qa_count=0)


def _contract(text: str, enabled: bool) -> dict:
    return _validate_rms_accepted_schedule_hlo(parse_hlo_module(text), enabled=enabled, **CENSUS)


def test_real_schedule_arm_passes_and_control_arm_fails_when_enabled() -> None:
    schedule = _contract(SCHEDULE.read_text(), True)
    assert schedule["passed"], schedule["violations"]
    assert schedule["conforming_rsqrt_count"] == 1 and schedule["tiled_module"] is True
    control = _contract(CONTROL.read_text(), True)
    assert not control["passed"]
    assert control["nonconforming_rsqrt_count"] == 1
    # Default contract: the control arm is clean, the schedule arm is refused.
    assert _contract(CONTROL.read_text(), False)["passed"]
    assert not _contract(SCHEDULE.read_text(), False)["passed"]


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("mixed_shape_rsqrt", "ROOT %rsqrt.5 = f32[32]{0:T(128)S(3)} rsqrt(", "ROOT %rsqrt.5 = f32[16]{0:T(128)S(3)} rsqrt("),
        ("row_40_reduce", "f32[32,6144]", "f32[40,6144]"),
        ("row_40_scale", "f32[32]{0:T(128)S(3)} rsqrt(", "f32[40]{0:T(128)S(3)} rsqrt("),
        ("wrong_axis", "dimensions={1}, to_apply=%region_0.1", "dimensions={0}, to_apply=%region_0.1"),
        ("wrong_epsilon", "constant(1e-05)", "constant(1e-06)"),
        ("not_a_square", "%square.4 = f32[32,6144]{1,0:T(8,128)} multiply(%param_0.325, %param_0.325)", "%square.4 = f32[32,6144]{1,0:T(8,128)} multiply(%param_0.325, %param_0.325.clone)"),
        ("wrong_layout", "%square.4 = f32[32,6144]{1,0:T(8,128)} multiply(", "%square.4 = f32[32,6144]{1,0:T(1,128)} multiply("),
        ("fused_producer", "%square.4 = f32[32,6144]{1,0:T(8,128)} multiply(%param_0.325, %param_0.325)", "%carry.9 = f32[32,6144]{1,0:T(8,128)} add(%param_0.325, %param_0.325)\n  %square.4 = f32[32,6144]{1,0:T(8,128)} multiply(%carry.9, %carry.9)"),
    ),
)
def test_hostile_mutations_of_the_real_schedule_arm_are_refused(label: str, old: str, new: str) -> None:
    text = SCHEDULE.read_text()
    assert old in text, label
    mutated = text.replace(old, new) if label == "row_40_reduce" else text.replace(old, new, 1)
    if label == "not_a_square":
        mutated = mutated.replace(
            "%param_0.325 = f32[32,6144]{1,0:T(8,128)S(3)} parameter(0)",
            "%param_0.325 = f32[32,6144]{1,0:T(8,128)S(3)} parameter(0)\n  %param_0.325.clone = f32[32,6144]{1,0:T(8,128)} copy(%param_0.325)",
            1,
        )
    result = _contract(mutated, True)
    assert not result["passed"], (label, result)


def test_dummy_32_row_rsqrt_does_not_excuse_an_unscheduled_rms() -> None:
    control = CONTROL.read_text()
    schedule = SCHEDULE.read_text()
    # A module carrying the control arm's own scalar/unscheduled rsqrt plus an
    # unrelated 32-row rsqrt with no reduce lineage must still be refused.
    dummy = control.replace(
        "ROOT %rsqrt.5 = f32[32]{0:T(128)S(3)} rsqrt(%add.742)",
        "%dummy_param = f32[32]{0:T(128)S(3)} parameter(1)\n  %dummy = f32[32]{0:T(128)S(3)} rsqrt(%dummy_param)\n  ROOT %rsqrt.5 = f32[32]{0:T(128)S(3)} rsqrt(%add.742)",
        1,
    )
    assert dummy != control
    result = _contract(dummy, True)
    assert not result["passed"]
    assert result["nonconforming_rsqrt_count"] >= 1
    # A lineage-less 32-row rsqrt alone is refused too.
    orphan = schedule.replace("rsqrt(%add.736)", "rsqrt(%param_0.323)", 1)
    assert orphan != schedule
    assert not _contract(orphan, True)["passed"]


def test_stablehlo_binding_requires_barrier_carried_32_row_scales() -> None:
    schedule = _validate_rms_accepted_schedule_stablehlo(SCHEDULE_STABLE.read_text(), enabled=True, **CENSUS)
    assert schedule["passed"], schedule
    assert schedule["conforming_rsqrt_count"] == 1 and schedule["barrier_count"] == 1
    # The control arm reduces the same 32 rows but squares an unbarriered add.
    control = _validate_rms_accepted_schedule_stablehlo(CONTROL_STABLE.read_text(), enabled=True, **CENSUS)
    assert not control["passed"]
    assert control["barrier_count"] == 0 and control["nonconforming_rsqrt_count"] == 1
    assert _validate_rms_accepted_schedule_stablehlo(CONTROL_STABLE.read_text(), enabled=False)["passed"]
    assert not _validate_rms_accepted_schedule_stablehlo(SCHEDULE_STABLE.read_text(), enabled=False)["passed"]


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("barrier_stripped", "stablehlo.optimization_barrier %528 : tensor<32x6144xf32>", "stablehlo.custom_barrier %528 : tensor<32x6144xf32>"),
        ("wrong_axis", "applies stablehlo.add across dimensions = [1] : (tensor<32x6144xf32>", "applies stablehlo.add across dimensions = [0] : (tensor<32x6144xf32>"),
        ("wrong_epsilon", "dense<9.99999974E-6>", "dense<9.99999974E-7>"),
        ("wrong_width", "dense<6.144000e+03>", "dense<6.143000e+03>"),
        ("mixed_shape", "stablehlo.rsqrt %536 : tensor<32x1xf32>", "stablehlo.rsqrt %536 : tensor<16x1xf32>"),
        ("max_combiner", "applies stablehlo.add across dimensions = [1] : (tensor<32x6144xf32>", "applies stablehlo.maximum across dimensions = [1] : (tensor<32x6144xf32>"),
    ),
)
def test_hostile_stablehlo_mutations_are_refused(label: str, old: str, new: str) -> None:
    text = SCHEDULE_STABLE.read_text()
    assert old in text, label
    result = _validate_rms_accepted_schedule_stablehlo(text.replace(old, new, 1), enabled=True, **CENSUS)
    assert not result["passed"], (label, result)
