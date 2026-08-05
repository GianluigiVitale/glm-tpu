from __future__ import annotations

from dataclasses import replace

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.model import (
    build_decoder_runtime_weight_layout,
    build_decoder_state_layout,
    build_pipeline_schedule,
)
from tests.greenfield.model.test_schedule import _plan


def test_production_runtime_weights_reconcile_protected_gate_b_files() -> None:
    plan = _plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)

    assert len(layout.specs) == 364
    assert len(layout.devices) == 32
    assert layout.runtime_bytes_per_chip == 26_068_042_432
    assert layout.source_leaf_count == 122_640
    assert layout.maximum_padding_bytes_per_chip == 3_719_717_632
    assert layout.minimum_padding_bytes_per_chip == 1_237_233_344
    assert layout.layout_hash == (
        "e239573af9e373874f424c3d80b0ab8f6f9cbcdd2810935a3d768120cdb9663f"
    )

    expected_stage_bytes = (
        23_163_143_616,
        24_830_809_088,
        22_351_307_904,
        22_348_324_800,
        24_830_809_088,
        22_351_307_904,
        24_830_809_088,
        22_824_128_448,
    )
    expected_stage_leaves = (3750, 4074, 3675, 3668, 4074, 3675, 4074, 3670)
    for stage in range(8):
        devices = layout.devices[stage * 4 : stage * 4 + 4]
        assert {device.source_bytes for device in devices} == {
            expected_stage_bytes[stage]
        }
        assert {device.source_leaf_count for device in devices} == {
            expected_stage_leaves[stage]
        }
        assert all(
            device.source_bytes + device.padding_bytes
            == layout.runtime_bytes_per_chip
            for device in devices
        )


def test_runtime_slot_bindings_preserve_exact_expert_identity_and_global_owners() -> None:
    plan = _plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    stage0_slot2 = layout.devices[2]
    by_name = {tensor.spec.name: tensor for tensor in stage0_slot2.tensors}

    experts = by_name["sparse.slot_00.experts.gate_proj.weight_bits"]
    assert experts.spec.shape == (64, 2048, 6144)
    assert experts.sources[0].name == (
        "model.layers.3.mlp.experts.128.gate_proj.weight"
    )
    assert experts.sources[-1].name == (
        "model.layers.3.mlp.experts.191.gate_proj.weight"
    )
    expert_scales = by_name["sparse.slot_00.experts.gate_proj.scale_inv"]
    assert expert_scales.spec.shape == (64, 16, 48)
    assert expert_scales.sources[0].shape == (16, 48)
    assert by_name["sparse.slot_09.router_weight"].is_padding
    assert not by_name["global.embedding"].is_padding
    assert by_name["global.final_norm"].is_padding
    assert by_name["global.lm_head"].is_padding

    stage1_slot2 = layout.devices[6]
    stage1 = {tensor.spec.name: tensor for tensor in stage1_slot2.tensors}
    assert not stage1["sparse.slot_09.router_weight"].is_padding
    assert stage1["global.embedding"].is_padding
    stage7_slot2 = layout.devices[30]
    stage7 = {tensor.spec.name: tensor for tensor in stage7_slot2.tensors}
    assert stage7["global.embedding"].is_padding
    assert not stage7["global.final_norm"].is_padding
    assert not stage7["global.lm_head"].is_padding


def test_uniform_weights_and_target_state_leave_explicit_precompile_headroom() -> None:
    plan = _plan()
    schedule = build_pipeline_schedule(plan)
    weights = build_decoder_runtime_weight_layout(plan, schedule)
    # Avoid the fixture plan's placeholder target-memory fields while checking
    # the independently derived maximum padded state geometry.
    short_state = build_decoder_state_layout(
        plan, schedule, context_capacity=8192
    )
    target_state_bytes = (
        short_state.maximum_padded_bytes_per_chip
        * (plan.target_context_length // 8192)
        - (short_state.stages[0].selected_index_bytes
           + short_state.stages[0].transport_bytes)
        * ((plan.target_context_length // 8192) - 1)
    )
    assert target_state_bytes == 1_090_555_904
    measured_one_layer_temporary_floor = 779_642_880
    hbm_limit = 33_014_413_312
    precompile_accounted = (
        weights.runtime_bytes_per_chip
        + target_state_bytes
        + measured_one_layer_temporary_floor
    )
    assert precompile_accounted == 27_938_241_216
    assert hbm_limit - precompile_accounted == 5_076_172_096


def test_runtime_weights_refuse_a_foreign_schedule() -> None:
    plan = _plan()
    schedule = replace(build_pipeline_schedule(plan), plan_hash="f" * 64)
    with pytest.raises(PlanValidationError, match="another plan"):
        build_decoder_runtime_weight_layout(plan, schedule)
