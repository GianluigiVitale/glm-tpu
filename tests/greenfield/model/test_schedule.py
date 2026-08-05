from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.model import build_pipeline_schedule
from glm_tpu.greenfield.types import (
    ExecutionPlan,
    ModelGeometry,
    PhysicalDevice,
    PhysicalTopology,
    PlanName,
    StageAssignment,
)


REPO = Path(__file__).resolve().parents[3]
PP8_RANGES = ((0, 12), (12, 22), (22, 31), (31, 40), (40, 50), (50, 59), (59, 69), (69, 78))


def _geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(config)


def _topology() -> PhysicalTopology:
    devices = tuple(
        PhysicalDevice(
            device_id=device,
            process_index=device // 4,
            local_device_id=device % 4,
            coordinates=(device % 2, (device // 2) % 4, device // 8),
            core_on_chip=0,
            platform="tpu",
            device_kind="TPU v4",
        )
        for device in range(32)
    )
    return PhysicalTopology(
        slice_name="db-v4-64-od",
        topology_shape=(2, 4, 4),
        devices=devices,
    )


def _plan(geometry: ModelGeometry | None = None) -> ExecutionPlan:
    geometry = _geometry() if geometry is None else geometry
    assignments = tuple(
        StageAssignment(
            stage_id=stage,
            process_index=stage,
            device_ids=tuple(range(stage * 4, stage * 4 + 4)),
            layer_start=start,
            layer_end_exclusive=end,
            persistent_weight_bytes=1,
            fp8_scale_bytes=1,
            kv_bytes_at_target_context=1,
            dsa_state_bytes=1,
            temporary_bytes=1,
            reserved_overlay_bytes=0,
        )
        for stage, (start, end) in enumerate(PP8_RANGES)
    )
    return ExecutionPlan(
        name=PlanName.PP8_LP4,
        geometry=geometry,
        topology=_topology(),
        target_context_length=262_144,
        pipeline_stages=8,
        local_parallel_size=4,
        stage_assignments=assignments,
        local_mesh_shape=(4,),
        residual_layout="stage_local_replicated",
        expert_layout="complete_expert_identity_lp4",
        kv_layout="stage_layer_context_sharded_lp4",
        transport="collective_permute_stage_ring",
    )


def test_real_pp8_schedule_covers_all_layers_and_padded_slots() -> None:
    schedule = build_pipeline_schedule(_plan())
    assert schedule.layer_count == 78
    assert schedule.maximum_dense_slots == 3
    assert schedule.maximum_sparse_slots == 10
    assert [stage.dense_slot_count for stage in schedule.stages] == [3, 0, 0, 0, 0, 0, 0, 0]
    assert [stage.sparse_slot_count for stage in schedule.stages] == [9, 10, 9, 9, 10, 9, 10, 9]
    assert all(stage.padded_dense_slot_count == 3 for stage in schedule.stages)
    assert all(stage.padded_sparse_slot_count == 10 for stage in schedule.stages)
    assert len(schedule.schedule_hash) == 64
    assert schedule.schedule_hash == build_pipeline_schedule(_plan()).schedule_hash


def test_real_pp8_indexshare_crossings_are_explicit_and_compact() -> None:
    schedule = build_pipeline_schedule(_plan())
    assert [item.boundary_layer for item in schedule.index_share_transfers] == [12, 31, 40, 59, 69]
    assert [item.producer_layer for item in schedule.index_share_transfers] == [10, 30, 38, 58, 66]
    assert all(item.byte_count == 8192 for item in schedule.index_share_transfers)
    assert [stage.assignment.layer_start for stage in schedule.stages if stage.incoming_index_state] == [12, 31, 40, 59, 69]


def test_layer_slots_are_kind_local_and_indexshare_uses_latest_full() -> None:
    schedule = build_pipeline_schedule(_plan())
    layers = [layer for stage in schedule.stages for layer in stage.layers]
    assert [(layer.layer_id, layer.dense_slot) for layer in layers[:3]] == [(0, 0), (1, 1), (2, 2)]
    assert layers[3].sparse_slot == 0
    assert layers[12].sparse_slot == 0
    assert layers[2].index_state_producer_layer == 2
    assert [layers[layer].index_state_producer_layer for layer in (3, 4, 5)] == [2, 2, 2]
    assert layers[6].index_state_producer_layer == 6


def test_schedule_refuses_shared_indexer_without_a_producer() -> None:
    geometry = _geometry()
    broken = replace(
        geometry,
        indexer_types=("shared", *geometry.indexer_types[1:]),
    )
    with pytest.raises(PlanValidationError, match="starts with IndexShare"):
        build_pipeline_schedule(_plan(broken))


def test_schedule_refuses_state_reuse_beyond_declared_group() -> None:
    geometry = _geometry()
    types = list(geometry.indexer_types)
    types[6] = "shared"
    broken = replace(geometry, indexer_types=tuple(types))
    with pytest.raises(PlanValidationError, match="beyond index_share_group_size"):
        build_pipeline_schedule(_plan(broken))
