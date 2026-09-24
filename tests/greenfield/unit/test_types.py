from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from glm_tpu.optimized.errors import GeometryValidationError, PlanValidationError, TopologyValidationError
from glm_tpu.optimized.geometry import (
    ExecutionPlan,
    ModelGeometry,
    PhysicalDevice,
    PhysicalTopology,
    PlanName,
    StageAssignment,
)


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(config)


def topology(*, reverse: bool = False) -> PhysicalTopology:
    devices = []
    device_id = 0
    for x in range(2):
        for y in range(4):
            for z in range(4):
                process = device_id // 4
                devices.append(
                    PhysicalDevice(
                        device_id=device_id,
                        process_index=process,
                        local_device_id=device_id % 4,
                        coordinates=(x, y, z),
                        core_on_chip=0,
                        platform="tpu",
                        device_kind="TPU v4",
                    )
                )
                device_id += 1
    if reverse:
        devices.reverse()
    return PhysicalTopology(
        slice_name="db-v4-64-od",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def assignments(stages: int, local_size: int) -> tuple[StageAssignment, ...]:
    result = []
    for stage in range(stages):
        layer_start = (78 * stage) // stages
        layer_end = (78 * (stage + 1)) // stages
        process = stage if stages == 8 else stage // 2
        result.append(
            StageAssignment(
                stage_id=stage,
                process_index=process,
                device_ids=tuple(range(stage * local_size, (stage + 1) * local_size)),
                layer_start=layer_start,
                layer_end_exclusive=layer_end,
                persistent_weight_bytes=100 + stage,
                fp8_scale_bytes=10,
                kv_bytes_at_target_context=20,
                dsa_state_bytes=30,
                temporary_bytes=40,
                reserved_overlay_bytes=50,
            )
        )
    return tuple(result)


def pp8_plan(geometry: ModelGeometry, **changes: object) -> ExecutionPlan:
    values = {
        "name": PlanName.PP8_LP4,
        "geometry": geometry,
        "topology": topology(),
        "target_context_length": 262_144,
        "pipeline_stages": 8,
        "local_parallel_size": 4,
        "stage_assignments": assignments(8, 4),
        "local_mesh_shape": (4,),
        "residual_layout": "stage_local_replicated",
        "expert_layout": "expert_identity_sharded_lp4",
        "kv_layout": "stage_layer_context_sharded_lp4",
        "transport": "collective_permute_stage_ring",
    }
    values.update(changes)
    return ExecutionPlan(**values)


def test_checked_in_glm_geometry_is_exact(geometry: ModelGeometry) -> None:
    assert geometry.model_id == "zai-org/GLM-5.2-FP8"
    assert geometry.num_layers == 78
    assert geometry.first_dense_layers == 3
    assert geometry.hidden_size == 6144
    assert geometry.num_routed_experts == 256
    assert geometry.routed_top_k == 8
    assert geometry.dsa_top_k == 2048
    assert geometry.dsa_indexer_heads == 32
    assert geometry.index_share_group_size == 4
    assert geometry.qk_nope_head_dim == 192
    assert geometry.qk_rope_head_dim == 64
    assert geometry.v_head_dim == 256
    assert geometry.num_nextn_predict_layers == 1
    assert geometry.fp8_block_shape == (128, 128)
    assert geometry.weight_storage_dtype == "fp8:e4m3"
    assert len(geometry.mlp_layer_types) == len(geometry.indexer_types) == 78


def test_geometry_refuses_inconsistent_layer_schedule(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(GeometryValidationError, match="first_dense_layers"):
        replace(
            geometry,
            mlp_layer_types=("sparse",) + geometry.mlp_layer_types[1:],
        )


def test_geometry_refuses_boolean_integer(geometry: ModelGeometry) -> None:
    with pytest.raises(GeometryValidationError, match="hidden_size"):
        replace(geometry, hidden_size=True)


def test_topology_is_canonical_and_content_addressed() -> None:
    ordered = topology()
    reversed_input = topology(reverse=True)
    assert ordered == reversed_input
    assert ordered.topology_hash == reversed_input.topology_hash
    assert len(ordered.topology_hash) == 64
    assert ordered.process_indices == tuple(range(8))


def test_topology_refuses_duplicate_coordinates() -> None:
    good = topology()
    duplicate = replace(good.devices[1], coordinates=good.devices[0].coordinates)
    with pytest.raises(TopologyValidationError, match="coordinates"):
        replace(good, devices=(good.devices[0], duplicate, *good.devices[2:]))


def test_pp8_round_trip_and_hash_are_deterministic(
    geometry: ModelGeometry,
) -> None:
    plan = pp8_plan(geometry)
    decoded = ExecutionPlan.from_json(plan.to_json())
    assert decoded == plan
    assert decoded.plan_hash == plan.plan_hash
    assert len(plan.plan_hash) == 64
    assert plan.to_json() == decoded.to_json()


def test_plan_hash_covers_topology_and_memory(geometry: ModelGeometry) -> None:
    plan = pp8_plan(geometry)
    changed_device = replace(
        plan.topology.devices[0], core_on_chip=1
    )
    changed_topology = replace(
        plan.topology,
        devices=(changed_device, *plan.topology.devices[1:]),
    )
    assert replace(plan, topology=changed_topology).plan_hash != plan.plan_hash

    first = replace(
        plan.stage_assignments[0],
        reserved_overlay_bytes=plan.stage_assignments[0].reserved_overlay_bytes + 1,
    )
    changed_memory = replace(
        plan,
        stage_assignments=(first, *plan.stage_assignments[1:]),
    )
    assert changed_memory.plan_hash != plan.plan_hash


def test_plan_normalizes_assignment_order(geometry: ModelGeometry) -> None:
    plan = pp8_plan(geometry)
    reversed_plan = pp8_plan(
        geometry, stage_assignments=tuple(reversed(plan.stage_assignments))
    )
    assert reversed_plan == plan
    assert reversed_plan.plan_hash == plan.plan_hash


def test_plans_are_immutable(geometry: ModelGeometry) -> None:
    plan = pp8_plan(geometry)
    with pytest.raises(FrozenInstanceError):
        plan.transport = "host_staging"  # type: ignore[misc]


def test_pp16_host_local_contract(geometry: ModelGeometry) -> None:
    plan = ExecutionPlan(
        name=PlanName.PP16_LP2,
        geometry=geometry,
        topology=topology(),
        target_context_length=262_144,
        pipeline_stages=16,
        local_parallel_size=2,
        stage_assignments=assignments(16, 2),
        local_mesh_shape=(2,),
        residual_layout="stage_local_replicated",
        expert_layout="expert_identity_sharded_lp2",
        kv_layout="stage_layer_context_sharded_lp2",
        transport="collective_permute_stage_ring",
    )
    counts = {}
    for stage in plan.stage_assignments:
        counts[stage.process_index] = counts.get(stage.process_index, 0) + 1
    assert set(counts.values()) == {2}


def test_plan_refuses_wrong_named_geometry(geometry: ModelGeometry) -> None:
    with pytest.raises(PlanValidationError, match="requires stages/local size"):
        pp8_plan(geometry, pipeline_stages=16)


def test_plan_refuses_layer_gap(geometry: ModelGeometry) -> None:
    stages = list(assignments(8, 4))
    stages[1] = replace(stages[1], layer_start=stages[1].layer_start + 1)
    with pytest.raises(PlanValidationError, match="gap-free"):
        pp8_plan(geometry, stage_assignments=tuple(stages))


def test_plan_refuses_cross_host_local_group(geometry: ModelGeometry) -> None:
    stages = list(assignments(8, 4))
    stages[0] = replace(stages[0], device_ids=(0, 1, 2, 4))
    stages[1] = replace(stages[1], device_ids=(3, 5, 6, 7))
    with pytest.raises(PlanValidationError, match="host-local"):
        pp8_plan(geometry, stage_assignments=tuple(stages))


def test_stage_refuses_negative_memory() -> None:
    with pytest.raises(PlanValidationError, match="temporary_bytes"):
        replace(assignments(8, 4)[0], temporary_bytes=-1)


def test_stage_accounted_bytes_includes_every_budget_class() -> None:
    stage = assignments(8, 4)[0]
    assert stage.accounted_bytes == 250
