from __future__ import annotations

import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PartitioningValidationError
from glm_tpu.greenfield.partitioning import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    DestinationShard,
    MemoryPolicy,
    PlacementLedger,
    PlacementRecipe,
    build_pipeline_plan,
)
from glm_tpu.greenfield.types import (
    ModelGeometry,
    PhysicalDevice,
    PhysicalTopology,
    PlanName,
)


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(config)


def topology() -> PhysicalTopology:
    devices = []
    device_id = 0
    for x in range(2):
        for y in range(4):
            for z in range(4):
                devices.append(
                    PhysicalDevice(
                        device_id=device_id,
                        process_index=device_id // 4,
                        local_device_id=device_id % 4,
                        coordinates=(x, y, z),
                        core_on_chip=0,
                        platform="tpu",
                        device_kind="TPU v4",
                    )
                )
                device_id += 1
    return PhysicalTopology(
        slice_name="db-v4-64-od",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def sharded_recipe(
    name: str,
    *,
    source_bytes: int,
    local_size: int,
    layer_id: int | None,
    load_set: str = BASE_LOAD_SET,
    value_class: str = "parameter",
) -> PlacementRecipe:
    assert source_bytes % local_size == 0
    width = source_bytes // local_size
    return PlacementRecipe(
        source_name=name,
        source_byte_count=source_bytes,
        source_dtype="F8_E4M3",
        layer_id=layer_id,
        load_set=load_set,
        layout="axis_sharded",
        value_class=value_class,
        shards=tuple(
            DestinationShard(
                device_slot=slot,
                shape=(width,),
                byte_count=width,
                axis=0,
                axis_start=slot * width,
                axis_end_exclusive=(slot + 1) * width,
            )
            for slot in range(local_size)
        ),
    )


def replicated_recipe(
    name: str,
    *,
    source_bytes: int,
    local_size: int,
    layer_id: int | None,
) -> PlacementRecipe:
    return PlacementRecipe(
        source_name=name,
        source_byte_count=source_bytes,
        source_dtype="BF16",
        layer_id=layer_id,
        load_set=BASE_LOAD_SET,
        layout="replicated",
        value_class="parameter",
        shards=tuple(
            DestinationShard(
                device_slot=slot,
                shape=(source_bytes // 2,),
                byte_count=source_bytes,
            )
            for slot in range(local_size)
        ),
    )


def ledger(geometry: ModelGeometry, local_size: int) -> PlacementLedger:
    recipes = []
    for layer in range(geometry.num_layers):
        weight_bytes = 400 if layer < 3 else 4_000
        recipes.append(
            sharded_recipe(
                f"model.layers.{layer}.synthetic.weight",
                source_bytes=weight_bytes,
                local_size=local_size,
                layer_id=layer,
            )
        )
        recipes.append(
            sharded_recipe(
                f"model.layers.{layer}.synthetic.weight_scale_inv",
                source_bytes=local_size * 4,
                local_size=local_size,
                layer_id=layer,
                value_class="fp8_scale",
            )
        )
    recipes.extend(
        (
            sharded_recipe(
                "model.embed_tokens.weight",
                source_bytes=local_size * 100,
                local_size=local_size,
                layer_id=None,
            ),
            sharded_recipe(
                "lm_head.weight",
                source_bytes=local_size * 100,
                local_size=local_size,
                layer_id=None,
            ),
            replicated_recipe(
                "model.norm.weight",
                source_bytes=16,
                local_size=local_size,
                layer_id=None,
            ),
            sharded_recipe(
                "model.layers.78.synthetic.weight",
                source_bytes=local_size * 200,
                local_size=local_size,
                layer_id=geometry.num_layers,
                load_set=MTP_LOAD_SET,
            ),
        )
    )
    return PlacementLedger(
        model_id=geometry.model_id,
        source_inventory_sha256="a" * 64,
        local_parallel_size=local_size,
        recipes=tuple(recipes),
    )


def policy(**changes: object) -> MemoryPolicy:
    values = {
        "target_context_length": 128,
        "hbm_limit_bytes": 10_000_000,
        "temporary_floor_bytes": 0,
        "reserved_overlay_bytes": 0,
        "provenance": "unit-test",
    }
    values.update(changes)
    return MemoryPolicy(**values)


def test_pp8_is_byte_balanced_contiguous_and_hashable(
    geometry: ModelGeometry,
) -> None:
    result = build_pipeline_plan(
        name=PlanName.PP8_LP4,
        geometry=geometry,
        topology=topology(),
        ledger=ledger(geometry, 4),
        memory_policy=policy(),
    )
    assert result.execution_plan.pipeline_stages == 8
    assert result.execution_plan.local_parallel_size == 4
    assert result.capacity_feasible
    assert not result.promotion_memory_proven
    assert len(result.plan_manifest_sha256) == 64
    assert result.stages[0].layer_start == 0
    assert result.stages[-1].layer_end_exclusive == 78
    assert all(
        left.layer_end_exclusive == right.layer_start
        for left, right in zip(result.stages, result.stages[1:])
    )
    assert sum(
        stage.layer_end_exclusive - stage.layer_start for stage in result.stages
    ) == 78
    assert result.stages[0].optional_mtp_slot_bytes == (200, 200, 200, 200)
    assert all(
        stage.optional_mtp_slot_bytes == (0, 0, 0, 0)
        for stage in result.stages[1:]
    )
    assert tuple(
        stage.layer_start for stage in result.stages[1:]
        if stage.incoming_index_share_crossing
    ) == result.index_share_crossings


def test_pp16_uses_two_stages_per_host_and_same_complete_layer_cover(
    geometry: ModelGeometry,
) -> None:
    result = build_pipeline_plan(
        name=PlanName.PP16_LP2,
        geometry=geometry,
        topology=topology(),
        ledger=ledger(geometry, 2),
        memory_policy=policy(),
    )
    counts: dict[int, int] = {}
    for stage in result.stages:
        counts[stage.process_index] = counts.get(stage.process_index, 0) + 1
    assert len(result.stages) == 16
    assert set(counts.values()) == {2}
    assert result.stages[-1].layer_end_exclusive == geometry.num_layers


def test_stage_assignment_refuses_per_chip_hbm_overflow(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(PartitioningValidationError, match="exceeds per-chip HBM"):
        build_pipeline_plan(
            name=PlanName.PP8_LP4,
            geometry=geometry,
            topology=topology(),
            ledger=ledger(geometry, 4),
            memory_policy=policy(hbm_limit_bytes=1),
        )


def test_plan_refuses_ledger_local_size_mismatch(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(PartitioningValidationError, match="local size"):
        build_pipeline_plan(
            name=PlanName.PP16_LP2,
            geometry=geometry,
            topology=topology(),
            ledger=ledger(geometry, 4),
            memory_policy=policy(),
        )
