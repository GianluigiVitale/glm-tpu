from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.partitioning import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    DestinationShard,
    MemoryPolicy,
    PlacementLedger,
    PlacementRecipe,
    SourceFile,
    SourceInventory,
    SourceTensor,
    build_layout_manifest,
    build_pipeline_plan,
    inspect_layout_manifest,
    write_layout_manifest,
)
from glm_tpu.greenfield.types import (
    ModelGeometry,
    PhysicalDevice,
    PhysicalTopology,
    PlanName,
)


REPO = Path(__file__).resolve().parents[3]


def geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    full = ModelGeometry.from_hf_config(config)
    return replace(
        full,
        num_layers=8,
        first_dense_layers=3,
        mlp_layer_types=("dense", "dense", "dense", *(["sparse"] * 5)),
        indexer_types=full.indexer_types[:8],
    )


def topology() -> PhysicalTopology:
    devices = []
    for device_id in range(32):
        x = device_id // 16
        within = device_id % 16
        y = within // 4
        z = within % 4
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
    return PhysicalTopology(
        slice_name="db-v4-64-od",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def fixture_contract():
    model = geometry()
    names = [f"model.layers.{layer}.synthetic.weight" for layer in range(8)]
    names.extend(
        (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.8.synthetic.weight",
        )
    )
    tensors = []
    recipes = []
    offset = 0
    for name in names:
        layer_id = None
        if name.startswith("model.layers."):
            layer_id = int(name.split(".")[2])
        load_set = MTP_LOAD_SET if layer_id == 8 else BASE_LOAD_SET
        tensor = SourceTensor(
            name=name,
            filename="model.safetensors",
            dtype="F8_E4M3",
            shape=(32,),
            data_offset_start=offset,
            data_offset_end=offset + 32,
        )
        offset += 32
        tensors.append(tensor)
        recipes.append(
            PlacementRecipe(
                source_name=name,
                source_byte_count=32,
                source_dtype="F8_E4M3",
                layer_id=layer_id,
                load_set=load_set,
                layout="axis_sharded",
                value_class="parameter",
                shards=tuple(
                    DestinationShard(
                        device_slot=slot,
                        shape=(8,),
                        byte_count=8,
                        axis=0,
                        axis_start=slot * 8,
                        axis_end_exclusive=(slot + 1) * 8,
                    )
                    for slot in range(4)
                ),
            )
        )
    inventory = SourceInventory(
        model_id=model.model_id,
        source_revision="fixture-v1",
        index_filename="model.safetensors.index.json",
        index_sha256="a" * 64,
        config_filename="config.json",
        config_sha256="b" * 64,
        declared_payload_bytes=offset,
        files=(
            SourceFile(
                filename="model.safetensors",
                file_bytes=offset,
                header_bytes=0,
                payload_bytes=offset,
                tensor_count=len(tensors),
                header_sha256="c" * 64,
            ),
        ),
        tensors=tuple(tensors),
    )
    ledger = PlacementLedger(
        model_id=model.model_id,
        source_inventory_sha256=inventory.inventory_sha256,
        local_parallel_size=4,
        recipes=tuple(recipes),
    )
    partition = build_pipeline_plan(
        name=PlanName.PP8_LP4,
        geometry=model,
        topology=topology(),
        ledger=ledger,
        memory_policy=MemoryPolicy(
            target_context_length=128,
            hbm_limit_bytes=10_000_000,
            temporary_floor_bytes=0,
            provenance="unit-test",
        ),
    )
    return inventory, ledger, partition


def manifest():
    inventory, ledger, partition = fixture_contract()
    return build_layout_manifest(
        inventory=inventory,
        ledger=ledger,
        partition=partition,
        source_uri="gs://driftbench-dsv4-uc/models/fixture",
        code_hash="d" * 40,
        topology_hash=partition.execution_plan.topology.topology_hash,
        plan_group_hash="e" * 64,
    )


def test_complete_layout_maps_every_leaf_to_physical_files_and_reconciles() -> None:
    value = manifest()
    assert len(value["placements"]) == 12
    assert value["source"]["leaf_count"] == 12
    assert value["source"]["payload_bytes"] == 12 * 32
    assert value["packed_payload_bytes"] == 12 * 32
    assert len(value["destination_files"]) == 36
    assert {
        destination["device_id"]
        for placement in value["placements"]
        for destination in placement["destinations"]
        if placement["load_set"] == BASE_LOAD_SET
    } == set(range(32))
    assert any(
        record["filename"].startswith("mtp_optional/")
        for record in value["destination_files"]
    )
    assert len(value["manifest_sha256"]) == 64


def test_layout_writer_is_append_only_and_inspector_rehashes(tmp_path: Path) -> None:
    value = manifest()
    output = tmp_path / "layout.json"
    write_layout_manifest(value, output)
    assert inspect_layout_manifest(output) == value
    with pytest.raises(CheckpointValidationError, match="overwrite"):
        write_layout_manifest(value, output)
    decoded = json.loads(output.read_text())
    decoded["packed_payload_bytes"] += 1
    output.write_text(json.dumps(decoded))
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        inspect_layout_manifest(output)


def test_layout_refuses_wrong_bucket_or_inventory_binding() -> None:
    inventory, ledger, partition = fixture_contract()
    kwargs = {
        "inventory": inventory,
        "ledger": ledger,
        "partition": partition,
        "source_uri": "gs://wrong/model",
        "code_hash": "d" * 40,
        "topology_hash": partition.execution_plan.topology.topology_hash,
        "plan_group_hash": "e" * 64,
    }
    with pytest.raises(CheckpointValidationError, match="approved"):
        build_layout_manifest(**kwargs)
    with pytest.raises(CheckpointValidationError, match="bind"):
        build_layout_manifest(
            **{
                **kwargs,
                "source_uri": "gs://driftbench-dsv4-uc/models/fixture",
                "ledger": replace(ledger, source_inventory_sha256="f" * 64),
            }
        )
