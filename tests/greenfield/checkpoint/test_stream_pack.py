from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.checkpoint import (
    build_destination_file_plans,
    build_destination_probe_plans,
    destination_groups,
    stream_pack_group,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.partitioning import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    DestinationShard,
    MemoryPolicy,
    PlacementLedger,
    PlacementRecipe,
    build_layout_manifest,
    build_pipeline_plan,
    read_source_inventory,
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
        devices.append(
            PhysicalDevice(
                device_id=device_id,
                process_index=device_id // 4,
                local_device_id=device_id % 4,
                coordinates=(x, within // 4, within % 4),
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


def fixture(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    model = geometry()
    source_root = tmp_path / "source"
    source_root.mkdir()
    tensors = {
        "model.layers.0.synthetic.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(4, 8),
        **{
            f"model.layers.{layer}.synthetic.weight": torch.arange(
                8, dtype=torch.float32
            ) + layer * 10
            for layer in range(1, 8)
        },
        "model.embed_tokens.weight": torch.arange(8, dtype=torch.float32) + 100,
        "lm_head.weight": torch.arange(8, dtype=torch.float32) + 200,
        "model.norm.weight": torch.arange(4, dtype=torch.float32) + 300,
        "model.layers.8.synthetic.weight": torch.arange(
            8, dtype=torch.float32
        ) + 400,
    }
    filename = "model.safetensors"
    save_file(tensors, source_root / filename)
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (source_root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {name: filename for name in tensors},
            }
        )
    )
    inventory = read_source_inventory(
        source_root,
        model_id=model.model_id,
        source_revision="fixture-v1",
        config_filename=None,
    )
    recipes = []
    for tensor in inventory.tensors:
        name = tensor.name
        layer_id = None
        if name.startswith("model.layers."):
            layer_id = int(name.split(".")[2])
        load_set = MTP_LOAD_SET if layer_id == 8 else BASE_LOAD_SET
        if name == "model.layers.0.synthetic.weight":
            shards = tuple(
                DestinationShard(
                    device_slot=slot,
                    shape=(4, 2),
                    byte_count=32,
                    axis=1,
                    axis_start=slot * 2,
                    axis_end_exclusive=(slot + 1) * 2,
                )
                for slot in range(4)
            )
            layout = "axis_sharded"
        elif name == "model.norm.weight":
            shards = tuple(
                DestinationShard(
                    device_slot=slot,
                    shape=tensor.shape,
                    byte_count=tensor.byte_count,
                )
                for slot in range(4)
            )
            layout = "replicated"
        elif load_set == MTP_LOAD_SET:
            shards = (
                DestinationShard(
                    device_slot=2,
                    shape=tensor.shape,
                    byte_count=tensor.byte_count,
                ),
            )
            layout = "expert_identity"
        else:
            shards = tuple(
                DestinationShard(
                    device_slot=slot,
                    shape=(2,),
                    byte_count=8,
                    axis=0,
                    axis_start=slot * 2,
                    axis_end_exclusive=(slot + 1) * 2,
                )
                for slot in range(4)
            )
            layout = "axis_sharded"
        recipes.append(
            PlacementRecipe(
                source_name=name,
                source_byte_count=tensor.byte_count,
                source_dtype=tensor.dtype,
                layer_id=layer_id,
                load_set=load_set,
                layout=layout,
                value_class="parameter",
                shards=shards,
            )
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
    layout = build_layout_manifest(
        inventory=inventory,
        ledger=ledger,
        partition=partition,
        source_uri="gs://driftbench-dsv4-uc/models/fixture",
        code_hash="a" * 40,
        topology_hash=partition.execution_plan.topology.topology_hash,
        plan_group_hash="b" * 64,
    )
    return source_root, layout


def pack_all(tmp_path: Path):
    source_root, layout = fixture(tmp_path)
    plans = build_destination_file_plans(layout)
    output_root = tmp_path / "packed"
    evidence = []
    for group in destination_groups(plans):
        outputs = {}
        handles = []
        try:
            for plan in group:
                path = output_root / plan.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("wb")
                handles.append(handle)
                outputs[plan.filename] = handle
            evidence.extend(
                stream_pack_group(
                    layout=layout,
                    plans=group,
                    source_root=source_root,
                    outputs=outputs,
                    chunk_bytes=64,
                )
            )
        finally:
            for handle in handles:
                handle.close()
    return source_root, layout, plans, output_root, evidence


def test_stream_pack_writes_exact_axis0_axis1_replica_and_identity_shards(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    _, layout, plans, output_root, evidence = pack_all(tmp_path)
    assert len(evidence) == len(plans)
    for record in evidence:
        path = output_root / record.filename
        assert path.stat().st_size == record.file_bytes
        assert sha256(path.read_bytes()).hexdigest() == record.sha256

    stage0 = layout["plan_manifest"]["stages"][0]
    for slot in range(4):
        path = output_root / f"base_decoder/stage_00/device_slot_{slot:02d}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = handle.get_tensor("model.layers.0.synthetic.weight")
        expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)[
            :, slot * 2 : (slot + 1) * 2
        ]
        assert torch.equal(actual, expected)
        assert stage0["device_ids"][slot] in set(range(32))

    final_stage = layout["plan_manifest"]["stages"][-1]["stage_id"]
    for slot in range(4):
        path = output_root / (
            f"base_decoder/stage_{final_stage:02d}/device_slot_{slot:02d}.safetensors"
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert handle.get_tensor("model.norm.weight").tolist() == [
                300.0,
                301.0,
                302.0,
                303.0,
            ]
    mtp_path = output_root / "mtp_optional/stage_00/device_slot_02.safetensors"
    with safe_open(mtp_path, framework="pt", device="cpu") as handle:
        assert handle.get_tensor("model.layers.8.synthetic.weight").tolist() == [
            400.0 + value for value in range(8)
        ]


def test_stream_pack_refuses_source_size_drift(tmp_path: Path) -> None:
    source_root, layout = fixture(tmp_path)
    plans = build_destination_file_plans(layout)
    group = destination_groups(plans)[0]
    with (source_root / "model.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    output_root = tmp_path / "packed"
    outputs = {}
    handles = []
    try:
        for plan in group:
            path = output_root / plan.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("wb")
            handles.append(handle)
            outputs[plan.filename] = handle
        with pytest.raises(CheckpointValidationError, match="size changed"):
            stream_pack_group(
                layout=layout,
                plans=group,
                source_root=source_root,
                outputs=outputs,
            )
    finally:
        for handle in handles:
            handle.close()


def test_stream_pack_can_resume_one_missing_axis_shard(tmp_path: Path) -> None:
    import torch
    from safetensors import safe_open

    source_root, layout = fixture(tmp_path)
    plans = build_destination_file_plans(layout)
    target = next(
        plan
        for plan in plans
        if plan.load_set == BASE_LOAD_SET
        and plan.stage_id == 0
        and plan.device_slot == 2
    )
    path = tmp_path / "resume" / target.filename
    path.parent.mkdir(parents=True)
    with path.open("wb") as output:
        evidence = stream_pack_group(
            layout=layout,
            plans=(target,),
            source_root=source_root,
            outputs={target.filename: output},
            chunk_bytes=64,
        )
    assert evidence[0].file_bytes == path.stat().st_size
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual = handle.get_tensor("model.layers.0.synthetic.weight")
    expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, 4:6]
    assert torch.equal(actual, expected)


def test_probe_plans_stream_only_selected_real_layout_leaf(tmp_path: Path) -> None:
    import torch
    from safetensors import safe_open

    source_root, layout = fixture(tmp_path)
    selected = "model.layers.0.synthetic.weight"
    plans = build_destination_probe_plans(layout, source_names=(selected,))
    assert len(plans) == 4
    assert all(tuple(tensor.name for tensor in plan.tensors) == (selected,)
               for plan in plans)

    with safe_open(
        source_root / "model.safetensors", framework="pt", device="cpu"
    ) as handle:
        expected_hash = sha256(
            handle.get_tensor(selected).numpy().tobytes()
        ).hexdigest()
    output_root = tmp_path / "probe"
    outputs = {}
    handles = []
    try:
        for plan in plans:
            path = output_root / plan.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("wb")
            handles.append(stream)
            outputs[plan.filename] = stream
        evidence = stream_pack_group(
            layout=layout,
            plans=plans,
            source_root=source_root,
            outputs=outputs,
            chunk_bytes=64,
            expected_source_sha256={selected: expected_hash},
        )
    finally:
        for stream in handles:
            stream.close()
    assert len(evidence) == 4
    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    for slot, plan in enumerate(plans):
        with safe_open(
            output_root / plan.filename, framework="pt", device="cpu"
        ) as handle:
            actual = handle.get_tensor(selected)
        assert torch.equal(actual, source[:, slot * 2 : (slot + 1) * 2])


def test_probe_plans_refuse_unknown_cross_stage_and_duplicate_sources(
    tmp_path: Path,
) -> None:
    _, layout = fixture(tmp_path)
    with pytest.raises(CheckpointValidationError, match="unknown source"):
        build_destination_probe_plans(layout, source_names=("missing",))
    with pytest.raises(CheckpointValidationError, match="nonempty and unique"):
        build_destination_probe_plans(
            layout,
            source_names=(
                "model.layers.0.synthetic.weight",
                "model.layers.0.synthetic.weight",
            ),
        )
    with pytest.raises(CheckpointValidationError, match="one load-set/stage"):
        build_destination_probe_plans(
            layout,
            source_names=(
                "model.layers.0.synthetic.weight",
                "model.norm.weight",
            ),
        )


def test_stream_pack_validates_exact_raw_source_tensor_hashes(
    tmp_path: Path,
) -> None:
    from safetensors import safe_open

    source_root, layout = fixture(tmp_path)
    plans = build_destination_file_plans(layout)
    group = destination_groups(plans)[0]
    filenames = {plan.filename for plan in group}
    names = {
        placement["source"]["name"]
        for placement in layout["placements"]
        if any(
            destination["filename"] in filenames
            for destination in placement["destinations"]
        )
    }
    with safe_open(
        source_root / "model.safetensors", framework="pt", device="cpu"
    ) as handle:
        expected_hashes = {
            name: sha256(handle.get_tensor(name).numpy().tobytes()).hexdigest()
            for name in names
        }
    outputs = {plan.filename: BytesIO() for plan in group}
    evidence = stream_pack_group(
        layout=layout,
        plans=group,
        source_root=source_root,
        outputs=outputs,
        chunk_bytes=64,
        expected_source_sha256=expected_hashes,
    )
    assert len(evidence) == len(group)

    bad_hashes = dict(expected_hashes)
    bad_hashes[sorted(bad_hashes)[0]] = "0" * 64
    with pytest.raises(CheckpointValidationError, match="source tensor SHA-256"):
        stream_pack_group(
            layout=layout,
            plans=group,
            source_root=source_root,
            outputs={plan.filename: BytesIO() for plan in group},
            chunk_bytes=64,
            expected_source_sha256=bad_hashes,
        )


def test_stream_pack_hash_validation_requires_complete_owners(
    tmp_path: Path,
) -> None:
    source_root, layout = fixture(tmp_path)
    plans = build_destination_file_plans(layout)
    target = destination_groups(plans)[0][0]
    names = {
        placement["source"]["name"]
        for placement in layout["placements"]
        if any(
            destination["filename"] == target.filename
            for destination in placement["destinations"]
        )
    }
    with pytest.raises(CheckpointValidationError, match="every destination shard"):
        stream_pack_group(
            layout=layout,
            plans=(target,),
            source_root=source_root,
            outputs={target.filename: BytesIO()},
            expected_source_sha256={name: "0" * 64 for name in names},
        )
