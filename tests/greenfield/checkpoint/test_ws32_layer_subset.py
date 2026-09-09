from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from glm_tpu.greenfield.checkpoint.ws32_layer_subset import (
    iter_ws32_layer_subset_host_tensors,
    read_ws32_layer_subset_metadata,
    ws32_expected_layer_names,
)
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
    Ws32RuntimePackConfig,
    _mapping_hash,
    pack_ws32_runtime_checkpoint,
    verify_ws32_runtime_checkpoint,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.partitioning import read_source_inventory
from glm_tpu.greenfield.sharding.ws32 import Ws32PhysicalMesh
from glm_tpu.greenfield.types import ModelGeometry
from tests.greenfield.checkpoint.test_ws32_runtime_checkpoint import (
    ROOT,
    _geometry,
    _seal,
)


def _physical() -> Ws32PhysicalMesh:
    # Non-identity device order makes accidental device-id == file-slot fail.
    ids = tuple(reversed(range(32)))
    rows = tuple(ids[row * 4 : row * 4 + 4] for row in range(8))
    return Ws32PhysicalMesh(
        device_ids=rows,
        feature_groups=rows,
        expert_groups=tuple(tuple(row[column] for row in rows) for column in range(4)),
    )


def _fixture(tmp_path: Path, *, num_layers: int = 2, missing: str | None = None):
    import torch
    from safetensors.torch import save_file

    geometry = replace(
        _geometry(),
        num_layers=num_layers,
        first_dense_layers=num_layers,
        mlp_layer_types=("dense",) * num_layers,
        indexer_types=("full",) * num_layers,
    )
    # Independent small source schema; do not create it from the selected-name helper.
    tensors = {"model.embed_tokens.weight": torch.ones((16, 8), dtype=torch.bfloat16)}
    norm_shapes = {
        "input_layernorm.weight": (8,),
        "post_attention_layernorm.weight": (8,),
        "self_attn.q_a_layernorm.weight": (8,),
        "self_attn.kv_a_layernorm.weight": (4,),
        "self_attn.indexer.k_norm.weight": (2,),
        "self_attn.indexer.k_norm.bias": (2,),
        "self_attn.indexer.weights_proj.weight": (8, 8),
    }
    matrix_shapes = {
        "self_attn.q_a_proj": (8, 8),
        "self_attn.kv_a_proj_with_mqa": (6, 8),
        "self_attn.q_b_proj": (32, 8),
        "self_attn.kv_b_proj": (32, 4),
        "self_attn.o_proj": (8, 16),
        "self_attn.indexer.wq_b": (16, 8),
        "self_attn.indexer.wk": (2, 8),
        "mlp.gate_proj": (16, 8),
        "mlp.up_proj": (16, 8),
        "mlp.down_proj": (8, 16),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        for index, (name, shape) in enumerate(norm_shapes.items()):
            tensors[prefix + name] = (
                torch.arange(torch.tensor(shape).prod())
                .reshape(shape)
                .to(torch.bfloat16)
                + layer
                + index
            )
        for index, (name, shape) in enumerate(matrix_shapes.items()):
            value = (
                torch.arange(shape[0] * shape[1]).reshape(shape) % 16 + layer + index
            )
            tensors[prefix + name + ".weight"] = value.to(torch.float8_e4m3fn)
            tensors[prefix + name + ".weight_scale_inv"] = torch.full(
                (shape[0] // 2, shape[1] // 2), 1.0 + index / 16
            )
    if missing is not None:
        tensors.pop(missing)
    source = tmp_path / "source"
    source.mkdir()
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        value.numel() * value.element_size()
                        for value in tensors.values()
                    )
                },
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    inventory = read_source_inventory(
        source,
        model_id=geometry.model_id,
        source_revision="unit-fixture",
        config_filename=None,
    )
    config = Ws32RuntimePackConfig(
        source,
        "gs://driftbench-dsv4-uc/models/unit-fixture",
        tmp_path / "packed",
        "a" * 40,
        _physical().mesh_hash,
    )
    manifest = pack_ws32_runtime_checkpoint(config, inventory, geometry)
    success = _seal(config.output_dir, manifest)
    common = dict(
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash=config.mesh_hash,
        expected_topology_hash="c" * 64,
        inventory=inventory,
        geometry=geometry,
    )
    return config, manifest, common, tensors


def _subset(root: Path, common: dict, **kwargs):
    options = dict(
        layer_ids=(0,), local_slots=tuple(range(32)), max_payload_bytes_per_chip=100_000
    )
    options.update(kwargs)
    return read_ws32_layer_subset_metadata(root, **common, **options)


def _flip(path: Path, offset: int) -> None:
    with path.open("r+b") as stream:
        stream.seek(offset)
        raw = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([raw[0] ^ 1]))


def test_explicit_embedding_is_bounded_and_full_tensor_verified(tmp_path):
    config, _, common, _ = _fixture(tmp_path)
    base = _subset(config.output_dir, common, layer_ids=(0, 1))
    extended = _subset(config.output_dir, common, layer_ids=(0, 1), include_embedding=True)
    schema = extended.metadata.plans[0].tensors
    embedding = next(t for t in schema if t.name == "model.embed_tokens.weight")
    assert extended.include_embedding and not base.include_embedding
    assert extended.payload_bytes_per_chip == base.payload_bytes_per_chip + embedding.byte_count
    assert len(extended.tensor_indices) == len(base.tensor_indices) + 1
    leaves = list(iter_ws32_layer_subset_host_tensors(extended))
    assert {t.name for _, t, _ in leaves} == (
        ws32_expected_layer_names(common["geometry"], 0)
        | ws32_expected_layer_names(common["geometry"], 1)
        | {"model.embed_tokens.weight"}
    )
    with pytest.raises(CheckpointValidationError, match="budget"):
        _subset(config.output_dir, common, layer_ids=(0, 1), include_embedding=True,
                max_payload_bytes_per_chip=base.payload_bytes_per_chip)
    plan = extended.metadata.plans[0]
    _flip(config.output_dir / plan.filename, len(plan.header) + embedding.data_offset_start)
    with pytest.raises(CheckpointValidationError, match="checksum"):
        list(iter_ws32_layer_subset_host_tensors(extended))
    # Existing layer-only integrity scope does not falsely certify this extra leaf.
    list(iter_ws32_layer_subset_host_tensors(base))
    with pytest.raises(ValueError, match="explicit bool"):
        _subset(config.output_dir, common, include_embedding=1)


def test_subset_only_reads_selected_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    config, _, common, _ = _fixture(tmp_path)
    subset = _subset(config.output_dir, common, layer_ids=(1,))
    schema = subset.metadata.plans[0].tensors
    assert len(subset.tensor_indices) == 27
    assert subset.tensor_indices[0] > 0  # Original ledger indices, not subset indices.
    expected_bytes = sum(schema[index].byte_count for index in subset.tensor_indices)
    observed_reads = []
    original_open = Path.open

    class Meter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            assert size >= 0, "unbounded checkpoint read"
            observed_reads.append(
                (Path(self.handle.name).name, self.handle.tell(), size)
            )
            return self.handle.read(size)

    def metered_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return Meter(handle) if path.suffix == ".safetensors" else handle

    monkeypatch.setattr(Path, "open", metered_open)
    total = 0
    for index, tensor, owners in iter_ws32_layer_subset_host_tensors(subset):
        assert index in subset.tensor_indices
        assert tensor.name.startswith("model.layers.1.")
        for slot, host, digest in owners:
            assert host.nbytes == tensor.byte_count
            assert not host.flags.writeable
            assert (
                digest == sha256(memoryview(host.view(np.uint8)).cast("B")).hexdigest()
            )
            assert np.isfinite(host).all()
            total += host.nbytes
    assert total == expected_bytes * 32
    allowed = {
        (
            plan.filename,
            len(plan.header) + schema[index].data_offset_start,
            schema[index].byte_count,
        )
        for plan in subset.metadata.plans
        for index in subset.tensor_indices
    }
    assert all(
        offset == 0 or (name, offset, size) in allowed
        for name, offset, size in observed_reads
    )
    assert sum(size for _, offset, size in observed_reads if offset) == total


def test_missing_complete_layer_refuses(tmp_path: Path) -> None:
    config, _, common, _ = _fixture(
        tmp_path, missing="model.layers.0.post_attention_layernorm.weight"
    )
    with pytest.raises(CheckpointValidationError, match="schema is incomplete"):
        _subset(config.output_dir, common)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_ids", ()),
        ("layer_ids", (True,)),
        ("layer_ids", (2,)),
        ("layer_ids", (0, 0)),
        ("local_slots", (0, 1, 2, 2)),
        ("local_slots", (0, 1)),
        ("max_payload_bytes_per_chip", 0),
        ("max_payload_bytes_per_chip", True),
    ],
)
def test_bad_selection_refuses(tmp_path: Path, field: str, value: object) -> None:
    config, _, common, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        _subset(config.output_dir, common, **{field: value})


def test_budget_refuses_before_payload(tmp_path: Path) -> None:
    config, _, common, _ = _fixture(tmp_path)
    with pytest.raises(CheckpointValidationError, match="exceeds per-chip budget"):
        _subset(config.output_dir, common, max_payload_bytes_per_chip=1)


@pytest.mark.parametrize(
    "mutation", ["selected", "unselected", "header", "truncate", "ledger_index"]
)
def test_integrity_scope(tmp_path: Path, mutation: str) -> None:
    config, manifest, common, _ = _fixture(tmp_path)
    subset = _subset(config.output_dir, common, layer_ids=(1,))
    plan = subset.metadata.plans[0]
    path = config.output_dir / plan.filename
    if mutation == "ledger_index":
        record = manifest["files"][0]
        index = subset.tensor_indices[0]
        record["tensor_sha256"][index] = record["tensor_sha256"][0]
        manifest["manifest_sha256"] = _mapping_hash(manifest, field="manifest_sha256")
        (config.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        success = _seal(config.output_dir, manifest)
        common.update(
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_success_sha256=success["success_sha256"],
        )
        subset = _subset(config.output_dir, common, layer_ids=(1,))
    elif mutation == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(plan.file_bytes - 1)
    else:
        index = subset.tensor_indices[0] if mutation == "selected" else 0
        offset = (
            8
            if mutation == "header"
            else len(plan.header) + plan.tensors[index].data_offset_start
        )
        _flip(path, offset)
    if mutation == "unselected":
        assert len(list(iter_ws32_layer_subset_host_tensors(subset))) == 27
        with pytest.raises(CheckpointValidationError, match="checksum drifted"):
            verify_ws32_runtime_checkpoint(config.output_dir, **common)
    else:
        with pytest.raises(
            CheckpointValidationError,
            match="checksum drifted|header drifted|missing/truncated",
        ):
            list(iter_ws32_layer_subset_host_tensors(subset))


def test_sparse_local_layout_and_header_recheck(tmp_path: Path) -> None:
    config, _, common, _ = _fixture(tmp_path)
    root = tmp_path / "local"
    root.mkdir()
    owned = (9, 13, 25, 29)
    for name in (
        "manifest.json",
        "SUCCESS",
        *(f"device_slot_{slot:02d}.safetensors" for slot in owned),
    ):
        shutil.copyfile(config.output_dir / name, root / name)
    subset = _subset(root, common, local_slots=owned)
    assert {
        slot
        for _, _, owners in iter_ws32_layer_subset_host_tensors(subset)
        for slot, _, _ in owners
    } == set(owned)
    with pytest.raises(
        CheckpointValidationError, match="foreign slot|missing/truncated"
    ):
        _subset(root, common, local_slots=(8, 13, 25, 29))
    shutil.copyfile(
        config.output_dir / "device_slot_00.safetensors",
        root / "device_slot_00.safetensors",
    )
    with pytest.raises(CheckpointValidationError, match="foreign slot"):
        list(iter_ws32_layer_subset_host_tensors(subset))


def test_layer_one_does_not_include_layer_ten(tmp_path: Path) -> None:
    config, _, common, _ = _fixture(tmp_path, num_layers=11)
    subset = _subset(config.output_dir, common, layer_ids=(1,))
    assert len(subset.tensor_indices) == 27
    assert all(
        subset.metadata.plans[0].tensors[index].name.startswith("model.layers.1.")
        for index in subset.tensor_indices
    )


def test_layer_names_match_existing_decoder_all_real_branches() -> None:
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig,
        _weight_name_leaves,
        ws32_decoder_weight_names,
    )

    geometry = ModelGeometry.from_hf_config(
        json.loads((ROOT / "configs/glm-5.2-fp8-config.json").read_text())
    )
    names = ws32_decoder_weight_names(
        Ws32DecoderConfig(geometry=geometry, context_capacity=8192)
    )
    for layer in range(geometry.num_layers):
        assert ws32_expected_layer_names(geometry, layer) == frozenset(
            _weight_name_leaves(names.layers[layer])
        )


@pytest.mark.parametrize("dtype", ["BF16", "F32", "U8"])
def test_hash_correct_nonfinite_selected_payload_refuses(
    tmp_path: Path, dtype: str
) -> None:
    import struct

    config, manifest, common, _ = _fixture(tmp_path)
    subset = _subset(config.output_dir, common)
    plan = subset.metadata.plans[0]
    index = next(
        index for index in subset.tensor_indices if plan.tensors[index].dtype == dtype
    )
    tensor = plan.tensors[index]
    invalid = {
        "BF16": struct.pack("<H", 0x7FC0),
        "F32": struct.pack("<f", float("nan")),
        "U8": b"\x7f",
    }[dtype]
    with (config.output_dir / plan.filename).open("r+b") as stream:
        offset = len(plan.header) + tensor.data_offset_start
        stream.seek(offset)
        stream.write(invalid)
        stream.seek(offset)
        raw = stream.read(tensor.byte_count)
    manifest["files"][0]["tensor_sha256"][index] = sha256(raw).hexdigest()
    manifest["manifest_sha256"] = _mapping_hash(manifest, field="manifest_sha256")
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    success = _seal(config.output_dir, manifest)
    common.update(
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
    )
    subset = _subset(config.output_dir, common)
    with pytest.raises(CheckpointValidationError, match="non-finite"):
        list(iter_ws32_layer_subset_host_tensors(subset))


@pytest.mark.parametrize(
    "field",
    [
        "expected_manifest_sha256",
        "expected_success_sha256",
        "expected_topology_hash",
        "expected_mesh_hash",
    ],
)
def test_subset_authenticates_full_metadata_pins(tmp_path: Path, field: str) -> None:
    config, _, common, _ = _fixture(tmp_path)
    common[field] = "0" * 64
    with pytest.raises(CheckpointValidationError):
        _subset(config.output_dir, common)


def test_subset_metadata_cannot_enter_full_loader(tmp_path: Path) -> None:
    from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
        load_ws32_runtime_checkpoint,
    )

    config, _, common, _ = _fixture(tmp_path)
    subset = _subset(config.output_dir, common)
    for wrong in (subset, subset.metadata):
        with pytest.raises(
            CheckpointValidationError, match="requires full-runtime verification"
        ):
            load_ws32_runtime_checkpoint(wrong, mesh=None, physical_mesh=None)


@pytest.mark.parametrize("include_embedding", [False, True])
def test_direct_owner_cpu32_loader(tmp_path: Path, include_embedding: bool) -> None:
    config, manifest, common, _ = _fixture(tmp_path)
    program = f"""
import json
from dataclasses import replace
from pathlib import Path
import jax
import numpy as np
from jax.sharding import Mesh
from safetensors import safe_open
from glm_tpu.greenfield.checkpoint.ws32_layer_subset import read_ws32_layer_subset_metadata,load_ws32_layer_subset
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.partitioning import read_source_inventory
from glm_tpu.greenfield.sharding.ws32 import Ws32PhysicalMesh
from glm_tpu.greenfield.types import ModelGeometry
root=Path({str(config.output_dir)!r})
geometry=ModelGeometry.from_dict(json.loads((root/'manifest.json').read_text())['geometry'])
inventory=read_source_inventory(Path({str(config.source_root)!r}),model_id=geometry.model_id,source_revision='unit-fixture',config_filename=None)
ids=tuple(reversed(range(32)))
rows=tuple(ids[r*4:r*4+4] for r in range(8))
physical=Ws32PhysicalMesh(device_ids=rows,feature_groups=rows,expert_groups=tuple(tuple(row[c] for row in rows) for c in range(4)))
mesh=Mesh(np.asarray(list(reversed(jax.devices())),dtype=object).reshape(8,4),('expert','feature'))
subset=read_ws32_layer_subset_metadata(root,layer_ids=(1,),local_slots=tuple(range(32)),max_payload_bytes_per_chip=100000,expected_manifest_sha256={manifest['manifest_sha256']!r},expected_success_sha256={common['expected_success_sha256']!r},expected_mesh_hash=physical.mesh_hash,expected_topology_hash={('c'*64)!r},inventory=inventory,geometry=geometry,include_embedding={include_embedding!r})
for wrong_subset,wrong_mesh in [(replace(subset,local_slots=(0,1,2,3)),mesh),(subset,Mesh(np.asarray(jax.devices(),dtype=object).reshape(8,4),('expert','feature')))]:
    try:
        load_ws32_layer_subset(wrong_subset,mesh=wrong_mesh,physical_mesh=physical)
    except CheckpointValidationError:
        pass
    else:
        raise AssertionError('accepted wrong ownership')
loaded=load_ws32_layer_subset(subset,mesh=mesh,physical_mesh=physical)
assert len(loaded.arrays)=={27 + int(include_embedding)}
assert loaded.include_embedding is {include_embedding!r}
assert loaded.integrity_scope=={('selected_layers_and_embedding_only_not_complete_checkpoint' if include_embedding else 'selected_layer_tensors_only_not_complete_checkpoint')!r}
slot_by_id={{device_id:slot for slot,device_id in enumerate(ids)}}
for name,array in loaded.arrays.items():
    for shard in array.addressable_shards:
        slot=slot_by_id[shard.device.id]
        with safe_open(root/f'device_slot_{{slot:02d}}.safetensors',framework='numpy') as handle:
            expected=handle.get_tensor(name)
        assert np.array_equal(np.asarray(shard.data),expected), (name,slot)
for record in loaded.local_device_slots:
    assert record['device_slot']==slot_by_id[record['device_id']]
    assert len(record['observed_selected_tensor_sha256'])=={27 + int(include_embedding)}
    assert 'file_sha256' not in record
print('SUBSET_CPU32_PASS')
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=32",
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SUBSET_CPU32_PASS" in result.stdout
