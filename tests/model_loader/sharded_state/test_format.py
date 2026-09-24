"""Tests of :mod:`glm_tpu.model_loader.sharded_state.format`."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.model_loader.sharded_state.format import RuntimePackConfig
from glm_tpu.model_loader.sharded_state.writer import (
    finalize_runtime_checkpoint,
    pack_runtime_checkpoint,
    pack_runtime_slots,
)
from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader.source_inventory import read_source_inventory
from glm_tpu.config.model import ModelGeometry
from glm_tpu.distributed.mesh import PhysicalMesh
from tests.fixtures.site import EXAMPLE_BUCKET, example_site, installed_site

# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (tests/config/test_model.py).
from tools.equivalence.fixture import config_json


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _example_site(tmp_path_factory):
    """The checkpoint format admits source URIs under the current site's approved buckets."""
    with installed_site(example_site(tmp_path_factory.mktemp("site"))) as site:
        yield site


def _geometry() -> ModelGeometry:
    full = ModelGeometry.from_hf_config(config_json())
    return replace(
        full,
        num_layers=1,
        first_dense_layers=1,
        hidden_size=8,
        dense_intermediate_size=16,
        num_routed_experts=8,
        routed_top_k=2,
        moe_intermediate_size=4,
        dsa_top_k=8,
        dsa_indexer_heads=8,
        dsa_indexer_head_dim=2,
        index_share_group_size=1,
        attention_heads=8,
        kv_heads=8,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=64,
        vocab_size=16,
        fp8_block_shape=(2, 2),
        mlp_layer_types=("dense",),
        indexer_types=("full",),
    )


def _fixture(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    filename = "model.safetensors"
    embedding = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8).to(torch.bfloat16)
    save_file({"model.embed_tokens.weight": embedding}, source / filename)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": embedding.numel() * embedding.element_size()},
                "weight_map": {"model.embed_tokens.weight": filename},
            }
        )
    )
    inventory = read_source_inventory(
        source,
        model_id="zai-org/GLM-5.2-FP8",
        source_revision="unit-fixture",
        config_filename=None,
    )
    output = tmp_path / "packed"
    config = RuntimePackConfig(
        source_root=source,
        source_uri=EXAMPLE_BUCKET + "models/unit-fixture",
        output_dir=output,
        code_hash="a" * 40,
        mesh_hash="b" * 64,
    )
    return embedding, inventory, config


def _seal(
    root: Path,
    manifest: dict[str, object],
    *,
    topology_hash: str = "c" * 64,
) -> dict[str, object]:
    source = manifest["source"]
    assert isinstance(source, dict)
    source_files = source["files"]
    assert isinstance(source_files, list)
    value: dict[str, object] = {
        "artifact_kind": "greenfield_ws32_runtime_checkpoint_success",
        "code_hash": manifest["code_hash"],
        "file_count": 32,
        "format_version": 1,
        "manifest_file_sha256": sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "mesh_hash": manifest["mesh_hash"],
        "packed_payload_bytes": manifest["packed_payload_bytes"],
        "performance_claim": False,
        "post_census_sha256": "d" * 64,
        "remote_preflight_sha256": "e" * 64,
        "remote_terminal_sha256": "f" * 64,
        "source_file_count": len(source_files),
        "source_inventory_sha256": source["inventory_sha256"],
        "tag": "greenfield_ws32_runtime_pack_20260815T010203123456789Z",
        "topology_hash": topology_hash,
        "tpu_initialized": False,
    }
    value["success_sha256"] = sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (root / "SUCCESS").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def test_ws32_runtime_packs_and_verifies_exact_32_final_owners(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    embedding, inventory, config = _fixture(tmp_path)
    manifest = pack_runtime_checkpoint(config, inventory, _geometry(), chunk_bytes=16)
    assert manifest["packed_payload_bytes"] == embedding.numel() * 2
    assert len(manifest["files"]) == 32
    assert len(manifest["tensor_schema"]) == 1
    assert not tuple(config.output_dir.glob("*.partial"))

    for slot in range(32):
        expert, feature = divmod(slot, 4)
        with safe_open(
            config.output_dir / f"device_slot_{slot:02d}.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            actual = handle.get_tensor("model.embed_tokens.weight")
        expected = embedding[
            expert * 2 : (expert + 1) * 2,
            feature * 2 : (feature + 1) * 2,
        ]
        assert torch.equal(actual, expected)

    with pytest.raises(CheckpointValidationError, match="lacks SUCCESS"):
        verify_runtime_checkpoint(
            config.output_dir,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_success_sha256="d" * 64,
            expected_mesh_hash="b" * 64,
            expected_topology_hash="c" * 64,
            inventory=inventory,
            geometry=_geometry(),
        )
    success = _seal(config.output_dir, manifest)
    verified = verify_runtime_checkpoint(
        config.output_dir,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash="b" * 64,
        expected_topology_hash="c" * 64,
        inventory=inventory,
        geometry=_geometry(),
    )
    assert len(verified.plans) == 32
    assert verified.records_by_slot[31]["feature_coordinate"] == 3

    path = config.output_dir / "device_slot_31.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes((value[0] ^ 1,)))
    # A launch rank hashes only its four final-owner files.  All structural
    # records are still checked, while the fleet sealer provides 0..31 hash
    # coverage across the eight disjoint rank subsets.
    verify_runtime_checkpoint(
        config.output_dir,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash="b" * 64,
        expected_topology_hash="c" * 64,
        inventory=inventory,
        geometry=_geometry(),
        verify_file_hash_slots=(0, 1, 2, 3),
    )
    with pytest.raises(CheckpointValidationError, match="checksum drifted"):
        verify_runtime_checkpoint(
            config.output_dir,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_success_sha256=success["success_sha256"],
            expected_mesh_hash="b" * 64,
            expected_topology_hash="c" * 64,
            inventory=inventory,
            geometry=_geometry(),
            verify_file_hash_slots=(31,),
        )
    with pytest.raises(ValueError, match="slot subset"):
        verify_runtime_checkpoint(
            config.output_dir,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_success_sha256=success["success_sha256"],
            expected_mesh_hash="b" * 64,
            expected_topology_hash="c" * 64,
            inventory=inventory,
            geometry=_geometry(),
            verify_file_hash_slots=(0, 0),
        )


def test_ws32_runtime_validation_failure_never_commits_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from glm_tpu.model_loader.sharded_state import writer as module

    _, inventory, config = _fixture(tmp_path)

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise CheckpointValidationError("injected final validation failure")

    monkeypatch.setattr(module, "_verify_runtime_value", fail)
    with pytest.raises(CheckpointValidationError, match="injected"):
        pack_runtime_checkpoint(config, inventory, _geometry(), chunk_bytes=16)
    assert not (config.output_dir / "manifest.json").exists()


def test_ws32_runtime_slot_pack_is_disjoint_and_nonterminal(
    tmp_path: Path,
) -> None:
    _, inventory, config = _fixture(tmp_path)
    records = pack_runtime_slots(
        config,
        inventory,
        _geometry(),
        device_slots=(0, 5, 31),
        chunk_bytes=16,
    )
    assert records["slots"] == [0, 5, 31]
    assert [record["device_slot"] for record in records["files"]] == [0, 5, 31]
    assert (config.output_dir / "slot_records.json").is_file()
    assert not (config.output_dir / "manifest.json").exists()
    assert {path.name for path in config.output_dir.glob("device_slot_*.safetensors")} == {
        "device_slot_00.safetensors",
        "device_slot_05.safetensors",
        "device_slot_31.safetensors",
    }


def test_ws32_runtime_distributed_records_finalize_once(tmp_path: Path) -> None:
    _, inventory, config = _fixture(tmp_path)
    records = pack_runtime_slots(
        config,
        inventory,
        _geometry(),
        device_slots=tuple(range(32)),
        chunk_bytes=16,
    )
    source_sha256 = {
        record.filename: sha256((config.source_root / record.filename).read_bytes()).hexdigest()
        for record in inventory.files
    }
    manifest = finalize_runtime_checkpoint(
        config,
        inventory,
        _geometry(),
        file_records=records["files"],
        source_file_sha256=source_sha256,
    )
    assert (
        manifest["manifest_sha256"] == json.loads((config.output_dir / "manifest.json").read_text())["manifest_sha256"]
    )
    with pytest.raises(FileExistsError, match="manifest exists"):
        finalize_runtime_checkpoint(
            config,
            inventory,
            _geometry(),
            file_records=records["files"],
            source_file_sha256=source_sha256,
        )


def test_ws32_runtime_loader_uses_only_exact_final_owner_shards(
    tmp_path: Path,
) -> None:
    _, inventory, config = _fixture(tmp_path)
    rows = tuple(tuple(range(row * 4, row * 4 + 4)) for row in range(8))
    physical = PhysicalMesh(
        device_ids=rows,
        feature_groups=rows,
        expert_groups=tuple(tuple(row * 4 + column for row in range(8)) for column in range(4)),
    )
    config = replace(config, mesh_hash=physical.mesh_hash)
    manifest = pack_runtime_checkpoint(config, inventory, _geometry(), chunk_bytes=16)
    success = _seal(config.output_dir, manifest)
    program = f"""\
import json
from pathlib import Path
import jax
import numpy as np
from jax.sharding import Mesh
from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
from glm_tpu.model_loader.sharded_state.loader import load_runtime_checkpoint
from glm_tpu.model_loader.source_inventory import read_source_inventory
from glm_tpu.distributed.mesh import PhysicalMesh
from glm_tpu.config.model import ModelGeometry
from glm_tpu.config.site import set_current_site
from tests.fixtures.site import example_site
set_current_site(example_site(Path({str(config.output_dir.parent / "site")!r})))
root=Path({str(config.output_dir)!r})
source=Path({str(config.source_root)!r})
manifest=json.loads((root/'manifest.json').read_text())
geometry=ModelGeometry.from_dict(manifest['geometry'])
inventory=read_source_inventory(source,model_id=geometry.model_id,source_revision='unit-fixture',config_filename=None)
verified=verify_runtime_checkpoint(root,expected_manifest_sha256={manifest["manifest_sha256"]!r},expected_success_sha256={success["success_sha256"]!r},expected_mesh_hash={physical.mesh_hash!r},expected_topology_hash={("c" * 64)!r},inventory=inventory,geometry=geometry)
rows=tuple(tuple(range(row*4,row*4+4)) for row in range(8))
physical=PhysicalMesh(device_ids=rows,feature_groups=rows,expert_groups=tuple(tuple(row*4+column for row in range(8)) for column in range(4)))
mesh=Mesh(np.asarray(jax.devices(),dtype=object).reshape(8,4),('expert','feature'))
loaded=load_runtime_checkpoint(verified,mesh=mesh,physical_mesh=physical)
observed=np.asarray(jax.device_get(loaded.arrays['model.embed_tokens.weight']))
expected=np.arange(16*8,dtype=np.float32).reshape(16,8).astype(jax.numpy.bfloat16)
print(json.dumps({{'content_exact':bool(np.array_equal(observed,expected)),'shape':list(observed.shape),'spec':str(loaded.arrays['model.embed_tokens.weight'].sharding.spec),'slots':len(loaded.local_device_slots)}}))
"""  # noqa: E501 (child program text)
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "content_exact": True,
        "shape": [16, 8],
        "spec": "P('expert', 'feature')",
        "slots": 32,
    }


def test_ws32_runtime_local_slot_layout_verifies_only_owned_slots(tmp_path: Path) -> None:
    """Streaming tmpfs layout (spec §21.5 route): a host root holds only its four slots."""
    import shutil

    _embedding, inventory, config = _fixture(tmp_path)
    manifest = pack_runtime_checkpoint(config, inventory, _geometry())
    success = _seal(config.output_dir, manifest)
    owned = (8, 12, 24, 28)
    local_root = tmp_path / "shm-root"
    local_root.mkdir()
    for name in ("manifest.json", "SUCCESS"):
        shutil.copy(config.output_dir / name, local_root / name)
    for slot in owned:
        shutil.copy(
            config.output_dir / f"device_slot_{slot:02d}.safetensors",
            local_root / f"device_slot_{slot:02d}.safetensors",
        )
    common = dict(
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash="b" * 64,
        expected_topology_hash="c" * 64,
        inventory=inventory,
        geometry=_geometry(),
    )
    # The default (full) layout refuses a four-slot root.
    with pytest.raises(CheckpointValidationError, match="missing or truncated"):
        verify_runtime_checkpoint(local_root, verify_file_hash_slots=owned, **common)
    verified = verify_runtime_checkpoint(local_root, verify_file_hash_slots=owned, local_slot_layout=True, **common)
    assert len(verified.plans) == 32 and set(verified.records_by_slot) == set(range(32))
    # Local layout still requires owned slots to be present, exact and hash-verified.
    with pytest.raises(ValueError, match="requires hash verification"):
        verify_runtime_checkpoint(local_root, local_slot_layout=True, **common)
    with pytest.raises(ValueError, match="requires hash verification"):
        verify_runtime_checkpoint(
            local_root, verify_file_hashes=False, verify_file_hash_slots=owned, local_slot_layout=True, **common
        )
    # Wrong ownership: slot 28 is present but not owned (foreign) and slot 29 is absent.
    with pytest.raises(CheckpointValidationError, match=r"foreign slot|missing or truncated"):
        verify_runtime_checkpoint(local_root, verify_file_hash_slots=(8, 12, 24, 29), local_slot_layout=True, **common)
    corrupt = local_root / "device_slot_12.safetensors"
    with corrupt.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(CheckpointValidationError, match="checksum drifted"):
        verify_runtime_checkpoint(local_root, verify_file_hash_slots=owned, local_slot_layout=True, **common)
    shutil.copy(config.output_dir / "device_slot_12.safetensors", corrupt)
    # A foreign slot file present in a local layout root is refused (no mixed layouts).
    shutil.copy(config.output_dir / "device_slot_00.safetensors", local_root / "device_slot_00.safetensors")
    with pytest.raises(CheckpointValidationError, match="foreign slot"):
        verify_runtime_checkpoint(local_root, verify_file_hash_slots=owned, local_slot_layout=True, **common)
