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

from glm_tpu.model_loader.sharded_state.writer import (
    finalize_runtime_checkpoint,
    pack_runtime_checkpoint,
    pack_runtime_slots,
)
from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.distributed.mesh import PhysicalMesh
from tests.fixtures import tiny_checkpoint
from tests.fixtures.site import example_site, installed_site


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _example_site(tmp_path_factory):
    """The checkpoint format admits source URIs under the current site's approved buckets."""
    with installed_site(example_site(tmp_path_factory.mktemp("site"))) as site:
        yield site


def test_ws32_runtime_packs_and_verifies_exact_32_final_owners(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    embedding, inventory, config = tiny_checkpoint.fixture(tmp_path)
    manifest = pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry(), chunk_bytes=16)
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
            geometry=tiny_checkpoint.geometry(),
        )
    success = tiny_checkpoint.seal(config.output_dir, manifest)
    verified = verify_runtime_checkpoint(
        config.output_dir,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash="b" * 64,
        expected_topology_hash="c" * 64,
        inventory=inventory,
        geometry=tiny_checkpoint.geometry(),
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
        geometry=tiny_checkpoint.geometry(),
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
            geometry=tiny_checkpoint.geometry(),
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
            geometry=tiny_checkpoint.geometry(),
            verify_file_hash_slots=(0, 0),
        )


def test_ws32_runtime_validation_failure_never_commits_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from glm_tpu.model_loader.sharded_state import writer as module

    _, inventory, config = tiny_checkpoint.fixture(tmp_path)

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise CheckpointValidationError("injected final validation failure")

    monkeypatch.setattr(module, "verify_runtime_value", fail)
    with pytest.raises(CheckpointValidationError, match="injected"):
        pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry(), chunk_bytes=16)
    assert not (config.output_dir / "manifest.json").exists()


def test_ws32_runtime_slot_pack_is_disjoint_and_nonterminal(
    tmp_path: Path,
) -> None:
    _, inventory, config = tiny_checkpoint.fixture(tmp_path)
    records = pack_runtime_slots(
        config,
        inventory,
        tiny_checkpoint.geometry(),
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
    _, inventory, config = tiny_checkpoint.fixture(tmp_path)
    records = pack_runtime_slots(
        config,
        inventory,
        tiny_checkpoint.geometry(),
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
        tiny_checkpoint.geometry(),
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
            tiny_checkpoint.geometry(),
            file_records=records["files"],
            source_file_sha256=source_sha256,
        )


@pytest.mark.cpu32
def test_ws32_runtime_loader_uses_only_exact_final_owner_shards(
    tmp_path: Path,
) -> None:
    _, inventory, config = tiny_checkpoint.fixture(tmp_path)
    rows = tuple(tuple(range(row * 4, row * 4 + 4)) for row in range(8))
    physical = PhysicalMesh(
        device_ids=rows,
        feature_groups=rows,
        expert_groups=tuple(tuple(row * 4 + column for row in range(8)) for column in range(4)),
    )
    config = replace(config, mesh_hash=physical.mesh_hash)
    manifest = pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry(), chunk_bytes=16)
    success = tiny_checkpoint.seal(config.output_dir, manifest)
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

    _embedding, inventory, config = tiny_checkpoint.fixture(tmp_path)
    manifest = pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry())
    success = tiny_checkpoint.seal(config.output_dir, manifest)
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
        geometry=tiny_checkpoint.geometry(),
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
