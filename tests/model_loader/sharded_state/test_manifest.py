"""Real tiny owner files: distributed assembly must match the retained packer."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from glm_tpu.model_loader.sharded_state.manifest import assemble_owner_manifest
from glm_tpu.model_loader.sharded_state import format as retained
from glm_tpu.model_loader.sharded_state import writer
from glm_tpu.model_loader.sharded_state import verify
from tests.fixtures import tiny_checkpoint
from tests.fixtures.site import example_site, installed_site


@pytest.fixture(autouse=True)
def _example_site(tmp_path_factory):
    """Packing and verification admit source URIs under the current site's approved buckets."""
    with installed_site(example_site(tmp_path_factory.mktemp("site"))) as site:
        yield site


@pytest.fixture
def owners(tmp_path):
    _, inventory, config = tiny_checkpoint.fixture(tmp_path)
    geometry = tiny_checkpoint.geometry()
    slots = {str(rank): list(range(rank * 4, rank * 4 + 4)) for rank in range(8)}
    records = [
        writer.pack_runtime_slots(
            replace(config, output_dir=tmp_path / f"host{rank}"), inventory, geometry, device_slots=slots[str(rank)]
        )
        for rank in range(8)
    ]
    hashes = {f.filename: sha256((config.source_root / f.filename).read_bytes()).hexdigest() for f in inventory.files}
    kwargs = dict(
        inventory=inventory,
        geometry=geometry,
        code_hash=config.code_hash,
        mesh_hash=config.mesh_hash,
        source_uri=config.source_uri,
        owner_records=records,
        host_to_slots=slots,
        source_file_sha256=hashes,
    )
    return kwargs, config


def test_distributed_manifest_matches_original_and_loads_owned_files(owners, tmp_path):
    import json

    kwargs, config = owners
    actual = assemble_owner_manifest(**kwargs)
    expected = writer.pack_runtime_checkpoint(config, kwargs["inventory"], kwargs["geometry"])
    assert actual == expected
    for rank in range(8):
        root = tmp_path / f"host{rank}"
        (root / "manifest.json").write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        success = tiny_checkpoint.seal(root, actual)
        verified = verify.verify_runtime_checkpoint(
            root,
            expected_manifest_sha256=actual["manifest_sha256"],
            expected_success_sha256=success["success_sha256"],
            expected_mesh_hash=config.mesh_hash,
            expected_topology_hash="c" * 64,
            inventory=kwargs["inventory"],
            geometry=kwargs["geometry"],
            verify_file_hashes=True,
            verify_file_hash_slots=kwargs["host_to_slots"][str(rank)],
            local_slot_layout=True,
        )
        assert verified.manifest == actual


@pytest.mark.parametrize("mutation", ["reorder", "duplicate", "checksum", "file_size", "source_missing"])
def test_incomplete_or_corrupted_owner_evidence_refused(owners, mutation):
    kwargs, _ = owners
    kwargs = dict(kwargs, owner_records=deepcopy(kwargs["owner_records"]))
    records = kwargs["owner_records"]
    if mutation == "reorder":
        records[0], records[1] = records[1], records[0]
    elif mutation == "duplicate":
        records[7] = records[0]
    elif mutation == "checksum":
        records[0]["record_sha256"] = "0" * 64
    elif mutation == "file_size":
        records[0]["files"][0]["file_bytes"] += 1
        records[0]["record_sha256"] = retained.mapping_hash(records[0], field="record_sha256")
    else:
        kwargs["source_file_sha256"] = {}
    with pytest.raises(ValueError):
        assemble_owner_manifest(**kwargs)


def test_pack_worker_stays_off_without_protected_cpu_invocation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from glm_tpu.engine.resident_protocol import PACK_WORKER_ENV_FLAG
    from glm_tpu.model_loader.pack_worker import preflight

    monkeypatch.delenv(PACK_WORKER_ENV_FLAG, raising=False)
    with pytest.raises(ValueError, match="protected CPU"):
        preflight(SimpleNamespace(output=tmp_path, code_hash="a" * 40))


def test_cleanup_authenticates_selected_worker_module(tmp_path, monkeypatch):
    import json
    import shlex
    from glm_tpu.executor import fleet as remote
    from glm_tpu.executor import multihost_executor as fleet
    from tests.fixtures.site import example_site

    captured = []
    monkeypatch.setattr(fleet, "remote_all", lambda *a, **k: captured.append(a[1]))
    hosts = [f"example-w-{rank}" for rank in range(8)]
    fleet.cleanup_owned(
        [],
        tmp_path,
        "a" * 40,
        hosts=hosts,
        fleet=example_site(tmp_path).fleet,
        helpers=remote.HelperTexts.from_package(),
        module="glm_tpu.model_loader.pack_worker",
    )
    words = shlex.split(captured[0])
    assert words[:2] == ["python3", "-c"] and len(words) == 4
    code = words[2]
    assert code == remote.helper_text("cleanup")  # the helper file itself, never templated
    assert json.loads(words[3]) == dict(
        root=str(tmp_path), hosts=hosts, pin="a" * 40, module="glm_tpu.model_loader.pack_worker"
    )
    assert "module.encode() not in command" in code
    assert "os.pidfd_open(pid)" in code
    assert 'fields[19] != owner["start_ticks"]' in code
    compile(code, "cleanup", "exec")
