"""Tests of :mod:`glm_tpu.model_loader.sharded_state.seal` on the tiny checkpoint (tests/fixtures/tiny_checkpoint.py):
the SUCCESS record is the one ``verify_runtime_checkpoint`` admits, the receipt and plan comparisons find every
difference, the canary re-derives exactly the packed tensor bytes, and the seal is installed once."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.model_loader.sharded_state import seal
from glm_tpu.model_loader.sharded_state.format import build_runtime_file_plans
from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
from glm_tpu.model_loader.sharded_state.writer import pack_runtime_checkpoint, pack_runtime_slots
from tests.fixtures import tiny_checkpoint
from tests.fixtures.site import example_site, installed_site

TAG = "greenfield_ws32_runtime_pack_" + "20000101T" + "0" * 15 + "Z"  # a synthetic pack run name
TOPOLOGY = "c" * 64
DIGESTS = dict(post_census_sha256="d" * 64, remote_preflight_sha256="e" * 64, remote_terminal_sha256="f" * 64)
HOST_TO_SLOTS = {str(rank): [rank + 8 * k for k in range(4)] for rank in range(8)}


@pytest.fixture
def packed(tmp_path: Path):
    """The tiny checkpoint packed whole (manifest, no SUCCESS) and packed again host by host (the eight receipts)."""
    with installed_site(example_site(tmp_path / "site")):
        _, inventory, config = tiny_checkpoint.fixture(tmp_path)
        manifest = pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry(), chunk_bytes=16)
        receipts = []
        for rank, slots in HOST_TO_SLOTS.items():
            host = replace(config, output_dir=tmp_path / f"host{rank}")
            receipts.append(pack_runtime_slots(host, inventory, tiny_checkpoint.geometry(), device_slots=slots))
        yield config, inventory, manifest, receipts


def test_success_record_is_what_verify_admits(packed, tmp_path: Path):
    config, inventory, manifest, _ = packed
    raw = (config.output_dir / "manifest.json").read_bytes()
    assert seal.json_bytes(manifest) == raw  # the manifest as the packer writes it
    value = seal.success_record(manifest, raw, tag=TAG, topology_hash=TOPOLOGY, **DIGESTS)
    (config.output_dir / "SUCCESS").write_bytes(seal.json_bytes(value))
    # the harness's synthetic seal is the same record with its own tag
    other = tiny_checkpoint.seal(config.output_dir, manifest, topology_hash=TOPOLOGY)
    assert {k: v for k, v in value.items() if k not in ("tag", "success_sha256")} == {
        k: v for k, v in other.items() if k not in ("tag", "success_sha256")
    }
    (config.output_dir / "SUCCESS").write_bytes(seal.json_bytes(value))
    with installed_site(example_site(tmp_path / "site")):
        verified = verify_runtime_checkpoint(
            config.output_dir,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_success_sha256=value["success_sha256"],
            expected_mesh_hash=manifest["mesh_hash"],
            expected_topology_hash=TOPOLOGY,
            inventory=inventory,
            geometry=tiny_checkpoint.geometry(),
        )
    assert verified.success == value


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(tag="pack-1"), "tag"),
        (dict(topology_hash="C" * 64), "topology_hash"),
        (dict(post_census_sha256="0"), "post_census_sha256"),
    ],
)
def test_success_record_refuses(packed, change, message):
    config, _, manifest, _ = packed
    raw = (config.output_dir / "manifest.json").read_bytes()
    options = dict(dict(tag=TAG, topology_hash=TOPOLOGY, **DIGESTS), **change)
    with pytest.raises(ValueError, match=message):
        seal.success_record(manifest, raw, **options)
    with pytest.raises(ValueError, match="not this manifest"):
        seal.success_record(dict(manifest, code_hash="b" * 40), raw, tag=TAG, topology_hash=TOPOLOGY, **DIGESTS)


def test_the_hosts_receipts_equal_the_whole_pack(packed):
    _, _, manifest, receipts = packed
    assert seal.compare_records(manifest, receipts, HOST_TO_SLOTS) == []


def test_compare_records_names_every_difference(packed):
    _, _, manifest, receipts = packed
    edited = [dict(r) for r in receipts]
    edited[2] = dict(edited[2], files=[dict(edited[2]["files"][0], sha256="0" * 64), *edited[2]["files"][1:]])
    edited[5] = dict(edited[5], mesh_hash="0" * 64)
    problems = seal.compare_records(manifest, edited, HOST_TO_SLOTS)
    assert problems == [f"slot {edited[2]['files'][0]['device_slot']}: ['sha256'] differ", "rank 5: mesh_hash differs"]
    swapped = dict(HOST_TO_SLOTS, **{"0": HOST_TO_SLOTS["1"], "1": HOST_TO_SLOTS["0"]})
    assert "rank 0: slots" in seal.compare_records(manifest, receipts, swapped)[0]
    missing = seal.compare_records(manifest, receipts[:7], HOST_TO_SLOTS)
    assert missing[0] == "7 owner receipts for 8 hosts" and "the receipts cover slots" in missing[-1]


def test_compare_plans(packed):
    _, inventory, manifest, _ = packed
    _, plans = build_runtime_file_plans(inventory, tiny_checkpoint.geometry(), mesh_hash=manifest["mesh_hash"])
    assert seal.compare_plans(plans, manifest) == []
    files = [dict(manifest["files"][0], header_sha256="0" * 64), *manifest["files"][1:]]
    assert seal.compare_plans(plans, dict(manifest, files=files)) == ["slot 0: ['header_sha256'] differ"]


def test_rederived_tensors_equal_the_packed_ones(packed):
    config, inventory, manifest, _ = packed
    _, plans = build_runtime_file_plans(inventory, tiny_checkpoint.geometry(), mesh_hash=manifest["mesh_hash"])
    for plan in (plans[0], plans[13], plans[31]):
        indices = seal.canary_tensors(plan, 0)
        derived = seal.rederive_tensors(config.source_root, inventory, tiny_checkpoint.geometry(), plan, indices)
        record = manifest["files"][plan.device_slot]
        assert [derived[i]["sha256"] for i in indices] == record["tensor_sha256"]
        assert all(derived[i]["covered"] for i in indices)


def test_a_changed_source_byte_is_seen_by_the_canary(packed):
    config, inventory, manifest, _ = packed
    _, plans = build_runtime_file_plans(inventory, tiny_checkpoint.geometry(), mesh_hash=manifest["mesh_hash"])
    path = config.source_root / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1  # the last element of the embedding: in the last rows' owner
    path.write_bytes(bytes(data))
    changed = []
    for plan in plans:
        derived = seal.rederive_tensors(config.source_root, inventory, tiny_checkpoint.geometry(), plan, [0])
        changed += (
            [plan.device_slot]
            if derived[0]["sha256"] != manifest["files"][plan.device_slot]["tensor_sha256"][0]
            else []
        )
    assert changed and len(changed) < 32


def test_canary_tensors_picks_distinct_indices(packed):
    _, inventory, manifest, _ = packed
    _, plans = build_runtime_file_plans(inventory, tiny_checkpoint.geometry(), mesh_hash=manifest["mesh_hash"])
    plan = plans[0]
    assert seal.canary_tensors(plan, 0) == list(range(len(plan.tensors)))
    picked = seal.canary_tensors(plan, 1)
    assert len(picked) == 1 and len(set(picked)) == 1
    with pytest.raises(ValueError):
        seal.canary_tensors(plan, -1)


def test_install_seal_writes_once(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    seal.install_seal(root, b"{}\n", b'{"a": 1}\n')
    assert (root / "manifest.json").read_bytes() == b"{}\n" and (root / "SUCCESS").read_bytes() == b'{"a": 1}\n'
    assert oct((root / "SUCCESS").stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError):
        seal.install_seal(root, b"[]\n", b"[]\n")
    assert (root / "manifest.json").read_bytes() == b"{}\n"


def test_json_bytes_is_the_indented_sorted_form():
    assert seal.json_bytes({"b": 1, "a": [2]}) == b'{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'
    assert sha256(seal.json_bytes({})).hexdigest() == sha256(b"{}\n").hexdigest()
    assert json.loads(seal.json_bytes({"x": 1})) == {"x": 1}
