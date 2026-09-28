"""Tests of the capture and binding side of :mod:`glm_tpu.distributed.topology`: the device inventory of the live
devices, the one slice shape the engine runs, and the binding derived from eight captures, which the runtime's own
``load_topology_binding`` must accept."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.distributed.mesh import build_physical_mesh
from glm_tpu.distributed.topology import (
    CAPTURE_NAMES,
    binding_bytes,
    derive_topology_binding,
    discover_topology,
    load_topology_binding,
    require_v4_64,
)
from glm_tpu.exceptions import TopologyValidationError
from tests.fixtures import topology as fleet
from tests.fixtures.site import example_site

CODE_HASH = "a" * 40


@pytest.fixture
def captures(tmp_path: Path) -> list[bytes]:
    return fleet.capture_all(tmp_path, example_site(tmp_path), CODE_HASH)


def staged(root: Path, binding: dict, captures: list[bytes]) -> tuple[Path, str]:
    """The binding as a run directory holds it (topology_rebinding.json beside topology_capture/) and its pin."""
    (root / "topology_capture").mkdir(parents=True)
    raw = binding_bytes(binding)
    (root / "topology_rebinding.json").write_bytes(raw)
    for name, data in zip(CAPTURE_NAMES, captures, strict=True):
        (root / "topology_capture" / name).write_bytes(data)
    return root, sha256(raw).hexdigest()


def test_discover_topology_records_what_the_devices_report():
    topology = discover_topology(fleet.devices(), slice_name="example-vm", local_device_ids=fleet.local_order())
    assert topology.topology_shape == (2, 4, 4) and len(topology.devices) == 32
    first = topology.devices[0].to_dict()
    assert first == dict(
        coordinates=[0, 0, 0],
        core_on_chip=0,
        device_id=0,
        device_kind="TPU v4",
        local_device_id=0,
        platform="tpu",
        process_index=0,
    )
    require_v4_64(topology)
    assert build_physical_mesh(topology).flattened_device_ids  # the mesh the runtime builds from it


@pytest.mark.parametrize(
    "devices,order,message",
    [
        ([], {}, "no devices"),
        (fleet.devices(), {}, "no observed local id"),
        ([fleet.device(0, local_hardware_id=3), *fleet.devices()[1:]], None, "disagrees with the runtime"),
        ([fleet.device(d, coords=(d % 2 + 1, (d // 2) % 4, d // 8)) for d in range(32)], None, "zero-based"),
        ([fleet.device(0, coords=(0, 0)), *fleet.devices()[1:]], None, "different dimensionality"),
    ],
    ids=["no devices", "no local order", "local id disagrees", "not zero-based", "dimensionality"],
)
def test_discover_topology_refuses_what_it_cannot_prove(devices, order, message):
    with pytest.raises(TopologyValidationError, match=message):
        discover_topology(
            devices, slice_name="example-vm", local_device_ids=fleet.local_order() if order is None else order
        )


@pytest.mark.parametrize(
    "change",
    [dict(device_kind="TPU v5"), dict(platform="gpu")],
    ids=["kind", "platform"],
)
def test_require_v4_64_refuses_another_slice(change):
    devices = [fleet.device(d, **change) for d in range(32)]
    topology = discover_topology(devices, slice_name="example-vm", local_device_ids=fleet.local_order())
    with pytest.raises(TopologyValidationError, match="not 32 TPU v4 chips"):
        require_v4_64(topology)


@pytest.mark.parametrize(
    "coords",
    [lambda d: (d % 4, (d // 4) % 4, d // 16), lambda d: (d % 8, d // 8, 0)],
    ids=["4x4x2 (a permutation the mesh does not map)", "8x4x1"],
)
def test_require_v4_64_refuses_another_shape(coords):
    devices = [fleet.device(d, coords=coords(d)) for d in range(32)]
    topology = discover_topology(devices, slice_name="example-vm", local_device_ids=fleet.local_order())
    with pytest.raises(TopologyValidationError, match="2x4x4"):
        require_v4_64(topology)


def test_a_first_binding_is_accepted_by_the_runtime(captures: list[bytes], tmp_path: Path):
    binding = derive_topology_binding(captures, all_hosts_idle_after=True, note="unit test")
    root, pin = staged(tmp_path / "run", binding, captures)
    identity = load_topology_binding(
        root,
        pin,
        expected_topology=binding["original_topology_sha256"],
        expected_mesh=binding["mesh_sha256"],
        original_fleet=binding["fleet_sha256"],
        slice_name="example-vm",
    )
    assert identity["hosts"] == fleet.HOSTS
    assert identity["jax_process_indices"] == list(fleet.PERMUTATION)
    assert identity["host_to_slots"] == binding["host_to_slots"]
    assert sorted(s for slots in binding["host_to_slots"].values() for s in slots) == list(range(32))
    assert binding["original_fleet_sha256"] == binding["fleet_sha256"]  # a first binding reassigns its own fleet
    assert binding["code_hash"] == CODE_HASH and binding["derived_by"] == "unit test"
    assert binding["launch_to_jax_process"] == {str(r): p for r, p in enumerate(fleet.PERMUTATION)}
    assert binding["capture_sha256"] == {n: sha256(c).hexdigest() for n, c in zip(CAPTURE_NAMES, captures, strict=True)}
    assert binding_bytes(binding) == (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode()


def test_a_rebinding_keeps_the_original_fleet_and_checks_the_expected_identity(captures: list[bytes], tmp_path: Path):
    first = derive_topology_binding(captures, all_hosts_idle_after=True, note="first")
    original = "e" * 64
    binding = derive_topology_binding(
        captures,
        all_hosts_idle_after=True,
        note="again",
        original_fleet_sha256=original,
        expected_topology_sha256=first["original_topology_sha256"],
        expected_mesh_sha256=first["mesh_sha256"],
        slice_name="example-vm",
    )
    assert binding["original_fleet_sha256"] == original and binding["fleet_sha256"] == first["fleet_sha256"]
    root, pin = staged(tmp_path / "run", binding, captures)
    load_topology_binding(
        root,
        pin,
        expected_topology=first["original_topology_sha256"],
        expected_mesh=first["mesh_sha256"],
        original_fleet=original,
        slice_name="example-vm",
    )


def _edit(captures: list[bytes], rank: int, **changes) -> list[bytes]:
    edited = list(captures)
    edited[rank] = json.dumps(dict(json.loads(captures[rank]), **changes)).encode()
    return edited


@pytest.mark.parametrize(
    "case,message",
    [
        ("seven captures", "exactly eight captures"),
        ("not idle", "not verified idle"),
        ("out of order", "launch order"),
        ("topology", "captured topology differs"),
        ("mesh", "captured physical mesh differs"),
        ("slice", "topology identity drifted"),
        ("contract", "topology fleet contract drifted"),
        ("hostname", "hostnames are not unique"),
        ("not a capture", "eight capture records"),
        ("no topology", "holds no topology"),
        ("original fleet", "64-hex"),
    ],
)
def test_derive_topology_binding_refuses(captures: list[bytes], case: str, message: str):
    options = dict(all_hosts_idle_after=True, note="unit test")
    raws = captures
    if case == "seven captures":
        raws = captures[:7]
    elif case == "not idle":
        options["all_hosts_idle_after"] = False
    elif case == "out of order":
        raws = [captures[1], captures[0], *captures[2:]]
    elif case == "topology":
        options["expected_topology_sha256"] = "0" * 64
    elif case == "mesh":
        options["expected_mesh_sha256"] = "0" * 64
    elif case == "slice":
        options["slice_name"] = "other-vm"
    elif case == "contract":
        contract = dict(json.loads(captures[3])["contract"], code_hash="b" * 40)
        raws = _edit(captures, 3, contract=contract)
    elif case == "hostname":
        raws = _edit(captures, 3, hostname=fleet.HOSTS[2])
    elif case == "not a capture":
        raws = [json.dumps({"launch_process_id": 0}).encode(), *captures[1:]]
    elif case == "no topology":
        raws = _edit(captures, 0, contract={"code_hash": CODE_HASH})
    elif case == "original fleet":
        options["original_fleet_sha256"] = "E" * 64
    with pytest.raises(ValueError, match=message):
        derive_topology_binding(raws, **options)
