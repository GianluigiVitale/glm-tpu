"""Authenticated topology identity of the WS32 fleet and retained host reassignment.

``validate_ws32_topology_fleet`` authenticates the eight launch-host topology captures (sealed
topology, launch-host to JAX-process permutation, fleet digest); ``_device_record`` is the device
description ``_initialize_runtime`` compares with each live device (both moved verbatim in S2a).
``apply_topology_binding`` authenticates a retained host reassignment on the unchanged physical
mesh before the initializer checks live devices. Nothing here initializes JAX or admits
checkpoint bytes or model graphs.
"""
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.config.model import PhysicalTopology


_TOPOLOGY_CAPTURE_KEYS = frozenset(
    {
        "captured_utc", "contract", "contract_hash", "fleet_contract_hashes",
        "fleet_local_device_ids_in_runtime_order",
        "hostname",
        "jax_device_count", "jax_local_device_count", "jax_process_count", "jax_process_index",
        "jax_version",
        "launch_process_id",
        "local_device_ids",
        "schema_version",
    }
)


def validate_ws32_topology_fleet(
    captures: tuple[Mapping[str, Any], ...],
    *,
    expected_topology_sha256: str,
    expected_fleet_sha256: str,
    slice_name: str,
) -> tuple[PhysicalTopology, tuple[Mapping[str, Any], ...], str]:
    """Authenticate the sealed launch-host to JAX-process permutation."""

    if len(captures) != 8 or any(
        not isinstance(item, Mapping) or set(item) != _TOPOLOGY_CAPTURE_KEYS
        for item in captures
    ):
        raise ValueError("WS32 topology fleet schema drifted")
    ordered = tuple(sorted(captures, key=lambda item: item["launch_process_id"]))
    if [item["launch_process_id"] for item in ordered] != list(range(8)):
        raise ValueError("WS32 topology launch identities drifted")
    contract = ordered[0]["contract"]
    contract_hash = sha256(
        json.dumps(
            contract,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if any(
        item["schema_version"] != 1
        or item["contract"] != contract
        or item["contract_hash"] != contract_hash
        or item["fleet_contract_hashes"] != [contract_hash] * 8
        or item["jax_device_count"] != 32
        or item["jax_local_device_count"] != 4
        or item["jax_process_count"] != 8
        or not isinstance(item["jax_version"], str)
        or not item["jax_version"]
        or not isinstance(item["captured_utc"], str)
        or not item["captured_utc"]
        for item in ordered
    ):
        raise ValueError("WS32 topology fleet contract drifted")
    if sorted(item["jax_process_index"] for item in ordered) != list(range(8)):
        raise ValueError("WS32 topology JAX process identities drifted")
    if len({item["hostname"] for item in ordered}) != 8:
        raise ValueError("WS32 topology hostnames are not unique")

    topology = PhysicalTopology.from_dict(contract["topology"])
    if topology.slice_name != slice_name or (
        topology.topology_hash != expected_topology_sha256
    ):
        raise ValueError("WS32 topology identity drifted")
    runtime_order = [
        [
            device.device_id
            for device in sorted(
                (
                    value
                    for value in topology.devices
                    if value.process_index == process_index
                ),
                key=lambda value: value.local_device_id,
            )
        ]
        for process_index in range(8)
    ]
    if runtime_order != [
        [process_index * 4 + offset for offset in range(4)]
        for process_index in range(8)
    ]:
        raise ValueError("WS32 topology runtime device order drifted")
    if any(
        item["fleet_local_device_ids_in_runtime_order"] != runtime_order
        or item["local_device_ids"]
        != runtime_order[item["jax_process_index"]]
        for item in ordered
    ):
        raise ValueError("WS32 topology local ownership drifted")
    projection = {
        "fleet_local_device_ids_in_runtime_order": runtime_order,
        "records": [
            {
                name: item[name]
                for name in (
                    "contract_hash",
                    "hostname",
                    "jax_process_index",
                    "launch_process_id",
                    "local_device_ids",
                )
            }
            for item in ordered
        ],
    }
    observed_fleet_sha256 = sha256(
        json.dumps(
            projection,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if observed_fleet_sha256 != expected_fleet_sha256:
        raise ValueError("WS32 topology fleet identity drifted")
    return topology, ordered, observed_fleet_sha256


def _device_record(device: object, *, local_device_id: int) -> dict[str, Any]:
    runtime_local_id = device.local_hardware_id
    if runtime_local_id is not None and int(runtime_local_id) != local_device_id:
        raise ValueError("runtime and captured local device ids disagree")
    return {
        "coordinates": [int(value) for value in device.coords],
        "core_on_chip": int(device.core_on_chip),
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "local_device_id": int(local_device_id),
        "platform": str(device.platform),
        "process_index": int(device.process_index),
    }


def load_topology_binding(root: Path, pin: str, *, expected_topology: str,
                          expected_mesh: str, original_fleet: str,
                          slice_name: str) -> dict:
    from glm_tpu.distributed.mesh import build_ws32_physical_mesh

    raw = (root / 'topology_rebinding.json').read_bytes()
    if not isinstance(pin, str) or len(pin) != 64 or sha256(raw).hexdigest() != pin:
        raise ValueError('topology rebinding hash differs')
    binding = json.loads(raw)
    if (binding.get('schema') != 'glm_perf_topology_rebinding_v1'
            or binding.get('physical_devices_identical') is not True
            or binding.get('all_hosts_idle_after') is not True
            or binding.get('original_topology_sha256') != expected_topology
            or binding.get('mesh_sha256') != expected_mesh
            or binding.get('original_fleet_sha256') != original_fleet):
        raise ValueError('topology rebinding changes the sealed physical identity')
    names = [f'topology.rank{i}.json' for i in range(8)]
    if set(binding.get('capture_sha256', {})) != set(names):
        raise ValueError('topology rebinding requires exactly eight captures')
    captures = []
    for name in names:
        payload = (root / 'topology_capture' / name).read_bytes()
        if sha256(payload).hexdigest() != binding['capture_sha256'][name]:
            raise ValueError('topology capture hash differs: ' + name)
        captures.append(json.loads(payload))
    topology, ordered, fleet = validate_ws32_topology_fleet(tuple(captures),
        expected_topology_sha256=expected_topology,
        expected_fleet_sha256=binding['fleet_sha256'], slice_name=slice_name)
    physical = build_ws32_physical_mesh(topology)
    if physical.mesh_hash != expected_mesh:
        raise ValueError('topology rebinding physical mesh differs')
    slots = {str(i): [s for s, device in enumerate(physical.flattened_device_ids)
                     if device in capture['local_device_ids']]
             for i, capture in enumerate(ordered)}
    if binding.get('host_to_slots') != slots:
        raise ValueError('topology rebinding slot ownership differs')
    if any(not capture['hostname'].endswith('-w-' + str(i))
           or capture['contract'].get('code_hash') != binding.get('code_hash')
           for i, capture in enumerate(ordered)):
        raise ValueError('topology rebinding host/source identity differs')
    return dict(sha256=pin, mesh_sha256=expected_mesh,
        topology_sha256=expected_topology, fleet_sha256=fleet,
        host_to_slots=slots, hosts=[capture['hostname'] for capture in ordered],
        jax_process_indices=[capture['jax_process_index'] for capture in ordered])


def apply_topology_binding(args, root: Path, pin: str | None) -> dict | None:
    """Override only the capture location and fleet mapping after authentication."""
    if pin is None:
        return None
    identity = load_topology_binding(root, pin,
        expected_topology=args.topology_sha256, expected_mesh=args.mesh_sha256,
        original_fleet=args.topology_fleet_sha256, slice_name=args.slice_name)
    args.topology_capture_root = root / 'topology_capture'
    args.topology_fleet_sha256 = identity['fleet_sha256']
    return identity
