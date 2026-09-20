"""Authenticated retained host reassignment on an unchanged physical WS32 mesh.

This authenticates staged captures before the original initializer checks live
devices. It neither initializes JAX nor admits checkpoint bytes or model graphs.
"""
from hashlib import sha256
import json
from pathlib import Path


def load_topology_binding(root: Path, pin: str, *, expected_topology: str,
                          expected_mesh: str, original_fleet: str,
                          slice_name: str) -> dict:
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import validate_ws32_topology_fleet
    from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh

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
