"""Host permutation admission must not loosen physical chip or fleet identity."""
from argparse import Namespace
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from glm_tpu.greenfield.types import PhysicalDevice, PhysicalTopology
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
from glm_tpu.perf.topology_binding import apply_topology_binding, summarize_topology_binding


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


@pytest.fixture
def fixture(tmp_path):
    topology = PhysicalTopology(slice_name='db-v4-64-od', topology_shape=(2, 4, 4),
        devices=tuple(PhysicalDevice(device_id=i, process_index=i//4, local_device_id=i%4,
            coordinates=(i//16, (i//4)%4, i%4), core_on_chip=0,
            platform='tpu', device_kind='TPU v4') for i in range(32)))
    mesh = build_ws32_physical_mesh(topology)
    contract = dict(code_hash='c'*40, topology=topology.to_dict())
    contract_hash = sha256(encoded(contract)).hexdigest()
    runtime_order = [list(range(i*4, i*4+4)) for i in range(8)]
    captures = [dict(schema_version=1, captured_utc='2026-09-20T12:00:00Z',
        contract=contract, contract_hash=contract_hash, fleet_contract_hashes=[contract_hash]*8,
        fleet_local_device_ids_in_runtime_order=runtime_order, hostname=f'test-w-{i}',
        jax_device_count=32, jax_local_device_count=4, jax_process_count=8,
        jax_process_index=(i+3)%8, jax_version='test', launch_process_id=i,
        local_device_ids=runtime_order[(i+3)%8]) for i in range(8)]
    projection = dict(fleet_local_device_ids_in_runtime_order=runtime_order,
        records=[{key: r[key] for key in ('contract_hash', 'hostname', 'jax_process_index',
            'launch_process_id', 'local_device_ids')} for r in captures])
    directory = tmp_path/'topology_capture'
    directory.mkdir()
    hashes = {}
    for i, capture in enumerate(captures):
        name = f'topology.rank{i}.json'
        payload = encoded(capture)
        (directory/name).write_bytes(payload)
        hashes[name] = sha256(payload).hexdigest()
    binding = dict(schema='glm_perf_topology_rebinding_v1', code_hash='c'*40,
        original_topology_sha256=topology.topology_hash, mesh_sha256=mesh.mesh_hash,
        original_fleet_sha256='a'*64, fleet_sha256=sha256(encoded(projection)).hexdigest(),
        physical_devices_identical=True, all_hosts_idle_after=True, capture_sha256=hashes,
        host_to_slots={str(i):[s for s,d in enumerate(mesh.flattened_device_ids)
            if d in capture['local_device_ids']] for i,capture in enumerate(captures)})
    args = Namespace(topology_sha256=topology.topology_hash, mesh_sha256=mesh.mesh_hash,
        topology_fleet_sha256='a'*64, topology_capture_root=tmp_path/'historical',
        slice_name='db-v4-64-od')
    return tmp_path, args, binding, captures


def seal(root, binding):
    payload = encoded(binding)
    (root/'topology_rebinding.json').write_bytes(payload)
    return sha256(payload).hexdigest()


def test_permutation_overrides_only_captures_and_fleet(fixture):
    root,args,binding,_ = fixture
    before = deepcopy(vars(args))
    pin = seal(root,binding)
    result = apply_topology_binding(args,root,pin)
    assert result['jax_process_indices'] == [3,4,5,6,7,0,1,2]
    assert vars(args) == dict(before, topology_capture_root=root/'topology_capture',
        topology_fleet_sha256=binding['fleet_sha256'])


def test_default_preserves_legacy_without_reading_any_files(fixture):
    root,args,_,_ = fixture
    before = deepcopy(vars(args))
    assert apply_topology_binding(args,root,None) is None
    assert vars(args) == before


@pytest.mark.parametrize('failure', ['pin', 'last_capture', 'missing_capture', 'mesh',
    'topology', 'old_fleet', 'fleet', 'slots', 'cleanup', 'physical'])
def test_corruption_refused_before_mutating_arguments(fixture, failure):
    root,args,binding,_ = fixture
    before = deepcopy(vars(args))
    if failure in ('last_capture','missing_capture'):
        path = root/'topology_capture/topology.rank7.json'
        if failure == 'last_capture':path.write_bytes(path.read_bytes()+b' ')
        else:path.unlink()
    key = dict(mesh='mesh_sha256',topology='original_topology_sha256',
        old_fleet='original_fleet_sha256',fleet='fleet_sha256').get(failure)
    if key:binding[key]='0'*64
    if failure == 'slots':binding['host_to_slots']['7']=[0,1,2,3]
    if failure == 'cleanup':binding['all_hosts_idle_after']=False
    if failure == 'physical':binding['physical_devices_identical']=False
    pin = seal(root,binding)
    if failure == 'pin':pin='0'*64
    with pytest.raises((ValueError,FileNotFoundError)):
        apply_topology_binding(args,root,pin)
    assert vars(args) == before


def test_summary_requires_controller_and_every_worker_binding(fixture, monkeypatch):
    root,args,binding,captures = fixture
    pin = seal(root,binding)
    import scripts.release.ws32_user_worker as worker
    monkeypatch.setattr(worker,'site_args',lambda _:deepcopy(args))
    rows=[dict(rank=i,hostname=c['hostname'],jax_process_index=c['jax_process_index'],
        topology_rebinding_sha256=pin,physical_identity=dict(mesh_sha256=args.mesh_sha256,
            topology_sha256=args.topology_sha256,fleet_sha256=binding['fleet_sha256'],
            local_slots=binding['host_to_slots'][str(i)])) for i,c in enumerate(captures)]
    controller=dict(topology_rebinding_sha256=pin)
    assert summarize_topology_binding(root,controller,rows)['sha256']==pin
    for field in ('hostname','jax_process_index','topology_rebinding_sha256','physical_identity'):
        bad=deepcopy(rows)
        bad[7][field]=rows[0][field] if field!='topology_rebinding_sha256' else None
        with pytest.raises(ValueError):summarize_topology_binding(root,controller,bad)
    with pytest.raises(ValueError):summarize_topology_binding(root,{},rows)
    assert summarize_topology_binding(root,{},[{} for _ in range(8)]) is None
