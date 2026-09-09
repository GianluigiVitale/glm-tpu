"""Saved real topology/runner binding; fixture runtime and selected load only."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_runtime as runtime_module
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield.ws32_prefill_budget_campaign import topology_bindings

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def original():
    root = Path('/home/gianl/glm-run') / protocol.ORIGINAL_TAG / 'first_window_collected'
    raw = (root / 'host_records/runner.rank0.json').read_bytes()
    physical, captures = topology_bindings()
    return raw, physical, captures


def setup(tmp_path, monkeypatch, original):
    raw, physical, captures = original
    prior = json.loads(raw)
    source = tmp_path / 'retained_reference' / protocol.original_names(0)[0]
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    tag = 'greenfield_fp8_ws32_dense_frontier_d01_20260909T140000000000000Z'
    record = dict(protocol=protocol.PROTOCOL, code_hash='a'*40, launch_rank=0,
                  tag=tag, diagnostic_only=True, programs={})
    preflight = dict(**record, hostname=prior['hostname'], selected_layer_ids=[0,1],
                     include_embedding=True, context_capacity=8192, host_main_rope_table=True,
                     selected_leaf_count=55, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
                     original_tag=protocol.ORIGINAL_TAG, original_ledger_sha256=protocol.LEDGER_SHA,
                     original_runner_sha256=sha256(raw).hexdigest(),
                     headers=[dict(device_slot=v['device_slot'], filename=f"{v['device_slot']}.fixture",
                                   file_bytes=123, header_sha256='b'*64)
                              for v in prior['local_device_slots']],
                     checkpoint_pins=dict(expected_success_sha256=prior['checkpoint_success_sha256']),
                     local_device_slots=prior['local_device_slots'])
    owners = {device: row['jax_process_index'] for row in captures for device in row['local_device_ids']}
    devices = [NS(id=d, platform='tpu', process_index=owners[d]) for d in physical.flattened_device_ids]
    local = [d for d in devices if d.process_index == prior['jax_process_index']]
    jax = NS(default_backend=lambda: 'tpu', process_count=lambda: 8, device_count=lambda:32,
             process_index=lambda: prior['jax_process_index'], local_devices=lambda:local)
    mesh = NS(devices=np.asarray(devices,object).reshape(8,4), axis_names=('expert','feature'))
    runtime = (jax,mesh,physical,NS(topology_hash=prior['topology_sha256']),prior['topology_fleet_sha256'])
    monkeypatch.setattr(runtime_module.socket,'gethostname',lambda: prior['hostname'])
    monkeypatch.setattr(protocol,'load_reference',lambda *a,**k:(prior,{'fixture':'cache replay tested separately'}))
    def save(): (tmp_path/'retained_preflight.json').write_text(json.dumps(preflight))
    save()
    return prior, preflight, record, runtime, save


@pytest.mark.parametrize('mutation', [None,'process','owners','backend','topology','mesh_order',
                                      'axes','header_slot','source_digest','preflight_pin','file_owner'])
def test_saved_real_owner_mapping_and_runtime_refusals(tmp_path,monkeypatch,original,mutation):
    prior, preflight, record, runtime, save = setup(tmp_path,monkeypatch,original)
    jax, mesh, physical, topology, fleet = runtime
    if mutation == 'process': jax.process_index=lambda:7
    elif mutation == 'owners': jax.local_devices=lambda:list(mesh.devices[0])
    elif mutation == 'backend': jax.default_backend=lambda:'cpu'
    elif mutation == 'topology': topology.topology_hash='0'*64
    elif mutation == 'mesh_order': mesh.devices=mesh.devices[::-1]
    elif mutation == 'axes': mesh.axis_names=('feature','expert')
    elif mutation == 'header_slot': preflight['headers'][0]['device_slot']=0
    elif mutation == 'source_digest': preflight['original_runner_sha256']='0'*64
    elif mutation == 'preflight_pin': preflight['code_hash']='0'*40
    elif mutation == 'file_owner':
        preflight['local_device_slots']=deepcopy(preflight['local_device_slots'])
        preflight['local_device_slots'][0]['file_sha256']='0'*64
    save()
    if mutation:
        with pytest.raises(ValueError): runtime_module.bind(root=tmp_path,record=record,runtime=runtime)
    else:
        _, _, slots, _ = runtime_module.bind(root=tmp_path,record=record,runtime=runtime)
        assert slots == {12:9,14:13,13:25,15:29}  # NOT rank*4 or device_id=slot.
        assert record['jax_process_index']==3 and record['original_runner_sha256']==sha256(original[0]).hexdigest()


@pytest.mark.parametrize('failure',[None,'inputs','headers','source','load','peer_load'])
def test_voted_bind_prepare_load_continuation_order(tmp_path,monkeypatch,original,failure):
    prior, preflight, record, runtime, save = setup(tmp_path,monkeypatch,original)
    events=[]
    records={v['device_slot']:{**header,'sha256':v['file_sha256']}
             for v,header in zip(prior['local_device_slots'],preflight['headers'],strict=True)}
    metadata=NS(records_by_slot=records,manifest={'source':{'inventory_sha256':prior['source_inventory_sha256']}})
    if failure=='headers': records[9]['header_sha256']='0'*64
    if failure=='source': metadata.manifest['source']['inventory_sha256']='0'*64
    subset=NS(metadata=metadata)
    prepared=NS(config=object(),tensor_names=('embedding','layer0','layer1'),
                manifest_sha256=prior['checkpoint_manifest_sha256'],payload_bytes_per_chip=protocol.PAYLOAD_BYTES)
    monkeypatch.setattr(runtime_module.preflight_module,'selected_metadata',lambda *a:(preflight['checkpoint_pins'],subset))
    monkeypatch.setattr(runtime_module.preparation,'prepare',lambda *a,**k:prepared)
    def host_inputs(*args):
        events.append('inputs')
        if failure=='inputs': raise ValueError('fixture host input failure')
        return np.arange(8155,dtype=np.int32),np.zeros((8192,64),np.float32)
    monkeypatch.setattr(runtime_module.execution,'host_inputs',host_inputs)
    def load(*args,**kwargs):
        events.append('load')
        if failure=='load': raise ValueError('fixture selected load failure')
        return NS(layer_ids=(0,1),include_embedding=True,payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
                  arrays={k:k for k in prepared.tensor_names},local_device_slots=prior['local_device_slots'],
                  integrity_scope='fixture_selected',device_memory_before=(),device_memory_after=())
    monkeypatch.setattr(runtime_module,'load_ws32_layer_subset',load)
    monkeypatch.setattr(runtime_module,'ws32_decoder_weight_names',lambda config:NS(embedding_local='embedding',layers=('layer0','layer1')))
    monkeypatch.setattr(runtime_module,'_bind_weight_name_tree',lambda names,arrays:(arrays['embedding'],(arrays['layer0'],arrays['layer1'])))
    import jax.sharding
    monkeypatch.setattr(jax.sharding,'NamedSharding',lambda *a:None)
    runtime[0].make_array_from_callback=lambda shape,sharding,callback:callback((slice(None),slice(None)))
    runtime[0].block_until_ready=lambda v:events.append('rope')
    def execute(**kwargs):
        events.append('execute')
        assert kwargs['embedding']=='embedding' and kwargs['layers']==('layer0','layer1')
        assert kwargs['local_slots']=={12:9,14:13,13:25,15:29}
        assert kwargs['prepared'] is prepared and kwargs['tokens'].shape==(8155,)
    monkeypatch.setattr(runtime_module.execution,'execute',execute)
    def consensus(ok):
        events.append(('vote',ok))
        return False if failure=='peer_load' and 'load' in events else ok
    def run(): runtime_module.execute_bound(root=tmp_path,record=record,repo=REPO,runtime=runtime,
                                          consensus=consensus,inspect_program=lambda *a:None)
    if failure:
        with pytest.raises((ValueError,RuntimeError)):run()
        assert 'execute' not in events and 'rope' not in events
        if failure in ('inputs','headers','source'):assert 'load' not in events
    else:
        run()
        assert [v for v in events if isinstance(v,str)]==['inputs','load','rope','execute']
        assert record['selected_layer_ids']==[0,1] and record['include_embedding']
        assert not record.get('numerical_promotion',False)
