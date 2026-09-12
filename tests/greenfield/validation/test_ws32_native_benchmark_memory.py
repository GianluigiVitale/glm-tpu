"""All-resident budget replay and real CPU compiler interfaces; not TPU HBM."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.greenfield import ws32_native_benchmark_memory as memory
from glm_tpu.greenfield.validation import ws32_prefill_memory as original


def record():
    analyses = {name:dict(argument_size_in_bytes=200,output_size_in_bytes=90,
        alias_size_in_bytes=80 if name not in ('cache_probe','cache_init') else 0,
        temp_size_in_bytes=50,generated_code_size_in_bytes=10) for name in memory.ROLES}
    rows = [dict(device_id=i,platform='tpu',process_index=0,
        buffers=[dict(bytes=80,identity_mode='physical_pointer',groups=['prefill_state','decode_state']),
                 dict(bytes=20,identity_mode='physical_pointer',groups=['prefill_state']),
                 dict(bytes=100,identity_mode='physical_pointer',groups=['retained_weights']),
                 dict(bytes=100,identity_mode='physical_pointer',groups=['__all_live_arrays__'])],
        accounted_resident_bytes=300,
        memory_stats=dict(bytes_in_use=350,peak_bytes_in_use=360,bytes_limit=memory.DEVICE_LIMIT))
        for i in range(4)]
    value=dict(schema_version=memory.SCHEMA,compiled_memory=analyses,cache_present=True,phase='cache_ready',
        donated_arguments=deepcopy(memory.OWNERS),reserve_bytes=memory.RESERVE,
        process_index=0,local_slots=[dict(device_id=i,slot=i) for i in range(4)],
        census=dict(schema_version=original.SCHEMA,includes_all_live_arrays=True,devices=rows))
    value['budgets']=memory.budgets(value)
    return value


def test_all_resident_code_and_unlabelled_arrays_count_alias_only_active_output():
    value=record()
    memory.validate_record(json.loads(json.dumps(value)))
    for name in memory.ROLES[:-1]:
        row=value['budgets'][name]['devices'][0]
        output=90 if name=='cache_probe' else 10
        assert row['estimated_peak_bytes']==350+60+output+50
        assert row['resident_code_bytes']==60
        assert value['compiled_memory'][name]['output_size_in_bytes']==90


def test_new_cache_is_budgeted_before_allocation_with_no_alias_credit():
    value=record()
    value['cache_present']=False
    value['phase']='before_cache'
    value['budgets']=memory.budgets(value)
    assert set(value['budgets'])=={'cache_init'}
    row=value['budgets']['cache_init']['devices'][0]
    assert row['estimated_peak_bytes']==350+60+90+50
    memory.validate_record(json.loads(json.dumps(value)))
    value['compiled_memory']['cache_init']['alias_size_in_bytes']=1
    with pytest.raises(ValueError,match='must not donate'): memory.budgets(value)


def test_real_admission_callback_preserves_and_replays_each_boundary(monkeypatch):
    stored=[]
    def make(compiled,state,**kwargs):
        value=record()
        value['cache_present']=state is not None
        value['phase']=kwargs['phase']
        value['budgets']=memory.budgets(value)
        return value
    monkeypatch.setattr(memory,'make_record',make)
    guard=memory.RequestMemoryAdmission(compiled=dict.fromkeys(memory.ROLES),devices=(),
        local_slots={},process_index=0,preserve=lambda stage,value:stored.append((stage,value)))
    with pytest.raises(ValueError,match='phase'): guard('cache_ready',{},object())
    for _ in range(2):
        guard('before_cache',{},None)
        guard('cache_ready',{},object())
        guard('prefill_done',{},object())
    assert [name for name,_ in stored]==['before_cache','cache_ready','prefill_done']*2
    assert all(value['cache_present']==(name!='before_cache') for name,value in stored)


def test_admission_callback_retains_refusal_before_raising(monkeypatch):
    value=record()
    value['cache_present']=False
    value['phase']='before_cache'
    value['census']['devices'][0]['memory_stats']['peak_bytes_in_use']=memory.DEVICE_LIMIT
    value['budgets']=memory.budgets(value)
    monkeypatch.setattr(memory,'make_record',lambda *a,**k:value)
    stored=[]
    guard=memory.RequestMemoryAdmission(compiled=dict.fromkeys(memory.ROLES),devices=(),
        local_slots={},process_index=0,preserve=lambda *entry:stored.append(entry))
    with pytest.raises(ValueError,match='memory fit'): guard('before_cache',{},None)
    assert len(stored)==1 and stored[0][0]=='before_cache'


def test_finished_prefill_budgets_only_decode_consumers_but_all_code():
    value=record()
    value['phase']='prefill_done'
    # The extra repaired-index root can alias the installed decoder index.
    # No new prefill input is authorized after this phase.
    for row in value['census']['devices']:
        row['buffers'][1]['groups']=[]
    value['budgets']=memory.budgets(value)
    assert set(value['budgets'])=={'decode','observer','cache_probe'}
    assert value['budgets']['decode']['devices'][0]['resident_code_bytes']==60
    memory.validate_record(value)


@pytest.mark.parametrize('case', ['missing_code','wrong_owner','missing_owner','duplicate_owner',
    'wrong_reserve','missing_state','weight_alias','no_physical_identity','excess_alias',
    'probe_donation','hidden_live_arrays','past_peak','output_budget','bool_memory'])
def test_unsafe_residency_refuses(case):
    value=record()
    row=value['census']['devices'][0]
    if case=='missing_code': del value['compiled_memory']['observer']
    elif case=='wrong_owner': row['process_index']=1
    elif case=='missing_owner': value['local_slots'].pop()
    elif case=='duplicate_owner': value['local_slots'][1]=value['local_slots'][0]
    elif case=='wrong_reserve': value['reserve_bytes']=0
    elif case=='missing_state': row['buffers'][0]['groups']=[]
    elif case=='weight_alias': row['buffers'][0]['groups'].append('retained_weights')
    elif case=='no_physical_identity': row['buffers'][0]['identity_mode']='distinct_object_upper_count'
    elif case=='excess_alias': value['compiled_memory']['decode']['alias_size_in_bytes']=81
    elif case=='probe_donation': value['compiled_memory']['cache_probe']['alias_size_in_bytes']=1
    elif case=='hidden_live_arrays': value['census']['includes_all_live_arrays']=False
    elif case=='past_peak': row['memory_stats']['peak_bytes_in_use']=memory.DEVICE_LIMIT
    elif case=='output_budget': value['budgets']['decode']['devices'][0]['estimated_peak_bytes']=0
    elif case=='bool_memory': value['compiled_memory']['decode']['temp_size_in_bytes']=True
    with pytest.raises(ValueError): memory.validate_record(value)


def test_real_compiled_donation_interface_and_all_live_census():
    code=r'''
import json
import jax, jax.numpy as jnp, numpy as np
from unittest.mock import patch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState
from scripts.greenfield import ws32_native_benchmark_memory as m
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object),('cpu',))
sharding=NamedSharding(mesh,P())
def a(value): return jax.device_put(jnp.asarray(value),sharding)
state=Ws32BatchedPrefillState(Ws32DecoderState(
    a([0.]),a([1.]),a([[0]]),a([0]),a([[0.]]),a([0]),a([[0]]),a([1]),a([True])),
    a([0.]),a(3),a(False))
weights=a([2.]); uniform=a(0.25)
roots=dict(raw_weights=weights,decode_weights=weights,wk=weights,exact_weights=weights,rope=weights)
def prefill(ids,count,s,w,wk,rope,u):
    return jax.tree.map(lambda v:v+u.astype(v.dtype),s)
def decode(ids,s,w,exact,rope,u):
    return jax.tree.map(lambda v:v+u.astype(v.dtype),s)
args=(a([1,2,3]),a(3),state,weights,weights,weights,uniform)
ds=(a([1]),state.decoder,weights,weights,weights,uniform)
compiled=dict(prefill_chunk=jax.jit(prefill,donate_argnums=(2,)).lower(*args).compile(),
    prefill_tail=jax.jit(prefill,donate_argnums=(2,)).lower(*args).compile(),
    decode=jax.jit(decode,donate_argnums=(1,)).lower(*ds).compile(),
    observer=jax.jit(decode,donate_argnums=(1,)).lower(*ds).compile(),
    cache_probe=jax.jit(lambda s:s.position+1).lower(state.decoder).compile(),
    cache_init=jax.jit(lambda prompt:jax.tree.map(lambda v:jnp.zeros_like(v)+prompt.astype(v.dtype),state)).lower(a(3)).compile())
class Stats:
    # Physical metadata/counters are FIXTURES, never measured TPU evidence.
    def __init__(self,d): self.id,self.platform,self.process_index=d.id,d.platform,0
    def memory_stats(self):
        return dict(bytes_in_use=1<<20,peak_bytes_in_use=1<<20,bytes_limit=m.DEVICE_LIMIT)
devices=tuple(Stats(d) for d in jax.devices())
capture=m.original.capture_resident_buffers
def fixture_census(*args,**kwargs):
    result=capture(*args,**kwargs)  # genuine CPU buffers, identity and compiler interfaces
    for row in result['devices']: row['platform']='tpu'  # ONLY the owner validator fixture
    return result
with patch.object(m.original,'capture_resident_buffers',fixture_census):
    value=m.make_record(compiled,state,retained_roots=roots,devices=devices,
                        local_slots={d.id:i for i,d in enumerate(devices)},process_index=0)
m.validate_record(json.loads(json.dumps(value)))
assert all(r['estimate_fits'] for r in value['budgets'].values())
compiled['decode']=jax.jit(decode).lower(*ds).compile()
try: m.make_record(compiled,state,retained_roots=roots,devices=devices,
                  local_slots={d.id:i for i,d in enumerate(devices)},process_index=0)
except ValueError as e: assert 'donation' in str(e)
else: raise AssertionError('undonated decoder admitted')
print('REAL_CPU_NATIVE_RESIDENCY_INTERFACES_PASS')
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[3],
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=4'),
        text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'REAL_CPU_NATIVE_RESIDENCY_INTERFACES_PASS' in result.stdout
