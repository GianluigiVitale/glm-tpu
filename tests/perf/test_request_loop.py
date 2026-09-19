"""D4 host failures, deterministic draws and complete CPU32 step equality."""
import os
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy, Ws32RequestSession
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
from glm_tpu.perf.request_loop import PackedDecodeResult, PackedRequestSession, request_uniform_values
from tests.greenfield.runtime.test_ws32_request_session import state, prefill


def setup(packed, *, mutate=None, vote=None, sink_error=False, outputs=(9,10), max_new=4):
    clock=[10.]
    calls, draws, events, votes, puts = [], [], [], [], []
    policy=RequestPolicy('request-a',42,3,max_new,20,256,(10,))
    bank=request_uniform_values(policy)
    def decode(token,previous,*uniform):
        i=len(calls)+1
        calls.append(token)
        draws.append(float(bank[i] if packed else uniform[0]))
        clock[0] += .25
        out=Ws32DecodeStepResult(state(3+i),np.array([outputs[i-1]],np.int32),np.zeros((1,1)))
        if not packed: return out
        status=np.array([outputs[i-1],1,3+i,4+i],np.int32)
        return PackedDecodeResult(out,mutate(status) if mutate else status)
    def replicate(value):
        puts.append(value)
        return value
    def sink(event):
        events.append(event)
        clock[0]+=.5
        if sink_error and event.index>0: raise OSError('sink failed after possible emission')
    def fleet(valid):
        votes.append(valid)
        return vote(valid,len(votes)) if vote else valid
    session=(PackedRequestSession if packed else Ws32RequestSession)(
        policy,decode_step=decode,replicate_uniform=replicate,fleet_all=fleet,deliver=sink,
        delivery_boundary='fake sink',request_started=1.,clock=lambda:clock[0])
    return session,calls,draws,events,votes,puts,clock


def test_packed_loop_preserves_events_rng_pause_eos_and_removes_boundary_work():
    a=setup(False); b=setup(True)
    for session,*_ in (a,b):
        session.next_uniform()
        session.accept_prefill(prefill())
        session.step()
        paused=session
        paused.step()
        assert session.finished and not session.failed
        assert session.decode_seconds==(.25,.25)
        assert session.ttft_seconds==9.5 and session.delivered_request_seconds==11.
        with pytest.raises(RuntimeError):session.step()
        session.release()
        assert session._state is None
    assert a[2]==b[2] and a[3]==b[3]
    assert [int(v[0]) for v in b[1]]==[7,9]
    assert len(a[4])==9 and len(b[4])==7  # Prefill unchanged, 3 -> 2 votes/decode.
    assert len(a[5])==3 and len(b[5])==1  # Only prefill requests a device_put.


@pytest.mark.parametrize('mutate',[
    lambda s:s[:3], lambda s:s.astype(np.int64),
    lambda s:np.array([-1,1,4,5],np.int32),
    lambda s:np.array([256,1,4,5],np.int32),
    lambda s:np.array([9,0,4,5],np.int32),
    lambda s:np.array([9,1,3,5],np.int32),
    lambda s:np.array([9,1,4,4],np.int32),
])
def test_invalid_compact_status_never_emits_or_retries(mutate):
    session,calls,_,events,*_=setup(True,mutate=mutate)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):session.step()
    assert session.failed and len(events)==1
    with pytest.raises(RuntimeError):session.step()
    assert len(calls)==1


@pytest.mark.parametrize('answer',[False,'yes'])
def test_remote_or_nonboolean_refusal_prevents_delivery(answer):
    session,_,_,events,*_=setup(True,vote=lambda v,n:answer if n==4 else v)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):session.step()
    assert session.failed and len(events)==1


def test_sink_failure_commits_once_then_poisoned():
    session,calls,_,events,*_=setup(True,sink_error=True)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError) as error:session.step()
    assert isinstance(error.value.__cause__,OSError)
    assert session.failed and len(events)==2 and len(session.events)==2
    assert session.delivered_request_seconds==9.5
    with pytest.raises(RuntimeError):session.step()
    assert len(calls)==1


def test_length_stop_without_extra_decode_and_first_token_stop():
    for maximum in (1,2):
        session,calls,*_=setup(True,max_new=maximum)
        session.accept_prefill(prefill())
        if maximum==2:session.step()
        assert session.finished and session.events[-1].finish_reason=='length'
        assert len(calls)==maximum-1


@pytest.mark.parametrize('invalid_time',[float('nan'),-100.])
def test_invalid_step_clock_never_emits(invalid_time):
    def corrupt(status):
        clock[0]=invalid_time
        return status
    session,_,_,events,_,_,clock=setup(True,mutate=corrupt)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):session.step()
    assert session.failed and len(events)==1


def test_remote_delivery_failure_is_terminal_after_commit():
    session,_,_,events,*_=setup(True,vote=lambda v,n:False if n==5 else v)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):session.step()
    assert session.failed and len(events)==2 and len(session.events)==2
    assert session.delivered_request_seconds==9.5


@pytest.mark.parametrize('greedy', [False, True])
def test_packed_decoder_uniform_bank_and_metadata_cpu32(greedy):
    code=r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b,ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from glm_tpu.perf.request_loop import build_packed_decoder_program,make_request_uniform_bank,pack_decode_metadata_mapped
from glm_tpu.perf.ws32_decoder_challenger import build_ws32_challenger_decoder_program,Ws32PerfOptions
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices()).reshape(8,4),('expert','feature'))
def put(x):return jax.device_put(x,NamedSharding(mesh,P()))
config,weights,wk=fixture(mesh)
resident=bf16_resident_weights(mesh,config,weights)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
kw=dict(sparse_attention_interpret=True,linear_interpret=True)
prefill=b.build_ws32_batched_prefill_program(mesh,config,block_rows=2,key_tile=128,**kw)
s=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
wk=tuple(put(x) for x in wk)
for toks,count in [([30,31],2),([32,-1],1)]:
 out=prefill.execute(put(jnp.array(toks,jnp.int32)),put(jnp.int32(count)),s,weights,wk,rope);s=out.state
ds,token=b.finish_ws32_batched_prefill(out)
policy=RequestPolicy('packed-cpu',7,3,4,config.context_capacity,config.geometry.vocab_size,(0,))
bank=make_request_uniform_bank(mesh,policy)
prompt=put(jnp.int32(3))
opts=Ws32PerfOptions(sampler='greedy' if GREEDY else 'nucleus_candidates',bf16_resident=True,lse_attention=not GREEDY,dsa_two_stage=True,
    candidates_per_shard=8,routed_projection=RoutedProjectionConfig(output_tile=128,contraction_tile=128))
base=build_ws32_challenger_decoder_program(mesh,config,options=opts,sampling=None if GREEDY else NucleusConfig(),**kw)
packed=build_packed_decoder_program(mesh,config,options=opts,sampling=None if GREEDY else NucleusConfig(),**kw)
for i in (1,2):
 extra=() if GREEDY else (put(jnp.float32(np.asarray(bank)[i])),)
 a=base.execute(token,ds,resident,rope,*extra)
 bank_args=() if GREEDY else (bank,prompt)
 z=packed.execute(token,ds,resident,rope,*bank_args)
 for aa,bb in zip(jax.tree.leaves(a),jax.tree.leaves(z.decoded)):
  np.testing.assert_array_equal(np.asarray(aa).view(np.uint8),np.asarray(bb).view(np.uint8))
 np.testing.assert_array_equal(np.asarray(z.metadata),[int(a.next_token[0]),1,3+i,4+i])
 token,ds=a.next_token,a.state
# Invalid RNG frontiers may use a clipped operand but must refuse admission.
if not GREEDY:
 for bad_prompt in (put(jnp.int32(-100)),put(jnp.int32(100))):
  z=packed.execute(token,ds,resident,rope,bank,bad_prompt)
  assert int(np.asarray(z.metadata)[1])==0
 bad_bank=bank.at[3].set(jnp.float32(float('nan')))
 z=packed.execute(token,ds,resident,rope,bad_bank,prompt)
 assert int(np.asarray(z.metadata)[1])==0
# A health failure on one owner/feature must reach every host's compact status.
def poison(t,h,p,l,v):
 h=h & ~((jax.lax.axis_index('expert')==7)&(jax.lax.axis_index('feature')==3))
 return pack_decode_metadata_mapped(t,h,p,l,v)
probe=jax.jit(jax.shard_map(poison,mesh=mesh,in_specs=(P(),)*5,out_specs=P(),check_vma=False))
z=probe(put(jnp.array([9],jnp.int32)),put(jnp.array([True])),put(jnp.array([4],jnp.int32)),put(jnp.array([5],jnp.int32)),put(jnp.bool_(True)))
for shard in z.addressable_shards: assert int(np.asarray(shard.data)[1])==0
print('packed CPU32: bitwise model outputs, deterministic RNG, all-owner health, invalid-frontier refusal')
'''
    code=code.replace('GREEDY',repr(greedy))
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32')
    p=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=900)
    assert p.returncode==0,p.stdout+p.stderr
