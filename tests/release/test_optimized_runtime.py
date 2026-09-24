"""Actual optimized host runtime with synthetic device results, CPU only."""
from types import SimpleNamespace
import numpy as np
import pytest

from glm_tpu.optimized import request
from glm_tpu.optimized.request_loop import PackedDecodeResult
from glm_tpu.optimized.runtime import OrdinaryRuntime
from glm_tpu.optimized.ws32_decoder import Ws32DecodeStepResult
from tests.greenfield.runtime.test_ws32_request_session import state, prefill


def fixture(monkeypatch,*,late=False):
    runtime=object.__new__(OrdinaryRuntime)
    runtime.capacity=8192
    runtime.concurrent_size=0
    runtime.active=False;runtime.put=lambda x:x
    runtime.weights=runtime.wk=runtime.rope=None
    runtime.record={'requests':[]};runtime.save=lambda record:None
    runtime.admit=lambda name:None;runtime.stats=lambda:[]
    runtime.initialize=lambda count:SimpleNamespace(decoder=state(0))
    runtime.vote=lambda value:value
    runtime.phase=lambda name,fn:fn()
    ticks=[1.];calls=[]
    def execute_prefill(*args):
        ticks[0]+=.5
        return prefill()
    runtime.prefill={114:execute_prefill,128:execute_prefill}
    def decode(token,previous,*args):
        i=len(calls)+1;calls.append(int(token[0]));ticks[0]+=100 if late else .25
        out=Ws32DecodeStepResult(state(3+i),np.array([8+i],np.int32),np.zeros((1,1)))
        return PackedDecodeResult(out,np.array([8+i,1,3+i,4+i],np.int32))
    runtime.decode=decode
    from jax.experimental import multihost_utils
    monkeypatch.setattr(multihost_utils,'process_allgather',lambda value:np.stack([value]*8))
    return runtime,ticks,calls


def test_fresh_request_delivery_and_terminal_release(monkeypatch):
    runtime,ticks,calls=fixture(monkeypatch)
    value=request.from_token_ids([30,31,32],request_id='fixture',max_new_tokens=3)
    events=[]
    tokens,report=runtime.generate(value,deliver=events.append,deadline=50,clock=lambda:ticks[0])
    np.testing.assert_array_equal(tokens,[7,9,10])
    assert calls==[7,9] and len(events)==3 and not runtime.active
    assert report['timed_decode_tokens']==2 and report['finish_reason']=='length'
    assert report['decode_wall_seconds']==.5
    assert runtime.record['requests']==[report]


def test_deadline_votes_before_delivery_and_never_retries(monkeypatch):
    runtime,ticks,calls=fixture(monkeypatch,late=True)
    value=request.from_token_ids([30,31,32],request_id='fixture',max_new_tokens=3)
    events=[]
    with pytest.raises(RuntimeError):
        runtime.generate(value,deliver=events.append,deadline=50,clock=lambda:ticks[0])
    assert len(events)==1 and runtime.active and calls==[7]
    with pytest.raises(RuntimeError):
        runtime.generate(value,deliver=events.append,deadline=500,clock=lambda:ticks[0])
    assert calls==[7] and not runtime.record['requests']


def test_capacity_mismatch_refuses_before_execution(monkeypatch):
    runtime,ticks,calls=fixture(monkeypatch)
    value=request.from_token_ids([1,2],request_id='long',max_new_tokens=3,
                                 context_capacity=request.LONG_CAPACITY)
    with pytest.raises(RuntimeError,match='capacity differs'):
        runtime.generate(value,deliver=lambda event:None,deadline=50,clock=lambda:ticks[0])
    assert not calls and not runtime.record['requests']


def test_next_question_starts_fresh_with_separate_delivery(monkeypatch):
    runtime,ticks,calls=fixture(monkeypatch)
    for name in ('first','second'):
        calls.clear();events=[]
        value=request.from_token_ids([30,31,32],request_id=name,max_new_tokens=3)
        tokens,report=runtime.generate(value,deliver=events.append,deadline=50,clock=lambda:ticks[0])
        assert [e.index for e in events]==[0,1,2]
        assert {e.request_id for e in events}=={name}
        np.testing.assert_array_equal(tokens,[7,9,10])
    assert len(runtime.record['requests'])==2
