"""D4 host failures, deterministic draws and complete CPU32 step equality."""
import os
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy, Ws32RequestSession
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
from glm_tpu.optimized.request_loop import PackedDecodeResult, PackedRequestSession, request_uniform_values
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
