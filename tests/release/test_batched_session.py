from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.optimized import request
from glm_tpu.optimized.batched_decode import BatchedDecodeResult
from glm_tpu.optimized.batched_session import BatchedSession


def fixture(monkeypatch, *, bad_lane=None, fail_delivery=False, deadline=100):
    monkeypatch.setattr('glm_tpu.optimized.batched_session.jax.block_until_ready',lambda x:x)
    values=[request.from_token_ids(list(range(1,lane+2)),request_id=f'lane-{lane}',
        max_new_tokens=lane+1,context_capacity=32768) for lane in range(8)]
    calls=[];events=[];ticks=[0.]
    def decode(tokens,state,active):
        calls.append(active.copy());ticks[0]+=1
        status=np.array([[10,1,len(v['prompt_ids'])+len(calls),
            len(v['prompt_ids'])+len(calls)+1] for v in values],np.int32)
        if len(calls)==1:status[2,0]=values[2]['eos_ids'][0]
        if bad_lane is not None:status[bad_lane,2]+=1
        return BatchedDecodeResult(state,tokens,status)
    def deliver(lane,event,round_index):
        if fail_delivery and round_index==1 and lane==2:raise OSError('sink failed')
        events.append((lane,event,round_index))
    session=BatchedSession(values,decode=decode,put=lambda x:x,vote=lambda x:bool(x),
        deliver=deliver,deadline=deadline,clock=lambda:ticks[0])
    status=np.array([[7,1,len(v['prompt_ids']),len(v['prompt_ids'])+1] for v in values],np.int32)
    return session,calls,events,status


def test_eight_streams_advance_together_and_stop_independently(monkeypatch):
    session,calls,events,status=fixture(monkeypatch)
    session.run(None,np.zeros((8,1),np.int32),status)
    assert len(calls)==7  # One graph call per round, never one call per lane.
    assert [len(v) for v in session.events]==[1,2,2,4,5,6,7,8]
    assert session.events[2][-1].finish_reason=='eos'
    assert not calls[0][0] and not calls[1][1] and not calls[1][2]
    assert {lane for lane,_,rnd in events if rnd==1}==set(range(1,8))
    for lane,rows in enumerate(session.events):
        assert [e.index for e in rows]==list(range(len(rows)))
        assert all(e.request_id==f'lane-{lane}' for e in rows)


@pytest.mark.parametrize('options',[{'bad_lane':4},{'fail_delivery':True},{'deadline':.5}])
def test_failure_poisoning_prevents_replay(monkeypatch,options):
    session,calls,events,status=fixture(monkeypatch,**options)
    with pytest.raises(RuntimeError):session.run(None,np.zeros((8,1),np.int32),status)
    assert session.failed and len(calls)==1
    before=list(events)
    with pytest.raises(RuntimeError):session.run(None,np.zeros((8,1),np.int32),status)
    assert events==before and len(calls)==1


def test_concurrent_payload_capacity_count_and_identity():
    values=[request.from_token_ids([1,2],request_id=f'r-{i}',max_new_tokens=10,
                                   context_capacity=32768) for i in range(8)]
    value=request.batch(values,concurrent=True)
    assert request.requests(value)==values and value['schema']==request.CONCURRENT_SCHEMA
    for n in (1,3,8):request.validate_payload(request.batch(values[:n],concurrent=True))
    for bad in ([],values+[values[0]]):
        with pytest.raises(ValueError):request.batch(bad,concurrent=True)
    with pytest.raises(ValueError):request.batch([request.from_token_ids([1],request_id='x',
        max_new_tokens=1)],concurrent=True)
    with pytest.raises(ValueError):request.from_token_ids([1,2],request_id='x',
        max_new_tokens=32767,context_capacity=32768)
    value['schema']=request.BATCH_SCHEMA
    with pytest.raises(ValueError):request.validate_payload(value)
