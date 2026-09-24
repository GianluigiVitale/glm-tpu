"""Packed greedy session: host failures, events and timing, policy refusals.

Until S2f the session and policy tests also ran the frozen sampled session and policy they replaced
(same events and timing, one fewer vote per decode, identical refusals at seed 0). That oracle is
archived at ``archive/research-20260922``; its final green run is recorded in the S2f commit message,
and the frozen policy's refusals at seed 0 are pinned below as data (``POLICY_CASES``).
"""
import numpy as np
import pytest

from glm_tpu.models.glm_moe_dsa.state import DecodeStepResult
from glm_tpu.models.glm_moe_dsa.model import PackedDecodeResult
from glm_tpu.engine.request_session import PackedRequestSession, RequestPolicy
from tests.engine.test_request_session import state, prefill


def setup(packed=True, *, mutate=None, vote=None, sink_error=False, outputs=(9,10), max_new=4):
    assert packed  # the greedy release session is the only session
    clock=[10.]
    calls, draws, events, votes = [], [], [], []
    def decode(token,previous,*uniform):
        i=len(calls)+1
        calls.append(token)
        draws.extend(float(u) for u in uniform)
        clock[0] += .25
        out=DecodeStepResult(state(3+i),np.array([outputs[i-1]],np.int32),np.zeros((1,1)))
        status=np.array([outputs[i-1],1,3+i,4+i],np.int32)
        return PackedDecodeResult(out,mutate(status) if mutate else status)
    def sink(event):
        events.append(event)
        clock[0]+=.5
        if sink_error and event.index>0: raise OSError('sink failed after possible emission')
    def fleet(valid):
        votes.append(valid)
        return vote(valid,len(votes)) if vote else valid
    session=PackedRequestSession(RequestPolicy('request-a',3,max_new,20,256,(10,)),decode_step=decode,
                                 fleet_all=fleet,deliver=sink,delivery_boundary='fake sink',
                                 request_started=1.,clock=lambda:clock[0])
    return session,calls,draws,events,votes,[],clock


def test_packed_loop_events_pause_eos_and_boundary_work():
    session,calls,draws,events,votes,_,_=setup()
    with pytest.raises(RuntimeError):session.next_uniform()  # greedy: no draw exists
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
    assert [(e.index,e.token_id,e.finish_reason) for e in events]==[(0,7,None),(1,9,None),(2,10,'eos')]
    assert draws==[]  # the greedy session never draws a uniform
    assert [int(v[0]) for v in calls]==[7,9]
    assert len(votes)==7  # prefill: 3 votes, then 2 per decode


# (arguments, outcome of the frozen sampled policy at seed 0 -- the release's value -- recorded from
# the final run of the frozen policy at S2f: None, or (exception type name, message))
POLICY_CASES = [
    (('request-a', 3, 4, 20, 256, (10,)), None),
    (('r', 1, 1, 2, 1, (0,)), None),
    (('', 3, 4, 20, 256, (10,)), ('ValueError', 'a nonempty request id is required')),
    ((None, 3, 4, 20, 256, (10,)), ('ValueError', 'a nonempty request id is required')),
    ((7, 3, 4, 20, 256, (10,)), ('ValueError', 'a nonempty request id is required')),
    (('\ud800', 3, 4, 20, 256, (10,)),
     ('UnicodeEncodeError', "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed")),
    (('ok', 0, 4, 20, 256, (10,)), ('ValueError', 'positive integer request dimensions required')),
    (('ok', True, 4, 20, 256, (10,)), ('ValueError', 'positive integer request dimensions required')),
    (('ok', 3, 4.0, 20, 256, (10,)), ('ValueError', 'positive integer request dimensions required')),
    (('ok', 3, 4, -20, 256, (10,)), ('ValueError', 'positive integer request dimensions required')),
    (('ok', 3, 4, 20, 0, (10,)), ('ValueError', 'positive integer request dimensions required')),
    (('ok', 3, 18, 20, 256, (10,)), ('ValueError', 'full registered generation cap must fit; no silent truncation')),
    (('ok', 3, 4, 20, 256, [10]), ('ValueError', 'unique in-vocabulary EOS ids required')),
    (('ok', 3, 4, 20, 256, ()), ('ValueError', 'unique in-vocabulary EOS ids required')),
    (('ok', 3, 4, 20, 256, (10, 10)), ('ValueError', 'unique in-vocabulary EOS ids required')),
    (('ok', 3, 4, 20, 256, (256,)), ('ValueError', 'unique in-vocabulary EOS ids required')),
    (('ok', 3, 4, 20, 256, (True,)), ('ValueError', 'unique in-vocabulary EOS ids required')),
    (('', 0, 0, 0, 0, ()), ('ValueError', 'a nonempty request id is required')),
    (('ok', 0, 0, 0, 0, [1, 1]), ('ValueError', 'positive integer request dimensions required')),
]


def _outcome(factory, *args):
    try:
        factory(*args)
    except Exception as exc:
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize(('args', 'expected'), POLICY_CASES)
def test_greedy_policy_refuses_exactly_like_the_frozen_policy_at_seed_zero(args, expected):
    outcome = _outcome(RequestPolicy, *args)
    assert (None if outcome is None else (outcome[0].__name__, outcome[1])) == expected
    assert 'seed' not in RequestPolicy.__dataclass_fields__


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
