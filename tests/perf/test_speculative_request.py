"""Host acceptance, cache root ownership and ambiguous delivery failures."""
import numpy as np
import pytest
import jax.numpy as jnp

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.perf.mtp_state import NativeState
from glm_tpu.perf.speculative_verify import VerificationProposal
from glm_tpu.perf.speculative_accept import greedy_acceptance
from glm_tpu.perf.speculative_plan import plan_verification
from glm_tpu.perf.speculative_request import accept_greedy, SpeculativeRequestSession
from tests.greenfield.runtime.test_ws32_request_session import state, prefill


def setup(*, predictions=(8,9,10), drafts=(8,9), maximum=7, fault=None, planned=False):
    now=[10.]; events=[]; calls=[]; roots=[]
    native=NativeState(state(3),np.array([8],np.int32),np.zeros((1,4)))
    def propose(pending, target, old, rows):
        calls.append(rows); roots.append(old)
        now[0]+=.2
        ids=np.asarray([int(pending[0]),*drafts][:rows],np.int32)
        values=np.asarray(predictions[:rows],np.int32)
        health=np.ones(rows,bool)
        if fault=='health':health[-1]=False
        proposal=VerificationProposal(None,None,None,None,None,values,None,
                                      np.ones((rows,4)),health)
        return ids,proposal
    def propose_planned(pending,target,old,rows,remaining):
        ids,proposal=propose(pending,target,old,rows)
        result=plan_verification(jnp.asarray(ids),proposal,jnp.asarray(target.position),
            jnp.int32(remaining),vocab_size=256,eos_ids=(10,))
        if fault=='metadata':result=result._replace(metadata=result.metadata.at[2].set(0))
        if fault=='device_count':result=result._replace(count=jnp.int32(0))
        return result
    def replicate_count(value):
        assert not planned, 'device plan must not upload its accepted count'
        return value
    def commit(old,proposal,count):
        now[0]+=.01
        return state(int(old.position[0])+int(count))
    def refresh(old,history,count):
        assert old is roots[-1]  # Never the recurrent draft root.
        assert np.array_equal(history.shifted_tokens,np.asarray(predictions[:calls[-1]]))
        assert int(history.position[0])==int(old.cache.position[0])
        now[0]+=.02
        cache=state(int(old.cache.position[0])+int(count))
        if fault=='refresh':cache=cache._replace(contract_valid=np.array([False]))
        return NativeState(cache,np.array([11],np.int32),np.full((1,4),2.))
    def sink(event):
        events.append(event);now[0]+=.03
        if fault=='sink' and event.index==2:raise OSError('ambiguous emission')
    session=SpeculativeRequestSession(RequestPolicy('spec-test',0,3,maximum,20,256,(10,)),
        decode_step=None,replicate_uniform=lambda x:x,fleet_all=lambda x:x,deliver=sink,
        delivery_boundary='test sink',request_started=1.,clock=lambda:now[0],
        native_state=native,propose=propose,commit=commit,refresh=refresh,replicate_count=replicate_count,
        propose_planned=propose_planned if planned else None,
        fleet_agree=lambda x:False if fault=='agreement' else True)
    session.accept_prefill(prefill())
    if fault=='alignment':session._native=native._replace(cache=state(4))
    return session,events,calls,roots


@pytest.mark.parametrize('drafts,expected', [((8,9),[8,9,10]),((8,77),[8,9]),((77,9),[8])])
@pytest.mark.parametrize('planned',[False,True])
def test_prefix_and_target_refresh(drafts,expected,planned):
    s,events,calls,roots=setup(drafts=drafts,planned=planned)
    old=s._native
    emitted=s.step()
    assert [e.token_id for e in emitted]==expected
    assert [e.index for e in emitted]==list(range(1,len(expected)+1))
    assert int(s._state.position[0])==3+len(expected)
    assert int(s._native.cache.position[0])==int(s._state.position[0])
    assert roots==[old] and calls==[3]
    assert s.rounds[0]['wall_seconds']>=.23+.03*len(expected)-1e-8
    assert s.finished==(expected[-1]==10)


@pytest.mark.parametrize('planned',[False,True])
def test_short_tail_budget_and_eos_no_extra_row(planned):
    s,events,calls,_=setup(maximum=2,planned=planned)
    assert s.step()[-1].finish_reason=='length' and calls==[1]
    with pytest.raises(RuntimeError):s.step()
    s.release();assert s._native is None and s._state is None
    s,events,calls,_=setup(predictions=(10,9,11),drafts=(10,9),planned=planned)
    assert len(s.step())==1 and s.finished and s.events[-1].finish_reason=='eos'


@pytest.mark.parametrize('fault',['alignment','health','agreement','refresh'])
@pytest.mark.parametrize('planned',[False,True])
def test_refusal_preserves_both_committed_roots_and_no_delivery(fault,planned):
    s,events,calls,_=setup(fault=fault,planned=planned)
    target,native=s._state,s._native
    with pytest.raises(RuntimeError):s.step()
    assert s.failed and len(events)==1 and s._state is target and s._native is native
    attempted=len(calls)
    with pytest.raises(RuntimeError):s.step()
    assert len(calls)==attempted


@pytest.mark.parametrize('planned',[False,True])
def test_partial_sink_failure_commits_batch_once_and_forbids_retry(planned):
    s,events,calls,_=setup(fault='sink',planned=planned)
    with pytest.raises(RuntimeError):s.step()
    assert s.failed and [e.token_id for e in events]==[7,8,9]
    assert int(s._state.position[0])==6 and int(s._native.cache.position[0])==6
    with pytest.raises(RuntimeError):s.step()
    assert calls==[3]


def test_host_acceptance_matches_device_rule_over_rejections_eos_and_caps():
    rng=np.random.default_rng(17)
    for rows in (1,2,3):
        for _ in range(30):
            ids=rng.integers(0,4,rows,dtype=np.int32)
            predictions=rng.integers(0,4,rows,dtype=np.int32)
            remaining=int(rng.integers(1,5))
            host=accept_greedy(ids,predictions,remaining,(0,))
            device=greedy_acceptance(jnp.asarray(ids),jnp.asarray(predictions),jnp.int32(remaining),eos_token_ids=(0,))
            assert host.count==int(device.emitted_count)
            assert host.accepted_drafts==int(device.accepted_draft_count)
            assert (host.reason=='eos')==bool(device.stopped_on_eos)
            assert (host.reason=='length')==bool(device.stopped_on_length)


@pytest.mark.parametrize('planned',[False,True])
def test_resume_after_rejection_keeps_accepted_frontier(planned):
    s,events,calls,roots=setup(drafts=(77,9),planned=planned)
    s.step()
    accepted=s._native
    assert int(accepted.cache.position[0])==4
    s.step()
    assert roots[-1] is accepted and int(s._native.cache.position[0])==5
    assert [e.index for e in events]==[0,1,2]


@pytest.mark.parametrize('fault',['metadata','device_count'])
def test_device_plan_fault_preserves_roots_and_forbids_delivery(fault):
    s,events,_,_=setup(fault=fault,planned=True)
    target,native=s._state,s._native
    with pytest.raises(RuntimeError):s.step()
    assert s.failed and len(events)==1 and s._state is target and s._native is native
