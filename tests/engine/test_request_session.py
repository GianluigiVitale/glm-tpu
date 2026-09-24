"""Request-control failures and timing, using fake math (not TPU proof)."""
from dataclasses import replace

import numpy as np
import pytest

from glm_tpu.engine.request_session import request_uniform, SampledRequestPolicy, Ws32RequestSession
from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, BatchedPrefillState, DecoderState, DecodeStepResult


def state(position=3, healthy=True):
    return DecoderState(
        np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1), np.int32),
        np.ones((1,), np.int32), np.zeros((1, 1), np.float32),
        np.array([position], np.int32), np.zeros((1, 1), np.int32),
        np.array([position + 1], np.int32), np.array([healthy]),
    )


def prefill(token=7, healthy=True):
    return BatchedPrefillResult(
        BatchedPrefillState(state(healthy=healthy), np.zeros((1,)),
                               np.array(3, np.int32), np.array(True)),
        np.array([token], np.int32),
    )


def setup(*, max_new=4, deliver=None, vote=None, outputs=(9, 10)):
    clock = [10.0]
    calls, draws, emitted = [], [], []
    policy = SampledRequestPolicy("request-a", 42, 3, max_new, 20, 256, (10,))

    def decode(token, previous, uniform):
        calls.append((token.copy(), previous))
        draws.append(float(uniform))
        clock[0] += .25
        return DecodeStepResult(state(3 + len(calls)),
                                   np.array([outputs[len(calls) - 1]], np.int32),
                                   np.zeros((1, 1)))

    def sink(event):
        emitted.append(event)
        clock[0] += .5
        if deliver is not None:
            deliver(event)

    session = Ws32RequestSession(
        policy, decode_step=decode, replicate_uniform=lambda x:x,
        fleet_all=vote or (lambda x:x), deliver=sink,
        delivery_boundary="fake unit-test sink", request_started=1.0,
        clock=lambda:clock[0],
    )
    return session, calls, draws, emitted


def test_first_token_delivery_eos_live_resume_and_rng_frontier():
    session, calls, draws, emitted = setup()
    assert float(session.next_uniform()) == request_uniform(seed=42, request_id="request-a", token_index=0)
    session.accept_prefill(prefill())
    assert session.ttft_seconds == 9.5 and calls == []
    assert session.events[0].token_id == 7
    paused_session = session
    paused_session.step()
    paused_session.step()
    assert [e.token_id for e in emitted] == [7, 9, 10]
    assert emitted[-1].finish_reason == "eos" and session.finished
    assert [int(x[0][0]) for x in calls] == [7, 9]
    assert draws == [request_uniform(seed=42, request_id="request-a", token_index=i) for i in (1, 2)]
    assert session.decode_seconds == (.25, .25)
    assert session.delivered_request_seconds == 11.0
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 2


@pytest.mark.parametrize("token,reason", [(7, "length"), (10, "eos")])
def test_first_token_can_finish_without_decode(token, reason):
    session, calls, _, emitted = setup(max_new=1)
    session.accept_prefill(prefill(token))
    assert emitted[0].finish_reason == reason and not calls
    with pytest.raises(RuntimeError):
        session.next_uniform()


def test_unhealthy_prefill_is_never_emitted_and_cannot_retry():
    session, calls, _, emitted = setup()
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill(healthy=False))
    assert session.failed and not emitted and not calls
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())


def test_sink_failure_preserves_generated_token_but_no_false_delivery_or_retry():
    def fail(event):
        raise OSError("sink disconnected after possibly writing bytes")
    session, calls, _, emitted = setup(deliver=fail)
    with pytest.raises(RuntimeError) as refused:
        session.accept_prefill(prefill())
    assert isinstance(refused.value.__cause__, OSError)
    assert session.failed and session.events[0].token_id == 7
    assert len(emitted) == 1 and session.ttft_seconds is None
    with pytest.raises(RuntimeError):
        session.step()
    assert calls == []


def test_remote_refusal_never_emits_local_healthy_token():
    session, _, _, emitted = setup(vote=lambda _:False)
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and emitted == []


def test_nonboolean_vote_cannot_admit_execution():
    session, calls, _, emitted = setup(vote=lambda _:"yes")
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and not calls and not emitted


def test_dispatch_exception_poisoned_without_duplicate_first_token():
    session, calls, _, emitted = setup(outputs=())
    session.accept_prefill(prefill())
    with pytest.raises(IndexError):
        session.step()
    assert session.failed and len(calls) == 1 and len(emitted) == 1
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 1


def test_wrong_frontier_or_token_cannot_be_delivered():
    for result in (prefill()._replace(next_token=np.array([-1], np.int32)),
                   prefill()._replace(state=prefill().state._replace(decoder=state(4)))):
        session, _, _, emitted = setup()
        with pytest.raises(RuntimeError):
            session.accept_prefill(result)
        assert not emitted and session.failed


def test_no_decode_before_prefill_and_no_second_prefill():
    session, calls, _, _ = setup()
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and not calls
    session, _, _, emitted = setup()
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and len(emitted) == 1


def test_request_policy_preserves_registered_cap():
    policy = SampledRequestPolicy("a", 0, 3, 4, 20, 256, (10,))
    for updates in ({"max_new_tokens": 18}, {"max_new_tokens": True},
                    {"eos_ids": ()}, {"eos_ids": (10, 10)},
                    {"eos_ids": (256,)}, {"seed": -1}, {"request_id": ""}):
        with pytest.raises(ValueError):
            replace(policy, **updates)
