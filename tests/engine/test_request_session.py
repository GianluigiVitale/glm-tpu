"""Request-control failures and timing, using fake math (not TPU proof).

The sequential ``RequestSession`` (one greedy token per step) and the concurrent ``BatchedSession``. Until S5 WU-E2
the first tests below ran the sampled base session and its seeded policy (``Ws32RequestSession``,
``SampledRequestPolicy``), which production never used; they now run the greedy session with the same fakes. The
packed-status tests after them came from ``tests/models/glm_moe_dsa/test_decode_program.py`` at S5 WU-E2. Until S2f
those also ran the frozen sampled session and policy they replaced (same events and timing, one fewer vote per
decode, identical refusals at seed 0). That oracle is archived at ``archive/research-20260922``; its final green run
is recorded in the S2f commit message, and the frozen policy's refusals at seed 0 are pinned below as data
(``POLICY_CASES``).
"""

from dataclasses import replace

import numpy as np
import pytest

from glm_tpu.engine.request_session import RequestPolicy, RequestSession, BatchedSession
from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, BatchedPrefillState, DecoderState, DecodeStepResult
from glm_tpu.engine import request
from glm_tpu.models.glm_moe_dsa.model import BatchedDecodeResult, PackedDecodeResult


def state(position=3, healthy=True):
    return DecoderState(
        np.zeros((1,)),
        np.zeros((1,)),
        np.zeros((1, 1), np.int32),
        np.ones((1,), np.int32),
        np.zeros((1, 1), np.float32),
        np.array([position], np.int32),
        np.zeros((1, 1), np.int32),
        np.array([position + 1], np.int32),
        np.array([healthy]),
    )


def prefill(token=7, healthy=True):
    return BatchedPrefillResult(
        BatchedPrefillState(state(healthy=healthy), np.zeros((1,)), np.array(3, np.int32), np.array(True)),
        np.array([token], np.int32),
    )


def setup(*, max_new=4, deliver=None, vote=None, outputs=(9, 10)):
    clock = [10.0]
    calls, emitted = [], []
    policy = RequestPolicy("request-a", 3, max_new, 20, 256, (10,))

    def decode(token, previous):
        calls.append((token.copy(), previous))
        clock[0] += 0.25
        output = outputs[len(calls) - 1]
        result = DecodeStepResult(state(3 + len(calls)), np.array([output], np.int32), np.zeros((1, 1)))
        return PackedDecodeResult(result, np.array([output, 1, 3 + len(calls), 4 + len(calls)], np.int32))

    def sink(event):
        emitted.append(event)
        clock[0] += 0.5
        if deliver is not None:
            deliver(event)

    session = RequestSession(
        policy,
        decode_step=decode,
        fleet_all=vote or (lambda x: x),
        deliver=sink,
        delivery_boundary="fake unit-test sink",
        request_started=1.0,
        clock=lambda: clock[0],
    )
    return session, calls, emitted


def test_first_token_delivery_eos_live_resume_and_frontier():
    session, calls, emitted = setup()
    session.accept_prefill(prefill())
    assert session.ttft_seconds == 9.5 and calls == []
    assert session.events[0].token_id == 7
    paused_session = session
    paused_session.step()
    paused_session.step()
    assert [e.token_id for e in emitted] == [7, 9, 10]
    assert emitted[-1].finish_reason == "eos" and session.finished
    assert [int(x[0][0]) for x in calls] == [7, 9]
    assert session.decode_seconds == (0.25, 0.25)
    assert session.delivered_request_seconds == 11.0
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 2


@pytest.mark.parametrize("token,reason", [(7, "length"), (10, "eos")])
def test_first_token_can_finish_without_decode(token, reason):
    session, calls, emitted = setup(max_new=1)
    session.accept_prefill(prefill(token))
    assert emitted[0].finish_reason == reason and not calls
    with pytest.raises(RuntimeError):
        session.step()
    assert not calls


def test_unhealthy_prefill_is_never_emitted_and_cannot_retry():
    session, calls, emitted = setup()
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill(healthy=False))
    assert session.failed and not emitted and not calls
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())


def test_sink_failure_preserves_generated_token_but_no_false_delivery_or_retry():
    def fail(event):
        raise OSError("sink disconnected after possibly writing bytes")

    session, calls, emitted = setup(deliver=fail)
    with pytest.raises(RuntimeError) as refused:
        session.accept_prefill(prefill())
    assert isinstance(refused.value.__cause__, OSError)
    assert session.failed and session.events[0].token_id == 7
    assert len(emitted) == 1 and session.ttft_seconds is None
    with pytest.raises(RuntimeError):
        session.step()
    assert calls == []


def test_remote_refusal_never_emits_local_healthy_token():
    session, _, emitted = setup(vote=lambda _: False)
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and emitted == []


def test_nonboolean_vote_cannot_admit_execution():
    session, calls, emitted = setup(vote=lambda _: "yes")
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and not calls and not emitted


def test_dispatch_exception_poisoned_without_duplicate_first_token():
    session, calls, emitted = setup(outputs=())
    session.accept_prefill(prefill())
    with pytest.raises(IndexError):
        session.step()
    assert session.failed and len(calls) == 1 and len(emitted) == 1
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 1


def test_wrong_frontier_or_token_cannot_be_delivered():
    for result in (
        prefill()._replace(next_token=np.array([-1], np.int32)),
        prefill()._replace(state=prefill().state._replace(decoder=state(4))),
    ):
        session, _, emitted = setup()
        with pytest.raises(RuntimeError):
            session.accept_prefill(result)
        assert not emitted and session.failed


def test_no_decode_before_prefill_and_no_second_prefill():
    session, calls, _ = setup()
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and not calls
    session, _, emitted = setup()
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.accept_prefill(prefill())
    assert session.failed and len(emitted) == 1


def test_request_policy_preserves_registered_cap():
    policy = RequestPolicy("a", 3, 4, 20, 256, (10,))
    for updates in (
        {"max_new_tokens": 18},
        {"max_new_tokens": True},
        {"eos_ids": ()},
        {"eos_ids": (10, 10)},
        {"eos_ids": (256,)},
        {"request_id": ""},
    ):
        with pytest.raises(ValueError):
            replace(policy, **updates)


def test_constructor_refuses_a_foreign_policy_before_reading_the_clock():
    reads = []

    def clock():
        reads.append(None)
        return 10.0

    arguments = dict(decode_step=None, fleet_all=None, deliver=None, delivery_boundary="fake sink", clock=clock)
    with pytest.raises(ValueError, match="an explicit greedy RequestPolicy is required"):
        RequestSession(("request-a", 3, 4, 20, 256, (10,)), request_started=11.0, **arguments)
    assert reads == []
    with pytest.raises(ValueError, match="request clock and named delivery boundary required"):
        RequestSession(RequestPolicy("request-a", 3, 4, 20, 256, (10,)), request_started=11.0, **arguments)
    assert len(reads) == 1


def packed_setup(packed=True, *, mutate=None, vote=None, sink_error=False, outputs=(9, 10), max_new=4):
    assert packed  # the greedy release session is the only session
    clock = [10.0]
    calls, draws, events, votes = [], [], [], []

    def decode(token, previous, *uniform):
        i = len(calls) + 1
        calls.append(token)
        draws.extend(float(u) for u in uniform)
        clock[0] += 0.25
        out = DecodeStepResult(state(3 + i), np.array([outputs[i - 1]], np.int32), np.zeros((1, 1)))
        status = np.array([outputs[i - 1], 1, 3 + i, 4 + i], np.int32)
        return PackedDecodeResult(out, mutate(status) if mutate else status)

    def sink(event):
        events.append(event)
        clock[0] += 0.5
        if sink_error and event.index > 0:
            raise OSError("sink failed after possible emission")

    def fleet(valid):
        votes.append(valid)
        return vote(valid, len(votes)) if vote else valid

    session = RequestSession(
        RequestPolicy("request-a", 3, max_new, 20, 256, (10,)),
        decode_step=decode,
        fleet_all=fleet,
        deliver=sink,
        delivery_boundary="fake sink",
        request_started=1.0,
        clock=lambda: clock[0],
    )
    return session, calls, draws, events, votes, [], clock


def test_packed_loop_events_pause_eos_and_boundary_work():
    session, calls, draws, events, votes, _, _ = packed_setup()
    session.accept_prefill(prefill())
    session.step()
    paused = session
    paused.step()
    assert session.finished and not session.failed
    assert session.decode_seconds == (0.25, 0.25)
    assert session.ttft_seconds == 9.5 and session.delivered_request_seconds == 11.0
    with pytest.raises(RuntimeError):
        session.step()
    session.release()
    assert session._state is None
    assert [(e.index, e.token_id, e.finish_reason) for e in events] == [(0, 7, None), (1, 9, None), (2, 10, "eos")]
    assert draws == []  # the greedy session never draws a uniform
    assert [int(v[0]) for v in calls] == [7, 9]
    assert len(votes) == 7  # prefill: 3 votes, then 2 per decode


# (arguments, outcome of the frozen sampled policy at seed 0 -- the release's value -- recorded from
# the final run of the frozen policy at S2f: None, or (exception type name, message))
POLICY_CASES = [
    (("request-a", 3, 4, 20, 256, (10,)), None),
    (("r", 1, 1, 2, 1, (0,)), None),
    (("", 3, 4, 20, 256, (10,)), ("ValueError", "a nonempty request id is required")),
    ((None, 3, 4, 20, 256, (10,)), ("ValueError", "a nonempty request id is required")),
    ((7, 3, 4, 20, 256, (10,)), ("ValueError", "a nonempty request id is required")),
    (
        ("\ud800", 3, 4, 20, 256, (10,)),
        ("UnicodeEncodeError", "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed"),
    ),
    (("ok", 0, 4, 20, 256, (10,)), ("ValueError", "positive integer request dimensions required")),
    (("ok", True, 4, 20, 256, (10,)), ("ValueError", "positive integer request dimensions required")),
    (("ok", 3, 4.0, 20, 256, (10,)), ("ValueError", "positive integer request dimensions required")),
    (("ok", 3, 4, -20, 256, (10,)), ("ValueError", "positive integer request dimensions required")),
    (("ok", 3, 4, 20, 0, (10,)), ("ValueError", "positive integer request dimensions required")),
    (("ok", 3, 18, 20, 256, (10,)), ("ValueError", "full registered generation cap must fit; no silent truncation")),
    (("ok", 3, 4, 20, 256, [10]), ("ValueError", "unique in-vocabulary EOS ids required")),
    (("ok", 3, 4, 20, 256, ()), ("ValueError", "unique in-vocabulary EOS ids required")),
    (("ok", 3, 4, 20, 256, (10, 10)), ("ValueError", "unique in-vocabulary EOS ids required")),
    (("ok", 3, 4, 20, 256, (256,)), ("ValueError", "unique in-vocabulary EOS ids required")),
    (("ok", 3, 4, 20, 256, (True,)), ("ValueError", "unique in-vocabulary EOS ids required")),
    (("", 0, 0, 0, 0, ()), ("ValueError", "a nonempty request id is required")),
    (("ok", 0, 0, 0, 0, [1, 1]), ("ValueError", "positive integer request dimensions required")),
]


def _outcome(factory, *args):
    try:
        factory(*args)
    except Exception as exc:
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize(("args", "expected"), POLICY_CASES)
def test_greedy_policy_refuses_exactly_like_the_frozen_policy_at_seed_zero(args, expected):
    outcome = _outcome(RequestPolicy, *args)
    assert (None if outcome is None else (outcome[0].__name__, outcome[1])) == expected
    assert "seed" not in RequestPolicy.__dataclass_fields__


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s[:3],
        lambda s: s.astype(np.int64),
        lambda s: np.array([-1, 1, 4, 5], np.int32),
        lambda s: np.array([256, 1, 4, 5], np.int32),
        lambda s: np.array([9, 0, 4, 5], np.int32),
        lambda s: np.array([9, 1, 3, 5], np.int32),
        lambda s: np.array([9, 1, 4, 4], np.int32),
    ],
)
def test_invalid_compact_status_never_emits_or_retries(mutate):
    session, calls, _, events, *_ = packed_setup(True, mutate=mutate)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and len(events) == 1
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 1


@pytest.mark.parametrize("answer", [False, "yes"])
def test_remote_or_nonboolean_refusal_prevents_delivery(answer):
    session, _, _, events, *_ = packed_setup(True, vote=lambda v, n: answer if n == 4 else v)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and len(events) == 1


def test_sink_failure_commits_once_then_poisoned():
    session, calls, _, events, *_ = packed_setup(True, sink_error=True)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError) as error:
        session.step()
    assert isinstance(error.value.__cause__, OSError)
    assert session.failed and len(events) == 2 and len(session.events) == 2
    assert session.delivered_request_seconds == 9.5
    with pytest.raises(RuntimeError):
        session.step()
    assert len(calls) == 1


def test_length_stop_without_extra_decode_and_first_token_stop():
    for maximum in (1, 2):
        session, calls, *_ = packed_setup(True, max_new=maximum)
        session.accept_prefill(prefill())
        if maximum == 2:
            session.step()
        assert session.finished and session.events[-1].finish_reason == "length"
        assert len(calls) == maximum - 1


@pytest.mark.parametrize("invalid_time", [float("nan"), -100.0])
def test_invalid_step_clock_never_emits(invalid_time):
    def corrupt(status):
        clock[0] = invalid_time
        return status

    session, _, _, events, _, _, clock = packed_setup(True, mutate=corrupt)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and len(events) == 1


def test_remote_delivery_failure_is_terminal_after_commit():
    session, _, _, events, *_ = packed_setup(True, vote=lambda v, n: False if n == 5 else v)
    session.accept_prefill(prefill())
    with pytest.raises(RuntimeError):
        session.step()
    assert session.failed and len(events) == 2 and len(session.events) == 2
    assert session.delivered_request_seconds == 9.5


def fixture(monkeypatch, *, bad_lane=None, fail_delivery=False, deadline=100):
    monkeypatch.setattr("glm_tpu.engine.request_session.jax.block_until_ready", lambda x: x)
    values = [
        request.from_token_ids(
            list(range(1, lane + 2)), request_id=f"lane-{lane}", max_new_tokens=lane + 1, context_capacity=32768
        )
        for lane in range(8)
    ]
    calls = []
    events = []
    ticks = [0.0]

    def decode(tokens, state, active):
        calls.append(active.copy())
        ticks[0] += 1
        status = np.array(
            [[10, 1, len(v["prompt_ids"]) + len(calls), len(v["prompt_ids"]) + len(calls) + 1] for v in values],
            np.int32,
        )
        if len(calls) == 1:
            status[2, 0] = values[2]["eos_ids"][0]
        if bad_lane is not None:
            status[bad_lane, 2] += 1
        return BatchedDecodeResult(state, tokens, status)

    def deliver(lane, event, round_index):
        if fail_delivery and round_index == 1 and lane == 2:
            raise OSError("sink failed")
        events.append((lane, event, round_index))

    session = BatchedSession(
        values,
        decode=decode,
        put=lambda x: x,
        vote=lambda x: bool(x),
        deliver=deliver,
        deadline=deadline,
        clock=lambda: ticks[0],
    )
    status = np.array([[7, 1, len(v["prompt_ids"]), len(v["prompt_ids"]) + 1] for v in values], np.int32)
    return session, calls, events, status


def test_eight_streams_advance_together_and_stop_independently(monkeypatch):
    session, calls, events, status = fixture(monkeypatch)
    session.run(None, np.zeros((8, 1), np.int32), status)
    assert len(calls) == 7  # One graph call per round, never one call per lane.
    assert [len(v) for v in session.events] == [1, 2, 2, 4, 5, 6, 7, 8]
    assert session.events[2][-1].finish_reason == "eos"
    assert not calls[0][0] and not calls[1][1] and not calls[1][2]
    assert {lane for lane, _, rnd in events if rnd == 1} == set(range(1, 8))
    for lane, rows in enumerate(session.events):
        assert [e.index for e in rows] == list(range(len(rows)))
        assert all(e.request_id == f"lane-{lane}" for e in rows)


@pytest.mark.parametrize("options", [{"bad_lane": 4}, {"fail_delivery": True}, {"deadline": 0.5}])
def test_failure_poisoning_prevents_replay(monkeypatch, options):
    session, calls, events, status = fixture(monkeypatch, **options)
    with pytest.raises(RuntimeError):
        session.run(None, np.zeros((8, 1), np.int32), status)
    assert session.failed and len(calls) == 1
    before = list(events)
    with pytest.raises(RuntimeError):
        session.run(None, np.zeros((8, 1), np.int32), status)
    assert events == before and len(calls) == 1


def test_concurrent_payload_capacity_count_and_identity():
    values = [
        request.from_token_ids([1, 2], request_id=f"r-{i}", max_new_tokens=10, context_capacity=32768) for i in range(4)
    ]
    value = request.batch(values, concurrent=True)
    assert request.requests(value) == values and value["schema"] == request.CONCURRENT_SCHEMA
    for n in (1, 3, 4):
        request.validate_payload(request.batch(values[:n], concurrent=True))
    extra = request.from_token_ids([1, 2], request_id="fifth", max_new_tokens=10, context_capacity=32768)
    for bad in ([], values + [extra], values[:3] + [values[0]]):
        with pytest.raises(ValueError):
            request.batch(bad, concurrent=True)
    with pytest.raises(ValueError):
        request.batch([request.from_token_ids([1], request_id="x", max_new_tokens=1)], concurrent=True)
    with pytest.raises(ValueError):
        request.from_token_ids([1, 2], request_id="x", max_new_tokens=32767, context_capacity=32768)
    value["schema"] = request.BATCH_SCHEMA
    with pytest.raises(ValueError):
        request.validate_payload(value)


def test_full_remaining_allowance_passes_old_caps_and_stops_per_lane(monkeypatch):
    monkeypatch.setattr("glm_tpu.engine.request_session.jax.block_until_ready", lambda x: x)
    values = [
        request.from_token_ids(
            [7] * (i + 1), request_id=f"full-{i}", max_new_tokens=32768 - i - 1, context_capacity=32768
        )
        for i in range(4)
    ]
    values = request.requests(request.batch(values, concurrent=True))
    round_index = [0]

    def status():
        rows = np.array(
            [[10, 1, len(v["prompt_ids"]) + round_index[0], len(v["prompt_ids"]) + round_index[0] + 1] for v in values],
            np.int32,
        )
        for lane, stop in ((0, 2050), (1, 3072), (2, values[2]["max_new_tokens"] - 1)):
            if round_index[0] == stop:
                rows[lane, 0] = values[lane]["eos_ids"][0]
        return rows

    def decode(tokens, state, active):
        round_index[0] += 1
        if round_index[0] > 2050:
            assert not active[0]
        if round_index[0] > 3072:
            assert not active[1]
        return BatchedDecodeResult(state, tokens, status())

    session = BatchedSession(
        values, decode=decode, put=lambda x: x, vote=bool, deliver=lambda *args: None, deadline=1, clock=lambda: 0.0
    )
    session.run(None, np.zeros((4, 1), np.int32), status())
    assert [len(events) for events in session.events] == [2051, 3073, 32765, 32764]
    assert [events[-1].finish_reason for events in session.events] == ["eos", "eos", "eos", "length"]
    assert [request.stop_cause(v, e[-1].finish_reason) for v, e in zip(values, session.events, strict=False)] == [
        "eos",
        "eos",
        "eos",
        "context_exhausted",
    ]
    capped = request.from_token_ids([7], request_id="capped", max_new_tokens=2048, context_capacity=32768)
    assert request.stop_cause(capped, "length") == "output_cap"
