"""Tests of :mod:`glm_tpu.engine.llm_engine`, CPU only, over synthetic device results: ``LLMEngine.generate``
(one fresh request: delivery, the deadline votes, no retry after a failure, the loaded capacity) and
``LLMEngine.generate_concurrent`` (the full prefill-to-batch host path with independent prompt lengths)."""

from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.engine import request
from glm_tpu.engine.llm_engine import LLMEngine
from glm_tpu.models.glm_moe_dsa.model import BatchedDecodeResult, PackedDecodeResult
from glm_tpu.models.glm_moe_dsa.state import DecodeStepResult
from glm_tpu.runner.tpu_runner import TPUModelRunner
from tests.engine.test_request_session import state, prefill


def fixture(monkeypatch, *, late=False):
    runtime = object.__new__(TPUModelRunner)
    runtime.capacity = 8192
    runtime.concurrent_size = 0
    runtime.put = lambda x: x
    runtime.weights = runtime.wk = runtime.rope = None
    runtime.record = {"requests": []}
    runtime.save = lambda record: None
    runtime.admit = lambda name: None
    runtime.stats = lambda: []
    runtime.initialize = lambda count: SimpleNamespace(decoder=state(0))
    runtime.vote = lambda value: value
    runtime.phase = lambda name, fn: fn()
    ticks = [1.0]
    calls = []

    def execute_prefill(*args):
        ticks[0] += 0.5
        return prefill()

    runtime.prefill = {114: execute_prefill, 128: execute_prefill}

    def decode(token, previous, *args):
        i = len(calls) + 1
        calls.append(int(token[0]))
        ticks[0] += 100 if late else 0.25
        out = DecodeStepResult(state(3 + i), np.array([8 + i], np.int32), np.zeros((1, 1)))
        return PackedDecodeResult(out, np.array([8 + i, 1, 3 + i, 4 + i], np.int32))

    runtime.decode = decode
    from jax.experimental import multihost_utils

    monkeypatch.setattr(multihost_utils, "process_allgather", lambda value: np.stack([value] * 8))
    return LLMEngine(runtime), ticks, calls


def test_fresh_request_delivery_and_terminal_release(monkeypatch):
    engine, ticks, calls = fixture(monkeypatch)
    value = request.from_token_ids([30, 31, 32], request_id="fixture", max_new_tokens=3)
    events = []
    tokens, report = engine.generate(value, deliver=events.append, deadline=50, clock=lambda: ticks[0])
    np.testing.assert_array_equal(tokens, [7, 9, 10])
    assert calls == [7, 9] and len(events) == 3 and not engine.active
    assert report["timed_decode_tokens"] == 2 and report["finish_reason"] == "length"
    assert report["decode_wall_seconds"] == 0.5
    assert engine.runner.record["requests"] == [report]


def test_deadline_votes_before_delivery_and_never_retries(monkeypatch):
    engine, ticks, calls = fixture(monkeypatch, late=True)
    value = request.from_token_ids([30, 31, 32], request_id="fixture", max_new_tokens=3)
    events = []
    with pytest.raises(RuntimeError):
        engine.generate(value, deliver=events.append, deadline=50, clock=lambda: ticks[0])
    assert len(events) == 1 and engine.active and calls == [7]
    with pytest.raises(RuntimeError):
        engine.generate(value, deliver=events.append, deadline=500, clock=lambda: ticks[0])
    assert calls == [7] and not engine.runner.record["requests"]


def test_capacity_mismatch_refuses_before_execution(monkeypatch):
    engine, ticks, calls = fixture(monkeypatch)
    value = request.from_token_ids([1, 2], request_id="long", max_new_tokens=3, context_capacity=request.LONG_CAPACITY)
    with pytest.raises(RuntimeError, match="capacity differs"):
        engine.generate(value, deliver=lambda event: None, deadline=50, clock=lambda: ticks[0])
    assert not calls and not engine.runner.record["requests"]


def test_next_question_starts_fresh_with_separate_delivery(monkeypatch):
    engine, ticks, calls = fixture(monkeypatch)
    for name in ("first", "second"):
        calls.clear()
        events = []
        value = request.from_token_ids([30, 31, 32], request_id=name, max_new_tokens=3)
        tokens, _report = engine.generate(value, deliver=events.append, deadline=50, clock=lambda: ticks[0])
        assert [e.index for e in events] == [0, 1, 2]
        assert {e.request_id for e in events} == {name}
        np.testing.assert_array_equal(tokens, [7, 9, 10])
    assert len(engine.runner.record["requests"]) == 2


def test_prefill_commits_each_history_before_shared_decode(monkeypatch):
    import jax
    from glm_tpu.models.glm_moe_dsa import state

    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr("jax.experimental.multihost_utils.process_allgather", lambda x: np.tile(x, (8, 1)))
    monkeypatch.setattr(state, "finish_batched_prefill", lambda out: (out.state.decoder, out.next_token))
    values = [
        request.from_token_ids([7] * length, request_id=f"lane{i}", max_new_tokens=4, context_capacity=32768)
        for i, length in enumerate((129, 257, 7))
    ]
    admissions = []
    inserts = []
    decodes = []
    events = []
    ticks = [0.0]

    def initialize(length):
        return SimpleNamespace(decoder=SimpleNamespace(position=0, contract_valid=np.array([True])))

    def prefill(block, count, fresh, *weights):
        ticks[0] += 0.1
        assert np.all(block[int(count) :] == -1)
        fresh.decoder.position += int(count)
        return SimpleNamespace(state=fresh, next_token=np.array([9], np.int32))

    def insert(bank, one, index):
        inserts.append(int(index))
        bank[int(index)] = one.position
        return bank

    def decode(tokens, bank, *rest):
        active = rest[-1]
        decodes.append(active.copy())
        ticks[0] += 1
        assert bank.tolist() == [129, 257, 7]
        status = np.array([[values[i]["eos_ids"][0], 1, int(p) + 1, int(p) + 2] for i, p in enumerate(bank)], np.int32)
        return BatchedDecodeResult(bank, tokens, status)

    r = SimpleNamespace(
        concurrent_size=3,
        capacity=32768,
        require=lambda ok, message: None if ok else (_ for _ in ()).throw(RuntimeError(message)),
        vote=lambda x: bool(x),
        put=lambda x: x,
        admit=admissions.append,
        initialize_batch=lambda lengths: np.zeros(3, np.int32),
        initialize=initialize,
        prefill={114: prefill, 128: prefill},
        insert_batch=insert,
        decode_batch=decode,
        weights=None,
        wk=None,
        rope=None,
        phase=lambda name, fn: fn(),
        stats=lambda: [],
        record={"requests": []},
        save=lambda x: None,
    )
    engine = LLMEngine(r)
    results, aggregate = engine.generate_concurrent(
        values, deliver=lambda *event: events.append(event), deadline=100, clock=lambda: ticks[0]
    )
    assert inserts == [0, 1, 2] and len(decodes) == 1 and len(events) == 6
    assert all(len(tokens) == 2 and report["finish_reason"] == "eos" for tokens, report in results)
    assert aggregate["decode_rounds"] == 1 and aggregate["aggregate_decode_tokens_per_second"] == 3
    assert admissions.count("cache_init") == 3 and admissions.count("batch_cache_init") == 1
    assert admissions[-1] == "batch_decode" and not engine.active
    engine.active = True
    with pytest.raises(RuntimeError, match="active or failed"):
        engine.generate_concurrent(values, deliver=lambda *args: None, deadline=100)
    assert len(decodes) == 1
