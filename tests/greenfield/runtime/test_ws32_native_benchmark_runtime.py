"""Actual request/input/session wiring with fake compiled math, CPU only."""
from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
    Ws32BatchedPrefillState, Ws32BatchedPrefillResult,
)
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState, Ws32DecodeStepResult
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from scripts.greenfield import ws32_native_benchmark_runtime as native
from scripts.greenfield import ws32_batched_prefill_runner as original


@pytest.mark.parametrize("length", [1, 113, 114, 115, 127, 128, 129, 2034, 8155])
def test_prompt_schedule_covers_every_live_token_with_two_fixed_shapes(length):
    blocks = native.prompt_blocks(length)
    assert sum(live for _, live, _ in blocks) == length
    assert all(physical in (114, 128) and 0 < live <= physical for _, live, physical in blocks)
    assert [i for offset, live, _ in blocks for i in range(offset, offset+live)] == list(range(length))
    if length == 2034:
        assert [physical for _, _, physical in blocks] == [128]*15+[114]


def build_runtime(monkeypatch, *, refusal=None):
    geometry = SimpleNamespace(vocab_size=256)
    raw = SimpleNamespace(exact_dsa=False, strategy_nd_dense=False, host_main_rope_table=True,
                          geometry=geometry, context_capacity=8192, full_index_slots=(0,))
    decode = SimpleNamespace(exact_dsa=True, host_main_rope_table=True, geometry=geometry,
                             context_capacity=8192)
    calls, allocations, admissions, emitted = [], [], [], []

    def fresh(mesh, config, *, prompt_length):
        state = Ws32DecoderState(
            np.zeros(1), np.zeros(1), np.zeros((1, 1), np.int32), np.ones(1, np.int32),
            np.zeros((1, 1), np.float32), np.array([0], np.int32),
            np.zeros((1, 1), np.int32), np.array([1], np.int32), np.array([True]),
        )
        allocations.append(state)
        return Ws32BatchedPrefillState(state, np.zeros(1),
                                      np.array(prompt_length, np.int32), np.array(False))

    class Compiled:
        def __init__(self, rows=None): self.rows = rows
        def memory_analysis(self): return None  # fake compiler, never admission evidence
        def __call__(self, *args):
            if self.rows is None:
                token, state, weights, exact, rope, uniform = args
                calls.append(('decode', int(token[0]), float(uniform)))
                return Ws32DecodeStepResult(state._replace(
                    position=state.position+1, context_lengths=state.context_lengths+1),
                    np.array([10], np.int32), np.zeros((1, 1)))
            tokens, count, state, weights, wk, rope, uniform = args
            assert tokens.shape == (self.rows,) and int(count) <= self.rows
            assert np.all(tokens[int(count):] == 0)
            calls.append(('prefill', int(state.decoder.position[0]), int(count), self.rows, float(uniform)))
            end = int(state.decoder.position[0]) + int(count)
            final = end == int(state.prompt_length)
            return Ws32BatchedPrefillResult(state._replace(decoder=state.decoder._replace(
                position=np.array([end], np.int32), context_lengths=np.array([end+1], np.int32)),
                finished=np.array(final)), np.array([7 if final else -1], np.int32))

    def authorize(stage, roots, state):
        admissions.append(stage)
        assert set(roots) == {'raw_weights', 'decode_weights', 'wk', 'exact_weights', 'rope'}
        if stage == refusal: raise ValueError('forced memory refusal')

    monkeypatch.setattr(native, 'make_ws32_batched_prefill_state', fresh)
    monkeypatch.setattr(native, 'replicated', lambda mesh, value:np.asarray(value))
    # Keep the original padding/count helper; replace only CPU device transfer.
    monkeypatch.setattr(original, 'replicated', lambda mesh, value:np.asarray(value))
    runtime = native.NativeBenchmarkRuntime(
        None, raw, decode, object(), object(), (np.zeros(1),), (object(),), np.zeros((8192, 64)),
        {114:Compiled(114), 128:Compiled(128)}, Compiled(), authorize, lambda ok:ok,
    )
    return runtime, calls, allocations, admissions, emitted


def test_two_real_host_schedules_fresh_cache_immediate_delivery_and_release(monkeypatch):
    runtime, calls, allocations, admissions, emitted = build_runtime(monkeypatch)
    for item, length in enumerate((2034, 115)):
        policy = RequestPolicy(f'item{item}', 42, length, 3, 8192, 256, (10,))
        session = runtime.start_request(np.arange(length, dtype=np.int32)%256, policy,
            deliver=emitted.append, delivery_boundary='test sink', request_started=runtime.clock())
        assert emitted[-1].token_id == 7 and not session.finished
        assert len([x for x in calls if x[0]=='decode']) == item
        with pytest.raises(RuntimeError): runtime.close_request()
        session.step()
        assert session.finished and emitted[-1].finish_reason == 'eos'
        assert session.ttft_seconds >= 0
        assert session.delivered_request_seconds >= session.ttft_seconds
        events = session.events
        runtime.close_request()
        assert runtime.active is None and session.events == events
        assert session._state is None and session._pending_token is None
    assert allocations[0] is not allocations[1]
    assert admissions == ['before_cache', 'cache_ready', 'prefill_done']*2
    assert all(allocation.position.tolist() == [0] for allocation in allocations)
    prefill = [x for x in calls if x[0]=='prefill']
    assert len(prefill) == 17 and prefill[-1][1:4] == (0, 115, 128)
    assert len(set(x[-1] for x in prefill[:16])) == 1


@pytest.mark.parametrize('stage', ['before_cache', 'cache_ready', 'prefill_done'])
def test_authorization_refusal_prevents_delivery_and_reuse(monkeypatch, stage):
    runtime, calls, allocations, _, emitted = build_runtime(monkeypatch, refusal=stage)
    policy = RequestPolicy('a', 0, 3, 4, 8192, 256, (10,))
    with pytest.raises(RuntimeError):
        runtime.start_request(np.array([1,2,3],np.int32), policy,
            deliver=emitted.append, delivery_boundary='test', request_started=runtime.clock())
    assert runtime.failed and not emitted
    assert not any(x[0]=='decode' for x in calls)
    if stage=='before_cache': assert not allocations and not calls
    with pytest.raises(RuntimeError):
        runtime.start_request(np.array([1,2,3],np.int32), policy,
            deliver=emitted.append, delivery_boundary='test', request_started=runtime.clock())


def test_failed_decode_cannot_be_followed_by_another_request(monkeypatch):
    runtime, _, _, _, emitted = build_runtime(monkeypatch)
    session = runtime.start_request(np.array([1,2,3],np.int32),
        RequestPolicy('a',0,3,4,8192,256,(10,)), deliver=emitted.append,
        delivery_boundary='test', request_started=runtime.clock())
    def fail(*args): raise RuntimeError('model dispatch failed')
    runtime.decode_compiled = fail
    with pytest.raises(RuntimeError): session.step()
    runtime.close_request()
    assert runtime.failed and runtime.active is None
