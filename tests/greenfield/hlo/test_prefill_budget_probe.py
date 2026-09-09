"""Missing-budget fixture/production-function tests, CPU only."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.greenfield import prefill_budget_probe as probe


def test_fixed_six_cases_cover_disjoint_physical_stripes_and_last_query():
    cases = probe.cases()
    assert len(cases) == len({c.name for c in cases}) == 6
    for capacity, prompt in probe.CAPACITIES:
        all_positions = probe.stripe_positions(capacity, 0, capacity)
        np.testing.assert_array_equal(np.sort(all_positions), np.arange(capacity))
        for expert, positions in enumerate(all_positions.reshape(8, -1)):
            assert np.all(np.diff(positions) > 0)
            assert np.all(positions // 64 % 8 == expert)
        selected = [c for c in cases if c.capacity == capacity]
        assert [c.last_valid_length for c in selected] == [2048, prompt // 2, prompt]
        for case in selected:
            np.testing.assert_array_equal(np.diff(case.valid_lengths), 1)
            assert case.valid_lengths[-1] == case.last_valid_length


@pytest.mark.parametrize(
    "values",
    [
        (131072, 127363, 127364),
        (131072, 131072, 2048),
        (True, 127363, 2048),
        (262144, 262144, 2048),
    ],
)
def test_unregistered_workloads_refuse(values):
    with pytest.raises(ValueError):
        probe.BudgetCase(*values)


def test_analytic_expected_scores_and_lowest_position_ties():
    lengths = np.asarray([0, 17, 2050, 127363, 262144], np.int32)
    for tied in (False, True):
        expected = probe.expected_selection(lengths, tied=tied)
        for row, length in enumerate(lengths):
            count = min(int(length), 2048)
            actual = expected["positions"][row, :count]
            assert len(set(actual)) == count and np.all(actual < length)
            assert np.all(expected["positions"][row, count:] == -1)
            assert np.isneginf(expected["scores"][row, count:]).all()
            # Independent bucket ordering, not another argsort implementation.
            ordered = []
            p = np.arange(length)
            values = np.zeros_like(p) if tied else (p * (row + 1) + row) % 251
            for score in range(250, -1, -1):
                ordered.extend(p[values == score].tolist())
                if len(ordered) >= count:
                    break
            np.testing.assert_array_equal(actual, np.asarray(ordered[:count], np.int32))


def test_delivery_conversion_and_callback_are_inside_ack_boundary():
    events = []
    times = iter((2.0, 2.25, 2.75))

    class Token:
        def __array__(self, dtype=None, copy=None):
            events.append("convert")
            return np.asarray([220], np.int32)

    def clock():
        events.append("clock")
        return next(times)

    report = probe.deliver_local_token(
        Token(), lambda token: events.append(token), clock=clock
    )
    assert events == ["clock", "convert", "clock", 220, "clock"]
    assert report["device_to_host_seconds"] == 0.25
    assert report["callback_ack_seconds"] == 0.5
    assert report["delivery_seconds"] == 0.75
    assert report["model_ttft_measured"] is False
    assert report["delivery_scope"] == probe.DELIVERY_SCOPE


def test_failed_or_invalid_delivery_never_returns_success():
    calls = []
    for token in (
        np.asarray([-1], np.int32),
        np.asarray([1, 2], np.int32),
        np.asarray([1.0], np.float32),
    ):
        with pytest.raises(ValueError):
            probe.deliver_local_token(token, calls.append)
    assert not calls

    def fail(token):
        raise RuntimeError("consumer rejected token")

    with pytest.raises(RuntimeError, match="consumer rejected"):
        probe.deliver_local_token(np.asarray([220], np.int32), fail)


@pytest.mark.parametrize("sorted_local_merge", [False, True])
def test_production_dsa_cpu32_analytic_and_tied_global_selection(sorted_local_merge):
    code = r"""
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield import prefill_budget_probe as b
mesh=Mesh(np.asarray(jax.devices()).reshape(8,4),('expert','feature'))
assert jax.default_backend()=='cpu' and jax.device_count()==32
capacity,rows,topk=4096,2,64
positions=b.stripe_positions(capacity,0,capacity)
keys=b.key_values(positions)
lengths=np.asarray([2050,4096],np.int32)
specs=(P(),P('expert',None),P(),P('expert'),P())
program=b.build_dsa_program(mesh,capacity=capacity,rows=rows,top_k=topk,sorted_local_merge=SORTED_LOCAL)
for tied in (False,True):
 q,h=b.query_inputs(rows=rows,tied=tied)
 args=tuple(jax.device_put(v,NamedSharding(mesh,s)) for v,s in zip((q,keys,h,positions,lengths),specs))
 result=program(*args)
 jax.block_until_ready(result)
 expected=b.expected_selection(lengths,top_k=topk,tied=tied)
 report=b.check_output(result,expected)
 assert report['passed'] and len(report['owners'])==32
 corrupted={k:v.copy() for k,v in expected.items()}
 corrupted['positions'][0,0]=-1
 try:b.check_output(result,corrupted)
 except ValueError:pass
 else:raise AssertionError('changed selected position accepted')
# Production-capacity inputs use callback-owned stripes, not replicated keys.
case=b.cases()[0]
args=b.make_dsa_inputs(mesh,case)
assert args[1].shape==(131072,128)
assert args[1].addressable_shards[0].data.shape==(16384,128)
assert args[3].addressable_shards[0].data.shape==(16384,)
assert args[4].shape==(32,)
abstract=b.build_dsa_program(mesh,capacity=case.capacity,sorted_local_merge=SORTED_LOCAL).lower(*args)
text=str(abstract.compiler_ir('stablehlo'))
assert 'all_gather' in text
assert 'callback' not in text and 'host_transfer' not in text
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield.run_short_decoder_ws32 import _compiled_memory
for capacity,prompt in b.CAPACITIES:
 case=b.BudgetCase(capacity,prompt,2048)
 values=b.make_dsa_inputs(mesh,case)
 lowered=b.build_dsa_program(mesh,capacity=capacity,sorted_local_merge=SORTED_LOCAL).lower(*values)
 compiled=lowered.compile()
 report=worker.inspect_program('dsa_c'+str(capacity),str(lowered.compiler_ir('stablehlo')),compiled.as_text(),_compiled_memory(compiled))
 assert report['passed'] and report['exchanged_candidate_leaves']=={'s32':1,'f32':1}
print('CPU32_BUDGET_DSA_PASS')
"""
    code = code.replace("SORTED_LOCAL", repr(sorted_local_merge))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32_BUDGET_DSA_PASS" in result.stdout


@pytest.mark.parametrize("failure", [None, "peer", "check", "deadline", "preserve"])
def test_sampler_fixed_order_votes_preservation_and_deadline(monkeypatch, failure):
    from types import SimpleNamespace

    events, saved = [], []
    current = 0.0

    def clock():
        nonlocal current
        current += 0.001
        return current

    def program(*args):
        events.append("execute")
        return object()

    def check(output, expected):
        events.append("check")
        if failure == "check":
            raise ValueError("invalid original")
        return {"passed": True}

    monkeypatch.setattr(probe, "check_output", check)

    def preserve(index, output):
        events.append("preserve")
        saved.append(index)
        if failure == "preserve":
            raise ValueError("publication failed")

    def phase(name, action):
        events.append(name)
        result = action()
        if failure == "peer" and name == "case/sample0/execute":
            assert saved == [0]
            raise ValueError("peer failed after local completion")
        return result

    calls = SimpleNamespace(phase=phase, record={"call_evidence": []})

    def call(label, graph, inputs, *, preserve):
        def execute():
            output = program(*inputs)
            calls.record["call_evidence"].append({"completed_call_seconds": 0.001})
            preserve(output)
            return output

        return phase(label + "/execute", execute)

    calls.call = call
    kwargs = dict(
        case_name="case",
        preserve=preserve,
        budget_started=-121 if failure == "deadline" else 0,
        clock=clock,
    )
    if failure:
        with pytest.raises(ValueError):
            probe.sample_dsa(calls, "dsa", (), {}, **kwargs)
        assert events.count("execute") == (0 if failure == "deadline" else 1)
        if failure != "deadline":
            assert saved and saved[0] == 0
        assert "case/sample1/execute" not in events
    else:
        report = probe.sample_dsa(calls, "dsa", (), {}, **kwargs)
        assert saved == [0, 6] and events.count("execute") == 7
        assert len(report["samples_seconds"]) == 5
        np.testing.assert_allclose(report["samples_seconds"], 0.001)
        assert report["model_prefill_measured"] is False
        assert events[-1] == "case/sample6_post_budget"


def test_initializer_reuses_production_allocation_and_waits(monkeypatch):
    from types import SimpleNamespace
    import jax
    from glm_tpu.greenfield.runtime import ws32_batched_prefill as production

    events = []
    sentinel = object()
    config = SimpleNamespace(context_capacity=262656)

    def allocate(mesh, cfg, *, prompt_length):
        assert mesh == "mesh" and cfg is config and prompt_length == 262144
        events.append("allocate")
        return sentinel

    monkeypatch.setattr(production, "make_ws32_batched_prefill_state", allocate)
    monkeypatch.setattr(
        jax,
        "block_until_ready",
        lambda state: events.append("ready") if state is sentinel else pytest.fail(),
    )
    times = iter((3.0, 3.25))
    state, report = probe.measure_initial_state(
        "mesh", config, prompt_length=262144, clock=lambda: next(times)
    )
    assert state is sentinel and events == ["allocate", "ready"]
    assert report["cache_initialization_seconds"] == 0.25
    assert (
        report["weights_loaded"] is False and report["initialized_prefix_length"] == 0
    )
