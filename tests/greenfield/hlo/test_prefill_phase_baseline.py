"""CPU-only sampling/trace/failure tests, never TPU performance evidence."""

from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from scripts.greenfield import prefill_phase_baseline as phase


def entries():
    return [
        dict(graph=name, completed=True, completed_call_seconds=1.0)
        for name in phase.SEQUENCE
    ]


def test_phase_sums_do_not_count_both_suffixes_as_one_path():
    result = phase.phase_summary(entries())
    assert result["shared_prefix_seconds"] == 8
    assert result["wide_suffix_seconds"] == 2
    assert result["narrow_suffix_seconds"] == 8
    assert result["wide_path_partial_phase_sum_seconds"] == 10
    assert result["narrow_path_partial_phase_sum_seconds"] == 16
    assert result["comparison_only_assembly_seconds"] == 1
    assert result["all_completed_call_seconds"] == 19
    assert result["scope"] == phase.SCOPE
    assert not result["independent_final_assembly_included"]


@pytest.mark.parametrize(
    "change", ["missing", "extra", "order", "incomplete", "nan", "negative", "bool"]
)
def test_phase_summary_refuses_wrong_inventory_and_times(change):
    rows = entries()
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(rows[0])
    elif change == "incomplete":
        rows[0]["completed"] = False
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["completed_call_seconds"] = {
            "nan": float("nan"),
            "negative": -1.0,
            "bool": True,
        }[change]
    with pytest.raises(ValueError):
        phase.phase_summary(rows)


def test_sampling_uses_shared_traversal_resets_cache_and_releases_generations(tmp_path):
    # Exercise the REAL extracted traversal with tiny synthetic model results.
    # No compiled-model or multi-host numerical/performance claim.
    initial = [None] * 20
    initial[0] = np.zeros((128, 2), np.float32)
    initial[1] = initial[0]
    initial[2:5] = [np.zeros(1) for _ in range(3)]
    initial[5:8] = [np.zeros((128, 2)), np.ones(128), np.zeros((128, 2))]
    initial[8:10] = [np.asarray(2553, np.int32), np.asarray(128, np.int32)]
    initial[16:18] = [object(), object()]
    initial[18] = np.ones((8, 4, 128), bool)
    initial[19] = np.zeros((128, 64))
    initial = tuple(initial)
    generations, starts, seen = [], [], []

    class Calls:
        root = tmp_path
        record = {}
        samples = []

        def phase(self, name, action):
            if name.endswith("/sample_complete"):
                assert all(ref() is None for ref in generations)
            return action()

        def call(self, label, name, args, *, preserve):
            seen.append((label, name))
            if name == "prepare_prefix":
                rows, offset, count, tile = args
                first = int(tile) * 32
                result = (
                    tuple(
                        r[:, :, first : first + 32] if i == 5 else r[first : first + 32]
                        for i, r in enumerate(rows)
                    ),
                    offset + first,
                    np.asarray(32),
                )
            elif name == "prefix":
                if label.endswith("/prefix0"):
                    starts.append(tuple(id(v) for v in args[2:5]))
                caches = tuple(np.asarray([len(generations)]) for _ in range(3))
                generations.extend(weakref.ref(v) for v in caches)
                result = (
                    args[0],
                    args[1],
                    *caches,
                    args[5],
                    args[6],
                    args[7],
                    args[18],
                    args[0],
                )
            elif name == "prepare_wide":
                norms, health, _ = args
                result = (
                    np.concatenate(norms),
                    np.ones(128, bool),
                    np.concatenate(health, axis=2),
                )
            elif name == "prepare_narrow":
                result = (args[0], np.ones(32, bool), args[1])
            elif name in ("candidate", "control"):
                n = len(args[0])
                result = (
                    args[0],
                    np.zeros((n, 8), np.int32),
                    np.zeros((n, 8)),
                    args[4],
                )
            else:
                assert name == "assemble"
                result = ((), ())
            preserve(result)
            self.samples.append(
                dict(graph=name, completed=True, completed_call_seconds=0.0)
            )
            return result

    trace = []
    calls = Calls()
    r = phase.run_samples(
        calls,
        values=initial,
        tiles=tuple(np.asarray(i) for i in range(4)),
        verify_component=lambda *args: None,
        verify_assembly=lambda *args: None,
        trace_start=lambda path: trace.append(("start", len(seen))),
        trace_stop=lambda: trace.append(("stop", len(seen))),
    )
    assert r["complete"] and len(r["wall_samples"]) == 10
    assert len(r["warmup_samples"]) == 3 and len(r["traced_samples"]) == 2
    assert trace == [("start", 13 * 19), ("stop", 15 * 19)]
    assert Counter(n for _, n in seen) == Counter(
        {n: c * 15 for n, c in phase.COUNTS.items()}
    )
    assert starts == [tuple(id(v) for v in initial[2:5])] * 15
    assert all(ref() is None for ref in generations)


@pytest.mark.parametrize(
    "failure", ["none", "local_start", "peer_start", "body", "local_stop", "peer_stop"]
)
def test_trace_entry_exit_is_voted_and_started_hosts_always_stop(tmp_path, failure):
    actions, phases = [], []

    class Calls:
        def phase(self, name, action):
            phases.append(name)
            value = action()
            if failure == "peer_start" and name.endswith("/start"):
                raise RuntimeError("peer start")
            if failure == "peer_stop" and name.endswith("/stop"):
                raise RuntimeError("peer stop")
            return value

    def start(path):
        actions.append("start")
        if failure == "local_start":
            raise RuntimeError("local start")

    def stop():
        actions.append("stop")
        if failure == "local_stop":
            raise RuntimeError("local stop")

    def run():
        with phase.voted_trace(Calls(), tmp_path / "trace", start=start, stop=stop):
            actions.append("body")
            if failure == "body":
                raise RuntimeError("body")

    if failure == "none":
        run()
    else:
        with pytest.raises(RuntimeError):
            run()
    assert phases == ["phase_trace/start", "phase_trace/stop"]
    assert ("stop" in actions) is (failure != "local_start")
    assert ("body" in actions) is (failure not in ("local_start", "peer_start"))


@pytest.mark.parametrize("failure", ["local_summary", "peer_summary", "clock"])
def test_sample_summary_failure_votes_before_trace_cleanup(
    tmp_path, monkeypatch, failure
):
    # Actual BudgetedCalls.phase publication/vote semantics, fake device traversal.
    votes, traces, traversals = [], [], []

    def consensus(ok):
        label = calls.record["current_phase"]
        votes.append((label, ok))
        return ok and not (
            failure == "peer_summary" and label == "traced0/sample_complete"
        )

    calls = phase.CompactPhaseCalls(
        root=tmp_path,
        record={},
        consensus=consensus,
        journal=SimpleNamespace(phase=lambda *a, **kw: None),
        local_slots={},
    )

    def traverse(calls, **kwargs):
        traversals.append(kwargs["case"])
        rows = entries()
        for row in rows:
            row["completed_call_seconds"] = 0.0
        if len(traversals) == 14 and failure == "local_summary":
            rows.pop()
        calls.samples.extend(rows)

    monkeypatch.setattr(phase, "execute_window", traverse)
    with pytest.raises((RuntimeError, ValueError)):
        phase.run_samples(
            calls,
            values=(),
            tiles=(),
            verify_component=lambda *a: None,
            verify_assembly=lambda *a: None,
            trace_start=lambda path: traces.append("start"),
            trace_stop=lambda: traces.append("stop"),
            clock=lambda: (
                float("nan") if failure == "clock" and len(traversals) == 14 else 0.0
            ),
        )
    assert traces == ["start", "stop"]
    assert traversals[-1] == "traced0" and len(traversals) == 14
    assert votes[-2:] == [
        ("traced0/sample_complete", failure == "peer_summary"),
        ("phase_trace/stop", True),
    ]
    assert not calls.record["phase_baseline"]["complete"]


def test_trace_groups_use_actual_names_and_disclose_mixed_suffixes(
    monkeypatch, tmp_path
):
    from scripts.analysis import parse_xplane

    programs = {
        n: f'HloModule jit_{"suffix" if n in ("candidate","control") else "candidate" if n=="prefix" else n}, is_scheduled=true'
        for n in phase.COUNTS
    }
    groups = phase.trace_groups(programs)
    assert groups["jit_suffix"]["expected_calls_per_core"] == 10
    assert groups["jit_suffix"]["mixed_graphs"]
    assert groups["jit_candidate"]["expected_calls_per_core"] == 8
    assert groups["jit_assemble"]["expected_calls_per_core"] == 2
    called = []

    def aggregate(root, step_module_re):
        group = next(g for g in groups.values() if g["module_regex"] == step_module_re)
        called.append(step_module_re)
        return dict(
            n_files=8, n_cores=64, steps_per_core=group["expected_calls_per_core"]
        )

    monkeypatch.setattr(parse_xplane, "aggregate_fleet", aggregate)
    r = phase.aggregate_phase_trace(tmp_path, programs)
    assert len(called) == 6 and r["programs"]["jit_suffix"]["selection"]["mixed_graphs"]
    assert not r["utilization_claim"]
    assert r["category_times_are_fleet_means_not_critical_path"]
    assert "intervening other" in r["cycle_metric_caveat"]
    for wrong in (
        dict(n_files=7, n_cores=64, steps_per_core=8),
        dict(n_files=8, n_cores=63, steps_per_core=8),
        dict(n_files=8, n_cores=64, steps_per_core=99),
    ):
        monkeypatch.setattr(parse_xplane, "aggregate_fleet", lambda *a, **kw: wrong)
        with pytest.raises(ValueError, match="coverage"):
            phase.aggregate_phase_trace(tmp_path, programs)


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "dispatch",
        "preserve",
        "postpeak",
        "archive",
        "peer_execute",
        "stream_cap",
        "call_cap",
    ],
)
def test_compact_calls_reuse_real_budgeted_timing_and_preserve_failures(
    tmp_path, monkeypatch, failure
):
    import jax
    from scripts.greenfield import prefill_window_worker as worker

    t = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: t[0])

    def complete(value):
        t[0] += 3
        return value

    monkeypatch.setattr(jax, "block_until_ready", complete)
    monkeypatch.setattr(jax, "local_devices", lambda: ())
    stats = [
        dict(
            device_id=i,
            platform="tpu",
            process_index=0,
            bytes_in_use=1024,
            peak_bytes_in_use=2048,
            bytes_limit=4 << 30,
        )
        for i in range(4)
    ]
    census = dict(devices=[dict(r, memory_stats=r) for r in stats])
    monkeypatch.setattr(worker, "capture_resident_buffers", lambda *a, **kw: census)

    def post(*args):
        if failure == "postpeak":
            raise RuntimeError("postpeak")
        return stats

    monkeypatch.setattr(worker, "capture_identified_device_memory", post)

    def consensus(ok):
        t[0] += 7
        return ok and not (
            failure == "peer_execute"
            and calls.record.get("current_phase") == "x/execute"
        )

    calls = phase.CompactPhaseCalls(
        root=tmp_path,
        record=dict(jax_process_index=0),
        consensus=consensus,
        journal=SimpleNamespace(phase=lambda *a, **kw: None),
        local_slots={i: i for i in range(4)},
        budgeter=lambda *a, **kw: dict(estimate_fits=True),
    )

    class Program:
        def memory_analysis(self):
            return SimpleNamespace(**{k: 0 for k in worker.MEMORY_FIELDS})

        def __call__(self, *args):
            t[0] += 2
            if failure == "dispatch":
                raise RuntimeError("dispatch")
            return (np.ones(1),)

    calls.programs = {"prefix": Program()}
    captures = []

    def preserve(value):
        t[0] += 11
        captures.append(True)
        if failure == "preserve":
            raise RuntimeError("preserve")

    if failure == "archive":
        monkeypatch.setattr(
            phase.gzip,
            "compress",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("archive")),
        )
    elif failure == "stream_cap":
        monkeypatch.setattr(phase, "MAX_WITNESS_BYTES", 0)
    elif failure == "call_cap":
        monkeypatch.setattr(phase, "MAX_WITNESS_CALLS", 0)
    if failure == "none":
        calls.call("x", "prefix", (), preserve=preserve)
    else:
        with pytest.raises((RuntimeError, OSError, ValueError)):
            calls.call("x", "prefix", (), preserve=preserve)
    if failure in ("archive", "stream_cap", "call_cap"):
        assert calls.record["call_evidence"] and not calls.samples
    else:
        original = json.loads(gzip.decompress(calls.evidence_path.read_bytes()))
        assert not calls.record["call_evidence"] and len(calls.samples) == 1
        if failure != "dispatch":
            assert (
                original["completed_call_seconds"] == 5
            )  # includes completion, excludes7s votes and11s capture
            assert captures
        else:
            assert not original["completed"] and not captures
