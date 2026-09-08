"""Numerical worker lifecycle/capture tests with CPU fixtures, not TPU proof."""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield import probe_ws32_prefill_layer as layer_worker
from scripts.greenfield.prefill_layer_evidence import owner_inputs


def make_calls(tmp_path, consensus=lambda ok: ok):
    record = dict(code_hash="a" * 40, launch_rank=0, jax_process_index=3, cases={})
    journal = worker.WindowNumericalJournal(
        tmp_path / "compile_journal.jsonl",
        dict(
            protocol=protocol.PROTOCOL,
            profile=admission.PROFILE,
            compile_only=False,
            code_hash=record["code_hash"],
            launch_rank=0,
        ),
    )
    return worker.BudgetedCalls(
        root=tmp_path,
        record=record,
        consensus=consensus,
        journal=journal,
        local_slots={9: 0, 13: 1, 25: 2, 29: 3},
    )


def fake_memory():
    rows = [
        dict(
            device_id=d,
            process_index=3,
            platform="tpu",
            buffers=[dict(bytes=340_000_000)],
            accounted_resident_bytes=340_000_000,
            memory_stats=dict(
                bytes_in_use=340_000_000,
                peak_bytes_in_use=350_000_000,
                bytes_limit=33_014_398_976,
            ),
        )
        for d in (9, 13, 25, 29)
    ]
    return dict(
        schema_version="ws32_prefill_resident_buffers_v1",
        includes_all_live_arrays=True,
        devices=rows,
    )


def test_call_memory_before_execution_completed_before_postflight(
    tmp_path, monkeypatch
):
    calls = make_calls(tmp_path)
    events = []
    pins = admission.registered_programs()

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.name]["compiled_memory"])

        def __call__(self, *args):
            events.append("execute")
            # Journal must already contain the successful memory vote.
            journal = (tmp_path / "compile_journal.jsonl").read_text()
            assert '"stage": "probe/memory"' in journal
            return np.zeros(1)

    calls.programs = {name: Program(name) for name in admission.PROGRAMS}

    def census(*args, **kwargs):
        events.append("census")
        return fake_memory()

    def post(*args):
        events.append("post")
        return [
            dict(
                device_id=r["device_id"],
                process_index=3,
                platform="tpu",
                **r["memory_stats"],
            )
            for r in fake_memory()["devices"]
        ]

    monkeypatch.setattr(worker, "capture_resident_buffers", census)
    monkeypatch.setattr(worker, "capture_identified_device_memory", post)
    calls.call(
        "probe",
        "candidate",
        (np.zeros(1),),
        preserve=lambda result: events.append("preserved"),
    )
    assert events == ["census", "execute", "preserved", "post"]
    evidence = calls.record["call_evidence"][0]
    assert evidence["completed"] and len(evidence["post_memory"]) == 4
    calls.journal.close()


@pytest.mark.parametrize("failure", ["peer", "local", "journal"])
def test_phase_failures_cannot_continue_or_hide_local_error(tmp_path, failure):
    votes = []

    def consensus(ok):
        votes.append(ok)
        return failure == "local"  # even malformed True cannot suppress local failure

    calls = make_calls(tmp_path, consensus)
    if failure == "journal":
        calls.journal.close()

    def action():
        if failure == "local":
            raise ValueError("local failed")

    with pytest.raises((ValueError, RuntimeError)):
        calls.phase("failure", action)
    assert votes == [failure == "peer"]
    calls.journal.close()


@pytest.mark.parametrize("failure", ["owner", "reserve", "allocation"])
def test_predispatch_refusal_never_calls_program(tmp_path, monkeypatch, failure):
    calls = make_calls(tmp_path)
    pins = admission.registered_programs()

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            memory = dict(pins[self.name]["compiled_memory"])
            if failure == "allocation":
                memory["temp_size_in_bytes"] += 1
            return SimpleNamespace(**memory)

        def __call__(self, *_):
            raise AssertionError("must refuse before executable dispatch")

    calls.programs = {n: Program(n) for n in admission.PROGRAMS}
    census = fake_memory()
    if failure == "owner":
        census["devices"][0]["process_index"] = 2
    if failure == "reserve":
        census["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 33_000_000_000
    monkeypatch.setattr(worker, "capture_resident_buffers", lambda *a, **k: census)
    with pytest.raises(ValueError):
        calls.call("refuse", "candidate", (np.zeros(1),), preserve=lambda result: None)
    assert calls.record["call_evidence"][0]["completed"] is False
    calls.journal.close()


@pytest.mark.parametrize("name", ["candidate", "control", "wk_decode", "wk_promote"])
@pytest.mark.parametrize("failure", ["post_memory", "execute_journal"])
def test_completed_originals_survive_postcall_refusal(
    tmp_path, monkeypatch, name, failure
):
    calls = make_calls(tmp_path)
    pins = admission.registered_programs()
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    executed = []

    class Program:
        def __init__(self, graph):
            self.graph = graph

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.graph]["compiled_memory"])

        def __call__(self, *args):
            executed.append(self.graph)
            return expected

    calls.programs = {n: Program(n) for n in admission.PROGRAMS}
    monkeypatch.setattr(
        worker, "capture_resident_buffers", lambda *a, **k: fake_memory()
    )
    path = tmp_path / "completed-original.npz"

    def post(*args):
        assert path.is_file(), "completed bytes must precede a postflight refusal"
        raise ValueError("post-call memory refused")

    monkeypatch.setattr(worker, "capture_identified_device_memory", post)
    if failure == "execute_journal":
        original = calls.journal.phase

        def phase(stage, **kwargs):
            if stage.endswith("/execute"):
                assert path.is_file()
                raise ValueError("execute journal refused")
            return original(stage, **kwargs)

        monkeypatch.setattr(calls.journal, "phase", phase)
    try:
        with pytest.raises(ValueError, match="refused"):
            for step in ("control3", "must_not_dispatch"):
                calls.call(
                    step,
                    name,
                    (np.zeros(1),),
                    preserve=lambda result: worker.save_arrays(
                        path, {"output": result}
                    ),
                )
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["output"], expected)
        assert executed == [name]
        assert calls.record["call_evidence"][0]["completed"] is True
    finally:
        calls.journal.close()


def fixture_output(host, slot):
    """Contract-valid mock outputs only; does not compute model arithmetic."""
    own = owner_inputs(host, slot)
    out = {
        k: own[k].copy()
        for k in ("kv", "index", "repair", "positions", "counts", "scores", "health")
    }
    out.update(
        output=np.zeros((128, 1536), protocol.BF16),
        residual=np.zeros((128, 1536), protocol.BF16),
        normalized=np.zeros((128, 1536), protocol.BF16),
        routes=np.tile(np.arange(8, dtype=np.int32), (128, 1)),
        route_weights=np.zeros((128, 8), np.float32),
    )
    offset, count = int(host["offset"]), int(host["count"])
    out["route_weights"][:count] = 1 / 8
    for row in range(count):
        n = min(offset + row + 1, 2048)
        out["positions"][row, :n] = np.arange(n)
        out["scores"][row, :n] = 0
        out["counts"][row] = n
    for _, page, row in protocol.written_addresses(slot, offset, count):
        for name in ("kv", "index", "repair"):
            out[name][page, row] = 0
    return out


@pytest.mark.parametrize("late_failure", [False, True])
def test_actual_case_loop_carries_three_caches_captures_and_replays(
    tmp_path, monkeypatch, late_failure
):
    calls = make_calls(tmp_path)
    host_now, result_observations, sequence = {}, {}, []
    previous_by_case = {}

    def inputs(host, *args):
        host_now.clear()
        host_now.update(host)
        return (
            tuple(
                host[k]
                for k in (
                    "update",
                    "residual",
                    "kv",
                    "index",
                    "repair",
                    "positions",
                    "counts",
                    "scores",
                    "offset",
                    "count",
                    "table",
                )
            )
            + (None,) * 7
            + (host["health"], host["rope"])
        )

    monkeypatch.setattr(layer_worker, "device_inputs", inputs)
    monkeypatch.setattr(
        worker, "local_observations", lambda r: result_observations[id(r)]
    )

    def call(phase, name, values, *, preserve):
        sequence.append(phase)
        case = phase.split("/")[0]
        tile = None if name == "candidate" else int(phase[-1])
        if tile is not None:
            expected = previous_by_case[case] if tile else host_now
            for index, key in enumerate(("kv", "index", "repair"), 2):
                assert values[index] is (expected[index] if tile else expected[key])
            assert int(values[9]) == max(
                0, min(32, protocol.CASES[case][1] - 32 * tile)
            )
        result = tuple(np.full(1, len(sequence)) for _ in range(12))
        observed = {}
        for device, slot in calls.local_slots.items():
            out = fixture_output(host_now, slot)
            if tile is not None:
                out = {
                    key: (
                        value
                        if key in ("kv", "index", "repair")
                        else value[tile * 32 : (tile + 1) * 32]
                    )
                    for key, value in out.items()
                }
            if late_failure and tile == 3:
                out["health"][-1] = False
            observed[device] = out
        result_observations[id(result)] = observed
        previous_by_case[case] = result
        preserve(result)
        return result

    calls.call = call
    try:
        if late_failure:
            with pytest.raises(ValueError, match="health failed"):
                worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
            with np.load(tmp_path / "boundary.npz", allow_pickle=False) as arrays:
                assert "actual_9__output" in arrays and "tile3_9__health" in arrays
                assert not arrays["tile3_9__health"][-1]
            assert len(sequence) == 5 and "competitive" not in calls.record["cases"]
        else:
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
            assert len(sequence) == 15
            for case in protocol.CASES:
                path = tmp_path / f"{case}.npz"
                replay = protocol.replay_case(
                    path, case=case, slots_by_device=calls.local_slots
                )
                assert replay == calls.record["cases"][case]["replay"]
                assert (
                    calls.record["cases"][case]["npz_sha256"]
                    == sha256(path.read_bytes()).hexdigest()
                )
                assert replay["passed"]
    finally:
        calls.journal.close()
