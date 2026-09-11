"""Real compiler writer/journal/call originals; fixture math, memory and captures."""

from hashlib import sha256
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_history_execution as execution
from scripts.greenfield import ws32_history_call_evidence as evidence
from scripts.greenfield import prefill_window_worker as parent
from glm_tpu.greenfield.validation.ws32_prefill_memory import MEMORY_FIELDS


@pytest.fixture
def tmp_path(tmp_path):
    root = tmp_path / "greenfield_fp8_ws32_history_frontier_l06_20260911T000000000000000Z" / "rank0"
    root.mkdir(parents=True)
    return root


def staged(tmp_path, monkeypatch, failure=None):
    import jax

    record = dict(protocol=execution.protocol.PROTOCOL, profile=execution.admission.PROFILE,
                  compile_only=False, diagnostic_only=True, programs={}, code_hash="a" * 40,
                  launch_rank=0, jax_process_index=3)
    events = []
    slots = dict(zip((12, 14, 13, 15), (9, 13, 25, 29)))
    stats = dict(bytes_limit=32 << 30, bytes_in_use=1 << 20, peak_bytes_in_use=2 << 20)
    devices = tuple(NS(id=d, platform="tpu", process_index=3, memory_stats=lambda: dict(stats))
                    for d in slots)
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    census = dict(devices=[dict(device_id=d.id, platform="tpu", process_index=3,
                                memory_stats=dict(stats)) for d in devices])
    monkeypatch.setattr(parent, "capture_resident_buffers", lambda *a, **k: census)
    monkeypatch.setattr(parent, "capture_identified_device_memory", lambda devices: [
        dict(device_id=d.id, platform="tpu", process_index=3, **stats) for d in devices])

    def budget(census, analyses, *, active_graph):
        assert tuple(analyses) == execution.protocol.PROGRAMS
        assert active_graph in analyses
        events.append(("budget", active_graph))
        return dict(estimate_fits=failure != "memory", fixture=True)
    monkeypatch.setattr(execution.runtime_module, "memory_budget", budget)

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return NS(**{k: 0 for k in MEMORY_FIELDS})

        def as_text(self):
            return self.name + " optimized fixture"

        def __call__(self, *args):
            events.append(("dispatch", self.name))
            if failure == "dispatch_" + self.name:
                raise ValueError("fixture device failure")
            return ("result", self.name, len(events))

    class Lowered:
        def __init__(self, name):
            self.name = name

        def compiler_ir(self, *, dialect):
            assert dialect == "stablehlo"
            return self.name + " stable fixture"

        def compile(self):
            events.append(("compile", self.name))
            if failure == "compile_" + self.name:
                raise ValueError("fixture compile failure")
            return Program(self.name)

    jobs = tuple((n, NS(lower=lambda *a, n=n: Lowered(n)), ()) for n in execution.protocol.PROGRAMS)
    layers = tuple(NS(dsa=NS(wk_bits_local=f"bits{l}", wk_scale_local=f"scales{l}")) for l in range(7))
    bound = execution.runtime_module.BoundHistoryRuntime(
        config=NS(), prepared=NS(), jobs=jobs, embedding=None, layers=layers,
        observer_layers=layers, raw_exact=tuple(range(4)), rope=None,
        prompt_tokens=np.zeros(8155, np.int32), originals={}, local_slots=slots)
    monkeypatch.setattr(execution.admission, "registration", lambda repo: {})

    def inspect(name, stable, optimized, memory, *, repo):
        assert all((tmp_path / f"{n}.optimized_hlo.txt").is_file() for n in execution.protocol.PROGRAMS)
        events.append(("admit", name))
        if failure == "admission":
            raise ValueError("fixture actual HLO refusal")
        return dict(passed=True, fixture=True)
    monkeypatch.setattr(execution.admission, "inspect_program", inspect)

    def capture_wk(root, record, local_slots, *, layer, name, value, inputs):
        events.append(("capture", f"layer{layer}/{name}"))
        assert layer in execution.protocol.PRODUCERS
        assert local_slots == slots and value[1] == name
        if name == "wk_decode":
            assert inputs == (f"bits{layer}", f"scales{layer}")
        else:
            assert len(inputs) == 1 and inputs[0][1] == "wk_decode"
        record.setdefault("materializer_originals", {})[f"layer{layer}/{name}"] = dict(fixture=True)
        if failure == "capture_" + name:
            raise ValueError("fixture capture refusal after retention")

    def capture_exact(root, record, actual_bound, *, name, value, inputs):
        assert actual_bound is bound and value[1] == name
        assert inputs == (bound.raw_exact,) if name == "exact_decode" else inputs[0][1] == "exact_decode"
        events.append(("capture", name))
        record.setdefault("materializer_originals", {})[name] = dict(fixture=True)
        if failure == "capture_" + name:
            raise ValueError("fixture capture refusal after retention")
    monkeypatch.setitem(sys.modules, "scripts.greenfield.ws32_history_materializers",
                        NS(capture_wk=capture_wk, capture_exact=capture_exact))

    def history(calls, **kwargs):
        assert kwargs["plan"] == execution.protocol.plan()
        assert kwargs["layers"] is bound.layers
        assert len(kwargs["wk"]) == 4 and all(v[1] == "wk_promote" for v in kwargs["wk"])
        assert kwargs["exact"][1] == "exact_promote"
        steps = execution.call_schedule()[10:]
        if failure == "missing_call":
            steps = steps[:-1]
        for phase, name in steps:
            calls.call(phase, name, (), preserve=lambda v: None)
        record["history"] = dict(complete=True, attribution_eligible=failure != "reproduction")
        if failure == "wrong_schedule":
            record["call_evidence"][10]["phase"] = "different"
    monkeypatch.setattr(execution.worker, "execute_history", history)

    def consensus(ok):
        if failure == "peer_admission" and record.get("current_phase") == "history/admission":
            return False
        return ok
    if failure == "finalize":
        original_close = execution.HistoryJournal.close
        def close(self):
            original_close(self)
            raise ValueError("fixture close failure")
        monkeypatch.setattr(execution.HistoryJournal, "close", close)
    if failure == "identity":
        record["profile"] = "old"
    if failure == "prior_call":
        record["call_evidence"] = [dict(old=True)]
    if failure == "prior_original":
        (tmp_path / "materializers").mkdir()

    def run():
        execution.execute(root=tmp_path, record=record, repo=tmp_path, mesh=None,
                          bound=bound, consensus=consensus)
    return run, record, events


def test_nine_compilers_then_331_calls_originals_and_journal(tmp_path, monkeypatch):
    run, record, events = staged(tmp_path, monkeypatch)
    run()
    assert [e[0] for e in events if e[0] in ("compile", "admit")] == ["compile"] * 9 + ["admit"] * 9
    originals = evidence.load_calls(tmp_path, record)
    assert tuple((e["phase"], e["graph"]) for e in originals) == execution.call_schedule()
    assert len([e for e in events if e[0] == "dispatch"]) == 331
    assert len([e for e in events if e[0] == "budget"]) == 331
    assert len(record["materializer_originals"]) == 10
    assert record["history_execution_complete"] is True
    assert record["numerical_promotion"] is record["performance_claim"] is False
    raw = (tmp_path / "compile_journal.jsonl").read_bytes()
    assert sha256(raw).hexdigest() == record["compile_journal_sha256"]
    journal = [json.loads(line) for line in raw.splitlines()]
    assert [r["graph"] for r in journal if r["stage"] == "inspected"] == list(execution.protocol.PROGRAMS)
    assert {r["artifact_kind"] for r in journal} == {execution.HistoryJournal.artifact_kind}


@pytest.mark.parametrize("failure", [*("compile_" + n for n in execution.protocol.PROGRAMS),
    "admission", "peer_admission", "memory", "identity", "prior_call", "prior_original",
    "dispatch_wk_decode", "capture_wk_decode", "capture_wk_promote",
    "dispatch_exact_decode", "capture_exact_decode", "capture_exact_promote"])
def test_refusal_retains_originals_and_prevents_successor(tmp_path, monkeypatch, failure):
    run, record, events = staged(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert record["status"] == "DIAGNOSTIC_FAILED"
    assert record["acquisition_phases"]["history/finalize"]["status"] in ("COMPLETE", "FAILED")
    if failure.startswith("compile_") or failure in ("admission", "peer_admission", "memory", "identity", "prior_call", "prior_original"):
        assert not any(e[0] == "dispatch" for e in events)
    if failure in ("admission", "peer_admission", "memory"):
        assert all((tmp_path / f"{n}.optimized_hlo.txt").is_file() for n in execution.protocol.PROGRAMS)
    if failure.startswith("capture_"):
        reference = record["call_evidence"][-1]
        original = json.loads((tmp_path / reference["original"]["path"]).read_bytes())
        assert original["completed"] is True
        assert events[-1][0] == "capture"


@pytest.mark.parametrize("failure", ["missing_call", "wrong_schedule", "reproduction", "finalize"])
def test_terminal_refusals_never_promote(tmp_path, monkeypatch, failure):
    run, record, events = staged(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert record["status"] == "DIAGNOSTIC_FAILED"
    assert record["numerical_promotion"] is record["performance_claim"] is False
    assert len(list((tmp_path / "call_records").glob("call*.json"))) in (330, 331)
    if failure != "finalize":
        assert record["history_execution_complete"] is False
    else:
        assert "fixture close failure" in record["finalization_error"]
