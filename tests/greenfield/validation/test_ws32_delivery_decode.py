"""Actual host materializer/call guards with fixture compute and TPU counters."""

import ast
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import weakref

import jax
import jax.numpy as jnp
import pytest

from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import ws32_delivery_decode as deferred
from scripts.greenfield import ws32_budgeted_calls as parent
from tests.greenfield.validation.test_ws32_delivery_phase_loading import FixtureJournal
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import fixture as memory_fixture

ROOT = Path(__file__).resolve().parents[3]
BASE = "5e94239a7f3a170a4c33aa4e23b0f7eb7835ca04"


def setup(tmp_path, monkeypatch, failure=None):
    raw = jnp.asarray([1], jnp.bfloat16)
    decoded = jnp.asarray([2], jnp.bfloat16)
    promoted = jnp.asarray([2], jnp.float32)
    events, refs = [], []
    census = deepcopy(memory_fixture()["records"][0]["prefill_execution"]["memory_admission"]["census"])
    process = census["devices"][0]["process_index"]
    slots = {d["device_id"]: i for i, d in enumerate(census["devices"])}
    devices = tuple(NS(id=d["device_id"], platform="tpu", process_index=process,
        memory_stats=lambda d=d: dict(d["memory_stats"])) for d in census["devices"])
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    def capture(*a, **k):
        value = deepcopy(census)
        if failure == "pre_memory":
            value["devices"][0]["memory_stats"]["peak_bytes_in_use"] = deferred.runtime.DEVICE_LIMIT
        return value
    monkeypatch.setattr(parent, "capture_resident_buffers", capture)
    def after(*a):
        rows = [dict(device_id=d.id, platform="tpu", process_index=process, **d.memory_stats()) for d in devices]
        if failure == "post_memory": rows[0]["peak_bytes_in_use"] = deferred.runtime.DEVICE_LIMIT
        return rows
    monkeypatch.setattr(parent, "capture_identified_device_memory", after)
    def decode(value):
        assert value is raw; events.append("decode"); return decoded
    def promote(value):
        assert value is decoded; events.append("promote"); return promoted
    monkeypatch.setattr(worker, "build_ws32_exact_dsa_materializer_program",
                        lambda *a: NS(decode=decode, promote=promote))
    monkeypatch.setattr(worker, "select_ws32_exact_dsa_raw_weights", lambda *a: raw)
    def report(**kwargs):
        if failure == "admission": raise ValueError("fixture graph refusal")
        return dict(passed=True, graph=kwargs["graph"])
    monkeypatch.setattr(worker, "_write_exact_materializer_graph", report)
    class Compiled:
        def __init__(self, fn): self.fn = fn
        def memory_analysis(self):
            # Not a TPU allocation: chosen nonzero code/output/scratch to
            # distinguish the first one-program and second two-program budgets.
            return NS(argument_size_in_bytes=64, output_size_in_bytes=32,
                      temp_size_in_bytes=16, generated_code_size_in_bytes=128,
                      alias_size_in_bytes=0)
        def __call__(self, value): return self.fn(value)
    class Jit:
        def __init__(self, fn): self.fn, self.cached = fn, None
        def lower(self, *a): return self
        def compile(self):
            if failure == "compile" and self.fn is promote:
                raise ValueError("fixture promotion compile failure")
            self.cached = Compiled(self.fn)
            refs.append(weakref.ref(self.cached))
            return self.cached
        def clear_cache(self):
            events.append("clear"); self.cached = None
    journal = FixtureJournal(tmp_path / "journal.jsonl", {"fixture_only": True})
    def consensus(ok):
        if failure == "peer" and "decode" in events: return False
        return ok
    calls = deferred.open_phase(root=tmp_path / "decode", journal=journal, consensus=consensus,
        local_slots=slots, identity=dict(jax_process_index=process, code_hash="a" * 40))
    args = NS(compile_only=False, hlo_dir=tmp_path, batched_prefill_profile=deferred.runtime.PROFILE,
        **{f"expected_{name}_{form}_sha256":"a"*64 for name in deferred.ROLES
           for form in ("stablehlo", "optimized_hlo")})
    kwargs = dict(jax=NS(jit=Jit, block_until_ready=jax.block_until_ready),
        args=args, mesh=None, config=NS(exact_dsa=True), weights=raw, batched_prefill=True,
        graphs={}, compile_seconds={}, compiled_memory={}, acquisition_journal=journal,
        begin_compile=lambda name: events.append(name), record_compile=lambda name: None)
    return kwargs, calls, events, refs, promoted


def test_materializer_budgets_both_completed_calls_and_releases_code(tmp_path, monkeypatch):
    kwargs, calls, events, refs, expected = setup(tmp_path, monkeypatch)
    try:
        result = worker._materialize_exact_decode(**kwargs, protected_calls=calls)
        record = deferred.finish(calls)
    finally:
        calls.journal.close()
    assert result is expected and record["complete"]
    assert events == ["exact_materialize", "decode", "exact_promote", "promote", "clear", "clear"]
    assert not calls.programs
    for i, row in enumerate(record["call_evidence"]):
        assert row["budget"]["resident_graphs"] == list(deferred.ROLES[:i+1])
        assert row["budget"]["required_reserve_bytes"] == 1 << 30
        assert row["budget"]["devices"][0]["resident_code_bytes"] == 128 * (i+1)
        assert row["output_schema"] == [dict(shape=[1], dtype="bfloat16" if i == 0 else "float32")]
        assert len(row["post_memory"]) == 4
    assert json.loads((calls.root / "runner.json").read_text()) == record
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("failure", ["admission", "compile", "pre_memory", "post_memory", "peer"])
def test_refusals_preserve_record_and_prevent_successor(tmp_path, monkeypatch, failure):
    kwargs, calls, events, refs, _ = setup(tmp_path, monkeypatch, failure)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            worker._materialize_exact_decode(**kwargs, protected_calls=calls)
    finally:
        calls.journal.close()
    assert "promote" not in events
    assert not calls.programs
    record = json.loads((calls.root / "runner.json").read_text())
    assert record["complete"] is False
    if failure in ("compile", "post_memory", "peer"):
        assert record["call_evidence"][0]["completed"]
        assert record["call_evidence"][0]["output_schema"]
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_default_helper_matches_original_two_call_effects(tmp_path, monkeypatch):
    kwargs, calls, events, refs, expected = setup(tmp_path, monkeypatch)
    source = subprocess.check_output(["git", "show", BASE + ":scripts/greenfield/run_short_decoder_ws32.py"],
                                     cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "_materialize_exact_decode")
    scope = dict(worker.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original_helper", "exec"), scope)
    try:
        old_result = scope["_materialize_exact_decode"](**kwargs)
        old_events = events[:]
        old_graphs, old_memory = deepcopy(kwargs["graphs"]), deepcopy(kwargs["compiled_memory"])
        events.clear()
        result = worker._materialize_exact_decode(**kwargs)
    finally:
        calls.journal.close()
    assert result is old_result is expected
    assert events == old_events and kwargs["graphs"] == old_graphs and kwargs["compiled_memory"] == old_memory


@pytest.mark.parametrize("mutation", ["missing", "extra", "alias", "limit", "all_live"])
def test_budget_refuses_invalid_inventory_or_memory(mutation):
    census = deepcopy(memory_fixture()["records"][0]["prefill_execution"]["memory_admission"]["census"])
    analysis = dict(argument_size_in_bytes=64, output_size_in_bytes=32,
                    temp_size_in_bytes=16, generated_code_size_in_bytes=128, alias_size_in_bytes=0)
    analyses = {n: dict(analysis) for n in deferred.ROLES}
    if mutation == "missing": analyses.pop(deferred.ROLES[0])
    if mutation == "extra": analyses["prefill_chunk"] = analysis
    if mutation == "alias": analyses[deferred.ROLES[1]]["alias_size_in_bytes"] = 1
    if mutation == "limit": census["devices"][0]["memory_stats"]["bytes_limit"] += 1
    if mutation == "all_live": census["includes_all_live_arrays"] = False
    with pytest.raises(ValueError): deferred.memory_budget(census, analyses, active_graph=deferred.ROLES[1])


def test_existing_directory_never_overwritten(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    path = root / "runner.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        deferred.open_phase(root=root, journal=None, consensus=lambda ok: ok,
                            local_slots={}, identity={})
    assert path.read_text() == "original"


def test_peer_failure_cannot_leave_complete_record(tmp_path, monkeypatch):
    kwargs, calls, _, _, _ = setup(tmp_path, monkeypatch)
    try:
        worker._materialize_exact_decode(**kwargs, protected_calls=calls)
        calls.consensus = lambda ok: False
        with pytest.raises(RuntimeError): deferred.finish(calls)
    finally:
        calls.journal.close()
    assert not json.loads((calls.root / "runner.json").read_text())["complete"]


def overlay_fixture(calls):
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
        Ws32StrategyNdDenseOverlay, strategy_nd_dense_tensor_names,
    )
    records = {(layer, *divmod(slot, 4)): dict(tensors={name: dict(byte_count=1024)
        for name in strategy_nd_dense_tensor_names(layer)})
        for slot in calls.local_slots.values() for layer in range(3)}
    return Ws32StrategyNdDenseOverlay(root=Path("/fixture/only"),
        manifest=dict(manifest_sha256="a"*64), records=records,
        manifest_file_sha256="b"*64, success_file_sha256="c"*64)


@pytest.mark.parametrize("failure", [None, "limit", "pre_reserve", "post_reserve", "missing_tensor"])
def test_overlay_budget_uses_verified_owner_payload_and_actual_peaks(tmp_path, monkeypatch, failure):
    _, calls, _, _, _ = setup(tmp_path, monkeypatch)
    overlay = overlay_fixture(calls)
    census = deepcopy(memory_fixture()["records"][0]["prefill_execution"]["memory_admission"]["census"])
    if failure == "limit": census["devices"][0]["memory_stats"]["bytes_limit"] += 1
    if failure == "pre_reserve":
        census["devices"][0]["memory_stats"]["peak_bytes_in_use"] = deferred.runtime.DEVICE_LIMIT
    if failure == "missing_tensor":
        next(iter(overlay.records.values()))["tensors"].popitem()
    monkeypatch.setattr(deferred, "capture_resident_buffers", lambda *a, **k: census)
    def after(devices):
        rows = [dict(device_id=row["device_id"], process_index=row["process_index"],
            platform="tpu", **row["memory_stats"]) for row in census["devices"]]
        if failure == "post_reserve": rows[0]["peak_bytes_in_use"] = deferred.runtime.DEVICE_LIMIT
        return rows
    monkeypatch.setattr(deferred, "capture_identified_device_memory", after)
    def exercise():
        calls.phase("overlay/preflight", lambda: deferred.overlay_preflight(overlay, calls, {"raw": ()}))
        calls.phase("overlay/complete", lambda: deferred.overlay_completed(calls))
    try:
        if failure:
            with pytest.raises(ValueError): exercise()
        else:
            exercise()
            report = calls.record["overlay_memory"]
            assert len(report["devices"]) == 4
            for row in report["devices"]:
                assert row["payload_bytes"] == 3 * 4 * 1024
                assert row["estimated_peak_bytes"] == 26_000_000_000
            assert report["temporary_payload_copies"] == 2
            assert report["reserve_bytes"] == 1 << 30
            assert len(calls.record["overlay_device_memory_after"]) == 4
    finally:
        calls.journal.close()


def test_overlay_refusal_precedes_loader_device_placement(tmp_path, monkeypatch):
    _, calls, _, _, _ = setup(tmp_path, monkeypatch)
    overlay = overlay_fixture(calls)
    monkeypatch.setattr(worker, "verify_ws32_strategy_nd_dense_overlay", lambda *a, **k: overlay)
    monkeypatch.setattr(worker, "load_ws32_strategy_nd_dense_overlay",
                        lambda *a, **k: pytest.fail("loader ran after preflight refusal"))
    args = NS(strategy_nd_dense_overlay_root=tmp_path,
        strategy_nd_dense_overlay_manifest_sha256="a"*64,
        strategy_nd_dense_overlay_manifest_file_sha256="b"*64,
        strategy_nd_dense_overlay_success_file_sha256="c"*64)
    def refuse(value):
        assert value is overlay
        raise ValueError("placement bound failed")
    try:
        with pytest.raises(ValueError, match="placement bound failed"):
            calls.phase("overlay/load", lambda: worker._load_dense_overlay(
                args=args, config=NS(strategy_nd_dense=True), mesh=None, physical_mesh=None,
                all_arrays={}, before_load=refuse))
    finally:
        calls.journal.close()


def test_actual_worker_passes_protected_owner_to_materializer_and_votes_overlay():
    tree = ast.parse((ROOT / "scripts/greenfield/run_short_decoder_ws32.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    branch = next(n for n in main.body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == "long_phase"
        and any(isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr == "open_phase" for v in ast.walk(n)))
    calls = [n for n in ast.walk(branch) if isinstance(n, ast.Call)]
    materialize = next(n for n in calls if isinstance(n.func, ast.Name)
                       and n.func.id == "_materialize_exact_decode")
    assert any(k.arg == "protected_calls" and isinstance(k.value, ast.Name)
               and k.value.id == "decode_calls" for k in materialize.keywords)
    phases = [n.args[0].value for n in calls if isinstance(n.func, ast.Attribute)
              and n.func.attr == "phase" and isinstance(n.args[0], ast.Constant)]
    assert all(name in phases for name in ("delivery_decode/overlay_load",
        "delivery_decode/weight_transition", "delivery_decode/overlay_memory_after"))
