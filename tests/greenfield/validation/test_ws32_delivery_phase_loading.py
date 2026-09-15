"""Real phase orchestration with fixture compiler/model/counters; no TPU claim."""

import ast
from dataclasses import replace
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import ws32_delivery_wk as wk
from scripts.greenfield import ws32_delivery_programs as programs
from scripts.greenfield import ws32_batched_prefill_runner as adapter
from scripts.greenfield import ws32_budgeted_calls as parent
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
from scripts.greenfield.ws32_phase_weights import PhaseWeights
from scripts.greenfield.ws32_history_call_evidence import load_calls
from tests.greenfield.runtime.test_ws32_phase_weights import config, base_arrays
# Sealed originals of the retired rolled-prefill worker test (b667f00f).
ORIGINAL = Path("/home/gianl/glm-run") / (
    "greenfield_fp8_ws32_prefill_expert_panel_phase_l6_20260909T033031628458338Z"
)
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import fixture as memory_fixture

ROOT = Path(__file__).resolve().parents[3]
BASE = "129ac4ba06a4537b3afb937a1444d3e3fde17c16"


class FixtureJournal(Ws32NumericalJournal):
    def _check_identity(self, identity):
        assert identity == {"fixture_only": True}


@pytest.fixture(scope="module")
def originals():
    root = ORIGINAL / "fleet/rank0"
    record = json.loads((root / "runner.json").read_text())
    return {name: ((root / f"{name}.stablehlo.mlir").read_text(),
                   (root / f"{name}.optimized_hlo.txt").read_text(),
                   record["programs"][name]["compiled_memory"]) for name in wk.ROLES}


def stage(tmp_path, monkeypatch, originals, failure=None):
    owner = PhaseWeights(base_arrays(config()), config())
    events, compiled_refs, functions = [], [], []
    fixture = memory_fixture()
    census = fixture["records"][0]["prefill_execution"]["memory_admission"]["census"]
    slots = {d["device_id"]: i for i, d in enumerate(census["devices"])}
    devices = tuple(NS(id=d["device_id"], platform="tpu", process_index=d["process_index"],
        memory_stats=lambda d=d: dict(d["memory_stats"])) for d in census["devices"])
    process = census["devices"][0]["process_index"]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(parent, "capture_resident_buffers", lambda *a, **k: census)
    monkeypatch.setattr(parent, "capture_identified_device_memory", lambda *a: [
        dict(device_id=d.id, platform="tpu", process_index=process, **d.memory_stats()) for d in devices])
    class Program:
        def __init__(self, name): self.name = name
        def memory_analysis(self): return NS(**originals[self.name][2])
        def as_text(self): return originals[self.name][1]
        def __call__(self, *values):
            events.append(("call", self.name))
            if failure == "dispatch": raise ValueError("fixture dispatch failure")
            return jax.device_put(np.zeros((128, 6144), dtype=(
                jnp.bfloat16 if self.name == "wk_decode" else np.float32)))
    class Function:
        def __init__(self, name): self.name, self.cached = name, None
        def lower(self, *values):
            events.append(("lower", self.name))
            if failure == "compile" and self.name == "wk_promote":
                raise ValueError("fixture compile failure")
            return self
        def compiler_ir(self, dialect): return originals[self.name][0]
        def compile(self):
            self.cached = Program(self.name)
            compiled_refs.append(weakref.ref(self.cached))
            return self.cached
        def clear_cache(self):
            events.append(("release", self.name))
            self.cached = None
    for name in wk.ROLES: functions.append(Function(name))
    monkeypatch.setattr(owner, "wk_jobs", lambda mesh: tuple(
        (fn.name, fn, ()) for fn in functions))
    # Production preserves four actual shards; this single-CPU fixture exercises
    # the call/persistence path, with output observation explicitly substituted.
    def preserve(value, entry, **kwargs):
        entry["output_fixture"] = dict(shape=list(value.shape), dtype=str(value.dtype))
        if failure == "output": raise ValueError("fixture observed boundary failure")
    monkeypatch.setattr(wk, "preserve_output", preserve)
    journal = FixtureJournal(tmp_path / "journal.jsonl", {"fixture_only": True})
    def consensus(ok):
        if failure == "peer" and owner.phase == "repair": return False
        if failure == "late_peer" and owner.phase == "prefill": return False
        return ok
    kwargs = dict(repo=ROOT, root=tmp_path / "wk", owner=owner, mesh=None, journal=journal,
        consensus=consensus, local_slots=slots,
        identity=dict(jax_process_index=process, code_hash="a" * 40))
    return kwargs, events, compiled_refs


def test_completed42_original_calls_release_code_and_keep_only_raw_wk(tmp_path, monkeypatch, originals):
    kwargs, events, refs = stage(tmp_path, monkeypatch, originals)
    try:
        record = wk.prepare(**kwargs)
    finally:
        kwargs["journal"].close()
    assert record["complete"] and kwargs["owner"].phase == "prefill"
    assert len(kwargs["owner"].wk) == 21 and kwargs["owner"].decode_weights is None
    assert [e[1] for e in events if e[0] == "call"] == list(wk.ROLES) * 21
    assert events[:2] == [("lower", name) for name in wk.ROLES]
    assert events[-2:] == [("release", name) for name in wk.ROLES]
    assert len(record["call_evidence"]) == 42
    resolved = load_calls(kwargs["root"], record, expected_calls=42)
    assert len(resolved) == 42
    with pytest.raises(ValueError): load_calls(kwargs["root"], record)
    for i, ref in enumerate(record["call_evidence"]):
        raw = (kwargs["root"] / ref["original"]["path"]).read_bytes()
        assert sha256(raw).hexdigest() == ref["original"]["sha256"]
        item = json.loads(raw)
        assert item["completed"] and item["budget"]["estimate_fits"]
        assert item["output_fixture"]["shape"] == [128, 6144]
        assert item["graph"] == wk.ROLES[i % 2] and len(item["post_memory"]) == 4
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert json.loads((kwargs["root"] / "runner.json").read_text())["complete"]


@pytest.mark.parametrize("failure", ["compile", "dispatch", "output", "peer", "late_peer"])
def test_phase_failure_no_successor_and_originals_survive(tmp_path, monkeypatch, originals, failure):
    kwargs, events, refs = stage(tmp_path, monkeypatch, originals, failure)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            wk.prepare(**kwargs)
    finally:
        kwargs["journal"].close()
    assert kwargs["owner"].phase == "failed"
    assert len([e for e in events if e[0] == "call"]) <= (42 if failure == "late_peer" else 1)
    assert (kwargs["root"] / "wk_decode.optimized_hlo.txt").exists()
    assert not json.loads((kwargs["root"] / "runner.json").read_text())["complete"]
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_original_materializer_and_overlay_bodies_preserved():
    name = "scripts/greenfield/run_short_decoder_ws32.py"
    before = subprocess.check_output(["git", "show", BASE + ":" + name], text=True)
    old = ast.parse(before)
    current = ast.parse((ROOT / name).read_text())
    class WithoutOptionalMemoryHook(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id in ("before_load", "protected_calls"):
                return None
            return self.generic_visit(node)
    current = WithoutOptionalMemoryHook().visit(current)
    main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    for helper, first, last in (("_load_dense_overlay", "dense_overlay", "load_seconds"),
                                ("_materialize_exact_decode", "exact_dsa_weights", "program")):
        def assignment(node, target):
            return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in node.targets)
        start = next(i for i, n in enumerate(main.body) if assignment(n, first))
        end = next(i for i in range(start + 1, len(main.body)) if assignment(main.body[i], last))
        fn = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == helper)
        assert [ast.dump(n, include_attributes=False) for n in main.body[start:end]] == [
            ast.dump(n, include_attributes=False) for n in fn.body[1:-1]]


def test_overlay_loader_keeps_verification_and_overlap_refusal(monkeypatch):
    cfg = config()
    args = NS(strategy_nd_dense_overlay_root=Path("/fixture/overlay"),
        strategy_nd_dense_overlay_manifest_sha256="a" * 64,
        strategy_nd_dense_overlay_manifest_file_sha256="b" * 64,
        strategy_nd_dense_overlay_success_file_sha256="c" * 64)
    events = []
    verified, loaded = object(), NS(arrays={"overlay": object()})
    def verify(path, **pins):
        events.append("verify")
        assert list(pins.values()) == ["a" * 64, "b" * 64, "c" * 64]
        return verified
    def load(checkpoint, **kwargs):
        events.append("load"); assert checkpoint is verified; return loaded
    monkeypatch.setattr(worker, "verify_ws32_strategy_nd_dense_overlay", verify)
    monkeypatch.setattr(worker, "load_ws32_strategy_nd_dense_overlay", load)
    arrays = {}
    assert worker._load_dense_overlay(args=args, config=cfg, mesh=None, physical_mesh=None,
        all_arrays=arrays, before_load=lambda value: events.append("budget") if value is verified else pytest.fail("unverified")) == (verified, loaded)
    assert arrays == loaded.arrays and events == ["verify", "budget", "load"]
    with pytest.raises(RuntimeError, match="aliases base"):
        worker._load_dense_overlay(args=args, config=cfg, mesh=None, physical_mesh=None, all_arrays=arrays)


@pytest.mark.parametrize("label", ["128k_d1_0", "256k_e0"])
def test_live_builder_keeps_registered_options_and_owned_e0(monkeypatch, label):
    from scripts.greenfield import ws32_prefill_owned_state as owned
    seen = []
    pair = {name: object() for name in ("prefill_chunk", "prefill_tail")}
    monkeypatch.setattr(adapter, "build_graph_pair", lambda mesh, cfg, plan, **options:
        seen.append((plan, options)) or pair)
    wrapped = object()
    monkeypatch.setattr(owned, "consume_state", lambda p: wrapped if p is pair["prefill_chunk"] else pytest.fail("wrong role"))
    result = programs.build_live_graph_pair(None, None, repo=ROOT, context_label=label)
    assert seen[0][0] == programs.long_plan(label)
    assert seen[0][1] == dict(paired_position_sort=True, rolled_prefix=True, expert_panels=True,
        sorted_local_merge=True, canonical_dense=True, key_tile=512,
        **(dict(pending_cache_rows=True, flat_pending_rows=True, capture_barrier=True) if label == "256k_e0" else {}))
    assert result == (dict.fromkeys(pair, wrapped) if label == "256k_e0" else pair)


def test_outer_long_entry_still_refuses_before_runtime(monkeypatch):
    args = NS(batched_prefill_profile=wk.runtime.PROFILE, delivery_context_label="256k_e0",
              prefill_mode=adapter.PREFILL_MODE, compile_only=False)
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    monkeypatch.setattr(worker, "_initialize_runtime", lambda *a: pytest.fail("unregistered launch"))
    with pytest.raises(ValueError): worker.main()


def test_actual_four_cpu_owner_output_hashes_and_nonfinite_capture():
    code = '''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield.ws32_delivery_wk import preserve_output
mesh = Mesh(np.array(jax.devices()), ('replica',))
assert len(jax.devices()) == 4
slots = {int(d.id): i for i,d in enumerate(jax.devices())}
for graph,dtype in [('wk_decode',jnp.bfloat16),('wk_promote',np.float32)]:
    host = np.ones((128,6144),dtype=dtype)
    value = jax.make_array_from_callback(host.shape, NamedSharding(mesh,P()), lambda _:host)
    entry = {}
    preserve_output(value,entry,graph=graph,slots=slots)
    assert len(entry['output']) == 4 and all(r['finite'] for r in entry['output'])
    host[0,0] = np.nan
    bad = jax.make_array_from_callback(host.shape, NamedSharding(mesh,P()), lambda _:host)
    entry = {}
    try: preserve_output(bad,entry,graph=graph,slots=slots)
    except ValueError: pass
    else: raise AssertionError('NaN accepted')
    assert len(entry['output']) == 4 and not any(r['finite'] for r in entry['output'])
print('four CPU owners: both dtypes/hash/finite/refusal PASS')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=4")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_phase_is_never_overwritten_on_preflight_failure(tmp_path, monkeypatch, originals):
    kwargs, events, refs = stage(tmp_path, monkeypatch, originals)
    kwargs["root"].mkdir()
    original = kwargs["root"] / "runner.json"
    original.write_text("historical original\n")
    try:
        with pytest.raises(FileExistsError): wk.prepare(**kwargs)
    finally:
        kwargs["journal"].close()
    assert original.read_text() == "historical original\n" and not events


def test_extracted_exact_materializer_executes_both_original_completed_boundaries(monkeypatch, tmp_path):
    events = []
    raw, decoded, promoted = object(), object(), object()
    def decode(value):
        assert value is raw; events.append("decode"); return decoded
    def promote(value):
        assert value is decoded; events.append("promote"); return promoted
    monkeypatch.setattr(worker, "build_ws32_exact_dsa_materializer_program", lambda *a: NS(decode=decode, promote=promote))
    monkeypatch.setattr(worker, "select_ws32_exact_dsa_raw_weights", lambda *a: raw)
    monkeypatch.setattr(worker, "_write_exact_materializer_graph", lambda **kwargs: {"passed": True})
    class Jit:
        def __init__(self, fn): self.fn = fn
        def lower(self, *a): return self
        def compile(self): return self
        def __call__(self, *a): return self.fn(*a)
        def memory_analysis(self): return NS(**{k:0 for k in wk.runtime.MEMORY_FIELDS})
        def clear_cache(self): events.append("clear")
    def ready(value):
        assert value in (decoded, promoted); events.append("ready"); return value
    args = NS(compile_only=False, hlo_dir=tmp_path, batched_prefill_profile="",
        **{f"expected_{name}_{form}_sha256":"a"*64 for name in ("exact_materialize", "exact_promote")
           for form in ("stablehlo", "optimized_hlo")})
    graphs, seconds, memory = {}, {}, {}
    result = worker._materialize_exact_decode(jax=NS(jit=Jit, block_until_ready=ready),
        args=args, mesh=None, config=NS(exact_dsa=True), weights=raw, batched_prefill=False,
        graphs=graphs, compile_seconds=seconds, compiled_memory=memory, acquisition_journal=None,
        begin_compile=lambda n: events.append(n), record_compile=lambda n: None)
    assert result is promoted
    assert events == ["exact_materialize", "decode", "ready", "exact_promote", "promote", "ready", "clear", "clear"]
    assert set(graphs) == set(seconds) == set(memory) == {"exact_materialize", "exact_promote"}
