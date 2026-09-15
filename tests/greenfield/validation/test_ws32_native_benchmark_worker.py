"""Real native cold orchestration with explicit model/TPU fixtures, not HBM proof."""
from dataclasses import replace
import gc
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

from scripts.greenfield import ws32_native_benchmark_worker as worker
from scripts.greenfield import run_short_decoder_ws32 as original
from scripts.greenfield import ws32_delivery_wk as wk
from scripts.greenfield import ws32_compile_originals as layer
from tests.greenfield.runtime.test_ws32_phase_weights import config, base_arrays
from tests.greenfield.validation.test_ws32_delivery_phase_loading import stage, originals

ROOT = Path(__file__).resolve().parents[3]


def identity():
    return dict(profile=worker.PROFILE, compile_only=False,
        context_capacity=worker.programs.PLAN.context_capacity, jax_process_index=0, launch_process_id=7, code_hash="a" * 40)


@pytest.mark.parametrize("key,value", [("profile", "old"), ("compile_only", True),
    ("context_capacity", 8192), ("jax_process_index", 8), ("launch_process_id", True),
    ("code_hash", "z" * 40)])
def test_journal_rejects_wrong_native_identity(tmp_path, key, value):
    data = identity()
    data[key] = value
    with pytest.raises(ValueError):
        worker.NativeBenchmarkJournal(tmp_path / "journal", data)
    assert not (tmp_path / "journal").exists()


def test_journal_records_native_without_promoting_old_profile(tmp_path):
    journal = worker.NativeBenchmarkJournal(tmp_path / "journal", identity())
    journal.phase("fixture", passed=True)
    journal.close()
    rows = [json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["identity"] == identity()
    assert all(not r["performance_claim"] and not r["numerical_claim"] for r in rows)


def test_native_WK_uses_original42_calls_and_releases_code(tmp_path, monkeypatch, originals):
    kwargs, events, refs = stage(tmp_path, monkeypatch, originals)
    try:
        record = wk.prepare(**kwargs, native_benchmark=True)
    finally:
        kwargs["journal"].close()
    assert record["complete"] and record["phase_contract"] == worker.PROFILE
    assert [name for action, name in events if action == "call"] == list(wk.ROLES) * 21
    assert kwargs["owner"].raw_weights is not None and len(kwargs["owner"].wk) == 21
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("failure", [None, "exact_materialize", "exact_promote", "release"])
def test_exact_completed_boundary_release_and_failure_order(tmp_path, monkeypatch, failure):
    events, refs = [], []
    raw = jnp.ones((2,), jnp.bfloat16)
    monkeypatch.setattr(worker.programs, "require_source", lambda repo: None)
    monkeypatch.setattr(worker.decoder, "select_ws32_exact_dsa_raw_weights", lambda w, c: raw)
    monkeypatch.setattr(worker.decoder, "build_ws32_exact_dsa_materializer_program",
        lambda m, c: NS(decode=lambda x: x + jnp.bfloat16(1), promote=lambda x: x.astype(jnp.float32)))
    class Calls:
        programs = {}
        record = {"call_evidence": []}
        def phase(self, name, action):
            events.append(name)
            result = action()
            if failure == "release" and name.endswith("code_released"):
                raise RuntimeError("fixture release")
            return result
        def call(self, phase, name, values, preserve):
            events.append("dispatch/" + name)
            assert len(self.programs) == (1 if name == "exact_materialize" else 2)
            if failure == name: raise RuntimeError("fixture dispatch")
            if name == "exact_promote":
                assert values[0].dtype == jnp.bfloat16
                np.testing.assert_array_equal(values[0], [2, 2])
            result = self.programs[name](*values)
            jax.block_until_ready(result)
            self.record["call_evidence"].append({"graph": name})
            preserve(result)
            return result
    calls = Calls()
    def compile_one(name, fn, values, **kwargs):
        compiled = fn.lower(*values).compile()
        calls.programs[name] = compiled
        refs.append(weakref.ref(compiled))
        return compiled
    monkeypatch.setattr(worker, "_compile_inspected", compile_one)
    kwargs = dict(mesh=None, config=NS(exact_dsa=True, strategy_nd_dense=True),
                  weights=None, calls=calls, repo=ROOT)
    if failure:
        with pytest.raises(RuntimeError): worker.materialize_exact(**kwargs)
    else:
        value = worker.materialize_exact(**kwargs)
        assert value.dtype == jnp.float32
        np.testing.assert_array_equal(value, [2, 2])
    assert calls.programs == {}
    assert events[-1] == "native_exact/code_released"
    if failure == "exact_materialize": assert "dispatch/exact_promote" not in events
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_loaded_ABI_uses_real_arrays_and_sharding():
    value = jax.device_put(np.ones((2,), np.float32))
    shape = jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
    worker._check_abstract((value,), (shape,), name="fixture")
    for actual in ((jnp.ones((3,)),), (jnp.ones((2,), jnp.int32),), [value]):
        with pytest.raises(ValueError): worker._check_abstract(actual, (shape,), name="fixture")


def cold_fixture(tmp_path, monkeypatch, failure=None):
    cfg = config()
    arrays = base_arrays(cfg)
    events = []
    devices = tuple(NS(id=i) for i in range(4))
    for name, value in (("default_backend", "tpu"), ("device_count", 32),
            ("process_count", 8), ("process_index", 0), ("local_devices", devices)):
        monkeypatch.setattr(jax, name, lambda value=value: value)
    monkeypatch.setattr(original, "_require_clean_code", lambda pin: events.append("source_clean"))
    monkeypatch.setattr(worker.programs, "require_source", lambda repo: None)
    pins = json.loads((ROOT / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    metadata = NS(manifest={"manifest_sha256": pins["expected_manifest_sha256"]},
                  success={"success_sha256": pins["expected_success_sha256"]})
    monkeypatch.setattr(worker.programs, "read_metadata", lambda repo: metadata)
    monkeypatch.setattr(layer, "authenticated_inventory", lambda *a: NS(inventory_sha256="inventory"))
    args = NS(expected_code_hash="a" * 40, process_id=0, context_capacity=worker.programs.PLAN.context_capacity,
        mesh_sha256="mesh", topology_sha256="topology", topology_fleet_sha256="fleet",
        checkpoint_transport="shm", source_inventory=Path("fixture"),
        checkpoint_root=Path("fixture"), checkpoint_manifest_sha256=pins["expected_manifest_sha256"],
        checkpoint_success_sha256=pins["expected_success_sha256"])
    recipe = json.loads((ROOT / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    for field, key in (
        ("strategy_nd_dense_overlay_manifest_sha256", "MANIFEST_SHA"),
        ("strategy_nd_dense_overlay_manifest_file_sha256", "MANIFEST_FILE_SHA"),
        ("strategy_nd_dense_overlay_success_file_sha256", "SUCCESS_FILE_SHA"),
    ):
        setattr(args, field, recipe["GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_" + key])
    if failure == "identity": args.checkpoint_manifest_sha256 = "bad"
    def verify(*a, **kwargs):
        events.append("verify")
        assert kwargs["verify_file_hashes"] is True and kwargs["local_slot_layout"] is True
        assert kwargs["verify_file_hash_slots"] == (0, 1, 2, 3)
        if failure == "verify": raise ValueError("fixture verification")
        return metadata
    monkeypatch.setattr(worker, "verify_ws32_runtime_checkpoint", verify)
    def load(*a, **kwargs):
        events.append("load")
        return NS(arrays=arrays, local_device_slots=[], device_memory_before=[], device_memory_after=[])
    monkeypatch.setattr(worker, "load_ws32_runtime_checkpoint", load)
    def prepare_wk(**kwargs):
        events.append("wk")
        assert kwargs["native_benchmark"] is True
        kwargs["owner"].wk = tuple(jnp.float32(i) for i in range(21))
        kwargs["owner"].phase = "prefill"
        return {"fixture": True}
    monkeypatch.setattr(wk, "prepare", prepare_wk)
    # No call to the destructive one-way begin_decode; both views must survive.
    monkeypatch.setattr(worker.PhaseWeights, "begin_decode", lambda *a: pytest.fail("one-way lifetime"))
    def overlay(**kwargs):
        events.append("overlay")
        if failure == "overlay": raise ValueError("fixture overlay")
        verified = NS(manifest={"manifest_sha256": "overlay"}, manifest_file_sha256="file", success_file_sha256="success")
        kwargs["before_load"](verified)
        raw = replace(cfg, exact_dsa=False, strategy_nd_dense=False)
        extra = set(jax.tree.leaves(worker.decoder.ws32_decoder_weight_names(cfg))) - set(
            jax.tree.leaves(worker.decoder.ws32_decoder_weight_names(raw)))
        kwargs["all_arrays"].update({name: jnp.int32(i) for i, name in enumerate(sorted(extra))})
        return verified, NS()
    monkeypatch.setattr(original, "_load_dense_overlay", overlay)
    monkeypatch.setattr(worker.preparation, "overlay_preflight", lambda *a: events.append("overlay_budget"))
    monkeypatch.setattr(worker.preparation, "overlay_completed", lambda *a: events.append("overlay_peak"))
    monkeypatch.setattr(worker.preparation, "finish", lambda calls: dict(calls.record, complete=True))
    def materialize(**kwargs):
        events.append("exact")
        return (jnp.float32(8),)
    monkeypatch.setattr(worker, "materialize_exact", materialize)
    class Function:
        def clear_cache(self): pass
    def prepare(*a, **kw):
        events.append("programs")
        inputs = (None, None, NS(prompt_length=None), None, None, None, None)
        return NS(programs={name: NS(execute=Function()) for name in ("prefill_chunk", "prefill_tail")},
                  inputs={name: inputs for name in ("prefill_chunk", "prefill_tail")})
    monkeypatch.setattr(worker.programs, "prepare", prepare)
    monkeypatch.setattr(worker.programs, "prepare_companions", lambda *a, **kw: NS(
        programs={name: Function() for name in ("observer", "decode", "cache_probe")},
        inputs={name: (None,) * 6 for name in ("observer", "decode", "cache_probe")}))
    monkeypatch.setattr(worker.programs, "build_cache_initializer", lambda *a: Function())
    monkeypatch.setattr(worker, "_check_abstract", lambda *a, **kw: events.append("abi/" + kw["name"]))
    def compile_one(name, fn, values, calls, **kwargs):
        events.append("compile/" + name)
        if failure == name: raise ValueError("fixture compile")
        calls.record["programs"][name] = {"admission": {"passed": True}}
        calls.programs[name] = Function()
        return calls.programs[name]
    monkeypatch.setattr(worker, "_compile_inspected", compile_one)
    # Production host rotary builder still runs (33MB), placement is a tiny
    # explicit fixture. No actual TPU or initializer allocation is attempted.
    from scripts.greenfield import ws32_batched_prefill_runner as batched
    monkeypatch.setattr(batched, "replicated", lambda mesh, host: jnp.float32(0))
    def bind(**kwargs):
        events.append("bind")
        assert len(kwargs["wk"]) == 21 and kwargs["raw_weights"] is not None
        assert kwargs["decode_weights"] is not None and kwargs["exact_weights"] is not None
        assert set(kwargs["compiled"]) == set(worker.memory.ROLES)
        return NS(resident_roots=lambda: {})
    monkeypatch.setattr(worker, "bind_admitted_runtime", bind)
    monkeypatch.setattr(worker.memory, "make_record", lambda *a, **kw: {"fixture": True})
    def validate(record):
        events.append("memory")
        if failure == "memory": raise ValueError("fixture memory")
    monkeypatch.setattr(worker.memory, "validate_record", validate)
    kwargs = dict(args=args, repo=ROOT, root=tmp_path / "cold", mesh=None,
        physical_mesh=NS(mesh_hash="mesh", flattened_device_ids=tuple(range(32))),
        topology=NS(topology_hash="topology"), fleet_sha="fleet", consensus=lambda ok: ok,
        preserve_memory=lambda *a: None)
    return kwargs, events


@pytest.mark.parametrize("failure", [None, "identity", "verify", "overlay", "decode", "memory"])
def test_complete_cold_load_order_and_fail_closed(tmp_path, monkeypatch, failure):
    kwargs, events = cold_fixture(tmp_path, monkeypatch, failure)
    if failure:
        with pytest.raises(ValueError): worker.load_runtime(**kwargs)
        assert not json.loads((kwargs["root"] / "runner.json").read_text())["complete"]
    else:
        loaded = worker.load_runtime(**kwargs)
        assert loaded.record["complete"]
        assert events.index("verify") < events.index("load") < events.index("wk")
        assert events.index("wk") < events.index("overlay") < events.index("exact") < events.index("programs")
        assert events[-2:] == ["bind", "memory"]
        assert [e for e in events if e.startswith("compile/")] == ["compile/" + n for n in worker.memory.ROLES]
    if failure in ("identity", "verify"): assert "load" not in events
    if failure == "overlay": assert "exact" not in events
    if failure == "decode": assert "bind" not in events


def test_peer_preflight_failure_does_not_create_or_load(tmp_path, monkeypatch):
    kwargs, events = cold_fixture(tmp_path, monkeypatch)
    kwargs["consensus"] = lambda ok: False
    with pytest.raises(RuntimeError, match="peer preflight"): worker.load_runtime(**kwargs)
    assert not kwargs["root"].exists() and "load" not in events


def test_wrong_overlay_refused_before_checkpoint_load(tmp_path, monkeypatch):
    kwargs, events = cold_fixture(tmp_path, monkeypatch)
    kwargs["args"].strategy_nd_dense_overlay_manifest_sha256 = "b" * 64
    with pytest.raises(ValueError, match="tested checkpoint recipe"):
        worker.load_runtime(**kwargs)
    assert not kwargs["root"].exists() and "load" not in events


def test_raw_mismatch_refuses_before_expensive_compile(tmp_path):
    events = []
    class Function:
        def lower(self): return self
        def compiler_ir(self, dialect): return "different raw"
        def compile(self): events.append("compile"); pytest.fail("compiled wrong RAW")
    with pytest.raises(ValueError, match="before compilation"):
        layer.compile_program(worker._RawCheckedFunction(Function(), (1, "0" * 64)),
                              (), "fixture", tmp_path, {})
    assert events == [] and (tmp_path / "fixture.stablehlo.mlir").read_text() == "different raw"


def test_actual_production_materializer_raw_through_native_worker():
    code = r'''
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax, numpy as np
from jax.sharding import Mesh
from scripts.greenfield import ws32_native_benchmark_worker as worker
from scripts.greenfield import ws32_native_benchmark_programs as programs
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
metadata=NS(manifest=json.loads(Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/manifest.json').read_text()))
pair=programs.prepare(mesh,metadata,repo=Path.cwd())
comp=programs.prepare_companions(mesh,pair,repo=Path.cwd())
from scripts.greenfield.ws32_phase_weights import PhaseWeights
from scripts.greenfield.ws32_dense_frontier_admission import RAW as WK_RAW
raw_config=worker.decoder.Ws32DecoderConfig(comp.config.geometry,worker.programs.PLAN.context_capacity,host_main_rope_table=True)
arrays=dict(zip(jax.tree.leaves(worker.decoder.ws32_decoder_weight_names(raw_config)),
                jax.tree.leaves(pair.inputs['prefill_chunk'][3]),strict=True))
owner=PhaseWeights(arrays,comp.config)
for name,fn,values in owner.wk_jobs(mesh):
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(fn.trace(*values).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    assert (len(raw),sha256(raw).hexdigest())==WK_RAW[name], (name,len(raw),sha256(raw).hexdigest())
    print('NATIVE_WK_RAW',name,len(raw),sha256(raw).hexdigest(),flush=True)
class Calls:
    programs={}
    record={'call_evidence':[]}
    def phase(self,name,action): return action()
    def call(self,phase,name,values,preserve):
        # No dispatch: shape of the ORIGINAL completed boundary feeds promotion.
        self.record['call_evidence'].append({'graph':name})
        return comp.inputs['exact_promote'][0] if name=='exact_materialize' else comp.inputs['decode'][3]
def raw_only(name,fn,values,*,calls,repo):
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(fn.trace(*values).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    assert (len(raw),sha256(raw).hexdigest())==programs.RAW[name], (name,len(raw),sha256(raw).hexdigest())
    print('NATIVE_PREPARATION_RAW',name,len(raw),sha256(raw).hexdigest(),flush=True)
    calls.programs[name]=NS()
    return calls.programs[name]
with patch.object(worker,'_compile_inspected',raw_only),patch('jax.device_put',side_effect=AssertionError('payload allocation')):
    value=worker.materialize_exact(mesh=mesh,config=comp.config,weights=comp.inputs['decode'][2],calls=Calls(),repo=Path.cwd())
assert value is comp.inputs['decode'][3]
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32"),
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("NATIVE_PREPARATION_RAW") == 2
    assert result.stdout.count("NATIVE_WK_RAW") == 2
