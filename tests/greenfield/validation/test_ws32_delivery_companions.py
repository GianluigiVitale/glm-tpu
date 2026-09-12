"""Original capacity HLO through actual writer/JSON/sealer; zero TPU calls."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_delivery_companions as companions
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield.ws32_acquisition_journal import Ws32DeliveryJournal, Ws32NumericalJournal

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/home/gianl/glm-run")


@pytest.mark.parametrize("label", ["128k_d1_0", "256k_e0"])
def test_actual_capacity_companions_worker_json_sealer(tmp_path, label):
    capacity = runtime.programs.long_plan(label).context_capacity
    db, tag, runner_sha = companions.ORIGINALS[capacity]
    raw_record = (BASE / tag / "runner.rank0.json").read_bytes()
    assert sha256(raw_record).hexdigest() == runner_sha
    original = json.loads(raw_record)
    pins = companions.raw_registration(label, historical=True)
    assert original["status"] == "SUCCESS"
    args = NS(batched_prefill_profile=runtime.PROFILE, context_label=label,
              exact_dsa=1, strategy_nd_dense=1, host_main_rope_table=1)
    for graph in companions.ROLES:
        stable = (BASE / tag / "hlo" / f"{graph}.stablehlo.mlir").read_text()
        optimized = (BASE / tag / "hlo" / f"{graph}.optimized_hlo.txt").read_text()
        assert sha256(stable.encode()).hexdigest() == original["graphs"][graph]["stablehlo_sha256"] == pins[graph]
        assert sha256(optimized.encode()).hexdigest() == original["graphs"][graph]["optimized_hlo_sha256"]
        kwargs = dict(graph=graph, lowered=NS(compiler_ir=lambda **_: stable),
            compiled=NS(as_text=lambda: optimized), hlo_dir=tmp_path,
            expected_stable=pins[graph], expected_optimized=runtime.hlo.FRESH_OPTIMIZED_MARKER,
            batched_profile=runtime.PROFILE, long_context_label=label)
        if graph.startswith("exact_"):
            report = worker._write_exact_materializer_graph(**kwargs)
        else:
            report, _, _ = worker._write_graph(**kwargs, hidden_size=6144,
                exact_dsa=True, strategy_nd_dense=True, host_main_rope_table=True)
        saved = json.loads(json.dumps(report, allow_nan=False))
        setattr(args, f"expected_{graph}_stablehlo_sha256", pins[graph])
        setattr(args, f"expected_{graph}_optimized_hlo_sha256", runtime.hlo.FRESH_OPTIMIZED_MARKER)
        assert sealer._replay_batched_graph(stable, optimized, graph=graph, args=args) == saved
        assert saved["passed"] and not saved["violations"]
        assert saved["maximum_group_size"] <= 8
        identity = saved["source_location_identity"]
        assert identity["optimized_identity_policy"] == runtime.hlo.FRESH_POLICY
        assert not identity["numerical_inheritance"] and not identity["dispatch_authorized"]
        for suffix, text in (("stablehlo.mlir", stable), ("optimized_hlo.txt", optimized)):
            path = tmp_path / f"{graph}.{suffix}"
            assert path.read_text() == text
            path.unlink()  # own ephemeral test copies, originals stay in BASE
        print(f"DB{db} original {graph}: worker/JSON/sealer PASS", flush=True)


def request(label):
    from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
    plan = runtime.programs.long_plan(label)
    entry = WS32_LONG_CONTEXT_PROFILES[label]
    pins = companions.raw_registration(label) | {
        role: values[1] for role, values in runtime.programs.raw_registration(label).items()}
    return NS(prefill_mode=worker.PREFILL_MODE, batched_prefill_profile=runtime.PROFILE,
        delivery_context_label=label, compile_only=0, prefill_chunk=128,
        context_capacity=plan.context_capacity, exact_dsa=1, strategy_nd_dense=1,
        host_main_rope_table=1, rotary_diagnostic=0, observer_steps=14 if label == "256k_e0" else 20,
        warmup=2, iterations=256 if label == "256k_e0" else 10, trace_steps=2,
        prefill_memory_reserve_bytes=runtime.RESERVE,
        prefill_budget_seconds=runtime.budget_seconds(label),
        long_context=entry["kind"], long_context_manifest_sha256=entry["manifest_sha256"],
        long_context_success_sha256=entry["success_sha256"], dsa_adjudication_record=None,
        dsa_adjudication_sha256="0" * 64,
        **{f"expected_{graph}_{form}_sha256": digest if form == "stablehlo" else "0" * 64
           for graph, digest in pins.items() for form in ("stablehlo", "optimized_hlo")})


@pytest.mark.parametrize("label", ["128k_d0_0", "128k_d0_05", "128k_d0_95", "128k_d1_0", "256k_e0"])
def test_fixed_request_and_actual_long_journal(tmp_path, label):
    args = request(label)
    runtime.require_request(args, context_label=label, prompt_length=runtime.programs.long_plan(label).prompt_length, repo=ROOT)
    identity = runtime.numerical_identity(label)
    assert identity["batched_prefill_plan"] == runtime.identity(label)
    assert identity["batched_prefill_plan"]["donate_argnums"] == ([2] if label == "256k_e0" else [])
    identity.update(compile_only=False, code_hash="a" * 40, hostname="fixture")
    path = tmp_path / "journal.jsonl"
    journal = Ws32DeliveryJournal(path, identity)
    journal.phase("runtime_initialize_started")
    journal.begin("wk_decode")
    journal.compiled("wk_decode", seconds=1.0, memory={"fixture": 1}, device_memory=[])
    journal.inspect("wk_decode", "fixture raw", "fixture optimized", lambda: {"passed": True})
    journal.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["identity"] == identity
    assert [r["stage"] for r in rows] == ["identity", "runtime_initialize_started", "lower_compile_started", "compiled", "raw_written", "inspected"]
    assert all(r["artifact_kind"] == Ws32DeliveryJournal.artifact_kind and not r["numerical_claim"] for r in rows)
    with pytest.raises(ValueError): Ws32NumericalJournal(tmp_path / "old.jsonl", identity)
    assert not (tmp_path / "old.jsonl").exists()
    with pytest.raises(FileExistsError): Ws32DeliveryJournal(path, identity)


@pytest.mark.parametrize("field,value", [
    ("prefill_mode", worker.SERIAL_PREFILL_MODE), ("batched_prefill_profile", ""),
    ("delivery_context_label", "128k_d0_0"), ("compile_only", 1),
    ("prefill_chunk", 32), ("context_capacity", 262144), ("observer_steps", 20),
    ("iterations", 10), ("exact_dsa", True), ("strategy_nd_dense", 0),
    ("host_main_rope_table", 0), ("rotary_diagnostic", 1), ("prefill_budget_seconds", float("nan")),
    ("prefill_memory_reserve_bytes", runtime.RESERVE-1), ("dsa_adjudication_record", Path("old.json")),
    ("long_context_manifest_sha256", "a"*64), ("long_context_success_sha256", "a"*64),
    ("expected_observer_stablehlo_sha256", "a"*64), ("expected_exact_promote_optimized_hlo_sha256", "a"*64),
])
def test_request_refuses_before_source_or_payload(monkeypatch, field, value):
    args = request("256k_e0")
    setattr(args, field, value)
    monkeypatch.setattr(runtime.programs, "require_source", lambda _: pytest.fail("late source check"))
    with pytest.raises(ValueError):
        runtime.require_request(args, context_label="256k_e0", prompt_length=262144, repo=ROOT)


@pytest.mark.parametrize("label", ["128k_d1_0", "256k_e0"])
def test_retained_raw_replay_does_not_authorize_historical_request(label):
    args = request(label)
    args.expected_observer_stablehlo_sha256 = companions.raw_registration(label, historical=True)["observer"]
    with pytest.raises(ValueError, match="fixed RAW"):
        runtime.require_request(args, context_label=label,
            prompt_length=runtime.programs.long_plan(label).prompt_length, repo=ROOT)


@pytest.mark.parametrize("field", ["validation_contract", "batched_prefill_plan", "long_workload", "batched_prefill_program_options", "compile_only"])
def test_journal_cannot_relabel_contract_before_creating_file(tmp_path, field):
    identity = deepcopy(runtime.numerical_identity("256k_e0"))
    identity["compile_only"] = False
    identity[field] = None
    with pytest.raises(ValueError): Ws32DeliveryJournal(tmp_path / "refused.jsonl", identity)
    assert not (tmp_path / "refused.jsonl").exists()


def test_fresh_digest_is_not_literal_identity_or_permission_to_skip_checks(monkeypatch):
    assert runtime.hlo.optimized_identity("actual", "0"*64) == (sha256(b"actual").hexdigest(), runtime.hlo.FRESH_POLICY)
    for expected in (None, "A"*64, "a"*64):
        with pytest.raises(ValueError): runtime.hlo.optimized_identity("actual", expected)
    with pytest.raises(ValueError): runtime.hlo.optimized_identity("", "0"*64)
    monkeypatch.setattr(companions.delivery.programs, "require_source", lambda _: pytest.fail("unbound RAW parsed"))
    with pytest.raises(ValueError, match="RAW"):
        companions.inspect_hlo("changed raw", "actual", repo=ROOT, context_label="256k_e0",
            graph="decode", expected_stable=companions.raw_registration("256k_e0")["decode"], expected_optimized="0"*64)


def test_current_production_companion_raw_before_expensive_prefill():
    # Reuse authenticated metadata/abstract prefill inputs and original decoder
    # builders. This is CPU TPU-target lowering, NOT TPU compilation/execution.
    source = r'''
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax, jax.numpy as jnp, numpy as np
import base64, re
from functools import lru_cache
from jax._src.interpreters import mlir
from jax._src.lib import tpu
from jax._src.lib.mlir import ir
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_decoder as dec
from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import _LOCAL_CONTRACT, strategy_nd_dense_tensor_names
from scripts.greenfield import ws32_delivery_programs as programs
from scripts.greenfield import ws32_delivery_companions as companions
from scripts.greenfield.run_short_decoder_ws32 import _geometry
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
old = Path.open
def checked(path, *a, **k):
    assert path.suffix not in ('.safetensors', '.bin'), path
    if '/glm-ws32-runtime/' in str(path): assert path.name in ('manifest.json','SUCCESS'), path
    return old(path, *a, **k)
def abstract(shape, dtype, spec=P()):
    return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))
def placed(values, specs):
    return jax.tree.map(lambda value,spec: abstract(value.shape,value.dtype,spec), values, specs)
body_pattern = re.compile(r'\\22body\\22: \\22([A-Za-z0-9+/=]+)\\22')
@lru_cache(maxsize=2048)
def body_text(body):
    with mlir.make_ir_context() as ctx:
        ctx.allow_unregistered_dialects = True
        tpu.register_dialect(ctx)
        mod = ir.Module.parse(base64.b64decode(body,validate=True),context=ctx)
        return mod.operation.get_asm(enable_debug_info=False)
with patch.object(Path, 'open', checked), patch('jax.device_put', side_effect=AssertionError('placement')), \
     patch('jax.stages.Compiled.__call__', side_effect=AssertionError('dispatch')), \
     patch('jax.stages.Lowered.compile', side_effect=AssertionError('compile')):
    metadata = programs.read_metadata(Path.cwd())
    for label in ('128k_d1_0','256k_e0'):
        pair = programs.prepare(mesh,metadata,repo=Path.cwd(),context_label=label)
        _,_,state,raw_weights,_,rope = pair.inputs['prefill_chunk']
        config = dec.Ws32DecoderConfig(_geometry(), programs.long_plan(label).context_capacity,
            exact_dsa=True,strategy_nd_dense=True,host_main_rope_table=True)
        raw_config = replace(config,exact_dsa=False,strategy_nd_dense=False)
        arrays = dict(zip(jax.tree.leaves(dec.ws32_decoder_weight_names(raw_config)),jax.tree.leaves(raw_weights),strict=True))
        arrays.update({name:abstract(shape,dtype,P(*spec)) for i in range(3)
            for name,(_,dtype,shape,spec) in zip(strategy_nd_dense_tensor_names(i),_LOCAL_CONTRACT,strict=True)})
        weights = dec.bind_ws32_decoder_weights(
            {name:arrays[name] for name in jax.tree.leaves(dec.ws32_decoder_weight_names(config))},config)
        materializer = dec.build_ws32_exact_dsa_materializer_program(mesh,config)
        raw = dec.select_ws32_exact_dsa_raw_weights(weights,config)
        decoded = placed(jax.eval_shape(materializer.decode,raw),dec.ws32_decoded_exact_dsa_specs(config))
        exact = placed(jax.eval_shape(materializer.promote,decoded),dec.ws32_exact_dsa_specs(config))
        decoder = dec.build_ws32_decoder_program(mesh,config)
        inputs = (abstract((1,),jnp.int32),state.decoder,weights,exact,rope)
        jobs = (
            ('exact_materialize',jax.jit(materializer.decode),(raw,)),
            ('exact_promote',jax.jit(materializer.promote),(decoded,)),
            ('observer',jax.jit(decoder.observe,donate_argnums=(1,)),inputs),
            ('decode',jax.jit(decoder.execute,donate_argnums=(1,)),inputs),
            ('cache_probe',jax.jit(decoder.probe_cache_write),(state.decoder,)),
        )
        for graph,fn,args in jobs:
            assert all(isinstance(v,jax.ShapeDtypeStruct) and v.sharding is not None for v in jax.tree.leaves(args))
            with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
                raw_text = str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
            digest=sha256(raw_text).hexdigest()
            print(label,graph,len(raw_text),digest,flush=True)
            tag = companions.ORIGINALS[config.context_capacity][1]
            original = (Path('/home/gianl/glm-run')/tag/'hlo'/f'{graph}.stablehlo.mlir').read_text()
            # Diagnostic only, never a runtime normalizer. The current RAW
            # gets its own literal registration; originals are not relabelled.
            current = raw_text.decode()
            assert body_pattern.sub('BODY',original) == body_pattern.sub('BODY',current)
            old_bodies,new_bodies = body_pattern.findall(original),body_pattern.findall(current)
            assert len(old_bodies) == len(new_bodies)
            changed = 0
            for old_body,new_body in zip(old_bodies,new_bodies,strict=True):
                if old_body != new_body:
                    assert body_text(old_body) == body_text(new_body)
                    changed += 1
            print('NONDEBUG_KERNEL_BODIES_EQUAL',label,graph,changed,flush=True)
print('PRODUCTION_LONG_COMPANION_RAW_PASS',flush=True)
'''
    flags = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip()
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, text=True,
        capture_output=True, timeout=480,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRODUCTION_LONG_COMPANION_RAW_PASS" in result.stdout
    import re
    observed = {(label, graph): digest for label, graph, digest in re.findall(
        r"(?m)^(128k_d1_0|256k_e0) (\w+) \d+ ([0-9a-f]{64})$", result.stdout)}
    expected = {(label, graph): digest for label in ("128k_d1_0", "256k_e0")
                for graph, digest in companions.raw_registration(label).items()}
    assert observed == expected
    print(result.stdout, flush=True)
