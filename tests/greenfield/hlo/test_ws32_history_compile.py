"""Seven-job metadata preparation and compiler-file refusal checks, CPU only."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts.greenfield import ws32_history_compile as compile_history
from scripts.greenfield import ws32_history_protocol as protocol

ROOT = Path(__file__).resolve().parents[3]


def test_production_metadata_and_declared_sharding_without_payload_or_dispatch():
    source = r'''
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_history_compile as candidate
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
original_open = Path.open
opened = []
def checked(path, *args, **kwargs):
    assert path.suffix not in ('.safetensors', '.bin'), path
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json', 'SUCCESS'), path
    opened.append(str(path))
    return original_open(path, *args, **kwargs)
with patch.object(Path, 'open', checked), \
     patch('jax.device_put', side_effect=AssertionError('concrete placement')), \
     patch('jax.stages.Compiled.__call__', side_effect=AssertionError('compiled dispatch')), \
     patch('jax.stages.Lowered.compile', side_effect=AssertionError('unexpected compilation')):
    metadata = candidate.read_metadata(Path.cwd())
    prepared = candidate.prepare(mesh, metadata, repo=Path.cwd())
    assert set(prepared.programs) == set(prepared.inputs) == set(candidate.PROGRAMS)
    assert 'wk_decode' not in prepared.programs and 'wk_promote' not in prepared.programs
    assert len(metadata.manifest['tensor_schema']) == 2310
    assert prepared.manifest_sha256 == metadata.manifest['manifest_sha256']
    assert prepared.source_inventory_sha256 == metadata.manifest['source']['inventory_sha256']
    assert all(isinstance(x, jax.ShapeDtypeStruct) and x.sharding is not None
               for x in jax.tree.leaves(prepared.inputs))
    for name in candidate.PROGRAMS:
        assert callable(prepared.programs[name].execute.lower)
    sizes = candidate.abstract_io_bytes(prepared, mesh)
    assert sizes == candidate.ABSTRACT_IO_BYTES, sizes
    for name, (arguments, outputs) in sizes.items():
        assert arguments <= candidate.memory_caps(name)['argument_size_in_bytes']
        assert outputs <= candidate.memory_caps(name)['output_size_in_bytes']
    for prefix in ('candidate', 'control'):
        assert prepared.inputs[prefix + '_b128'][0].shape == (128,)
        assert prepared.inputs[prefix + '_b114'][0].shape == (114,)
    decoded = prepared.inputs['exact_promote'][0]
    exact = prepared.inputs['observer'][8]
    assert len(decoded) == len(exact) == 4
    assert decoded[0].wq_b_weight_local.sharding.shard_shape(decoded[0].wq_b_weight_local.shape) == (1024, 2048)
    assert len(exact[0].wq_b_weight_aliases) == 4
    assert all(v.sharding.shard_shape(v.shape) == (1024, 2048) for v in exact[0].wq_b_weight_aliases)
    with patch('jax._src.tpu_custom_call.get_ir_version', return_value=None):
        for name in candidate.PROGRAMS:
            raw = str(prepared.programs[name].execute.trace(*prepared.inputs[name]).lower(
                lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
            assert (len(raw), sha256(raw).hexdigest()) == candidate.RAW[name], name
            print('HISTORY_RAW_PIN', name, len(raw), sha256(raw).hexdigest(), flush=True)
assert any(p.endswith('/manifest.json') for p in opened)
print('HISTORY_SEVEN_METADATA_PASS', flush=True)
'''
    flags = f"{os.environ.get('XLA_FLAGS', '').strip()} --xla_force_host_platform_device_count=32".strip()
    result = subprocess.run([sys.executable, "-c", source], text=True, cwd=ROOT,
        capture_output=True, timeout=600,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HISTORY_SEVEN_METADATA_PASS" in result.stdout


@pytest.mark.parametrize("name", list(compile_history.SOURCE_SHA256))
def test_diagnostic_source_mutation_refuses(monkeypatch, name):
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + (b"changed" if p == ROOT / name else b""))
    with pytest.raises(ValueError, match="diagnostic source"):
        compile_history.require_source(ROOT)


def test_overlay_metadata_is_verified_during_pre_runtime_read(monkeypatch):
    from glm_tpu.greenfield.checkpoint import ws32_strategy_nd_dense as overlay

    def refuse(*args, **kwargs):
        assert set(kwargs) == {"expected_manifest_sha256", "expected_manifest_file_sha256", "expected_success_file_sha256"}
        raise ValueError("injected overlay metadata refusal")

    monkeypatch.setattr(compile_history, "RAW", {name: (1, "a" * 64) for name in compile_history.PROGRAMS})
    monkeypatch.setattr(compile_history.canonical, "read_metadata", lambda repo: object())
    monkeypatch.setattr(overlay, "verify_ws32_strategy_nd_dense_overlay", refuse)
    with pytest.raises(ValueError, match="injected overlay"):
        compile_history.read_metadata(ROOT)


def test_caps_cover_each_declared_output_and_reject_unknown_jobs():
    assert compile_history.memory_caps("exact_decode")["output_size_in_bytes"] == 128 << 20
    assert compile_history.memory_caps("exact_promote")["output_size_in_bytes"] == 256 << 20
    for name in compile_history.PROGRAMS:
        caps = compile_history.memory_caps(name)
        assert caps == protocol.memory_caps(name)
        assert caps["argument_size_in_bytes"] == 2 << 30
        assert caps["temp_size_in_bytes"] == 1 << 30
        assert caps["generated_code_size_in_bytes"] == 128 << 20
        assert caps["alias_size_in_bytes"] == 0
        assert compile_history.ABSTRACT_IO_BYTES[name][1] <= caps["output_size_in_bytes"]
        caps["output_size_in_bytes"] = 0
        assert compile_history.memory_caps(name)["output_size_in_bytes"] > 0
    with pytest.raises(ValueError, match="unregistered"):
        compile_history.memory_caps("wk_decode")
    with pytest.raises(ValueError, match="unregistered"):
        protocol.memory_caps("foreign")


def _originals(root, monkeypatch):
    record = dict(kernel=compile_history.KERNEL, protocol=compile_history.PROTOCOL,
        profile=compile_history.PROFILE, compile_only=True, weights_loaded=False,
        model_executable_calls=0, numerical_claim=False, performance_claim=False, programs={})
    pins = {}
    for name in compile_history.PROGRAMS:
        stable, optimized = f"raw {name}".encode(), f"optimized {name}".encode()
        (root / f"{name}.stablehlo.mlir").write_bytes(stable)
        (root / f"{name}.optimized_hlo.txt").write_bytes(optimized)
        pins[name] = (len(stable), sha256(stable).hexdigest())
        arguments, outputs = compile_history.ABSTRACT_IO_BYTES[name]
        record["programs"][name] = dict(stablehlo_sha256=pins[name][1],
            optimized_hlo_sha256=sha256(optimized).hexdigest(),
            compiled_memory=dict(argument_size_in_bytes=arguments, output_size_in_bytes=outputs,
                                 alias_size_in_bytes=0, temp_size_in_bytes=0, generated_code_size_in_bytes=1))
    monkeypatch.setattr(compile_history, "RAW", pins)
    return record


def test_all_seven_originals_are_bound_before_any_cap_check(tmp_path, monkeypatch):
    record = _originals(tmp_path, monkeypatch)
    report = compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)
    assert set(report["graphs"]) == set(compile_history.PROGRAMS)
    assert not report["numerical_claim"] and not report["performance_claim"]
    read, observed = Path.read_bytes, set()

    def tracked(path):
        if path.parent == tmp_path:
            observed.add(path.name)
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    record["programs"][compile_history.PROGRAMS[0]]["compiled_memory"]["temp_size_in_bytes"] = 2 << 30
    with pytest.raises(ValueError, match="allocation"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)
    assert observed == {f"{name}.{form}" for name in compile_history.PROGRAMS
                        for form in ("stablehlo.mlir", "optimized_hlo.txt")}


@pytest.mark.parametrize("field,value", [("compile_only", False), ("weights_loaded", True),
    ("model_executable_calls", 1), ("model_executable_calls", False), ("numerical_claim", True),
    ("performance_claim", True), ("kernel", "other"), ("protocol", "other"), ("profile", "other")])
def test_wrong_identity_or_dispatch_declarations_refuse(tmp_path, monkeypatch, field, value):
    record = _originals(tmp_path, monkeypatch)
    record[field] = value
    with pytest.raises(ValueError, match="identity or no-dispatch"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


@pytest.mark.parametrize("name", compile_history.PROGRAMS)
def test_every_graph_is_required_and_digest_bound(tmp_path, monkeypatch, name):
    record = _originals(tmp_path, monkeypatch)
    incomplete = deepcopy(record)
    del incomplete["programs"][name]
    with pytest.raises(ValueError, match="seven"):
        compile_history.validate_preserved_pair(tmp_path, incomplete, repo=ROOT)
    path = tmp_path / f"{name}.optimized_hlo.txt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="graph identity"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


@pytest.mark.parametrize("field,value", [("output_size_in_bytes", 257 << 20), ("alias_size_in_bytes", 1),
    ("argument_size_in_bytes", 3 << 30), ("temp_size_in_bytes", -1), ("temp_size_in_bytes", True),
    ("generated_code_size_in_bytes", 129 << 20)])
def test_allocation_caps_and_types_refuse(tmp_path, monkeypatch, field, value):
    record = _originals(tmp_path, monkeypatch)
    record["programs"]["exact_promote"]["compiled_memory"][field] = value
    with pytest.raises(ValueError, match="allocation"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


def test_unregistered_raw_table_refuses_without_admission(tmp_path, monkeypatch):
    record = _originals(tmp_path, monkeypatch)
    monkeypatch.setattr(compile_history, "RAW", {})
    with pytest.raises(ValueError, match="preregistered"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


@pytest.mark.parametrize("memory", [None, [], {}, {"temp_size_in_bytes": 0}])
def test_incomplete_memory_inventory_refuses_after_original_identity(tmp_path, monkeypatch, memory):
    record = _originals(tmp_path, monkeypatch)
    record["programs"]["candidate_b128"]["compiled_memory"] = memory
    with pytest.raises(ValueError, match="allocation"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


def test_modified_stable_original_cannot_repin_itself_from_record(tmp_path, monkeypatch):
    record = _originals(tmp_path, monkeypatch)
    path = tmp_path / "observer.stablehlo.mlir"
    changed = path.read_bytes() + b"modified source"
    path.write_bytes(changed)
    record["programs"]["observer"]["stablehlo_sha256"] = sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="graph identity"):
        compile_history.validate_preserved_pair(tmp_path, record, repo=ROOT)


@pytest.mark.parametrize("mutation", ["missing", "extra", "zero", "boolean", "negative", "bad_sha", "upper_sha"])
def test_invalid_raw_registration_refuses_before_metadata_runtime_work(monkeypatch, mutation):
    pins = {name: (1, "a" * 64) for name in compile_history.PROGRAMS}
    name = compile_history.PROGRAMS[0]
    if mutation == "missing":
        del pins[name]
    elif mutation == "extra":
        pins["unregistered"] = (1, "a" * 64)
    elif mutation == "zero":
        pins[name] = (0, "a" * 64)
    elif mutation == "boolean":
        pins[name] = (True, "a" * 64)
    elif mutation == "negative":
        pins[name] = (-1, "a" * 64)
    elif mutation == "bad_sha":
        pins[name] = (1, "a" * 63)
    else:
        pins[name] = (1, "A" * 64)
    monkeypatch.setattr(compile_history, "RAW", pins)
    monkeypatch.setattr(compile_history, "require_source", lambda repo:
                        pytest.fail("invalid raw registration reached metadata/source work"))
    with pytest.raises(ValueError, match="preregis"):
        compile_history.read_metadata(ROOT)
