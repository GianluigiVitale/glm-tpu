"""Distinct full-model metadata recipe; old sources/profiles stay fail-closed."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_canonical_prefill_compile as candidate
from scripts.greenfield import ws32_dense_canonical_source as reduced
from scripts.greenfield import ws32_rolled_prefill_compile as original


ROOT = Path(__file__).resolve().parents[3]


def test_current_three_file_scope_and_old_guard_refusal():
    candidate.require_source(ROOT)
    assert set(candidate.MODEL_SOURCE_OVERRIDES) == {
        *reduced.MODEL_SOURCE_OVERRIDES,
        "glm_tpu/greenfield/runtime/ws32_batched_prefill.py",
    }
    with pytest.raises(ValueError, match="unrelated frozen"):
        reduced.require_source(ROOT)
    with pytest.raises(ValueError, match="source differs"):
        admission.require_acquired_model_source(
            ROOT, profile=admission.FROZEN_8K_PROFILE
        )
    with pytest.raises(ValueError, match="not registered"):
        admission.profile_is_paired(candidate.PROFILE)


@pytest.mark.parametrize("change", ["runtime", "evidence", "unrelated"])
def test_source_and_prerequisite_changes_refuse(monkeypatch, change):
    read = Path.read_bytes
    name = (
        "glm_tpu/greenfield/runtime/ws32_batched_prefill.py"
        if change == "runtime"
        else next(iter(candidate.PREREQUISITES))
    )
    if change == "unrelated":

        def changed(command, **kwargs):
            assert all(
                ":(exclude)" + p in command for p in candidate.MODEL_SOURCE_OVERRIDES
            )
            assert admission.FROZEN_SOURCE_PIN in command
            return SimpleNamespace(returncode=1)

        monkeypatch.setattr(candidate.subprocess, "run", changed)
    else:
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: read(p) + (b"changed" if p == ROOT / name else b""),
        )
    with pytest.raises(ValueError, match="canonical"):
        candidate.require_source(ROOT)


def test_modes_are_exclusive_and_strict():
    for kwargs in (
        {"full_canonical": 1},
        {"canonical_dense": True, "full_canonical": True},
    ):
        with pytest.raises(ValueError):
            original.read_metadata(ROOT, **kwargs)
    with pytest.raises(ValueError, match="static bool"):
        original.prepare(None, None, repo=ROOT, full_canonical=1)


def test_real_metadata_abstract_pair_without_payload_or_device_put():
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_paired_prefill.py")
    )
    source = r"""
from pathlib import Path
from unittest.mock import patch
import jax,numpy as np
from jax.sharding import Mesh
from scripts.greenfield import ws32_canonical_prefill_compile as candidate
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
opened=[];old=Path.open
def checked(path,*args,**kwargs):
    if '/glm-ws32-runtime/' in str(path):assert path.name in ('manifest.json','SUCCESS'),path
    assert path.suffix not in ('.safetensors','.bin'),path
    opened.append(str(path));return old(path,*args,**kwargs)
with patch.object(Path,'open',checked),patch('jax.device_put',side_effect=AssertionError('payload placement')):
    metadata=candidate.read_metadata(Path.cwd())
    prepared=candidate.prepare(mesh,metadata,repo=Path.cwd())
assert any(p.endswith('/manifest.json') for p in opened)
assert len(metadata.manifest['tensor_schema'])==2310
assert set(prepared.programs)==set(prepared.inputs)==set(candidate.PROGRAMS)
for name,rows in (('prefill_chunk',128),('prefill_tail',114)):
    assert prepared.inputs[name][0].shape==(rows,)
    assert prepared.programs[name].canonical_dense
    assert all(isinstance(x,jax.ShapeDtypeStruct) for x in jax.tree.leaves(prepared.inputs[name]))
    assert len(prepared.inputs[name][3].layers)==78
assert candidate.program_options()['canonical_dense'] is True
print('CANONICAL_FULL_METADATA_PASS',flush=True)
"""
    assert "CANONICAL_FULL_METADATA_PASS" in helpers["_cpu"](source, timeout=120)


def test_preserved_pair_file_integrity_memory_and_no_admission(tmp_path, monkeypatch):
    record = {"programs": {}}
    raw = {}
    acquired = admission.short_acquisition(ROOT)["fleet"][0]["compiled"]
    for name in candidate.PROGRAMS:
        stable, optimized = f"raw {name}".encode(), f"actual {name}".encode()
        (tmp_path / f"{name}.stablehlo.mlir").write_bytes(stable)
        (tmp_path / f"{name}.optimized_hlo.txt").write_bytes(optimized)
        raw[name] = (len(stable), sha256(stable).hexdigest())
        record["programs"][name] = dict(
            stablehlo_sha256=sha256(stable).hexdigest(),
            optimized_hlo_sha256=sha256(optimized).hexdigest(),
            compiled_memory=acquired[name]["memory"],
        )
    monkeypatch.setattr(candidate, "RAW", raw)
    result = candidate.validate_preserved_pair(tmp_path, record, repo=ROOT)
    assert not result["numerical_claim"] and not result["performance_claim"]
    for name in candidate.PROGRAMS:
        bad = deepcopy(record)
        del bad["programs"][name]
        with pytest.raises(ValueError, match="both"):
            candidate.validate_preserved_pair(tmp_path, bad, repo=ROOT)
        bad = deepcopy(record)
        bad["programs"][name]["compiled_memory"]["temp_size_in_bytes"] = 1 << 31
        with pytest.raises(ValueError, match="allocation"):
            candidate.validate_preserved_pair(tmp_path, bad, repo=ROOT)
        path = tmp_path / f"{name}.optimized_hlo.txt"
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b"changed")
        with pytest.raises(ValueError, match="identity"):
            candidate.validate_preserved_pair(tmp_path, record, repo=ROOT)
        path.write_bytes(original_bytes)
    monkeypatch.setattr(candidate, "RAW", {})
    with pytest.raises(ValueError, match="both"):
        candidate.validate_preserved_pair(tmp_path, record, repo=ROOT)
