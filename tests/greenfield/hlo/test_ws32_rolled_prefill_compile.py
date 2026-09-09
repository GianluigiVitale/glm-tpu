"""Metadata-only compiler preparation; no TPU or numerical execution."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import runpy

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_rolled_prefill_compile as candidate

ROOT = Path(__file__).resolve().parents[3]


def test_real_metadata_only_abstract_inputs_reproduce_registered_graphs():
    # This is a local retained-metadata integration check, not a payload fixture
    # or evidence that another machine has those metadata files installed.
    checkpoint = Path(
        "/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z"
    )
    if not (checkpoint / "manifest.json").exists():
        pytest.skip("requires retained WS32 manifest/SUCCESS, not checkpoint payloads")
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_paired_prefill.py")
    )
    source = r"""
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_rolled_prefill_compile as candidate
from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
root=Path.cwd()
opened=[]
original=Path.open
def checked(path,*args,**kwargs):
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json','SUCCESS'),path
    assert path.suffix not in ('.safetensors','.bin'),path
    opened.append(str(path))
    return original(path,*args,**kwargs)
with patch.object(Path,'open',checked):
    metadata=candidate.read_metadata(root)
assert any(p.endswith('/manifest.json') for p in opened)
assert len(metadata.manifest['tensor_schema'])==2310
with patch('jax.device_put',side_effect=AssertionError('allocated concrete inputs')):
    prepared=candidate.prepare(mesh,metadata,repo=root)
assert set(prepared.programs)==set(prepared.inputs)=={'prefill_chunk','prefill_tail'}
assert all(isinstance(x,jax.ShapeDtypeStruct) for v in prepared.inputs.values() for x in jax.tree.leaves(v))
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
registered=admission.rolled_registration(root)
for name in prepared.programs:
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(prepared.programs[name].execute.trace(*prepared.inputs[name]).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    pins=registered['graphs'][name]
    assert len(raw)==pins['stablehlo_bytes'],(name,len(raw))
    assert sha256(raw).hexdigest()==pins['stablehlo_sha256'],(name,sha256(raw).hexdigest())
print('METADATA_ONLY_BOTH_RAW_GRAPHS_PASS',flush=True)
"""
    assert "METADATA_ONLY_BOTH_RAW_GRAPHS_PASS" in helpers["_cpu"](source, timeout=300)


def test_preserved_pair_requires_both_actual_files_and_allocations(
    tmp_path, monkeypatch
):
    record = {"programs": {}}
    registration = {"graphs": {}}
    original = admission.short_acquisition(ROOT)["fleet"][0]["compiled"]
    for name in ("prefill_chunk", "prefill_tail"):
        stable, optimized = f"raw {name}", f"actual {name}"
        (tmp_path / f"{name}.stablehlo.mlir").write_text(stable)
        (tmp_path / f"{name}.optimized_hlo.txt").write_text(optimized)
        registration["graphs"][name] = dict(
            stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            stablehlo_bytes=len(stable),
        )
        record["programs"][name] = dict(
            stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
            compiled_memory=original[name]["memory"],
        )
    read_registration = admission.rolled_registration
    # Fake raw texts isolate the file/capture boundary; actual memory caps still
    # come from the real bound registration, not an arbitrary test allowance.
    registration["memory"] = read_registration(ROOT)["memory"]
    monkeypatch.setattr(admission, "rolled_registration", lambda repo: registration)
    result = candidate.validate_preserved_pair(tmp_path, record, repo=ROOT)
    assert not result["numerical_claim"] and not result["performance_claim"]
    assert "model_calls" not in result
    for name in registration["graphs"]:
        partial = deepcopy(record)
        del partial["programs"][name]
        with pytest.raises(ValueError, match="both graphs"):
            candidate.validate_preserved_pair(tmp_path, partial, repo=ROOT)
        bad = deepcopy(record)
        bad["programs"][name]["compiled_memory"]["temp_size_in_bytes"] = 1 << 31
        with pytest.raises(ValueError, match="allocation"):
            candidate.validate_preserved_pair(tmp_path, bad, repo=ROOT)
        path = tmp_path / f"{name}.optimized_hlo.txt"
        original_text = path.read_text()
        path.write_text(original_text + "changed")
        with pytest.raises(ValueError, match="identity"):
            candidate.validate_preserved_pair(tmp_path, record, repo=ROOT)
        path.write_text(original_text)
