"""Retained real originals and production abstract preparation; CPU only."""

import ast
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.greenfield import prefill_rolled_window as rolled
from scripts.greenfield.prefill_layer_evidence import encode_arrays
from scripts.greenfield.prefill_window_worker import save_arrays

ROOT = Path("/home/gianl/glm-run") / (
    "greenfield_fp8_ws32_prefill_expert_panel_phase_l6_20260909T033031628458338Z"
)


@pytest.fixture(scope="module")
def references():
    # Missing protected originals are an admission failure, never a skipped test.
    return [rolled.load_reference(ROOT, rank=rank) for rank in range(8)]


def test_all32_original_control_assemblies_replay(references):
    slots = []
    for ref in references:
        slots.extend(ref.slots.values())
        assert len(ref.sources) == 2
        result = rolled.compare_observations(ref.controls, ref)
        assert result["passed"] and result["source_db"] == 600
        assert not result["independent_canonical_dsa_claim"]
        assert not result["performance_claim"]
    assert sorted(slots) == list(range(32))


@pytest.mark.parametrize("field", ["positions", "routes", "kv", "output", "health"])
def test_original_candidate_defects_refuse(references, field):
    ref = references[0]
    device = next(iter(ref.slots))
    actual = {d: dict(v) for d, v in ref.controls.items()}
    value = actual[device][field].copy()
    if field == "positions":
        value[0, 0] = -1
    elif field == "routes":
        value[0, :2] = value[0, :2][::-1]
    elif field == "kv":
        value[:] = 0  # Includes untouched historical rows.
    elif field == "output":
        value[0, 0] = np.nan
    else:
        value[0] = False
    actual[device][field] = value
    with pytest.raises(ValueError):
        rolled.compare_observations(actual, ref)


def test_candidate_capsule_replay_and_input_owner_refusals(tmp_path, references):
    ref = references[0]
    arrays = encode_arrays("input", ref.inputs)
    for device, fields in ref.controls.items():
        arrays.update(encode_arrays(f"actual_{device}", fields))
    path = tmp_path / "candidate.npz"
    save_arrays(path, arrays)
    assert rolled.replay_candidate(path, ref)["passed"]
    wrong = dict(ref.inputs)
    wrong["update"] = wrong["update"].copy()
    wrong["update"][0, 0] += np.asarray(1, dtype=wrong["update"].dtype)
    with pytest.raises(ValueError, match="fixture"):
        rolled.replay_candidate(path, replace(ref, inputs=wrong))
    with pytest.raises(ValueError, match="owners"):
        rolled.compare_observations({}, ref)
    arrays["unknown"] = np.asarray(0)
    save_arrays(path, arrays)
    with pytest.raises(ValueError, match="inventory"):
        rolled.replay_candidate(path, ref)


def test_bound_source_mutation_size_and_rank_refuse(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    digest = sha256(b"original").hexdigest()
    assert rolled._read_bound(path, digest, 8) == b"original"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="bytes"):
        rolled._read_bound(path, digest, 8)
    with pytest.raises(ValueError, match="size"):
        rolled._read_bound(path, digest, 9)
    for rank in (-1, 8, True):
        with pytest.raises(ValueError, match="rank"):
            rolled.load_reference(ROOT, rank=rank)


def test_actual_production_rolled_programs_lower_without_payloads():
    source = Path("tests/greenfield/hlo/test_prefill_completed_window.py").read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_completed_window_production_shapes_without_payloads"
    )
    fixture = ast.literal_eval(fn.body[0].value).split("programs=prepare_programs", 1)[
        0
    ]
    code = (
        fixture
        + r"""
from hashlib import sha256
from unittest.mock import patch
from jax._src import tpu_custom_call
from scripts.greenfield.prefill_rolled_window import prepare_programs,PROGRAMS
from scripts.greenfield.prefill_paired_sort_admission import registered_programs
from scripts.greenfield.prefill_layer_programs import build_layer_programs
programs=prepare_programs(mesh=mesh,config=config,weights=w)
assert tuple(n for n,_,_ in programs)==PROGRAMS
proof={}
with patch.object(tpu_custom_call,'get_ir_version',return_value=None):
 for name,fn,args in programs:
  assert all(isinstance(a,jax.ShapeDtypeStruct) for a in jax.tree.leaves(args))
  out=jax.eval_shape(fn,*args)
  raw=str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo'))
  digest=sha256(raw.encode()).hexdigest()
  if name!='candidate':assert digest==registered_programs()[name]['stablehlo_sha256']
  else:
   assert len(out)==12 and out[0].shape==(128,6144) and out[10].shape==(8,4,128)
   assert out[2].shape==(8,8,64,640) and out[3].shape==(8,8,64,128)
   assert 'stablehlo.while' in raw
   assert raw.count('name = "greenfield_prefill_expert_panel_raw_fp8"')==3
  proof[name]={'sha256':digest,'bytes':len(raw.encode())}
specs=input_specs(w,programs[-1][2][14])
for bad in ({'rolled_prefix':True}, {'expert_panels':True},
            {'sorted_local_merge':1},
            {'candidate_window':True,'rolled_prefix':True,'capture_boundaries':True}):
 try:build_layer_programs(mesh,specs,full_indexer=True,sparse_mlp=True,**bad)
 except ValueError:pass
 else:raise AssertionError('bad candidate options accepted')
print('ROLLED_PRODUCTION_RAW='+json.dumps(proof,sort_keys=True),flush=True)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
