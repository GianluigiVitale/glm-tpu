"""Use the existing production abstract fixture, no checkpoint or TPU load."""

import ast
import os
from pathlib import Path
import subprocess
import sys


def test_actual_completed_suffix_opt_in_and_default_raw_identity():
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
from scripts.greenfield.prefill_paired_sort_admission import registered_programs
original=registered_programs()
proof={}
with patch.object(tpu_custom_call,'get_ir_version',return_value=None):
 for flag in (False,True):
  programs=prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True,
                           paired_position_sort=True,expert_panels=flag)
  for name,fn,args in programs:
   raw=str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo'))
   digest=sha256(raw.encode()).hexdigest()
   if not flag or name != 'candidate':
    assert digest==original[name]['stablehlo_sha256'],(flag,name,digest)
   else:
    assert digest!=original[name]['stablehlo_sha256']
    assert raw.count('name = "greenfield_prefill_expert_panel_raw_fp8"')==3,raw.count('greenfield_prefill_expert_panel_raw_fp8')
    shape=jax.eval_shape(fn,*args)
    rows=128 if name=='candidate' else 32
    assert shape[0].shape==(rows,6144) and shape[1].shape==(rows,8)
    assert shape[3].shape==(8,4,rows)
    proof[name]={'stablehlo_sha256':digest,'bytes':len(raw.encode())}
for flag in (1,'yes'):
 try:prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True,expert_panels=flag)
 except ValueError:pass
 else:raise AssertionError('nonbool opt-in accepted')
try:prepare_programs(mesh=mesh,config=config,weights=w,expert_panels=True)
except ValueError:pass
else:raise AssertionError('panel in noncompleted path accepted')
print('PANEL_PRODUCTION_RAW='+json.dumps(proof,sort_keys=True))
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
    assert "PANEL_PRODUCTION_RAW=" in result.stdout
    print(result.stdout.strip())
