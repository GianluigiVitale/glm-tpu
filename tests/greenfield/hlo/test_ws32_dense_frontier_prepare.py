"""Current corrective source preparation/lowering, not historical RAW identity."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_production_abstract_dense01_without_payloads():
    checkpoint = Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z')
    if not (checkpoint / 'manifest.json').exists():
        pytest.skip('requires retained production metadata')
    source = r'''
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_dense_frontier_prepare import prepare,compiler_programs
from scripts.greenfield import ws32_dense_canonical as candidate
from scripts.greenfield.ws32_dense_frontier_program import build_program
from scripts.greenfield.ws32_dense_frontier_admission import RAW
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
opened = Path.open
def checked(path, *args, **kwargs):
    assert path.suffix not in ('.safetensors', '.bin'), path
    return opened(path, *args, **kwargs)
with patch.object(Path, 'open', checked), patch('jax.device_put', side_effect=AssertionError('concrete allocation')):
    p = candidate.prepare(mesh, repo=Path.cwd())
assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(p.inputs))
assert len(p.tensor_names) == 55, len(p.tensor_names)  # 27 per dense/full-index layer + embedding
assert p.config.geometry.num_layers == 78
assert not p.config.exact_dsa and not p.config.strategy_nd_dense
assert p.inputs[0].shape == (128,)
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
with patch('jax._src.tpu_custom_call.get_ir_version', return_value=None):
    candidate_ir = p.program.trace(*p.inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')
    raw = str(candidate_ir).encode()
assert raw
assert (len(raw),sha256(raw).hexdigest()) == candidate.RAW[candidate.GRAPH]
assert 'greenfield_ws32_prefill_dense_canonical' in candidate_ir.operation.get_asm(enable_debug_info=True)
candidate_raw=raw
with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
    default=build_program(mesh,p.config)
    explicit=build_program(mesh,p.config,canonical_dense=False)
    default_ir=default.trace(*p.inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')
    default_raw=str(default_ir).encode()
    explicit_raw=str(explicit.trace(*p.inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
assert default_raw == explicit_raw
assert 'greenfield_ws32_prefill_dense_canonical' not in default_ir.operation.get_asm(enable_debug_info=True)
assert candidate_raw != default_raw
# Historical RAW registrations remain unchanged. Current debug locations have
# moved; this is default selection, not reproduction of the acquired graph.
print('DENSE_PRODUCTION_ABSTRACT', len(p.tensor_names), p.payload_bytes_per_chip,
      p.manifest_sha256, len(raw), sha256(raw).hexdigest(), flush=True)
jobs=candidate.compiler_programs(p,mesh)
assert tuple(n for n,_,_ in jobs)==('wk_decode','wk_promote','dense01_canonical')
for name,fn,values in jobs[:2]:
    assert all(isinstance(x,jax.ShapeDtypeStruct) for x in jax.tree.leaves(values))
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(fn.trace(*values).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    assert (len(raw),sha256(raw).hexdigest()) == RAW[name]
    print('DENSE_WK_ABSTRACT',name,len(raw),sha256(raw).hexdigest(),flush=True)
'''
    result = subprocess.run([sys.executable, '-c', source], text=True, capture_output=True,
                            timeout=180, env=dict(os.environ, JAX_PLATFORMS='cpu',
                            XLA_FLAGS='--xla_force_host_platform_device_count=32'))
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
    assert 'DENSE_PRODUCTION_ABSTRACT' in result.stdout
