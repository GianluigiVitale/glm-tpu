"""Production metadata and abstract shapes, explicitly not TPU compilation."""

import os
import subprocess
import sys


def test_selected_production_frontier_without_payload_or_placement():
    source = r'''
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_history_frontier_prepare import prepare
assert jax.default_backend()=='cpu'
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
opened=Path.open
def checked(path,*args,**kwargs):
    assert path.suffix not in ('.safetensors','.bin'),path
    return opened(path,*args,**kwargs)
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
with patch.object(Path,'open',checked),patch('jax.device_put',side_effect=AssertionError('concrete placement')):
    p=prepare(mesh,repo=Path.cwd())
    assert set(p.programs)==set(p.inputs)=={'canonical_b128','canonical_b114','live32_b128','live32_b114'}
    assert all(isinstance(x,jax.ShapeDtypeStruct) for x in jax.tree.leaves(p.inputs))
    assert len(p.tensor_names)==201
    assert p.payload_bytes_per_chip==1424692176
    assert p.cache_bytes_per_chip_per_branch==11272192
    assert p.wk_bytes_per_chip==12582912
    assert p.config.geometry.num_layers==78
    assert not p.config.exact_dsa and not p.config.strategy_nd_dense
    for name,fn in p.programs.items():
        out=jax.eval_shape(fn,*p.inputs[name])
        rows=p.inputs[name][0].shape[0]
        assert tuple(map(len,out.caches))==(7,4,4)
        assert len(out.boundaries)==7 and len(out.producers)==4
        for boundary in out.boundaries:
            assert boundary.update.shape==boundary.residual.shape==boundary.normalized_input.shape==(rows,6144)
            assert boundary.health.shape==(8,4,rows)
        for event in out.producers:
            assert event.positions.shape==event.scores.shape==(rows,2048)
            assert event.counts.shape==(rows,)
        assert out.healthy.shape==()
        print('HISTORY_ABSTRACT',name,flush=True)
'''
    result = subprocess.run([sys.executable, "-c", source], text=True,
        capture_output=True, timeout=300,
        env=dict(os.environ, JAX_PLATFORMS="cpu",
                 XLA_FLAGS="--xla_force_host_platform_device_count=32"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("HISTORY_ABSTRACT") == 4
