"""Actual CPU32 allocation/placement for the protected diagnostic adapter."""

import os
import subprocess
import sys


def test_independent_owner_caches_and_physical_b128_host_inputs():
    code=r'''
import jax
import numpy as np
from jax.sharding import Mesh
from scripts.greenfield.ws32_dense_frontier_worker import fresh_caches,independent_caches,inputs
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
a,b=fresh_caches(mesh),fresh_caches(mesh)
jax.block_until_ready((a,b))
independent_caches(a,b)
for branch in (a,b):
    for layer in branch:
        for array in layer:
            assert str(array.dtype)=='bfloat16'
            for shard in array.addressable_shards:
                assert np.asarray(shard.data).view(np.uint16).sum()==0
try:
    independent_caches(a,a);raise AssertionError('cache alias accepted')
except ValueError:pass
ids=np.arange(32,dtype=np.int32)+40
v=inputs(mesh,ids,64,b,None,None,None,None)
jax.block_until_ready(v[:4])
assert v[0].shape==(128,)
assert np.array_equal(np.asarray(v[0])[:32],ids)
assert np.all(np.asarray(v[0])[32:]==0)
assert int(v[1])==32 and int(v[2])==64
assert np.array_equal(np.asarray(v[3]),np.arange(16,dtype=np.int32)[None])
assert v[4] is b
print('DENSE_INPUTS_CPU32_PASS')
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=90,
                          env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'))
    assert result.returncode==0,result.stdout+result.stderr
    assert 'DENSE_INPUTS_CPU32_PASS' in result.stdout
