"""All feature groups must preserve their exact routed-row reduction."""
import os
import subprocess
import sys


def test_owner_compact_feature_sum_cpu32_bitwise():
    code=r'''
import jax,jax.numpy as jnp,numpy as np
from jax import lax
from jax.sharding import Mesh,PartitionSpec as P
from glm_tpu.perf.compact_feature_reduce import owner_compact_feature_sum
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rows,width=512,128
rng=np.random.default_rng(178)
def body(values,starts,counts):
    values=values[0,0];start,count=starts[0],counts[0]
    actual=owner_compact_feature_sum(values,start,count,capacity=128)
    expected=lax.psum(values,'feature').astype(jnp.bfloat16)
    return actual[None],expected[None]
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P('expert','feature'),P('expert'),P('expert')),
    out_specs=(P('expert'),P('expert')),check_vma=False))
for counts in ([64]*8,[512,0,0,0,0,0,0,0],[0,3,0,128,0,129,0,252],[0,0,0,0,0,0,512,0]):
    counts=np.array(counts,np.int32);starts=np.cumsum(counts,dtype=np.int32)-counts
    values=rng.normal(size=(8,4,2,rows,width)).astype(np.float32)
    # Original panel unpacker supplies exact positive-zero unowned rows.
    mask=(np.arange(rows)[None]>=starts[:,None])&(np.arange(rows)[None]<starts[:,None]+counts[:,None])
    values=np.where(mask[:,None,None,:,None],values,np.float32(0))
    actual,expected=fn(jnp.asarray(values),jnp.asarray(starts),jnp.asarray(counts))
    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8),np.ascontiguousarray(expected).view(np.uint8))
# Invalid span metadata must select the full-buffer path without clipping data.
values=jnp.asarray(rng.normal(size=(8,4,2,rows,width)),jnp.float32)
starts=jnp.asarray([-1,513,0,0,2147483647,0,1,0],jnp.int32)
counts=jnp.asarray([0,0,-1,513,1,2147483647,512,512],jnp.int32)
actual,expected=fn(values,starts,counts)
np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8),np.ascontiguousarray(expected).view(np.uint8))
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
