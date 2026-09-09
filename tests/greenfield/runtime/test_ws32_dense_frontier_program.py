"""CPU mechanism and original-runtime relation, not DB604 TPU reproduction."""

import os
import subprocess
import sys


def test_two_dense_layers_same_physical128_cache_carry_cpu32():
    code = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_dense_frontier_program import build_program
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(v,s=P()):return jax.device_put(v,NamedSharding(mesh,s))
wk=tuple(put(w) for w in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
cache_spec=P('expert',None,None,None)
caches=tuple(tuple(put(jnp.zeros((8,config.page_count,64,width),jnp.bfloat16),cache_spec)
                   for width in (640,128,128)) for _ in range(2))
table=put(jnp.arange(config.page_count,dtype=jnp.int32)[None])
tokens=put(jnp.arange(128,dtype=jnp.int32)+30)
fn=build_program(mesh,config,interpret=True)
args=(tokens,put(jnp.int32(128)),put(jnp.int32(0)),table,caches,
      weights.embedding_local,weights.layers[:2],wk[:2],rope)
compiled=fn.lower(*args).compile()
actual=compiled(*args);jax.block_until_ready(actual)
assert all(np.asarray(v[10]).all() for v in actual)
print('DENSE_FRONTIER_WIDE_EXECUTED',flush=True)
def bits_equal(a,b):
    x,y=np.asarray(a),np.asarray(b)
    assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
opts=dict(key_tile=512,sparse_attention_interpret=True,linear_interpret=True,
          mlp_window=True,rolled_prefix=True,expert_panels=True,
          paired_position_sort=True,sorted_local_merge=True)
full=b.build_ws32_batched_prefill_program(mesh,config,block_rows=128,**opts)
state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=256)
reference=full.execute(tokens,put(jnp.int32(128)),state,weights,wk,rope)
jax.block_until_ready(reference)
assert not bool(reference.state.finished)
for layer in range(2):
    for k,v in enumerate((reference.state.decoder.kv_cache_local,
                          reference.state.decoder.index_cache_local,
                          reference.state.repaired_index_local)):
        # Original shape[pages,512,width] versus explicit owner[8,pages,64,width].
        original=np.asarray(v[layer]).reshape(config.page_count,8,64,-1).transpose(1,0,2,3)
        bits_equal(actual[layer][k+2],original)
print('DENSE_FRONTIER_CPU_ORIGINAL_FIRST2_CACHE_MATCH',flush=True)
current=caches
for offset in (0,32,64,96):
    ids=put(jnp.concatenate((tokens[offset:offset+32],jnp.zeros(96,jnp.int32))))
    args=(ids,put(jnp.int32(32)),put(jnp.int32(offset)),table,current,
          weights.embedding_local,weights.layers[:2],wk[:2],rope)
    result=compiled(*args);jax.block_until_ready(result)
    assert all(np.asarray(v[10]).all() for v in result)
    if offset==32:
        poison=compiled(ids.at[32:].set(-2147483648),*args[1:])
        for v,p in zip(jax.tree.leaves(result),jax.tree.leaves(poison),strict=True):bits_equal(v,p)
    current=tuple(tuple(v[2:5]) for v in result)
for layer in range(2):
    for k in range(3):bits_equal(actual[layer][k+2],current[layer][k])
bad=compiled(tokens,put(jnp.int32(128)),put(jnp.int32(32)),table,caches,
             weights.embedding_local,weights.layers[:2],wk[:2],rope)
assert all(not np.asarray(v[10]).any() for v in bad)
try:
    fn(tokens[:32],*args[1:]);raise AssertionError('accepted physicalB32')
except ValueError:pass
print('DENSE_FRONTIER_CARRY_PADDING_SPAN_PASS',flush=True)
'''
    result = subprocess.run([sys.executable, "-c", code], text=True,
        capture_output=True, timeout=420,
        env=dict(os.environ, JAX_PLATFORMS="cpu",
                 XLA_FLAGS="--xla_force_host_platform_device_count=32"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DENSE_FRONTIER_CARRY_PADDING_SPAN_PASS" in result.stdout
