"""End-to-end P2 bitwise proof with independent frozen/challenger programs."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("lse_attention", [False, True])
def test_prefill_p2_program_matches_frozen_and_preserves_module_bindings(lse_attention):
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.kernels import ws32_prefill_dsa as ds, ws32_prefill_layer as layer, ws32_prefill_window as window
from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
import glm_tpu.perf.prefill_challenger as perf
seen=[]
one_pass=perf.prefill_dsa_one_pass_mapped
def observed_selector(*args,**kwargs):
    seen.append(True)
    return one_pass(*args,**kwargs)
perf.prefill_dsa_one_pass_mapped=observed_selector
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(v): return jax.device_put(v,NamedSharding(mesh,P()))
config,weights,wk=fixture(mesh)
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
originals=[(m,n,getattr(m,n)) for m,n in [(ds,'ws32_prefill_dsa_from_query_mapped'),(layer,'ws32_prefill_dsa_mapped'),(window,'ws32_prefill_transformer_layer_mapped'),(b,'ws32_prefill_layer_window_mapped'),(b,'ws32_batched_prefill_mapped')]]
# Exercise the rolled window, including the final one-live-row padded block.
opts=dict(block_rows=2,key_tile=128,mlp_window=True,rolled_prefix=True,paired_position_sort=True,sorted_local_merge=True,sparse_attention_interpret=True,linear_interpret=True)
frozen=b.build_ws32_batched_prefill_program(mesh,config,**opts)
challenger=build_ws32_prefill_challenger_program(mesh,config,lse_attention=LSE_ATTENTION,**opts)
assert all(getattr(m,n) is v for m,n,v in originals)
a=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
bstate=a
for tokens,count in [([30,31],2),([32,-1],1)]:
    x=frozen.execute(put(jnp.array(tokens,jnp.int32)),put(jnp.int32(count)),a,weights,wk,rope)
    y=challenger.execute(put(jnp.array(tokens,jnp.int32)),put(jnp.int32(count)),bstate,weights,wk,rope)
    for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)): np.testing.assert_array_equal(np.asarray(xx),np.asarray(yy))
    assert bool(np.asarray(y.state.decoder.contract_valid).all())
    a,bstate=x.state,y.state
assert bool(np.asarray(bstate.finished))
assert len(seen) >= len(config.full_index_slots), 'challenger failed to use P2 selector'
# Already-finished state must refuse another append atomically in both programs.
tokens,count=put(jnp.array([33,-1],jnp.int32)),put(jnp.int32(1))
x=frozen.execute(tokens,count,a,weights,wk,rope)
y=challenger.execute(tokens,count,bstate,weights,wk,rope)
for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)): np.testing.assert_array_equal(np.asarray(xx),np.asarray(yy))
assert not bool(np.asarray(y.state.decoder.contract_valid).all())
assert all(getattr(m,n) is v for m,n,v in originals)
print('P2 complete prefill state/token/cache bitwise equal; atomic refusal and frozen globals intact')
'''
    code = code.replace("LSE_ATTENTION", repr(lse_attention))
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
