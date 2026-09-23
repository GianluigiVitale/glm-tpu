"""End-to-end proof: the production prefill program against the independent frozen FP8 program."""
import os
import subprocess
import sys

import pytest


def test_prefill_p2_program_matches_frozen_and_preserves_module_bindings():
    lse_attention, bf16_resident, canonical, pending = True, True, True, False
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.kernels import ws32_prefill_dsa as ds, ws32_prefill_layer as layer, ws32_prefill_window as window
from glm_tpu.optimized.prefill import build_prefill_program
import glm_tpu.optimized.prefill_dsa as perf  # the production DSA selects through its one-pass selector
seen=[]
one_pass=perf.prefill_dsa_one_pass_mapped
def observed_selector(*args,**kwargs):
    seen.append(True)
    return one_pass(*args,**kwargs)
perf.prefill_dsa_one_pass_mapped=observed_selector
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(v): return jax.device_put(v,NamedSharding(mesh,P()))
config,weights,wk=fixture(mesh,panel_geometry=CANONICAL)
from glm_tpu.optimized.bf16_resident import bf16_resident_weights
resident = bf16_resident_weights(mesh,config,weights) if BF16_RESIDENT else weights
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
originals=[(m,n,getattr(m,n)) for m,n in [(ds,'ws32_prefill_dsa_from_query_mapped'),(layer,'ws32_prefill_dsa_mapped'),(window,'ws32_prefill_transformer_layer_mapped'),(b,'ws32_prefill_layer_window_mapped'),(b,'ws32_batched_prefill_mapped')]]
# Exercise the rolled window, including the final one-live-row padded block.
opts=dict(block_rows=2,key_tile=128,mlp_window=True,rolled_prefix=True,paired_position_sort=True,sorted_local_merge=True,sparse_attention_interpret=True,linear_interpret=True)
if PENDING: opts.update(pending_cache_rows=True,flat_pending_rows=True,capture_barrier=True)
if CANONICAL: opts.update(block_rows=128,canonical_dense=True,expert_panels=True)
frozen=b.build_ws32_batched_prefill_program(mesh,config,**opts)
# the production program hard-wires the admitted profile (window, rolled prefix, panels, canonical
# dense, one-pass selector); only the block size and the interpret flags are arguments
challenger=build_prefill_program(mesh,config,block_rows=opts['block_rows'],sparse_attention_interpret=True,linear_interpret=True)
assert all(getattr(m,n) is v for m,n,v in originals)
a=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
bstate=a
blocks=[([30,31,32]+[-1]*125,3)] if CANONICAL else [([30,31],2),([32,-1],1)]
for tokens,count in blocks:
    x=frozen.execute(put(jnp.array(tokens,jnp.int32)),put(jnp.int32(count)),a,weights,wk,rope)
    y=challenger.execute(put(jnp.array(tokens,jnp.int32)),put(jnp.int32(count)),bstate,resident,wk,rope)
    for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)):
        aa,bb=np.asarray(xx),np.asarray(yy)
        if BF16_RESIDENT and (jnp.issubdtype(xx.dtype,jnp.floating)):
            np.testing.assert_allclose(aa.astype(np.float32),bb.astype(np.float32),rtol=.02,atol=.0625)
        else: np.testing.assert_array_equal(aa,bb)
    assert bool(np.asarray(y.state.decoder.contract_valid).all())
    a,bstate=x.state,y.state
assert bool(np.asarray(bstate.finished))
assert len(seen) >= len(config.full_index_slots), 'challenger failed to use P2 selector'
# Already-finished state must refuse another append atomically in both programs.
committed=bstate
tokens,count=put(jnp.array([33]+[-1]*(opts["block_rows"]-1),jnp.int32)),put(jnp.int32(1))
x=frozen.execute(tokens,count,a,weights,wk,rope)
y=challenger.execute(tokens,count,bstate,resident,wk,rope)
for field in ('kv_cache_local','index_cache_local','position','context_lengths'):
    np.testing.assert_array_equal(np.asarray(getattr(committed.decoder,field)),np.asarray(getattr(y.state.decoder,field)))
np.testing.assert_array_equal(np.asarray(x.state.decoder.contract_valid),np.asarray(y.state.decoder.contract_valid))
assert not bool(np.asarray(y.state.decoder.contract_valid).all())
assert all(getattr(m,n) is v for m,n,v in originals)
print('P2 complete prefill: bitwise raw path / bounded BF16 path; same token, atomic refusal, frozen globals intact')
# Long-context ownership uses the SAME body with exclusive state donation.
# Check actual state/token outputs against the non-donating implementation.
owned=jax.jit(challenger.execute,donate_argnums=(2,))
owned_state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
plain_state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
for token_list,live in blocks:
    token_array,count_array=put(jnp.array(token_list,jnp.int32)),put(jnp.int32(live))
    plain=challenger.execute(token_array,count_array,plain_state,resident,wk,rope)
    donated=owned(token_array,count_array,owned_state,resident,wk,rope)
    jax.block_until_ready((plain,donated))
    for expected,actual in zip(jax.tree.leaves(plain),jax.tree.leaves(donated)):
        np.testing.assert_array_equal(np.asarray(expected),np.asarray(actual))
    plain_state,owned_state=plain.state,donated.state
print('Exclusive prefill state donation preserves every output leaf')
'''
    code = code.replace("LSE_ATTENTION", repr(lse_attention)).replace("BF16_RESIDENT",repr(bf16_resident)).replace("CANONICAL",repr(canonical)).replace("PENDING",repr(pending))
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
