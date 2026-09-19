"""P3 wide IndexShare prefix proofs; full DSA/repair and dense stay narrow."""
import ast
import inspect
import os
import subprocess
import sys
import pytest


def test_wide_mirrors_change_only_explicit_row_guards():
    from glm_tpu.greenfield.kernels import ws32_prefill_layer as layer, prefill_cache as cache, ws32_prefill_attention as attention
    from glm_tpu.perf import wide_prefill_primitives as wide
    for module,name in ((layer,'ws32_prefill_transformer_layer_mapped'),(cache,'write_prefill_cache_block'),(attention,'_require_block')):
        expected=inspect.getsource(getattr(module,name)).replace('shape[0] <= 32','shape[0] <= 128')
        expected=expected.replace('requires1..32 BF16','requires1..128 BF16')
        if module is layer:
            expected=expected.replace('    rows = hidden_update_local.shape[0]\n',
                "    rows = hidden_update_local.shape[0]\n    if rows > 32 and (not prefix_only or dsa_weights is not None or dense_weights is not None):\n        raise ValueError('wide rows require a sparse IndexShare prefix')\n")
        assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(getattr(wide,name))))


@pytest.mark.parametrize('options',[dict(wide_indexshare=1),dict(wide_indexshare=True),
    dict(wide_indexshare=True,bf16_resident=True,lse_attention=True,owned_key_capacity=128,
         mlp_window=True,rolled_prefix=True,expert_panels=True,canonical_dense=True,pending_cache_rows=True)])
def test_wide_options_refuse_unsupported_compositions(options):
    from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
    with pytest.raises(ValueError,match='wide IndexShare'):
        build_ws32_prefill_challenger_program(None,None,**options)


def test_wide_indexshare_complete_cpu32():
    code = r'''

import jax,jax.numpy as jnp,numpy as np
from dataclasses import replace
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b,ws32_decoder as d
from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program as build
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(v):return jax.device_put(v,NamedSharding(mesh,P()))
config,raw,wk=fixture(mesh,panel_geometry=True)
config=replace(config,geometry=replace(config.geometry,dsa_top_k=512))
weights=bf16_resident_weights(mesh,config,raw)
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
opts=dict(key_tile=128,mlp_window=True,rolled_prefix=True,expert_panels=True,canonical_dense=True,
    paired_position_sort=True,sorted_local_merge=True,sparse_attention_interpret=True,linear_interpret=True,
    lse_attention=True,bf16_resident=True)
small={n:build(mesh,config,block_rows=n,**opts) for n in (114,128)}
# Exercise a B128 chunk and B114 final tail. Compare every cache,
# selected position, route-derived activation effect, token and frontier.
for rows in (242,):
    large=build(mesh,config,block_rows=rows,pooled_moe=True,
                owned_key_capacity=128,wide_indexshare=True,**opts)
    tokens=put(jnp.asarray(np.arange(rows,dtype=np.int32)%256))
    initial=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=rows)
    expected=initial
    for first in range(0,rows,128):
        n=min(128,rows-first)
        ref=small[n].execute(tokens[first:first+n],put(jnp.int32(n)),expected,weights,wk,rope)
        expected=ref.state
    actual=large.execute(tokens,put(jnp.int32(rows)),initial,weights,wk,rope)
    assert bool(np.asarray(actual.state.finished)) and bool(np.asarray(actual.state.decoder.contract_valid).all())
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(ref)):
        np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(e).view(np.uint8))
    # Rejection must preserve the entire committed state except the health latch.
    rejected=large.execute(tokens,put(jnp.int32(rows)),actual.state,weights,wk,rope)
    assert not bool(np.asarray(rejected.state.decoder.contract_valid).all())
    preserved=rejected.state._replace(decoder=rejected.state.decoder._replace(contract_valid=actual.state.decoder.contract_valid))
    for a,e in zip(jax.tree.leaves(preserved),jax.tree.leaves(actual.state)):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(e))
    assert int(np.asarray(rejected.next_token)[0])==-1
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=900)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('offset,count',[(60,128),(504,114),(1020,128),(1536,0)])
def test_wide_cache_stripes_pages_causality_and_atomic_refusal(offset,count):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from glm_tpu.greenfield.kernels.reference.attention import StageLocalKvLayout
    from glm_tpu.perf.wide_prefill_primitives import write_prefill_cache_block
    layout=StageLocalKvLayout(local_parallel_size=8,packed_cache_width=128)
    table=np.asarray([[2,0,1]],np.int32)
    cache=jnp.full((3,64,128),-1,jnp.bfloat16)
    rows=jnp.arange(128*128,dtype=jnp.float32).reshape(128,128).astype(jnp.bfloat16)
    rows=rows.at[count:].set(jnp.nan)
    call=jax.jit(lambda rows,table,start,count,owner:write_prefill_cache_block(
        cache,rows,table,start,count,owner,layout=layout))
    for owner in range(8):
        out=call(rows,jnp.asarray(table),jnp.int32(offset),jnp.int32(count),jnp.int32(owner))
        expected=np.asarray(cache).copy()
        for i in range(count):
            pos=offset+i
            if (pos%512)//64==owner:expected[table[0,pos//512],pos%64]=np.asarray(rows[i])
        assert bool(out.valid)
        np.testing.assert_array_equal(np.asarray(out.cache),expected)
        np.testing.assert_array_equal(np.asarray(out.causal_lengths),np.where(np.arange(128)<count,offset+np.arange(128)+1,0))
    for start,n,t,r in ((2147483647,count,table,rows),(offset,129,table,rows),
            (offset,count,np.asarray([[0,0,1]],np.int32),rows),
            (0,128,table,rows.at[127].set(jnp.inf))):
        # The alias check only applies to pages in the populated prefix.
        if start==offset and n==count and count and offset+count<=512:continue
        out=call(r,jnp.asarray(t),jnp.int32(start),jnp.int32(n),jnp.int32(0))
        assert not bool(out.valid)
        np.testing.assert_array_equal(np.asarray(out.cache),np.asarray(cache))
        assert not np.asarray(out.row_valid).any()
