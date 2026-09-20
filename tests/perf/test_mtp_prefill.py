"""Constrained prefill mirror and CPU32 bootstrap-hidden/state boundary checks."""
import ast
import inspect
import os
import subprocess
import sys

from glm_tpu.greenfield.runtime import ws32_batched_prefill as frozen
from glm_tpu.perf import mtp_prefill


def test_prefill_mirror_only_adds_hidden_export_and_health():
    expected = inspect.getsource(frozen.ws32_batched_prefill_mapped)
    expected = expected.replace(') -> Ws32BatchedPrefillResult:', ') -> MtpPrefillResult:')
    expected = expected.replace(
        '    healthy = _all_owners_healthy(span_valid & jnp.all(~live | health) & head_health)',
        '''    normalized_hidden, hidden_health = export_hidden_mapped(
        update, residual, weights.final_norm_weight_local,
        hidden_size=config.geometry.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    healthy = _all_owners_healthy(
        span_valid & jnp.all(~live | health) & head_health
        & jnp.all(~live | hidden_health)
    )''')
    expected = expected.replace('    return Ws32BatchedPrefillResult(\n', '    return MtpPrefillResult(\n')
    expected = expected.replace('        jnp.where(healthy & final, token, -1),\n    )', '''        jnp.where(healthy & final, token, -1),
        jnp.where((healthy & live)[:, None], normalized_hidden, jnp.bfloat16(0)),
        healthy & live,
    )''')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(mtp_prefill.ws32_batched_prefill_mapped)))


def test_mtp_prefill_hidden_and_atomic_state_cpu32():
    code = r'''
from dataclasses import replace
import json
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b,ws32_decoder as d
from glm_tpu.greenfield.kernels.ws32_io import ws32_logits_mapped,ws32_greedy_sample_mapped
from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
from glm_tpu.perf.mtp_prefill import export_hidden_mapped
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
config,raw,wk=fixture(mesh,panel_geometry=True)
# Keep one dense and one full-index MoE layer, including canonical B114/B128
# placement, DSA, repair and all physical owners without a 78-layer CPU run.
config=replace(config,geometry=replace(config.geometry,num_layers=2,first_dense_layers=1,
    mlp_layer_types=('dense','sparse'),indexer_types=('full','full')))
raw=raw._replace(layers=(raw.layers[0],raw.layers[6]))
weights=bf16_resident_weights(mesh,config,raw)
wk=tuple(put(v) for v in (wk[0],wk[3]))
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
opts=dict(lse_attention=True,bf16_resident=True,key_tile=128,mlp_window=True,
    rolled_prefix=True,expert_panels=True,canonical_dense=True,
    paired_position_sort=True,sorted_local_merge=True,
    sparse_attention_interpret=True,linear_interpret=True)
original=b.ws32_batched_prefill_mapped
programs={rows:(build_ws32_prefill_challenger_program(mesh,config,block_rows=rows,**opts),
    build_ws32_prefill_challenger_program(mesh,config,block_rows=rows,export_mtp_hidden=True,**opts))
    for rows in (128,114)}
initial=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
a=c=initial

def exact(x,y):
    assert jax.tree.structure(x)==jax.tree.structure(y)
    for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)):
        for xs,ys in zip(xx.addressable_shards,yy.addressable_shards):
            aa,bb=np.asarray(xs.data),np.asarray(ys.data)
            if xx.dtype==jnp.bfloat16:aa,bb=aa.view(np.uint16),bb.view(np.uint16)
            np.testing.assert_array_equal(aa,bb)

def zeros(x):
    for shard in x.addressable_shards:assert not np.asarray(shard.data).any()

for rows,tokens,count in ((128,[30,31],2),(114,[32],1)):
    ids=put(jnp.asarray(tokens+[-1]*(rows-count),jnp.int32));n=put(jnp.int32(count))
    ordinary,export=programs[rows]
    x=jax.block_until_ready(ordinary.execute(ids,n,a,weights,wk,rope))
    y=jax.block_until_ready(export.execute(ids,n,c,weights,wk,rope))
    exact(x,b.Ws32BatchedPrefillResult(y.state,y.next_token))
    assert np.asarray(y.state.decoder.contract_valid).all()
    np.testing.assert_array_equal(np.asarray(y.hidden_valid),np.arange(rows)<count)
    zeros(y.normalized_hidden_local[count:])
    assert np.isfinite(np.asarray(y.normalized_hidden_local,dtype=np.float32)).all()
    if rows==114:
        # The exported last row, passed directly through the head (no second
        # norm), must predict exactly the original prefill next token.
        def head(h,w):
            logits=ws32_logits_mapped(h,w,vocab_size=config.geometry.vocab_size)
            return ws32_greedy_sample_mapped(logits,vocab_size=config.geometry.vocab_size).token_id
        sample=jax.jit(jax.shard_map(head,mesh=mesh,
            in_specs=(P(None,'feature'),P('expert','feature')),
            out_specs=P(),check_vma=False))
        exact(sample(y.normalized_hidden_local[:1],weights.lm_head_local),y.next_token)
    a,c=x.state,y.state
# Finished state refuses atomically; invalid hidden outputs cannot be consumed.
export=programs[114][1]
y=export.execute(ids,n,c,weights,wk,rope)
for key in ('kv_cache_local','index_cache_local','position','context_lengths'):
    exact(getattr(y.state.decoder,key),getattr(c.decoder,key))
assert not np.asarray(y.state.decoder.contract_valid).any()
zeros(y.normalized_hidden_local);zeros(y.hidden_valid)
# An invalid last physical replica's final norm must refuse even on an
# intermediate block, where the original next-token head is not executed.
parts=[]
for i,device in enumerate(mesh.devices.flat):
    part=np.ones(config.geometry.hidden_size//4,dtype=np.dtype(jnp.bfloat16))
    if i==31:part[0]=np.nan
    parts.append(jax.device_put(part,device))
badnorm=jax.make_array_from_single_device_arrays((config.geometry.hidden_size,),
    NamedSharding(mesh,P('feature')),parts)
bad=weights._replace(final_norm_weight_local=badnorm)
y=programs[128][1].execute(put(jnp.asarray([30,31]+[-1]*126,jnp.int32)),
    put(jnp.int32(2)),initial,bad,wk,rope)
for key in ('kv_cache_local','index_cache_local','position','context_lengths'):
    exact(getattr(y.state.decoder,key),getattr(initial.decoder,key))
for shard in y.state.decoder.contract_valid.addressable_shards:assert not np.asarray(shard.data).any()
zeros(y.normalized_hidden_local);zeros(y.hidden_valid)
assert b.ws32_batched_prefill_mapped is original
# Independent final RMSNorm expression, with nonzero carried residual and
# nontrivial norm weights. Check every row/feature shard, including full B128.
rng=np.random.default_rng(20260920);H=256;maximum=0.
for rows in (1,114,128):
    u=jnp.asarray(rng.normal(size=(rows,H)),jnp.bfloat16)
    r=jnp.asarray(rng.normal(size=(rows,H)),jnp.bfloat16)
    w=jnp.asarray(rng.normal(size=(H,)),jnp.bfloat16)
    fn=jax.jit(jax.shard_map(lambda u,r,w:export_hidden_mapped(u,r,w,hidden_size=H,epsilon=1e-5),
        mesh=mesh,in_specs=(P(None,'feature'),P(None,'feature'),P('feature')),
        out_specs=(P(None,'feature'),P()),check_vma=False))
    actual,health=fn(u,r,w)
    # The norm consumes the unrounded FP32 sum; only the separately carried
    # residual rounds that sum to BF16 in the pinned target head.
    summed=u.astype(jnp.float32)+r.astype(jnp.float32)
    expected=((summed*jax.lax.rsqrt(jnp.mean(summed*summed,axis=1,keepdims=True)+1e-5)).astype(jnp.bfloat16)*w).astype(jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(actual,dtype=np.float32),np.asarray(expected,dtype=np.float32),rtol=.01,atol=.015625)
    maximum=max(maximum,float(jnp.max(jnp.abs(actual.astype(jnp.float32)-expected.astype(jnp.float32)))))
    assert np.asarray(health).all()
print(json.dumps(dict(scope='CPU32 synthetic two-layer resident canonical P1/P2 B128+B114; intermediate and final export',
    ordinary_state_and_token='bitwise on all 32 owners',independent_norm_max_abs=maximum,
    invalid_last_replica='atomic refusal on all 32 owners',trained_tpu_admitted=False)))
'''
    env = dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32')
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
