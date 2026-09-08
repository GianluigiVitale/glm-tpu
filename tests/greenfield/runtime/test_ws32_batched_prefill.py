"""CPU composition proof, not TPU numerical or performance promotion."""

import os
import subprocess
import sys


def test_layer_major_two_chunk_state_and_decode_handoff_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import (
    build_ws32_main_rope_table,ws32_decoder_weight_specs,ws32_decode_mapped,ws32_decode_result_specs,
)
from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_transformer_layer_mapped
from glm_tpu.greenfield.kernels.ws32_io import ws32_split_final_sample_mapped
from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh)
def put(v,s=P()):return jax.device_put(v,NamedSharding(mesh,s))
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=533)
assert config.full_index_slots==(0,1,2,6)
# Synthetic populated-prefix fixture only, not a real checkpoint-resume claim.
rng=np.random.default_rng(517)
ds=state.decoder
ds=ds._replace(
    kv_cache_local=put(jnp.asarray(rng.normal(0,.1,ds.kv_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
    index_cache_local=put(jnp.asarray(rng.normal(0,.1,ds.index_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
    position=put(jnp.array([505],jnp.int32)),context_lengths=put(jnp.array([506],jnp.int32)),
    block_tables=put(jnp.array([[2,0,1]],jnp.int32)),
)
state=state._replace(decoder=ds,repaired_index_local=jnp.full_like(state.repaired_index_local,7))
program=b.build_ws32_batched_prefill_program(mesh,config,block_rows=17,key_tile=128,sparse_attention_interpret=True,linear_interpret=True)
tokens=put(jnp.arange(17,dtype=jnp.int32)+30)
compiled=program.execute.lower(tokens,put(jnp.int32(17)),state,weights,wk,rope).compile()
print('compiled full eight-layer block',flush=True)
ops=[x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
fg=tuple(tuple(range(e*4,e*4+4)) for e in range(8));eg=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert all(x.replica_groups in (fg,eg) for x in ops)
first=compiled(tokens,put(jnp.int32(17)),state,weights,wk,rope)
assert np.asarray(first.state.decoder.contract_valid).all()
assert first.next_token.tolist()==[-1] and not bool(first.state.finished)
assert first.state.decoder.position.tolist()==[522] and first.state.decoder.context_lengths.tolist()==[523]
assert not np.array_equal(first.state.decoder.index_cache_local,first.state.repaired_index_local)
try:b.finish_ws32_batched_prefill(first);raise AssertionError('premature decode')
except ValueError:pass

# Independently wire each layer through completed layer boundaries. This compares
# composition/ownership, not independent math: kernels are the admitted ones.
qspec=ws32_decoder_weight_specs(config)
layer_cache=P(None,'expert',None)
selection=(P(),P(),P())
out_specs=(P(None,'feature'),P(None,'feature'),layer_cache,layer_cache,layer_cache,*selection,P('expert','feature',None))
ins=(P(None,'feature'),P(None,'feature'),layer_cache,layer_cache,layer_cache,*selection,P(),P(),P(),P(),P('expert','feature',None))
reference_programs={}
def reference_block(tokens,count,start_state):
    offset=int(start_state.decoder.position[0]); live=int(count);rows=tokens.shape[0]
    embed=jax.jit(jax.shard_map(lambda ids,w:b.ws32_prefill_embedding_mapped(ids,w,jnp.int32(live),vocab_size=256).residual_local,mesh=mesh,in_specs=(P(),P('expert','feature')),out_specs=P(None,'feature'),check_vma=False))
    u=embed(tokens,weights.embedding_local);r=jnp.zeros_like(u)
    kv=start_state.decoder.kv_cache_local;ic=start_state.decoder.index_cache_local;rc=start_state.repaired_index_local
    s=put(jnp.full((rows,128),-1,jnp.int32));n=put(jnp.zeros(rows,jnp.int32));sc=put(jnp.full((rows,128),-jnp.inf,jnp.float32))
    health=put(jnp.ones((8,4,rows),jnp.bool_),P('expert','feature',None))
    witnesses=[]
    for layer,slot in ((0,0),(1,1),(2,2),(3,None),(4,None),(5,None),(6,3),(7,None)):
        w=weights.layers[layer]
        # NONE index inputs on shared layers are unchanged even though the
        # producer's compact selection must carry through, unlike per-layer KV.
        k=wk[slot] if slot is not None else None
        def run(u,r,c,i,z,s,n,sc,off,count,table,hrope,h,w,k):
            a=ws32_prefill_transformer_layer_mapped(u,r,c,i,z,s,n,sc,off,count,table,w.qkv_a,w.attention,w.dsa,k,w.post_attention_norm_weight_local,w.dense,w.moe,h[0,0],main_rope_table_rows=hrope,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract,key_tile=128,sparse_attention_config=SparseMlaConfig(segment_block=128),sparse_attention_interpret=True,linear_interpret=True)
            return a.output_local,a.carried_residual_local,a.cache_local,a.unrepaired_index_cache,a.repaired_index_cache,a.selected_positions,a.selected_valid_counts,a.selected_scores,a.contract_valid[None,None]
        kind=(slot is not None,layer<3)
        if kind not in reference_programs:
            reference_programs[kind]=jax.jit(jax.shard_map(run,mesh=mesh,in_specs=(*ins,qspec.layers[layer],None if k is None else P()),out_specs=out_specs,check_vma=False))
        fn=reference_programs[kind]
        prior=(s,n,sc)
        out=fn(u,r,kv[layer],ic[0 if slot is None else slot],rc[0 if slot is None else slot],s,n,sc,put(jnp.int32(offset)),put(jnp.int32(live)),start_state.decoder.block_tables,rope[offset:offset+rows],health,w,k)
        u,r=out[:2];kv=kv.at[layer].set(out[2]);s,n,sc=out[5:8];health=out[8]
        if slot is not None:ic=ic.at[slot].set(out[3]);rc=rc.at[slot].set(out[4])
        else:
            for x,y in zip(prior,(s,n,sc)):np.testing.assert_array_equal(x,y)
        assert np.asarray(health).all()
        witnesses.append((np.asarray(u),np.asarray(r),np.asarray(s)))
    # Distinct full producers replace selection; shared2→3 and6→7 retain it.
    np.testing.assert_array_equal(witnesses[2][2],witnesses[3][2])
    np.testing.assert_array_equal(witnesses[6][2],witnesses[7][2])
    assert not np.array_equal(witnesses[2][2],witnesses[6][2])
    head=jax.jit(jax.shard_map(lambda u,r,n,w:ws32_split_final_sample_mapped(u,r,n,w,hidden_size=512,vocab_size=256).token_id,mesh=mesh,in_specs=(P(None,'feature'),P(None,'feature'),P('feature'),P('expert','feature')),out_specs=P(),check_vma=False))
    token=head(u[live-1:live],r[live-1:live],weights.final_norm_weight_local,weights.lm_head_local)
    return kv,ic,rc,s[live-1:live],n[live-1:live],sc[live-1:live],token
ref=reference_block(tokens,17,state)
print('completed first block reference',flush=True)
for x,y in zip((first.state.decoder.kv_cache_local,first.state.decoder.index_cache_local,first.state.repaired_index_local,first.state.decoder.selected_positions,first.state.decoder.selected_valid_counts,first.state.decoder.selected_scores),ref[:6]):np.testing.assert_array_equal(x,y)

# Final padded tail: invalid IDs and NaN rotary rows outside live11 are ignored.
tail=put(jnp.concatenate((jnp.arange(11,dtype=jnp.int32)+90,jnp.full(6,-2147483648,jnp.int32))))
nan_rope=rope.at[533:539].set(jnp.nan)
last=compiled(tail,put(jnp.int32(11)),first.state,weights,wk,nan_rope)
assert bool(last.state.finished) and np.asarray(last.state.decoder.contract_valid).all()
assert last.state.decoder.position.tolist()==[533] and last.state.decoder.context_lengths.tolist()==[534]
ref=reference_block(tail,11,first.state)
print('completed tail reference',flush=True)
for x,y in zip((last.state.decoder.kv_cache_local,last.state.decoder.index_cache_local,last.state.repaired_index_local,last.state.decoder.selected_positions,last.state.decoder.selected_valid_counts,last.state.decoder.selected_scores,last.next_token),(ref[0],ref[2],ref[2],*ref[3:])):np.testing.assert_array_equal(x,y)
decoder,token=b.finish_ws32_batched_prefill(last)
np.testing.assert_array_equal(decoder.index_cache_local,last.state.repaired_index_local)
# Decode actually consumes the returned one-row state/token, on the raw CPU
# reference path only. Promoted exact-alias TPU handoff still needs §21 proof.
decode=jax.jit(jax.shard_map(lambda t,s,w,rope:ws32_decode_mapped(t,s,w,config=config,main_rope_table=rope,sparse_attention_interpret=True,linear_interpret=True),mesh=mesh,in_specs=(P(),b.ws32_decoder_state_specs(),qspec,P()),out_specs=ws32_decode_result_specs(),check_vma=False))
step=decode(token,decoder,weights,rope)
assert step.state.position.tolist()==[534] and np.asarray(step.state.contract_valid).all()
print('completed raw decoder handoff',flush=True)

def unchanged_failed(result,old):
    assert not np.asarray(result.state.decoder.contract_valid).any()
    assert result.next_token.tolist()==[-1]
    for name in old.decoder._fields:
        if name!='contract_valid':np.testing.assert_array_equal(getattr(result.state.decoder,name),getattr(old.decoder,name))
    np.testing.assert_array_equal(result.state.repaired_index_local,old.repaired_index_local)
    assert bool(result.state.finished)==bool(old.finished)
unchanged_failed(compiled(tail,put(jnp.int32(11)),last.state,weights,wk,rope),last.state)
for count in (0,-1,18,2147483647):unchanged_failed(compiled(tokens,put(jnp.int32(count)),state,weights,wk,rope),state)
for change in ({'position':put(jnp.array([2147483647],jnp.int32))},{'context_lengths':put(jnp.array([505],jnp.int32))},{'block_tables':put(jnp.array([[2,2,1]],jnp.int32))}):
    bad=state._replace(decoder=state.decoder._replace(**change))
    unchanged_failed(compiled(tokens,put(jnp.int32(17)),bad,weights,wk,rope),bad)
unchanged_failed(compiled(tokens.at[0].set(-1),put(jnp.int32(17)),state,weights,wk,rope),state)
# Repaired history must not leak into prompt computations in the second chunk.
altered=first.state._replace(repaired_index_local=jnp.full_like(first.state.repaired_index_local,-9))
changed=compiled(tail,put(jnp.int32(11)),altered,weights,wk,rope)
for name in ('kv_cache_local','selected_positions','selected_scores'):
    np.testing.assert_array_equal(getattr(changed.state.decoder,name),getattr(last.state.decoder,name))
np.testing.assert_array_equal(changed.next_token,last.next_token)
# One-owner health corruption on a FINAL block cannot cause divergent head
# collectives or commit healthy peers. Keep this in one compiled invocation.
def poison(t,c,s,w,k,r):
    own_bad=(lax.axis_index('expert')==3)&(lax.axis_index('feature')==2)
    ds=s.decoder._replace(contract_valid=s.decoder.contract_valid & ~own_bad)
    return b.ws32_batched_prefill_mapped(t,c,s._replace(decoder=ds),w,k,r,config=config,key_tile=128,sparse_attention_interpret=True,linear_interpret=True)
poison=jax.jit(jax.shard_map(poison,mesh=mesh,in_specs=(P(),P(),b.ws32_batched_prefill_state_specs(),qspec,tuple(P() for _ in wk),P()),out_specs=b.Ws32BatchedPrefillResult(b.ws32_batched_prefill_state_specs(),P()),check_vma=False))
unchanged_failed(poison(tail,put(jnp.int32(11)),first.state,weights,wk,rope),first.state)
print('CPU32_LAYER_MAJOR_TWO_CHUNKS_HANDOFF_PASS')
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=32",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32_LAYER_MAJOR_TWO_CHUNKS_HANDOFF_PASS" in result.stdout


def test_production_78_layer_schema_without_allocating_weights():
    code = r"""
import json
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig,Ws32DecoderState,ws32_decoder_weight_names,_bind_weight_name_tree
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.errors import PlanValidationError
tpu_info._get_tpu_info=lambda:tpu_info.TpuInfo.from_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
geometry=ModelGeometry.from_hf_config(json.loads(Path('configs/glm-5.2-fp8-config.json').read_text()))
config=Ws32DecoderConfig(geometry,8192,host_main_rope_table=True)
schema=json.loads(Path('tests/greenfield/hlo/fixtures/prefill_layer_schema.json').read_text())['tensor_schema']
types={'BF16':jnp.bfloat16,'F32':jnp.float32,'U8':jnp.uint8}
def abstract(shape,dtype,spec=P()):return jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh,spec))
source={t['name']:t for t in schema}
names=ws32_decoder_weight_names(config)
arrays={}
for layer in range(78):
    # Full producers after layer2 have MoE trees and layer0's DSA schema.
    for name in jax.tree.leaves(names.layers[layer]):
        suffix=name.split(f'model.layers.{layer}.',1)[1]
        template=f'model.layers.{0 if layer<3 or ".indexer." in name else 3}.{suffix}'
        t=source[template]
        arrays[name]=abstract(tuple(t['global_shape']),types[t['dtype']],P(*t['partition_spec']))
for name in (names.embedding_local,names.lm_head_local):arrays[name]=abstract((geometry.vocab_size,geometry.hidden_size),jnp.bfloat16,P('expert','feature'))
arrays[names.final_norm_weight_local]=abstract((geometry.hidden_size,),jnp.bfloat16,P('feature'))
weights=_bind_weight_name_tree(names,arrays)
ds=Ws32DecoderState(
    abstract(config.kv_cache_shape,jnp.bfloat16,P(None,None,'expert',None)),
    abstract(config.index_cache_shape,jnp.bfloat16,P(None,None,'expert',None)),
    abstract((1,2048),jnp.int32),abstract((1,),jnp.int32),abstract((1,2048),jnp.float32),
    abstract((1,),jnp.int32),abstract((1,16),jnp.int32),abstract((1,),jnp.int32),abstract((1,),jnp.bool_),
)
state=b.Ws32BatchedPrefillState(ds,ds.index_cache_local,abstract((),jnp.int32),abstract((),jnp.bool_))
wk=tuple(abstract((128,6144),jnp.float32) for _ in config.full_index_slots)
rope=abstract(config.main_rope_table_shape,jnp.bfloat16)
for rows in (17,11):
    fn=b.build_ws32_batched_prefill_program(mesh,config,block_rows=rows).execute
    out=jax.eval_shape(fn,abstract((rows,),jnp.int32),abstract((),jnp.int32),state,weights,wk,rope)
    assert out.state.decoder.kv_cache_local.shape==config.kv_cache_shape
    assert out.state.decoder.index_cache_local.shape==out.state.repaired_index_local.shape==config.index_cache_shape
    assert out.state.decoder.selected_positions.shape==(1,2048)
    assert out.next_token.shape==(1,) and out.state.finished.shape==()
for changed in (replace(config,exact_dsa=True),replace(config,strategy_nd_dense=True),replace(config,host_main_rope_table=False)):
    try:b.build_ws32_batched_prefill_program(mesh,changed,block_rows=17);raise AssertionError('unsupported config accepted')
    except PlanValidationError:pass
for rows in (0,33,True):
    try:b.build_ws32_batched_prefill_program(mesh,config,block_rows=rows);raise AssertionError('invalid row count accepted')
    except PlanValidationError:pass
print('REAL_78_LAYER_BATCHED_PREFILL_SCHEMA_PASS')
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=32",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_78_LAYER_BATCHED_PREFILL_SCHEMA_PASS" in result.stdout
