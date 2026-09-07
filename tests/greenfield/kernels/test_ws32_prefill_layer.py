"""CPU32 full-layer composition; no real-weight or TPU admission."""

import json
import os
import subprocess
import sys


def test_full_prefill_layer_branches_and_router_on_cpu32():
    program = r"""
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32QkvAWeights,Ws32AttentionWeights,Ws32DsaWeights,Ws32DenseWeights,Ws32MoeWeights,
    ws32_transformer_layer_mapped,
)
from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_transformer_layer_mapped,ws32_prefill_router_mapped
from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract,route_glm_noaux_tc_logits
from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng=np.random.default_rng(677)
def bf(shape):return jnp.asarray(rng.normal(0,.1,shape),jnp.bfloat16)
def bits(shape):return jnp.asarray(np.asarray(rng.normal(0,.02,shape),ml_dtypes.float8_e4m3fn).view(np.uint8))
def scales(shape):return jnp.asarray(rng.uniform(.5,1.5,shape),jnp.float32)
def put(v,s):return jax.device_put(v,NamedSharding(mesh,s))
def mapped(fn,ins,outs):return jax.jit(jax.shard_map(fn,mesh=mesh,in_specs=ins,out_specs=outs,check_vma=False))
R=17; count=11; offset=505
dc=DsaNumericalContract(hidden_size=512,q_lora_rank=128,top_k=16)
ac=MlaNumericalContract(num_heads=16,top_k=16)
mc=GlmMoeNumericalContract(hidden_size=512,intermediate_size=128,num_experts=64,stage_size=8)
kw=dict(dsa_contract=dc,attention_contract=ac,moe_contract=mc,sparse_attention_config=SparseMlaConfig(segment_block=8),sparse_attention_interpret=True,linear_interpret=True)
qspec=Ws32QkvAWeights(P('feature'),P(None,'feature'),P(None,'feature'),P(),P(None,'feature'),P(None,'feature'),P())
qw=Ws32QkvAWeights(jnp.ones(512,jnp.bfloat16),bits((128,512)),scales((1,4)),jnp.ones(128,jnp.bfloat16),bits((576,512)),scales((5,4)),jnp.ones(512,jnp.bfloat16))
aspec=Ws32AttentionWeights(P('expert',None),P('expert',None),P('expert',None),P('expert',None),P('feature','expert'),P('feature','expert'))
aw=Ws32AttentionWeights(bits((4096,128)),scales((32,1)),bits((7168,512)),scales((56,4)),bits((512,4096)),scales((4,32)))
dspec=Ws32DsaWeights(P('expert',None),P('expert',None),P(None,'feature'),P(None,'feature'),P(),P(),P('expert','feature'))
dw=Ws32DsaWeights(bits((4096,128)),scales((32,1)),bits((128,512)),scales((1,4)),jnp.ones(128,jnp.bfloat16),bf((128,)),bf((32,512)))
denspec=Ws32DenseWeights(P('expert','feature'),P('expert','feature'),P('expert','feature'),P('expert','feature'),P('feature','expert'),P('feature','expert'))
dense=Ws32DenseWeights(bits((1024,512)),scales((8,4)),bits((1024,512)),scales((8,4)),bits((512,1024)),scales((4,8)))
mospec=Ws32MoeWeights(P('expert','feature'),P('expert'),P('expert',None,'feature'),P('expert',None,'feature'),P('expert',None,'feature'),P('expert',None,'feature'),P('expert','feature',None),P('expert','feature',None),P(None,'feature'),P(None,'feature'),P(None,'feature'),P(None,'feature'),P('feature',None),P('feature',None))
moe=Ws32MoeWeights(bf((64,512)),jnp.zeros(64,jnp.float32),bits((64,128,512)),scales((64,1,4)),bits((64,128,512)),scales((64,1,4)),bits((64,512,128)),scales((64,4,1)),bits((128,512)),scales((1,4)),bits((128,512)),scales((1,4)),bits((512,128)),scales((4,1)))
cache=bf((8,3,64,640));index=bf((8,3,64,128));repair=jnp.full_like(index,7)
selections=np.full((R,16),-1,np.int32)
for i in range(count):selections[i]=np.arange(offset+i+1-16,offset+i+1)
counts=jnp.where(jnp.arange(R)<count,16,0).astype(jnp.int32)
scores=jnp.where(jnp.arange(R)[:,None]<count,jnp.ones((R,16),jnp.float32),-jnp.inf)
angles=jnp.asarray(rng.normal(size=(R,32)),jnp.float32)
rope=jnp.concatenate((jnp.cos(angles),jnp.sin(angles)),axis=1).astype(jnp.bfloat16)
values=(bf((R,512)),bf((R,512)),cache,index,repair,jnp.asarray(selections),counts,scores,jnp.int32(offset),jnp.int32(count),jnp.asarray([[2,0,1]],jnp.int32),qw,aw,dw,bf((128,512)).astype(jnp.float32),jnp.ones(512,jnp.bfloat16),dense,moe,jnp.ones((8,4,R),jnp.bool_),rope)
ins=(P(None,'feature'),P(None,'feature'),P('expert',None,None,None),P('expert',None,None,None),P('expert',None,None,None),P(),P(),P(),P(),P(),P(),qspec,aspec,dspec,P(),P('feature'),denspec,mospec,P('expert','feature',None),P())
values=jax.tree.map(put,values,ins)
outs=(P(None,'feature'),P(None,'feature'),P('expert',None,None,None),P('expert',None,None,None),P('expert',None,None,None),P(),P(),P(),P(),P(),P('expert','feature',None))
def body(full,moe_kind,*v):
    u,r,c,ic,rc,s,n,sc,o,count,t,q,a,d,k,post,dense,moe,health,hrope=v
    result=ws32_prefill_transformer_layer_mapped(u,r,c[0],ic[0],rc[0],s,n,sc,o,count,t,q,a,d if full else None,k if full else None,post,None if moe_kind else dense,moe if moe_kind else None,health[0,0],main_rope_table_rows=hrope,key_tile=128,**kw)
    return (result.output_local,result.carried_residual_local,result.cache_local[None],result.unrepaired_index_cache[None],result.repaired_index_cache[None],result.selected_positions,result.selected_valid_counts,result.selected_scores,result.route_indices,result.route_weights,result.contract_valid[None,None])
functions={(f,m):mapped(lambda *v,full=f,sparse=m:body(full,sparse,*v),ins,outs) for f,m in ((False,False),(True,True),(True,False),(False,True))}
report={}
for case,fn in functions.items():
    compiled=fn.lower(*values).compile()
    out=compiled(*values)
    assert np.asarray(out[-1]).all(),case
    assert np.all(np.asarray(out[0])[count:]==0) and np.all(np.asarray(out[1])[count:]==0)
    assert np.all(np.asarray(out[9])[count:]==0)
    if not case[0]:
        for i in (3,4,5,6,7):np.testing.assert_array_equal(out[i],values[i])
    changed=list(values);changed[18]=values[18].at[3,2,4].set(False)
    bad=fn(*changed)
    assert not bool(bad[-1][3,2,4]),case
    ops=[x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
    fg=tuple(tuple(range(e*4,e*4+4)) for e in range(8));eg=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
    assert all(x.replica_groups in (fg,eg) for x in ops)
    report[str(case)]=sorted(x.maximum_group_size for x in ops)

# The shared/dense branch compares directly to the previous complete one-row layer.
def old_body(*v):
    u,r,c,ic,rc,s,n,sc,o,count,t,q,a,d,k,post,dense,moe,health,hrope=v
    x=ws32_transformer_layer_mapped(u,r,c[0],ic[0],s,n,sc,o[None],t,(o+1)[None],q,a,None,post,dense,None,health[0,0],indexer_kind='shared',mlp_kind='dense',main_rope_table_row=hrope[0],**kw)
    return (x.output_local,x.carried_residual_local,x.cache_local[None],x.index_cache_local[None],rc,x.selected_positions,x.selected_valid_counts,x.selected_scores,x.route_indices,x.route_weights,x.contract_valid[None,None])
old=mapped(old_body,ins,outs)
actual=functions[(False,False)](*values)
current=values[2]; pieces=[]
for i in range(count):
    v=list(values)
    for slot in (0,1,5,6,7,19):v[slot]=v[slot][i:i+1]
    v[18]=v[18][:,:,i:i+1];v[2]=current;v[8]=jnp.int32(offset+i)
    ref=old(*v);pieces.append(ref);current=ref[2]
for idx in (0,1):
    expected=jnp.pad(jnp.concatenate([x[idx] for x in pieces]),((0,R-count),(0,0)))
    np.testing.assert_array_equal(actual[idx],expected)
np.testing.assert_array_equal(actual[2],current)

# Shared-index invalid producer health and live NaN scores must not disappear.
v=list(values);v[7]=v[7].at[0,0].set(jnp.nan)
assert not np.asarray(functions[(False,True)](*v)[-1])[:,:,0].any()
# Full-indexer + MoE: changed repaired history/future rows cannot alter earlier rows.
fn=functions[(True,True)];base=fn(*values)
v=list(values);v[4]=jnp.full_like(v[4],-9)
other=fn(*v)
for idx in (0,1,2,3,5,6,7,8,9):np.testing.assert_array_equal(base[idx],other[idx])
v=list(values);v[0]=v[0].at[count-1].set(3)
future=fn(*v)
for idx in (0,1,5,6,7,8,9):np.testing.assert_array_equal(base[idx][:count-1],future[idx][:count-1])
v=list(values)
for idx in (0,1,19):v[idx]=v[idx].at[count:].set(jnp.nan)
tail=fn(*v)
assert np.asarray(tail[-1]).all()
for idx in range(10):np.testing.assert_array_equal(base[idx],tail[idx])

# Standalone router: exactly representable products/sums allow an independent
# dense FP32 logits reference, including correction-only selection and ties.
rspec=(P(None,'feature'),P('expert','feature'),P('expert'),P())
def router_body(x,w,b,l):
    ids,weights,health=ws32_prefill_router_mapped(x,w,b,l)
    return ids,weights,health[None,None]
router=mapped(router_body,rspec,(P(),P(),P('expert','feature',None)))
x=jnp.asarray(rng.integers(-2,3,(R,512))*.125,jnp.bfloat16)
w=jnp.asarray(rng.integers(-2,3,(64,512))*.125,jnp.bfloat16)
b=jnp.asarray(rng.normal(0,.2,64),jnp.float32);live=jnp.arange(R)<11
rv=tuple(put(v,s) for v,s in zip((x,w,b,live),rspec))
ids,rw,health=router(*rv);assert np.asarray(health).all()
expected=route_glm_noaux_tc_logits(x.astype(jnp.float32)@w.astype(jnp.float32).T,b)
np.testing.assert_array_equal(ids[:11],expected[0][:11]);np.testing.assert_array_equal(rw[:11],expected[1][:11])
assert np.all(np.asarray(rw)[11:]==0)
zeros=router(rv[0],jnp.zeros_like(rv[1]),jnp.zeros_like(rv[2]),rv[3])
np.testing.assert_array_equal(zeros[0],np.broadcast_to(np.arange(8),(R,8)))
biased=router(rv[0],jnp.zeros_like(rv[1]),rv[2].at[32:40].set(10),rv[3])
np.testing.assert_array_equal(np.sort(np.asarray(biased[0][:11]),axis=1),np.broadcast_to(np.arange(32,40),(11,8)))
np.testing.assert_array_equal(biased[1][:11],np.full((11,8),.125,np.float32))
assert not np.asarray(router(rv[0],rv[1],rv[2].at[0].set(jnp.nan),rv[3])[-1])[:,:,:11].any()
print(json.dumps({'branches':4,'shared_dense_old_cpu_exact':True,'router_exact':True,'groups':report}))
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert (
        report["branches"] == 4
        and report["shared_dense_old_cpu_exact"]
        and report["router_exact"]
    )
