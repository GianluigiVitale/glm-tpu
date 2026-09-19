"""Global tile maxima preserve the probability-scale boundary, not FP32 sum order."""
import os
import subprocess
import sys


def test_global_tile_attention_cpu8():
    code=r'''
from dataclasses import replace
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P
from glm_tpu.perf.global_tile_attention import global_tile_attention_mapped
from glm_tpu.perf.lse_attention import lse_attention_mapped
from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract,StageLocalKvLayout,SparseAttentionResult,gather_stage_local_selected_kv_aligned
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig,pregathered_sparse_mla_pallas
mesh=Mesh(np.asarray(jax.devices(),object),('expert',))
c=MlaNumericalContract(num_heads=8,kv_lora_rank=128,qk_nope_head_dim=64,qk_rope_head_dim=64,qk_head_dim=128,v_head_dim=128,packed_cache_width=256,top_k=128)
l=StageLocalKvLayout(logical_page_size=512,local_parallel_size=8,packed_cache_width=256)
rng=np.random.default_rng(51)
q=jnp.asarray(rng.normal(size=(3,8,128))*.3,jnp.bfloat16)
r=jnp.asarray(rng.normal(size=(3,8,64))*.3,jnp.bfloat16)
cache=jnp.asarray(rng.normal(size=(8,2,64,256)),jnp.bfloat16)
tables=jnp.tile(jnp.array([[1,0]],jnp.int32),(3,1));length=jnp.full((3,),1024,jnp.int32)
config=SparseMlaConfig(segment_block=16)
def body(q,r,cache,pos,cnt):
    selection=SelectedPositions(pos,cnt)
    args=(q,r,cache[0],tables,selection,length)
    opts=dict(contract=c,layout=l,config=config,interpret=True,validate_finite=True)
    result=global_tile_attention_mapped(*args,**opts)
    old=lse_attention_mapped(*args,**opts)
    aligned=gather_stage_local_selected_kv_aligned(cache[0],tables,selection,length,layout=l,owner_index=jax.lax.axis_index('expert'))
    frozen=pregathered_sparse_mla_pallas(q,r,jax.lax.psum(aligned.values,'expert'),cnt,
        contract=replace(c,num_heads=1),config=config,interpret=True,prefill=True)
    return result,old.output,frozen
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,'expert'),P(None,'expert'),P('expert'),P(),P()),
    out_specs=(SparseAttentionResult(P(None,'expert'),P(None,'expert'),P()),P(None,'expert'),P(None,'expert')),check_vma=False))
errors=[]
for positions in ([],[0],list(range(64)),list(range(0,1024,8)),rng.choice(1024,128,replace=False).tolist()):
    n=len(positions)
    pos=jnp.tile(jnp.array([positions+[-1]*(128-n)],jnp.int32),(3,1)).at[1].set(-1)
    cnt=jnp.full((3,),n,jnp.int32).at[1].set(0)
    result,old,frozen=fn(q,r,cache,pos,cnt)
    error=float(jnp.max(jnp.abs(result.output.astype(jnp.float32)-frozen.astype(jnp.float32))))
    errors.append((error,float(jnp.max(jnp.abs(old.astype(jnp.float32)-frozen.astype(jnp.float32))))))
    assert bool(result.contract_valid.all()) and bool(jnp.isfinite(result.output).all())
    assert error <= .0078125,errors
    assert bool((result.output[1]==0).all()) and bool(jnp.isneginf(result.logsumexp[1]).all())
pos=jnp.tile(jnp.array([[0]+[-1]*127],jnp.int32),(3,1));cnt=jnp.ones((3,),jnp.int32)
result,_,_=fn(q,r,cache.at[0,1,0,0].set(jnp.nan),pos,cnt)
assert not bool(result.contract_valid.any())
result,_,_=fn(q,r,cache.at[0,1,3,0].set(jnp.nan),pos,cnt)
assert bool(result.contract_valid.all())
result,_,_=fn(q,r,cache,jnp.tile(jnp.array([[0,0]+[-1]*126],jnp.int32),(3,1)),jnp.full((3,),2,jnp.int32))
assert not bool(result.contract_valid.any())
print(errors)
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=8'),timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
