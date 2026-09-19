"""Feature-row splitting must preserve per-query results and failed health."""
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('options',[dict(feature_row_attention=1),dict(feature_row_attention=True),
    dict(feature_row_attention=True,lse_attention=True,bf16_resident=True,mlp_window=True)])
def test_feature_rows_require_replicated_prefill_composition(options):
    from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
    with pytest.raises(ValueError,match='feature-row attention'):
        build_ws32_prefill_challenger_program(None,None,**options)


def test_feature_row_attention_cpu32_bitwise():
    code=r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P
from glm_tpu.perf.feature_row_attention import feature_row_lse_attention
from glm_tpu.perf.lse_attention import lse_attention_mapped
from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract,StageLocalKvLayout,SparseAttentionResult
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng=np.random.default_rng(912)
c=MlaNumericalContract(num_heads=8,kv_lora_rank=128,qk_nope_head_dim=64,
    qk_rope_head_dim=64,qk_head_dim=128,v_head_dim=128,packed_cache_width=256,top_k=128)
l=StageLocalKvLayout(logical_page_size=512,local_parallel_size=8,packed_cache_width=256)
rows=8
q=jnp.asarray(rng.normal(size=(rows,8,128))*.3,jnp.bfloat16)
r=jnp.asarray(rng.normal(size=(rows,8,64))*.3,jnp.bfloat16)
cache=jnp.asarray(rng.normal(size=(8,2,64,256)),jnp.bfloat16)
tables=jnp.tile(jnp.array([[1,0]],jnp.int32),(rows,1))
length=jnp.full((rows,),1024,jnp.int32)
positions=np.full((rows,128),-1,np.int32);counts=np.zeros(rows,np.int32)
# Every feature slice sees different rows, including empties, balanced owners,
# concentrated fallback and cut boundaries. Padded future keys remain masked.
choices=[[],[0],list(range(64)),list(range(0,1024,8)),
         rng.choice(1024,128,replace=False).tolist(),[513],list(range(512,544)),[]]
for i,values in enumerate(choices):
    counts[i]=len(values);positions[i,:len(values)]=values
def body(q,r,cache,p,n):
    args=(q,r,cache[0],tables,SelectedPositions(p,n),length)
    opts=dict(contract=c,layout=l,config=SparseMlaConfig(segment_block=16),
              interpret=True,validate_finite=True,owned_key_capacity=32)
    return lse_attention_mapped(*args,**opts),feature_row_lse_attention(*args,**opts)
spec=SparseAttentionResult(P(None,'expert'),P(None,'expert'),P())
fn=jax.jit(jax.shard_map(body,mesh=mesh,
    in_specs=(P(None,'expert'),P(None,'expert'),P('expert'),P(),P()),out_specs=(spec,spec),check_vma=False))
def check(cache,p=positions,n=counts):
    a,b=fn(q,r,cache,jnp.asarray(p),jnp.asarray(n))
    for x,y in zip(a,b):
        np.testing.assert_array_equal(np.ascontiguousarray(x).view(np.uint8),np.ascontiguousarray(y).view(np.uint8))
    return b
assert np.asarray(check(cache).contract_valid).all()
# Selected poison occurs only in the second feature slice, and health must be
# restored to precisely the original rows; unselected poison is not an operand.
bad=check(cache.at[0,1,0,0].set(jnp.nan))
assert not bool(bad.contract_valid[1]) and bool(bad.contract_valid[0])
malformed=positions.copy();malformed[5,:2]=[513,513]
bad_counts=counts.copy();bad_counts[5]=2
assert not bool(check(cache,malformed,bad_counts).contract_valid[5])
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
