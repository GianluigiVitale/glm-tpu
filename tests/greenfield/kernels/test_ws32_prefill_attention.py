"""Forced32 causal attention assembly admission; no TPU/performance claim."""

import json
import os
import subprocess
import sys


def test_batched_attention_matches_serial_cache_and_outputs_on_cpu32():
    program = r"""
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32AttentionWeights, Ws32PreparedAttention, Ws32QkvAWeights,
    ws32_index_share_attention_mapped, ws32_prepare_attention_mapped,
)
from glm_tpu.greenfield.kernels.ws32_prefill_attention import (
    ws32_prefill_index_share_attention_mapped, ws32_prefill_prepare_attention_mapped,
)
from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from glm_tpu.optimized.reference.attention import MlaNumericalContract
from glm_tpu.optimized.hlo_contract import parse_hlo_module
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng = np.random.default_rng(915)
def bf(shape):
    return jnp.asarray(rng.normal(0,.1,shape),jnp.bfloat16)
def bits(shape):
    return jnp.asarray(np.asarray(rng.normal(0,.04,shape),ml_dtypes.float8_e4m3fn).view(np.uint8))
def scale(shape):
    return jnp.asarray(rng.uniform(.5,1.5,shape),jnp.float32)
def put(v,s):
    return jax.device_put(v,NamedSharding(mesh,s))
def mapped(fn,ins,outs):
    return jax.jit(jax.shard_map(fn,mesh=mesh,in_specs=ins,out_specs=outs,check_vma=False))
R = 17
res = put(bf((R,512)),P(None,'feature'))
prepared = Ws32PreparedAttention(res,res,put(bf((R,128)),P()),put(bf((R,576)),P()))
prep_spec = Ws32PreparedAttention(P(None,'feature'),P(None,'feature'),P(),P())
weight_spec = Ws32AttentionWeights(P('expert',None),P('expert',None),P('expert',None),P('expert',None),P('feature','expert'),P('feature','expert'))
weights = Ws32AttentionWeights(bits((4096,128)),scale((32,1)),bits((7168,512)),scale((56,4)),bits((512,4096)),scale((4,32)))
weights = jax.tree.map(put,weights,weight_spec)
cache = put(bf((8,3,64,640)),P('expert',None,None,None))
table = put(jnp.asarray([[2,0,1]],jnp.int32),P())
# Distinct host-style BF16 cos/sin rows, shared between both implementations.
angles = jnp.asarray(rng.normal(size=(R,32)),jnp.float32)
rope = put(jnp.concatenate((jnp.cos(angles),jnp.sin(angles)),axis=1).astype(jnp.bfloat16),P())
contract = MlaNumericalContract(num_heads=16,top_k=16)
kwargs = dict(contract=contract,sparse_attention_config=SparseMlaConfig(segment_block=8),sparse_attention_interpret=True,linear_interpret=True)
ins=(P(None,'feature'),prep_spec,P('expert',None,None,None),P(),P(),P(),P(),P(),weight_spec,P())
outs=(P(None,'feature'),P('expert',None,None,None),P('expert','feature',None))
def batch_body(r,p,c,s,n,offset,count,t,w,h):
    result=ws32_prefill_index_share_attention_mapped(r,p,c[0],s,n,offset,count,t,w,main_rope_table_rows=h,**kwargs)
    return result.output_local,result.cache_local[None],result.contract_valid[None,None]
def serial_body(r,p,c,s,n,offset,count,t,w,h):
    result=ws32_index_share_attention_mapped(r,p,c[0],s,n,offset[None],t,(offset+1)[None],w,main_rope_table_row=h[0],**kwargs)
    return result.output_local,result.cache_local[None],result.contract_valid[None,None]
batch,serial=mapped(batch_body,ins,outs),mapped(serial_body,ins,outs)
def selections(offset, count):
    p=np.full((R,16),-1,np.int32)
    n=np.zeros(R,np.int32)
    for i in range(count):
        end=offset+i+1
        n[i]=min(end,16)
        # Ascending here, and the consumer canonicalizes supplied DSA score order.
        p[i,:n[i]]=np.arange(end-n[i],end)
    return put(jnp.asarray(p),P()),put(jnp.asarray(n),P())
def run(offset,count,p=prepared,h=rope,s=None,n=None,c=cache,r=res):
    if s is None:s,n=selections(offset,count)
    return batch(r,p,c,s,n,put(jnp.int32(offset),P()),put(jnp.int32(count),P()),table,weights,h)
for offset,count in ((55,17),(505,17),(0,11)):
    s,n=selections(offset,count)
    out,end,health=run(offset,count,s=s,n=n)
    assert np.asarray(health).all()
    expected=[]
    current=cache
    for i in range(count):
        step=serial(res[i:i+1],jax.tree.map(lambda v:v[i:i+1],prepared),current,s[i:i+1],n[i:i+1],jnp.int32(offset+i),jnp.int32(1),table,weights,rope[i:i+1])
        assert np.asarray(step[2]).all()
        expected.append(step[0]);current=step[1]
    expected=jnp.pad(jnp.concatenate(expected),((0,R-count),(0,0)))
    np.testing.assert_array_equal(np.asarray(out).view(np.uint16),np.asarray(expected).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(end).view(np.uint16),np.asarray(current).view(np.uint16))

# Editing a future live row's key/query must not change earlier query outputs.
base=run(55,17)
changed=prepared._replace(q_residual=prepared.q_residual.at[-1].set(2),current_kv=prepared.current_kv.at[-1].set(3))
future=run(55,17,p=changed)
np.testing.assert_array_equal(base[0][:-1],future[0][:-1])
assert not np.array_equal(base[0][-1],future[0][-1])
# Padded garbage is ignored; zero-live block is a healthy exact cache no-op.
poison=jax.tree.map(lambda v:v.at[11:].set(jnp.nan),prepared)
partial=run(0,11,p=poison,h=rope.at[11:].set(jnp.nan),r=res.at[11:].set(jnp.nan))
np.testing.assert_array_equal(partial[0],run(0,11)[0]);assert np.asarray(partial[2]).all()
empty=run(0,0,p=jax.tree.map(lambda v:jnp.full_like(v,jnp.nan),prepared))
assert np.asarray(empty[2]).all() and np.all(np.asarray(empty[0])==0)
np.testing.assert_array_equal(empty[1],cache)
# Future selection is rejected despite that key being physically present in the block.
s,n=selections(55,17)
assert not np.asarray(run(55,17,s=s.at[0,-1].set(56),n=n)[2])[:,:,0].any()
assert not np.asarray(run(55,17,s=s,n=n.at[0].set(15))[2])[:,:,0].any()
bad=run(2147483647,17,s=s,n=n)
assert not np.asarray(bad[2]).any();np.testing.assert_array_equal(bad[1],cache)
bad_key=prepared._replace(current_kv=prepared.current_kv.at[0,0].set(jnp.nan))
bad=run(55,17,p=bad_key)
assert not np.asarray(bad[2]).any();np.testing.assert_array_equal(bad[1],cache)
bad_query=prepared._replace(q_residual=prepared.q_residual.at[0,0].set(jnp.nan))
assert not np.asarray(run(55,17,p=bad_query)[2])[:,:,0].any()
# Position40 is selected by row0, owned by expert0 in physical page2.
old_nan=cache.at[0,2,40,0].set(jnp.nan)
assert not np.asarray(run(55,17,c=old_nan)[2])[:,:,0].any()
# Rotaries are explicit live operands even if the null sink hides their NaNs.
assert not np.asarray(run(55,17,h=rope.at[0,0].set(jnp.nan))[2])[:,:,0].any()
# The executed CPU graph contains only the two expert8 attention exchanges.
compiled=batch.lower(res,prepared,cache,s,n,jnp.int32(55),jnp.int32(17),table,weights,rope).compile()
ops=[x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
groups=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert len(ops)==2 and all(x.opcode=='all-reduce' and x.replica_groups==groups for x in ops)

# Independently compare multirow preparation with the existing raw-projection path.
qspec=Ws32QkvAWeights(P('feature'),P(None,'feature'),P(None,'feature'),P(),P(None,'feature'),P(None,'feature'),P())
qw=Ws32QkvAWeights(jnp.ones(512,jnp.bfloat16),bits((128,512)),scale((1,4)),jnp.ones(128,jnp.bfloat16),bits((576,512)),scale((5,4)),jnp.ones(512,jnp.bfloat16))
qw=jax.tree.map(put,qw,qspec)
pb=mapped(lambda r,w:ws32_prefill_prepare_attention_mapped(r,w,linear_interpret=True),(P(None,'feature'),qspec),prep_spec)
ps=mapped(lambda r,w:ws32_prepare_attention_mapped(r,w,hidden_size=512,linear_interpret=True),(P(None,'feature'),qspec),prep_spec)
actual=pb(res,qw)
expected=jax.tree.map(lambda *xs:jnp.concatenate(xs),*[ps(res[i:i+1],qw) for i in range(R)])
for a,e in zip(actual,expected):np.testing.assert_array_equal(a,e)
print(json.dumps({'rows':17,'causal_cache_cases':3,'groups':[8,8],'preparation':'exact_cpu'}))
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
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rows": 17,
        "causal_cache_cases": 3,
        "groups": [8, 8],
        "preparation": "exact_cpu",
    }
