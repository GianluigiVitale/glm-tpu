"""Source-derived MTP projection on CPU32; no native drafter or TPU admission."""
import os
import subprocess
import sys


def test_mtp_projection_cpu32():
    code = r'''
import json
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.perf.mtp_projection import MtpProjectionWeights,build_mtp_projection
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
rng=np.random.default_rng(19);H=256
def bf(shape):return jnp.asarray(rng.normal(size=shape),jnp.bfloat16)
enorm,hnorm=bf((H,)),bf((H,));projection=bf((H,2*H))/jnp.bfloat16(16)
weights=MtpProjectionWeights(put(enorm,P('feature')),put(hnorm,P('feature')),put(projection,P('feature',None)))
fn=build_mtp_projection(mesh,hidden_size=H,epsilon=1e-5)
maximum=0.
for rows in (1,3,114,128):
    embedding,previous=bf((rows,H)),bf((rows,H))
    positions=jnp.arange(rows,dtype=jnp.int32)
    args=(put(embedding,P(None,'feature')),put(previous,P(None,'feature')),put(positions),weights)
    actual=jax.block_until_ready(fn(*args))
    # Independent unsharded formula from vLLM's IR RMSNorm; reductions/dot
    # shapes differ, so assert an explicit numerical boundary, not exactness.
    def norm(x,w):
        x=x.astype(jnp.float32)
        return ((x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-5)).astype(jnp.bfloat16)*w).astype(jnp.bfloat16)
    joined=jnp.concatenate((norm(jnp.where((positions==0)[:,None],0,embedding),enorm),norm(previous,hnorm)),axis=1)
    reference=jax.lax.dot_general(joined,projection,(((1,),(1,)),((),())),preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(actual.hidden_local,dtype=np.float32),np.asarray(reference,dtype=np.float32),rtol=.01,atol=.015625)
    error=float(jnp.max(jnp.abs(actual.hidden_local.astype(jnp.float32)-reference.astype(jnp.float32))))
    maximum=max(maximum,error)
    assert np.asarray(actual.contract_valid).all()
    assert len(actual.hidden_local.addressable_shards)==32
    for shard in actual.hidden_local.addressable_shards:
        np.testing.assert_allclose(np.asarray(shard.data,dtype=np.float32),np.asarray(reference[shard.index],dtype=np.float32),rtol=.01,atol=.015625)
    # Pos0 must be independent of its embedding and dependent on h[0].
    altered=embedding.at[0].set(bf((H,))*32)
    other=fn(put(altered,P(None,'feature')),args[1],args[2],weights)
    np.testing.assert_array_equal(np.asarray(other.hidden_local).view(np.uint16),np.asarray(actual.hidden_local).view(np.uint16))
    zero_hidden=fn(args[0],put(previous.at[0].set(0),P(None,'feature')),args[2],weights)
    assert not np.array_equal(np.asarray(zero_hidden.hidden_local[0]),np.asarray(actual.hidden_local[0]))
    # Changes to a later query must not affect any preceding query.
    if rows>1:
        future=fn(put(embedding.at[-1].set(17),P(None,'feature')),put(previous.at[-1].set(-11),P(None,'feature')),args[2],weights)
        np.testing.assert_array_equal(np.asarray(future.hidden_local[:-1]).view(np.uint16),np.asarray(actual.hidden_local[:-1]).view(np.uint16))

# Invalid input on a non-first feature/expert replica must refuse on all32.
rows=3;shape=(rows,H)
def poisoned(index):
    result=np.ones((rows,H//4),dtype=np.float32)
    if index[1].start==H//2:result[1,0]=np.nan
    return result.astype(np.dtype(jnp.bfloat16))
bad=jax.make_array_from_callback(shape,NamedSharding(mesh,P(None,'feature')),poisoned)
result=fn(bad,put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.arange(rows,dtype=jnp.int32)),weights)
for shard in result.contract_valid.addressable_shards:assert not bool(np.asarray(shard.data)[1])
# Deliberately corrupt only the last physical replica, beyond the ordinary
# replicated-input contract, to verify the explicit all-owner health reduction.
parts=[]
for i,device in enumerate(mesh.devices.flat):
    part=np.ones((rows,H//4),dtype=np.dtype(jnp.bfloat16))
    if i==31:part[1,0]=np.nan
    parts.append(jax.device_put(part,device))
one_bad=jax.make_array_from_single_device_arrays(shape,NamedSharding(mesh,P(None,'feature')),parts)
result=fn(one_bad,put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.arange(rows,dtype=jnp.int32)),weights)
for shard in result.contract_valid.addressable_shards:assert not bool(np.asarray(shard.data)[1])
negative=fn(put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.array([-1,1,2],jnp.int32)),weights)
assert not np.asarray(negative.contract_valid)[0]
for malformed in (weights._replace(enorm_local=weights.enorm_local.astype(jnp.float32)),weights._replace(eh_projection_local=weights.eh_projection_local[:,:H])):
    try:fn(put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.ones(shape,jnp.bfloat16),P(None,'feature')),put(jnp.arange(rows,dtype=jnp.int32)),malformed)
    except ValueError:pass
    else:raise AssertionError('invalid weights admitted')
print(json.dumps(dict(cpu_devices=32,rows=[1,3,114,128],hidden_size=H,maximum_absolute_error=maximum,
    reference='pinned vLLM IR RMSNorm expression plus unsharded BF16/FP32 dot',rtol=.01,atol=.015625,
    position_zero_mask=True,hidden_zero_retained=True,causal_rows=True,all_owner_refusal=True,
    native_drafter_complete=False,tpu_measured=False)))
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
