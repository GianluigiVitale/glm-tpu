"""Actual CPU32 arithmetic/scheduling tests; not TPU row-placement reproduction."""

import os
import subprocess
import sys


def test_canonical_dense_cpu32():
    code = r"""
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
from glm_tpu.greenfield.kernels.ws32_prefill_dense_canonical import ws32_prefill_dense_canonical_mapped as candidate
from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_mlp_mapped
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
cfg,weights,_=fixture(mesh,panel_geometry=True)
spec=ws32_decoder_weight_specs(cfg).layers[0].dense
def put(v,s):return jax.device_put(v,NamedSharding(mesh,s))
def body(x,live,dense):
 out=candidate(x[0,0],live,dense,moe_contract=cfg.moe_contract,linear_interpret=True)
 return tuple(v[None,None] for v in out)
def ref_body(x,live,dense):
 rows=x.shape[2];v=jnp.where(live[:,None],x[0,0],0)
 v=jnp.pad(v,((0,128-rows),(0,0))); m=jnp.pad(live,((0,128-rows),))
 out=[]
 for start in (0,32,64,96):
  part=jnp.pad(v[start:start+32],((0,96),(0,0)))
  mask=jnp.pad(m[start:start+32],((0,96),))
  y,ids,w,h=ws32_prefill_mlp_mapped(part,mask,dense,None,
       moe_contract=cfg.moe_contract,linear_interpret=True,expert_panels=True)
  h=h & (~mask | jnp.all(jnp.isfinite(y),axis=1));y=jnp.where(mask[:,None],y,0)
  out.append((y[:32],ids[:32],w[:32],h[:32]))
 return tuple(jnp.concatenate([o[j] for o in out])[:rows][None,None] for j in range(4))
def original_body(x,live,dense):
 values=jnp.where(live[:,None],x[0,0],0)
 y,ids,w,h=ws32_prefill_mlp_mapped(values,live,dense,None,
      moe_contract=cfg.moe_contract,linear_interpret=True,expert_panels=True)
 h=h & (~live | jnp.all(jnp.isfinite(y),axis=1));y=jnp.where(live[:,None],y,0)
 return tuple(v[None,None] for v in (y,ids,w,h))
def wrap(fn):return jax.jit(jax.shard_map(fn,mesh=mesh,
 in_specs=(P('expert','feature'),P(),spec),out_specs=(P('expert','feature'),)*4,check_vma=False))
fn,ref=wrap(body),wrap(ref_body)
original=wrap(original_body)
rng=np.random.RandomState(606)
width=cfg.geometry.hidden_size//4
for rows,counts in ((128,(0,1,31,32,33,63,64,65,95,96,97,127,128)),(114,(0,91,96,97,113,114))):
 # Distinct owner inputs remain simultaneous; expert axis is not independent.
 data=(rng.randn(8,4,rows,width)*.01).astype(jnp.bfloat16)
 x=put(data,P('expert','feature'))
 for count in counts:
  live=put(np.arange(rows)<count,P())
  got=fn(x,live,weights.layers[0].dense);expected=ref(x,live,weights.layers[0].dense)
  jax.block_until_ready((got,expected))
  for a,b in zip(got,expected,strict=True):
   a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
  # Independent full-row original, not another copy of the placement schedule.
  full=original(x,live,weights.layers[0].dense)
  for a,b in zip(got,full,strict=True): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
  assert np.asarray(got[3]).all() and np.count_nonzero(np.asarray(got[0])[:,:,count:])==0
  assert (np.asarray(got[1])==-1).all() and not np.asarray(got[2]).any()
  poison=data.copy();poison[:,:,count:]=np.nan
  poisoned=fn(put(poison,P('expert','feature')),live,weights.layers[0].dense)
  for a,b in zip(got,poisoned,strict=True): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
 # A live NaN remains an invalid result rather than disappearing in padding.
 bad=data.copy();bad[:,:,0]=np.nan
 result=fn(put(bad,P('expert','feature')),put(np.ones(rows,bool),P()),weights.layers[0].dense)
 assert not np.asarray(result[3])[:,:,0].any()
 # One physical owner's NaN in a later tile must stay at that output row.
 bad=data.copy();bad[2,1,33,0]=np.nan
 result=fn(put(bad,P('expert','feature')),put(np.ones(rows,bool),P()),weights.layers[0].dense)
 full=original(put(bad,P('expert','feature')),put(np.ones(rows,bool),P()),weights.layers[0].dense)
 health=np.asarray(result[3]);assert np.array_equal(health,np.asarray(full[3]))
 assert not health[:,:,33].all() and health[:,:,:33].all() and health[:,:,34:].all()
for rows,dtype in ((32,jnp.bfloat16),(128,jnp.float32)):
 try:
  candidate(jnp.zeros((rows,width),dtype),jnp.ones(rows,bool),weights.layers[0].dense,
      moe_contract=cfg.moe_contract,linear_interpret=True)
  raise AssertionError('accepted invalid shape/dtype')
 except ValueError:pass
print('CANONICAL_DENSE_CPU32_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=240,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CANONICAL_DENSE_CPU32_PASS" in result.stdout
