"""CPU32 diagnostic observation/suffix mechanisms, not original TPU identity."""

import os
import subprocess
import sys


def test_observed_dense_window_and_completed_suffix_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_dense_frontier_program import build_program
from scripts.greenfield.ws32_dense_norm_boundary import build_completed_dense_suffix,build_capture_program,build_owner_packet_suffix
from scripts.greenfield.prefill_window_boundary import capture_owner_arrays
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(v,s=P()):return jax.device_put(v,NamedSharding(mesh,s))
def same(a,b):
 x,y=np.asarray(a),np.asarray(b)
 assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
wk=tuple(put(w) for w in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
cache_spec=P('expert',None,None,None)
caches=tuple(tuple(put(jnp.zeros((8,config.page_count,64,width),jnp.bfloat16),cache_spec)
                   for width in (640,128,128)) for _ in range(2))
table=put(jnp.arange(config.page_count,dtype=jnp.int32)[None])
tokens=put(jnp.arange(128,dtype=jnp.int32)+30)
args=(tokens,put(jnp.int32(128)),put(jnp.int32(0)),table,caches,
      weights.embedding_local,weights.layers[:2],wk[:2],rope)
original=build_program(mesh,config,interpret=True).lower(*args).compile()
observed=build_capture_program(mesh,config,interpret=True).lower(*args).compile()
slots={int(d.id):slot for slot,d in enumerate(mesh.devices.flat)}
suffix=build_completed_dense_suffix(mesh,config,interpret=True)
owner_suffix=build_owner_packet_suffix(mesh,config,interpret=True)
current=caches
for count,offset in ((128,0),(32,0),(32,32),(32,64),(32,96)):
 ids=tokens if count==128 else put(jnp.pad(tokens[offset:offset+32],(0,96)))
 inputs=(ids,put(jnp.int32(count)),put(jnp.int32(offset)),table,current,
         weights.embedding_local,weights.layers[:2],wk[:2],rope)
 ref=original(*inputs); out,packet=observed(*inputs); jax.block_until_ready((ref,out,packet))
 for a,b in zip(jax.tree.leaves(ref),jax.tree.leaves(out),strict=True):same(a,b)
 owner=capture_owner_arrays(packet,slots_by_device=slots)
 assert len(owner)==32 and all(len(v)==11 for v in owner.values())
 for d,values in owner.items():
  assert values['post_norm/summed'].shape==(128,config.geometry.hidden_size//4)
  np.testing.assert_array_equal(values['boundary/live'],np.arange(128)<count)
  summed=values['post_norm/update'].astype(np.float32)+values['post_norm/residual'].astype(np.float32)
  same(values['post_norm/summed'],summed)
  same(values['post_norm/carried'],summed.astype(jnp.bfloat16))
  same(values['boundary/normalized_mlp'][:count],values['post_norm/normalized'][:count])
  assert np.count_nonzero(values['boundary/normalized_mlp'][count:])==0
 # Reassemble original feature shards for CPU mechanism test only. Hardware
 # replay must place local slices directly, never gather global hidden to host.
 normalized=np.concatenate([owner[int(mesh.devices[0,f].id)]['boundary/normalized_mlp']
                            for f in range(4)],axis=1)
 replay,health=suffix(put(normalized,P(None,'feature')),
                      put(jnp.arange(128)<count),weights.layers[0].dense)
 jax.block_until_ready((replay,health)); same(out[0][0],replay); assert np.asarray(health).all()
 # Device packet feeds the real suffix directly; both result axes remain owned.
 owned=owner_suffix(packet['boundary/normalized_mlp'],put(jnp.int32(0)),
                    put(jnp.int32(count)),weights.layers[0].dense)
 jax.block_until_ready(owned)
 owned_rows=capture_owner_arrays(dict(output=owned[0],health=owned[1]),slots_by_device=slots)
 for device,fields in owned_rows.items():
  feature=slots[device]%4; width=config.geometry.hidden_size//4
  same(fields['output'],np.asarray(replay)[:,feature*width:(feature+1)*width])
  assert fields['health'].all()
 if count==128:
  wide_packet=packet['boundary/normalized_mlp']
  wide_rows=owned_rows
  for start in (0,32,64,96):
   cross=owner_suffix(wide_packet,put(jnp.int32(start)),put(jnp.int32(32)),weights.layers[0].dense)
   cross_rows=capture_owner_arrays(dict(output=cross[0],health=cross[1]),slots_by_device=slots)
   for device,fields in cross_rows.items():
    same(fields['output'][:32],wide_rows[device]['output'][start:start+32])
    assert np.count_nonzero(fields['output'][32:])==0 and fields['health'].all()
 if count==32:
  current=tuple(tuple(layer[2:5]) for layer in out)
 if offset==32:
  poisoned=observed(ids.at[32:].set(-2147483648),*inputs[1:])
  for a,b in zip(jax.tree.leaves((out,packet)),jax.tree.leaves(poisoned),strict=True):same(a,b)
 if count==128: current=caches
try:
 suffix(put(normalized[:32],P(None,'feature')),put(jnp.ones(32,bool)),weights.layers[0].dense)
 raise AssertionError('accepted physicalM32 suffix')
except ValueError:pass
for start,count in ((-1,32),(1,32),(128,32),(32,128),(0,0),(0,129)):
 invalid=owner_suffix(wide_packet,put(jnp.int32(start)),put(jnp.int32(count)),weights.layers[0].dense)
 assert not np.asarray(invalid[1]).any()
for start,count in ((jnp.float32(0),jnp.int32(32)),(jnp.int32(0),jnp.asarray([32],jnp.int32))):
 try:
  owner_suffix(wide_packet,put(start),put(count),weights.layers[0].dense)
  raise AssertionError('accepted invalid suffix metadata')
 except ValueError:pass
# Do not let equal synthetic expert replicas conceal owner-axis mistakes.
distinct=np.asarray(wide_packet).copy()
for expert in range(8): distinct[expert] *= np.asarray(expert+1,dtype=jnp.bfloat16)
owned=owner_suffix(put(distinct,P('expert','feature')),put(jnp.int32(0)),put(jnp.int32(128)),weights.layers[0].dense)
observed_owned=capture_owner_arrays(dict(output=owned[0],health=owned[1]),slots_by_device=slots)
from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_mlp_mapped
from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
# Dense weights are intermediate-sharded over expert: the original suffix has
# an expert8 down reduction. Retain ALL distinct expert operands simultaneously;
# separate replicated-input evaluations would be a different mathematical call.
def direct_body(packet,dense):
 value=ws32_prefill_mlp_mapped(packet[0,0],jnp.ones(128,bool),dense,None,
        moe_contract=config.moe_contract,linear_interpret=True,expert_panels=True)[0]
 return value[None,None]
direct=jax.jit(jax.shard_map(direct_body,mesh=mesh,
 in_specs=(P('expert','feature'),ws32_decoder_weight_specs(config).layers[0].dense),
 out_specs=P('expert','feature'),check_vma=False))
expected=direct(put(distinct,P('expert','feature')),weights.layers[0].dense)
same(owned[0],expected)
bad=observed(tokens,put(jnp.int32(128)),put(jnp.int32(32)),table,caches,
             weights.embedding_local,weights.layers[:2],wk[:2],rope)[0]
assert all(not np.asarray(layer[10]).any() for layer in bad)
print('DENSE_NORM_ORIGINALS_PACKETS_SUFFIX_PADDING_CPU32_PASS',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=420,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DENSE_NORM_ORIGINALS_PACKETS_SUFFIX_PADDING_CPU32_PASS" in result.stdout
