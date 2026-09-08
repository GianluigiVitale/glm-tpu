"""Completed prefix/suffix CPU32 composition, not protected TPU evidence."""

import os
import subprocess
import sys


def test_completed_prefix_window_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from scripts.greenfield.prefill_layer_programs import build_layer_programs
from scripts.greenfield.probe_ws32_prefill_layer import input_specs,device_inputs
from scripts.greenfield.prefill_completed_window import build_completed_window_programs,prefix_inputs,suffix_inputs,assemble_result
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wks=fixture(mesh);w=weights.layers[6]
wk=jax.device_put(wks[3],NamedSharding(mesh,P()))
specs=input_specs(w,wk)
opts=dict(full_indexer=True,sparse_mlp=True,key_tile=128,
 dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,
 moe_contract=config.moe_contract,linear_interpret=True,sparse_attention_interpret=True)
prefix,suffix=build_completed_window_programs(mesh,specs,**opts)
old,_=build_layer_programs(mesh,specs,**opts)
rng=np.random.RandomState(77)
host={
 'update':np.asarray(rng.randn(128,512)*.01,dtype='bfloat16'),
 'residual':np.asarray(rng.randn(128,512)*.01,dtype='bfloat16'),
 'kv':np.asarray(rng.randn(8,3,64,640)*.01,dtype='bfloat16'),
 'index':np.asarray(rng.randn(8,3,64,128)*.01,dtype='bfloat16'),
 'repair':np.full((8,3,64,128),7,dtype='bfloat16'),
 'positions':np.full((128,128),-1,np.int32),'counts':np.zeros(128,np.int32),
 'scores':np.full((128,128),-np.inf,np.float32),'offset':np.asarray(505,np.int32),
 'count':np.asarray(128,np.int32),'table':np.asarray([[2,0,1]],np.int32),
 'health':np.ones((8,4,128),np.bool_),
 'rope':build_ws32_main_rope_table(config)[505:633],
}
values=device_inputs(host,specs,w,wk,mesh)
initial_caches=[np.asarray(v).copy() for v in values[2:5]]
prefixes=[];narrow=[]
for tile in range(4):
 inputs=prefix_inputs(values,tile,prefixes[-1] if prefixes else None)
 p=prefix(*inputs);jax.block_until_ready(p)
 assert len(p)==10 and p[0].dtype==jnp.bfloat16
 prefixes.append(p)
 s=suffix(*suffix_inputs([p],jnp.int32(32),w.dense,w.moe));jax.block_until_ready(s)
 narrow.append(s)
 # Existing fused B32 reference on the same prior cache/input, CPU-only.
 o=old(*inputs);jax.block_until_ready(o)
 for left,right in ((s[0],o[0]),(s[1],o[8]),(s[2],o[9]),(s[3],o[10]),
                    (p[1],o[1]),(p[5],o[5]),(p[7],o[7])):
  np.testing.assert_array_equal(left,right)
 for i in (2,3,4):np.testing.assert_array_equal(p[i],o[i])
wide_args=suffix_inputs(prefixes,jnp.int32(128),w.dense,w.moe)
wide=suffix(*wide_args);jax.block_until_ready(wide)
for i in range(4):np.testing.assert_array_equal(wide[i],jnp.concatenate([s[i] for s in narrow],axis=2 if i==3 else 0))
result=assemble_result(prefixes,wide);assert len(result)==12
assert result[0].shape==(128,512) and result[10].shape==(8,4,128)
for i in (2,3,4):assert result[i] is prefixes[-1][i]
for a,b in zip(initial_caches,values[2:5]):np.testing.assert_array_equal(a,b)
# Same executable on partial final tile; poison never leaks into active rows.
tail=list(values);tail[9]=jnp.int32(33)
tail[0]=tail[0].at[33:].set(jnp.nan);tail[1]=tail[1].at[33:].set(jnp.nan)
pt=[]
for tile in range(4):
 p=prefix(*prefix_inputs(tuple(tail),tile,pt[-1] if pt else None));jax.block_until_ready(p);pt.append(p)
st=suffix(*suffix_inputs(pt,jnp.int32(33),w.dense,w.moe));jax.block_until_ready(st)
assert np.asarray(st[3]).all() and not np.asarray(st[0][33:]).any() and not np.asarray(st[2][33:]).any()
np.testing.assert_array_equal(st[0][:33],wide[0][:33])
for i in (2,3,4):np.testing.assert_array_equal(pt[-1][i],pt[1][i])
# Prefix health reaches suffix and never becomes a committed state here.
bad=list(wide_args);bad[4]=bad[4].at[3,2,9].set(False)
bad_out=suffix(*bad);jax.block_until_ready(bad_out)
assert not bool(bad_out[3][3,2,9])
bad=list(values);bad[18]=bad[18].at[3,2,0].set(False)
bad_prefix=prefix(*prefix_inputs(tuple(bad),0));jax.block_until_ready(bad_prefix)
assert not bool(bad_prefix[8][3,2,0])
# Malformed span refuses health, never wraps INTMAX into another valid cache row.
bad=list(values);bad[8]=jnp.int32(2147483647)
bad_prefix=prefix(*prefix_inputs(tuple(bad),0));jax.block_until_ready(bad_prefix)
assert not np.asarray(bad_prefix[8]).any()
try:prefix_inputs(values,1,None)
except ValueError:pass
else:raise AssertionError('missing causal carry accepted')
try:build_completed_window_programs(mesh,specs,**opts,capture_boundaries=True)
except ValueError:pass
else:raise AssertionError('observation mode accepted')
for count in (jnp.float32(1.5),jnp.bool_(True),jnp.asarray([1],jnp.int32),1):
 try:suffix_inputs(prefixes,count,w.dense,w.moe)
 except ValueError:pass
 else:raise AssertionError('malformed suffix count accepted')
for count in (jnp.int32(-1),jnp.int32(129)):
 args=suffix_inputs(prefixes,count,w.dense,w.moe)
 assert not np.asarray(args[-1]).any()
print('COMPLETED_WINDOW_CPU32_PASS',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETED_WINDOW_CPU32_PASS" in result.stdout


def test_completed_window_production_shapes_without_payloads():
    code = r"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.prefill_window_acquisition import prepare_programs
from scripts.greenfield.prefill_completed_window import build_completed_window_programs
from scripts.greenfield.probe_ws32_prefill_layer import input_specs
from scripts.greenfield.run_short_decoder_ws32 import _geometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig,ws32_decoder_weight_names,_bind_weight_name_tree
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
schema=json.loads(Path('tests/greenfield/hlo/fixtures/prefill_layer_schema.json').read_text())['tensor_schema']
config=Ws32DecoderConfig(_geometry(),4096)
names=ws32_decoder_weight_names(config).layers[6]
source={t['name']:t for t in schema};arrays={}
types={'BF16':jnp.bfloat16,'F32':jnp.float32,'U8':jnp.uint8}
for name in jax.tree.leaves(names):
 suffix=name.split('model.layers.6.',1)[1]
 t=source[f'model.layers.{0 if ".indexer." in name else 3}.{suffix}']
 arrays[name]=jax.ShapeDtypeStruct(tuple(t['global_shape']),types[t['dtype']],sharding=NamedSharding(mesh,P(*t['partition_spec'])))
w=_bind_weight_name_tree(names,arrays)
programs=prepare_programs(mesh=mesh,config=config,weights=w)
args=programs[-1][2]
prefix,suffix=build_completed_window_programs(mesh,input_specs(w,args[14]),full_indexer=True,sparse_mlp=True,key_tile=512,
 dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
p=jax.eval_shape(prefix,*args)
assert len(p)==10 and p[0].shape==(32,6144) and p[8].shape==(8,4,32)
assert p[2].shape==(8,8,64,640) and p[3].shape==(8,8,64,128)
for rows in (32,128):
 h=jax.ShapeDtypeStruct((rows,6144),jnp.bfloat16,sharding=NamedSharding(mesh,P(None,'feature')))
 live=jax.ShapeDtypeStruct((rows,),jnp.bool_,sharding=NamedSharding(mesh,P()))
 valid=jax.ShapeDtypeStruct((8,4,rows),jnp.bool_,sharding=NamedSharding(mesh,P('expert','feature',None)))
 s=jax.eval_shape(suffix,h,live,w.dense,w.moe,valid)
 assert len(s)==4 and s[0].shape==(rows,6144) and s[1].shape==(rows,8) and s[3].shape==(8,4,rows)
completed=prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True)
assert tuple(n for n,_,_ in completed)==('wk_decode','wk_promote','prefix','candidate','control')
assert completed[3][1] is completed[4][1]
for name,fn,args in completed:
 for leaf in jax.tree.leaves(args):assert isinstance(leaf,jax.ShapeDtypeStruct)
 out=jax.eval_shape(fn,*args)
 if name=='prefix':assert len(out)==10 and out[0].shape==(32,6144)
 if name in ('candidate','control'):
  rows=128 if name=='candidate' else 32
  assert len(out)==4 and out[0].shape==(rows,6144) and out[3].shape==(8,4,rows)
try:prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True,capture_boundaries=True)
except ValueError:pass
else:raise AssertionError('mixed acquisition mode accepted')
print('COMPLETED_WINDOW_PRODUCTION_SCHEMA_PASS',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETED_WINDOW_PRODUCTION_SCHEMA_PASS" in result.stdout
