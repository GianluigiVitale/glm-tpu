"""Executing CPU32 capture composition; not TPU numerical evidence."""

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.greenfield.prefill_window_boundary import compare_original_outputs
from scripts.greenfield.prefill_layer_numerical import FIELDS


@pytest.mark.parametrize("field", FIELDS)
def test_original_signature_report_never_silently_accepts_changed_field(field):
    original = {n: np.zeros((2, 2), np.float32) for n in FIELDS}
    observed = {n: v.copy() for n, v in original.items()}
    same = compare_original_outputs(observed, original)
    assert same["signature_reproduced"] and same["all_outputs_reproduced"]
    observed[field][0, 0] = 1
    report = compare_original_outputs(observed, original)
    assert not report["all_outputs_reproduced"]
    assert not report["fields"][field]["byte_identical"]
    assert not report["numerical_admission"] and not report["performance_claim"]
    if field in ("positions", "counts", "scores", "routes", "route_weights"):
        assert not report["signature_reproduced"]
    observed.pop(field)
    with pytest.raises(ValueError, match="all12"):
        compare_original_outputs(observed, original)


def test_window_boundary_captures_actual_operands_without_tracer_escape_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from scripts.greenfield.prefill_layer_programs import build_layer_programs
from scripts.greenfield.probe_ws32_prefill_layer import input_specs, device_inputs
from scripts.greenfield.prefill_window_protocol import control_inputs
from scripts.greenfield.prefill_window_boundary import capture_owner_arrays,build_same_input_router_program
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
# Deliberately nonidentity device placement; observation owners are mesh slots.
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wks=fixture(mesh); w=weights.layers[6]
wk=jax.device_put(wks[3],NamedSharding(mesh,P()))
specs=input_specs(w,wk)
opts=dict(full_indexer=True,sparse_mlp=True,key_tile=128,
    dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,
    moe_contract=config.moe_contract,linear_interpret=True,sparse_attention_interpret=True)
plain,_=build_layer_programs(mesh,specs,candidate_window=True,**opts)
capture,_=build_layer_programs(mesh,specs,candidate_window=True,capture_boundaries=True,**opts)
small,_=build_layer_programs(mesh,specs,capture_boundaries=True,**opts)
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
original=plain(*values)
captured,obs=capture(*values);jax.block_until_ready((captured,obs))
assert len(captured)==12 and len(obs)==4*(8+9+4)+8+4==96
for a,b in zip(original,captured):np.testing.assert_array_equal(a,b)
assert all(a.shape[:2]==(8,4) for a in obs.values())
assert 'tile32/dsa/query' in obs and 'router/partial' in obs
slots={int(d.id):i for i,d in enumerate(mesh.devices.flat)}
owners=capture_owner_arrays(obs,slots_by_device=slots)
assert set(owners)==set(slots)
for device,slot in slots.items():
    for key in obs:np.testing.assert_array_equal(owners[device][key],obs[key][slot//4,slot%4])
bad=dict(slots);keys=list(bad);bad[keys[0]],bad[keys[1]]=bad[keys[1]],bad[keys[0]]
try:capture_owner_arrays(obs,slots_by_device=bad)
except ValueError as e:assert 'physical slot' in str(e)
else:raise AssertionError('wrong physical mapping passed')
np.testing.assert_array_equal(obs['router_selection/indices'][0,0],captured[8])
np.testing.assert_array_equal(obs['router_selection/weights'][0,0],captured[9])
joined=np.concatenate([np.asarray(obs[f'tile{i}/attention_mlp_boundary/normalized_mlp']) for i in (0,32,64,96)],axis=2)
np.testing.assert_array_equal(joined,obs['router/input'])
np.testing.assert_array_equal(obs['router/input'],obs['router/clean'])
# Preserve completed BF16 data, then assemble feature-sharded replay inputs.
router=build_same_input_router_program(mesh)
router_input=jnp.concatenate([obs['router/input'][0,f] for f in range(4)],axis=1)
router_input=jax.device_put(router_input,NamedSharding(mesh,P(None,'feature')))
live=jax.device_put(jnp.ones(128,jnp.bool_),NamedSharding(mesh,P()))
replayed,replay_obs=router(router_input,w.moe.router_weight_local,w.moe.correction_bias_local,live)
jax.block_until_ready((replayed,replay_obs))
np.testing.assert_array_equal(replayed[0],captured[8])
np.testing.assert_array_equal(replayed[1],captured[9])
np.testing.assert_array_equal(replay_obs['router/input'],obs['router/input'])
for start in (0,32,64,96):
    narrow,_=router(router_input[start:start+32],w.moe.router_weight_local,w.moe.correction_bias_local,live[:32])
    np.testing.assert_array_equal(narrow[0],replayed[0][start:start+32])
    np.testing.assert_array_equal(narrow[1],replayed[1][start:start+32])
partial=np.asarray(obs['router/partial'])
np.testing.assert_allclose(np.sum(partial,axis=1),np.asarray(obs['router/local_logits'])[:,0],rtol=1e-5,atol=1e-6)
for start in (0,32,64,96):
    q=f'tile{start}/post_norm/'
    summed=np.asarray(obs[q+'update']).astype(np.float32)+np.asarray(obs[q+'residual']).astype(np.float32)
    np.testing.assert_array_equal(summed,obs[q+'summed'])
    np.testing.assert_array_equal(np.asarray(obs[q+'carried']),summed.astype('bfloat16'))
    sq=np.sum(np.asarray(obs[q+'local_square_sum']),axis=1)
    np.testing.assert_allclose(sq,np.asarray(obs[q+'square_sum'])[:,0],rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(obs[f'tile{start}/attention_mlp_boundary/normalized_mlp'],obs[q+'normalized'])
    np.testing.assert_array_equal(obs[f'tile{start}/dsa/causal_lengths'][0,0],np.arange(506+start,538+start))
    # Explicit page ordering, not physical cache flattening.
    for owner in range(8):
        np.testing.assert_array_equal(obs[f'tile{start}/dsa/logical_positions'][owner,0],
            (np.arange(3)[:,None]*512+owner*64+np.arange(64)[None,:]).reshape(-1))
previous=None
for tile in range(4):
    control,observations=small(*control_inputs(values,tile,previous))
    jax.block_until_ready((control,observations))
    assert len(observations)==33
    for name in ('dsa/query','dsa/head_weights','dsa/keys','dsa/current_keys',
                 'post_norm/summed','attention_mlp_boundary/normalized_mlp'):
        np.testing.assert_array_equal(obs[f'tile{tile*32}/{name}'],observations[name])
    previous=control
# Repeat same executable and retrace another builder: no escaped/stale collector.
again,again_obs=capture(*values)
for key in obs:np.testing.assert_array_equal(obs[key],again_obs[key])
shape=jax.eval_shape(build_layer_programs(mesh,specs,candidate_window=True,capture_boundaries=True,**opts)[0],*values)
assert set(shape[1])==set(obs)
assert all(shape[1][k].shape==obs[k].shape for k in obs)
# Padded poison stays observable where meaningful, but never reaches router math.
tail=list(values);tail[9]=jax.device_put(jnp.int32(33),NamedSharding(mesh,P()))
tail[0]=tail[0].at[33:].set(jnp.nan);tail[1]=tail[1].at[33:].set(jnp.nan)
tail_output,tail_obs=capture(*tail);jax.block_until_ready((tail_output,tail_obs))
assert np.asarray(tail_output[10]).all()
assert not np.asarray(tail_obs['router/live'][:,:,33:]).any()
assert not np.asarray(tail_obs['router/clean'][:,:,33:]).any()
assert not np.asarray(tail_output[9][33:]).any()
print('BOUNDARY_CAPTURE_CPU32_PASS',flush=True)
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
    assert "BOUNDARY_CAPTURE_CPU32_PASS" in result.stdout


def test_production_capture_schema_and_explicit_output_bytes_without_weights():
    code = r"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import prefill_window_protocol as p
from scripts.greenfield.prefill_window_boundary import build_boundary_programs
from scripts.greenfield.probe_ws32_prefill_layer import input_specs,device_inputs
from scripts.greenfield.run_short_decoder_ws32 import _geometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig,ws32_decoder_weight_names,_bind_weight_name_tree
from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
schema=json.loads(Path('tests/greenfield/hlo/fixtures/prefill_layer_schema.json').read_text())['tensor_schema']
config=Ws32DecoderConfig(_geometry(),p.CAPACITY)
names=ws32_decoder_weight_names(config).layers[6]
source={t['name']:t for t in schema};arrays={}
types={'BF16':jnp.bfloat16,'F32':jnp.float32,'U8':jnp.uint8}
for name in jax.tree.leaves(names):
    suffix=name.split('model.layers.6.',1)[1]
    t=source[f'model.layers.{0 if ".indexer." in name else 3}.{suffix}']
    arrays[name]=jax.ShapeDtypeStruct(tuple(t['global_shape']),types[t['dtype']],sharding=NamedSharding(mesh,P(*t['partition_spec'])))
w=_bind_weight_name_tree(names,arrays)
wk=jax.ShapeDtypeStruct((128,6144),jnp.float32,sharding=NamedSharding(mesh,P()))
specs=input_specs(w,wk)
host=p.host_case('boundary',build_rotary_table_host(p.CAPACITY,rotary_dim=64,theta=8e6))
values=device_inputs(host,specs,w,wk,mesh)
wide,small=build_boundary_programs(mesh,specs,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
for fn,args,rows,prefix in ((wide,values,128,'tile0/'),(small,p.control_inputs(values,0),32,'')):
    outputs,obs=jax.eval_shape(fn,*args)
    assert len(outputs)==12 and outputs[0].shape==(rows,6144)
    assert len(obs)==(96 if rows==128 else 33)
    assert obs['router/input'].shape==(8,4,rows,1536)
    assert obs['router/weight'].shape==(8,4,32,1536)
    assert obs['router/partial'].shape==(8,4,rows,32)
    assert obs['router/logits'].shape==(8,4,rows,256)
    assert obs[prefix+'dsa/query'].shape==(8,4,32,32,128)
    assert obs[prefix+'dsa/keys'].shape==(8,4,512,128)
    assert obs[prefix+'post_norm/summed'].shape==(8,4,32,1536)
    assert obs[prefix+'dsa/current_keys'].dtype==jnp.bfloat16
    assert obs[prefix+'post_norm/summed'].dtype==jnp.float32
    per_chip=sum(np.prod(v.shape[2:])*v.dtype.itemsize for v in obs.values())
    # Output payload only. Compiler temporaries and live-buffer/runtime peak
    # are additional and cannot be inferred from this shape calculation.
    assert per_chip<8*1024*1024
    print('CAPTURE_SCHEMA',rows,'bytes_per_chip',per_chip,flush=True)
print('BOUNDARY_PRODUCTION_SCHEMA_PASS',flush=True)
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
    print(result.stdout)
    assert "BOUNDARY_PRODUCTION_SCHEMA_PASS" in result.stdout
