"""Check exact cache/output behavior and the FP32 DSA-score rounding boundary."""
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('rowwise_dsa', [False, True], ids=['batched_dsa','rowwise_dsa'])
def test_batched_verifier_attention_cpu32(rowwise_dsa):
    code = r'''
import json,jax,jax.numpy as jnp,numpy as np
from jax import lax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_decoder as d
from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
from glm_tpu.perf.bf16_resident import bf16_resident_weights,bf16_weight_specs,attention_layer_bf16
from glm_tpu.perf.speculative_attention import attention_rows_bf16
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config,raw,_=fixture(mesh,panel_geometry=True)
weights=bf16_resident_weights(mesh,config,raw)
rng=np.random.default_rng(4501)
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
def bf(shape):return jnp.asarray(rng.normal(0,.3,shape),jnp.bfloat16)
rows=5
hidden=put(bf((rows,config.geometry.hidden_size)),P(None,'feature'))
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
kv=put(bf((3,512,config.attention_contract.packed_cache_width)),P(None,'expert'))
index=put(bf((3,512,config.dsa_contract.head_dim)),P(None,'expert'))
blocks=put(jnp.array([[2,0,1]],jnp.int32))
ids=put(jnp.tile(jnp.arange(128,dtype=jnp.int32),(rows,1)))
scores=put(jnp.zeros((rows,128),jnp.float32))
def body(h,kv,index,ids,counts,scores,pos,blocks,layer,rope):
    batched=attention_rows_bf16(h,h,kv,index,ids,counts,scores,pos,blocks,pos+1,
        layer,rope,config=config,interpret=True,rowwise_dsa=ROWWISE_DSA)
    def one(caches,values):
        i,x,sel,count,score=values
        p=pos+i
        r=attention_layer_bf16(x[None],*caches,sel[None],count[None],score[None],p,
            blocks,p+1,layer.qkv_a,layer.attention,layer.dsa,normalized=x[None],
            dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,
            cache_layout=config.cache_layout,sparse_attention_config=SparseMlaConfig(segment_block=128),
            sparse_attention_interpret=True,main_rope_table_row=rope[p[0]],
            dsa_two_stage=True,lse_attention=False)
        return (r.cache_local,r.index_cache_local),(r.output_local[0],r.selected_positions[0],
            r.selected_valid_counts[0],r.selected_scores[0],r.contract_valid[0])
    caches,values=lax.scan(one,(kv,index),(jnp.arange(rows,dtype=jnp.int32),h,ids,counts,scores))
    ordinary=type(batched)(values[0],*caches,*values[1:])
    return batched,ordinary
specs=(P(None,'feature'),P(None,'expert'),P(None,'expert'),P(),P(),P(),P(),P(),None,P())
out_specs=(P(None,'feature'),P(None,'expert'),P(None,'expert'),P(),P(),P(),P())
from glm_tpu.greenfield.kernels.ws32_layer import Ws32AttentionLayerResult
out_specs=Ws32AttentionLayerResult(*out_specs)
def same(a,b,label):
    for x,y in zip(a.addressable_shards,b.addressable_shards):
        np.testing.assert_array_equal(np.ascontiguousarray(x.data).view(np.uint8),
            np.ascontiguousarray(y.data).view(np.uint8),err_msg=label)
for layer_id in (0,3):
    ins=specs[:8]+(bf16_weight_specs(config).layers[layer_id],specs[9])
    fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=ins,out_specs=(out_specs,out_specs),check_vma=False))
    for start in (61,127,510):
        counts=put(jnp.minimum(jnp.arange(rows,dtype=jnp.int32)+start+1,128))
        valid_ids=put(jnp.where(jnp.arange(128)[None,:]<counts[:,None],ids,-1))
        args=(hidden,kv,index,valid_ids,counts,scores,put(jnp.array([start],jnp.int32)),blocks,weights.layers[layer_id],rope)
        actual,expected=fn(*args)
        assert np.asarray(actual.contract_valid).all()
        for name in actual._fields:
            if name == 'selected_scores':
                x,y=np.asarray(actual.selected_scores),np.asarray(expected.selected_scores)
                np.testing.assert_array_equal(np.isfinite(x),np.isfinite(y))
                live=np.isfinite(y)
                error=float(np.max(np.abs(x[live]-y[live]),initial=0))
                print(json.dumps(dict(layer=layer_id,start=start,dsa_score_max_abs=error)),flush=True)
                # Batched FP32 score contractions have a documented rounding
                # boundary; this fixture guard is not a global top-k proof.
                if ROWWISE_DSA:
                    same(actual.selected_scores,expected.selected_scores,'rowwise scores')
                else:
                    np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-6)
            else:
                same(getattr(actual,name),getattr(expected,name),f'{layer_id}/{start}/{name}')
        changed,_=fn(hidden.at[1:].set(-hidden[1:]),*args[1:])
        same(changed.output_local[:1],actual.output_local[:1],'future draft output')
        same(changed.selected_positions[:1],actual.selected_positions[:1],'future draft selection')
'''
    code = code.replace('ROWWISE_DSA', str(rowwise_dsa))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
