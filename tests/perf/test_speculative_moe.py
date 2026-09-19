"""Compare pooled expert work with independent ordinary one-row MoE calls."""
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('small_expert_tiles', [False, True], ids=['prefill_panels','m8_tiles'])
def test_canonical_moe_routes_cpu32(small_expert_tiles):
    code = r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.perf.bf16_resident import bf16_resident_weights,bf16_weight_specs,dense_bf16
from glm_tpu.perf.speculative_moe import moe_rows_bf16,dense_rows_bf16
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig,ws32_moe_grouped_routes_mapped
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config,raw,_=fixture(mesh,panel_geometry=True)
weights=bf16_resident_weights(mesh,config,raw)
contract=config.moe_contract
rng=np.random.default_rng(813)
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
def same(a,b):
    # Inspect every expert replica as well as all feature shards.
    for x,y in zip(a.addressable_shards,b.addressable_shards):
        np.testing.assert_array_equal(np.ascontiguousarray(x.data).view(np.uint8),
                                      np.ascontiguousarray(y.data).view(np.uint8))
def body(x,ids,rw,m):
    ordinary=jnp.concatenate([ws32_moe_grouped_routes_mapped(x[i:i+1],ids[i:i+1],rw[i:i+1],
        *m[2:8],None,None,None,None,None,None,contract=contract,
        config=RoutedProjectionConfig(output_tile=128,contraction_tile=128),interpret=True,
        shared_bf16=(m.shared_gate_local,m.shared_up_local,m.shared_down_local))
        for i in range(x.shape[0])])
    actual,ok=moe_rows_bf16(x,ids,rw,m,contract=contract,interpret=True,
                          small_expert_tiles=SMALL_EXPERT_TILES)
    return ordinary,actual,ok[None,None]
fn=jax.jit(jax.shard_map(body,mesh=mesh,
    in_specs=(P(None,'feature'),P(),P(),bf16_weight_specs(config).layers[3].moe),
    out_specs=(P(None,'feature'),P(None,'feature'),P('expert','feature')),check_vma=False))
for rows in (2,3,5):
    x=put(jnp.asarray(rng.normal(size=(rows,config.geometry.hidden_size)),jnp.bfloat16),P(None,'feature'))
    rw=rng.uniform(.05,1,(rows,contract.top_k)).astype(np.float32)
    rw=put(rw/rw.sum(axis=1,keepdims=True))
    for concentrated in (True,False):
        # Concentrated routes leave seven owners empty and reuse each weight
        # panel across rows. Spread routes exercise restore permutations.
        ids=np.tile(np.arange(contract.top_k,dtype=np.int32),(rows,1))
        if not concentrated:
            ids=(np.arange(rows,dtype=np.int32)[:,None]+ids*contract.local_experts)%contract.num_experts
            ids=ids[:,::-1].copy()
        expected,actual,ok=fn(x,put(ids),rw,weights.layers[3].moe)
        assert np.asarray(ok).all()
        same(actual,expected)
    # Invalid metadata must fail on every owner, not just the busy owner.
    for invalid in (put(ids).at[0,0].set(-1),put(ids).at[0,0].set(contract.num_experts),
                    put(ids).at[0,0].set(ids[0,1])):
        _,_,ok=fn(x,invalid,rw,weights.layers[3].moe)
        assert not np.asarray(ok).any()
    for value in (-.1,float('nan'),float('inf')):
        _,_,ok=fn(x,put(ids),rw.at[0,0].set(value),weights.layers[3].moe)
        assert not np.asarray(ok).any()
def dense_body(x,w):
    return (jnp.concatenate([dense_bf16(x[i:i+1],w,expert_axis='expert',feature_axis='feature')
                            for i in range(x.shape[0])]),dense_rows_bf16(x,w))
dense=jax.jit(jax.shard_map(dense_body,mesh=mesh,
    in_specs=(P(None,'feature'),bf16_weight_specs(config).layers[0].dense),
    out_specs=(P(None,'feature'),P(None,'feature')),check_vma=False))
same(*dense(x,weights.layers[0].dense))
'''
    code = code.replace('SMALL_EXPERT_TILES', str(small_expert_tiles))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
