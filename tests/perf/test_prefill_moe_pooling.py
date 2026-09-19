"""Pool routed rows across B128 boundaries without changing MoE arithmetic.

This exercises only the already-existing from-routes suffix. It does not raise
the full prefill builder's row limit or certify attention/dense placement.
"""
import os
import subprocess
import sys


def test_pooled_prefill_moe_cpu32():
    code = r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
from glm_tpu.perf.function_bindings import bind_dependencies
from glm_tpu.perf.prefill_bf16 import resident_matmul,resident_matmul_f32
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
contract=GlmMoeNumericalContract(hidden_size=1024,intermediate_size=256,num_experts=64,top_k=4,stage_size=8)
rng=np.random.default_rng(9521)
def put(value,spec=P()):return jax.device_put(value,NamedSharding(mesh,spec))
def bits(shape):return jax.lax.bitcast_convert_type(jnp.asarray(rng.normal(0,.025,shape),jnp.float8_e4m3fn),jnp.uint8)
gs,ds=P('expert',None,'feature'),P('expert','feature',None)
tables=(put(bits((64,256,1024)),gs),put(jnp.asarray(rng.uniform(.2,1.5,(64,2,8)),jnp.float32),gs),
        put(bits((64,256,1024)),gs),put(jnp.asarray(rng.uniform(.2,1.5,(64,2,8)),jnp.float32),gs),
        put(bits((64,1024,256)),ds),put(jnp.asarray(rng.uniform(.2,1.5,(64,8,2)),jnp.float32),ds),
        put(jnp.asarray(rng.normal(0,.025,(256,1024)),jnp.bfloat16),P(None,'feature')),None,
        put(jnp.asarray(rng.normal(0,.025,(256,1024)),jnp.bfloat16),P(None,'feature')),None,
        put(jnp.asarray(rng.normal(0,.025,(1024,256)),jnp.bfloat16),P('feature')),None)
specs=(gs,gs,gs,gs,ds,ds,P(None,'feature'),None,P(None,'feature'),None,P('feature'),None)
moe=bind_dependencies(ws32_prefill_moe_from_routes_mapped,
    fp8_block_matmul_f32=resident_matmul_f32,fp8_block_matmul=resident_matmul)
def body(x,ids,weights,tables):
    return moe(x,ids,weights,*tables,contract=contract,expert_panels=True,interpret=True)
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,'feature'),P(),P(),specs),
                        out_specs=(P(None,'feature'),P()),check_vma=False))
for rows,concentrated in ((512,False),(498,False),(1024,False),(512,True),(498,True),(1024,True)):
    x=put(jnp.asarray(rng.normal(0,.2,(rows,1024)),jnp.bfloat16),P(None,'feature'))
    ids=np.tile(np.arange(4,dtype=np.int32),(rows,1))
    if not concentrated:ids=(np.arange(rows,dtype=np.int32)[:,None]+np.arange(4,dtype=np.int32)[None,:]*16)%64
    ids=put(ids)
    weights=rng.uniform(.1,1.,(rows,4)).astype(np.float32)
    weights=put(weights/weights.sum(axis=1,keepdims=True))
    pooled,valid=fn(x,ids,weights,tables)
    parts=[]
    for start in range(0,rows,128):
        value,ok=fn(x[start:start+128],ids[start:start+128],weights[start:start+128],tables)
        assert bool(np.asarray(ok));parts.append(np.asarray(value))
    assert bool(np.asarray(valid)) and np.isfinite(np.asarray(pooled)).all()
    expected=np.concatenate(parts)
    np.testing.assert_array_equal(np.ascontiguousarray(pooled).view(np.uint8),np.ascontiguousarray(expected).view(np.uint8))
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
