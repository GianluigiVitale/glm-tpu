"""Forced32 CPU MoE comparisons: no TPU or production-prefill claim."""

import json
import os
import subprocess
import sys


def test_grouped_moe_matches_row_path_with_skew_and_local_collectives():
    program = r"""
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32 import ws32_moe_pallas_from_routes_mapped
from glm_tpu.greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from scripts.greenfield.probe_ws32_prefill_moe import build_mapped
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
contract=GlmMoeNumericalContract(hidden_size=128,intermediate_size=32,num_experts=64,
                               top_k=8,stage_size=8,fp8_block_shape=(32,32))
rng=np.random.default_rng(771)
B=17
hidden=np.asarray(rng.normal(0,0.15,(B,128)),ml_dtypes.bfloat16)
hidden[4]=0
def bits(shape):
 return np.asarray(rng.normal(0,0.1,shape),ml_dtypes.float8_e4m3fn).view(np.uint8)
def scales(shape):
 return rng.uniform(0.25,1.5,shape).astype(np.float32)
weights=(bits((64,32,128)),scales((64,1,4)),bits((64,32,128)),scales((64,1,4)),
         bits((64,128,32)),scales((64,4,1)),bits((32,128)),scales((1,4)),
         bits((32,128)),scales((1,4)),bits((128,32)),scales((4,1)))
weights[1][3]=0
specs=(P(None,'feature'),P(),P(),P('expert',None,'feature'),P('expert',None,'feature'),
       P('expert',None,'feature'),P('expert',None,'feature'),P('expert','feature',None),
       P('expert','feature',None),P(None,'feature'),P(None,'feature'),P(None,'feature'),
       P(None,'feature'),P('feature',None),P('feature',None))
batch,one=build_mapped(mesh,contract=contract,interpret=True)
captured_batch,captured_one=build_mapped(mesh,contract=contract,interpret=True,capture_boundaries=True)
fp32_batch,_=build_mapped(mesh,contract=contract,interpret=True,fp32_route_sum=True)
fp32_capture,_=build_mapped(mesh,contract=contract,interpret=True,capture_boundaries=True,fp32_route_sum=True)
report={}
for case in ('distributed','one_owner'):
 if case=='distributed':
  routes=np.stack([rng.choice(64,8,replace=False) for _ in range(B)]).astype(np.int32)
 else:
  routes=np.stack([rng.permutation(np.arange(16,24)) for _ in range(B)]).astype(np.int32)
 rw=rng.uniform(0.01,1.0,(B,8)).astype(np.float32); rw/=rw.sum(axis=1,keepdims=True)
 rw[2,3]=0
 values=tuple(jax.device_put(v,NamedSharding(mesh,s)) for v,s in zip((hidden,routes,rw,*weights),specs))
 compiled=batch.lower(*values).compile()
 actual,health=compiled(*values)
 assert np.asarray(health).shape==(8,4) and np.asarray(health).all()
 reference=jnp.concatenate([one(values[0][i:i+1],values[1][i:i+1],values[2][i:i+1],*values[3:]) for i in range(B)])
 np.testing.assert_array_equal(np.asarray(actual).view(np.uint16),np.asarray(reference).view(np.uint16))
 captured_y,captured_health,cb=captured_batch(*values)
 cr=[captured_one(values[0][i:i+1],values[1][i:i+1],values[2][i:i+1],*values[3:]) for i in range(B)]
 np.testing.assert_array_equal(np.asarray(captured_y).view(np.uint16),np.asarray(actual).view(np.uint16))
 np.testing.assert_array_equal(np.asarray(jnp.concatenate([v[0] for v in cr])).view(np.uint16),np.asarray(reference).view(np.uint16))
 assert np.asarray(captured_health).all()
 fp32_y,fp32_health,fp32_cb=fp32_capture(*values)
 fp32_plain,fp32_plain_health=fp32_batch(*values)
 np.testing.assert_array_equal(np.asarray(fp32_y).view(np.uint16),np.asarray(fp32_plain).view(np.uint16))
 assert np.asarray(fp32_health).all() and np.asarray(fp32_plain_health).all()
 parts=np.asarray(fp32_cb['weighted_routes']).astype(np.float32)
 expected_sum=parts.sum(axis=3,dtype=np.float32)
 np.testing.assert_array_equal(np.asarray(fp32_cb['local_sum_operand']),expected_sum)
 assert np.asarray(fp32_cb['local_sum_operand']).dtype==np.float32
 for name,array in cb.items():
  expected=np.concatenate([np.asarray(v[1][name]) for v in cr],axis=2)
  observed=np.asarray(array)
  dtype=np.uint32 if observed.dtype==np.float32 else np.uint16
  np.testing.assert_array_equal(observed.view(dtype),expected.view(dtype),err_msg=name)
 hlo=parse_hlo_module(compiled.as_text())
 cs=[i for i in hlo.instructions if i.is_collective]
 feature=tuple(tuple(range(e*4,e*4+4)) for e in range(8))
 expert=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
 assert all(i.opcode=='all-reduce' and i.replica_groups in (feature,expert) for i in cs), [(i.opcode,i.replica_groups) for i in cs]
 # CPU XLA combines the independent routed/shared feature reductions into
 # one tuple all-reduce. Check the actual lowered count, not source calls.
 assert sorted(i.maximum_group_size for i in cs)==[4,8], [(i.opcode,i.maximum_group_size) for i in cs]
 report[case]={'exact':True,'groups':[4,8]}
 bad=list(values); bad[1]=bad[1].at[0,1].set(bad[1][0,0])
 _,health=batch(*bad)
 assert not np.asarray(health).any()
 for invalid_weight in (-1.0,np.nan,np.inf):
  bad=list(values); bad[2]=bad[2].at[0,0].set(invalid_weight)
  _,health=batch(*bad)
  assert not np.asarray(health).any()
print(json.dumps(report))
"""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "distributed": {"exact": True, "groups": [4, 8]},
        "one_owner": {"exact": True, "groups": [4, 8]},
    }
