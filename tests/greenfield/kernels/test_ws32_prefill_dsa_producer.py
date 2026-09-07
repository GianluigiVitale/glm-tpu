"""CPU32 DSA producer/dual-cache admission, not TPU or decoder equivalence."""

import json
import os
import subprocess
import sys


def test_prefill_dsa_producer_keeps_causal_and_repair_state_separate():
    program = r"""
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_layer import Ws32DsaWeights, Ws32PreparedAttention
from glm_tpu.greenfield.kernels.ws32_prefill_dsa import ws32_prefill_dsa_inputs_mapped,ws32_prefill_dsa_mapped
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract,dsa_scores,exact_topk
from glm_tpu.greenfield.kernels.reference.prefill_index import repair_stage_local_prompt_index_cache
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng=np.random.default_rng(456)
def bf(shape):return jnp.asarray(rng.normal(0,.1,shape),jnp.bfloat16)
def bits(shape):return jnp.asarray(np.asarray(rng.normal(0,.03,shape),ml_dtypes.float8_e4m3fn).view(np.uint8))
def scale(shape):return jnp.asarray(rng.uniform(.5,1.5,shape),jnp.float32)
def put(v,s):return jax.device_put(v,NamedSharding(mesh,s))
def mapped(fn,ins,outs):return jax.jit(jax.shard_map(fn,mesh=mesh,in_specs=ins,out_specs=outs,check_vma=False))
R=17
contract=DsaNumericalContract(hidden_size=512,q_lora_rank=128,top_k=16)
n=put(bf((R,512)),P(None,'feature'))
prepared=Ws32PreparedAttention(n,n,put(bf((R,128)),P()),put(bf((R,576)),P()))
pspec=Ws32PreparedAttention(P(None,'feature'),P(None,'feature'),P(),P())
wspec=Ws32DsaWeights(P('expert',None),P('expert',None),P(None,'feature'),P(None,'feature'),P(),P(),P('expert','feature'))
weights=Ws32DsaWeights(bits((4096,128)),scale((32,1)),bits((128,512)),scale((1,4)),jnp.ones(128,jnp.bfloat16),bf((128,)),bf((32,512)))
weights=jax.tree.map(put,weights,wspec)
# The repair leaf is already materialized; deliberately different to make buffer
# contamination detectable, not to claim production weight provenance.
wk=put(bf((128,512)).astype(jnp.float32),P())
cache=put(bf((8,3,64,128)),P('expert',None,None,None))
repaired=put(jnp.full_like(cache,7),P('expert',None,None,None))
table=put(jnp.asarray([[2,0,1]],jnp.int32),P())
ins=(pspec,P('expert',None,None,None),P('expert',None,None,None),P(),P(),P(),wspec,P())
outs=(P('expert',None,None,None),P('expert',None,None,None),P(),P(),P(),P('expert','feature'))
def body(p,c,r,o,n,t,w,k):
    v=ws32_prefill_dsa_mapped(p,c[0],r[0],o,n,t,w,k,contract=contract,key_tile=128,linear_interpret=True)
    return v.unrepaired_index_cache[None],v.repaired_index_cache[None],v.selected_positions,v.selected_valid_counts,v.selected_scores,v.contract_valid[None,None]
fn=mapped(body,ins,outs)
def inputs_body(p,pos,live,w):
    v=ws32_prefill_dsa_inputs_mapped(p,pos,live,w,contract=contract,linear_interpret=True)
    return v.query,v.head_weights,v.keys,v.normalized_full,v.contract_valid[None,None]
inputs=mapped(inputs_body,(pspec,P(),P(),wspec),(P(),P(),P(),P(),P('expert','feature',None)))
def run(offset,count,p=prepared,r=repaired,c=cache,t=table,w=weights):
    return fn(p,c,r,jnp.int32(offset),jnp.int32(count),t,w,wk)
for offset,count in ((55,17),(505,17),(0,11)):
    out=run(offset,count)
    assert np.asarray(out[-1]).all()
    pos=jnp.arange(R,dtype=jnp.int32)+offset
    live=jnp.arange(R)<count
    produced=inputs(prepared,pos,live,weights)
    # Independent row geometry guards query/head gather transpose and input masking.
    singles=[inputs(jax.tree.map(lambda v:v[i:i+1],prepared),pos[i:i+1],live[i:i+1],weights) for i in range(R)]
    for idx in (0,2,3):np.testing.assert_array_equal(produced[idx],jnp.concatenate([v[idx] for v in singles]))
    # FP32 GEMM/GEMV head sums can differ by association. Judge BOTH against
    # independent FP64 arithmetic with a dimension-derived forward-error bound,
    # not a tolerance fitted to the observed difference. BF16 products fit F32.
    x=np.asarray(prepared.normalized_local,dtype=np.float64).copy()
    x[count:]=0
    hw=np.asarray(weights.head_weight_local,dtype=np.float64)
    math_head=(x @ hw.T) * (32**-.5)
    unit_roundoff=2.0**-24
    operation_count=contract.hidden_size+8
    gamma=operation_count*unit_roundoff/(1-operation_count*unit_roundoff)
    bound=gamma*(np.abs(x) @ np.abs(hw).T)*(32**-.5)
    for head_values in (produced[1],jnp.concatenate([v[1] for v in singles])):
        assert np.all(np.abs(np.asarray(head_values,dtype=np.float64)-math_head)<=bound)
    expected=np.asarray(cache).copy()
    for i in range(count):
        p=offset+i;expected[(p%512)//64,int(table[0,p//512]),p%64]=np.asarray(produced[2][i])
    np.testing.assert_array_equal(out[0],expected)
    # CPU full score row from logical UNREPAIRED cache, per-query causal limits.
    logical=jnp.stack([out[0][:,int(page)] for page in np.asarray(table[0])]).reshape(1536,128)
    scores=dsa_scores(produced[0],logical,produced[1],precision='default')
    lengths=jnp.where(live,pos+1,0)
    expected_selection=exact_topk(scores,lengths,top_k=16)
    np.testing.assert_array_equal(out[2],expected_selection.positions)
    np.testing.assert_array_equal(out[3],expected_selection.valid_counts)
    chosen=jnp.take_along_axis(scores,jnp.maximum(out[2],0),axis=1)
    expected_scores=jnp.where(jnp.arange(16)[None]<out[3][:,None],chosen,-jnp.inf)
    np.testing.assert_array_equal(out[4],expected_scores)
    # Existing M64 reference reconstructs every retained repair row identically.
    reference=[]
    for owner in range(8):
        reference.append(repair_stage_local_prompt_index_cache(
            repaired[owner],produced[3],table,wk,weights.key_norm_weight,weights.key_norm_bias,jnp.int32(owner),
            contract=contract,local_rows_per_page=64,local_parallel_size=8,prompt_chunk=64,
            position_offset=offset,valid_rows=count))
    np.testing.assert_array_equal(out[1],jnp.stack(reference))
    # Arbitrarily different repaired history cannot influence prompt selections.
    other=run(offset,count,r=jnp.full_like(repaired,-13))
    for idx in (0,2,3,4):np.testing.assert_array_equal(out[idx],other[idx])

# Later live inputs cannot influence earlier queries or key prefixes.
base=run(55,17)
future=jax.tree.map(lambda v:v.at[-1].set(3),prepared)
altered=run(55,17,p=future)
for idx in (2,3,4):np.testing.assert_array_equal(base[idx][:-1],altered[idx][:-1])
# Empty and padded rows ignore NaNs, including non-used projection inputs.
poison=jax.tree.map(lambda v:v.at[11:].set(jnp.nan),prepared)
partial=run(0,11,p=poison)
assert np.asarray(partial[-1]).all()
for idx in range(5):np.testing.assert_array_equal(partial[idx],run(0,11)[idx])
empty=run(0,0,p=jax.tree.map(lambda v:jnp.full_like(v,jnp.nan),prepared))
assert np.asarray(empty[-1]).all()
np.testing.assert_array_equal(empty[0],cache);np.testing.assert_array_equal(empty[1],repaired)
assert np.all(np.asarray(empty[2])==-1) and np.all(np.asarray(empty[3])==0)
# Invalid offset/page mappings leave BOTH complete caches untouched.
for offset,count,t in ((2147483647,17,table),(-2147483648,17,table),(0,2147483647,table),(505,17,table.at[0,1].set(2))):
    bad=run(offset,count,t=t)
    assert not np.asarray(bad[-1]).any()
    np.testing.assert_array_equal(bad[0],cache);np.testing.assert_array_equal(bad[1],repaired)
# Historical NaNs and live normalized/query poison must fail health.
assert not np.asarray(run(55,17,c=cache.at[0,2,40,0].set(jnp.nan))[-1]).all()
for field in ('normalized_local','q_residual'):
    bad=prepared._replace(**{field:getattr(prepared,field).at[0,0].set(jnp.nan)})
    assert not np.asarray(run(55,17,p=bad)[-1]).all()
# Zero heads create exact all-score ties; lowest positions must win for every row.
tie=run(55,17,w=weights._replace(head_weight_local=jnp.zeros_like(weights.head_weight_local)))
np.testing.assert_array_equal(tie[2],np.broadcast_to(np.arange(16),(R,16)))
compiled=fn.lower(prepared,cache,repaired,jnp.int32(55),jnp.int32(17),table,weights,wk).compile()
ops=[x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
fg=tuple(tuple(range(e*4,e*4+4)) for e in range(8))
eg=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert all(x.replica_groups in (fg,eg) for x in ops)
assert sum(x.opcode=='all-gather' and x.replica_groups==eg for x in ops)==3
assert sum(x.opcode=='all-gather' and x.replica_groups==fg for x in ops)==1
reductions=[x for x in ops if x.opcode=='all-reduce']
assert len(reductions)==1 and reductions[0].replica_groups==fg
assert sorted((s.dtype,s.dimensions) for s in reductions[0].result_shapes)==[('f32',(17,4)),('f32',(17,128))]
print(json.dumps({'rows':R,'cases':3,'groups':sorted(x.maximum_group_size for x in ops),'causal_dual_cache':True}))
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
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["rows"] == 17 and report["cases"] == 3 and report["causal_dual_cache"]
    assert report["groups"] == [4, 4, 8, 8, 8]
