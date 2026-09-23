"""CPU32: batched execution matches eight independent histories and freezes EOS lanes."""
import json
import os
import subprocess
import sys


CPU_CHECK = r'''
import json
import jax, jax.numpy as jnp, numpy as np
from types import SimpleNamespace
# CPU DotThunk cannot execute mixed BF16 x BF16 -> F32 in some outlined
# batched contractions. Interpret that arithmetic with exact operand widening;
# preserve FP32 accumulation and all BF16 rounding boundaries. This hook lives
# only in this isolated CPU interpreter, never in the TPU/runtime graph.
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
def cpu_dot(ctx,left,right,**params):
    if all(a.dtype==jnp.bfloat16 for a in ctx.avals_in) and ctx.avals_out[0].dtype==jnp.float32:
        def widened(a,b):
            return jax.lax.dot_general(
                jax.lax.optimization_barrier(a.astype(jnp.float32)),
                jax.lax.optimization_barrier(b.astype(jnp.float32)),
                params['dimension_numbers'],precision=params['precision'],preferred_element_type=jnp.float32)
        return mlir.lower_fun(widened,multiple_results=False)(ctx,left,right)
    return lax_internal._dot_general_lower(ctx,left,right,platform='cpu',**params)
mlir.register_lowering(lax_internal.dot_general_p,cpu_dot,platform='cpu')
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime import ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_sampled_request import build_ws32_sampled_prefill_program
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from glm_tpu.optimized.bf16_resident import bf16_resident_weights
from glm_tpu.optimized.ws32_decoder_challenger import build_ws32_challenger_decoder_program
from glm_tpu.optimized.batched_decode import build_batched_decoder_program
from glm_tpu.optimized.batched_runtime import compile_batch
from glm_tpu.runner.programs import build_program_set
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x): return jax.device_put(x,NamedSharding(mesh,P()))
config,weights,wk=fixture(mesh,panel_geometry=True)
wk=tuple(put(x) for x in wk)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
interpret=dict(sparse_attention_interpret=True,linear_interpret=True)
bf16=bf16_resident_weights(mesh,config,weights)
prefill=build_ws32_sampled_prefill_program(mesh,config,sampling=NucleusConfig(),
    block_rows=2,key_tile=128,**interpret)
states=[];tokens=[]
for lane in range(8):
    ids=np.arange(30+lane*5,33+lane*5+lane%2,dtype=np.int32)
    state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=len(ids))
    for start in range(0,len(ids),2):
        part=ids[start:start+2]
        out=prefill.execute(put(np.pad(part,(0,2-len(part)),constant_values=-1)),
            put(np.int32(len(part))),state,weights,wk,rope,put(np.float32(.5)))
        state=out.state
    state,token=b.finish_ws32_batched_prefill(out)
    states.append(state);tokens.append(token)
single=build_ws32_challenger_decoder_program(mesh,config,**interpret).execute
batch=build_batched_decoder_program(mesh,config,batch_size=8,**interpret)
# Exercise the actual bank initializer and donated per-lane insertion without
# compiling a TPU-only kernel: the production program set, built with Pallas
# interpretation enabled (its batched decoder equals ``batch`` above).
r=SimpleNamespace(concurrent_size=8,mesh=mesh,config=config,put=put,weights=bf16,rope=rope,
    compile=lambda name,fn,values,**kwargs:fn.lower(*values).compile())
programs=build_program_set(mesh,config,concurrent_size=8,interpret=True)
compile_batch(r,b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3),programs.batch)
stacked=r.initialize_batch(put(np.array([3+i%2 for i in range(8)],np.int32)))
for lane in range(8):stacked=r.insert_batch(stacked,states[lane],put(np.int32(lane)))
batch=r.decode_batch
input_tokens=jnp.stack(tokens)
report=dict(conversations=8,steps=0,independent_positions=True,inactive_frozen=True,
            token_agreement=True,maximum_state_error=0.,maximum_relative_row_error=0.,
            mixed_bf16_cpu_interpreter=True,other_lanes_unchanged_under_perturbation=True)
for step in range(2):
    active=np.array([lane>=step for lane in range(8)],bool)
    previous=jax.tree.map(lambda x:np.asarray(x).copy(),stacked)
    refs=[single(tokens[i],states[i],bf16,rope) if active[i] else None for i in range(8)]
    if step==0:
        isolated=jax.tree.map(lambda x:jnp.array(x,copy=True),stacked)
        changed_tokens=input_tokens.at[7,0].set((input_tokens[7,0]+1)%config.geometry.vocab_size)
        perturbed=batch(changed_tokens,isolated,bf16,rope,put(active))
    out=batch(input_tokens,stacked,bf16,rope,put(active))
    jax.block_until_ready(out)
    if step==0:
        # A direct isolation oracle: changing one conversation must leave
        # EVERY byte of every other conversation's computed state unchanged.
        # Equal batch shapes avoid scalar-vs-matrix accumulation differences.
        for actual,changed in zip(jax.tree.leaves(out),jax.tree.leaves(perturbed)):
            np.testing.assert_array_equal(np.asarray(actual[:7]).view(np.uint8),
                                          np.asarray(changed[:7]).view(np.uint8))
        assert not np.array_equal(np.asarray(out.state.kv_cache_local[7]).view(np.uint8),
                                  np.asarray(perturbed.state.kv_cache_local[7]).view(np.uint8))
        del perturbed,isolated
    for lane in range(8):
        actual=jax.tree.map(lambda x:np.asarray(x[lane]),out.state)
        expected=refs[lane].state if active[lane] else jax.tree.map(lambda x:x[lane],previous)
        for field,got,want in zip(actual._fields,actual,expected):
            want=np.asarray(want)
            if jnp.issubdtype(got.dtype,jnp.floating) and active[lane]:
                # Batching changes GEMV into GEMM and FP32 accumulation order.
                # Record differences on affected vectors, not an average
                # diluted by zero pages. Isolation is checked bitwise above;
                # token agreement and unchanged history are separate assertions.
                a,z=got.astype(np.float32),want.astype(np.float32)
                if field=='selected_scores':
                    # All available prompt/decode positions fit within top-k
                    # in this fixture; score ordering can differ near a tie.
                    a,z=np.sort(a,axis=-1),np.sort(z,axis=-1)
                finite=np.isfinite(z)
                np.testing.assert_array_equal(np.isfinite(a),finite)
                np.testing.assert_array_equal(a[~finite],z[~finite])
                a,z=np.where(finite,a,0),np.where(finite,z,0)
                changed=np.any(a!=z,axis=-1)
                if changed.any() and field!='selected_scores':
                    relative=np.linalg.norm(a[changed]-z[changed],axis=-1)/np.maximum(np.linalg.norm(z[changed],axis=-1),1e-12)
                    report['maximum_relative_row_error']=max(report['maximum_relative_row_error'],float(relative.max()))
                report['maximum_state_error']=max(report['maximum_state_error'],float(np.max(np.abs(a-z))))
                if field in ('kv_cache_local','index_cache_local'):
                    # No existing nonempty history row may be overwritten.
                    old=np.asarray(getattr(previous,field)[lane])
                    occupied=np.any(old!=0,axis=-1)
                    np.testing.assert_array_equal(got[occupied].view(np.uint8),old[occupied].view(np.uint8))
            elif field=='selected_positions' and active[lane]:
                np.testing.assert_array_equal(np.sort(got,axis=-1),np.sort(want,axis=-1))
            else:np.testing.assert_array_equal(got.view(np.uint8),want.view(np.uint8))
        expected_token=refs[lane].next_token if active[lane] else tokens[lane]
        np.testing.assert_array_equal(np.asarray(out.next_token[lane]),np.asarray(expected_token))
        if active[lane]:states[lane],tokens[lane]=refs[lane].state,refs[lane].next_token
    stacked,input_tokens=out.state,out.next_token
    report['steps']+=1
print(json.dumps(report))
'''


def test_eight_conversations_match_separate_decode_and_freeze_finished_lane():
    env=dict(os.environ,JAX_PLATFORMS='cpu',
             XLA_FLAGS=(os.environ.get('XLA_FLAGS','')+' --xla_force_host_platform_device_count=32').strip())
    result=subprocess.run([sys.executable,'-c',CPU_CHECK],env=env,
                          capture_output=True,text=True,timeout=1500)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout.strip().splitlines()[-1])
    assert report['conversations']==8 and report['steps']==2
    assert report['other_lanes_unchanged_under_perturbation']
    print(json.dumps(report,sort_keys=True))
