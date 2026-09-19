"""The layerwise probe follows the complete decoder's one-step semantics."""
import os
import subprocess
import sys


def test_greedy_trail_advances_its_own_state_and_resets_between_variants():
    from typing import NamedTuple
    import jax.numpy as jnp
    import numpy as np
    from glm_tpu.perf.decode_diagnostics import collect_greedy_trail
    class State(NamedTuple):
        position: object
        contract_valid: object
    class Result(NamedTuple):
        state: State
        next_token: object
        final_residual_local: object
    initial=State(jnp.int32(0),jnp.array([True]))
    token=jnp.array([0],jnp.int32)
    for increment,healthy in ((1,True),(2,False)):
        calls=[]
        def step(index,t,state):
            calls.append(index)
            assert int(state.position)==index-1
            return Result(State(state.position+1,jnp.array([healthy])),t+increment,jnp.ones((1,4),jnp.bfloat16))
        values,report=collect_greedy_trail(step,token,initial,np.arange(29,dtype=np.int32),
                                         cache_check=lambda s:bool(s.position==28))
        np.testing.assert_array_equal(values,np.arange(29,dtype=np.int32)*increment)
        assert calls==list(range(1,29)) and report['steps']==28
        assert report['healthy'] is healthy and report['all_steps_finite']
        assert report['final_cache_finite']
        assert report['token_comparison']['all_equal'] is (increment==1)
        assert int(initial.position)==0 and int(token[0])==0


def test_layerwise_diagnostic_cpu32():
    code = r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_batched_prefill as pre,ws32_decoder as dec
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions,build_ws32_challenger_decoder_program
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
from glm_tpu.perf.decode_diagnostics import diagnose_layerwise
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x):return jax.device_put(x,NamedSharding(mesh,P()))
config,raw,wk=fixture(mesh)
weights=bf16_resident_weights(mesh,config,raw)
state=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=3).decoder
rope=put(jnp.asarray(dec.build_ws32_main_rope_table(config),jnp.bfloat16))
token=put(jnp.array([30],jnp.int32))
options=Ws32PerfOptions(sampler='greedy',bf16_resident=True,lse_attention=True,dsa_two_stage=True,
    routed_projection=RoutedProjectionConfig(output_tile=128,contraction_tile=128))
kw=dict(sparse_attention_interpret=True,linear_interpret=True)
full=build_ws32_challenger_decoder_program(mesh,config,options=options,**kw).execute(token,state,weights,rope)
names=[]
def compile(name,fn,args):
    names.append(name)
    return fn.lower(*args).compile()
split,report=diagnose_layerwise(mesh,config,options,token,state,weights,rope,compile_program=compile,**kw)
assert len(names)==len(set(names))==5
assert len(report['layers'])==8
assert report['head_healthy'] and np.asarray(full.state.contract_valid).all()
assert report['embedding']['nonzero']>0
for row in report['layers']:
    assert row['healthy'] and row['normalized_input']['finite'] and row['normalized_input']['nonzero']>0
np.testing.assert_array_equal(np.asarray(split[0]),np.asarray(full.next_token))
a,b=np.asarray(split[2]).astype(np.float64),np.asarray(full.final_residual_local).astype(np.float64)
# Separate layer executables change fused BF16 rounding. This probe is a
# documented numerical diagnostic, not an exact replacement.
relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b))
max_abs=float(np.max(np.abs(a-b)))
print(dict(relative_l2=relative_l2,max_abs=max_abs),flush=True)
assert relative_l2<.01 and max_abs<=.0625
# No mutation of input frontier/cache on this diagnostic execution.
assert int(np.asarray(state.position)[0])==0
assert not np.asarray(state.kv_cache_local).any()
'''
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=600)
    assert result.returncode == 0, result.stdout+result.stderr
