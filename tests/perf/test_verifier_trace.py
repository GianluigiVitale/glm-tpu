"""Observe the existing target bodies; qualify only this synthetic CPU fixture."""
import os
import subprocess
import sys


def test_trace_comparison_detects_probe_and_instrumentation_changes():
    from typing import NamedTuple
    import numpy as np
    from glm_tpu.perf.verifier_trace import TargetTrace, compare_trace_window

    class State(NamedTuple):
        position: object
        contract_valid: object
    class Result(NamedTuple):
        state: object
        next_token: object
        final_residual_local: object
    class Proposal(NamedTuple):
        predictions: object
        final_residual_local: object
        contract_valid: object

    states = [State(np.array([i]), np.array([True])) for i in range(3)]
    results = [Result(states[i+1], np.array([21+i], np.int32), np.full((1,4), i)) for i in range(2)]
    def trace(i, bad=False):
        hidden = np.full((2,1,4), i, np.float32)
        return TargetTrace(hidden,hidden,hidden,np.zeros((2,1,2),np.int32),
            np.ones((2,1),np.int32),np.zeros((2,1,2),np.float32),hidden[0],
            np.array([[999 if bad else 21+i,22+i]],np.int32),np.array([[10.,9.]],np.float32))
    traces = [trace(0),trace(1)]
    candidate = TargetTrace(*(np.concatenate([t[i] for t in traces],axis=1 if i<6 else 0)
                              for i in range(9)))
    proposal = Proposal(np.array([21,22],np.int32),np.concatenate([r.final_residual_local for r in results]),np.ones(2,bool))
    health = []
    def compare(a,b):return dict(bitwise_equal=bool(np.array_equal(a,b)))
    def run(bad_probe=False, changed=False):
        def ordinary(token,state):
            i=int(state.position[0]);return results[i],trace(i,bad_probe)
        return compare_trace_window(np.array([20,21],np.int32),states,results,proposal,
            traced_ordinary=ordinary,traced_verifier=lambda t,s:(proposal._replace(
                predictions=proposal.predictions+int(changed)),candidate),
            ready=lambda x:x,put=np.asarray,compare=compare,healthy=lambda x,name:health.append(name))
    good=run()
    assert good['ordinary_head_matches_prediction'] and good['verifier_head_matches_prediction']
    assert all(c['bitwise_equal'] for layer in good['layers'] for c in layer['comparisons'].values())
    assert good['ordinary_logit_margin']==good['verifier_logit_margin']==[1.,1.]
    assert not run(bad_probe=True)['ordinary_head_matches_prediction']
    changed=run(changed=True)
    assert not changed['verifier_instrumentation']['predictions_equal']
    assert not changed['verifier_head_matches_prediction']
    assert health[:3]==['traced_verify','traced_ordinary_0','traced_ordinary_1']


def test_trace_geometry_head_order_and_uninstrumented_fixture_agreement():
    code = r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b,ws32_decoder as d
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions,build_ws32_challenger_decoder_program
from glm_tpu.perf.speculative_verify import build_verifier
from glm_tpu.perf.verifier_trace import build_target_trace
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
config,raw,wk=fixture(mesh,panel_geometry=True)
weights=bf16_resident_weights(mesh,config,raw)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
interpret=dict(sparse_attention_interpret=True,linear_interpret=True)
prefill=b.build_ws32_batched_prefill_program(mesh,config,block_rows=3,key_tile=128,**interpret)
initial=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
result=prefill.execute(put(jnp.array([30,31,32],jnp.int32)),put(jnp.int32(3)),initial,raw,tuple(put(x) for x in wk),rope)
state,token=b.finish_ws32_batched_prefill(result)
options=Ws32PerfOptions(sampler='greedy',bf16_resident=True,dsa_two_stage=True,
    routed_projection=RoutedProjectionConfig(output_tile=128,contraction_tile=128))
ordinary=build_ws32_challenger_decoder_program(mesh,config,options=options,**interpret).execute
verify=build_verifier(mesh,config,canonical_mlp=True,batched_attention=True,
    small_expert_tiles=True,rowwise_dsa=True,**interpret)
for rows,base,opt in [(1,ordinary,options),(3,verify,None)]:
    tokens=put(jnp.concatenate((token,jnp.array([65,32],jnp.int32)))[:rows])
    trace_fn=build_target_trace(mesh,config,ordinary_options=opt,**interpret)
    baseline=jax.block_until_ready(base(tokens,state,weights,rope))
    observed,trace=jax.block_until_ready(trace_fn(tokens,state,weights,rope))
    expected=observed.next_token if opt is not None else observed.predictions
    original=baseline.next_token if opt is not None else baseline.predictions
    np.testing.assert_array_equal(expected,original)
    np.testing.assert_array_equal(trace.top_ids[:,0],expected)
    assert trace.normalized_inputs.shape==(config.geometry.num_layers,rows,config.geometry.hidden_size)
    assert trace.hidden_updates.shape==trace.carried_residuals.shape==trace.normalized_inputs.shape
    assert trace.selected_positions.shape==(config.geometry.num_layers,rows,config.geometry.dsa_top_k)
    assert trace.top_scores.shape==(rows,2)
    assert np.isfinite(np.asarray(trace.top_scores)).all()
    assert np.all(np.asarray(trace.top_scores[:,0])>=np.asarray(trace.top_scores[:,1]))
    if opt is None:
        np.testing.assert_array_equal(trace.final_normalized,observed.normalized_hidden_local)
    tied,tt=jax.block_until_ready(trace_fn(tokens,state,weights._replace(lm_head_local=jnp.zeros_like(weights.lm_head_local)),rope))
    np.testing.assert_array_equal(tt.top_ids,np.tile(np.array([[0,1]],np.int32),(rows,1)))
    np.testing.assert_array_equal(tt.top_scores,np.zeros((rows,2),np.float32))
print('CPU trace covers all layers and exact lowest-ID tied head ordering; trained instrumentation remains unqualified')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        timeout=600)
    assert result.returncode == 0, result.stdout+result.stderr
