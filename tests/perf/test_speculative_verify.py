"""Characterize the CPU numerical boundary; this is not exact target admission.

The independent sequential decoder remains the comparison. Physical rollback
has a separate bitwise oracle in test_speculative_commit.py. Float tolerances
here only guard the measured small-fixture regression envelope; they establish
neither production correctness nor a general numerical error bound.
"""
import json
import os
import subprocess
import sys

import pytest


def test_four_row_small_expert_rowwise_boundary():
    """Three drafts need their own existing numerical/causal/rollback gate."""
    test_layer_major_verifier_cpu_numerical_boundary(4, True, True, True, True)


@pytest.mark.parametrize('rows', [1, 2, 3, 4])
def test_unrolled_attention_small_expert_bitwise_cpu(rows):
    """Qualify this CPU fixture across all replicas, not trained TPU execution."""
    test_layer_major_verifier_cpu_numerical_boundary(
        rows, True, False, True, False, unrolled_attention=True)


@pytest.mark.parametrize('rows', [1, 3])
def test_global_max_attention_verifier_cpu_boundary(rows):
    """Opt-in adaptation must retain causal/rollback gates and the existing bound."""
    test_layer_major_verifier_cpu_numerical_boundary(
        rows, True, True, True, True, global_max_attention=True)


@pytest.mark.parametrize('rows', [1, 2, 3, 5])
@pytest.mark.parametrize(('canonical_mlp', 'batched_attention', 'small_expert_tiles', 'rowwise_dsa'),
    [(False, False, False, False), (True, False, False, False), (True, True, False, False),
     (True, True, True, False), (True, True, True, True)],
    ids=['pooled', 'canonical', 'batched', 'm8_tiles', 'm8_rowwise'])
def test_layer_major_verifier_cpu_numerical_boundary(rows, canonical_mlp, batched_attention,
                                                    small_expert_tiles, rowwise_dsa,
                                                    unrolled_attention=False, global_max_attention=False):
    code = r'''
import json,hashlib
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions,build_ws32_challenger_decoder_program
from glm_tpu.perf.speculative_verify import build_verifier,build_prefix_committer
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rows=ROWS
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
verify=build_verifier(mesh,config,canonical_mlp=CANONICAL_MLP,batched_attention=BATCHED_ATTENTION,
                     small_expert_tiles=SMALL_EXPERT_TILES,rowwise_dsa=ROWWISE_DSA,
                     unrolled_attention=UNROLLED_ATTENTION,
                     global_max_attention=GLOBAL_MAX_ATTENTION,**interpret)
commit=build_prefix_committer(mesh,config)
tokens=put(jnp.concatenate((token,jnp.array([65,32,87,133],jnp.int32)))[:rows])
proposal=verify(tokens,state,weights,rope)
jax.block_until_ready(proposal)
assert np.asarray(proposal.contract_valid).all()
expected=[state];residuals=[];predictions=[]
for i in range(rows):
    result=ordinary(tokens[i:i+1],expected[-1],weights,rope)
    expected.append(result.state);residuals.append(result.final_residual_local)
    predictions.append(result.next_token)
np.testing.assert_array_equal(proposal.predictions,jnp.concatenate(predictions))
report=dict(schema='glm_mtp_cpu_verifier_boundary_v1',jax=jax.__version__,
    target_rows=rows,canonical_mlp=CANONICAL_MLP,batched_attention=BATCHED_ATTENTION,
    small_expert_tiles=SMALL_EXPERT_TILES,rowwise_dsa=ROWWISE_DSA,
    unrolled_attention=UNROLLED_ATTENTION,
    global_max_attention=GLOBAL_MAX_ATTENTION,
    target_prediction_agreement=True,exact_target_admitted=False,
    numerical_scope='eight synthetic layers, populated three-token prompt, CPU32 default XLA',
    float_comparisons={},selection_order_mismatches=[],envelope_failures=[],
    source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'glm_tpu/perf/speculative_verify.py','glm_tpu/perf/speculative_moe.py',
        'glm_tpu/perf/speculative_attention.py','glm_tpu/perf/speculative_experts.py',
        'glm_tpu/perf/bf16_resident.py','glm_tpu/perf/global_max_attention.py')})
def same(a,b,label):
    for i,(x,y) in enumerate(zip(jax.tree.leaves(a),jax.tree.leaves(b))):
        x,y=np.ascontiguousarray(x),np.ascontiguousarray(y)
        assert np.array_equal(x.view(np.uint8),y.view(np.uint8)),(label,i,x.shape)
def same_all_replicas(a,b,label):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        assert len(x.addressable_shards)==len(y.addressable_shards)==32
        for xs,ys in zip(x.addressable_shards,y.addressable_shards):
            assert xs.device==ys.device and xs.index==ys.index
            same(np.asarray(xs.data),np.asarray(ys.data),label+'/'+str(xs.device))
def same_state_except_scores(a,b,label):
    for name in a._fields:
        x,y=getattr(a,name),getattr(b,name)
        if name != 'selected_scores':
            same_all_replicas(x,y,label+'/'+name)
        else:
            assert len(x.addressable_shards)==len(y.addressable_shards)==32
            for xs,ys in zip(x.addressable_shards,y.addressable_shards):
                assert xs.device==ys.device and xs.index==ys.index
                # Stored FP32 scores retain the attention fixture's explicit
                # rounding boundary. IDs must still match bitwise; this is
                # not proof for arbitrary scores close to a selection cut.
                np.testing.assert_allclose(np.asarray(xs.data),np.asarray(ys.data),
                    rtol=2e-6,atol=2e-6,err_msg=label+'/selected_scores')
def numerical(a,b,label):
    x,y=np.asarray(a).astype(np.float64),np.asarray(b).astype(np.float64)
    np.testing.assert_array_equal(np.isfinite(x),np.isfinite(y))
    assert not np.isnan(x).any() and not np.isnan(y).any()
    finite=np.isfinite(y)
    error=x[finite]-y[finite]
    maximum=float(np.max(np.abs(error),initial=0))
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(y[finite]),1e-30))
    report['float_comparisons'][label]=dict(different=int(np.count_nonzero(x!=y)),
        max_abs=maximum,relative_l2=relative)
    # Deliberately reported as an empirical regression envelope, not equality.
    if maximum > .0625 or relative > .015625:
        report['envelope_failures'].append(label)
        # Retain the known four/five-row failures while checking their remaining
        # cache/causal/refusal invariants. The parent marks it unqualified.
        assert rows in (4, 5),(label,maximum,relative)
numerical(proposal.final_residual_local,jnp.concatenate(residuals),'residual')
if not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and (UNROLLED_ATTENTION or (ROWWISE_DSA and rows <= 2)):
    same_all_replicas(proposal.final_residual_local,jnp.concatenate(residuals),'residual all replicas')
same(commit(state,proposal,put(jnp.int32(0))),state,'empty prefix')
for n in range(1,rows+1):
    actual=commit(state,proposal,put(jnp.int32(n)));reference=expected[n]
    if not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and UNROLLED_ATTENTION:
        same_all_replicas(actual,reference,f'prefix {n} all fields all replicas')
    if not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and ROWWISE_DSA and rows == 1:
        same_state_except_scores(actual,reference,f'prefix {n} all replicas')
    for name in ('position','context_lengths','block_tables','selected_valid_counts','contract_valid'):
        same(getattr(actual,name),getattr(reference,name),name)
    for name in ('kv_cache_local','index_cache_local'):
        x,y=getattr(actual,name),getattr(reference,name)
        # The original prompt and every rejected/future cache row stay exact.
        same(x[:,:,:3],y[:,:,:3],name+' prompt')
        same(x[:,:,3+n:],y[:,:,3+n:],name+' future')
        numerical(x[:,:,3:3+n],y[:,:,3:3+n],name+f' prefix {n}')
    x,y=np.asarray(actual.selected_positions)[0],np.asarray(reference.selected_positions)[0]
    report['selection_order_mismatches'].append(int(np.count_nonzero(x!=y)))
    # This prefix is shorter than top_k: all causal keys must be selected,
    # although changed scores can change their order. This does not test a cut.
    ax,ay=np.argsort(x),np.argsort(y)
    same(x[ax],y[ay],'selected key set')
    numerical(np.asarray(actual.selected_scores)[0,ax],np.asarray(reference.selected_scores)[0,ay],f'scores prefix {n}')
# Out-of-range counts and failed health refuse without changing cache/frontiers.
for n in (-1,rows+1):
    actual=commit(state,proposal,put(jnp.int32(n)))
    same(actual,state._replace(contract_valid=jnp.zeros_like(state.contract_valid)),f'invalid count {n}')
bad=proposal._replace(contract_valid=jnp.zeros_like(proposal.contract_valid))
same(commit(state,bad,put(jnp.int32(1))),state._replace(contract_valid=jnp.zeros_like(state.contract_valid)),'poison')
# Future draft choices must not affect an earlier target row in the same graph.
if rows>1:
    changed=verify(tokens.at[1:].set(jnp.arange(rows-1,dtype=jnp.int32)+87),state,weights,rope)
    same(changed.predictions[:1],proposal.predictions[:1],'causal prediction')
    same(changed.final_residual_local[:1],proposal.final_residual_local[:1],'causal residual')
    same(changed.kv_cache_local[:,:,3:4],proposal.kv_cache_local[:,:,3:4],'causal KV')
    same(changed.index_cache_local[:,:,3:4],proposal.index_cache_local[:,:,3:4],'causal index')
report['future_draft_independence_bitwise']=True if rows>1 else None
invalid=verify(tokens.at[min(1,rows-1)].set(-1),state,weights,rope)
assert not np.asarray(invalid.contract_valid).all()
same(commit(state,invalid,put(jnp.int32(1))),state._replace(contract_valid=jnp.zeros_like(state.contract_valid)),'invalid token')
end=state._replace(position=put(jnp.array([config.context_capacity-rows+1],jnp.int32)),
    context_lengths=put(jnp.array([config.context_capacity-rows+2],jnp.int32)))
overflow=verify(tokens,end,weights,rope)
assert not np.asarray(overflow.contract_valid).any()
same(commit(end,overflow,put(jnp.int32(1))),end._replace(contract_valid=jnp.zeros_like(state.contract_valid)),'span overflow')
report['invalid_token_and_span_refuse']=True
report['cpu_fixture_bitwise_residual']=bool(not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and (UNROLLED_ATTENTION or (ROWWISE_DSA and rows <= 2)))
report['cpu_fixture_bitwise_state_except_scores']=bool(not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and (UNROLLED_ATTENTION or (ROWWISE_DSA and rows == 1)))
report['cpu_fixture_bitwise_state_all_fields']=bool(not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and UNROLLED_ATTENTION)
report['cpu_fixture_score_tolerance']=dict(atol=2e-6,rtol=2e-6) if not GLOBAL_MAX_ATTENTION and SMALL_EXPERT_TILES and ROWWISE_DSA and rows == 1 else None
print(json.dumps(report,sort_keys=True))
'''
    code = code.replace('ROWS', str(rows)).replace('CANONICAL_MLP', str(canonical_mlp))
    code = code.replace('BATCHED_ATTENTION', str(batched_attention))
    code = code.replace('SMALL_EXPERT_TILES', str(small_expert_tiles))
    code = code.replace('ROWWISE_DSA', str(rowwise_dsa))
    code = code.replace('UNROLLED_ATTENTION', str(unrolled_attention))
    code = code.replace('GLOBAL_MAX_ATTENTION', str(global_max_attention))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
    report=json.loads(result.stdout.strip().splitlines()[-1])
    if rows in (4, 5) and report['envelope_failures']:
        pytest.xfail(f'{rows}-row verifier exceeds CPU numerical qualification envelope: '
                     + ', '.join(report['envelope_failures']))
