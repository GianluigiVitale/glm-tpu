from typing import NamedTuple
import numpy as np
import pytest

from glm_tpu.perf import speculative_diagnostics as module

class State(NamedTuple):
    position:np.ndarray
    cache:np.ndarray
    contract_valid:np.ndarray
class Result(NamedTuple):
    state:State
    next_token:np.ndarray
    final_residual_local:np.ndarray
class Proposal(NamedTuple):
    predictions:np.ndarray
    cache:np.ndarray
    contract_valid:np.ndarray

def environment(bad_prediction=False,bad_commit=False,bad_reference=False,unhealthy=False):
    initial=State(np.array([3],np.int32),np.zeros(64,np.int32),np.array([True]))
    def ordinary(token,state):
        cache=state.cache.copy();cache[state.position[0]]=token[0]
        return Result(state._replace(position=state.position+1,cache=cache),token+1+int(bad_reference),token)
    def verify(tokens,state):
        cache=state.cache.copy();start=int(state.position[0]);cache[start:start+tokens.size]=tokens
        pred=tokens+1
        if bad_prediction:pred=pred.copy();pred[0]+=1
        return Proposal(pred,cache,np.array([not unhealthy]))
    def commit(state,proposal,count):
        cache=proposal.cache.copy();end=int(state.position[0])+int(count)
        if not bad_commit:cache[end:]=state.cache[end:]
        return state._replace(position=np.array([end],np.int32),cache=cache)
    def healthy(value,label):
        if not np.asarray(value).all():raise RuntimeError(label)
    return initial,dict(ordinary=ordinary,verify=verify,commit=commit,replicate=np.asarray,
        ready=lambda x:x,healthy=healthy,compare=lambda a,b:dict(bitwise_equal=bool(np.array_equal(a,b))))

@pytest.mark.parametrize('rows',[2,3,5])
def test_trail(rows):
    initial,kw=environment()
    expected=np.arange(10,39,dtype=np.int32)
    actual,report,candidate,reference=module.compare_reference_trail(expected,initial,rows=rows,**kw)
    np.testing.assert_array_equal(actual,expected[1:])
    assert report['compared']==report['matches']==28
    assert report['ordinary_predictions_equal'] and report['target_predictions_equal']
    assert all(x['bitwise_equal'] for x in report['final_state_comparisons'].values())
    assert sum(b['live_rows'] for b in report['blocks'])==28
    assert candidate.position[0]==31 and np.count_nonzero(candidate.cache[31:])==0
    assert np.count_nonzero(initial.cache)==0
    assert not report['measured_speculative_throughput'] and report['teacher_forced']
    assert 'tokens' not in report and 'predictions' not in report

@pytest.mark.parametrize('bad',['prediction','reference','commit'])
def test_rejections_visible(bad):
    initial,kw=environment(bad_prediction=bad=='prediction',bad_reference=bad=='reference',bad_commit=bad=='commit')
    # Padding uses zero IDs; start with nonzero untouched future cache values
    # so a missing restoration cannot be hidden by initial zeros.
    initial=initial._replace(cache=np.full(64,999,np.int32))
    _,r,_,_=module.compare_reference_trail(np.arange(10,39,dtype=np.int32),initial,rows=3,**kw)
    if bad=='prediction':assert not r['target_predictions_equal'] and r['first_mismatch_index']==1
    if bad=='reference':assert not r['ordinary_predictions_equal']
    if bad=='commit':assert not r['final_state_comparisons']['cache']['bitwise_equal']

def test_health_aborts():
    initial,kw=environment(unhealthy=True)
    with pytest.raises(RuntimeError,match='verifier_0'):
        module.compare_reference_trail(np.arange(10,39,dtype=np.int32),initial,rows=3,**kw)


def fleet():
    import copy
    records = []
    initial,kw = environment()
    expected = np.arange(10,39,dtype=np.int32)
    reports = {str(n):module.compare_reference_trail(expected,initial,rows=n,**kw)[1] for n in (2,3)}
    fields = ('kv_cache_local','index_cache_local','selected_positions','selected_valid_counts',
        'selected_scores','position','block_tables','context_lengths','contract_valid')
    names = {f'speculative_{kind}_{n}' for kind in ('verify','commit') for n in (2,3)}
    for rank in range(8):
        r = dict(rank=rank,diagnose_speculative_verifier=True,decode_lse_attention=False,
            token_comparison=dict(all_equal=True),input_identity=dict(reference_sha256=reports['2']['full_reference_sha256']),
            speculative_verifier=copy.deepcopy(reports),phases=dict(verifier_reference_admission=dict(passed=True)),
            speculative_programs={name:dict(stablehlo_sha256='a'*64,optimized_hlo_sha256='b'*64,
                hlo_admission=dict(passed=True),memory_admission=dict(passed=True)) for name in names})
        for name in names:
            for prefix in ('compile_','graph_consensus_','hlo_','memory_'):
                r['phases'][prefix+name]=dict(passed=True)
        for n in (2,3):
            d = r['speculative_verifier'][str(n)]
            d['warmup_pairs']=5
            d['verifier_options']=dict(canonical_mlp=True,batched_attention=True)
            d['memory_after']=[dict(device_id=rank*4+i,bytes_in_use=10,peak_bytes_in_use=20,bytes_limit=30) for i in range(4)]
            d['final_state_comparisons']={name:dict(bitwise_equal=True,finite=True,differing_elements=0,
                local_replica_elements=4,max_abs=0.,relative_l2=0.) for name in fields}
            label=f'speculative_verify_{n}'
            for suffix in ('_warm_first','_warm','_trail'):
                r['phases'][label+suffix]=dict(passed=True)
            for start in range(0,28,n):
                for kind in ('verifier','commit'):
                    r['phases'][f'{label}_{kind}_{start}']=dict(passed=True)
            for i in range(28):r['phases'][f'{label}_ordinary_{i}']=dict(passed=True)
        records.append(r)
    return records


def test_fleet_summary_is_explicitly_not_speculative_throughput():
    import json
    ranks=fleet()
    # Inject private-like material at several depths; none may be published.
    ranks[0]['speculative_verifier']['2']['private_payload']='SYNTHETIC_SECRET'
    ranks[0]['speculative_verifier']['2']['blocks'][0]['tokens']='SYNTHETIC_SECRET'
    ranks[0]['speculative_verifier']['2']['final_state_comparisons']['kv_cache_local']['raw']='SYNTHETIC_SECRET'
    report=module.summarize_reference_trails(ranks)
    assert report['variants']['2']['target_predictions_equal']
    assert report['variants']['3']['ordinary_predictions_equal']
    assert not report['measured_speculative_throughput'] and not report['serving_admitted']
    assert 'SYNTHETIC_SECRET' not in json.dumps(report)


@pytest.mark.parametrize('case', ['rank','variant','phase','graph','admission','matches','block',
    'timing','memory','state','comparison','reference','options'])
def test_incomplete_or_inconsistent_fleet_refused(case):
    ranks=fleet();r=ranks[-1];d=r['speculative_verifier']['3']
    if case=='rank':r['rank']=0
    elif case=='variant':del r['speculative_verifier']['2']
    elif case=='phase':del r['phases']['speculative_verify_3_commit_27']
    elif case=='graph':r['speculative_programs']['speculative_verify_3']['stablehlo_sha256']='c'*64
    elif case=='admission':r['speculative_programs']['speculative_commit_2']['memory_admission']['passed']=False
    elif case=='matches':d['matches']=27
    elif case=='block':d['blocks'].pop()
    elif case=='timing':d['blocks'][0]['verify_seconds']=float('nan')
    elif case=='memory':d['memory_after'][-1]['device_id']=0
    elif case=='state':del d['final_state_comparisons']['index_cache_local']
    elif case=='comparison':d['final_state_comparisons']['kv_cache_local']['max_abs']=-1.
    elif case=='reference':d['full_reference_sha256']='0'*64
    elif case=='options':d['verifier_options']['batched_attention']=False
    with pytest.raises(ValueError):module.summarize_reference_trails(ranks)


def test_negative_prediction_and_nonfirst_owner_cache_result_preserved():
    ranks=fleet();d=ranks[-1]['speculative_verifier']['3']
    d.update(matches=27,target_predictions_equal=False,first_mismatch_index=28,observed_sha256='c'*64)
    d['blocks'][-1].update(matches=0,predictions_equal=False)
    d['final_state_comparisons']['kv_cache_local'].update(bitwise_equal=False,differing_elements=1,max_abs=.5,relative_l2=.1)
    report=module.summarize_reference_trails(ranks)
    assert not report['variants']['3']['target_predictions_equal']
    assert not report['variants']['3']['fleet_predictions_agree']
    assert not report['variants']['3']['ranks'][-1]['final_state_comparisons']['kv_cache_local']['bitwise_equal']


def test_explicit_small_expert_options_require_fleet_and_variant_agreement():
    ranks=fleet()
    options=dict(canonical_mlp=True,batched_attention=True,small_expert_tiles=True,rowwise_dsa=True)
    for rank in ranks:
        rank['speculative_verifier_options']=dict(options)
        for d in rank['speculative_verifier'].values():
            d['verifier_options']=dict(options)
    assert module.summarize_reference_trails(ranks)['verifier_options']==options
    ranks[-1]['speculative_verifier']['3']['verifier_options']['small_expert_tiles']=False
    with pytest.raises(ValueError,match='scope differs'):
        module.summarize_reference_trails(ranks)
    ranks[-1]['speculative_verifier']['3']['verifier_options']=dict(options)
    ranks[-1]['speculative_verifier_options']['rowwise_dsa']=False
    with pytest.raises(ValueError,match='options differ'):
        module.summarize_reference_trails(ranks)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_explicit_options_are_static_booleans(value):
    ranks=fleet()
    for rank in ranks:
        rank['speculative_verifier_options']=dict(canonical_mlp=True,batched_attention=True,
                                                 small_expert_tiles=value,rowwise_dsa=False)
    with pytest.raises(ValueError,match='options differ'):
        module.summarize_reference_trails(ranks)


def test_real_acquisition_summary_requires_diagnostic_on_every_host(tmp_path):
    import json
    from tests.perf.test_real_validation import fake_completed_fleet
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    for d in fleet():
        path=tmp_path/f"validation.rank{d['rank']}.json"
        r=json.loads(path.read_text())
        for k in ('diagnose_speculative_verifier','decode_lse_attention','speculative_programs','speculative_verifier'):
            r[k]=d[k]
        r['input_identity']['reference_sha256']=d['input_identity']['reference_sha256']
        r['phases'].update(d['phases'])
        path.write_text(json.dumps(r))
    result=summarize_real_validation(tmp_path)
    assert result['db610_token_check_passed']
    assert result['speculative_verifier']['variants']['3']['target_predictions_equal']
    path=tmp_path/'validation.rank7.json';r=json.loads(path.read_text())
    del r['speculative_verifier'];path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='missing verifier diagnostic'):
        summarize_real_validation(tmp_path)


def written_fleet():
    ranks=fleet()
    for r in ranks:
        r['speculative_cache_comparison_scope']='whole_and_written_span_v1'
        count=56 if r['rank'] in (0,7) else 0
        for n in (2,3):
            r['phases'][f'speculative_verify_{n}_written_cache_comparison']=dict(passed=True)
            r['speculative_verifier'][str(n)]['written_cache_comparisons']={
                name:dict(bitwise_equal=True,finite=True,differing_elements=0,
                    local_replica_elements=count*size,max_abs=0.,relative_l2=0.,
                    logical_start=2034,logical_stop=2062,local_replica_rows=count,
                    global_unique_elements=28*size,expected_feature_replicas=4)
                for name,size in [('kv_cache_local',78*640),('index_cache_local',20*128)]}
    return ranks


def test_written_cache_summary_requires_exact_fleet_coverage_and_preserves_errors():
    ranks=written_fleet()
    c=ranks[7]['speculative_verifier']['3']['written_cache_comparisons']['kv_cache_local']
    c.update(bitwise_equal=False,differing_elements=2,max_abs=.25,relative_l2=.02)
    result=module.summarize_reference_trails(ranks)
    assert result['cache_comparison_scope']=='whole_and_written_span_v1'
    assert result['variants']['3']['ranks'][7]['written_cache_comparisons']['kv_cache_local']==c
    assert not result['serving_admitted']


@pytest.mark.parametrize('bad',[
    'missing_scope','missing_field','partial_scope','span','elements','empty_error',
    'missing_phase','coverage','replicas','extra_payload',
])
def test_written_cache_summary_refuses_incomplete_or_inconsistent_evidence(bad):
    ranks=written_fleet();r=ranks[6]
    d=r['speculative_verifier']['3'];c=d['written_cache_comparisons']['kv_cache_local']
    if bad=='missing_scope':
        for item in ranks:item.pop('speculative_cache_comparison_scope')
    if bad=='missing_field':del d['written_cache_comparisons']['index_cache_local']
    if bad=='partial_scope':r.pop('speculative_cache_comparison_scope')
    if bad=='span':c['logical_start']=0
    if bad=='elements':c['local_replica_elements']=1
    if bad=='empty_error':c.update(bitwise_equal=False,max_abs=.1)
    if bad=='missing_phase':r['phases'].pop('speculative_verify_3_written_cache_comparison')
    if bad=='coverage':c.update(local_replica_rows=1,local_replica_elements=78*640)
    if bad=='replicas':c['expected_feature_replicas']=1
    if bad=='extra_payload':c['cache_values']=[1]
    with pytest.raises(ValueError):module.summarize_reference_trails(ranks)


@pytest.mark.parametrize('extra',[['--decode-lse-attention'],['--prefill-block-rows','512']])
def test_worker_refuses_mixed_baseline_before_acquisition(extra):
    import os
    import subprocess
    import sys
    r=subprocess.run([sys.executable,'tools/perf_real_validation.py','--diagnose-speculative-verifier',*extra],
        capture_output=True,text=True,env=dict(os.environ,JAX_PLATFORMS='cpu'),timeout=30)
    assert r.returncode!=0 and 'require the canonical D1/D8/D10' in r.stderr


@pytest.mark.parametrize('small_tiles',[False,True],ids=['baseline','small_tiles'])
def test_teacher_forced_diagnostic_real_jax_cpu32(small_tiles):
    import os
    import subprocess
    import sys
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
from glm_tpu.perf.speculative_verify import build_verifier,build_prefix_committer
from glm_tpu.perf.speculative_diagnostics import compare_reference_trail
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
config,raw,wk=fixture(mesh,panel_geometry=True)
weights=bf16_resident_weights(mesh,config,raw)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
interpret=dict(sparse_attention_interpret=True,linear_interpret=True)
prefill=b.build_ws32_batched_prefill_program(mesh,config,block_rows=3,key_tile=128,**interpret)
result=prefill.execute(put(jnp.array([30,31,32],jnp.int32)),put(jnp.int32(3)),
    b.make_ws32_batched_prefill_state(mesh,config,prompt_length=3),raw,tuple(put(x) for x in wk),rope)
initial,token=b.finish_ws32_batched_prefill(result)
options=Ws32PerfOptions(sampler='greedy',bf16_resident=True,dsa_two_stage=True,
    routed_projection=RoutedProjectionConfig(output_tile=256,contraction_tile=256))
ordinary=build_ws32_challenger_decoder_program(mesh,config,options=options,**interpret).execute
expected=[int(np.asarray(token)[0])];state=initial
for _ in range(5):
    output=jax.block_until_ready(ordinary(token,state,weights,rope))
    state,token=output.state,output.next_token
    expected.append(int(np.asarray(token)[0]))
expected=np.asarray(expected,np.int32)
def healthy(v,label):assert np.asarray(v).all(),label
def compare(a,b):
    x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
    return dict(bitwise_equal=bool(np.array_equal(x.view(np.uint8),y.view(np.uint8))))
for rows in (2,3):
    verify=build_verifier(mesh,config,canonical_mlp=True,batched_attention=True,
                         small_expert_tiles=SMALL_TILES,rowwise_dsa=SMALL_TILES,**interpret)
    commit=build_prefix_committer(mesh,config)
    actual,report,candidate,reference=compare_reference_trail(expected,initial,rows=rows,
        verify=lambda t,s:verify(t,s,weights,rope),commit=commit,
        ordinary=lambda t,s:ordinary(t,s,weights,rope),replicate=put,
        ready=jax.block_until_ready,healthy=healthy,compare=compare)
    assert report['compared']==5 and report['ordinary_predictions_equal']
    assert report['target_predictions_equal'],report
    assert candidate.position[0]==reference.position[0]==8
    for name in ('position','context_lengths','block_tables','contract_valid'):
        assert report['final_state_comparisons'][name]['bitwise_equal'],name
    for name in ('kv_cache_local','index_cache_local'):
        assert compare(getattr(candidate,name)[:,:,8:],getattr(initial,name)[:,:,8:])['bitwise_equal']
    assert report['blocks'][-1]['padded_rows']==rows-(5%rows)
'''
    code=code.replace('SMALL_TILES',str(small_tiles))
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=900)
    assert result.returncode == 0,result.stdout+result.stderr
