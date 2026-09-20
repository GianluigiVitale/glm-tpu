"""Fail closed on incomplete, mismatched or overstated native measurements."""
from copy import deepcopy

import pytest

from glm_tpu.perf.native_summary import summarize_native_rows


def fixture():
    programs={'native_prefill_128','native_prefill_114','native_ordinary',
        *(f'native_refresh_{n}' for n in (1,2,3,8)),
        *(f'native_{kind}_{n}' for kind in ('inputs','verify','commit') for n in (1,2,3))}
    graph=dict(hlo_admission=dict(passed=True),memory_admission=dict(passed=True),
               stablehlo_sha256='a'*64,optimized_hlo_sha256='b'*64,compiled_memory={})
    policy=dict(request_id='fake-native',seed=0,prompt_tokens=2034,max_new_tokens=29,
                context_capacity=8192,vocab_size=256,eos_ids=[10])
    base=dict(healthy=True,speculative=False,sampling='greedy',decode_steps=28,emitted=29,
        timed_votes=56,warm_steps_excluded=0,delivery_boundary='rank0 private JSONL token write+flush; no network transport',
        excludes_cold_load_compile=True,decode_rate_excludes_prefill=True,token_sha256='c'*64,
        wall_seconds=2.,tokens_per_second=14.,finish_reason='length',
        ttft_seconds=1.1,request_wall_seconds=3.2,vote_wall_seconds=.1,
        p50_ms=70.,p99_ms=80.,model_step_p50_ms=65.)
    variants={}
    for n,rounds,accepted in ((2,14,[14]),(3,10,[9,9])):
        variants[str(n)]=dict(rows=n,emitted=29,decode_tokens=28,rounds=rounds,
            accepted_drafts=sum(accepted),proposed_drafts=sum(accepted),
            accepted_by_position=accepted,proposals_by_position=accepted,
            finish_reason='length',healthy=True,all_host_token_agreement=True,finite_caches=True,
            includes_draft_verify_rejections_refresh_commit_votes_delivery=True,
            excludes_cold_load_compile=True,decode_rate_excludes_prefill=True,
            warm_steps_excluded=0,delivery_boundary=base['delivery_boundary'],token_sha256='c'*64,
            wall_seconds=4.,tokens_per_second=7.,proposal_seconds=2.1,refresh_commit_seconds=.2,
            host_vote_seconds=.2,host_agreement_seconds=.1,ttft_seconds=1.3,request_wall_seconds=5.5,
            bootstrap_seconds=.2,accepted_per_round=28/rounds,paired_wall_speedup=.5,p50_round_ms=100.,
            component_seconds=dict(draft=.1,verify=2.,commit=.1,refresh=.1),
            component_timing_scope='synchronized device calls inside measured wall time',
            prefill=dict(prompt_tokens=2034,wall_seconds=1.,prompt_tokens_per_second=2034.),
            ordinary_agreement=dict(baseline_token_sha256='c'*64,all_equal=True,first_mismatch=None,
                                    multirow_numerical_boundary=True),memory_after=[])
    case=dict(policy=policy,prompt_sha256='d'*64,export_db610_parity=True,
        prefill=dict(prompt_tokens=2034,wall_seconds=1.,prompt_tokens_per_second=2034.),
        live_memory_admission={k:dict(passed=True) for k in programs},ordinary=base,speculative=variants)
    native=dict(schema='glm_native_mtp_comparison_v1',complete=True,sampled=False,independent_native_reference=False,
        pack_index=dict(plan_sha256='e'*64,manifest_sha256='f'*64),
        cases=dict(db610=case),load_admission=dict(passed=True),resident_admission=dict(passed=True))
    phases={prefix+name:dict(passed=True) for name in programs
            for prefix in ('compile_','graph_consensus_','hlo_','memory_')}
    phases.update({k:dict(passed=True) for k in ['db610_ordinary','db610_ordinary_agreement','db610_ordinary_finite',
        'db610_bootstrap','db610_resident_admission','db610_export_parity',*(f'db610_warm_{n}' for n in (1,2,3)),
        *(f'native.db610.r{n}{suffix}' for n in (2,3) for suffix in ('_generation','_agreement','_finite'))]})
    rows=[]
    for rank in range(8):
        row=dict(rank=rank,native_mtp=deepcopy(native),native_pack_index_sha256='1'*64,
            native_programs={k:deepcopy(graph) for k in programs},phases=deepcopy(phases),
            input_identity=dict(prompt_sha256='d'*64,reference_sha256='c'*64),
            decode=dict(memory_after=[dict(device_id=rank*4+i) for i in range(4)]))
        for value in row['native_mtp']['cases']['db610']['speculative'].values():
            value['memory_after']=[dict(device_id=rank*4+i,peak_bytes_in_use=100,bytes_limit=200) for i in range(4)]
        rows.append(row)
    controller=dict(native_pack_index_sha256='1'*64)
    index=dict(plan_sha256='e'*64,rank_manifests={str(i):dict(manifest_sha256='f'*64) for i in range(8)})
    return rows,controller,index


def test_native_aggregate_does_not_turn_negative_speed_into_success_or_leak_payloads():
    rows,controller,index=fixture()
    for r in rows:
        r['native_mtp']['private_text']='do not export'
        r['native_mtp']['cases']['db610']['speculative']['2']['ordinary_agreement']['private_text']='do not export'
    result=summarize_native_rows(rows,controller,index)
    assert 'do not export' not in str(result)
    assert result['cases']['db610']['speculative']['2']['timings']['paired_wall_speedup']==dict(min=.5,max=.5)
    assert result['all_host_tokens_agree'] and result['independent_native_reference'] is False
    baseline=result['cases']['db610']['ordinary']
    assert baseline['timings']['ttft_seconds']==dict(min=1.1,max=1.1)
    assert baseline['decode_steps']==28 and baseline['finish_reason']=='length'


def unprofiled_fixture():
    rows,controller,index=fixture()
    controller['native_component_timing']='none'
    for row in rows:
        row['native_component_timing']='none'
        for case in row['native_mtp']['cases'].values():
            for result in case['speculative'].values():
                result['component_seconds']=None
                result['component_timing_scope']='disabled; request synchronization and wall timing retained'
    return rows,controller,index


def test_unprofiled_summary_keeps_wall_costs_and_does_not_invent_component_times():
    result=summarize_native_rows(*unprofiled_fixture())
    assert result['component_timing']=='none'
    variant=result['cases']['db610']['speculative']['3']
    assert variant['component_seconds'] is None
    assert variant['timings']['tokens_per_second']==dict(min=7.,max=7.)
    assert variant['timings']['host_vote_seconds']==dict(min=.2,max=.2)
    assert variant['ordinary_agreement']['all_equal'] is True


@pytest.mark.parametrize('fault', ['controller','rank','scope','fabricated_components','phase_overlap'])
def test_unprofiled_summary_rejects_mixed_or_overstated_timings(fault):
    rows,controller,index=unprofiled_fixture()
    d=rows[-1]['native_mtp']['cases']['db610']['speculative']['3']
    if fault=='controller':controller.pop('native_component_timing')
    if fault=='rank':rows[-1].pop('native_component_timing')
    if fault=='scope':d['component_timing_scope']='synchronized device calls inside measured wall time'
    if fault=='fabricated_components':d['component_seconds']=dict(draft=0,verify=0,commit=0,refresh=0)
    if fault=='phase_overlap':d['proposal_seconds']=d['wall_seconds']
    with pytest.raises(ValueError):summarize_native_rows(rows,controller,index)


def test_native_timing_option_requires_pack_before_runtime_initialization(monkeypatch):
    from tools.perf_real_validation import main
    monkeypatch.setattr('sys.argv',['perf_real_validation.py','--native-component-timing','none'])
    with pytest.raises(ValueError,match='requires a pinned native pack'):
        main()


@pytest.mark.parametrize('field,value', [('ttft_seconds',.5),('request_wall_seconds',2.),
    ('vote_wall_seconds',3.),('p50_ms',float('nan')),('p99_ms',0.),('model_step_p50_ms',-1.)])
def test_paired_ordinary_latency_is_validated_on_every_host(field,value):
    rows,controller,index=fixture()
    rows[-1]['native_mtp']['cases']['db610']['ordinary'][field]=value
    with pytest.raises(ValueError,match='ordinary paired latency'):
        summarize_native_rows(rows,controller,index)


@pytest.mark.parametrize('fault', ['partial','graph','pack','health','rate','counts','position_counts',
    'timing','components','tokens','comparison','memory','phase','export'])
def test_native_bad_receipt_refused(fault):
    rows,controller,index=fixture();r=rows[-1]
    native=r['native_mtp'];case=native['cases']['db610'];d=case['speculative']['3']
    if fault=='partial':native['complete']=False
    if fault=='graph':r['native_programs']['native_verify_3']['optimized_hlo_sha256']='2'*64
    if fault=='pack':native['pack_index']['manifest_sha256']='0'*64
    if fault=='health':d['finite_caches']=False
    if fault=='rate':d['tokens_per_second']=30.
    if fault=='counts':d['decode_tokens']=29
    if fault=='position_counts':d['accepted_by_position']=[10,10]
    if fault=='timing':d['ttft_seconds']=.1
    if fault=='components':d['component_seconds']['draft']=10.
    if fault=='tokens':d['token_sha256']='0'*64
    if fault=='comparison':d['ordinary_agreement']['all_equal']=False
    if fault=='memory':d['memory_after'][0]['peak_bytes_in_use']=201
    if fault=='phase':r['phases']['native.db610.r3_finite']['passed']=False
    if fault=='export':case['ordinary']['token_sha256']='0'*64
    with pytest.raises(ValueError):summarize_native_rows(rows,controller,index)


def test_registered_suite_repeats_have_separate_results_and_pinned_identity():
    import numpy as np
    from hashlib import sha256
    from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
    rows,controller,index=fixture()
    ids=np.arange(2034,dtype=np.int32)%256
    policy=RequestPolicy('fake-native',0,2034,29,8192,256,(10,))
    suite=[('prose_repeat1',ids,policy,None),('prose_repeat2',ids,policy,None)]
    controller['native_suite_sha256']='3'*64
    for r in rows:
        r['native_suite_sha256']='3'*64
        for label,_,_,_ in suite:
            case=deepcopy(r['native_mtp']['cases']['db610'])
            case['prompt_sha256']=sha256(ids.tobytes()).hexdigest()
            case['export_db610_parity']=False
            r['native_mtp']['cases'][label]=case
            r['phases'].update({key.replace('db610',label):deepcopy(value)
                for key,value in list(r['phases'].items()) if 'db610' in key})
    out=summarize_native_rows(rows,controller,index,suite_cases=suite)
    assert set(out['cases'])=={'db610','prose_repeat1','prose_repeat2'}
    rows[-1]['native_suite_sha256']='4'*64
    with pytest.raises(ValueError):summarize_native_rows(rows,controller,index,suite_cases=suite)
