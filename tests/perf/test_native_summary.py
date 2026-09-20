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
        wall_seconds=2.,tokens_per_second=14.)
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
