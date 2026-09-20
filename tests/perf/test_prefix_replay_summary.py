import copy
import json
from hashlib import sha256

import numpy as np
import pytest

from glm_tpu.perf.prefix_replay_summary import FIELDS, summarize_prefix_replay


def fixture():
    identity = dict(reference_sha256='a'*64, offsets=[0, 2])
    cases = [('code', np.arange(8, dtype=np.int32), np.arange(16, dtype=np.int32), (0, 2), identity)]
    ranks = []
    for rank in range(8):
        memory = [dict(device_id=4*rank+i, bytes_in_use=10, peak_bytes_in_use=20, bytes_limit=30) for i in range(4)]
        phases = {'prefix_replay_reference_admission':dict(passed=True), 'replay_code_first_token':dict(passed=True)}
        variants = {}
        for rows in (1, 2, 3):
            label = f'replay_code_r{rows}'
            required = {label+'_comparison', label+'_historical_reference', label+'_advance_0', label+'_advance_1'}
            for kind in ('verify', 'commit'):
                required.update(p+label+'_'+kind for p in ('compile_', 'graph_consensus_', 'hlo_', 'memory_'))
            windows = []
            for offset in (0, 2):
                required.add(label+f'_verify_{offset}')
                required.update(label+f'_reference_{offset}_{i}' for i in range(rows))
                prefixes = []
                for count in range(rows+1):
                    required.add(label+f'_commit_{offset}_{count}')
                    comparison = dict(bitwise_equal=True, finite=True, differing_elements=0,
                                      local_replica_elements=4, max_abs=0., relative_l2=0.)
                    states = {f:dict(comparison) for f in FIELDS}
                    for f in ('kv_cache_local', 'index_cache_local'):
                        states[f]['differing_layers_or_slots'] = []
                    written = {f:dict(comparison, logical_start=8+offset,
                        logical_stop=8+offset+count, expected_feature_replicas=4)
                        for f in ('kv_cache_local', 'index_cache_local')} if count else {}
                    prefixes.append(dict(consumed=count, state_comparisons=states, details=dict(written_cache=written)))
                windows.append(dict(input_offset=offset, predictions_equal=True,
                    differing_prediction_rows=[], residual_comparison=dict(comparison), prefixes=prefixes))
            phases.update({p:dict(passed=True) for p in required})
            variants[str(rows)] = dict(schema='glm_perf_same_prefix_v1', target_rows=rows,
                reference_sha256='a'*64, reset_to_ordinary_state=True, ordinary_reference_equal=True,
                measured_speculative_throughput=False, windows=windows, first_prediction_mismatch=None,
                memory_after=copy.deepcopy(memory))
        ranks.append(dict(rank=rank, complete=True, prefix_replay_sha256='b'*64,
            phases=phases, decode=dict(memory_after=memory), prefix_replay=dict(
                schema='glm_perf_prefix_replay_rank_v1', measured_speculative_throughput=False,
                full_index_slot_by_layer=list(range(78)),
                cases=dict(code=dict(identity=copy.deepcopy(identity), variants=variants)))))
    return ranks, cases


def test_all_owners_and_no_private_payloads():
    ranks, cases = fixture()
    ranks[0]['prefix_replay']['cases']['code']['raw_prompt'] = 'PRIVATE_SENTINEL'
    r = summarize_prefix_replay(ranks, cases, 'b'*64)
    assert not r['serving_admitted'] and not r['measured_speculative_throughput']
    assert 'PRIVATE_SENTINEL' not in json.dumps(r)
    assert r['cases']['code']['variants']['3']['windows'][0]['predictions_equal']


@pytest.mark.parametrize('bad', ['rank', 'complete', 'digest', 'reference', 'variant', 'window',
                               'prefix', 'phase', 'memory', 'span', 'nan', 'field', 'accounting', 'slots'])
def test_incomplete_or_inconsistent_evidence_refused(bad):
    ranks, cases = fixture()
    r = ranks[-1]; d = r['prefix_replay']['cases']['code']['variants']['3']
    w = d['windows'][0]; p = w['prefixes'][1]
    if bad == 'rank': r['rank'] = 0
    if bad == 'complete': r['complete'] = False
    if bad == 'digest': r['prefix_replay_sha256'] = 'c'*64
    if bad == 'reference': d['ordinary_reference_equal'] = False
    if bad == 'variant': del r['prefix_replay']['cases']['code']['variants']['1']
    if bad == 'window': d['windows'].pop()
    if bad == 'prefix': w['prefixes'].pop()
    if bad == 'phase': del r['phases']['replay_code_r3_commit_0_1']
    if bad == 'memory': d['memory_after'][1]['device_id'] = d['memory_after'][0]['device_id']
    if bad == 'span': p['details']['written_cache']['kv_cache_local']['logical_stop'] += 1
    if bad == 'nan': w['residual_comparison']['max_abs'] = float('nan')
    if bad == 'field': del p['state_comparisons']['selected_positions']
    if bad == 'accounting': d['first_prediction_mismatch'] = 1
    if bad == 'slots': r['prefix_replay']['full_index_slot_by_layer'][1] = 0
    with pytest.raises(ValueError): summarize_prefix_replay(ranks, cases, 'b'*64)


def test_negative_result_on_last_owner_is_preserved():
    ranks, cases = fixture()
    d = ranks[-1]['prefix_replay']['cases']['code']['variants']['3']
    w = d['windows'][0]
    w.update(predictions_equal=False, differing_prediction_rows=[1])
    d['first_prediction_mismatch'] = 2
    c = w['prefixes'][1]['state_comparisons']['kv_cache_local']
    c.update(bitwise_equal=False, differing_elements=1, max_abs=.5, relative_l2=.1,
             differing_layers_or_slots=[4])
    r = summarize_prefix_replay(ranks, cases, 'b'*64)['cases']['code']['variants']['3']
    assert not r['windows'][0]['predictions_equal']
    assert not r['windows'][0]['fleet_mismatch_pattern_agrees']
    assert r['windows'][0]['prefixes'][1]['cache_layers'] == [4]


def traced_fixture():
    ranks,cases=fixture()
    comp=dict(bitwise_equal=True,finite=True,differing_elements=0,local_replica_elements=4,max_abs=0.,relative_l2=0.)
    proposal_fields=('kv_cache_local','index_cache_local','selected_positions','selected_valid_counts',
        'selected_scores','predictions','final_residual_local','normalized_hidden_local','contract_valid')
    layer_fields=('normalized_inputs','hidden_updates','carried_residuals','selected_positions','selected_counts','selected_scores')
    for rank in ranks:
        rank['prefix_replay_trace']=rank['prefix_replay']['trace_layers']=True
        for n,r in rank['prefix_replay']['cases']['code']['variants'].items():
            rows=int(n);label=f'replay_code_r{n}'
            for graph in ('replay_code_ordinary_trace',label+'_trace'):
                for p in ('compile_','graph_consensus_','hlo_','memory_'):
                    rank['phases'][p+graph]=dict(passed=True)
            for w in r['windows']:
                start=w['input_offset']
                digest=sha256(cases[0][2][start+1:start+rows+1].tobytes()).hexdigest()
                w['prediction_sha256']=w['ordinary_prediction_sha256']=digest
                for suffix in ('traced_verify',*(f'traced_ordinary_{i}' for i in range(rows))):
                    rank['phases'][label+f'_trace_{start}_'+suffix]=dict(passed=True)
                w['trace']=dict(compiler_outputs_changed=True,
                    layers=[dict(layer=i,comparisons={f:dict(comp) for f in layer_fields}) for i in range(78)],
                    ordinary_instrumentation=[dict(prediction_equal=True,head_matches_prediction=True,
                        residual=dict(comp),state={f:dict(comp) for f in FIELDS}) for _ in range(rows)],
                    verifier_instrumentation=dict(predictions_equal=True,fields={f:dict(comp) for f in proposal_fields}),
                    final_normalized=dict(comp),verifier_head_matches_prediction=True,
                    ordinary_head_matches_prediction=True,top_two_ids_equal_by_row=[True]*rows,
                    verifier_top_score=[2.]*rows,ordinary_top_score=[2.]*rows,
                    verifier_logit_margin=[.5]*rows,ordinary_logit_margin=[.5]*rows)
    return ranks,cases


def test_trace_does_not_hide_instrumentation_changes():
    ranks,cases=traced_fixture()
    t=ranks[-1]['prefix_replay']['cases']['code']['variants']['3']['windows'][0]['trace']
    t['verifier_instrumentation']['predictions_equal']=False
    t['layers'][4]['comparisons']['hidden_updates'].update(bitwise_equal=False,differing_elements=1,max_abs=.1)
    r=summarize_prefix_replay(ranks,cases,'b'*64,trace_layers=True)
    diagnostic=r['cases']['code']['variants']['3']['windows'][0]['trace_by_rank'][-1]
    assert not diagnostic['instrumentation_bitwise_stable']
    assert diagnostic['first_differing_layer']==4


@pytest.mark.parametrize('bad',['missing','layer','margin','phase','unregistered','head','hash','reference_hash'])
def test_incomplete_trace_refused(bad):
    ranks,cases=traced_fixture()
    w=ranks[-1]['prefix_replay']['cases']['code']['variants']['3']['windows'][0]
    if bad=='missing':del w['trace']
    if bad=='layer':w['trace']['layers'].pop()
    if bad=='margin':w['trace']['verifier_logit_margin'][0]=float('nan')
    if bad=='phase':del ranks[-1]['phases']['replay_code_r3_trace_0_traced_verify']
    if bad=='head':w['trace']['ordinary_head_matches_prediction']=False
    if bad=='hash':del w['prediction_sha256']
    if bad=='reference_hash':w['ordinary_prediction_sha256']='0'*64
    with pytest.raises(ValueError):
        summarize_prefix_replay(ranks,cases,'b'*64,trace_layers=bad!='unregistered')


def test_trace_preserves_cross_host_prediction_disagreement():
    ranks,cases=traced_fixture()
    for rank in ranks:
        report=rank['prefix_replay']['cases']['code']['variants']['3']
        window=report['windows'][0]
        window.update(predictions_equal=False,differing_prediction_rows=[1],prediction_sha256='c'*64)
        report['first_prediction_mismatch']=2
    ranks[-1]['prefix_replay']['cases']['code']['variants']['3']['windows'][0]['prediction_sha256']='d'*64
    result=summarize_prefix_replay(ranks,cases,'b'*64,trace_layers=True)
    window=result['cases']['code']['variants']['3']['windows'][0]
    assert window['fleet_mismatch_pattern_agrees']
    assert not window['fleet_prediction_hashes_agree']
