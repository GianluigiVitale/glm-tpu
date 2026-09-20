"""Host-paired receipt checks must not qualify missing or divergent execution."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.perf.ordinary_suite import ordinary_order, output_agreement, summarize_ordinary_suite
from tests.perf.test_question_summary import prepare


def fixture(root, compare=True):
    prepare(root)
    rows=[json.loads((root/f'validation.rank{i}.json').read_text()) for i in range(8)]
    ids=np.array([1,2,3],np.int32)
    policy=RequestPolicy('fresh',0,3,4,8192,256,(255,))
    cases=[(f'prose_repeat{i}',ids,policy,None) for i in (1,2)]
    digests={label:'d'*64 for label,*_ in cases}
    control=dict(ordinary_suite_sha256='a'*64,ordinary_suite_compare_prefill=compare,
                 ordinary_prefill_admission_sha256=None)
    if compare:
        proof=dict(schema='glm_perf_real_globalmax_prefill_v1',all_hosts_idle_after=True,
            fleet_summary_passed=True,summary=dict(db610_token_check_passed=True,
                prefill_global_max_attention=True,checkpoint=rows[0]['checkpoint']))
        raw=json.dumps(proof).encode();(root/'ordinary_prefill_admission.json').write_bytes(raw)
        control['ordinary_prefill_admission_sha256']=sha256(raw).hexdigest()
    generated=np.array([10,11,12,13],np.int32)
    for row in rows:
        row.update(control)
        report=dict(schema='glm_ordinary_suite_v1',complete=True,cases={},
            suite_sha256=control['ordinary_suite_sha256'],compare_prefill=compare,
            same_decode_program=True,fresh_state_per_mode=True,
            profiling='no per-component blocking timers',
            memory_scope='allocator process high-water mark through each completed mode')
        row['ordinary_suite']=report
        for prefix in ('compile_','graph_consensus_','hlo_','memory_'):
            row['phases'][prefix+'ordinary_suite_packed']=dict(passed=True)
        for label,*_ in cases:
            identity=dict(file_sha256='d'*64,prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(),
                policy=json.loads(json.dumps(asdict(policy))))
            order=list(ordinary_order(label,compare))
            case=dict(identity=identity,planned_order=order,execution_order=order,modes={})
            report['cases'][label]=case
            for mode in order:
                entry={k:deepcopy(row[k]) for k in ('question','question_identity','question_prefill','phases')}
                entry['question_identity']=identity
                entry['question']['token_sha256']=sha256(generated.tobytes()).hexdigest()
                entry['ordinary_agreement']=output_agreement(generated,generated)
                entry['paired_wall_speedup']=1.
                case['modes'][mode]=entry
                for step in ('prefill','decode','health'):
                    row['phases'][f'suite_{label}_{mode}_warm_{step}']=dict(passed=True)
                if row['rank']==0:np.savez(root/f'suite_{label}_{mode}.generated.rank0.npz',tokens=generated)
    return rows,control,cases,digests


@pytest.mark.parametrize('compare',[False,True])
def test_pairs_are_validated_in_alternating_order_without_private_payload(tmp_path,compare):
    rows,control,cases,digests=fixture(tmp_path,compare)
    result=summarize_ordinary_suite(rows,control,cases,digests,root=tmp_path)
    second=result['cases']['prose_repeat2']
    assert second['execution_order']==(['optimized_ordinary','ordinary'] if compare else ['ordinary'])
    assert second['modes']['ordinary']['paired_wall_speedup']==1
    assert 'DO_NOT_PUBLISH' not in json.dumps(result)
    assert result['serving_admitted'] is False


@pytest.mark.parametrize('fault',['missing_host','mode','order','warmup','policy','timing',
    'private_tokens','claimed_agreement','admission','admission_weights','input_pin','profiling'])
def test_incomplete_or_misreported_pair_is_refused(tmp_path,fault):
    rows,control,cases,digests=fixture(tmp_path)
    row=rows[-1];case=row['ordinary_suite']['cases']['prose_repeat2'];entry=case['modes']['ordinary']
    if fault=='missing_host':rows.pop()
    elif fault=='mode':del case['modes']['optimized_ordinary']
    elif fault=='order':case['execution_order']=list(reversed(case['execution_order']))
    elif fault=='warmup':del row['phases']['suite_prose_repeat2_ordinary_warm_health']
    elif fault=='policy':entry['question_identity']['policy']['max_new_tokens']=5
    elif fault=='timing':entry['question']['tokens_per_second']=100
    elif fault=='private_tokens':np.savez(tmp_path/'suite_prose_repeat2_ordinary.generated.rank0.npz',tokens=np.array([9,9,9,9],np.int32))
    elif fault=='claimed_agreement':
        for r in rows:r['ordinary_suite']['cases']['prose_repeat2']['modes']['ordinary']['ordinary_agreement']['all_equal']=False
    elif fault=='admission':control['ordinary_prefill_admission_sha256']='b'*64
    elif fault=='admission_weights':row['checkpoint']['inventory_sha256']='b'*64
    elif fault=='input_pin':row['ordinary_suite_sha256']='c'*64
    else:row['ordinary_suite']['profiling']='blocking'
    with pytest.raises(ValueError):summarize_ordinary_suite(rows,control,cases,digests,root=tmp_path)


def test_differing_tokens_remain_a_failed_agreement_not_a_summary_error(tmp_path):
    rows,control,cases,digests=fixture(tmp_path)
    base=np.array([10,11,12,13],np.int32);other=np.array([10,99,12,13],np.int32)
    label='prose_repeat1';mode='optimized_ordinary'
    np.savez(tmp_path/f'suite_{label}_{mode}.generated.rank0.npz',tokens=other)
    for row in rows:
        entry=row['ordinary_suite']['cases'][label]['modes'][mode]
        entry['question']['token_sha256']=sha256(other.tobytes()).hexdigest()
        entry['ordinary_agreement']=output_agreement(other,base)
    result=summarize_ordinary_suite(rows,control,cases,digests,root=tmp_path)
    assert result['cases'][label]['modes'][mode]['ordinary_agreement']['first_mismatch']==1
    assert not result['cases'][label]['modes'][mode]['ordinary_agreement']['all_equal']


def test_runner_uses_finished_disposable_warm_prefix_and_fresh_measured_roots(tmp_path,monkeypatch):
    """Run the real request loop with stub model calls, including a padded tail."""
    from types import SimpleNamespace
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from glm_tpu.greenfield.runtime import ws32_batched_prefill as pre
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
    from glm_tpu.perf import request_loop, prefill_challenger
    from glm_tpu.perf.ordinary_suite import run_ordinary_suite
    from tests.greenfield.runtime.test_ws32_request_session import state
    created=[];starts=[];decoded=[]
    def cache(position):
        value=state(position)
        return value._replace(kv_cache_local=jnp.zeros(1),index_cache_local=jnp.zeros(1))
    def make(mesh,config,*,prompt_length):
        created.append(prompt_length)
        return pre.Ws32BatchedPrefillState(cache(0),np.zeros(1),np.int32(prompt_length),np.array(False))
    def prefill(block,count,old,*_):
        start=int(old.decoder.position[0]);end=start+int(count)
        assert end<=int(old.prompt_length)
        starts.append((start,end,int(old.prompt_length)))
        final=end==int(old.prompt_length)
        return pre.Ws32BatchedPrefillResult(old._replace(decoder=cache(end),finished=np.array(final)),
            np.array([7 if final else -1],np.int32))
    def decode(token,old,*_):
        assert int(token[0])>=0, 'warm decode must never consume unfinished-prefill token -1'
        position=int(old.position[0])+1;next_token=int(token[0])+1
        decoded.append((int(old.position[0]),int(token[0])))
        result=Ws32DecodeStepResult(cache(position),np.array([next_token],np.int32),np.zeros((1,1)))
        return request_loop.PackedDecodeResult(result,np.array([next_token,1,position,position+1],np.int32))
    monkeypatch.setattr(pre,'make_ws32_batched_prefill_state',make)
    monkeypatch.setattr(request_loop,'build_packed_decoder_program',lambda *a,**kw:SimpleNamespace(execute=decode))
    monkeypatch.setattr(prefill_challenger,'build_ws32_prefill_challenger_program',lambda *a,**kw:SimpleNamespace(execute=prefill))
    monkeypatch.setattr(multihost_utils,'process_allgather',lambda x:np.tile(x,(8,1)))
    ids=np.arange(129,dtype=np.int32);policy=RequestPolicy('fresh',0,129,4,8192,256,(255,))
    cases=[(f'prose_repeat{i}',ids,policy,None) for i in (1,2)]
    (tmp_path/'case-prose.json').write_text('{}')
    record=dict(token_comparison=dict(all_equal=True),phases={})
    def phase(name,action):
        value=action();record['phases'][name]=dict(passed=True);return value
    def compile_model(name,fn,args):
        for prefix in ('compile_','graph_consensus_','hlo_','memory_'):
            record['phases'][prefix+name]=dict(passed=True)
        return fn
    def require(value,message):
        if not value:raise RuntimeError(message)
    run_ordinary_suite(root=tmp_path,cases=cases,suite_sha256='a'*64,compare_prefill=True,
        mesh=None,config=SimpleNamespace(context_capacity=8192),weights=None,wk=None,rope=None,
        prefill={114:prefill,128:prefill},prefill_options={},decode_options=None,rank=0,
        record=record,phase=phase,require=require,compile_model=compile_model,
        admit=lambda *a:None,stats=lambda:[],fleet_all=bool,put=np.asarray,save=lambda:None)
    report=record['ordinary_suite']
    assert report['complete']
    assert created.count(128)==4 and created.count(129)==5
    assert starts.count((0,128,128))==4 # Disposable finished warm prefixes.
    assert starts.count((0,128,129))==4 and starts.count((128,129,129))==4
    assert len(decoded)==4+4*3 # One warm step and three delivered decode steps per mode.
    assert report['cases']['prose_repeat2']['execution_order']==['optimized_ordinary','ordinary']
    for label,*_ in cases:
        for mode in ('ordinary','optimized_ordinary'):
            value=report['cases'][label]['modes'][mode]
            assert value['question']['emitted']==4 and value['question']['timed_votes']==6
            assert value['ordinary_agreement']['all_equal']
            events=[json.loads(s) for s in (tmp_path/f'suite_{label}_{mode}.tokens.rank0.jsonl').read_text().splitlines()]
            assert len(events)==4


def test_full_fleet_summary_authenticates_registered_inputs_and_shared_graphs(tmp_path):
    from pathlib import Path
    from glm_tpu.perf.real_validation import summarize_real_validation
    from glm_tpu.user_request import TOKENIZER_FILES,TEMPLATE_SHA
    rows,control,cases,digests=fixture(tmp_path)
    model=json.loads((Path(__file__).resolve().parents[2]/'configs/glm-5.2-fp8-config.json').read_text())
    eos=model['eos_token_id'];eos=[eos] if type(eos) is int else eos
    ids=cases[0][1]
    question=dict(schema='glm_perf_question_v1',request_id='fresh',prompt_ids=ids.tolist(),
        prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(),max_new_tokens=4,
        tokenizer_files=TOKENIZER_FILES,chat_template_sha256=TEMPLATE_SHA,
        thinking='on/max',decode_policy='greedy')
    raw=json.dumps(question).encode();(tmp_path/'case-prose.json').write_bytes(raw)
    question_pin=sha256(raw).hexdigest()
    suite=dict(schema='glm_native_mtp_suite_v1',cases=[dict(label='prose',repeats=2,question_sha256=question_pin)])
    raw=json.dumps(suite).encode();(tmp_path/'native_suite.json').write_bytes(raw)
    control['ordinary_suite_sha256']=sha256(raw).hexdigest()
    path=tmp_path/'controller_identity.json';original=json.loads(path.read_text());original.update(control)
    path.write_text(json.dumps(original))
    for rank,row in enumerate(rows):
        row['ordinary_suite_sha256']=control['ordinary_suite_sha256']
        row['ordinary_suite']['suite_sha256']=control['ordinary_suite_sha256']
        for key in ('question','question_identity','question_prefill'):row.pop(key)
        row['programs'].pop('question_packed')
        for name in ('ordinary_suite_packed','ordinary_opt_prefill_128','ordinary_opt_prefill_114'):
            row['programs'][name]=deepcopy(row['programs']['decode'])
        for label,*_ in cases:
            case=row['ordinary_suite']['cases'][label]
            case['identity']['file_sha256']=question_pin
            case['identity']['policy'].update(vocab_size=model['vocab_size'],eos_ids=eos)
            for entry in case['modes'].values():entry['question_identity']=deepcopy(case['identity'])
        (tmp_path/f'validation.rank{rank}.json').write_text(json.dumps(row))
    result=summarize_real_validation(tmp_path)
    assert len(result['ordinary_suite']['cases'])==2
    assert 'DO_NOT_PUBLISH' not in json.dumps(result)
    path=tmp_path/'validation.rank7.json';row=json.loads(path.read_text())
    row['programs']['ordinary_opt_prefill_114']['stablehlo_sha256']='bad'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='graph disagreement'):summarize_real_validation(tmp_path)
