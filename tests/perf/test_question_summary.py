from copy import deepcopy
import json
import pytest
from glm_tpu.perf.real_validation import summarize_real_validation
from tests.perf.test_real_validation import fake_completed_fleet


def prepare(root):
    fake_completed_fleet(root)
    controller=root/'controller_identity.json';v=json.loads(controller.read_text());v['question_sha256']='d'*64;controller.write_text(json.dumps(v))
    for rank in range(8):
        p=root/f'validation.rank{rank}.json';r=json.loads(p.read_text())
        r['programs']['question_packed']=deepcopy(r['programs']['decode'])
        for name in ('question_reference_admission','compile_question_packed','graph_consensus_question_packed',
                     'hlo_question_packed','memory_question_packed','question_prefill_0','question_prefill_health_0',
                     'question_generation','question_token_agreement','question_cache_check'):
            r['phases'][name]=dict(passed=True)
        r['question_identity']=dict(file_sha256='d'*64,prompt_ids_sha256='e'*64,
            policy=dict(request_id='fresh',seed=0,prompt_tokens=3,max_new_tokens=4,context_capacity=8192,vocab_size=256,eos_ids=[255]))
        r['question_prefill']=dict(prompt_tokens=3,wall_seconds=1.,prompt_tokens_per_second=3.)
        r['question']=dict(healthy=True,all_host_token_agreement=True,final_cache_finite=True,
            excludes_cold_load_compile=True,decode_rate_excludes_prefill=True,speculative=False,sampling='greedy',
            warm_steps_excluded=0,decode_steps=3,emitted=4,timed_votes=6,finish_reason='length',token_sha256='f'*64,
            delivery_boundary='rank0 private JSONL token write+flush; no network transport',
            wall_seconds=.3,ttft_seconds=1.1,request_wall_seconds=1.41,vote_wall_seconds=.03,tokens_per_second=10.,
            p50_ms=100.,p99_ms=105.,model_step_p50_ms=90.,
            windows=[dict(first_decode_step=0,count=3,wall_seconds=.3,tokens_per_second=10.,private='DO_NOT_PUBLISH')],
            memory_after=deepcopy(r['decode']['memory_after']),tokens=['DO_NOT_PUBLISH'])
        p.write_text(json.dumps(r))


def test_question_scope_and_no_raw_payload(tmp_path):
    prepare(tmp_path);r=summarize_real_validation(tmp_path)
    assert r['question']['timings']['tokens_per_second']==dict(min=10.,max=10.)
    assert r['question']['emitted']==4 and r['question']['finish_reason']=='length'
    assert 'DO_NOT_PUBLISH' not in json.dumps(r)


@pytest.mark.parametrize('fault',['missing','phase','token','count','vote','rate','window','policy','memory','graph','scope'])
def test_question_incomplete_or_disagreeing_evidence_refused(tmp_path,fault):
    prepare(tmp_path);p=tmp_path/'validation.rank7.json';r=json.loads(p.read_text())
    if fault=='missing':del r['question']
    elif fault=='phase':del r['phases']['question_prefill_health_0']
    elif fault=='token':r['question']['token_sha256']='a'*64
    elif fault=='count':r['question']['emitted']=3
    elif fault=='vote':r['question']['timed_votes']=5
    elif fault=='rate':r['question']['tokens_per_second']=100.
    elif fault=='window':r['question']['windows'][0]['first_decode_step']=1
    elif fault=='policy':r['question_identity']['policy']['max_new_tokens']=5
    elif fault=='memory':r['question']['memory_after'][0]['device_id']=0
    elif fault=='graph':r['programs']['question_packed']['stablehlo_sha256']='different'
    else:r['question']['speculative']=True
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize_real_validation(tmp_path)
