from hashlib import sha256
import json
import numpy as np
import pytest

from glm_tpu.perf.long_question import load_question,question_blocks,measure_question
from glm_tpu.user_request import TOKENIZER_FILES,TEMPLATE_SHA
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
from glm_tpu.perf.request_loop import PackedDecodeResult
from tests.greenfield.runtime.test_ws32_request_session import state,prefill


@pytest.mark.parametrize('n',[1,114,115,128,129,242,243,256,2034])
def test_question_block_padding_and_counts(n):
    ids=np.arange(n,dtype=np.int32)
    blocks=question_blocks(ids)
    assert all(len(b) in (114,128) and 0<c<=len(b) and np.all(b[c:]==-1) for b,c in blocks)
    np.testing.assert_array_equal(np.concatenate([b[:c] for b,c in blocks]),ids)


def test_question_admission_rejects_wrong_identity_and_budget(tmp_path):
    value=dict(schema='glm_perf_question_v1',request_id='test-long',prompt_ids=[1,2,3],
        prompt_ids_sha256=sha256(np.array([1,2,3],np.int32).tobytes()).hexdigest(),max_new_tokens=10,
        tokenizer_files=TOKENIZER_FILES,chat_template_sha256=TEMPLATE_SHA,thinking='on/max',decode_policy='greedy')
    path=tmp_path/'question.json'
    def load():
        path.write_text(json.dumps(value))
        return load_question(path,sha256(path.read_bytes()).hexdigest(),capacity=20,vocab_size=256,eos_ids=(255,))
    assert load()[1].max_new_tokens==10
    with pytest.raises(ValueError,match='digest'):load_question(path,'0'*64,capacity=20,vocab_size=256,eos_ids=(255,))
    for key,bad in [('max_new_tokens',18),('prompt_ids',[True,2,3]),('thinking','off'),('decode_policy','sampled'),('prompt_ids_sha256','0'*64)]:
        old=value[key];value[key]=bad
        with pytest.raises(ValueError):load()
        value[key]=old


@pytest.mark.parametrize('eos,emitted,reason',[(255,4,'length'),(10,3,'eos'),(7,1,'eos')])
def test_question_speed_includes_all_votes_and_delivery(eos,emitted,reason):
    clock=[10.];calls=[];delivered=[]
    def decode(token,previous):
        calls.append(int(token[0]));clock[0]+=.2
        index=len(calls)
        result=Ws32DecodeStepResult(state(3+index),np.array([8+index],np.int32),np.zeros((1,1)))
        return PackedDecodeResult(result,np.array([8+index,1,3+index,4+index],np.int32))
    def vote(valid):clock[0]+=.01;return valid
    def deliver(event):delivered.append(event);clock[0]+=.05
    policy=RequestPolicy('fresh-question',0,3,4,20,256,(eos,))
    tokens,report,final=measure_question(policy,prefill(),decode_step=decode,
        replicate_uniform=lambda x:x,fleet_all=vote,deliver=deliver,request_started=9.,clock=lambda:clock[0])
    assert len(tokens)==len(delivered)==emitted
    assert report['decode_steps']==emitted-1 and report['finish_reason']==reason
    assert report['timed_votes']==2*(emitted-1)
    assert report['wall_seconds']==pytest.approx(.27*(emitted-1))
    assert report['tokens_per_second']==(pytest.approx(1/.27) if emitted>1 else None)
    assert report['token_sha256']==sha256(tokens.tobytes()).hexdigest()
    assert report['answer_correctness']=='not yet assessed'


def test_failed_delivery_is_not_retried_or_reported_as_success():
    calls=[]
    def fail(event):calls.append(event);raise OSError('sink failed')
    with pytest.raises(RuntimeError,match='state/delivery'):
        measure_question(RequestPolicy('bad-sink',0,3,4,20,256,(255,)),prefill(),
            decode_step=lambda *a:None,replicate_uniform=lambda x:x,fleet_all=bool,
            deliver=fail,request_started=0.)
    assert len(calls)==1
