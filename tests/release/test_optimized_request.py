"""Fixed-profile, privacy and budget gates before any accelerator import."""
from copy import deepcopy
import pytest
from glm_tpu.optimized import request


def test_fixed_profile_and_full_budget():
    value=request.from_token_ids([7,8],request_id='ordinary-1',max_new_tokens=8190)
    request.validate(value)
    assert value['decode_policy']=='greedy' and value['context_capacity']==8192
    with pytest.raises(ValueError):
        request.from_token_ids([7,8],request_id='ordinary-1',max_new_tokens=8191)


@pytest.mark.parametrize('field,value', [('decode_policy','sampled'),('context_capacity',166912),
    ('max_new_tokens',True),('request_id','changed'),('thinking','off'),('seed',0),
    ('prompt_ids',[7,9]),('eos_ids',[0]),('model_revision','0'*40),
    ('model_id','zai-org/GLM-5.2-FP8'),('chat_template_sha256',request.legacy.TEMPLATE_SHA)])
def test_changed_request_never_inherits_admission(field,value):
    body=request.from_token_ids([7,8],request_id='ordinary-1',max_new_tokens=8)
    body[field]=value
    with pytest.raises(ValueError):request.validate(body)


def test_private_file_digest(tmp_path):
    from glm_tpu.user_request import canonical
    body=request.from_token_ids([7],request_id='ordinary-1',max_new_tokens=8)
    path=tmp_path/'request.json';path.write_bytes(canonical(body))
    assert request.read(path)==body
    with pytest.raises(ValueError):request.read(path,expected_sha256='0'*64)


def test_long_context_preserves_full_output_budget():
    value=request.from_token_ids([7]*131072,request_id='long',max_new_tokens=32768,
                                 context_capacity=request.LONG_CAPACITY)
    request.validate(value)
    assert value['context_capacity']==166912
    with pytest.raises(ValueError):
        request.from_token_ids([7]*131073,request_id='long',max_new_tokens=1,
                               context_capacity=request.LONG_CAPACITY)
    with pytest.raises(ValueError):
        request.from_token_ids([7]*131072,request_id='long',max_new_tokens=35841,
                               context_capacity=request.LONG_CAPACITY)


def test_ten_distinct_requests_bound_order_and_capacity():
    values=[request.from_token_ids([i+1],request_id=f'q-{i}',max_new_tokens=4)
            for i in range(10)]
    body=request.batch(values)
    assert request.requests(body)==values
    changed=deepcopy(body);changed['requests'].reverse()
    with pytest.raises(ValueError):request.validate_payload(changed)
    for bad in ([],values+[values[0]],[values[0],values[0]]):
        with pytest.raises(ValueError):request.batch(bad)
    long=request.from_token_ids([7],request_id='long',max_new_tokens=4,
                                context_capacity=request.LONG_CAPACITY)
    with pytest.raises(ValueError):request.batch([values[0],long])


@pytest.mark.parametrize('capacity,prompt_length,explicit_cap,expected',[
    (166912,186,None,163840),(166912,131072,None,35840),(166912,186,32768,32768),
    (32768,186,None,32582),(32768,32767,None,1),(8192,186,None,8006),
])
def test_preparation_allocates_available_output_without_truncating_prompt(
        monkeypatch,tmp_path,capacity,prompt_length,explicit_cap,expected):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    repo=Path(__file__).resolve().parents[2]
    ids=[7]*prompt_length
    def tokenize(messages, **kwargs):
        assert kwargs['enable_thinking'] is True and kwargs['reasoning_effort']=='max'
        assert kwargs['add_generation_prompt'] is True
        assert messages==[dict(role='user',content='synthetic question')]
        return ids
    tokenizer=SimpleNamespace(apply_chat_template=tokenize)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:tokenizer)))
    monkeypatch.setattr(request.model,'TOKENIZER_FILES',{})
    messages=tmp_path/'messages.json'
    messages.write_text(json.dumps([dict(role='user',content='synthetic question')]))
    output=tmp_path/'request.json'
    request.prepare_file(messages_path=messages,output=output,repo=repo,
        tokenizer_root=tmp_path,request_id='output-budget',max_new_tokens=explicit_cap,
        context_capacity=capacity)
    value=request.read(output)
    assert value['prompt_ids']==ids
    assert value['max_new_tokens']==expected
    assert len(ids)+expected<=value['context_capacity']


def test_old_prepared_requests_cannot_be_relabelled():
    body=request.from_token_ids([7],request_id='old',max_new_tokens=1024)
    body['schema']='glm_ws32_optimized_request_v1'
    body.pop('model_id');body.pop('model_revision')
    body['chat_template_sha256']=request.legacy.TEMPLATE_SHA
    unsigned={k:v for k,v in body.items() if k!='request_sha256'}
    from hashlib import sha256
    body['request_sha256']=sha256(request.legacy.canonical(unsigned)).hexdigest()
    with pytest.raises(ValueError):request.validate(body)


@pytest.mark.parametrize('prompt_length',[32768,32769])
def test_exhausted_context_never_truncates_or_writes(monkeypatch,tmp_path,prompt_length):
    from pathlib import Path
    from types import SimpleNamespace
    repo=Path(__file__).resolve().parents[2]
    tokenizer=SimpleNamespace(apply_chat_template=lambda *args,**kwargs:[7]*prompt_length)
    template=(repo/request.model.TEMPLATE_PATH).read_text()
    with pytest.raises(ValueError):
        request.from_messages([dict(role='user',content='synthetic')],tokenizer=tokenizer,
            chat_template=template,request_id='full',context_capacity=32768)
