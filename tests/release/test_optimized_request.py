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
    ('prompt_ids',[7,9]),('eos_ids',[0])])
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
