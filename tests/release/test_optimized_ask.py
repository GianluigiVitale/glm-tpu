"""Private question preparation and dispatch boundaries, without a model call."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu import cli, user_request
from glm_tpu.optimized import ask, request


def test_cli_passes_questions_without_opening_devices(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(ask,'main',lambda args:seen.append(args) or 0)
    assert cli.main(['ask','a difficult question','--prepare-only'])==0
    assert seen[0].question=='a difficult question' and seen[0].context=='128k'
    assert cli.main(['ask','--questions',str(tmp_path/'questions.json')])==0
    assert seen[1].question is None
    with pytest.raises(SystemExit):cli.main(['ask','question','--questions','file'])


def test_batch_files_private_and_isolated(monkeypatch,tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    root=tmp_path/'private'
    def prepare(**kwargs):
        messages=json.loads(kwargs['messages_path'].read_text())
        number=int(messages[0]['content'].split()[-1])
        value=request.from_token_ids([number+1],request_id=kwargs['request_id'],
            max_new_tokens=kwargs['max_new_tokens'],context_capacity=kwargs['context_capacity'])
        kwargs['output'].write_bytes(user_request.canonical(value))
        kwargs['output'].chmod(0o600)
    monkeypatch.setattr(request,'prepare_file',prepare)
    path=ask.prepare_questions([f'question {i}' for i in range(10)],repo=repo,
        tokenizer_root=tmp_path,output_root=root,context_capacity=request.LONG_CAPACITY,
        max_new_tokens=32768)
    values=request.requests(request.read(path))
    assert [v['prompt_ids'] for v in values]==[[i+1] for i in range(10)]
    assert len({v['request_id'] for v in values})==10
    assert root.stat().st_mode&0o077==0
    assert all(p.stat().st_mode&0o077==0 for p in root.iterdir())
    with pytest.raises(FileExistsError):
        ask.prepare_questions(['new'],repo=repo,tokenizer_root=tmp_path,output_root=root,
            context_capacity=request.LONG_CAPACITY,max_new_tokens=10)


@pytest.mark.parametrize('questions',[[],['']*10,['q']*11,['q',None]])
def test_invalid_batch_never_prepares_or_dispatches(tmp_path,questions):
    root=tmp_path/'private'
    with pytest.raises(ValueError):
        ask.prepare_questions(questions,repo=tmp_path/'repo',tokenizer_root=tmp_path,
            output_root=root,context_capacity=request.LONG_CAPACITY,max_new_tokens=32768)
    assert not root.exists()
