"""Private question preparation and dispatch boundaries, without a model call."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.entrypoints.cli import main as cli
from glm_tpu.utils import json_utils
from glm_tpu.entrypoints.cli import ask
from glm_tpu.engine import request


def test_cli_passes_questions_without_opening_devices(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(ask,'main',lambda args:seen.append(args) or 0)
    assert cli.main(['ask','a difficult question','--prepare-only'])==0
    assert seen[0].question=='a difficult question' and seen[0].context=='32k'
    assert cli.main(['ask','--questions',str(tmp_path/'questions.json')])==0
    assert seen[1].question is None
    assert cli.main(['ask','question','--keep-loaded'])==0
    assert seen[2].keep_loaded
    with pytest.raises(SystemExit):cli.main(['ask','question','--questions','file'])


def test_batch_files_private_and_isolated(monkeypatch,tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    root=tmp_path/'private'
    def prepare(**kwargs):
        messages=json.loads(kwargs['messages_path'].read_text())
        number=int(messages[0]['content'].split()[-1])
        value=request.from_token_ids([number+1],request_id=kwargs['request_id'],
            max_new_tokens=kwargs['max_new_tokens'],context_capacity=kwargs['context_capacity'])
        kwargs['output'].write_bytes(json_utils.canonical(value))
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


@pytest.mark.parametrize('context,expected',[('128k',None),('8k',None),('32k',None)])
def test_default_budget_reaches_preparation_without_launch(monkeypatch,tmp_path,context,expected):
    from tests.fixtures.site import write_example_site
    seen=[]
    monkeypatch.setattr(ask,'prepare_questions',lambda *args,**kwargs:seen.append(kwargs) or tmp_path/'request.json')
    site=write_example_site(tmp_path/'site.toml')
    args=SimpleNamespace(questions=None,question='question',context=context,
        max_new_tokens=None,prepare_only=True,site=site)
    assert ask.main(args)==0
    assert seen[0]['max_new_tokens']==expected
    # private inputs go under the site's run root; the tokenizer is the site's model path
    assert seen[0]['output_root'].parent==tmp_path/'runs' and seen[0]['tokenizer_root']==tmp_path/'model'


def test_ask_refuses_without_a_site_file(tmp_path):
    from glm_tpu.config.site import SiteConfigError
    args=SimpleNamespace(questions=None,question='question',context='32k',max_new_tokens=None,
        prepare_only=True,site=tmp_path/'absent.toml')
    with pytest.raises(SiteConfigError,match='no site file'):ask.main(args)


def test_concurrent_cli_is_explicit_and_invalid_count_never_writes(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(ask,'main',lambda args:seen.append(args) or 0)
    assert cli.main(['ask','--questions','private.json','--context','32k','--concurrent'])==0
    assert seen[0].concurrent and seen[0].context=='32k'
    for count,capacity in ((5,32768),(8,32768),(9,32768),(3,8192)):
        root=tmp_path/f'private-{count}'
        with pytest.raises(ValueError):
            ask.prepare_questions(['q']*count,repo=tmp_path/'repo',tokenizer_root=tmp_path,
                output_root=root,context_capacity=capacity,max_new_tokens=20,concurrent=True)
        assert not root.exists()
