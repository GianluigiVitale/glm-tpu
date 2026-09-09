"""Actual BudgetedCalls/persistence with fixture compute and memory, not TPU."""

from hashlib import sha256
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_worker as worker
from tests.greenfield.validation.test_ws32_prefill_frontier_worker import setup as original_setup


def setup(tmp_path, monkeypatch, failure=None):
    calls, config, events = original_setup(tmp_path, monkeypatch,
                                          'post_memory' if failure == 'post_memory' else None)
    calls.budgeter = worker.memory_budget
    calls.record['call_evidence'] = [dict(phase=f'layer{layer}/{name}', graph=name, completed=True)
                                    for layer in (0, 1) for name in worker.PROGRAMS[:2]]
    prompt = np.arange(8155, dtype=np.int32)
    monkeypatch.setattr(worker, 'PROMPT_SHA', sha256(prompt.tobytes()).hexdigest())
    allocated = []
    def fresh(mesh):
        value = ((NS(branch=len(allocated), offset=0),)*3,)*2
        allocated.append(value)
        return value
    monkeypatch.setattr(worker, 'fresh_caches', fresh)
    monkeypatch.setattr(worker, 'independent_caches', lambda a,b: a is not b)
    def inputs(mesh, tokens, offset, caches, *rest):
        assert np.array_equal(tokens, prompt[offset:offset+len(tokens)])
        assert caches[0][0].offset == offset
        return (np.pad(tokens,(0,128-len(tokens))), len(tokens), offset, caches)
    monkeypatch.setattr(worker, 'inputs', inputs)
    base = calls.programs['prefill_chunk']
    class Program:
        memory_analysis = base.memory_analysis
        def __call__(self, tokens, count, offset, caches):
            branch = caches[0][0].branch
            events.append(('dense_dispatch', count, offset, branch))
            if failure == 'dispatch':
                raise ValueError('fixture dispatch failure')
            cache = NS(branch=branch, offset=offset+count)
            return ((0,0,cache,cache,cache,0,0,0,0,0,0,0),)*2
    calls.programs = {name: Program() for name in worker.PROGRAMS}
    def capture(result, *, count, keep_caches, **kwargs):
        arrays = {'row_output': np.zeros((count, 2), np.uint16)}
        if keep_caches:
            for slot in calls.local_slots.values():
                for layer in (0,1):
                    for family,width in (('kv',640),('index',128),('repair',128)):
                        arrays[f'slot{slot}_layer{layer}__{family}'] = np.zeros((16,64,width),np.uint16)
        return arrays,dict(count=count, keep_caches=keep_caches, valid=failure!='health',errors=[])
    monkeypatch.setattr(worker, 'capture', capture)
    def compare(witness, *, branch, slot, caches):
        assert set(caches)=={'kv','index','repair'}
        assert caches['kv'].shape==(2,16,64,640)
        return dict(branch=branch,slot=slot,reproduced=failure!='reproduction')
    monkeypatch.setattr(worker,'compare_owner',compare)
    return calls, config, prompt, events


def execute(calls, config, prompt):
    worker.execute_five_calls(calls,mesh=None,config=config,prompt_tokens=prompt,
                              embedding=None,layers=None,wk=None,rope=None,witness={})


def test_nine_calls_fixed_schedule_originals_and_both_branch_comparison(tmp_path,monkeypatch):
    calls,config,prompt,events=setup(tmp_path,monkeypatch)
    execute(calls,config,prompt)
    assert [e[1:] for e in events if e[0]=='dense_dispatch']==[
        (128,0,0),(32,0,1),(32,32,1),(32,64,1),(32,96,1)]
    assert len(calls.record['call_evidence'])==9
    assert len(list(tmp_path.glob('*.npz')))==5
    report=json.loads((tmp_path/'comparison.json').read_text())
    assert len(report['owners'])==8 and report['reproduced']
    assert not report['numerical_promotion'] and not report['performance_claim']


@pytest.mark.parametrize('failure',['health','post_memory','dispatch','reproduction'])
def test_preserve_completed_outputs_and_vote_before_successor(tmp_path,monkeypatch,failure):
    calls,config,prompt,events=setup(tmp_path,monkeypatch,failure)
    with pytest.raises(ValueError):
        execute(calls,config,prompt)
    assert len([e for e in events if e[0]=='dense_dispatch'])==(5 if failure=='reproduction' else 1)
    assert (tmp_path/'wide_final.npz').exists()==(failure!='dispatch')
    assert (tmp_path/'narrow_128.npz').exists()==(failure=='reproduction')
    assert events[-1]==('vote',False)
    if failure=='reproduction':
        assert not json.loads((tmp_path/'comparison.json').read_text())['reproduced']


@pytest.mark.parametrize('change',['prompt','wk','budget'])
def test_preflight_refuses_before_any_model_dispatch(tmp_path,monkeypatch,change):
    calls,config,prompt,events=setup(tmp_path,monkeypatch)
    if change=='prompt':prompt[0]+=1
    elif change=='wk':calls.record['call_evidence'].pop()
    else:calls.budgeter=lambda *a,**k:{}
    with pytest.raises(ValueError,match='inventory'):
        execute(calls,config,prompt)
    assert not any(e[0]=='dense_dispatch' for e in events)
