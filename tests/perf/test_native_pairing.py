"""Exercise actual callback ordering, deferred pairing and abort boundaries."""
from hashlib import sha256

import numpy as np
import pytest

from glm_tpu.perf.native_pairing import comparison_order, run_paired_modes


@pytest.mark.parametrize('label,expected', [
    ('db610',('ordinary','r2','r3')), ('question',('ordinary','r2','r3')),
    ('code_repeat1',('ordinary','r2','r3')), ('code_repeat2',('r3','r2','ordinary')),
    ('code_repeat3',('ordinary','r2','r3'))])
def test_explicit_repeat_schedule(label,expected):
    assert comparison_order(label,'alternating')==expected
    assert comparison_order(label,'ordinary_first')==('ordinary','r2','r3')


def result(tokens,rate):
    return dict(token_sha256=sha256(tokens.tobytes()).hexdigest(),tokens_per_second=rate)


@pytest.mark.parametrize('label',['code_repeat1','code_repeat2'])
def test_pairing_uses_actual_ordinary_output_even_when_it_runs_last(label):
    calls=[];observations=[]
    def ordinary():
        calls.append('ordinary')
        tokens=np.array([1,2,3,4],np.int32)
        return tokens,result(tokens,10.)
    def speculative(rows):
        calls.append('r'+str(rows))
        tokens=np.array([1,2,9,4] if rows==2 else [1,2,3],np.int32)
        return tokens,result(tokens,12. if rows==2 else 15.)
    def observe(mode,report,completed):
        assert 'ordinary_agreement' not in report
        observations.append((mode,completed))
    r=run_paired_modes(label,order_policy='alternating',ordinary=ordinary,
        speculative=speculative,observe=observe)
    assert tuple(calls)==comparison_order(label,'alternating')
    assert observations[-1][1]==tuple(calls)
    assert r['r2']['ordinary_agreement']['first_mismatch']==2
    assert r['r3']['ordinary_agreement']['first_mismatch']==3
    for mode,rate in [('r2',1.2),('r3',1.5)]:
        assert not r[mode]['ordinary_agreement']['all_equal']
        assert r[mode]['ordinary_agreement']['baseline_token_sha256']==r['ordinary']['token_sha256']
        assert r[mode]['paired_wall_speedup']==rate


def test_preserves_completed_tokens_when_callbacks_reuse_a_buffer():
    shared=np.array([1,2],np.int32)
    def ordinary():
        shared[:]=[1,2]
        return shared,result(shared,10.)
    def speculative(rows):
        shared[:]=[1,3] if rows==3 else [1,2]
        return shared,result(shared,15.)
    r=run_paired_modes('code_repeat2',order_policy='alternating',ordinary=ordinary,
        speculative=speculative,observe=lambda *a:None)
    assert not r['r3']['ordinary_agreement']['all_equal']
    assert r['r2']['ordinary_agreement']['all_equal']


def test_callback_failure_aborts_without_retry_or_later_mode():
    calls=[]
    def fail(rows):
        calls.append(rows)
        raise RuntimeError('refused')
    with pytest.raises(RuntimeError,match='refused'):
        run_paired_modes('code_repeat2',order_policy='alternating',ordinary=lambda:pytest.fail('late ordinary'),
            speculative=fail,observe=lambda *a:pytest.fail('incomplete result'))
    assert calls==[3]


@pytest.mark.parametrize('fault',['digest','rate','dtype'])
def test_invalid_callback_evidence_is_not_paired(fault):
    tokens=np.array([1,2],np.int32);report=result(tokens,10.)
    if fault=='digest':report['token_sha256']='0'*64
    if fault=='rate':report['tokens_per_second']=float('nan')
    if fault=='dtype':tokens=tokens.astype(np.int64)
    with pytest.raises(ValueError):
        run_paired_modes('code_repeat1',order_policy='alternating',ordinary=lambda:(tokens,report),
            speculative=lambda _:pytest.fail('late speculative'),observe=lambda *a:None)
