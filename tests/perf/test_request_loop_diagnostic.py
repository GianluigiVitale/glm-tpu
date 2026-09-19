import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
from glm_tpu.perf.request_loop import PackedDecodeResult
from glm_tpu.perf.request_loop_diagnostic import measure_request_trail
from tests.greenfield.runtime.test_ws32_request_session import state,prefill


@pytest.mark.parametrize('packed',[False,True])
def test_fixed_reference_loop_counts_only_post_warmup_delivery(packed):
    clock=[10.]
    calls=[]
    def decode(token,previous,*uniform):
        calls.append(int(token[0]));clock[0]+=.2
        index=len(calls)
        result=Ws32DecodeStepResult(state(3+index),np.array([8+index],np.int32),np.zeros((1,1)))
        return PackedDecodeResult(result,np.array([8+index,1,3+index,4+index],np.int32)) if packed else result
    def vote(valid):clock[0]+=.01;return valid
    policy=RequestPolicy('greedy-diagnostic',7,3,4,20,256,(255,))
    values,report,final,last=measure_request_trail(policy,prefill(),np.array([7,9,10,11],np.int32),
        decode_step=decode,packed=packed,replicate_uniform=lambda x:x,fleet_all=vote,
        warm_steps=1,clock=lambda:clock[0])
    np.testing.assert_array_equal(values,[7,9,10,11])
    assert calls==[7,9,10]
    assert report['samples']==2 and report['warm_steps']==1 and report['decode_steps']==3
    assert report['timed_votes']==(4 if packed else 6)
    assert report['timed_uniform_transfers']==(0 if packed else 2)
    assert report['token_comparison']['all_equal'] and report['finish_reason']=='length'
    assert report['wall_seconds']==pytest.approx(.44 if packed else .46)
    assert report['model_step_p50_ms']==pytest.approx(200.)
    assert int(final.position[0])==6 and int(last.next_token[0])==11
    assert not any(k in report for k in ('tokens','prompt','expected','state'))


def test_early_eos_is_a_short_failed_reference_trail():
    policy=RequestPolicy('early',7,3,4,20,256,(9,))
    result=Ws32DecodeStepResult(state(4),np.array([9],np.int32),np.zeros((1,1)))
    values,report,_,_=measure_request_trail(policy,prefill(),np.array([7,9,10,11],np.int32),
        decode_step=lambda *a:result,packed=False,replicate_uniform=lambda x:x,
        fleet_all=bool,warm_steps=1)
    assert len(values)==2 and report['samples']==0 and report['tokens_per_second'] is None
    assert not report['token_comparison']['all_equal']
    assert report['token_comparison']['matches']==2 and report['token_comparison']['first_mismatch_index']==2
