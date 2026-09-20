"""Compiled compact acceptance agrees with an independent host prefix oracle."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.perf.speculative_plan import plan_verification
from glm_tpu.perf.speculative_request import accept_greedy, read_device_plan
from glm_tpu.perf.speculative_verify import VerificationProposal


def proposal(predictions, health=None):
    n=len(predictions)
    return VerificationProposal(None,None,None,None,None,jnp.asarray(predictions,jnp.int32),
        None,None,jnp.ones(n,bool) if health is None else jnp.asarray(health,bool))


def test_compiled_device_plan_every_rejection_eos_and_tail():
    fn=jax.jit(partial(plan_verification,vocab_size=256,eos_ids=(90,91)))
    for rows in (1,2,3):
        for rejection in range(rows):
            for eos_at in range(rows+1):
                targets=np.arange(10,10+rows,dtype=np.int32)
                if eos_at<rows:targets[eos_at]=90
                ids=np.concatenate((np.array([7],np.int32),targets[:-1])).copy()
                if rejection<rows-1:ids[rejection+1]=200
                for remaining in range(1,rows+2):
                    result=fn(jnp.asarray(ids),proposal(targets),jnp.array([510],jnp.int32),jnp.int32(remaining))
                    expected=accept_greedy(ids,targets,remaining,(90,91))
                    x,y,actual=read_device_plan(np.asarray(result.metadata),rows=rows,position=510,
                        remaining=remaining,pending=7,vocab_size=256,eos_ids=(90,91))
                    np.testing.assert_array_equal(x,ids);np.testing.assert_array_equal(y,targets)
                    assert actual==expected and int(result.count)==expected.count
                    np.testing.assert_array_equal(result.proposal.predictions,targets)


@pytest.mark.parametrize('fault',['health','input','prediction','budget','frontier'])
def test_unhealthy_device_plan_never_authorizes_cache_rows(fault):
    ids=jnp.array([7,10,11],jnp.int32);p=proposal([10,11,12]);remaining=jnp.int32(3)
    position=jnp.array([510],jnp.int32)
    if fault=='health':p=p._replace(contract_valid=p.contract_valid.at[-1].set(False))
    if fault=='input':ids=ids.at[1].set(256)
    if fault=='prediction':p=p._replace(predictions=p.predictions.at[2].set(-1))
    if fault=='budget':remaining=jnp.int32(0)
    if fault=='frontier':position=jnp.array([-1],jnp.int32)
    result=jax.jit(partial(plan_verification,vocab_size=256,eos_ids=(90,)))(ids,p,position,remaining)
    assert int(result.count)==0 and int(result.metadata[5])==0
    with pytest.raises(ValueError):
        read_device_plan(np.asarray(result.metadata),rows=3,position=510,remaining=3,
                         pending=7,vocab_size=256,eos_ids=(90,))


@pytest.mark.parametrize('column',range(6))
def test_corrupt_plan_header_cannot_commit(column):
    result=plan_verification(jnp.array([7,10,11],jnp.int32),proposal([10,11,12]),
        jnp.array([510],jnp.int32),jnp.int32(3),vocab_size=256,eos_ids=(90,))
    bad=np.asarray(result.metadata).copy();bad[column]+=1
    with pytest.raises(ValueError):
        read_device_plan(bad,rows=3,position=510,remaining=3,pending=7,vocab_size=256,eos_ids=(90,))
