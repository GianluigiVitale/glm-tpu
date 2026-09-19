"""Exact score/position ordering for the payload-sort candidate."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa import merge_topk_candidates_with_scores
from glm_tpu.perf.dsa_payload_merge import sort_payload_candidate_merge


@pytest.mark.parametrize('geometry',[(8,3,8,16),(8,32,512,2048)])
def test_payload_sort_bitwise_for_ties_signed_zero_masks_and_nonfinite(geometry):
    owners,rows,width,top_k=geometry
    rng=np.random.default_rng(512)
    opts=dict(top_k=top_k,global_context_size=owners*width,paired_position_sort=True)
    compare=jax.jit(lambda s,p,n:(sort_payload_candidate_merge(s,p,n,**opts),
                                merge_topk_candidates_with_scores(s,p,n,**opts)))
    for trial in range(12):
        scores=rng.integers(-5,6,size=(owners,rows,width)).astype(np.float32)
        positions=np.stack([rng.permutation(owners*width) for _ in range(rows)],axis=0)
        positions=positions.reshape(rows,owners,width).transpose(1,0,2).astype(np.int32)
        lengths=np.linspace(0,owners*width,rows,dtype=np.int32)
        if trial%3==0:
            scores.fill(-0.0);scores.ravel()[::3]=0.0
        if trial%3==1:
            scores=np.asarray(rng.normal(size=scores.shape),np.float32)
        scores=np.where(positions<lengths[None,:,None],scores,-np.inf)
        positions=np.where(positions<lengths[None,:,None],positions,-1)
        if trial==9:scores[-1,-1,0]=np.nan
        if trial==10:scores[-1,-1,0]=np.inf
        if trial==11:
            # Exercise negative and positive NaN payloads through fallback.
            scores[-1,-1,:2]=np.asarray([0xffc00123,0x7fc00456],np.uint32).view(np.float32)
        actual,expected=compare(jnp.asarray(scores),jnp.asarray(positions),jnp.asarray(lengths))
        for a,e in zip(actual,expected):
            np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(e).view(np.uint8))
