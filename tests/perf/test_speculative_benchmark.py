"""Verifier benchmark comparisons must cover every local replica."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.perf_speculative_verify import local_comparison


def array(values):
    return SimpleNamespace(shape=(2, 2), dtype=np.dtype('float32'),
        addressable_shards=[SimpleNamespace(device=i,index=(slice(None),slice(None)),
            data=np.asarray(x,np.float32)) for i,x in enumerate(values)])


def test_comparison_sees_nonfirst_owner_error_and_nonfinite():
    values=[np.ones((2,2)) for _ in range(4)]
    baseline=array(values)
    assert local_comparison(baseline,baseline)['bitwise_equal']
    values[3]=np.ones((2,2))*2
    report=local_comparison(array(values),baseline)
    assert not report['bitwise_equal'] and report['finite']
    assert report['differing_elements']==4 and report['local_replica_elements']==16
    assert report['max_abs']==1 and report['relative_l2']==.5
    values[3][0,0]=np.nan
    report=local_comparison(array(values),baseline)
    assert not report['finite'] and not report['bitwise_equal']


def test_comparison_refuses_incomplete_or_misaligned_owners():
    baseline=array([np.ones((2,2)) for _ in range(4)])
    with pytest.raises(ValueError,match='four'):
        local_comparison(array([np.ones((2,2))]),baseline)
    other=array([np.ones((2,2)) for _ in range(4)])
    other.addressable_shards[3].device=0
    with pytest.raises(ValueError,match='owner'):
        local_comparison(other,baseline)
