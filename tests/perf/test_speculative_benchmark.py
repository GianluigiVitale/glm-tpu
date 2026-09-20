"""Verifier benchmark comparisons must cover every local replica."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.perf_speculative_verify import local_comparison, local_cache_span_comparison


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


def cache_array(values, owners=(0,0,1,1)):
    return SimpleNamespace(shape=(2,3,16,3),dtype=np.dtype('float32'),
        addressable_shards=[SimpleNamespace(device=i,
            index=(slice(None),slice(None),slice(owner*2,owner*2+2),slice(None)),
            data=np.asarray(value,np.float32)) for i,(owner,value) in enumerate(zip(owners,values,strict=True))])


def test_written_span_uses_physical_pages_and_excludes_large_prompt_values():
    # Logical page 1 is physical page 0; positions 17..19 cross owners 0/1.
    table=np.array([[2,0,1]],np.int32)
    values=[np.full((2,3,2,3),1000.,np.float32) for _ in range(4)]
    for owner,x in zip((0,0,1,1),values):
        x[:,0,1 if owner==0 else slice(None),:]=1
    baseline=cache_array(values)
    changed=[x.copy() for x in values]
    for owner,x in zip((0,0,1,1),changed):
        x[:,0,1 if owner==0 else slice(None),:]=2
        x[:,2,:,:]=3000  # Old prompt values must not dilute this span's error.
    report=local_cache_span_comparison(cache_array(changed),baseline,table,17,20)
    assert report['logical_start']==17 and report['logical_stop']==20
    assert report['local_replica_rows']==6
    assert report['local_replica_elements']==report['differing_elements']==36
    assert report['global_unique_elements']==18 and report['expected_feature_replicas']==4
    assert report['max_abs']==report['relative_l2']==1
    assert report['finite'] and not report['bitwise_equal']


def test_span_reports_empty_local_owners_without_counting_unwritten_cache():
    values=[np.ones((2,3,2,3),np.float32) for _ in range(4)]
    a=cache_array(values)
    report=local_cache_span_comparison(a,a,np.array([[2,0,1]],np.int32),14,16)
    assert report['local_replica_rows']==report['local_replica_elements']==0
    assert report['bitwise_equal'] and report['finite'] and report['relative_l2']==0


def test_span_crosses_logical_page_and_expert_boundaries():
    owners=(7,7,0,0)
    values=[np.ones((2,3,2,3),np.float32) for _ in owners]
    changed=[x.copy() for x in values]
    # Rows 14/15 are in physical page 2 on owner7; 16/17 in page0 on owner0.
    for owner,x in zip(owners,changed):x[:,2 if owner==7 else 0,:,:]=2
    report=local_cache_span_comparison(cache_array(changed,owners),cache_array(values,owners),
        np.array([[2,0,1]],np.int32),14,18)
    assert report['local_replica_rows']==8 and report['local_replica_elements']==48
    assert report['differing_elements']==48 and report['relative_l2']==1


def test_span_sees_error_in_nonfirst_replica_and_nonfinite():
    values=[np.ones((2,3,2,3),np.float32) for _ in range(4)]
    a=cache_array(values)
    changed=[x.copy() for x in values];changed[3][:,0,0,:]=np.nan
    report=local_cache_span_comparison(cache_array(changed),a,np.array([[0,1,2]],np.int32),2,3)
    assert not report['finite'] and not report['bitwise_equal']
    assert report['differing_elements']==6 and report['local_replica_elements']==12


@pytest.mark.parametrize('table,start,stop',[
    (np.array([[0,0,2]],np.int32),0,1),
    (np.array([[0,1,3]],np.int32),0,1),
    (np.array([[0,1,2]],np.int64),0,1),
    (np.array([[0,1,2]],np.int32),-1,1),
    (np.array([[0,1,2]],np.int32),0,49),
    (np.array([[0,1,2]],np.int32),1,1),
])
def test_span_refuses_invalid_mapping_or_range(table,start,stop):
    a=cache_array([np.ones((2,3,2,3),np.float32) for _ in range(4)])
    with pytest.raises(ValueError,match='valid WS32 span'):
        local_cache_span_comparison(a,a,table,start,stop)


def test_span_refuses_unexpected_partition():
    a=cache_array([np.ones((2,3,2,3),np.float32) for _ in range(4)])
    a.addressable_shards[-1].index=(slice(0,1),slice(None),slice(2,4),slice(None))
    with pytest.raises(ValueError,match='expert row sharding'):
        local_cache_span_comparison(a,a,np.array([[0,1,2]],np.int32),0,3)
