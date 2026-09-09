"""Production-shape addressable capture and mutation checks, no TPU device."""

from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_capture as capture

SLOTS = {9:0, 13:1, 25:8, 29:31}


def outputs():
    shapes=((128,6144),(128,6144),(8,16,64,640),(8,16,64,128),(8,16,64,128),
            (128,2048),(128,),(128,2048),(128,8),(128,8),(8,4,128),(128,6144))
    dtypes=(ml_dtypes.bfloat16,)*5+(np.int32,np.int32,np.float32,np.int32,np.float32,np.bool_,ml_dtypes.bfloat16)
    result=[]
    for _ in (0,1):
        layer=[]
        for field,(shape,dtype) in enumerate(zip(shapes,dtypes,strict=True)):
            shards=[]
            for device,slot in SLOTS.items():
                index=[slice(None)]*len(shape)
                if field in (2,3,4,10):index[0]=slice(slot//4,slot//4+1)
                if field==10:index[1]=slice(slot%4,slot%4+1)
                if field in (0,1,11):index[1]=slice(slot%4*1536,(slot%4+1)*1536)
                local=tuple(len(range(*s.indices(n))) for s,n in zip(index,shape))
                data=np.full(local,field==10,dtype=dtype)
                shards.append(NS(device=NS(id=device,platform='tpu',process_index=3),
                                 index=tuple(index),data=data))
            layer.append(NS(shape=shape,dtype=np.dtype(dtype),addressable_shards=shards))
        result.append(tuple(layer))
    return tuple(result)


def test_endpoint_complete_cache_bytes_and_live_rows():
    arrays,report=capture.capture(outputs(),local_slots=SLOTS,process_index=3,count=128,keep_caches=True)
    assert report['valid'] and len(arrays)==96
    assert sum(v.nbytes for v in arrays.values()) < 48<<20
    for slot in SLOTS.values():
        caches=capture.cache_bits(arrays,slot)
        assert caches['kv'].shape==(2,16,64,640)
        assert caches['kv'].dtype==np.uint16
        assert arrays[f'slot{slot}_layer0__output'].shape==(128,1536)


def test_intermediate_does_not_read_cache_payload():
    result=outputs()
    class Unreadable:
        def __init__(self,value):self.shape,self.dtype=value.shape,value.dtype
        def __array__(self,*args,**kwargs):raise AssertionError('downloaded intermediate cache')
    for layer in result:
        for field in (2,3,4):
            for shard in layer[field].addressable_shards:shard.data=Unreadable(shard.data)
    arrays,report=capture.capture(result,local_slots=SLOTS,process_index=3,count=32,keep_caches=False)
    assert report['valid'] and len(arrays)==72
    assert arrays['slot0_layer0__output'].shape==(32,1536)
    assert arrays['slot0_layer0__health'].shape==(128,)


@pytest.mark.parametrize('field',[0,1,2,3,4,9,11])
def test_nonfinite_originals_returned_before_refusal(field):
    result=outputs()
    result[1][field].addressable_shards[0].data.flat[0]=np.nan
    arrays,report=capture.capture(result,local_slots=SLOTS,process_index=3,count=128,keep_caches=True)
    assert not report['valid'] and report['errors'] and arrays


@pytest.mark.parametrize('mutation',['index','device','platform','process','missing','global_shape'])
def test_physical_identity_or_layout_refuses(mutation):
    result=outputs()
    array=result[0][2]
    shard=array.addressable_shards[0]
    if mutation=='index':shard.index=(slice(1,2),*shard.index[1:])
    elif mutation=='device':shard.device.id=987
    elif mutation=='platform':shard.device.platform='cpu'
    elif mutation=='process':shard.device.process_index=4
    elif mutation=='missing':array.addressable_shards.pop()
    else:array.shape=(8,8,64,640)
    with pytest.raises(ValueError):
        capture.capture(result,local_slots=SLOTS,process_index=3,count=128,keep_caches=True)
