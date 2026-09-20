from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import struct
import pytest
from glm_tpu.perf.mtp_source_reader import NativeGenerationReader
from tests.perf.test_mtp_pack import fixture


def setup():
    plan,original,_,_=fixture();raw_header=b'12345678'
    file=replace(plan.files[0],header_bytes=16,file_bytes=plan.files[0].payload_bytes+16,header_sha256=sha256(raw_header).hexdigest())
    plan=replace(plan,files=(file,));calls=[]
    class Blob:
        generation=77;size=file.file_bytes;crc32c='fixture-crc';content_encoding=None
        def reload(self,**kw):calls.append(('metadata',kw))
        def download_as_bytes(self,**kw):
            calls.append(('range',kw))
            if kw['start']==0:return struct.pack('<Q',8)+raw_header
            old=fixture()[0].files[0]
            return original(old,kw['start']-8,kw['end']-kw['start']+1)
    blob=Blob()
    class Bucket:
        location='US-CENTRAL2'
        def reload(self,**kw):calls.append(('bucket',kw))
        def blob(self,name,**kw):
            assert name=='models/GLM-5.2-FP8/fixture.safetensors' and kw=={'generation':77}
            return blob
    bucket=Bucket()
    class Client:
        def bucket(self,name):assert name=='driftbench-dsv4-uc';return bucket
    audit=dict(source_inventory_sha256=plan.source_inventory_sha256,canonical_source_uri='gs://driftbench-dsv4-uc/models/GLM-5.2-FP8',
        objects=[dict(object='models/GLM-5.2-FP8/fixture.safetensors',generation=77,bytes=file.file_bytes,crc32c='fixture-crc',sealed_sha256='d'*64)])
    return plan,audit,Client(),blob,bucket,calls


def test_only_native_ranges_from_exact_audited_generation():
    plan,audit,client,blob,bucket,calls=setup()
    reader=NativeGenerationReader(plan,audit,client=client)
    source=plan.sources[0];file=plan.files[0]
    raw=reader(file,file.header_bytes+source.data_offset_start,source.byte_count)
    assert len(raw)==source.byte_count
    for kind,kwargs in calls:
        assert kwargs['retry'] is None
        if kind!='bucket':assert kwargs['if_generation_match']==77
    with pytest.raises(ValueError,match='audited native'):reader(file,0,file.file_bytes)
    assert reader.identity['range_reads_generation_bound']
    assert not reader.identity['full_source_object_sha256_recomputed']


@pytest.mark.parametrize('change',['region','generation','size','crc','encoding','header'])
def test_generation_header_and_location_refusal(change):
    plan,audit,client,blob,bucket,_=setup()
    if change=='region':bucket.location='EU'
    elif change=='generation':blob.generation=78
    elif change=='size':blob.size-=1
    elif change=='crc':blob.crc32c='wrong'
    elif change=='encoding':blob.content_encoding='gzip'
    else:plan=replace(plan,files=(replace(plan.files[0],header_sha256='0'*64),))
    with pytest.raises(ValueError):NativeGenerationReader(plan,audit,client=client)
