"""Read only audited native tensor ranges from immutable regional generations."""
from hashlib import sha256
import struct


class NativeGenerationReader:
    def __init__(self,plan,audit,*,client):
        if (audit.get('source_inventory_sha256')!=plan.source_inventory_sha256
                or audit.get('canonical_source_uri')!='gs://driftbench-dsv4-uc/models/GLM-5.2-FP8'):
            raise ValueError('native source audit identity differs')
        prefix='models/GLM-5.2-FP8/'
        files={f.filename:f for f in plan.files}
        objects=audit.get('objects',[])
        if (len(objects)!=len(files) or {r.get('object') for r in objects}!={prefix+n for n in files}):
            raise ValueError('native source object set differs')
        bucket=client.bucket('driftbench-dsv4-uc');bucket.reload(retry=None,timeout=60)
        if bucket.location!='US-CENTRAL2':raise ValueError('native source bucket region differs')
        self._files=files;self._blobs={};identities=[]
        for row in objects:
            filename=row['object'][len(prefix):];file=files[filename]
            generation=row.get('generation')
            if type(generation) is not int or generation<=0:raise ValueError('native source generation is invalid')
            blob=bucket.blob(row['object'],generation=generation)
            blob.reload(if_generation_match=generation,retry=None,timeout=60)
            if (int(blob.generation)!=generation or blob.size!=file.file_bytes or blob.size!=row.get('bytes')
                    or blob.crc32c!=row.get('crc32c') or getattr(blob,'content_encoding',None) not in (None,'identity')):
                raise ValueError('native source object generation/size/checksum differs')
            header=blob.download_as_bytes(start=0,end=file.header_bytes-1,
                if_generation_match=generation,raw_download=True,retry=None,timeout=180,checksum=None)
            if (len(header)!=file.header_bytes or len(header)<8
                    or struct.unpack('<Q',header[:8])[0]!=file.header_bytes-8
                    or sha256(header[8:]).hexdigest()!=file.header_sha256):
                raise ValueError('native source header differs from authenticated inventory')
            self._blobs[filename]=(blob,generation)
            identities.append(dict(object=row['object'],generation=generation,bytes=blob.size,
                crc32c=blob.crc32c,header_sha256=file.header_sha256,
                retained_sealed_sha256=row.get('sealed_sha256')))
        self._allowed={(s.filename,files[s.filename].header_bytes+s.data_offset_start,s.byte_count) for s in plan.sources}
        self.identity=dict(bucket='driftbench-dsv4-uc',location='US-CENTRAL2',objects=sorted(identities,key=lambda x:x['object']),
            range_reads_generation_bound=True,full_source_object_sha256_recomputed=False)

    def __call__(self,file,start,length):
        if (file!=self._files.get(file.filename) or (file.filename,start,length) not in self._allowed):
            raise ValueError('read is not an audited native tensor range')
        blob,generation=self._blobs[file.filename]
        raw=blob.download_as_bytes(start=start,end=start+length-1,if_generation_match=generation,
            raw_download=True,retry=None,timeout=180,checksum=None)
        if len(raw)!=length:raise ValueError('native range response is truncated')
        return raw
