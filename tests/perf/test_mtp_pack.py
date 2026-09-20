"""Exact native binary placement/loading and corruption refusal on tiny payloads."""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import Ws32RuntimeTensorPlan
from glm_tpu.greenfield.partitioning.source_inventory import SourceTensor,SourceFile
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from glm_tpu.greenfield.sharding.ws32 import Ws32PhysicalMesh
from glm_tpu.perf.mtp_checkpoint import mtp_source_placements
from glm_tpu.perf.mtp_pack import NativePackPlan,canonical,pack_native_slots,verify_native_pack
from tests.greenfield.checkpoint.test_ws32_runtime_checkpoint import _geometry


def fixture():
    geometry=replace(_geometry(),dsa_top_k=128,max_position_embeddings=128)
    config=Ws32DecoderConfig(geometry,32,packed_cache_width=70,sparse_segment_block=128,host_main_rope_table=True)
    arrays={
      'model.layers.1.enorm.weight':np.arange(8,dtype=np.float32).astype(ml_dtypes.bfloat16),
      'model.layers.1.eh_proj.weight':(np.arange(128,dtype=np.float32).reshape(8,16)/16).astype(ml_dtypes.bfloat16),
    }
    for e in range(8):arrays[f'model.layers.1.mlp.experts.{e}.gate_proj.weight']=np.arange(e*8,e*8+32,dtype=np.uint8).reshape(4,8)
    sources=[];payload=b''
    for name,value in arrays.items():
        dtype='F8_E4M3' if value.dtype==np.uint8 else 'BF16';raw=value.tobytes()
        sources.append(SourceTensor(name,'fixture.safetensors',dtype,value.shape,len(payload),len(payload)+len(raw)))
        payload+=raw
    file=SourceFile('fixture.safetensors',8+len(payload),8,len(payload),len(sources),'a'*64)
    ps=tuple(p for s in sources for p in mtp_source_placements(s,config))
    schemas={p.destination_name:p for p in ps if p.slot==0};tensors=[];offset=0
    for name,p in sorted(schemas.items()):
        end=offset+prod(p.destination_shape)*{'U8':1,'BF16':2}[p.destination_dtype]
        tensors.append(Ws32RuntimeTensorPlan(name,p.destination_dtype,p.destination_shape,p.global_shape,p.partition_spec,offset,end));offset=end
    plan=NativePackPlan('a'*64,tuple(sources),(file,),ps,tuple(tensors),'b'*64)
    calls=[];content=b'header00'+payload
    def read(source_file,start,length):
        assert source_file==file;calls.append((start,length));return content[start:start+length]
    return plan,read,calls,arrays


def physical():
    rows=tuple(tuple(range(i*4,i*4+4)) for i in range(8))
    return Ws32PhysicalMesh(device_ids=rows,feature_groups=rows,
        expert_groups=tuple(tuple(i*4+j for i in range(8)) for j in range(4)))


def test_native_all_slot_bytes_match_independent_slices(tmp_path):
    plan,read,calls,arrays=fixture();root=tmp_path/'native';mesh=physical()
    digest=pack_native_slots(plan,root,range(32),read_source=read,mesh_sha256=mesh.mesh_hash,source_identity={'fixture':'bounded'})
    manifest=verify_native_pack(root,digest,plan,slots=range(32),mesh_sha256=mesh.mesh_hash)
    assert len(calls)==len(plan.sources)
    for slot in range(32):
        e,f=divmod(slot,4);content=(root/f'slot-{slot:02d}.bin').read_bytes()
        for t in plan.tensors:
            raw=content[t.data_offset_start:t.data_offset_end]
            if t.name=='mtp.enorm.weight':expected=arrays['model.layers.1.enorm.weight'][f*2:(f+1)*2]
            elif t.name=='mtp.eh_proj.weight':expected=arrays['model.layers.1.eh_proj.weight'][f*2:(f+1)*2,:]
            else:expected=arrays[f'model.layers.1.mlp.experts.{e}.gate_proj.weight'][:,f*2:(f+1)*2][None]
            assert raw==expected.tobytes()
            assert manifest['files'][str(slot)]['tensor_sha256'][t.name]==sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):pack_native_slots(plan,root,range(32),read_source=read,mesh_sha256=mesh.mesh_hash,source_identity={'fixture':True})


def test_native_subset_reads_only_needed_expert_and_replicated_tables(tmp_path):
    plan,read,calls,_=fixture();mesh=physical();root=tmp_path/'native'
    digest=pack_native_slots(plan,root,(16,17,18,19),read_source=read,mesh_sha256=mesh.mesh_hash,source_identity={'fixture':True})
    assert len(calls)==3
    verify_native_pack(root,digest,plan,slots=(16,17,18,19),mesh_sha256=mesh.mesh_hash)
    with pytest.raises(ValueError,match='owner identity'):verify_native_pack(root,digest,plan,slots=range(32),mesh_sha256=mesh.mesh_hash)
    with pytest.raises(ValueError,match='digest'):verify_native_pack(root,'0'*64,plan,slots=(16,17,18,19),mesh_sha256=mesh.mesh_hash)
    path=root/'slot-19.bin';raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='file identity'):verify_native_pack(root,digest,plan,slots=(16,17,18,19),mesh_sha256=mesh.mesh_hash)


@pytest.mark.parametrize('kind',['short','nonfinite'])
def test_source_failure_does_not_publish_manifest(tmp_path,kind):
    plan,read,_,_=fixture();root=tmp_path/'native'
    def fail(file,start,count):
        raw=read(file,start,count)
        return raw[:-1] if kind=='short' else b'\xff'*len(raw)
    with pytest.raises(ValueError):pack_native_slots(plan,root,(0,),read_source=fail,mesh_sha256='a'*64,source_identity={'fixture':True})
    assert not (root/'manifest.json').exists()
    assert (root/'slot-00.partial').exists()


def test_native_direct_loader_cpu32(tmp_path):
    plan,read,_,_=fixture();root=tmp_path/'native';mesh=physical()
    digest=pack_native_slots(plan,root,range(32),read_source=read,mesh_sha256=mesh.mesh_hash,source_identity={'fixture':True})
    code=r'''
import jax,numpy as np
from pathlib import Path
from jax.sharding import Mesh
from tests.perf.test_mtp_pack import fixture,physical
from glm_tpu.perf.mtp_pack import load_native_arrays
plan,_,_,expected=fixture();physical=physical()
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
arrays=load_native_arrays(Path(ROOT),DIGEST,plan,mesh=mesh,physical_mesh=physical)
wanted={'mtp.enorm.weight':expected['model.layers.1.enorm.weight'],
        'mtp.eh_proj.weight':expected['model.layers.1.eh_proj.weight'],
        'model.layers.0.mlp.experts.gate_proj.weight_bits':np.stack([expected[f'model.layers.1.mlp.experts.{e}.gate_proj.weight'] for e in range(8)])}
assert set(arrays)==set(wanted)
for name,array in arrays.items():
    np.testing.assert_array_equal(np.asarray(array),wanted[name])
    assert len(array.addressable_shards)==32
    for s in array.addressable_shards:np.testing.assert_array_equal(np.asarray(s.data),wanted[name][s.index])
print('native exact raw arrays on all32 final owners; no target I/O loaded')
'''.replace('ROOT',repr(str(root))).replace('DIGEST',repr(digest))
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=120)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('change',['mesh','plan','schema','source_missing','source_range','tensor_hash_missing','filename'])
def test_rehashed_manifest_cannot_change_expected_plan_or_coverage(tmp_path,change):
    plan,read,_,_=fixture();root=tmp_path/'native';mesh=physical()
    digest=pack_native_slots(plan,root,(0,),read_source=read,mesh_sha256=mesh.mesh_hash,source_identity={'fixture':True})
    path=root/'manifest.json';value=json.loads(path.read_text())
    if change=='mesh':value['mesh_sha256']='0'*64
    elif change=='plan':value['plan_sha256']='0'*64
    elif change=='schema':value['tensors'][0]['global_shape'][0]+=1
    elif change=='source_missing':value['source_reads'].pop()
    elif change=='source_range':value['source_reads'][0]['start']+=1
    elif change=='tensor_hash_missing':value['files']['0']['tensor_sha256'].pop(next(iter(value['files']['0']['tensor_sha256'])))
    else:value['files']['0']['filename']='../elsewhere'
    path.write_bytes(canonical(value)+b'\n');digest=sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):verify_native_pack(root,digest,plan,slots=(0,),mesh_sha256=mesh.mesh_hash)
