"""Native-only checkpoint packing/loading with explicit source and owner identity.

No acquisition or TPU initialization occurs on import. The controller supplies
an authenticated generation-bound reader, workload/storage leases and manifest
pins. Files contain exact final-owner raw tables; base embedding/head are absent.
"""
from contextlib import ExitStack
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path

import numpy as np

from ..greenfield.checkpoint.ws32_runtime_checkpoint import (
    Ws32RuntimeTensorPlan,_flat_contiguous_offset,_pwrite_all,_validate_finite_chunk,
)
from ..greenfield.runtime.ws32_decoder import ws32_decoder_weight_names,_weight_name_leaves
from .mtp_checkpoint import mtp_source_placements,PROJECTION_NAMES
from .mtp_draft import mtp_config

_BYTES={'U8':1,'BF16':2,'F32':4}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def file_sha(path):
    h=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class NativePackPlan:
    source_inventory_sha256: str
    sources: tuple
    files: tuple
    placements: tuple
    tensors: tuple
    plan_sha256: str

    @property
    def payload_bytes_per_slot(self):return self.tensors[-1].data_offset_end


def build_native_pack_plan(inventory,target_config):
    if inventory.model_id!=target_config.geometry.model_id:
        raise ValueError('native inventory/model identity differs')
    sources=tuple(s for s in inventory.tensors if s.layer_id==target_config.geometry.num_layers)
    if not sources:raise ValueError('native source layer is absent')
    placements=tuple(p for source in sources for p in mtp_source_placements(source,target_config))
    expected=set(_weight_name_leaves(ws32_decoder_weight_names(mtp_config(target_config))))
    expected-={'model.embed_tokens.weight','lm_head.weight'}
    expected.update(PROJECTION_NAMES)
    schemas,intervals,written={},{},{}
    for p in placements:
        key=(p.slot,p.destination_name)
        schema=(p.destination_dtype,p.destination_shape,p.global_shape,p.partition_spec)
        if schemas.setdefault(key,schema)!=schema:raise ValueError('native destination schema differs')
        for starts,stops in intervals.setdefault(key,[]):
            if all(max(a,c)<min(b,d) for a,b,c,d in zip(starts,stops,p.destination_starts,p.destination_stops)):
                raise ValueError('native destination writes overlap')
        intervals[key].append((p.destination_starts,p.destination_stops))
        written[key]=written.get(key,0)+p.byte_count
    for slot in range(32):
        if {name for owner,name in schemas if owner==slot}!=expected:
            raise ValueError('native destination table set is incomplete')
        for name in expected:
            dtype,shape,*_=schemas[slot,name]
            if schemas[slot,name]!=schemas[0,name] or written[slot,name]!=prod(shape)*_BYTES[dtype]:
                raise ValueError('native destination coverage or per-slot schema differs')
    offset=0;tensors=[]
    for name in sorted(expected):
        dtype,local,global_shape,spec=schemas[0,name]
        end=offset+prod(local)*_BYTES[dtype]
        tensors.append(Ws32RuntimeTensorPlan(name,dtype,local,global_shape,spec,offset,end));offset=end
    digest=sha256(canonical(dict(inventory_sha256=inventory.inventory_sha256,
        sources=[s.to_dict() for s in sources],placements=[p.to_dict() for p in placements],
        tensors=[asdict(t) for t in tensors]))).hexdigest()
    filenames={s.filename for s in sources}
    return NativePackPlan(inventory.inventory_sha256,sources,
        tuple(f for f in inventory.files if f.filename in filenames),placements,tuple(tensors),digest)


def pack_native_slots(plan,output,slots,*,read_source,mesh_sha256,source_identity):
    """Write one host's owned slots; caller's reader authenticates each range.

    read_source(SourceFile,start,length) returns exactly one tensor's original
    bytes from a bound source generation. A new output directory is mandatory;
    failures leave partial originals and never publish a completion manifest.
    """
    slots=tuple(sorted(slots));output=Path(output)
    if not slots or len(set(slots))!=len(slots) or any(type(x) is not int or not 0<=x<32 for x in slots):
        raise ValueError('native owner slots must be unique integers in 0..31')
    if len(mesh_sha256)!=64 or not source_identity:raise ValueError('native source/mesh identity required')
    output.mkdir(mode=0o700,parents=False,exist_ok=False)
    schemas={t.name:t for t in plan.tensors};files={f.filename:f for f in plan.files}
    by_source={s.name:[] for s in plan.sources}
    for p in plan.placements:
        if p.slot in slots:by_source[p.source_name].append(p)
    source_reads=[]
    with ExitStack() as stack:
        handles={slot:stack.enter_context((output/f'slot-{slot:02d}.partial').open('xb+')) for slot in slots}
        for f in handles.values():f.truncate(plan.payload_bytes_per_slot)
        for source in plan.sources:
            placements=by_source[source.name]
            if not placements:continue
            start=files[source.filename].header_bytes+source.data_offset_start
            raw=read_source(files[source.filename],start,source.byte_count)
            if type(raw) is not bytes or len(raw)!=source.byte_count:raise ValueError('native source range length differs')
            _validate_finite_chunk(raw,'U8' if source.dtype=='F8_E4M3' else source.dtype)
            source_reads.append(dict(name=source.name,filename=source.filename,start=start,bytes=len(raw),sha256=sha256(raw).hexdigest()))
            elements=np.frombuffer(raw,dtype=f'V{source.byte_count//prod(source.shape)}').reshape(source.shape)
            for p in placements:
                tensor=schemas[p.destination_name]
                selected=elements[tuple(slice(a,b) for a,b in zip(p.source_starts,p.source_stops))].tobytes(order='C')
                if len(selected)!=p.byte_count:raise ValueError('native slice byte count differs')
                flat,elements_count=_flat_contiguous_offset(p.destination_shape,p.destination_starts,p.destination_stops)
                if elements_count*_BYTES[tensor.dtype]!=len(selected):raise ValueError('native contiguous span differs')
                _pwrite_all(handles[p.slot].fileno(),selected,tensor.data_offset_start+flat*_BYTES[tensor.dtype])
        for f in handles.values():f.flush();os.fsync(f.fileno())
    records={}
    for slot in slots:
        path=output/f'slot-{slot:02d}.partial';hashes={}
        with path.open('rb') as f:
            for t in plan.tensors:
                f.seek(t.data_offset_start);raw=f.read(t.byte_count)
                if len(raw)!=t.byte_count:raise ValueError('native packed tensor truncated')
                _validate_finite_chunk(raw,t.dtype);hashes[t.name]=sha256(raw).hexdigest()
        final=output/f'slot-{slot:02d}.bin';path.rename(final)
        records[str(slot)]=dict(filename=final.name,bytes=final.stat().st_size,
                               sha256=file_sha(final),tensor_sha256=hashes)
    manifest=dict(schema='glm_native_mtp_pack_v1',plan_sha256=plan.plan_sha256,
        source_inventory_sha256=plan.source_inventory_sha256,source_identity=source_identity,
        mesh_sha256=mesh_sha256,slots=list(slots),tensors=[asdict(t) for t in plan.tensors],
        files=records,source_reads=source_reads,shared_target_io_included=False)
    raw=canonical(manifest)+b'\n'
    with (output/'manifest.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return sha256(raw).hexdigest()


def verify_native_pack(root,manifest_sha256,plan,*,slots,mesh_sha256):
    root=Path(root);raw=(root/'manifest.json').read_bytes()
    if sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('native manifest digest differs')
    value=json.loads(raw)
    if (set(value)!={'schema','plan_sha256','source_inventory_sha256','source_identity',
                    'mesh_sha256','slots','tensors','files','source_reads','shared_target_io_included'}
            or not value.get('source_identity')
            or value.get('schema')!='glm_native_mtp_pack_v1' or value.get('plan_sha256')!=plan.plan_sha256
            or value.get('source_inventory_sha256')!=plan.source_inventory_sha256
            or value.get('mesh_sha256')!=mesh_sha256 or value.get('slots')!=sorted(slots)
            or canonical(value.get('tensors'))!=canonical([asdict(t) for t in plan.tensors])
            or value.get('shared_target_io_included') is not False
            or set(value.get('files',{}))!={str(x) for x in slots}):
        raise ValueError('native manifest plan or owner identity differs')
    needed={p.source_name for p in plan.placements if p.slot in slots}
    reads=value['source_reads'];sources={s.name:s for s in plan.sources};files={f.filename:f for f in plan.files}
    if len(reads)!=len(needed) or {r.get('name') for r in reads}!=needed:
        raise ValueError('native source read coverage differs')
    for row in reads:
        s=sources[row['name']]
        if (set(row)!={'name','filename','start','bytes','sha256'} or row['filename']!=s.filename
                or row['start']!=files[s.filename].header_bytes+s.data_offset_start
                or row['bytes']!=s.byte_count or type(row['sha256']) is not str
                or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('native source read identity differs')
    for slot in slots:
        row=value['files'][str(slot)];name=f'slot-{slot:02d}.bin';path=root/name
        if (row.get('filename')!=name or path.is_symlink() or not path.is_file()
                or path.stat().st_size!=plan.payload_bytes_per_slot or row.get('bytes')!=plan.payload_bytes_per_slot
                or file_sha(path)!=row.get('sha256')
                or set(row.get('tensor_sha256',{}))!={t.name for t in plan.tensors}):
            raise ValueError('native owner file identity differs')
    return value


def load_native_arrays(root,manifest_sha256,plan,*,mesh,physical_mesh):
    """Verify and direct-load only addressable owners; no base-table copies."""
    import jax
    import ml_dtypes
    from jax.sharding import NamedSharding,PartitionSpec as P
    if (tuple(mesh.axis_names)!=('expert','feature') or tuple(mesh.devices.shape)!=(8,4)
            or tuple(tuple(int(d.id) for d in row) for row in mesh.devices)!=physical_mesh.device_ids):
        raise ValueError('native physical mesh identity differs')
    slot_by_id={int(d.id):i for i,d in enumerate(mesh.devices.flat)}
    addressable=tuple(d for d in mesh.devices.flat if d in jax.local_devices())
    expected=32 if jax.default_backend()=='cpu' and jax.process_count()==1 else 4
    if len(addressable)!=expected or set(addressable)!=set(jax.local_devices()):
        raise ValueError('native addressable owner set differs')
    slots=tuple(sorted(slot_by_id[int(d.id)] for d in addressable))
    manifest=verify_native_pack(root,manifest_sha256,plan,slots=slots,mesh_sha256=physical_mesh.mesh_hash)
    root=Path(root);arrays={};dtypes={'U8':np.uint8,'F32':np.float32,'BF16':ml_dtypes.bfloat16}
    with ExitStack() as stack:
        handles={slot:stack.enter_context((root/f'slot-{slot:02d}.bin').open('rb')) for slot in slots}
        for t in plan.tensors:
            sharding=NamedSharding(mesh,P(*t.partition_spec))
            devices=tuple(sharding.addressable_devices_indices_map(t.global_shape))
            if set(devices)!=set(addressable) or sharding.shard_shape(t.global_shape)!=t.local_shape:
                raise ValueError('native tensor owner shape differs')
            locals_=[]
            for device in devices:
                slot=slot_by_id[int(device.id)];f=handles[slot]
                f.seek(t.data_offset_start);raw=f.read(t.byte_count)
                if sha256(raw).hexdigest()!=manifest['files'][str(slot)]['tensor_sha256'][t.name]:
                    raise ValueError('native tensor changed while loading')
                _validate_finite_chunk(raw,t.dtype)
                host=np.frombuffer(raw,dtype=dtypes[t.dtype]).reshape(t.local_shape)
                local=jax.device_put(host,device);local.block_until_ready()
                if tuple(local.devices())!=(device,):raise ValueError('native tensor missed its owner')
                locals_.append(local)
            arrays[t.name]=jax.make_array_from_single_device_arrays(t.global_shape,sharding,locals_)
    return arrays
