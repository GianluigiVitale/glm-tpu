"""Leased native-MTP-only pack worker; never initializes JAX's TPU backend.

Controller must hold workload/pod/sync/cron leases, verify idle and disable SSH
retries. It provides an immutable source manifest and a pinned physical-owner
map from a prior authenticated run; later TPU loading rechecks the live mesh.
"""
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pack-base',type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True)
    p.add_argument('--owner-map-sha256',required=True)
    p.add_argument('--code-hash',required=True)
    args=p.parse_args();os.umask(0o077)
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('native pack must explicitly use CPU only')
    root=args.output.resolve()
    if root.is_relative_to(REPO) or not root.is_dir():raise ValueError('native receipts require an existing private run directory')
    for name,digest in (('source_manifest.json',args.source_manifest_sha256),('owner_map.json',args.owner_map_sha256)):
        if sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('native worker input digest differs')
    manifest=json.loads((root/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        path=REPO/name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('native immutable source file differs')
    owner_map=json.loads((root/'owner_map.json').read_text())
    rank=int(socket.gethostname().rsplit('-w-',1)[1])
    if (owner_map['hosts'][str(rank)]['hostname']!=socket.gethostname()
            or sorted(x for h in owner_map['hosts'].values() for x in h['slots'])!=list(range(32))):
        raise ValueError('native authenticated host/owner map differs')
    slots=tuple(owner_map['hosts'][str(rank)]['slots']);mesh_sha=owner_map['mesh_sha256']
    pack_base=args.pack_base.absolute()
    if (pack_base!=pack_base.resolve() or not pack_base.is_relative_to(Path('/dev/shm/glm-mtp-native'))
            or any(p.is_symlink() for p in (pack_base,*pack_base.parents))):
        raise ValueError('native pack must use a new private tmpfs child')
    pack_base.mkdir(mode=0o700,parents=True,exist_ok=True)
    destination=pack_base/f'rank{rank}'
    if destination.exists():raise ValueError('native output already exists; no retry or overwrite')
    record=dict(schema='glm_native_mtp_acquisition_rank_v1',rank=rank,hostname=socket.gethostname(),
        code_hash=args.code_hash,source_manifest_sha256=args.source_manifest_sha256,
        owner_map_sha256=args.owner_map_sha256,mesh_sha256=mesh_sha,slots=slots,
        output=str(destination),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        complete=False,tpu_initialized=False)
    receipt=root/f'acquisition.rank{rank}.json'
    def save():
        temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');temp.replace(receipt)
    save()
    try:
        from google.cloud import storage
        from glm_tpu.greenfield.partitioning.source_inventory import inspect_source_inventory
        from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
        from glm_tpu.greenfield.types import ModelGeometry
        from glm_tpu.perf.mtp_pack import build_native_pack_plan,pack_native_slots,verify_native_pack
        from glm_tpu.perf.mtp_source_reader import NativeGenerationReader
        audit=json.loads((REPO/'docs/perf/mtp-source-audit-20260919.json').read_text())
        inventory=inspect_source_inventory(Path('/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json'))
        if inventory.inventory_sha256!=audit['source_inventory_sha256']:raise ValueError('native source inventory identity differs')
        config=Ws32DecoderConfig(ModelGeometry.from_hf_config(json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_text())),8192,host_main_rope_table=True)
        plan=build_native_pack_plan(inventory,config)
        required=plan.payload_bytes_per_slot*len(slots)+3*max(s.byte_count for s in plan.sources)+(1<<30)
        fs=os.statvfs(pack_base)
        if fs.f_bavail*fs.f_frsize<required:raise ValueError('native pack lacks tmpfs headroom')
        record.update(plan_sha256=plan.plan_sha256,source_inventory_sha256=inventory.inventory_sha256,
            bytes_per_slot=plan.payload_bytes_per_slot,required_tmpfs_headroom=required)
        save();print('MTP_ACQUIRE_PLAN '+plan.plan_sha256,flush=True)
        reader=NativeGenerationReader(plan,audit,client=storage.Client())
        record['source_identity']=reader.identity;save()
        reads=[0,0];started=time.perf_counter()
        def read(file,start,length):
            raw=reader(file,start,length);reads[0]+=1;reads[1]+=len(raw)
            if reads[0]%100==0:print('MTP_ACQUIRE_READS '+str(reads[0]),flush=True)
            return raw
        digest=pack_native_slots(plan,destination,slots,read_source=read,mesh_sha256=mesh_sha,source_identity=reader.identity)
        verified=verify_native_pack(destination,digest,plan,slots=slots,mesh_sha256=mesh_sha)
        record.update(complete=True,manifest_sha256=digest,files=verified['files'],
            tensor_reads=reads[0],source_bytes_read=reads[1],seconds=time.perf_counter()-started,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save();print('MTP_ACQUIRE_DONE '+digest,flush=True)
    except Exception as exc:
        record['failure']=dict(kind=type(exc).__name__,message=str(exc));save();raise


if __name__=='__main__':main()
