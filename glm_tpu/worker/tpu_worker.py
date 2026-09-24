"""Private ordinary worker; source staging and fleet leases belong to its controller."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import time

from glm_tpu.engine import _s3_user_request as legacy
from glm_tpu.config.site import SiteConfig, get_current_site, set_current_site
from glm_tpu.engine import request
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.config import _s3_model as model
from glm_tpu.config._s3_model import site_args

# The staged source root this worker runs from, and its own path in the source manifest (both
# derived from this file; source_root refuses a file that is not WORKER_MODULE's).
REPO = protocol.source_root(__file__, protocol.WORKER_MODULE)
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()
# Site values (run root, model path, fleet naming) come from the run's staged,
# controller-resolved site.json (--site-sha256); workers never read $HOME config.


def persist(path, value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def private(path):
    legacy._plain(path)
    info=path.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise ValueError('optimized input namespace must be owner-only')


def preflight(args):
    """Authenticate inputs and the entire deployed archive before opening devices."""
    root=args.output
    if (os.environ.get(protocol.WORKER_ENV_FLAG)!='1'
            or re.fullmatch(r'optimized_request_[0-9]{8}T[0-9]{12}Z',root.name) is None
            or re.fullmatch(r'[0-9a-f]{40}',args.code_hash) is None
            or not 1<=args.wall_seconds<=86400):
        raise ValueError('protected optimized controller identity required')
    private(root)
    for name in ('request.json','source_manifest.json','site.json','topology_rebinding.json'):
        private(root/name)
    site=SiteConfig.from_staged(root/'site.json',args.site_sha256)
    if root.parent!=site.paths.run_root:
        raise ValueError('protected optimized controller identity required')
    set_current_site(site)
    host,port=args.coordinator_address.rsplit(':',1)
    ipaddress.ip_address(host)
    if port!='8476':raise ValueError('coordinator port differs')
    if args.coordinator_address!=site.fleet.coordinator_address:
        raise ValueError('coordinator address differs from the staged site')
    raw=legacy.read_bounded(root/'source_manifest.json',4<<20)
    if sha256(raw).hexdigest()!=args.source_manifest_sha256:
        raise ValueError('source manifest digest differs')
    manifest=json.loads(raw)
    if not isinstance(manifest,dict) or SELF not in manifest:
        raise ValueError('source manifest missing worker')
    for name,digest in manifest.items():
        path=REPO/name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('deployed source differs')
    # Frozen-source comparison is done against Git by the controller before
    # archiving. This worker authenticates those exact bytes without requiring
    # a Git database on the eight archive deployments.
    model.verified_template(REPO, site.paths.model_path)
    value=request.read(root/'request.json',expected_sha256=args.request_file_sha256)
    rank=site.fleet.host_rank(socket.gethostname())
    if rank is None or not 0<=rank<site.fleet.num_hosts:raise ValueError('worker rank differs')
    if (root/f'runner.rank{rank}.json').exists() or (root/f'native.rank{rank}').exists():
        raise ValueError('worker cannot retry an existing namespace')
    args.process_id=rank
    args=site_args(args,site)
    model.require_site(args)
    from glm_tpu.distributed.topology import apply_topology_binding
    binding=apply_topology_binding(args,root,args.topology_rebinding_sha256)
    args.context_capacity=value['context_capacity']
    return value,binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    for name in ('code-hash','source-manifest-sha256','request-file-sha256',
                 'site-sha256','topology-rebinding-sha256','coordinator-address'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--keep-loaded',action='store_true')
    args=parser.parse_args(argv)
    os.umask(0o077)
    value,binding=preflight(args)
    if args.keep_loaded and value.get('schema')==request.CONCURRENT_SCHEMA:
        raise ValueError('resident mode currently uses sequential ordinary requests')
    if args.preflight_only:
        import sys
        import jax, jaxlib, libtpu, numpy
        paths=dict(python=Path(sys.executable),jax=Path(jax.__file__),
            jaxlib=Path(jaxlib.__file__).parent/'libjax_common.so',
            libtpu=Path(libtpu.__file__).parent/'libtpu.so')
        print(json.dumps(dict(hostname=socket.gethostname(),jax=jax.__version__,
            jaxlib=jaxlib.__version__,numpy=numpy.__version__,python=sys.version,
            sha256={name:sha256(path.read_bytes()).hexdigest() for name,path in paths.items()})))
        return 0
    root,rank=args.output,args.process_id
    record=dict(schema='glm_optimized_worker_v1',code_hash=args.code_hash,
        source_manifest_sha256=args.source_manifest_sha256,
        request_file_sha256=args.request_file_sha256,request_sha256=value['request_sha256'],
        rank=rank,hostname=socket.gethostname(),topology_binding=binding,complete=False)
    path=root/f'runner.rank{rank}.json'
    persist(path,record)
    try:
        from glm_tpu.distributed import parallel_state
        from glm_tpu.runner.tpu_runner import OrdinaryRuntime
        started=time.perf_counter()
        jax,mesh,physical,topology,fleet_sha=parallel_state._initialize_runtime(args)
        native=root/f'native.rank{rank}';native.mkdir()
        pending=request.requests(value)
        concurrent=value.get('schema')==request.CONCURRENT_SCHEMA
        runtime=OrdinaryRuntime(args=args,repo=REPO,root=native,mesh=mesh,
            physical=physical,topology=topology,fleet_sha=fleet_sha,
            vote=parallel_state._batched_fleet_all,save=lambda v:persist(native/'runtime.json',v),
            context_capacity=value['context_capacity'],
            **(dict(concurrent_size=len(pending)) if concurrent else {}))
        deadline=started+args.wall_seconds
        # Warm the actual graphs on a disposable request/cache. No warm tokens
        # are delivered; the measured request always starts from fresh state.
        if concurrent:
            reports,aggregate=run_concurrent(runtime,pending,root,rank,deadline)
        else:
            def save_requests(reports):
                record['requests']=reports;persist(path,record)
            reports=run_queued(runtime,pending,value,root,rank,deadline,save=save_requests)
        report=reports[0] if value.get('schema') not in (request.BATCH_SCHEMA,request.CONCURRENT_SCHEMA) else dict(
            requests=reports,emitted=sum(r['emitted'] for r in reports),
            token_sha256=sha256(legacy.canonical([r['token_sha256'] for r in reports])).hexdigest(),
            scheduling='batched concurrent decode' if concurrent else 'queued; one request generates at a time',
            context_capacity=value['context_capacity'],**(dict(batch=aggregate) if concurrent else {}))
        record.update(complete=True,request=report,requests=reports,jax_process_index=jax.process_index(),
            cold_load_compile_seconds=runtime.record['cold_load_compile_seconds'],
            worker_wall_seconds=time.perf_counter()-started,
            programs=runtime.record['programs'],physical_identity=runtime.record['physical_identity'])
        persist(path,record)
        if args.keep_loaded:
            resident_loop(runtime,record,root,rank,args.wall_seconds)
        return 0
    except Exception as exc:
        import traceback
        with (root/f'failure.rank{rank}.log').open('x') as stream:
            stream.write(traceback.format_exc())
        record['failure_type']=type(exc).__name__;persist(path,record)
        # The private originals contain partial delivery. No retry or traceback
        # with user inputs is emitted through the controller channel.
        return 1


def resident_loop(runtime,record,root,rank,wall_seconds,*,stream=None):
    """Reuse one loaded runtime via its controller-owned stdin; no network listener."""
    import sys
    stream=sys.stdin if stream is None else stream
    sequence=0
    while True:
        # All result files exist before rank zero announces this round ready.
        runtime.phase('resident_ready',lambda:None)
        if rank==0:persist(root/'resident-ready.json',dict(sequence=sequence))
        line=stream.readline(legacy.PAYLOAD_CAP+1024)
        def decode_command():
            if not line or len(line)>legacy.PAYLOAD_CAP+512:
                raise ValueError('resident controller disconnected or oversized command')
            command=json.loads(line)
            if command=={'stop':True}:return None
            if set(command)!={'sequence','request'} or command['sequence']!=sequence+1:
                raise ValueError('resident sequence differs')
            value=command['request'];request.validate_payload(value)
            if value['context_capacity']!=runtime.capacity or value.get('schema')==request.CONCURRENT_SCHEMA:
                raise ValueError('resident request must match loaded ordinary context')
            return value
        value=runtime.phase('resident_command',decode_command)
        if value is None:return
        sequence+=1
        job=root/f'resident-{sequence:04d}'
        runtime.phase('resident_directory',lambda:job.mkdir(mode=0o700))
        deadline=time.perf_counter()+wall_seconds
        reports=run_queued(runtime,request.requests(value),value,job,rank,deadline,
                           save=lambda reports:None,warmup=False)
        report=reports[0] if value.get('schema')!=request.BATCH_SCHEMA else dict(
            requests=reports,emitted=sum(r['emitted'] for r in reports),
            token_sha256=sha256(legacy.canonical([r['token_sha256'] for r in reports])).hexdigest())
        result=dict(record,request_sha256=value['request_sha256'],request=report,requests=reports,
                    resident_sequence=sequence,complete=True)
        persist(job/f'runner.rank{rank}.json',result)


def run_queued(runtime,pending,value,root,rank,deadline,*,save,warmup=True):
    from transformers import AutoTokenizer
    first=pending[0]
    warm_ids=first['prompt_ids'] if value['context_capacity']==request.CAPACITY else first['prompt_ids'][:128]
    warm=request.from_token_ids(warm_ids,request_id=first['request_id'],
                                max_new_tokens=min(2,first['max_new_tokens']),
                                context_capacity=value['context_capacity'])
    if warmup:runtime.generate(warm,deliver=lambda event:None,deadline=deadline)
    reports=[]
    for index,item in enumerate(pending):
        item_root=root/f'item{index:03d}' if value.get('schema')==request.BATCH_SCHEMA else root
        if item_root!=root:runtime.phase('request_directory',lambda:item_root.mkdir(mode=0o700))
        stream=runtime.phase('open_tokens',lambda:(item_root/'tokens.jsonl').open('x') if rank==0 else None)
        def deliver(event):
            if stream is not None:
                stream.write(json.dumps(asdict(event),sort_keys=True)+'\n');stream.flush()
        try:tokens,report=runtime.generate(item,deliver=deliver,deadline=deadline)
        finally:
            if stream is not None:stream.close()
        def write_answer():
            if rank==0:
                tokenizer=AutoTokenizer.from_pretrained(get_current_site().paths.model_path,local_files_only=True,trust_remote_code=False)
                with (item_root/'answer.txt').open('x') as stream:
                    stream.write(tokenizer.decode(tokens.tolist(),skip_special_tokens=False))
        runtime.phase('write_answer',write_answer)
        report.update(request_id=item['request_id'],output_directory=item_root.name,
            output_budget_tokens=item['max_new_tokens'],
            stop_cause=request.stop_cause(item,report['finish_reason']))
        reports.append(report)
        save(reports)
    return reports


def run_concurrent(runtime,pending,root,rank,deadline):
    from contextlib import ExitStack
    from transformers import AutoTokenizer
    warm=[request.from_token_ids(item['prompt_ids'][:128],request_id=item['request_id'],
        max_new_tokens=min(2,item['max_new_tokens']),context_capacity=item['context_capacity'])
        for item in pending]
    runtime.generate_concurrent(warm,deliver=lambda *args:None,deadline=deadline)
    with ExitStack() as stack:
        directories=[];streams=[]
        for index in range(len(pending)):
            item_root=root/f'item{index:03d}'
            runtime.phase('request_directory',lambda:item_root.mkdir(mode=0o700))
            directories.append(item_root)
            stream=runtime.phase('open_tokens',lambda:(item_root/'tokens.jsonl').open('x') if rank==0 else None)
            streams.append(stack.enter_context(stream) if stream is not None else None)
        def deliver(lane,event,round_index):
            stream=streams[lane]
            if stream is not None:
                stream.write(json.dumps(dict(asdict(event),batch_round=round_index),sort_keys=True)+'\n')
                stream.flush()
        results,aggregate=runtime.generate_concurrent(pending,deliver=deliver,deadline=deadline)
    reports=[]
    for item,item_root,(tokens,report) in zip(pending,directories,results,strict=True):
        def write_answer():
            if rank==0:
                tokenizer=AutoTokenizer.from_pretrained(get_current_site().paths.model_path,local_files_only=True,trust_remote_code=False)
                with (item_root/'answer.txt').open('x') as stream:
                    stream.write(tokenizer.decode(tokens.tolist(),skip_special_tokens=False))
        runtime.phase('write_answer',write_answer)
        report.update(request_id=item['request_id'],output_directory=item_root.name,
            output_budget_tokens=item['max_new_tokens'],
            stop_cause=request.stop_cause(item,report['finish_reason']))
        reports.append(report)
    return reports,aggregate


if __name__=='__main__':
    raise SystemExit(main())
