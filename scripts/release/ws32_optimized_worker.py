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

from glm_tpu import user_request as legacy
from glm_tpu.optimized import request, model
from scripts.release.ws32_user_worker import site_args

TOKENIZER = model.TOKENIZER_ROOT

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = Path('/home/gianl/glm-run')


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
    if (os.environ.get('GLM_OPTIMIZED_REQUEST')!='1'
            or root.parent!=RUN_ROOT
            or re.fullmatch(r'optimized_request_[0-9]{8}T[0-9]{12}Z',root.name) is None
            or re.fullmatch(r'[0-9a-f]{40}',args.code_hash) is None
            or not 1<=args.wall_seconds<=86400):
        raise ValueError('protected optimized controller identity required')
    private(root)
    for name in ('request.json','source_manifest.json','topology_rebinding.json'):
        private(root/name)
    host,port=args.coordinator_address.rsplit(':',1)
    ipaddress.ip_address(host)
    if port!='8476':raise ValueError('coordinator port differs')
    raw=legacy.read_bounded(root/'source_manifest.json',4<<20)
    if sha256(raw).hexdigest()!=args.source_manifest_sha256:
        raise ValueError('source manifest digest differs')
    manifest=json.loads(raw)
    if not isinstance(manifest,dict) or 'scripts/release/ws32_optimized_worker.py' not in manifest:
        raise ValueError('source manifest missing worker')
    for name,digest in manifest.items():
        path=REPO/name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('deployed source differs')
    # Frozen-source comparison is done against Git by the controller before
    # archiving. This worker authenticates those exact bytes without requiring
    # a Git database on the eight archive deployments.
    model.verified_template(REPO, TOKENIZER)
    value=request.read(root/'request.json',expected_sha256=args.request_file_sha256)
    rank=int(socket.gethostname().rsplit('-w-',1)[1])
    if not 0<=rank<8:raise ValueError('worker rank differs')
    if (root/f'runner.rank{rank}.json').exists() or (root/f'native.rank{rank}').exists():
        raise ValueError('worker cannot retry an existing namespace')
    args.process_id=rank
    args=site_args(args)
    model.require_site(args)
    from glm_tpu.optimized.topology_binding import apply_topology_binding
    binding=apply_topology_binding(args,root,args.topology_rebinding_sha256)
    args.context_capacity=value['context_capacity']
    return value,binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    for name in ('code-hash','source-manifest-sha256','request-file-sha256',
                 'topology-rebinding-sha256','coordinator-address'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv)
    os.umask(0o077)
    value,binding=preflight(args)
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
        from scripts.greenfield import run_short_decoder_ws32 as original
        from glm_tpu.optimized.runtime import OrdinaryRuntime
        started=time.perf_counter()
        jax,mesh,physical,topology,fleet_sha=original._initialize_runtime(args)
        native=root/f'native.rank{rank}';native.mkdir()
        pending=request.requests(value)
        concurrent=value.get('schema')==request.CONCURRENT_SCHEMA
        runtime=OrdinaryRuntime(args=args,repo=REPO,root=native,mesh=mesh,
            physical=physical,topology=topology,fleet_sha=fleet_sha,
            vote=original._batched_fleet_all,save=lambda v:persist(native/'runtime.json',v),
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
        return 0
    except Exception as exc:
        import traceback
        with (root/f'failure.rank{rank}.log').open('x') as stream:
            stream.write(traceback.format_exc())
        record['failure_type']=type(exc).__name__;persist(path,record)
        # The private originals contain partial delivery. No retry or traceback
        # with user inputs is emitted through the controller channel.
        return 1


def run_queued(runtime,pending,value,root,rank,deadline,*,save):
    from transformers import AutoTokenizer
    first=pending[0]
    warm_ids=first['prompt_ids'] if value['context_capacity']==request.CAPACITY else first['prompt_ids'][:128]
    warm=request.from_token_ids(warm_ids,request_id=first['request_id'],
                                max_new_tokens=min(2,first['max_new_tokens']),
                                context_capacity=value['context_capacity'])
    runtime.generate(warm,deliver=lambda event:None,deadline=deadline)
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
                tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,trust_remote_code=False)
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
                tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,trust_remote_code=False)
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
