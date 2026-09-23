"""One private greedy request on the retained 32-chip site; no automatic retries.

The site's [launch] policy decides which checkout may launch (branch patterns,
origin, clean, pushed; glm_tpu.executor.launch_policy), and the staged source is
`git archive` of the pinned commit of that checkout, which must be this
controller's own (--repo and the site's paths.repo may only name it: the
controller's code and the remote helper texts come from it). This controller
holds the workload leases
through authenticated cleanup, and never creates or resizes resources. Every
site value (fleet, interpreters, paths, pins, locks) comes from the validated
site file (--site, else $GLM_TPU_SITE_CONFIG, else
$GLM_TPU_CONFIG_ROOT/site.toml); the resolved configuration is staged to every
host as site.json and bound by --site-sha256. Remote work is done by the
stdlib helper programs of glm_tpu.executor.remote, sent as
`<interpreter> -c <file text> <one JSON argument>` (glm_tpu.executor.fleet);
their SHA-256s are recorded in the run's helpers.json. After the workers end
(stop, completion or failure) every host's records are collected once: a record
the resident receipt already holds is skipped when equal, and a different one is
kept as final/<name> and listed in controller_terminal.json (divergent_records);
nothing is overwritten, and a failure is reported only after collection.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import time

from glm_tpu import user_request as legacy
from glm_tpu.config.site import DEFAULT_HOST_RANK_REGEX, SiteConfig, rank_matches, set_current_site
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.executor import fleet as remote
from glm_tpu.executor import launch_policy
from glm_tpu.optimized import request
from glm_tpu.utils import io_utils
from scripts.release import ws32_optimized_worker as worker

REPO=Path(__file__).resolve().parents[2]
MODULE=protocol.WORKER_MODULE


def require(value,message):
    if not value:raise ValueError(message)


def source_identity(repo,policy):
    """The commit to stage: the site's [launch] policy applied to the checkout, which must be this
    controller's own (this launcher's and the glm_tpu package's), so that the proof covers them."""
    launch_policy.require_controller_checkout(repo,REPO,launch_policy.package_checkout())
    return launch_policy.source_identity(repo,policy).pin


def ssh_commands(fleet):
    project=['--project='+fleet.project] if fleet.project else []
    result=subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh',fleet.tpu_name,
        '--zone='+fleet.zone,*project,'--worker=all','--dry-run','--command=true'],capture_output=True,text=True,check=True)
    commands=[shlex.split(line) for line in result.stdout.splitlines() if line.startswith('/usr/bin/ssh ')]
    require(len(commands)==fleet.num_hosts,'SSH discovery must return eight hosts')
    for command in commands:
        require(command[-2:]==['--','true'],'SSH discovery command differs')
        command[:]=[s.replace('StrictHostKeyChecking=no','StrictHostKeyChecking=yes') for s in command if s!='-t']
        command[1:1]=['-o','BatchMode=yes','-o','ConnectionAttempts=1','-o','ConnectTimeout=30',
                      '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
        alias=next(s.split('=',1)[1] for s in command if s.startswith('HostKeyAlias='))
        require(subprocess.run(['ssh-keygen','-F',alias,'-f',str(fleet.known_hosts)],
            capture_output=True).returncode==0,'SSH host key is unknown')
    return commands


def remote_all(commands,command,root,label,*,payload=None,check=True):
    def one(rank):
        with (root/f'{label}.rank{rank}.log').open('xb') as stream:
            return subprocess.run(commands[rank][:-1]+[command],input=payload,
                stdout=stream,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=8) as pool:codes=list(pool.map(one,range(8)))
    if check:require(not any(codes),label+' failed on one or more hosts; see private originals')
    return codes


def idle(commands,root,label,fleet,hosts=None):
    """``IDLE <host>`` from every host (no libtpu holder, no live worker of this run); before
    staging ``hosts`` is None and the observed hostnames are authenticated here by rank."""
    remote_all(commands,remote.command(fleet,'idle_probe',dict(root=str(root),hosts=hosts)),root,label)
    observed=[]
    for rank in range(8):
        found=[s[5:] for s in (root/f'{label}.rank{rank}.log').read_text().splitlines() if s.startswith('IDLE ')]
        require(len(found)==1 and fleet.rank_matches(found[0],rank),'idle observation host differs')
        observed+=found
    require(len(set(observed))==8,'idle observations contain duplicate hosts')
    require(hosts is None or observed==list(hosts),'idle observation host differs')
    return observed


def stage_bundle(repo,pin,root,raw,site):
    archive=subprocess.check_output(['git','archive','--format=tar',pin],cwd=repo)
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
        for member in tar:
            require(member.isfile() or member.isdir(),'release archive contains a non-regular entry')
            if member.isfile():files['source/'+member.name]=tar.extractfile(member).read()
    manifest={name.removeprefix('source/'):sha256(data).hexdigest() for name,data in files.items()}
    manifest_raw=legacy.canonical(manifest)+b'\n'
    binding_dir=site.topology.binding_dir
    files.update({'request.json':raw,'source_manifest.json':manifest_raw,'site.json':site.resolved_json(),
                  'topology_rebinding.json':(binding_dir/'topology_rebinding.json').read_bytes()})
    require(sha256(files['topology_rebinding.json']).hexdigest()==site.topology.binding_sha256,'site rebinding changed')
    binding=json.loads(files['topology_rebinding.json'])
    for rank in range(8):
        name=f'topology.rank{rank}.json';data=(binding_dir/'captures'/name).read_bytes()
        require(sha256(data).hexdigest()==binding['capture_sha256'][name],'topology capture differs')
        files['topology_capture/'+name]=data
    result=io.BytesIO()
    with tarfile.open(fileobj=result,mode='w:gz') as tar:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            tar.addfile(info,io.BytesIO(data))
    return result.getvalue(),sha256(manifest_raw).hexdigest()


def cleanup_owned(commands,root,pin,*,hosts,fleet,module=MODULE):
    """Terminate only this invocation's authenticated process, never an unknown holder."""
    remote_all(commands,remote.command(fleet,'cleanup',dict(root=str(root),hosts=hosts,pin=pin,module=module)),
               root,'cleanup_owned',check=False)


def fetch_records(commands,directory,root,label,names,*,hosts,fleet):
    """Per rank, the named records of ``directory`` on that host: {name: bytes}."""
    remote_all(commands,remote.command(fleet,'fetch',dict(dir=str(directory),hosts=hosts,names=names)),root,label)
    return [{name:base64.b64decode(data,validate=True) for name,data in
             json.loads((root/f'{label}.rank{rank}.log').read_text()).items()} for rank in range(8)]


def summarize(rows,pin,request_sha,*,idle_after=True,host_rank_regex=DEFAULT_HOST_RANK_REGEX):
    require(len(rows)==8 and [r['rank'] for r in rows]==list(range(8)),'incomplete fleet result')
    for rank,row in enumerate(rows):
        require(row.get('complete') is True and row['code_hash']==pin
            and row['request_sha256']==request_sha and rank_matches(row['hostname'],rank,host_rank_regex),
            'worker identity/completion differs')
    require(len({r['request']['token_sha256'] for r in rows})==1,'worker outputs differ')
    require(len({r['request']['emitted'] for r in rows})==1,'worker output lengths differ')
    for row in rows[1:]:
        for name,program in rows[0]['programs'].items():
            require(all(program[k]==row['programs'][name][k] for k in
                ('stablehlo_sha256','optimized_hlo_sha256')),'worker graph identities differ')
    return dict(passed=True,code_hash=pin,request=rows[0]['request'],
        all_ranks_agree=True,all_hosts_idle_after=idle_after,
        limits='Retained-site greedy execution; completed answers and capacity coverage require separate checks.')


def resident_controller(commands,running,root,pin,value,wall_seconds,print_answers,*,hosts,fleet,
                        host_rank_regex=DEFAULT_HOST_RANK_REGEX):
    """Hold workload leases while serving an owner-only, ordered private inbox."""
    inbox=root/protocol.INBOX_DIR;inbox.mkdir(mode=0o700)
    sequence=0;pending=True
    deadline=time.monotonic()+wall_seconds+60
    while True:
        require(all(p.poll() is None for p in running),'resident worker exited unexpectedly')
        ready=root/protocol.READY_FILE
        if pending and ready.exists() and json.loads(ready.read_text())['sequence']==sequence:
            job=protocol.result_dir(root,sequence)
            fetched=fetch_records(commands,job,root,f'resident-collect-{sequence:04d}',['runner.rank{rank}.json'],
                                  hosts=hosts,fleet=fleet)
            require(all(set(files)=={protocol.runner_file(rank)} for rank,files in enumerate(fetched)),
                    'resident worker record missing')
            rows=[json.loads(files[protocol.runner_file(rank)]) for rank,files in enumerate(fetched)]
            result=summarize(rows,pin,value['request_sha256'],idle_after=False,host_rank_regex=host_rank_regex)
            for rank,row in enumerate(rows):
                if rank:worker.persist(job/protocol.runner_file(rank),row)
            result.update(model_retained=True,resident_sequence=sequence,
                          cleanup='intentionally deferred until explicit stop or worker failure')
            worker.persist(job/protocol.MEASUREMENT_FILE,result)
            print(protocol.STDOUT_RESIDENT_RESULT+str(job/protocol.MEASUREMENT_FILE),flush=True)
            if print_answers:
                for index,item in enumerate(request.requests(value)):
                    answer_root=job/f'item{index:03d}' if value.get('schema')==request.BATCH_SCHEMA else job
                    print('\n'+item['request_id']+'\n'+(answer_root/'answer.txt').read_text(),flush=True)
            pending=False
        if pending:
            require(time.monotonic()<=deadline,'resident inference deadline expired')
        else:
            stop=inbox/protocol.STOP_FILE
            next_input=inbox/protocol.inbox_name(sequence+1)
            if stop.exists():
                worker.private(stop)
                require(json.loads(legacy.read_bounded(stop,1024))=={'stop':True},'invalid resident stop')
                for process in running:
                    process.stdin.write(protocol.STOP_COMMAND);process.stdin.flush()
                return
            if next_input.exists():
                worker.private(next_input)
                raw=legacy.read_bounded(next_input,legacy.PAYLOAD_CAP)
                next_value=json.loads(raw);request.validate_payload(next_value)
                require(next_value['context_capacity']==value['context_capacity'] and
                        next_value.get('schema')!=request.CONCURRENT_SCHEMA,
                        'resident input differs from loaded context/scheduling')
                sequence+=1;value=next_value
                command=protocol.encode_command(sequence,value)
                for process in running:
                    process.stdin.write(command);process.stdin.flush()
                pending=True;deadline=time.monotonic()+wall_seconds+60
        # This is controller process supervision, not model-turn polling.
        time.sleep(2)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=7200)
    parser.add_argument('--print-answers',action='store_true',help='print completed local outputs after cleanup')
    parser.add_argument('--keep-loaded',action='store_true')
    parser.add_argument('--site',type=Path,help='site file (default: $GLM_TPU_SITE_CONFIG, else $GLM_TPU_CONFIG_ROOT/site.toml)')
    parser.add_argument('--repo',type=Path,help='git checkout to stage; must be this controller\'s own (default: the site paths.repo, else this checkout)')
    args=parser.parse_args(argv)
    require(1<=args.wall_seconds<=86400,'wall deadline must be 1..86400 seconds')
    site=SiteConfig.load(args.site)
    set_current_site(site)
    repo=launch_policy.resolve_repo(site,args.repo,default=REPO)
    fleet=site.fleet
    worker.private(args.request)
    require(not any(args.request.resolve().is_relative_to(p) for p in (repo,REPO)),'private request must be outside Git')
    raw=legacy.read_bounded(args.request,legacy.PAYLOAD_CAP)
    value=json.loads(raw);request.validate_payload(value)
    require(not args.keep_loaded or value.get('schema')!=request.CONCURRENT_SCHEMA,
            'resident mode currently uses sequential ordinary requests')
    pin=source_identity(repo,site.launch)
    os.umask(0o077)
    root=site.paths.run_root/('optimized_request_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir(mode=0o700)
    print(protocol.STDOUT_RUN+str(root),flush=True)
    worker.persist(root/protocol.HELPERS_FILE,remote.helpers_record())
    with ExitStack() as stack:
        locks=[]
        for path,blocking in [(p,False) for p in site.locks.workload]+[(p,True) for p in site.locks.sync]:
            stream=stack.enter_context(open(path,'a'))
            # A live model owner is a refusal. A scheduled source backup only
            # delays staging: wait on its lock without retrying any workload.
            flags=fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(stream,flags);locks.append(stream)
        commands=ssh_commands(fleet);hosts=idle(commands,root,'idle_before',fleet)
        require(hosts[0]==socket.gethostname(),'controller must run on authenticated rank0')
        bundle,manifest_sha=stage_bundle(repo,pin,root,raw,site)
        site_sha=site.resolved_sha256()
        remote_all(commands,remote.command(fleet,'stage_bundle',dict(root=str(root),digest=sha256(bundle).hexdigest(),
            hosts=hosts)),root,'stage',payload=bundle)
        # CPU-only preflight on every host precedes the single fleet dispatch.
        command=[fleet.worker_python,'-m',MODULE,'--output',str(root),'--code-hash',pin,
            '--source-manifest-sha256',manifest_sha,'--request-file-sha256',sha256(raw).hexdigest(),
            '--site-sha256',site_sha,
            '--topology-rebinding-sha256',site.topology.binding_sha256,'--coordinator-address',fleet.coordinator_address,
            '--wall-seconds',str(args.wall_seconds)]
        if args.keep_loaded:command.append('--keep-loaded')
        preflight=remote.preflight_command(fleet,root,command)
        remote_all(commands,preflight,root,'preflight')
        environments=[json.loads((root/f'preflight.rank{rank}.log').read_text()) for rank in range(8)]
        require([r['hostname'] for r in environments]==hosts,'preflight hosts differ')
        require(len({json.dumps({k:v for k,v in r.items() if k!='hostname'},sort_keys=True)
            for r in environments})==1,'fleet environments differ')
        require(environments[0]['jax']==environments[0]['jaxlib']=='0.10.1','retained JAX version differs')
        worker.persist(root/'controller_identity.json',dict(code_hash=pin,source_manifest_sha256=manifest_sha,
            request_file_sha256=sha256(raw).hexdigest(),request_sha256=value['request_sha256'],site_sha256=site_sha,
            controller_pid=os.getpid(),controller_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            automatic_workload_retries=False,hosts=hosts,environment=environments[0]))
        for stream in locks[len(site.locks.workload):]:fcntl.flock(stream,fcntl.LOCK_UN)
        wrapper=remote.command(fleet,'start_worker',dict(root=str(root),hosts=hosts,pin=pin,
            worker_python=fleet.worker_python,pythonpath=list(fleet.worker_pythonpath),module=MODULE,
            env={'JAX_PLATFORMS':'tpu',protocol.WORKER_ENV_FLAG:'1'},argv=command[3:]))
        running=[]
        failed=False
        failure=None
        try:
            for rank in range(8):
                log=stack.enter_context((root/f'run.rank{rank}.log').open('xb'))
                running.append(subprocess.Popen(commands[rank][:-1]+[wrapper],
                    stdin=subprocess.PIPE if args.keep_loaded else subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT))
            if args.keep_loaded:
                try:
                    resident_controller(commands,running,root,pin,value,args.wall_seconds,args.print_answers,
                                        hosts=hosts,fleet=fleet,host_rank_regex=fleet.host_rank_regex)
                except Exception as exc:
                    # A resident failure is reported after cleanup and collection, so every
                    # host's record (e.g. a failure_type rewrite) is preserved first.
                    failure=exc;failed=True
            deadline=time.monotonic()+args.wall_seconds+60
            while failure is None and any(p.poll() is None for p in running):
                if any(p.poll() not in (None,0) for p in running) or time.monotonic()>deadline:
                    failed=True;break
                time.sleep(10)
        except BaseException:
            failed=True;raise
        finally:
            if failed:cleanup_owned(commands,root,pin,hosts=hosts,fleet=fleet)
            try:idle(commands,root,'idle_after',fleet,hosts)
            except Exception:
                print('Cleanup unresolved; workload leases retained. Inspect '+str(root),flush=True)
                # An operator must authenticate cleanup before ending this
                # controller. There is no auto-retry or lease-release timeout.
                while True:time.sleep(30)
        codes=[p.wait() for p in running]
        divergent=[];collected=False
        try:
            fetched=fetch_records(commands,root,root,'collect',['runner.rank{rank}.json','worker_started.rank{rank}.json'],
                                  hosts=hosts,fleet=fleet)
            for rank in range(1,8):
                for name,payload in fetched[rank].items():
                    io_utils.write_collected(root,name,payload,divergent)
            collected=True
        finally:
            worker.persist(root/protocol.CONTROLLER_TERMINAL_FILE,dict(codes=codes,all_hosts_idle=True,
                collected=collected,divergent_records=divergent))
        if failure is not None:raise failure
        require(not any(codes),'request failed; partial originals preserved, no retry')
        rows=[json.loads((root/f'runner.rank{rank}.json').read_text()) for rank in range(8)]
        summary=summarize(rows,pin,value['request_sha256'],host_rank_regex=fleet.host_rank_regex)
        worker.persist(root/protocol.SUMMARY_FILE,summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
        if args.print_answers and not args.keep_loaded:  # the resident loop printed each answer
            for index,item in enumerate(request.requests(value)):
                item_root=root/f'item{index:03d}' if value.get('schema') in (request.BATCH_SCHEMA,request.CONCURRENT_SCHEMA) else root
                print('\n'+item['request_id']+'\n'+(item_root/'answer.txt').read_text(),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
