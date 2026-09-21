"""One private greedy request on the retained 32-chip site; no automatic retries.

Source must be clean, committed and pushed. This controller holds the workload
leases through authenticated cleanup, and never creates or resizes resources.
"""
import argparse
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
from glm_tpu.optimized import request
from scripts.release import ws32_optimized_worker as worker

REPO=Path(__file__).resolve().parents[2]
PYTHON='/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12'
SITE='/home/gianl/vllm-env/lib/python3.12/site-packages'
BINDING=Path('/home/gianl/glm-run/perf_topology_diagnostic_20260920T125236Z/derived_binding')
BINDING_SHA='66f468aeeaa286079bb2c2c782640acbd61e153912bf1aea1007637bf6429bde'
LOCKS=('/home/gianl/.glm-tpu-workload.lock','/home/gianl/glm-run/.glm_pod_workload.lock',
       '/opt/glm-tpu/locks/glm_tpu_rsync.lock','/home/gianl/.glm-tpu-rsync.lock')
MODULE='scripts.release.ws32_optimized_worker'


def require(value,message):
    if not value:raise ValueError(message)


def source_identity(repo):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip(),'source must be clean')
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=repo,text=True).strip()
    require(branch=='main' or branch.startswith('release/'),'use a release branch or main')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip()
    require(remote=='git@github.com:GianluigiVitale/glm-tpu.git','private release origin differs')
    published=subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/'+branch],cwd=repo,text=True)
    require(published.split()[0]==pin,'source must be pushed before deployment')
    from scripts.greenfield.ws32_native_benchmark_programs import require_source
    require_source(repo)
    return pin


def ssh_commands():
    result=subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh','db-v4-64-od',
        '--zone=us-central2-b','--worker=all','--dry-run','--command=true'],capture_output=True,text=True,check=True)
    commands=[shlex.split(line) for line in result.stdout.splitlines() if line.startswith('/usr/bin/ssh ')]
    require(len(commands)==8,'SSH discovery must return eight hosts')
    for command in commands:
        require(command[-2:]==['--','true'],'SSH discovery command differs')
        command[:]=[s.replace('StrictHostKeyChecking=no','StrictHostKeyChecking=yes') for s in command if s!='-t']
        command[1:1]=['-o','BatchMode=yes','-o','ConnectionAttempts=1','-o','ConnectTimeout=30',
                      '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
        alias=next(s.split('=',1)[1] for s in command if s.startswith('HostKeyAlias='))
        require(subprocess.run(['ssh-keygen','-F',alias,'-f','/home/gianl/.ssh/google_compute_known_hosts'],
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


def idle(commands,root,label):
    from scripts.greenfield.watch_ws32_run import REMOTE
    probe=REMOTE[REMOTE.index('import hashlib'):REMOTE.index('tag, pin =')]+'''
if libtpu_holders():raise RuntimeError('libtpu is owned')
root=pathlib.Path(ROOT)
rank=int(socket.gethostname().rsplit('-w-',1)[1])
marker=root/f'worker_started.rank{rank}.json'
if marker.exists():
    owner=json.loads(marker.read_text())
    path=pathlib.Path('/proc')/str(owner['pid'])/'stat'
    if path.exists():
        fields=path.read_text().rsplit(')',1)[1].split()
        if fields[19]==owner['start_ticks'] and fields[0] not in ('Z','X'):
            raise RuntimeError('request worker is live')
print('IDLE '+socket.gethostname())
'''.replace('ROOT',repr(str(root)))
    remote_all(commands,'python3 -c '+shlex.quote(probe),root,label)
    hosts=[]
    for rank in range(8):
        found=[s[5:] for s in (root/f'{label}.rank{rank}.log').read_text().splitlines() if s.startswith('IDLE ')]
        require(len(found)==1 and found[0].endswith('-w-'+str(rank)),'idle observation host differs')
        hosts+=found
    require(len(set(hosts))==8,'idle observations contain duplicate hosts')
    return hosts


def stage_bundle(repo,pin,root,raw):
    archive=subprocess.check_output(['git','archive','--format=tar',pin],cwd=repo)
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
        for member in tar:
            require(member.isfile() or member.isdir(),'release archive contains a non-regular entry')
            if member.isfile():files['source/'+member.name]=tar.extractfile(member).read()
    manifest={name.removeprefix('source/'):sha256(data).hexdigest() for name,data in files.items()}
    manifest_raw=legacy.canonical(manifest)+b'\n'
    files.update({'request.json':raw,'source_manifest.json':manifest_raw,
                  'topology_rebinding.json':(BINDING/'topology_rebinding.json').read_bytes()})
    require(sha256(files['topology_rebinding.json']).hexdigest()==BINDING_SHA,'site rebinding changed')
    binding=json.loads(files['topology_rebinding.json'])
    for rank in range(8):
        name=f'topology.rank{rank}.json';data=(BINDING/'captures'/name).read_bytes()
        require(sha256(data).hexdigest()==binding['capture_sha256'][name],'topology capture differs')
        files['topology_capture/'+name]=data
    result=io.BytesIO()
    with tarfile.open(fileobj=result,mode='w:gz') as tar:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            tar.addfile(info,io.BytesIO(data))
    return result.getvalue(),sha256(manifest_raw).hexdigest()


def cleanup_owned(commands,root,pin,*,module=MODULE):
    """Terminate only this invocation's authenticated process, never an unknown holder."""
    code='''import json,os,pathlib,signal,socket,time
root=pathlib.Path(ROOT);rank=int(socket.gethostname().rsplit('-w-',1)[1])
marker=root/f'worker_started.rank{rank}.json'
if not marker.exists():raise SystemExit(0)
owner=json.loads(marker.read_text());pid=owner['pid'];proc=pathlib.Path('/proc')/str(pid)
if not proc.exists():raise SystemExit(0)
fd=os.pidfd_open(pid)
try:
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if fields[19]!=owner['start_ticks'] or fields[0] in ('Z','X'):raise SystemExit(0)
    if owner['hostname']!=socket.gethostname() or owner['code_hash']!=PIN or owner['boot_id']!=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('cleanup identity differs')
    argv=(proc/'cmdline').read_bytes().split(b'\\0')
    if MODULE.encode() not in argv or str(root).encode() not in argv or PIN.encode() not in argv:
        raise RuntimeError('cleanup argv differs')
    signal.pidfd_send_signal(fd,signal.SIGKILL)
finally:os.close(fd)
'''.replace('ROOT',repr(str(root))).replace('PIN',repr(pin)).replace('MODULE',repr(module))
    remote_all(commands,'python3 -c '+shlex.quote(code),root,'cleanup_owned',check=False)


def summarize(rows,pin,request_sha):
    require(len(rows)==8 and [r['rank'] for r in rows]==list(range(8)),'incomplete fleet result')
    for rank,row in enumerate(rows):
        require(row.get('complete') is True and row['code_hash']==pin
            and row['request_sha256']==request_sha and row['hostname'].endswith('-w-'+str(rank)),
            'worker identity/completion differs')
    require(len({r['request']['token_sha256'] for r in rows})==1,'worker outputs differ')
    require(len({r['request']['emitted'] for r in rows})==1,'worker output lengths differ')
    for row in rows[1:]:
        for name,program in rows[0]['programs'].items():
            require(all(program[k]==row['programs'][name][k] for k in
                ('stablehlo_sha256','optimized_hlo_sha256')),'worker graph identities differ')
    return dict(passed=True,code_hash=pin,request=rows[0]['request'],
        all_ranks_agree=True,all_hosts_idle_after=True,
        limits='Retained-site greedy execution; completed answers and capacity coverage require separate checks.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=7200)
    parser.add_argument('--print-answers',action='store_true',help='print completed local outputs after cleanup')
    args=parser.parse_args(argv)
    require(1<=args.wall_seconds<=86400,'wall deadline must be 1..86400 seconds')
    worker.private(args.request)
    require(not args.request.resolve().is_relative_to(REPO),'private request must be outside Git')
    raw=legacy.read_bounded(args.request,legacy.PAYLOAD_CAP)
    value=json.loads(raw);request.validate_payload(value)
    pin=source_identity(REPO)
    os.umask(0o077)
    root=worker.RUN_ROOT/('optimized_request_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir(mode=0o700)
    print('RUN '+str(root),flush=True)
    with ExitStack() as stack:
        locks=[]
        for index,path in enumerate(LOCKS):
            stream=stack.enter_context(open(path,'a'))
            # A live model owner is a refusal. A scheduled source backup only
            # delays staging: wait on its lock without retrying any workload.
            flags=fcntl.LOCK_EX|(fcntl.LOCK_NB if index<2 else 0)
            fcntl.flock(stream,flags);locks.append(stream)
        commands=ssh_commands();hosts=idle(commands,root,'idle_before')
        require(hosts[0]==socket.gethostname(),'controller must run on authenticated rank0')
        bundle,manifest_sha=stage_bundle(REPO,pin,root,raw)
        code='''import hashlib,io,os,pathlib,socket,sys,tarfile
os.umask(0o077);root=pathlib.Path(ROOT)
rank=int(socket.gethostname().rsplit('-w-',1)[1])
if rank:root.mkdir(mode=0o700)
data=sys.stdin.buffer.read()
if hashlib.sha256(data).hexdigest()!=DIGEST:raise RuntimeError('staging transport differs')
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:tar.extractall(root,filter='data')
'''.replace('ROOT',repr(str(root))).replace('DIGEST',repr(sha256(bundle).hexdigest()))
        remote_all(commands,shlex.join([PYTHON,'-c',code]),root,'stage',payload=bundle)
        # CPU-only preflight on every host precedes the single fleet dispatch.
        command=[PYTHON,'-m',MODULE,'--output',str(root),'--code-hash',pin,
            '--source-manifest-sha256',manifest_sha,'--request-file-sha256',sha256(raw).hexdigest(),
            '--topology-rebinding-sha256',BINDING_SHA,'--coordinator-address','192.168.0.37:8476',
            '--wall-seconds',str(args.wall_seconds)]
        preflight='cd '+shlex.quote(str(root/'source'))+' && '+shlex.join([
            'env','JAX_PLATFORMS=cpu','GLM_OPTIMIZED_REQUEST=1',
            'PYTHONPATH='+str(root/'source')+':'+SITE,*command,'--preflight-only'])
        remote_all(commands,preflight,root,'preflight')
        environments=[json.loads((root/f'preflight.rank{rank}.log').read_text()) for rank in range(8)]
        require([r['hostname'] for r in environments]==hosts,'preflight hosts differ')
        require(len({json.dumps({k:v for k,v in r.items() if k!='hostname'},sort_keys=True)
            for r in environments})==1,'fleet environments differ')
        require(environments[0]['jax']==environments[0]['jaxlib']=='0.10.1','retained JAX version differs')
        worker.persist(root/'controller_identity.json',dict(code_hash=pin,source_manifest_sha256=manifest_sha,
            request_file_sha256=sha256(raw).hexdigest(),request_sha256=value['request_sha256'],
            controller_pid=os.getpid(),controller_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            automatic_workload_retries=False,hosts=hosts,environment=environments[0]))
        for stream in locks[2:]:fcntl.flock(stream,fcntl.LOCK_UN)
        wrapper='''import json,os,pathlib,socket
root=pathlib.Path(ROOT);rank=int(socket.gethostname().rsplit('-w-',1)[1])
owner=dict(pid=os.getpid(),hostname=socket.gethostname(),code_hash=PIN,
    boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    start_ticks=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
with (root/f'worker_started.rank{rank}.json').open('x') as stream:json.dump(owner,stream)
os.chdir(root/'source');os.environ.update(JAX_PLATFORMS='tpu',GLM_OPTIMIZED_REQUEST='1',PYTHONPATH=str(root/'source')+':'+SITE)
os.execv(PYTHON,COMMAND)
'''.replace('ROOT',repr(str(root))).replace('PIN',repr(pin)).replace('SITE',repr(SITE)).replace('PYTHON,COMMAND',repr(PYTHON)+','+repr(command))
        running=[]
        failed=False
        try:
            for rank in range(8):
                log=stack.enter_context((root/f'run.rank{rank}.log').open('xb'))
                running.append(subprocess.Popen(commands[rank][:-1]+['python3 -c '+shlex.quote(wrapper)],stdout=log,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+args.wall_seconds+60
            while any(p.poll() is None for p in running):
                if any(p.poll() not in (None,0) for p in running) or time.monotonic()>deadline:
                    failed=True;break
                time.sleep(10)
        except BaseException:
            failed=True;raise
        finally:
            if failed:cleanup_owned(commands,root,pin)
            try:idle(commands,root,'idle_after')
            except Exception:
                print('Cleanup unresolved; workload leases retained. Inspect '+str(root),flush=True)
                # An operator must authenticate cleanup before ending this
                # controller. There is no auto-retry or lease-release timeout.
                while True:time.sleep(30)
        codes=[p.wait() for p in running]
        worker.persist(root/'controller_terminal.json',dict(codes=codes,all_hosts_idle=True))
        fetch='''import base64,json,pathlib,socket
root=pathlib.Path(ROOT);rank=int(socket.gethostname().rsplit('-w-',1)[1])
print(json.dumps({name:base64.b64encode((root/name).read_bytes()).decode() for name in
    (f'runner.rank{rank}.json',f'worker_started.rank{rank}.json') if (root/name).exists()}))
'''.replace('ROOT',repr(str(root)))
        remote_all(commands,'python3 -c '+shlex.quote(fetch),root,'collect')
        import base64
        for rank in range(1,8):
            for name,payload in json.loads((root/f'collect.rank{rank}.log').read_text()).items():
                with (root/name).open('xb') as stream:stream.write(base64.b64decode(payload,validate=True))
        require(not any(codes),'request failed; partial originals preserved, no retry')
        rows=[json.loads((root/f'runner.rank{rank}.json').read_text()) for rank in range(8)]
        summary=summarize(rows,pin,value['request_sha256'])
        worker.persist(root/'summary.json',summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
        if args.print_answers:
            for index,item in enumerate(request.requests(value)):
                item_root=root/f'item{index:03d}' if value.get('schema') in (request.BATCH_SCHEMA,request.CONCURRENT_SCHEMA) else root
                print('\n'+item['request_id']+'\n'+(item_root/'answer.txt').read_text(),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
