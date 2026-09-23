"""Offline gates for the optimized controller; never contact TPU hosts."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts.release import launch_ws32_optimized_request as launch
from scripts.release import ws32_optimized_worker as worker
from tests.fixtures.site import example_mapping, example_site, write_example_site


def rows():
    return [dict(rank=i,hostname=f'fixture-w-{i}',complete=True,code_hash='a'*40,
        request_sha256='b'*64,request=dict(token_sha256='c'*64,emitted=3),
        programs=dict(decode=dict(stablehlo_sha256='d'*64,optimized_hlo_sha256='e'*64)))
        for i in range(8)]


def test_complete_fleet_required():
    assert launch.summarize(rows(),'a'*40,'b'*64)['passed']
    for bad in (rows()[:7],list(reversed(rows()))):
        with pytest.raises(ValueError):launch.summarize(bad,'a'*40,'b'*64)


@pytest.mark.parametrize('field,value', [('complete',False),('code_hash','f'*40),
    ('request_sha256','f'*64),('hostname','fixture-w-0')])
def test_identity_and_failure_cannot_publish(field,value):
    fleet=rows();fleet[7][field]=value
    with pytest.raises(ValueError):launch.summarize(fleet,'a'*40,'b'*64)


def test_graph_or_output_disagreement_cannot_publish():
    for section,key in [('request','token_sha256'),('request','emitted'),
                        ('programs','decode')]:
        fleet=rows()
        if section=='programs':fleet[7][section][key]['optimized_hlo_sha256']='f'*64
        else:fleet[7][section][key]='different'
        with pytest.raises(ValueError):launch.summarize(fleet,'a'*40,'b'*64)


def test_resident_result_never_claims_cleanup():
    result=launch.summarize(rows(),'a'*40,'b'*64,idle_after=False)
    assert result['passed'] and not result['all_hosts_idle_after']


def test_resident_reuses_runtime_and_has_explicit_stop(monkeypatch,tmp_path):
    import io
    from glm_tpu.optimized import request
    value=request.from_token_ids([7],request_id='fixture',max_new_tokens=2,context_capacity=32768)
    runtime=SimpleNamespace(capacity=32768,phase=lambda name,action:action())
    seen=[]
    def generate(actual,pending,value,root,rank,deadline,**kwargs):
        seen.append((actual,kwargs['warmup'],root.name))
        return [dict(emitted=2,token_sha256='c'*64)]
    monkeypatch.setattr(worker,'run_queued',generate)
    commands=json.dumps(dict(sequence=1,request=value))+'\n'+json.dumps(dict(stop=True))+'\n'
    worker.resident_loop(runtime,rows()[0],tmp_path,0,3600,stream=io.StringIO(commands))
    assert seen==[(runtime,False,'resident-0001')]
    assert json.loads((tmp_path/'resident-ready.json').read_text())=={'sequence':1}
    assert json.loads((tmp_path/'resident-0001/runner.rank0.json').read_text())['request_sha256']==value['request_sha256']


@pytest.mark.parametrize('case',['disconnect','sequence','capacity'])
def test_resident_refuses_ambiguous_or_incompatible_input(tmp_path,case):
    import io
    from glm_tpu.optimized import request
    value=request.from_token_ids([7],request_id='fixture',max_new_tokens=2,
                                context_capacity=8192 if case=='capacity' else 32768)
    line='' if case=='disconnect' else json.dumps(dict(sequence=2 if case=='sequence' else 1,request=value))+'\n'
    runtime=SimpleNamespace(capacity=32768,phase=lambda name,action:action())
    with pytest.raises(ValueError):
        worker.resident_loop(runtime,rows()[0],tmp_path,0,3600,stream=io.StringIO(line))
    assert not (tmp_path/'resident-0001').exists()


def test_resident_controller_keeps_idle_model_past_inference_deadline(monkeypatch,tmp_path):
    import io
    from glm_tpu.optimized import request
    value=request.from_token_ids([7],request_id='fixture',max_new_tokens=2)
    fleet=rows()
    for row in fleet:row['request_sha256']=value['request_sha256']
    worker.persist(tmp_path/'resident-ready.json',dict(sequence=0))
    def collect(commands,command,root,label):
        for rank,row in enumerate(fleet):worker.persist(root/f'{label}.rank{rank}.log',row)
    monkeypatch.setattr(launch,'remote_all',collect)
    elapsed=[0]
    monkeypatch.setattr(launch.time,'monotonic',lambda:elapsed[0])
    def wait(seconds):
        elapsed[0]+=1000
        if elapsed[0]>=2000:
            stop=tmp_path/'inbox/stop.json';stop.write_text('{"stop":true}');stop.chmod(0o600)
    monkeypatch.setattr(launch.time,'sleep',wait)
    processes=[SimpleNamespace(poll=lambda:None,stdin=io.BytesIO()) for _ in range(8)]
    launch.resident_controller([],processes,tmp_path,'a'*40,value,1,False)
    assert elapsed[0]>=2000
    assert all(p.stdin.getvalue()==b'{"stop":true}\n' for p in processes)
    result=json.loads((tmp_path/'resident-measurement.json').read_text())
    assert result['model_retained'] and not result['all_hosts_idle_after']


def test_private_input_rejects_public_permissions_and_symlink(tmp_path):
    path=tmp_path/'request.json';path.write_text('{}');path.chmod(0o644)
    with pytest.raises(ValueError):worker.private(path)
    path.chmod(0o600);worker.private(path)
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError):worker.private(link)


def test_worker_default_off_before_model_import(monkeypatch,tmp_path):
    monkeypatch.delenv('GLM_OPTIMIZED_REQUEST',raising=False)
    with pytest.raises(ValueError,match='protected'):
        worker.preflight(SimpleNamespace(output=tmp_path,code_hash='a'*40,wall_seconds=100))


def test_migration_refuses_inherited_glm52_checkpoint():
    from scripts.release.ws32_user_worker import site_args as legacy_site_args
    args=legacy_site_args(SimpleNamespace())
    with pytest.raises(ValueError,match='GLM-5.3 runtime checkpoint'):
        worker.model.require_site(args)


def test_ssh_unknown_host_never_dispatches(monkeypatch,tmp_path):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if argv[0]=='gcloud':
            return SimpleNamespace(stdout='\n'.join(
                f'/usr/bin/ssh -o HostKeyAlias=host{i} -o StrictHostKeyChecking=no example -- true' for i in range(8)))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(launch.subprocess,'run',run)
    fleet=example_site(tmp_path).fleet
    with pytest.raises(ValueError,match='unknown'):launch.ssh_commands(fleet)
    assert len(calls)==2
    assert calls[0][5:7]==[fleet.tpu_name,'--zone='+fleet.zone]  # the site's TPU VM and zone
    assert calls[1][-2:]==['-f',str(fleet.known_hosts)]


def test_ssh_failure_is_not_retried(monkeypatch,tmp_path):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(launch.subprocess,'run',run)
    commands=[['ssh',str(i),'--','true'] for i in range(8)]
    with pytest.raises(ValueError):launch.remote_all(commands,'command',tmp_path,'run')
    assert len(calls)==8 and {argv[1] for argv in calls}=={str(i) for i in range(8)}


@pytest.mark.parametrize('held_index',[0,2])
def test_model_owner_refuses_but_backup_waits_before_any_ssh(monkeypatch,tmp_path,held_index):
    import fcntl,os,threading
    from concurrent.futures import ThreadPoolExecutor
    from glm_tpu.optimized import request
    from glm_tpu.user_request import canonical
    paths=tuple(str(tmp_path/f'lock{i}') for i in range(4))
    import subprocess
    repo=tmp_path/'source';repo.mkdir();monkeypatch.setattr(launch,'REPO',repo)
    subprocess.run(['git','init','-q',str(repo)],check=True)  # the launch checkout must be a git top level
    runs=tmp_path/'runs';runs.mkdir()
    site=write_example_site(tmp_path/'site.toml',example_mapping(tmp_path,paths=dict(run_root=str(runs)),
        locks=dict(workload=list(paths[:2]),sync=list(paths[2:]))))
    path=tmp_path/'input.json'
    path.write_bytes(canonical(request.from_token_ids([7],request_id='fixture',max_new_tokens=2)))
    path.chmod(0o600)
    ready=threading.Event();dispatched=threading.Event()
    def identity(repo,policy):ready.set();return 'a'*40
    monkeypatch.setattr(launch,'source_identity',identity)
    class EndBeforeSSH(Exception):pass
    def ssh(fleet):dispatched.set();raise EndBeforeSSH
    monkeypatch.setattr(launch,'ssh_commands',ssh)
    previous_umask=os.umask(0o077)
    try:
        with open(paths[held_index],'a') as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(launch.main,['--request',str(path),'--site',str(site)])
                assert ready.wait(5)
                if held_index==0:
                    with pytest.raises(BlockingIOError):pending.result(timeout=5)
                    assert not dispatched.is_set()
                else:
                    assert not dispatched.wait(.1)
                    fcntl.flock(owner,fcntl.LOCK_UN)
                    with pytest.raises(EndBeforeSSH):pending.result(timeout=5)
                    assert dispatched.is_set()
    finally:os.umask(previous_umask)
