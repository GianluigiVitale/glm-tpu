"""Offline gates for the optimized controller; never contact TPU hosts."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts.release import launch_ws32_optimized_request as launch
from scripts.release import ws32_optimized_worker as worker


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


def test_private_input_rejects_public_permissions_and_symlink(tmp_path):
    path=tmp_path/'request.json';path.write_text('{}');path.chmod(0o644)
    with pytest.raises(ValueError):worker.private(path)
    path.chmod(0o600);worker.private(path)
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError):worker.private(link)


def test_dirty_source_refused_before_network(monkeypatch,tmp_path):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return b' M private.py\n'
    monkeypatch.setattr(launch.subprocess,'check_output',run)
    with pytest.raises(ValueError,match='clean'):launch.source_identity(tmp_path)
    assert calls==[['git','status','--porcelain']]


def test_worker_default_off_before_model_import(monkeypatch,tmp_path):
    monkeypatch.delenv('GLM_OPTIMIZED_REQUEST',raising=False)
    with pytest.raises(ValueError,match='protected'):
        worker.preflight(SimpleNamespace(output=tmp_path,code_hash='a'*40,wall_seconds=100))


def test_ssh_unknown_host_never_dispatches(monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if argv[0]=='gcloud':
            return SimpleNamespace(stdout='\n'.join(
                f'/usr/bin/ssh -o HostKeyAlias=host{i} -o StrictHostKeyChecking=no example -- true' for i in range(8)))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(launch.subprocess,'run',run)
    with pytest.raises(ValueError,match='unknown'):launch.ssh_commands()
    assert len(calls)==2


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
    monkeypatch.setattr(launch,'LOCKS',paths)
    repo=tmp_path/'source';repo.mkdir();monkeypatch.setattr(launch,'REPO',repo)
    runs=tmp_path/'runs';runs.mkdir();monkeypatch.setattr(worker,'RUN_ROOT',runs)
    path=tmp_path/'input.json'
    path.write_bytes(canonical(request.from_token_ids([7],request_id='fixture',max_new_tokens=2)))
    path.chmod(0o600)
    ready=threading.Event();dispatched=threading.Event()
    def identity(repo):ready.set();return 'a'*40
    monkeypatch.setattr(launch,'source_identity',identity)
    class EndBeforeSSH(Exception):pass
    def ssh():dispatched.set();raise EndBeforeSSH
    monkeypatch.setattr(launch,'ssh_commands',ssh)
    previous_umask=os.umask(0o077)
    try:
        with open(paths[held_index],'a') as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(launch.main,['--request',str(path)])
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
