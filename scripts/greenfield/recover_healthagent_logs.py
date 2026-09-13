"""One-off, owner-authorized lossless recovery of the Sep13 OOM log backlog.

Fixed hosts/files only. Archive on worker0's persistent disk before removing a
closed original. No log truncation, container restart or cloud-resource change.
Run with CPU platform under the existing controller's two leases.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from scripts.greenfield.evict_reviewed_local_copies import _leases
from scripts.greenfield.launch_ws32_native_benchmark import ssh, persist

ROOT = Path('/home/gianl/glm-run/health-log-recovery-20260913')
SUFFIX = '.glm-recovery-20260913'
PIDS = {0: 522686, 1: 152578, 4: 136809}


def remote(rank: int, body: str) -> dict:
    command = shlex.join(['sudo', '/usr/bin/python3', '-c', body])
    out = ssh(command, workers=str(rank), timeout=300)
    lines = [s.removeprefix('LOG_RECOVERY ') for s in out.splitlines()
             if s.startswith('LOG_RECOVERY ')]
    if len(lines) != 1:
        raise ValueError('missing unique remote receipt')
    return json.loads(lines[0])


def rotate(rank: int) -> dict:
    # Rename rather than copytruncate; rsyslog HUP only reopens its log files.
    return remote(rank, f'''
import os,pathlib,stat,signal,subprocess,json,time,fcntl
pid={PIDS[rank]}
assert pathlib.Path('/proc',str(pid),'exe').resolve()==pathlib.Path('/usr/sbin/rsyslogd')
lock=open('/var/lib/logrotate/status','r+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[]
for name in ('kern.log','syslog'):
 p=pathlib.Path('/var/log')/name;q=pathlib.Path(str(p)+{SUFFIX!r})
 s=p.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not q.exists()
 rows.append(dict(path=str(q),original=str(p),ino=s.st_ino,dev=s.st_dev,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode)))
for r in rows:
 os.rename(r['original'],r['path'])
 fd=os.open(r['original'],os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,r['mode'])
 os.fchown(fd,r['uid'],r['gid']);os.fchmod(fd,r['mode']);os.fsync(fd);os.close(fd)
os.kill(pid,signal.SIGHUP)
subprocess.run(['logger','-t','glm-health-log-recovery','Reopened after lossless OOM log rotation'],check=True)
for r in rows:
 for attempt in range(20):
  held=subprocess.run(['fuser',r['path']],capture_output=True)
  if held.returncode==1 and not held.stdout.strip() and not held.stderr.strip():break
  time.sleep(.25)
 else:raise RuntimeError('rotated log remains held; preserve it')
 s=os.stat(r['path']);assert s.st_ino==r['ino'] and s.st_dev==r['dev'] and s.st_nlink==1
 r.update(size=s.st_size,mtime_ns=s.st_mtime_ns)
assert pathlib.Path('/proc',str(pid),'exe').resolve()==pathlib.Path('/usr/sbin/rsyslogd')
fd=os.open('/var/log',os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print('LOG_RECOVERY '+json.dumps(dict(rank={rank},logging_pid=pid,files=rows)),flush=True)
''')


def archive(rank: int, row: dict) -> dict:
    path = ROOT / f'rank{rank}-{Path(row["original"]).name}.gz'
    # gcloud writes progress on stderr; gzip/SHA verification rejects any stdout
    # contamination and preserves the original. Never replace existing archives.
    command = ['gcloud','compute','tpus','tpu-vm','ssh','db-v4-64-od',
               '--zone','us-central2-b',f'--worker={rank}',
               '--command='+shlex.join(['sudo','gzip','-1','-c',row['path']])]
    with path.open('xb') as output, (path.with_suffix('.stderr')).open('xb') as err:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err)
        total = 0
        try:
            assert child.stdout is not None
            while block := child.stdout.read(1 << 20):
                total += len(block)
                if total > 3 << 30 or shutil.disk_usage(ROOT).free < 1 << 30:
                    raise ValueError('compressed archive/floor cap exceeded; preserve source')
                output.write(block)
            if child.wait() != 0:
                raise ValueError('compression transport failed; preserve source')
            output.flush(); os.fsync(output.fileno())
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=10)
    digest=hashlib.sha256(); size=0
    with gzip.open(path,'rb') as stream:
        while block := stream.read(1 << 20):
            size += len(block)
            if size > row['size']:
                raise ValueError('inflated size overflow; preserve source')
            digest.update(block)
    if size != row['size']:
        raise ValueError('inflated size differs; preserve source')
    fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY)
    os.fsync(fd);os.close(fd)
    # SHA verification is on the closed remote original immediately before its
    # descriptor-relative unlink. The durable compressed copy already exists.
    result=remote(rank,f'''
import os,hashlib,json,subprocess
r={row!r};p=r['path']
held=subprocess.run(['fuser',p],capture_output=True)
assert held.returncode==1 and not held.stdout.strip() and not held.stderr.strip()
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd)
assert (s.st_dev,s.st_ino,s.st_nlink,s.st_size,s.st_mtime_ns)==(r['dev'],r['ino'],1,r['size'],r['mtime_ns'])
h=hashlib.sha256()
with os.fdopen(fd,'rb') as stream:
 while b:=stream.read(8<<20):h.update(b)
assert h.hexdigest()=={digest.hexdigest()!r}
held=subprocess.run(['fuser',p],capture_output=True)
assert held.returncode==1 and not held.stdout.strip() and not held.stderr.strip()
d=os.open('/var/log',os.O_RDONLY|os.O_DIRECTORY)
s=os.stat(os.path.basename(p),dir_fd=d,follow_symlinks=False)
assert (s.st_dev,s.st_ino,s.st_nlink,s.st_size,s.st_mtime_ns)==(r['dev'],r['ino'],1,r['size'],r['mtime_ns'])
os.unlink(os.path.basename(p),dir_fd=d);os.fsync(d);os.close(d)
print('LOG_RECOVERY '+json.dumps(dict(rank={rank},removed=p,size=r['size'],sha256=h.hexdigest(),recovery={str(path)!r})),flush=True)
''')
    result['compressed_bytes']=total
    persist(path.with_suffix('.receipt.json'),result)
    return result


def main() -> None:
    os.umask(0o077)
    with _leases():
        ROOT.mkdir(mode=0o700,exist_ok=False)
        # Worker0 first: reclaim its log bytes before staging peer archives.
        for rank in (0,1,4):
            record=rotate(rank)
            persist(ROOT/f'rotation.rank{rank}.json',record)
            print('ROTATED',rank,flush=True)
            for row in record['files']:
                print('ARCHIVED',json.dumps(archive(rank,row)),flush=True)


if __name__=='__main__':
    main()
