"""Serialized native benchmark launch/attach on the EXISTING eight-host pod.

No infrastructure lifecycle calls. Reuses original root idle census, exact
process observer and both leases. A timeout never triggers a replacement run.
An interrupted controller can attach to the SAME tag/pin; ended workers retain
their original partials and independently attempt bounded publication.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time
from typing import Any

from scripts.greenfield import watch_ws32_run as watch
from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield import ws32_native_benchmark_collect as requests
from scripts.greenfield.ws32_native_benchmark_protocol import REPO, canonical
from scripts.greenfield.fp8_baseline_guard import census_command, validate_fleet
from scripts.greenfield.ws32_history_preflight import _plain_path

from scripts.release.ws32_host_ops import (
    PYTHON, BRANCH, ssh, persist, reviewed_branch, sync_command, source_preflight,
    validate_attach, markers, publication_state, observe_originals,
)
CAPSULE = "results/native-benchmark-registration-20260912/requests.json"
CAPSULE_GENERATION = 1789251974418204
CAPSULE_BYTES = 580513
CAPSULE_SHA = "daaf3f402c7ef684411bb8ab530f4c2adaebdc8b976d375356882d8ce00478ea"


def worker_command(tag: str, pin: str, coordinator: str) -> str:
    cold._identity(tag, pin, 0)
    host, port = coordinator.rsplit(":",1)
    ipaddress.ip_address(host)
    if port != "8476": raise ValueError("native coordinator port differs")
    args = [PYTHON, "-m", "scripts.greenfield.launch_ws32_native_benchmark",
            "--worker-child", "--tag", tag, "--code-hash", pin, "--coordinator", coordinator]
    return ("set -euo pipefail; cd "+shlex.quote(str(REPO))+"; "
        "nohup env JAX_PLATFORMS=cpu PYTHONPATH="+shlex.quote(str(REPO))+" "+shlex.join(args)+
        " > /home/gianl/glm-run/"+tag+"/supervisor.rank${HOSTNAME##*-w-}.log 2>&1 </dev/null &")


def prepare_worker(tag: str, pin: str) -> None:
    """Small pinned capsule only; verify existing tmpfs/checkpoint mount recipe."""
    from google.cloud import storage
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    _require_clean_code(pin)
    rank = int(socket.gethostname().rsplit("-w-",1)[1])
    cold._identity(tag, pin, rank)
    root = watch.RUN_ROOT / tag
    _plain_path(root)
    if rank == 0:
        if not root.is_dir(): raise ValueError("controller run root missing")
    else:
        root.mkdir(exist_ok=False)
    if shutil.disk_usage(root).free < 6 << 30:
        raise ValueError("native host launch floor below6GiB")
    recipe = json.loads((REPO/"configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    checkpoint = Path(recipe["GLM_GREENFIELD_WS32_CHECKPOINT_ROOT"])
    mount = subprocess.check_output(["findmnt","-T",str(checkpoint),"-n","-o","FSTYPE"],text=True).strip()
    if mount != "tmpfs" or len(list(checkpoint.glob("device_slot_*.safetensors"))) != 4:
        raise ValueError("existing final-layout tmpfs checkpoint missing; no recreation")
    overlay = recipe["GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_ROOT"]
    mount = subprocess.check_output(["findmnt","-T",overlay,"-n","-o","SOURCE,FSTYPE"],text=True)
    if "driftbench-dsv4-uc" not in mount or "fuse.gcsfuse" not in mount:
        raise ValueError("overlay does not use approved regional mount")
    bucket = cold._bucket(storage.Client())
    blob = bucket.blob(CAPSULE, generation=CAPSULE_GENERATION)
    blob.reload()
    data = cold._download(blob, cap=CAPSULE_BYTES)
    if len(data) != CAPSULE_BYTES or sha256(data).hexdigest() != CAPSULE_SHA:
        raise ValueError("native registered request capsule changed")
    requests._write_once(root/"requests.json",data)
    print(f"NATIVE_PREPARED {rank}",flush=True)


def worker_child(tag: str, pin: str, coordinator: str) -> int:
    """Waitable own process only; no TPU/node lifecycle or automatic retries."""
    rank = int(socket.gethostname().rsplit("-w-",1)[1])
    cold._identity(tag,pin,rank)
    root = watch.RUN_ROOT/tag
    protocol = REPO/"configs/greenfield-native-benchmark-protocol.json"
    args = [PYTHON,"-u","scripts/greenfield/run_short_decoder_ws32.py",
        "--native-benchmark-request",str(root/"requests.json"),
        "--native-benchmark-protocol",str(protocol),"--protocol-sha256",sha256(protocol.read_bytes()).hexdigest(),
        "--expected-code-hash",pin,"--process-id",str(rank),"--coordinator-address",coordinator,
        "--output",str(root/f"runner.rank{rank}.json")]
    env=dict(os.environ,JAX_PLATFORMS="tpu",XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        GLM_GREENFIELD_NATIVE_BENCHMARK="1",GLM_GREENFIELD_RUN_TAG=tag,PYTHONPATH=str(REPO))
    path=root/f"runner.rank{rank}.log"
    _plain_path(path)
    with path.open("xb") as log:
        child=subprocess.Popen(["timeout","--signal=TERM","--kill-after=60","86700",*args],
            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        rc=child.wait()
    persist(root/f"ended.rank{rank}.json",dict(tag=tag,code_hash=pin,rank=rank,
        host=socket.gethostname(),worker_exit_code=rc,supervisor_pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    publication_error = None
    try:
        published=subprocess.run([PYTHON,"-m","scripts.greenfield.ws32_native_benchmark_collect",
            "--tag",tag,"--code-hash",pin,"--rank",str(rank)],cwd=REPO,
            env=dict(os.environ,JAX_PLATFORMS="cpu",PYTHONPATH=str(REPO)),timeout=1200).returncode
    except Exception as exc:
        published = 1
        publication_error = f"{type(exc).__name__}: {exc}"
    persist(root/f"published.rank{rank}.json",dict(tag=tag,code_hash=pin,rank=rank,
        worker_exit_code=rc,publish_exit_code=published,error=publication_error))
    return rc or published


def replay_collected(root: Path, tag: str, pin: str) -> dict:
    """Original cold + same-request controls/score replay; final DB seal separate."""
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_entry as entry
    from scripts.greenfield import ws32_native_benchmark_evidence as evidence
    from scripts.greenfield.ws32_native_benchmark_result import replay_requests
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    from scripts.analysis.parse_xplane import aggregate_fleet
    from transformers import AutoTokenizer
    import re
    destination=root/"collected"
    # Reuse the worker's fixed datalocation recipe, without initializing JAX.
    args=entry.parse_args(["--native-benchmark-request",str(root/"requests.json"),
        "--native-benchmark-protocol",str(REPO/"configs/greenfield-native-benchmark-protocol.json"),
        "--protocol-sha256","unused-no-worker-preflight", "--expected-code-hash",pin,
        "--process-id","0","--coordinator-address","unused-no-backend",
        "--output",str(root/"runner.rank0.json")])
    captures=tuple(json.loads((args.topology_capture_root/f"topology.rank{r}.json").read_bytes()) for r in range(8))
    topology,captures,fleet_sha=original.validate_ws32_topology_fleet(captures,
        expected_topology_sha256=args.topology_sha256,expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name=args.slice_name)
    physical=original.build_ws32_physical_mesh(topology)
    if physical.mesh_hash!=args.mesh_sha256:raise ValueError("native replay mesh differs")
    config=Ws32DecoderConfig(original._geometry(),args.context_capacity,exact_dsa=True,
        strategy_nd_dense=True,host_main_rope_table=True)
    cold_report=evidence.replay_cold_fleet(root=destination,repo=REPO,pin=pin,captures=captures,
        physical_mesh=physical,topology_hash=args.topology_sha256,fleet_hash=fleet_sha,
        full_index_layers=config.full_index_slots)
    parents=[json.loads((destination/f"native.rank{r}"/"runner.json").read_bytes()) for r in range(8)]
    payload=json.loads((root/"requests.json").read_bytes())
    plan=json.loads(args.native_benchmark_protocol.read_bytes())
    for name,expected in plan['tokenizer_files'].items():
        if sha256((entry.TOKENIZER/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("replay tokenizer identity differs")
    tokenizer=AutoTokenizer.from_pretrained(entry.TOKENIZER,local_files_only=True,trust_remote_code=False)
    report=replay_requests(root=destination,payload=payload,plan=plan,pin=pin,parents=parents,
        tokenizer=tokenizer,full_index_layers=config.full_index_slots)
    if report['completed_requests']:
        traces=report['trace_originals']
        if len(traces)!=8 or {r['rank'] for r in traces}!=set(range(8)):
            raise ValueError("sampled campaign lacks fresh8host traces")
        text=(destination/'native.rank0/observer.optimized_hlo.txt').read_text()
        module=re.match(r'HloModule ([A-Za-z0-9_.-]+),',text)
        if module is None:raise ValueError("sampled observer module identity missing")
        trace=aggregate_fleet(destination,step_module_re=r'(?<![A-Za-z0-9_])'+re.escape(module[1])+r'(?![A-Za-z0-9_])',
                              allow_single_step=True)
        if trace['n_files']!=8 or trace['n_cores']!=64 or trace['steps_per_core']!=1:
            raise ValueError("sampled observer trace lacks exact8host64core single-step coverage")
        report.update(trace_physical_coverage_verified=True,trace=trace)
    report['cold']=cold_report
    persist(root/'request_replay.json',report)
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--code-hash",required=True)
    parser.add_argument("--reviewed-branch", default=BRANCH, type=reviewed_branch,
                        help="published owner branch (default: main); pin must equal its remote HEAD")
    parser.add_argument("--attach",action="store_true")
    parser.add_argument("--prepare-worker",action="store_true")
    parser.add_argument("--worker-child",action="store_true")
    parser.add_argument("--coordinator")
    args=parser.parse_args()
    cold._identity(args.tag,args.code_hash,0)
    if args.prepare_worker:
        prepare_worker(args.tag,args.code_hash); return 0
    if args.worker_child:
        return worker_child(args.tag,args.code_hash,args.coordinator)
    from google.cloud import storage
    source_preflight(args.code_hash, branch=args.reviewed_branch)
    root=watch.RUN_ROOT/args.tag
    _plain_path(root)
    with ExitStack() as stack:
        for path in watch.LOCKS:
            handle=stack.enter_context(path.open("a"))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.attach:
            root.mkdir(exist_ok=False)
            if shutil.disk_usage(root).free<6<<30: raise ValueError("restore6GiB controller launch floor")
            bucket=cold._bucket(storage.Client())
            live=sum(int(blob.size) for blob in bucket.list_blobs())
            if live+(10<<30)>=2_500_000_000_000: raise ValueError("native storage budget exceeds live cap")
            if next(iter(bucket.list_blobs(prefix=f"results/{args.tag}/",max_results=1)),None) is not None:
                raise ValueError("native remote tag already exists")
            pre=ssh(census_command()); validate_fleet(pre)
            requests._write_once(root/"census_pre.txt",pre.encode())
            sync=ssh(sync_command(args.code_hash, branch=args.reviewed_branch),timeout=300); markers(sync,"NATIVE_SYNC_OK")
            requests._write_once(root/"sync.txt",sync.encode())
            prepare=shlex.join([PYTHON,"-m","scripts.greenfield.launch_ws32_native_benchmark",
                "--prepare-worker","--tag",args.tag,"--code-hash",args.code_hash])
            prepared=ssh("cd "+shlex.quote(str(REPO))+" && JAX_PLATFORMS=cpu "+prepare,timeout=300)
            markers(prepared,"NATIVE_PREPARED")
            requests._write_once(root/"prepared.txt",prepared.encode())
            address=ssh("hostname -I",workers="0").strip().split()[0]
            ipaddress.ip_address(address)
            persist(root/"launch.json",dict(tag=args.tag,code_hash=args.code_hash,coordinator=address+":8476",
                reviewed_branch=args.reviewed_branch,
                storage_live_before=live,planned_archive_cap=10<<30,started_utc=datetime.now(timezone.utc).isoformat()))
            ssh(worker_command(args.tag,args.code_hash,address+":8476"))
        else:
            launch=json.loads((root/"launch.json").read_bytes())
            validate_attach(launch, tag=args.tag, pin=args.code_hash, branch=args.reviewed_branch)
        # Original observation receipt is append-only. A transient timeout never
        # releases leases to launch a replacement. Unknown state is retried.
        receipt=stack.enter_context((root/"native_watch.jsonl").open("a+"))
        receipt.seek(0); previous=receipt.read(); receipt.seek(0,os.SEEK_END)
        original=None
        if previous and not previous.endswith("\n"):
            raise ValueError("partial watch record; preserve and diagnose before attach")
        for line in previous.splitlines():
            row=json.loads(line)
            if row.get("tag") != args.tag or row.get("pin") != args.code_hash:
                raise ValueError("watch receipt identity differs")
            if row.get("fleet") is not None:
                observed = watch.parse_fleet("\n".join(watch.PREFIX+json.dumps(r) for r in row["fleet"]), args.tag, args.code_hash)
                original = observe_originals(original, observed)
        idle=0
        while True:
            row=dict(utc=datetime.now(timezone.utc).isoformat(),tag=args.tag,pin=args.code_hash)
            try:
                observed=watch.observe(args.tag,args.code_hash)
                original = observe_originals(original, observed)
                if all(not r["processes"] and not r["holders"] for r in observed):
                    publication = publication_state(args.tag, args.code_hash, observed)
                    row["publication"] = publication
                    idle = idle+1 if all(r["published"] is not None for r in publication) else 0
                else:
                    idle = 0
                row.update(status="OBSERVED",fleet=observed)
            except Exception as exc:
                idle=0; row.update(status="UNKNOWN_RETRYING",error=f"{type(exc).__name__}: {exc}")
            receipt.write(json.dumps(row,sort_keys=True)+"\n"); receipt.flush(); os.fsync(receipt.fileno())
            if idle>=2: break
            time.sleep(30)
        post=ssh(census_command()); validate_fleet(post)
        # A same-tag collection retry gets a FRESH idle census above, but keeps
        # the original archived post-census bytes. Never overwrite its history.
        if (root/"census_post.txt").exists():
            validate_fleet((root/"census_post.txt").read_text())
        else:
            requests._write_once(root/"census_post.txt",post.encode())
        print("NATIVE_FLEET_IDLE collection/replay required; no SUCCESS claimed",flush=True)
        if any(r["published"]["publish_exit_code"] != 0 for r in publication):
            raise RuntimeError("workers ended but original publication failed; recover same-tag files, never rerun model")
        client=storage.Client(); bucket=cold._bucket(client)
        blobs={b.name:b for b in bucket.list_blobs(prefix=f"results/{args.tag}/")}
        destination=root/"collected"
        destination.mkdir(exist_ok=True)
        request_receipts=[requests.collect(destination,args.tag,args.code_hash,rank,client,blobs)
                          for rank in range(8)]
        cold_receipts=cold.collect_fleet(destination=destination,tag=args.tag,pin=args.code_hash,client=client,blobs=blobs)
        persist(root/"collection.json",dict(complete=True,quality_proven=False,final_seal_required=True))
        report=replay_collected(root,args.tag,args.code_hash)
        from scripts.greenfield.ws32_native_benchmark_archive import archive
        requests._write_once(root/'protocol.json', (REPO/'configs/greenfield-native-benchmark-protocol.json').read_bytes())
        result=archive(root=root,tag=args.tag,pin=args.code_hash,report=report,
            cold_receipts=cold_receipts,request_receipts=request_receipts,blobs=blobs,client=client)
        print(json.dumps(result),flush=True)
        return 0


if __name__=="__main__":
    raise SystemExit(main())
