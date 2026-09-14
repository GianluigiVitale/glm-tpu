"""Default-off serialized single-user controller on the EXISTING eight hosts.

No infrastructure lifecycle calls or benchmark registration. Collection is
followed by original-evidence replay, DB linkage and regional archive sealing.
Real release admission remains required. Never launch beside an active benchmark.
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

from glm_tpu import user_request
from scripts.release import ws32_user_worker as worker
from scripts.release import ws32_user_transport as transport
from scripts.release.ws32_user_evidence import replay_collected
from scripts.release.ws32_user_archive import seal
from scripts.greenfield import launch_ws32_native_benchmark as shared
from scripts.greenfield import watch_ws32_run as watch
from scripts.greenfield import ws32_native_benchmark_collect as originals
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.fp8_baseline_guard import census_command, validate_fleet

REPO, PYTHON = worker.REPO, shared.PYTHON
MODULE = "scripts.release.launch_ws32_user_request"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def persist(path, value):
    with transport.private_writes():
        originals._write_once(path, user_request.canonical(value)+b"\n")


def request_object(tag):
    return f"results/{tag}/private_input/request.json"


def validate_transport(args):
    worker.identity(args.tag, args.code_hash, 0)
    if (not isinstance(args.request_file_sha256, str) or len(args.request_file_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.request_file_sha256)
            or type(args.request_generation) is not int or args.request_generation <= 0
            or type(args.request_bytes) is not int or not 0 < args.request_bytes <= user_request.PAYLOAD_CAP
            or type(args.wall_seconds) is not int or not 1 <= args.wall_seconds <= 86400):
        raise ValueError("user original transport/deadline identity differs")
    host, port = args.coordinator.rsplit(":", 1)
    ipaddress.ip_address(host)
    if port != "8476":
        raise ValueError("user coordinator port differs")


def role_command(args, role):
    validate_transport(args)
    if role not in ("prepare", "worker", "publish"):
        raise ValueError("unknown user worker role")
    return [PYTHON, "-m", MODULE, "--role", role, "--tag", args.tag,
        "--code-hash", args.code_hash, "--request-file-sha256", args.request_file_sha256,
        "--request-generation", str(args.request_generation), "--request-bytes", str(args.request_bytes),
        "--wall-seconds", str(args.wall_seconds), "--coordinator", args.coordinator]


def remote_command(args, role):
    command = shlex.join(role_command(args, role))
    prefix = "set -euo pipefail; umask 077; cd " + shlex.quote(str(REPO)) + "; "
    env = "env JAX_PLATFORMS=cpu GLM_GREENFIELD_USER_REQUEST=1 PYTHONPATH=" + shlex.quote(str(REPO)) + " "
    if role == "worker":
        return prefix + "nohup " + env + command + " > /home/gianl/glm-run/" + args.tag + "/supervisor.rank${HOSTNAME##*-w-}.log 2>&1 </dev/null &"
    return prefix + env + command


def worker_args(args, rank):
    root = worker.RUN_ROOT / args.tag
    return worker.parse_args(["--user-request", str(root / "request.json"),
        "--request-file-sha256", args.request_file_sha256, "--expected-code-hash", args.code_hash,
        "--process-id", str(rank), "--coordinator-address", args.coordinator,
        "--wall-seconds", str(args.wall_seconds), "--output", str(root / f"runner.rank{rank}.json")])


def prepare_worker(args):
    from google.cloud import storage
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    validate_transport(args)
    _require_clean_code(args.code_hash)
    rank = int(socket.gethostname().rsplit("-w-", 1)[1])
    worker.identity(args.tag, args.code_hash, rank)
    root = worker.RUN_ROOT / args.tag
    user_request._plain(root)
    if rank != 0:
        root.mkdir(mode=0o700, exist_ok=False)
    transport.private_directory(root)
    if shutil.disk_usage(root).free < 6 << 30:
        raise ValueError("user worker requires the original6GiB launch floor")
    site = worker_args(args, rank)
    mount = subprocess.check_output(["findmnt", "-T", str(site.checkpoint_root), "-n", "-o", "FSTYPE"], text=True).strip()
    if mount != "tmpfs" or len(list(site.checkpoint_root.glob("device_slot_*.safetensors"))) != 4:
        raise ValueError("existing final-layout RAM checkpoint missing; no recreation")
    overlay_mount = subprocess.check_output(["findmnt", "-T", str(site.strategy_nd_dense_overlay_root), "-n", "-o", "SOURCE,FSTYPE"], text=True)
    if "driftbench-dsv4-uc" not in overlay_mount or "fuse.gcsfuse" not in overlay_mount:
        raise ValueError("user overlay requires the approved regional mount")
    bucket = transport.approved_bucket(storage.Client())
    blob = bucket.blob(request_object(args.tag), generation=args.request_generation)
    blob.reload(if_generation_match=args.request_generation)
    raw = transport.cold._download(blob, cap=args.request_bytes)
    if len(raw) != args.request_bytes or sha256(raw).hexdigest() != args.request_file_sha256:
        raise ValueError("user original request bytes differ")
    with transport.private_writes():
        originals._write_once(root / "request.json", raw)
    worker.preflight(site)  # no backend; full local input/source/tokenizer checks
    print(f"USER_PREPARED {rank}", flush=True)


def publish_worker(args):
    from google.cloud import storage
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    validate_transport(args)
    _require_clean_code(args.code_hash)
    rank = int(socket.gethostname().rsplit("-w-", 1)[1])
    value = transport.publish(root=worker.RUN_ROOT / args.tag, tag=args.tag, pin=args.code_hash,
        rank=rank, request_file_sha256=args.request_file_sha256, client=storage.Client())
    print(json.dumps(value, sort_keys=True), flush=True)


def worker_child(args):
    validate_transport(args)
    rank = int(socket.gethostname().rsplit("-w-", 1)[1])
    site = worker_args(args, rank)
    root = site.output.parent
    transport.private_directory(root)
    command = [PYTHON, "-u", "scripts/greenfield/run_short_decoder_ws32.py",
        "--user-request", str(site.user_request), "--request-file-sha256", args.request_file_sha256,
        "--expected-code-hash", args.code_hash, "--process-id", str(rank),
        "--coordinator-address", args.coordinator, "--wall-seconds", str(args.wall_seconds),
        "--output", str(site.output)]
    env = dict(os.environ, JAX_PLATFORMS="tpu", XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        GLM_GREENFIELD_USER_REQUEST="1", GLM_GREENFIELD_RUN_TAG=args.tag, PYTHONPATH=str(REPO))
    worker_started, worker_error = False, None
    try:
        worker.preflight(site)
        with transport.private_writes(), (root / f"runner.rank{rank}.log").open("xb") as log:
            child = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=60",
                str(args.wall_seconds+300), *command], cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            worker_started = True
            rc = child.wait()
    except Exception as exc:
        if worker_started:
            # A failed wait cannot establish process exit. No fabricated ended
            # marker, replacement worker or publication over an unknown owner.
            raise
        rc, worker_error = 1, type(exc).__name__
    persist(root / f"ended.rank{rank}.json", dict(tag=args.tag, code_hash=args.code_hash,
        rank=rank, request_file_sha256=args.request_file_sha256, host=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        worker_exit_code=rc, supervisor_pid=os.getpid(), worker_started=worker_started,
        worker_error_type=worker_error))
    error = None
    try:
        published = subprocess.run(role_command(args, "publish"), cwd=REPO,
            env=dict(os.environ, JAX_PLATFORMS="cpu", GLM_GREENFIELD_USER_REQUEST="1", PYTHONPATH=str(REPO)),
            timeout=1200).returncode
    except Exception as exc:
        published, error = 1, type(exc).__name__
    persist(root / f"published.rank{rank}.json", dict(tag=args.tag, code_hash=args.code_hash,
        rank=rank, request_file_sha256=args.request_file_sha256, worker_exit_code=rc,
        publish_exit_code=published, error=error))
    return rc or published


def watch_originals(root, args):
    """Wait for the same original processes; unknown state never restarts work."""
    original = None
    with transport.private_writes(), (root / "user_watch.jsonl").open("a+") as stream:
        stream.seek(0)
        previous = stream.read()
        if previous and not previous.endswith("\n"):
            raise ValueError("partial user watch record; preserve and diagnose")
        for line in previous.splitlines():
            row = json.loads(line)
            if row.get("tag") != args.tag or row.get("pin") != args.code_hash:
                raise ValueError("user watch receipt identity differs")
            if row.get("fleet") is not None:
                observed = watch.parse_fleet("\n".join(watch.PREFIX+json.dumps(r) for r in row["fleet"]), args.tag, args.code_hash)
                original = shared.observe_originals(original, observed)
        stream.seek(0, os.SEEK_END)
        idle = 0
        while True:
            row = dict(utc=stamp(), tag=args.tag, pin=args.code_hash)
            try:
                observed = watch.observe(args.tag, args.code_hash)
                original = shared.observe_originals(original, observed)
                if all(not r["processes"] and not r["holders"] for r in observed):
                    publication = shared.publication_state(args.tag, args.code_hash, observed)
                    for state in publication:
                        for marker in (state["ended"], state["published"]):
                            if marker is not None and marker.get("request_file_sha256") != args.request_file_sha256:
                                raise ValueError("user publication belongs to another input")
                    row["publication"] = publication
                    idle = idle+1 if all(r["published"] is not None for r in publication) else 0
                else:
                    idle = 0
                row.update(status="OBSERVED", fleet=observed)
            except Exception as exc:
                idle = 0
                row.update(status="UNKNOWN_RETRYING", error_type=type(exc).__name__)
            stream.write(json.dumps(row, sort_keys=True)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            if idle >= 2:
                return original, publication
            time.sleep(30)  # autonomous watchdog, not repeated model-agent polling


def controller(args):
    from google.cloud import storage
    if not socket.gethostname().endswith("-w-0"):
        raise ValueError("user controller and both leases must live on worker0")
    shared.source_preflight(args.code_hash, branch=args.reviewed_branch)
    root = worker.RUN_ROOT / args.tag
    user_request._plain(root)
    with ExitStack() as stack:
        for path in watch.LOCKS:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.attach:
            if args.request is None or args.request.resolve().is_relative_to(REPO.resolve()):
                raise ValueError("prepared user input outside Git is required")
            raw = user_request.read_bounded(args.request, user_request.PAYLOAD_CAP)
            user_request.validate(json.loads(raw))
            if shutil.disk_usage(worker.RUN_ROOT).free < 6 << 30:
                raise ValueError("restore6GiB controller launch floor")
            root.mkdir(mode=0o700, exist_ok=False)
            bucket = transport.approved_bucket(storage.Client())
            live = sum(int(blob.size) for blob in bucket.list_blobs())
            if live + (10 << 30) >= 2_500_000_000_000:
                raise ValueError("user archive would exceed regional live-storage cap")
            if next(iter(bucket.list_blobs(prefix=f"results/{args.tag}/", max_results=1)), None) is not None:
                raise ValueError("user remote run namespace already exists")
            pre = shared.ssh(census_command())
            validate_fleet(pre)
            with transport.private_writes():
                originals._write_once(root / "census_pre.txt", pre.encode())
                originals._write_once(root / "request.json", raw)
            sync = shared.ssh(shared.sync_command(args.code_hash, branch=args.reviewed_branch), timeout=300)
            shared.markers(sync, "NATIVE_SYNC_OK")
            with transport.private_writes():
                originals._write_once(root / "sync.txt", sync.encode())
            uploaded = publish_exact(bucket, request_object(args.tag), root / "request.json",
                                      digest_file(root / "request.json"), compressed=False)
            args.request_generation, args.request_bytes = int(uploaded["generation"]), len(raw)
            args.request_file_sha256 = sha256(raw).hexdigest()
            address = shared.ssh("hostname -I", workers="0").strip().split()[0]
            ipaddress.ip_address(address)
            args.coordinator = address+":8476"
            prepared = shared.ssh(remote_command(args, "prepare"), timeout=300)
            shared.markers(prepared, "USER_PREPARED")
            with transport.private_writes():
                originals._write_once(root / "prepared.txt", prepared.encode())
            persist(root / "launch.json", dict(tag=args.tag, code_hash=args.code_hash,
                reviewed_branch=args.reviewed_branch, coordinator=args.coordinator,
                request_file_sha256=args.request_file_sha256, request_generation=args.request_generation,
                request_bytes=args.request_bytes, wall_seconds=args.wall_seconds,
                storage_live_before=live, planned_archive_cap=10 << 30, input_object=uploaded,
                started_utc=stamp(), benchmark=False))
            try:
                shared.ssh(remote_command(args, "worker"))
            except Exception as exc:
                # SSH may have delivered some/all commands. Keep both leases
                # and observe the originals; never repeat an ambiguous dispatch.
                persist(root / "dispatch_unknown.json", dict(tag=args.tag, code_hash=args.code_hash,
                    utc=stamp(), error_type=type(exc).__name__, retry_authorized=False))
        else:
            transport.private_directory(root)
            launch = json.loads(user_request.read_bounded(root / "launch.json", 64 << 10))
            shared.validate_attach(launch, tag=args.tag, pin=args.code_hash, branch=args.reviewed_branch)
            if launch.get("benchmark") is not False:
                raise ValueError("user attach cannot adopt a benchmark")
            for field in ("request_file_sha256", "request_generation", "request_bytes", "wall_seconds", "coordinator"):
                setattr(args, field, launch[field])
            validate_transport(args)
            raw = user_request.read_bounded(root / "request.json", user_request.PAYLOAD_CAP)
            if len(raw) != args.request_bytes or sha256(raw).hexdigest() != args.request_file_sha256:
                raise ValueError("user attach original input changed")
            user_request.validate(json.loads(raw))
        original, publication = watch_originals(root, args)
        post = shared.ssh(census_command())
        validate_fleet(post)
        if (root / "census_post.txt").exists():
            validate_fleet((root / "census_post.txt").read_text())
        else:
            with transport.private_writes():
                originals._write_once(root / "census_post.txt", post.encode())
        if any(r["published"]["publish_exit_code"] != 0 for r in publication):
            raise RuntimeError("original user publication failed; recover same-tag originals, never regenerate")
        client = storage.Client()
        bucket = transport.approved_bucket(client)
        blobs = {b.name: b for b in bucket.list_blobs(prefix=f"results/{args.tag}/")}
        destination = root / "collected"
        destination.mkdir(mode=0o700, exist_ok=True)
        report = transport.collect(destination=destination, tag=args.tag, pin=args.code_hash,
            request_file_sha256=args.request_file_sha256, original_fleet=original, client=client, blobs=blobs)
        persist(root / "user_collection.json", report)
        replay = replay_collected(root, args.tag, args.code_hash)
        persist(root / "user_replay.json", replay)
        result = seal(root=root, tag=args.tag, pin=args.code_hash, report=replay,
                      collection=report, blobs=blobs, client=client)
        print(json.dumps(result), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("controller", "prepare", "worker", "publish"), default="controller")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--reviewed-branch", default="main", type=shared.reviewed_branch)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--request-file-sha256")
    parser.add_argument("--request-generation", type=int)
    parser.add_argument("--request-bytes", type=int)
    parser.add_argument("--coordinator")
    args = parser.parse_args(argv)
    if os.environ.get("GLM_GREENFIELD_USER_REQUEST") != "1":
        raise ValueError("user launch is default-off; deployment admission required")
    worker.identity(args.tag, args.code_hash, 0)
    if args.attach and (args.request is not None or args.wall_seconds is not None
                       or args.request_generation is not None or args.request_bytes is not None
                       or args.request_file_sha256 is not None or args.coordinator is not None):
        raise ValueError("attach must reuse the original request, deadline and transport; no overrides")
    if args.wall_seconds is None:
        args.wall_seconds = 3600
    if not 1 <= args.wall_seconds <= 86400:
        raise ValueError("user operational deadline must be1..86400seconds")
    if args.role != "controller" and args.attach:
        raise ValueError("only the controller can attach")
    if args.role == "prepare":
        prepare_worker(args)
        return 0
    if args.role == "publish":
        publish_worker(args)
        return 0
    if args.role == "worker":
        return worker_child(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
