#!/usr/bin/env python3
"""Observe an existing WS32 fleet after controller loss, without starting work.

Holds both user leases while polling exact processes over authenticated SSH.
Unknown observations retain the leases and retry. Completion means only that the
original processes disappeared and libtpu has no holders; original outputs still
need collection, the protected census and the existing recovery sealer.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


LOCKS = (
    Path("/home/gianl/glm-run/.glm_pod_workload.lock"),
    Path("/home/gianl/.glm-tpu-rsync.lock"),
)
RUN_ROOT = Path("/home/gianl/glm-run")
PREFIX = "WS32_WATCH "
# This program reads /proc and local files only. The script path must be an
# individual argv element, so the observer cannot match its own command text.
REMOTE = r'''
import hashlib, json, os, pathlib, socket, subprocess, sys
tag, pin = sys.argv[1:]
hostname = socket.gethostname()
rank = int(hostname.rsplit("-w-", 1)[1])
root = pathlib.Path("/home/gianl/glm-run") / tag
output = str(root / ("runner.rank%d.json" % rank))
processes = []
for path in pathlib.Path("/proc").iterdir():
    if not path.name.isdecimal():
        continue
    try:
        raw = (path / "cmdline").read_bytes()
        argv = [s.decode() for s in raw.split(b"\0") if s]
        scripts = ("scripts/greenfield/run_short_decoder_ws32.py",
                   "/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_short_decoder_ws32.py")
        if not any(arg in scripts for arg in argv[1:3]):
            continue
        def option(name):
            if argv.count(name) != 1:
                raise RuntimeError("missing/duplicate option " + name)
            return argv[argv.index(name) + 1]
        if option("--output") != output or option("--expected-code-hash") != pin:
            raise RuntimeError("another WS32 run is present")
        first = (path / "stat").read_text().rsplit(")", 1)[1].split()
        executable = os.readlink(path / "exe")
        second = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if first[19] != second[19] or (path / "cmdline").read_bytes() != raw:
            raise RuntimeError("process changed during observation")
        if second[0] not in ("Z", "X"):
            processes.append(dict(pid=int(path.name), start_ticks=second[19],
                parent_pid=int(second[1]), executable=executable,
                argv_sha256=hashlib.sha256(raw).hexdigest()))
    except (FileNotFoundError, ProcessLookupError):
        continue
holder = subprocess.run(["sudo", "-n", "fuser", "/tmp/libtpu_lockfile"],
    capture_output=True, text=True, timeout=15)
if holder.returncode not in (0, 1) or (holder.returncode == 1 and holder.stderr.strip()):
    raise RuntimeError("cannot establish libtpu holder state")
holders = sorted(int(p) for p in holder.stdout.split())
if bool(holders) != (holder.returncode == 0):
    raise RuntimeError("inconsistent fuser output")
progress = ""
log = root / ("runner.rank%d.log" % rank)
if log.exists():
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        lines = stream.read(4096).decode(errors="replace").splitlines()
    progress = next((s for s in reversed(lines) if s.startswith("GREENFIELD_WS32_PREFILL_CHUNK")), "")
print("WS32_WATCH " + json.dumps(dict(rank=rank, host=hostname, tag=tag, pin=pin,
    boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    processes=processes, holders=holders, progress=progress,
    output_present=pathlib.Path(output).is_file())))
'''


def parse_fleet(text: str, tag: str, pin: str) -> list[dict]:
    """Require one complete, correctly addressed observation from every rank."""
    rows = [json.loads(line[len(PREFIX):]) for line in text.splitlines()
            if line.startswith(PREFIX)]
    if (len(rows) != 8 or {r["rank"] for r in rows} != set(range(8))
            or len({r["host"] for r in rows}) != 8):
        raise ValueError("incomplete or duplicate fleet observation")
    for row in rows:
        if row["tag"] != tag or row["pin"] != pin:
            raise ValueError("run identity drifted")
        if not row["host"].endswith("-w-" + str(row["rank"])):
            raise ValueError("host/rank mismatch")
        if len(row["processes"]) > 1:
            raise ValueError("multiple runner processes on one host")
    return sorted(rows, key=lambda row: row["rank"])


def require_original_fleet(original: list[dict], observed: list[dict]) -> None:
    """PID reuse, reboots or replacement runners are unknown, never completion."""
    for old, new in zip(original, observed, strict=True):
        for key in ("rank", "host", "boot_id", "tag", "pin"):
            if old[key] != new[key]:
                raise ValueError("fleet identity changed: " + key)
        if new["processes"] and new["processes"] != old["processes"]:
            raise ValueError("original process was replaced")


def resume_baseline(text: str, tag: str, pin: str) -> list[dict] | None:
    """Replay the append-only receipt; never silently adopt replacement PIDs."""
    if text and not text.endswith("\n"):
        raise ValueError("incomplete watch receipt; preserve and diagnose before resume")
    original = None
    for line in text.splitlines():
        record = json.loads(line)
        if record["tag"] != tag or record["pin"] != pin:
            raise ValueError("watch receipt run identity drifted")
        if record["status"] == "UNKNOWN_RETRYING":
            continue
        if record["status"] not in ("OBSERVED", "READY_FOR_CENSUS"):
            raise ValueError("unknown watch receipt status")
        rows = parse_fleet("\n".join(PREFIX + json.dumps(r) for r in record["fleet"]), tag, pin)
        if original is None:
            if not all(len(r["processes"]) == 1 for r in rows):
                raise ValueError("receipt lacks the original eight-process baseline")
            original = rows
        require_original_fleet(original, rows)
    return original


def observe(tag: str, pin: str) -> list[dict]:
    payload = base64.b64encode(REMOTE.encode()).decode()
    command = shlex.join([
        "/usr/bin/python3", "-c",
        "import base64;exec(base64.b64decode(" + repr(payload) + "))", tag, pin,
    ])
    result = subprocess.run([
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", "db-v4-64-od",
        "--zone", "us-central2-b", "--worker=all", "--command=" + command,
    ], capture_output=True, text=True, timeout=55)
    if result.returncode:
        raise RuntimeError("SSH observation failed: " + result.stderr[-1000:])
    return parse_fleet(result.stdout, tag, pin)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--once", action="store_true", help="read-only fleet snapshot; no leases")
    args = parser.parse_args()
    if (not re.fullmatch(r"greenfield_ws32_short_decoder_[a-z0-9_]+_[0-9]{8}T[0-9]{15}Z", args.tag)
            or not re.fullmatch(r"[0-9a-f]{40}", args.code_hash)):
        parser.error("invalid exact run tag or pin")
    if args.once:
        print(json.dumps(observe(args.tag, args.code_hash), sort_keys=True))
        return 0
    run = RUN_ROOT / args.tag
    if not run.is_dir():
        parser.error("existing run directory required")
    with ExitStack() as stack:
        for path in LOCKS:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = stack.enter_context((run / "watch.jsonl").open("a+"))
        receipt.seek(0)
        original = resume_baseline(receipt.read(), args.tag, args.code_hash)
        receipt.seek(0, os.SEEK_END)
        idle_observations = 0
        while True:
            record = {"utc": datetime.now(timezone.utc).isoformat(),
                      "tag": args.tag, "pin": args.code_hash}
            try:
                current = observe(args.tag, args.code_hash)
                if original is None:
                    if not all(len(r["processes"]) == 1 for r in current):
                        raise ValueError("initial attachment needs eight live runners")
                    original = current
                require_original_fleet(original, current)
                idle = all(not r["processes"] and not r["holders"] for r in current)
                idle_observations = idle_observations + 1 if idle else 0
                record.update(status="READY_FOR_CENSUS" if idle_observations >= 2 else "OBSERVED",
                              fleet=current, numerical_success_claim=False)
            except (ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                idle_observations = 0
                record.update(status="UNKNOWN_RETRYING", error=str(error))
            receipt.write(json.dumps(record, sort_keys=True) + "\n")
            receipt.flush()
            os.fsync(receipt.fileno())
            print(json.dumps(record, sort_keys=True), flush=True)
            if idle_observations >= 2:
                return 0
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
