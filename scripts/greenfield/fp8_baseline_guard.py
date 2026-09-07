#!/usr/bin/env python3
"""Read-only baseline admission for the bounded FP8 mechanism probe.

Uses the existing WS32 lock observer and publisher, not another TPU supervisor.
The outer wrapper owns both leases and the full process/container census.
"""
from __future__ import annotations

import argparse
import ast
import base64
import inspect
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess

from scripts.greenfield import watch_ws32_run


def accelerator_snapshot() -> dict[str, tuple[int, int, int, int]]:
    """Bind the four canonical character devices; no open or device mutation."""
    paths = sorted(Path("/dev").glob("accel*"))
    if paths != [Path(f"/dev/accel{i}") for i in range(4)]:
        raise RuntimeError("unexpected accelerator device set")
    result = {}
    for path in paths:
        value = path.lstat()
        if not stat.S_ISCHR(value.st_mode):
            raise RuntimeError("accelerator path is not a character device")
        result[str(path)] = (value.st_dev, value.st_ino, value.st_mode, value.st_rdev)
    if len({v[3] for v in result.values()}) != 4:
        raise RuntimeError("duplicate accelerator identity")
    return result


def require_accelerators_idle() -> dict:
    """Root fuser observes open devices independently of the libtpu lock path."""
    if os.geteuid() != 0:
        raise RuntimeError("accelerator census requires root observation")
    before = accelerator_snapshot()
    for path in before:
        result = subprocess.run(
            ["env", "LC_ALL=C", "fuser", path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
            raise RuntimeError(
                f"accelerator busy or holder observation unknown: {path}"
            )
    if accelerator_snapshot() != before:
        raise RuntimeError("accelerator identity changed during observation")
    # A starting Python runner may not yet have opened an accelerator. Match
    # script argv, not the source text inside this observer's command carrier.
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            args = (path / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        for arg in args[1:3]:
            if b"scripts/greenfield/" in arg and arg.endswith(b".py"):
                raise RuntimeError(f"greenfield runner remains live: {path.name}")
    return before


def census_command() -> str:
    tree = ast.parse(watch_ws32_run.REMOTE)
    lock_function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "libtpu_holders"
    )
    source = (
        "import os, pathlib, socket, stat, subprocess, json\nfrom pathlib import Path\n"
        + ast.unparse(lock_function)
        + "\n"
        + inspect.getsource(accelerator_snapshot)
        + "\n"
        + inspect.getsource(require_accelerators_idle)
        + "\n"
        + "if libtpu_holders(): raise RuntimeError('libtpu holders remain')\n"
        + "devices = require_accelerators_idle()\n"
        + "print('FP8_IDLE ' + json.dumps(dict(host=socket.gethostname(), "
        "boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), devices=devices)))\n"
    )
    encoded = base64.b64encode(source.encode()).decode()
    program = f"import base64; exec(base64.b64decode({encoded!r}))"
    return "sudo -n /home/gianl/vllm-env/bin/python -c " + shlex.quote(program)


def validate_fleet(text: str) -> None:
    rows = [
        json.loads(s[len("FP8_IDLE ") :])
        for s in text.splitlines()
        if s.startswith("FP8_IDLE ")
    ]
    if len(rows) != 8 or len({r["host"] for r in rows}) != 8:
        raise ValueError("incomplete/duplicate authenticated idle fleet")
    if {r["host"].rsplit("-w-", 1)[-1] for r in rows} != set(map(str, range(8))):
        raise ValueError("idle fleet ranks differ")
    if any(not r["boot_id"] or len(r["devices"]) != 4 for r in rows):
        raise ValueError("missing fleet device/boot evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("census-command", "validate-fleet"))
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.action == "census-command":
        print(census_command())
    else:
        if args.file is None:
            parser.error("validate-fleet requires --file")
        validate_fleet(args.file.read_text())


if __name__ == "__main__":
    main()
