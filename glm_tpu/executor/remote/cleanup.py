"""Remote helper: terminate only this run's authenticated worker on this host, never an unknown holder.

One JSON argument: ``root`` (run directory), ``hosts`` (authenticated hostnames; rank = position of
this host), ``pin`` (the staged commit) and ``module`` (the worker module the process must run).
Nothing to do (exit 0) when there is no start marker, no such process, or the pid was reused (start
ticks differ, or a zombie). Otherwise the process must match the marker's hostname, code hash and
boot id, and its argv must contain ``module``, ``root`` and ``pin``; only then it receives
``SIGKILL`` through a pidfd opened before the checks. A mismatch raises (non-zero exit).

Runs on ``fleet.helper_python``. Standard library only; sent as ``<interpreter> -c <this text>
<json>``, never imported on a host.
"""

import json
import os
import pathlib
import signal
import socket
import sys

KEYS = ("hosts", "module", "pin", "root")


def arguments(argv, keys):
    if len(argv) != 1:
        raise SystemExit("expected exactly one JSON argument")
    value = json.loads(argv[0])
    if not isinstance(value, dict) or sorted(value) != sorted(keys):
        raise SystemExit("helper arguments differ")
    return value


def rank_of(hosts):
    """This host's position in the authenticated host list (refuses a host outside it)."""
    hostname = socket.gethostname()
    if (not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts)
            or len(set(hosts)) != len(hosts) or hostname not in hosts):
        raise RuntimeError("host is not in the authenticated fleet")
    return hosts.index(hostname)


def main(argv):
    args = arguments(argv, KEYS)
    root, pin, module = pathlib.Path(args["root"]), args["pin"], args["module"]
    marker = root / ("worker_started.rank%d.json" % rank_of(args["hosts"]))
    if not marker.exists():
        return 0
    owner = json.loads(marker.read_text())
    pid = int(owner["pid"])
    proc = pathlib.Path("/proc") / str(pid)
    if not proc.exists():
        return 0
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return 0
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] != owner["start_ticks"] or fields[0] in ("Z", "X"):
            return 0
        boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if owner["hostname"] != socket.gethostname() or owner["code_hash"] != pin or owner["boot_id"] != boot_id:
            raise RuntimeError("cleanup identity differs")
        command = (proc / "cmdline").read_bytes().split(b"\0")
        if module.encode() not in command or str(root).encode() not in command or pin.encode() not in command:
            raise RuntimeError("cleanup argv differs")
        signal.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
