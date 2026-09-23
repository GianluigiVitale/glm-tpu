"""Remote helper: record this host's worker start marker, then become the worker process.

One JSON argument: ``root`` (run directory), ``hosts`` (authenticated hostnames; rank = position
of this host), ``pin`` (the staged commit), ``worker_python``, ``pythonpath`` (entries after
``<root>/source``), ``module`` (the worker module run with ``-m``), ``env`` (the handshake
variables, e.g. ``JAX_PLATFORMS`` and the worker flag) and ``argv`` (the worker's arguments).

The marker ``worker_started.rank<r>.json`` (pid, hostname, code_hash, boot_id, start_ticks) is
created with ``open("x")`` before any heavy import; this same process then ``chdir``-s to
``<root>/source`` and ``execv``-s ``worker_python -m module argv...``, so the pid and start ticks
the cleanup helper authenticates are the worker's own.

Runs on ``fleet.helper_python``. Standard library only; sent as ``<interpreter> -c <this text>
<json>``, never imported on a host.
"""

import json
import os
import pathlib
import re
import socket
import sys

KEYS = ("argv", "env", "hosts", "module", "pin", "pythonpath", "root", "worker_python")
DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


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


def strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def main(argv):
    args = arguments(argv, KEYS)
    env, python = args["env"], args["worker_python"]
    if (not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
            or "PYTHONPATH" in env or not strings(args["argv"]) or not strings(args["pythonpath"])
            or any(os.pathsep in entry for entry in args["pythonpath"]) or not isinstance(python, str)
            or not os.path.isabs(python) or not isinstance(args["module"], str)
            or DOTTED.fullmatch(args["module"]) is None):
        raise SystemExit("worker start arguments differ")
    root = pathlib.Path(args["root"])
    rank = rank_of(args["hosts"])
    owner = dict(pid=os.getpid(), hostname=socket.gethostname(), code_hash=args["pin"],
                 boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                 start_ticks=pathlib.Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    with (root / ("worker_started.rank%d.json" % rank)).open("x") as stream:
        json.dump(owner, stream)
    os.chdir(root / "source")
    os.environ.update(env)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(root / "source"), *args["pythonpath"]])
    os.execv(python, [python, "-m", args["module"], *args["argv"]])
    return 1  # unreachable: execv replaces this process or raises


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
