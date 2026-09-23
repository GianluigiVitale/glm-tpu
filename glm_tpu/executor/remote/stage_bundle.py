"""Remote helper: receive the staged bundle on stdin, verify its digest and extract it.

One JSON argument: ``root`` (the run directory; created 0700 on every host but rank 0, where the
controller created it), ``digest`` (SHA-256 of the gzipped tar on stdin) and ``hosts`` (the
authenticated hostnames; rank = position of this host). Extraction uses ``tarfile``'s ``data``
filter, so it runs on ``fleet.worker_python`` (3.12), as the 181c013e staging program did.

Standard library only; sent as ``<interpreter> -c <this text> <json>``, never imported on a host.
"""

import hashlib
import io
import json
import os
import pathlib
import socket
import sys
import tarfile

KEYS = ("digest", "hosts", "root")


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
    os.umask(0o077)
    root = pathlib.Path(args["root"])
    if rank_of(args["hosts"]):
        root.mkdir(mode=0o700)
    data = sys.stdin.buffer.read()
    if hashlib.sha256(data).hexdigest() != args["digest"]:
        raise RuntimeError("staging transport differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
