"""Remote helper: print named records of one directory as base64 JSON (one line).

One JSON argument: ``dir`` (the directory), ``hosts`` (authenticated hostnames; rank = position of
this host) and ``names`` (file names; ``{rank}`` is replaced by this host's rank, e.g.
``runner.rank{rank}.json``). Prints ``{"<name>": "<base64 of the bytes>", ...}`` for the named
regular files that exist (sorted keys); a name with a path separator refuses.

Runs on ``fleet.helper_python``. Standard library only; sent as ``<interpreter> -c <this text>
<json>``, never imported on a host.
"""

import base64
import json
import pathlib
import re
import socket
import sys

KEYS = ("dir", "hosts", "names")
TEMPLATE = re.compile(r"[A-Za-z0-9_.-]*(\{rank\})?[A-Za-z0-9_.-]*")


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
    directory, names = pathlib.Path(args["dir"]), args["names"]
    rank = rank_of(args["hosts"])
    if not isinstance(names, list) or not names:
        raise SystemExit("fetch names differ")
    found = {}
    for template in names:
        if not isinstance(template, str) or TEMPLATE.fullmatch(template) is None:
            raise SystemExit("fetch names differ")
        name = template.replace("{rank}", str(rank))
        if name in ("", ".", ".."):
            raise SystemExit("fetch names differ")
        path = directory / name
        if path.is_file():
            found[name] = base64.b64encode(path.read_bytes()).decode()
    print(json.dumps(found, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
