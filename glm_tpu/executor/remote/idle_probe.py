"""Remote helper: this host is idle -- no libtpu holder and no live worker of this run.

One JSON argument: ``root`` (the run directory; it need not exist yet) and ``hosts`` (null before
staging, else the authenticated hostnames: this host must be one of them). Checked: the
``/tmp/libtpu_lockfile`` holders (``libtpu_holders`` is the fleet collector's guard, verbatim), and
every start marker of the run that belongs to this host (its own rank's marker when ``hosts`` is
given, and every marker whose recorded hostname is this host's) whose process is alive with the
recorded start ticks. Prints ``IDLE <hostname>``; any other outcome raises (non-zero exit).

Runs on ``fleet.helper_python`` (the system ``python3``), which must be Python >= 3.10. Standard
library only; sent as ``<interpreter> -c <this text> <json>``, never imported on a host.
"""

import json
import pathlib
import socket
import subprocess
import sys

KEYS = ("hosts", "root")


def libtpu_holders():
    # Kept standalone for authenticated source-only SSH deployment. Mirror the
    # collector guard; neither this snapshot nor path absence is a fleet census.
    lock = pathlib.Path('/tmp/libtpu_lockfile')
    def identity():
        try:
            value = lock.lstat()
        except FileNotFoundError:
            return None
        return (value.st_dev, value.st_ino, value.st_mode)
    before = identity()
    result = subprocess.run(['sudo', '-n', 'env', 'LC_ALL=C', 'fuser', str(lock)],
        capture_output=True, text=True, timeout=15)
    after = identity()
    if before != after:
        raise RuntimeError('libtpu lock changed during observation')
    if before is None:
        if (result.returncode == 1 and not result.stdout.strip()
                and result.stderr.strip() == 'Specified filename /tmp/libtpu_lockfile does not exist.'):
            return []
        raise RuntimeError('cannot establish absent libtpu lock state')
    if (result.returncode not in (0, 1)
            or (result.returncode == 1 and result.stderr.strip())
            or (result.returncode == 0 and result.stderr.strip() != str(lock) + ':')):
        raise RuntimeError('cannot establish libtpu holder state')
    holders = sorted(int(p) for p in result.stdout.split())
    if bool(holders) != (result.returncode == 0) or any(p <= 0 for p in holders):
        raise RuntimeError('inconsistent fuser output')
    return holders


def arguments(argv, keys):
    if len(argv) != 1:
        raise SystemExit("expected exactly one JSON argument")
    value = json.loads(argv[0])
    if not isinstance(value, dict) or sorted(value) != sorted(keys):
        raise SystemExit("helper arguments differ")
    return value


def alive(owner):
    """Whether the process a start marker names still runs with the recorded start ticks."""
    try:
        fields = (pathlib.Path("/proc") / str(int(owner["pid"])) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == owner["start_ticks"] and fields[0] not in ("Z", "X")


def main(argv):
    if sys.version_info < (3, 10):
        sys.stderr.write("remote helpers need Python >= 3.10 (fleet.helper_python)\n")
        return 2
    args = arguments(argv, KEYS)
    root, hosts = pathlib.Path(args["root"]), args["hosts"]
    hostname = socket.gethostname()
    markers = []
    if hosts is not None:
        if (not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts)
                or len(set(hosts)) != len(hosts) or hostname not in hosts):
            raise RuntimeError("host is not in the authenticated fleet")
        markers.append(root / ("worker_started.rank%d.json" % hosts.index(hostname)))
    if libtpu_holders():
        raise RuntimeError("libtpu is owned")
    if root.is_dir():
        for marker in sorted(root.glob("worker_started.rank*.json")):
            if marker not in markers and json.loads(marker.read_text()).get("hostname") == hostname:
                markers.append(marker)
    for marker in markers:
        if marker.exists() and alive(json.loads(marker.read_text())):
            raise RuntimeError("request worker is live")
    print("IDLE " + hostname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
