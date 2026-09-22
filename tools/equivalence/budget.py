"""CPU budget rule on the TPU controller host (D25): read-only live-run detection.

``live_tpu_run()`` is true when any of these holds:

* a process whose ``/proc/<pid>/cmdline`` names the controller, worker or pack-worker module
  (181c013e names and the post-refactor names);
* ``/tmp/libtpu_lockfile`` is open by a visible process (scan of ``/proc/*/fd`` links);
* a workload lock inode of the site is present in ``/proc/locks``.

The two *sync* (rsync) locks are deliberately not indicators: a five-minute cron backup holds
them routinely, which would make every heavy gate refuse intermittently without any TPU run.
While a run is live, heavy gates (production fingerprints, CPU execution goldens, any CPU32
selection) refuse; light gates run under ``os.nice(19)`` with single-threaded Eigen.
Nothing here writes, signals or opens a device.
"""

from __future__ import annotations

import os
from pathlib import Path

MODULES = (
    b"scripts.release.launch_ws32_optimized_request",
    b"scripts.release.ws32_optimized_worker",
    b"scripts.release.ws32_pack_worker",
    b"glm_tpu.executor.multihost_executor",
    b"glm_tpu.worker.tpu_worker",
    b"glm_tpu.model_loader.pack_worker",
)
LIBTPU_LOCK = "/tmp/libtpu_lockfile"
REFUSAL = "a TPU run is live on this host; run heavy gates elsewhere or later"


def _workload_locks() -> list[str]:
    """Workload lock paths: the 181c013e launcher constants (first two entries are the workload
    leases; S1 moves them to the site file, and this function follows)."""
    try:
        from scripts.release import launch_ws32_optimized_request as launch
    except Exception:  # detection must never fail on an import problem
        return []
    return [str(path) for path in list(getattr(launch, "LOCKS", ()))[:2]]


def live_processes() -> list[int]:
    own = os.getpid()
    hits = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(module in argv for module in MODULES):
            hits.append(int(entry.name))
    return sorted(hits)


def libtpu_holders() -> list[int]:
    holders = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                if os.readlink(fd) == LIBTPU_LOCK:
                    holders.append(int(entry.name))
                    break
        except OSError:
            continue
    return sorted(holders)


def held_workload_locks() -> list[str]:
    try:
        table = Path("/proc/locks").read_text()
    except OSError:
        return []
    held = []
    for path in _workload_locks():
        try:
            inode = os.stat(path).st_ino
        except OSError:
            continue
        rows = [line.split() for line in table.splitlines()]
        if any(len(row) > 5 and row[5].rsplit(":", 1)[-1] == str(inode) for row in rows):
            held.append(path)
    return held


def live_tpu_run() -> bool:
    return bool(live_processes() or libtpu_holders() or held_workload_locks())


def light_mode() -> None:
    """Throttle a light gate while a run is live (never used to bypass a heavy-gate refusal)."""
    os.nice(19)
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()


def report() -> dict[str, object]:
    return dict(live=live_tpu_run(), processes=len(live_processes()), libtpu_holders=len(libtpu_holders()),
                workload_locks=len(held_workload_locks()))
