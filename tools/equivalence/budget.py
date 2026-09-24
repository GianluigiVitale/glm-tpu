"""CPU budget rule on the TPU controller host (D25): read-only live-run detection.

``live_tpu_run()`` is true when any of these holds:

* a process whose ``/proc/<pid>/cmdline`` names the controller, worker or pack-worker module
  (181c013e names and the post-refactor names);
* ``/tmp/libtpu_lockfile`` is open by a visible process (scan of ``/proc/*/fd`` links);
* a workload lock inode of the site is present in ``/proc/locks``;
* the site file exists but cannot be loaded (invalid, wrong owner or mode, unreadable): the
  workload locks are then unknown, so detection is *indeterminate* and counts as live. No site
  file at all means no workload locks on this host (process and libtpu checks only).

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
INDETERMINATE = (
    "live-run detection is indeterminate: the site file exists but cannot be loaded, so its "
    "workload locks are unknown; fix the site file or run heavy gates elsewhere"
)


def _workload_locks() -> list[str] | None:
    """Workload lock paths: the site file's ``locks.workload`` (S1a moved them there from the
    181c013e launcher constants). No site file on this host: none. A site file that exists but
    cannot be loaded, or a site location that cannot be resolved: None (indeterminate)."""
    try:
        from glm_tpu import envs

        location = envs.GLM_TPU_SITE_CONFIG
    except Exception:  # e.g. a relative GLM_TPU_SITE_CONFIG / GLM_TPU_CONFIG_ROOT
        return None
    if not os.path.lexists(location):
        return []
    try:
        from glm_tpu.config.site import SiteConfig

        return [str(path) for path in SiteConfig.load(location).locks.workload]
    except Exception:  # invalid, unsafe or unreadable: the locks are unknown
        return None


def indeterminate() -> bool:
    return _workload_locks() is None


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
    for path in _workload_locks() or []:
        try:
            inode = os.stat(path).st_ino
        except OSError:
            continue
        rows = [line.split() for line in table.splitlines()]
        if any(len(row) > 5 and row[5].rsplit(":", 1)[-1] == str(inode) for row in rows):
            held.append(path)
    return held


def live_tpu_run() -> bool:
    """True while a run is live -- or when that cannot be established (:func:`indeterminate`)."""
    return bool(indeterminate() or live_processes() or libtpu_holders() or held_workload_locks())


def refusal_reason() -> str:
    """Why heavy gates refuse right now (only meaningful while :func:`live_tpu_run` is true)."""
    return INDETERMINATE if indeterminate() else REFUSAL


def light_mode() -> None:
    """Throttle a light gate while a run is live (never used to bypass a heavy-gate refusal)."""
    os.nice(19)
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()


def report() -> dict[str, object]:
    return dict(
        live=live_tpu_run(),
        indeterminate=indeterminate(),
        processes=len(live_processes()),
        libtpu_holders=len(libtpu_holders()),
        workload_locks=len(held_workload_locks()),
    )
