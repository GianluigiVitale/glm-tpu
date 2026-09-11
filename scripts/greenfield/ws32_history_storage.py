"""Fixed controller-only write bounds for the untimed history diagnostic.

These limits do not replace the transport's 512 MiB published / 544 MiB local
rank limits or its 10 GiB whole-prefix limit. Controller categories total
469 MiB, below the 504 MiB usable portion of its 512 MiB envelope; the remaining
8 MiB is reserved for publication metadata. Failed writes retain their bounded
partial original at its final name. No writer replaces an existing original.
"""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sqlite3
import stat
import sys
import time
from typing import BinaryIO


MIB = 1 << 20
LIMITS = dict(database=256 * MIB, runner=32 * MIB, summary=33 * MIB,
              hlo=12 * MIB, sources=104 * MIB, control=32 * MIB)
CONTROLLER_BYTES = 512 * MIB
METADATA_RESERVE = 8 * MIB
CHUNK_BYTES = 64 << 10
DB_SQL_METADATA_LIMIT = 64 << 10
DB_GROWTH_RESERVE = MIB
FILES = {
    "results_ckpt.db": ("database", LIMITS["database"]),
    "runner.json": ("runner", LIMITS["runner"]),
    "summary.json": ("summary", LIMITS["summary"]),
    "hlo/candidate.optimized_hlo.txt": ("hlo", LIMITS["hlo"]),
    "source_tiles/manifest.json": ("sources", MIB),
    # Nonidentity topology makes producer ownership uneven (rank4 ~31.6 MiB).
    **{f"source_tiles/rank{i}.npz": ("sources", 34 * MIB) for i in range(8)},
    **{name: ("control", 2 * MIB) for name in (
        "fleet_sync.log", "retained_preflight.log", "coordinator.log",
        "fleet_launch.log", "archive_receipts.json", "failure_archive_receipts.json",
    )},
    **{f"{kind}_{phase}.txt": ("control", MIB)
       for kind in ("census", "devices") for phase in ("pre", "post", "failure_exit")},
    "runner.log": ("control", 16 * MIB),
    "orchestrator.log": ("control", 512 << 10),
    "failure_orchestrator.log": ("control", 512 << 10),
    "evidence.sha256": ("control", MIB),
    "results_ckpt.db-journal": ("control", MIB),
    "SUCCESS": ("control", 64 << 10),
}
APPEND = frozenset(("orchestrator.log", "evidence.sha256"))
_TAG = re.compile(r"greenfield_fp8_ws32_history_frontier_l06_[0-9]{8}T[0-9]+Z")


def _plain(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("history controller symlink refused")


def _target(path: Path) -> tuple[Path, str]:
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("history controller parent traversal refused")
    path = path.absolute()
    _plain(path)
    root = next((p for p in path.parents if _TAG.fullmatch(p.name)), None)
    if root is None or not root.is_dir():
        raise ValueError("history controller run root/tag differs")
    name = path.relative_to(root).as_posix()
    if name not in FILES:
        raise ValueError(f"history controller unregistered path: {name}")
    return root, name


@contextmanager
def _locked(root: Path):
    # The existing directory is the lock inode: no extra evidence/lock artifact.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _usage(root: Path) -> dict[str, int]:
    totals = dict.fromkeys(LIMITS, 0)
    for path in root.iterdir():
        if re.fullmatch(r"rank[0-7]", path.name) or path.name == "fleet":
            # Authenticated rank originals have their own transport bounds.
            continue
        if path.name == "references":
            from scripts.greenfield.ws32_history_transport import _reference_bytes

            _plain(path)
            for rank in path.iterdir():
                _plain(rank)
                if (not re.fullmatch(r"rank[0-7]", rank.name) or not rank.is_dir()
                        or any(p.name != "retained_reference" for p in rank.iterdir())):
                    raise ValueError("history controller retained-reference layout differs")
                _reference_bytes(rank)
            continue
        paths = path.rglob("*") if path.is_dir() else (path,)
        _plain(path)
        for item in paths:
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("history controller nonregular path refused")
            name = item.relative_to(root).as_posix()
            if name not in FILES:
                raise ValueError(f"history controller unregistered existing path: {name}")
            kind, cap = FILES[name]
            size = item.stat().st_size
            if size > cap:
                raise ValueError(f"history controller file budget exceeded: {name}")
            totals[kind] += size
    if any(totals[kind] > cap for kind, cap in LIMITS.items()):
        raise ValueError("history controller category budget exceeded")
    return totals


def _reserve(root: Path, name: str, additional: int) -> None:
    if type(additional) is not int or additional < 0:
        raise ValueError("history controller byte count differs")
    path = root / name
    _plain(path)
    kind, cap = FILES[name]
    size = path.stat().st_size if path.exists() else 0
    totals = _usage(root)
    if size + additional > cap or totals[kind] + additional > LIMITS[kind]:
        raise ValueError(f"history controller write budget exceeded: {name}")
    if sum(totals.values()) + additional > CONTROLLER_BYTES - METADATA_RESERVE:
        raise ValueError("history controller aggregate budget exceeded")


def write_bytes(path: Path, data: bytes) -> None:
    """Create one registered original, checking all bytes before its first write."""
    if not isinstance(data, bytes):
        raise TypeError("history controller writer requires bytes")
    root, name = _target(path)
    with _locked(root):
        _reserve(root, name, len(data))
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())


def write_json(path: Path, value) -> None:
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def run_child(action, *, timeout: float | None = None) -> None:
    """Run one existing local action; reap only its owned session on every exit."""
    if timeout is not None and (type(timeout) not in (int, float)
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("history child timeout differs")

    def terminated(signum, frame):
        raise InterruptedError("history child parent interrupted")

    previous = signal.signal(signal.SIGTERM, terminated)
    pid = None
    try:
        pid = os.fork()
        if pid == 0:
            try:
                os.setsid()
                action()
            except BaseException:
                os._exit(1)
            os._exit(0)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            # Keep the known child unreaped while signaling its own group, so
            # neither its PID nor group identity can be recycled underneath us.
            observed = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT |
                                 (os.WNOHANG if deadline is not None else 0))
            if observed is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("history bounded child timed out")
            time.sleep(0.02)
        if observed.si_code != os.CLD_EXITED or observed.si_status != 0:
            raise RuntimeError("history bounded child failed")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if pid:
            # Also remove unexpected surviving local descendants after success.
            # This never signals the controller's group or a remote process.
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Parent interruption may precede the child's setsid().
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.waitpid(pid, 0)
        signal.signal(signal.SIGTERM, previous)


def run_logged(action, output: Path) -> None:
    """Bound the unchanged SSH helper's direct file descriptor in a child.

    The four fixed RPC logs reserve their entire per-file allowance before the
    child can write. The directory lock keeps concurrent shell journals from
    consuming that reservation. No infrastructure command is constructed here.
    """
    root, name = _target(output)
    if name not in ("fleet_sync.log", "retained_preflight.log", "coordinator.log", "fleet_launch.log"):
        raise ValueError("history controller RPC log name differs")
    with _locked(root):
        if (root / name).exists():
            raise FileExistsError(root / name)
        _reserve(root, name, FILES[name][1])
        def limited():
            cap = FILES[name][1]
            resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
            action()

        try:
            run_child(limited)
        except Exception as error:
            raise RuntimeError(f"history bounded RPC failed; retained log: {name}") from error
        _usage(root)
        if not (root / name).is_file():
            raise RuntimeError(f"history bounded RPC failed; retained log: {name}")
        with (root / name).open("rb") as saved:
            os.fsync(saved.fileno())


def stream(root: Path, name: str, source: BinaryIO, *, append: bool = False,
           tee: BinaryIO | None = None) -> None:
    """Bound each chunk before writing; a refused chunk is never persisted."""
    actual, registered = _target(Path(root) / name)
    if actual != Path(root).absolute() or name != registered:
        raise ValueError("history controller stream root/name differs")
    if append and name not in APPEND:
        raise ValueError("history controller append is restricted to journals")
    target = actual / name
    with _locked(actual):
        _reserve(actual, name, 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        output = target.open("ab" if append else "xb", buffering=0)
    try:
        while chunk := source.read(CHUNK_BYTES):
            with _locked(actual):
                _reserve(actual, name, len(chunk))
                output.write(chunk)
            if tee is not None:
                tee.write(chunk)
                tee.flush()
    finally:
        os.fsync(output.fileno())
        output.close()


def database_budget(db_path: Path, *, raw_output: str | None = None,
                    sql_metadata: dict | None = None) -> dict:
    """Read-only prospective check, before launch or any provenance INSERT.

    The launch reserves twice the 32 MiB aggregate allowance plus 1 MiB for
    SQLite row/index/schema growth. Accounting repeats the check with its exact
    single serialization and <=64 KiB of fixed SQL metadata. This conservative
    projection does not cover unrelated concurrent database writers; the later
    pinned-snapshot hard cap remains independently enforced.
    """
    if raw_output is None:
        if sql_metadata is not None:
            raise ValueError("history launch database metadata differs")
        raw_size, metadata_size = LIMITS["runner"], 0
    else:
        if not isinstance(raw_output, str) or not isinstance(sql_metadata, dict):
            raise ValueError("history accounting serialization/metadata differs")
        raw_size = len(raw_output.encode())
        metadata_size = len(json.dumps(sql_metadata, sort_keys=True, allow_nan=False).encode())
        if raw_size > LIMITS["runner"] or metadata_size > DB_SQL_METADATA_LIMIT:
            raise ValueError("history accounting serialization budget exceeded before INSERT")
    path = Path(db_path).absolute()
    _plain(path)
    page_size, page_count = 4096, 0
    # The protected shell separately requires its canonical DB to exist. An
    # absent test-local DB can safely be budgeted before provenance creates it.
    if path.exists():
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
            source.execute("BEGIN")
            source.execute("SELECT count(*) FROM sqlite_master").fetchone()
            page_size = source.execute("PRAGMA page_size").fetchone()[0]
            page_count = source.execute("PRAGMA page_count").fetchone()[0]
            source.rollback()
    current = page_size * page_count
    projected = current + 2 * raw_size + DB_GROWTH_RESERVE
    cap = min(LIMITS["database"], FILES["results_ckpt.db"][1])
    if projected > cap:
        raise ValueError("history database projection exceeds snapshot budget before INSERT/launch")
    return dict(schema="history_controller_db_projection_v1",
                phase="prelaunch_maximum" if raw_output is None else "preinsert_actual",
                source_page_size=page_size, source_page_count=page_count,
                source_bytes=current, raw_output_bytes=raw_size,
                sql_metadata_bytes=metadata_size, payload_growth_allowance=2 * raw_size,
                sqlite_growth_reserve=DB_GROWTH_RESERVE,
                projected_bytes=projected, snapshot_cap_bytes=cap,
                concurrent_growth_included=False)


def snapshot(db_path: Path, dest: Path) -> None:
    """Copy a pinned SQLite read snapshot, bounded before destination creation.

    Holding the source read transaction prevents source growth from expanding
    this backup, including WAL writers. Progress and max_page_count are additional
    checks, not a substitute for pinning the source snapshot before counting it.
    """
    root, name = _target(dest)
    if name != "results_ckpt.db":
        raise ValueError("history controller database destination differs")
    source_path = Path(db_path).absolute()
    _plain(source_path)
    with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as source:
        source.execute("BEGIN")
        source.execute("SELECT count(*) FROM sqlite_master").fetchone()
        page_size = source.execute("PRAGMA page_size").fetchone()[0]
        page_count = source.execute("PRAGMA page_count").fetchone()[0]
        expected = page_size * page_count
        with _locked(root):
            _reserve(root, name, expected)
            target = root / name
            with target.open("xb"):
                pass
            try:
                with closing(sqlite3.connect(target)) as output:
                    output.execute(f"PRAGMA page_size={page_size}")
                    output.execute(f"PRAGMA max_page_count={FILES[name][1] // page_size}")

                    def progress(status, remaining, total):
                        if total != page_count or target.stat().st_size > expected:
                            raise ValueError("history controller database snapshot grew")

                    source.backup(output, pages=128, progress=progress)
                    if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("history controller database snapshot integrity failed")
            finally:
                # A failure keeps the bounded original for exact-name recovery.
                if target.stat().st_size > expected:
                    raise ValueError("history controller database snapshot exceeded pinned size")
        source.rollback()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()
    stream(args.root, args.name, sys.stdin.buffer, append=args.append,
           tee=sys.stdout.buffer if args.tee else None)


if __name__ == "__main__":
    main()
