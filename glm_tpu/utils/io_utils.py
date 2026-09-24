"""Owner-only, create-once file writes for run directories (DESIGN 6.8, H1).

:func:`create_private_exclusive` creates a file ``0600`` with ``O_CREAT | O_EXCL | O_NOFOLLOW``:
it never replaces, truncates or follows anything that already exists.

:func:`write_collected` is how the controller stores a record fetched from a host. A resident run
already holds the sequence-0 records (``runner.rank{r}.json``) when the final collection fetches
the same names again; the normal path fetches identical bytes, but a worker that failed after
sequence 0 rewrote its record with ``failure_type``. So the function never overwrites and never
raises on a difference: an equal record (same bytes, or the same JSON value) is skipped, and a
different one is preserved next to it as ``final/<name>`` and reported in ``divergent``.

Standard library only.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import stat


FINAL_DIR = "final"


def create_private_exclusive(path: Path, payload: bytes) -> None:
    """Create ``path`` with mode ``0600`` and write ``payload`` (fsync-ed). Raises
    ``FileExistsError`` if anything -- file, directory or symlink -- already exists there."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _same_record(path: Path, payload: bytes) -> bool:
    """Whether ``path`` is a regular file (not a symlink) holding ``payload``'s bytes or JSON value."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return False
        existing = path.read_bytes()
    except FileNotFoundError:
        return False
    if existing == payload:
        return True
    try:
        return json.loads(existing) == json.loads(payload)
    except ValueError:  # not JSON (UnicodeDecodeError is a ValueError): only byte equality counts
        return False


def write_collected(root: Path, name: str, payload: bytes, divergent: list[str]) -> None:
    """Write a record fetched from a host exactly once; never overwrite, never raise on difference.

    ``root/name`` absent: created (:func:`create_private_exclusive`). Present and equal (bytes or
    JSON value): nothing to do -- the resident receipt already holds it. Present and different
    (or not a regular file): ``root/name`` is left untouched, the fetched bytes are kept as
    ``root/final/<name>`` (``final/<name>.<n>`` if that name is taken by other content) and
    ``name`` is appended to ``divergent``. I/O errors (a full disk, permissions) still raise."""
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise ValueError(f"collected record name {name!r} must be a plain file name")
    root = Path(root)
    target = root / name
    try:
        create_private_exclusive(target, payload)
        return
    except FileExistsError:
        pass
    if _same_record(target, payload):
        return
    final = root / FINAL_DIR
    final.mkdir(mode=0o700, exist_ok=True)
    for index in itertools.count():
        candidate = final / (name if index == 0 else f"{name}.{index}")
        try:
            create_private_exclusive(candidate, payload)
            break
        except FileExistsError:
            if _same_record(candidate, payload):
                break
    if name not in divergent:
        divergent.append(name)


def _plain(path: Path) -> Path:
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("request paths must not traverse symlinks")
    return path


def read_bounded(path: Path, cap: int) -> bytes:
    path = _plain(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        facts = os.fstat(stream.fileno())
        if not stat.S_ISREG(facts.st_mode) or not 0 < facts.st_size <= cap:
            raise ValueError("request input is not a bounded regular file")
        raw = stream.read(cap + 1)
    if not 0 < len(raw) <= cap:
        raise ValueError("request input changed beyond its byte cap")
    return raw


def atomic(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        os.chmod(tmp, 0o600)
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# Site values (run root, model path, fleet naming) come from the run's staged,
# controller-resolved site.json (--site-sha256); workers never read $HOME config.


def persist(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def private(path):
    _plain(path)
    info = path.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("optimized input namespace must be owner-only")
