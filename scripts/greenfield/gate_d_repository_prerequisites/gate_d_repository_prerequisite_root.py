#!/usr/bin/python3
"""Root-only, no-replace primitives for Gate-D repository prerequisites.

The reviewed root-owned controller transports its retained, committed source
through ``python -c`` under an exact
``sudo env -i /usr/bin/python3 -I -S -B`` boundary.
It creates only immutable lock files or the exact recovery controller.  It
never opens a repository, imports JAX, or touches TPU state.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

OPT = Path("/opt")
GLM_ROOT = Path("/opt/glm-tpu")
LOCK_ROOT = Path("/opt/glm-tpu/locks")
BIN_ROOT = Path("/opt/glm-tpu/bin")
CONTROLLER_NAME = "recover_gate_d_worker_repositories.py"
CONTROLLER_SHA256 = "fd8950de504df483a3d86420b489406dbf93c3f68de04faf230bc85f402b2225"
CONTROLLER_BYTES = 44013
ALLOWED_LOCKS = {
    "gate_d_repo_recovery.lock",
    "glm_pod_workload.lock",
    "glm_tpu_rsync.lock",
}
_RENAME_NOREPLACE = 1
_CARRIER_PATTERN = re.compile(r"gate_d_repo_prerequisite_[0-9]{8}T[0-9]{15}Z")


class PrerequisiteError(RuntimeError):
    pass


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _validate_directory(fd: int, path: Path, *, uid: int, gid: int) -> None:
    value = os.fstat(fd)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != uid
        or value.st_gid != gid
        or stat.S_IMODE(value.st_mode) != 0o755
        or os.listxattr(fd)
    ):
        raise PrerequisiteError(f"unsafe exact directory: {path}")


def _open_directory(path: Path, *, uid: int, gid: int) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _validate_directory(fd, path, uid=uid, gid=gid)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_or_create_directory(
    parent_fd: int,
    parent_path: Path,
    name: str,
    *,
    uid: int,
    gid: int,
) -> int:
    created = False
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    fd = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        if created:
            os.fchown(fd, uid, gid)
            os.fchmod(fd, 0o755)
            os.fsync(parent_fd)
        _validate_directory(fd, parent_path / name, uid=uid, gid=gid)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _validate_lock(
    parent_fd: int, name: str, fd: int, *, uid: int, gid: int
) -> dict[str, int | str]:
    held = os.fstat(fd)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or held.st_uid != uid
        or held.st_gid != gid
        or stat.S_IMODE(held.st_mode) != 0o666
        or held.st_size != 0
        or os.listxattr(fd)
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise PrerequisiteError(f"unsafe lock identity: {name}")
    return {
        "dev": held.st_dev,
        "gid": held.st_gid,
        "ino": held.st_ino,
        "mode": f"{stat.S_IMODE(held.st_mode):04o}",
        "name": name,
        "nlink": held.st_nlink,
        "size": held.st_size,
        "uid": held.st_uid,
    }


def ensure_locks(
    opt: Path,
    glm_root: Path,
    lock_root: Path,
    names: tuple[str, ...],
    *,
    uid: int,
    gid: int,
) -> list[dict[str, int | str]]:
    if not names or set(names) - ALLOWED_LOCKS or len(names) != len(set(names)):
        raise PrerequisiteError("unsafe lock-name set")
    opt_fd = _open_directory(opt, uid=uid, gid=gid)
    try:
        glm_fd = _open_or_create_directory(opt_fd, opt, glm_root.name, uid=uid, gid=gid)
    finally:
        os.close(opt_fd)
    try:
        lock_fd = _open_or_create_directory(
            glm_fd, glm_root, lock_root.name, uid=uid, gid=gid
        )
    finally:
        os.close(glm_fd)
    records: list[dict[str, int | str]] = []
    try:
        for name in names:
            descriptor = -1
            created = False
            try:
                try:
                    descriptor = os.open(
                        name,
                        os.O_RDWR
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        0o666,
                        dir_fd=lock_fd,
                    )
                    created = True
                except FileExistsError:
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=lock_fd,
                    )
                if created:
                    os.fchown(descriptor, uid, gid)
                    os.fchmod(descriptor, 0o666)
                    os.fsync(descriptor)
                    os.fsync(lock_fd)
                records.append(
                    _validate_lock(lock_fd, name, descriptor, uid=uid, gid=gid)
                )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    finally:
        os.close(lock_fd)
    return records


def _rename_noreplace(parent_fd: int, source: str, target: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.renameat2
    except AttributeError as error:
        raise PrerequisiteError("renameat2 unavailable") from error
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), target)


def _validate_controller(
    parent_fd: int,
    name: str,
    fd: int,
    *,
    uid: int,
    gid: int,
) -> dict[str, int | str]:
    held = os.fstat(fd)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    raw = _read_fd(fd)
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or held.st_uid != uid
        or held.st_gid != gid
        or stat.S_IMODE(held.st_mode) != 0o555
        or held.st_size != CONTROLLER_BYTES
        or digest != CONTROLLER_SHA256
        or os.listxattr(fd)
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise PrerequisiteError("installed controller identity mismatch")
    return {
        "bytes": held.st_size,
        "dev": held.st_dev,
        "gid": held.st_gid,
        "ino": held.st_ino,
        "mode": f"{stat.S_IMODE(held.st_mode):04o}",
        "name": name,
        "sha256": digest,
        "uid": held.st_uid,
    }


def install_controller(
    bin_root: Path,
    raw: bytes,
    *,
    uid: int,
    gid: int,
) -> dict[str, int | str]:
    if (
        len(raw) != CONTROLLER_BYTES
        or hashlib.sha256(raw).hexdigest() != CONTROLLER_SHA256
    ):
        raise PrerequisiteError("controller input identity mismatch")
    parent_fd = _open_directory(bin_root, uid=uid, gid=gid)
    try:
        try:
            target_fd = os.open(
                CONTROLLER_NAME,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            target_fd = -1
        if target_fd >= 0:
            try:
                return _validate_controller(
                    parent_fd,
                    CONTROLLER_NAME,
                    target_fd,
                    uid=uid,
                    gid=gid,
                )
            finally:
                os.close(target_fd)

        staging = f".{CONTROLLER_NAME}.{CONTROLLER_SHA256[:16]}.tmp"
        staging_fd = os.open(
            staging,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o555,
            dir_fd=parent_fd,
        )
        published = False
        try:
            view = memoryview(raw)
            while view:
                written = os.write(staging_fd, view)
                view = view[written:]
            os.fchown(staging_fd, uid, gid)
            os.fchmod(staging_fd, 0o555)
            os.fsync(staging_fd)
            _validate_controller(
                parent_fd,
                staging,
                staging_fd,
                uid=uid,
                gid=gid,
            )
            _rename_noreplace(parent_fd, staging, CONTROLLER_NAME)
            published = True
            os.fsync(parent_fd)
        finally:
            os.close(staging_fd)
            if not published:
                try:
                    os.unlink(staging, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        target_fd = os.open(
            CONTROLLER_NAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            return _validate_controller(
                parent_fd,
                CONTROLLER_NAME,
                target_fd,
                uid=uid,
                gid=gid,
            )
        finally:
            os.close(target_fd)
    finally:
        os.close(parent_fd)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    locks = subparsers.add_parser("locks")
    locks.add_argument("--lock", action="append", required=True)
    subparsers.add_parser("install-controller")
    return parser.parse_args()


def _validate_privileged_boundary() -> None:
    if (
        os.geteuid() != 0
        or sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or sys.dont_write_bytecode is not True
        or set(os.environ)
        != {"GLM_GATE_D_PREREQUISITE_CARRIER", "LANG", "LC_ALL", "PATH"}
        or os.environ["LANG"] != "C"
        or os.environ["LC_ALL"] != "C"
        or os.environ["PATH"] != "/usr/bin:/bin"
        or _CARRIER_PATTERN.fullmatch(os.environ["GLM_GATE_D_PREREQUISITE_CARRIER"])
        is None
    ):
        raise PrerequisiteError("invalid privileged execution boundary")


def main() -> int:
    _validate_privileged_boundary()
    arguments = _arguments()
    if arguments.action == "locks":
        records = ensure_locks(
            OPT,
            GLM_ROOT,
            LOCK_ROOT,
            tuple(arguments.lock),
            uid=0,
            gid=0,
        )
        output: dict[str, object] = {
            "host": os.uname().nodename,
            "locks": records,
            "marker": "PREREQUISITE_LOCKS_OK",
        }
    else:
        raw = sys.stdin.buffer.read()
        record = install_controller(
            BIN_ROOT,
            raw,
            uid=0,
            gid=0,
        )
        output = {
            "controller": record,
            "host": os.uname().nodename,
            "marker": "PREREQUISITE_CONTROLLER_OK",
        }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
