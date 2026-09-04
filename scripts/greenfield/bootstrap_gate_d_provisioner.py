#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Publish the Gate-D provisioner from sealed, exact committed bytes.

The unprivileged entry point is itself executed from a sealed memfd.  It reads
the provisioner from an exact Git object, seals those bytes in a second memfd,
and asks the root entry point in this same sealed source image to publish them
without replacing an existing pathname.
"""

from __future__ import annotations

import ctypes
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
PROVISIONER_REPO_PATH = "scripts/greenfield/provision_gate_d_runtime_archive.py"
PROVISIONER_TARGET = Path("/opt/glm-tpu/bin/provision_gate_d_runtime_archive.py")
PROVISIONER_SHA256 = (
    "2582e9b6c91fe11a8489a2359f830ab786b5856463ea896172da1f7ab79b4319"
)
DIRECTORIES = (
    (Path("/opt/glm-tpu"), "bin"),
    (Path("/usr/local"), "libexec"),
    (Path("/usr/local/libexec"), "glm-tpu"),
)
UNPRIVILEGED_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
ROOT_ENVIRONMENT = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_REQUIRED_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 0x0001)
    | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
    | getattr(fcntl, "F_SEAL_GROW", 0x0004)
    | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
)
_RENAME_NOREPLACE = 1
_PROC_FD = re.compile(r"/proc/(?:self|[1-9][0-9]*)/fd/([0-9]+)\Z")


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("short sealed payload write")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while block := os.read(descriptor, 1024 * 1024):
        chunks.append(block)
    return b"".join(chunks)


def _sealed_memfd(name: str, raw: bytes) -> int:
    descriptor = os.memfd_create(
        name, os.MFD_CLOEXEC | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    )
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o400)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if _read_all(descriptor) != raw:
            raise RuntimeError("sealed payload readback drifted")
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _REQUIRED_SEALS)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) != _REQUIRED_SEALS:
            raise RuntimeError("sealed payload lacks the exact seal set")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_sealed_path(path: Path, expected_sha256: str) -> bytes:
    if _PROC_FD.fullmatch(str(path)) is None:
        raise RuntimeError(f"source is not an exact proc descriptor path: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 0
            or fcntl.fcntl(descriptor, _F_GET_SEALS) != _REQUIRED_SEALS
            or os.listxattr(descriptor)
        ):
            raise RuntimeError("source descriptor is not an immutable sealed memfd")
        raw = _read_all(descriptor)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            identity(before) != identity(after)
            or len(raw) != before.st_size
            or sha256(raw).hexdigest() != expected_sha256
        ):
            raise RuntimeError("sealed source identity or digest drifted")
        return raw
    finally:
        os.close(descriptor)


def _require_root_chain(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe root path: {path}")
    for component in reversed((path, *path.parents)):
        metadata = os.stat(component, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"unsafe root path component: {component}")


def _require_directory_fd(descriptor: int, *, mode: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != mode
        or os.listxattr(descriptor)
    ):
        raise RuntimeError("unsafe root-owned directory descriptor")


def _ensure_root_directory(parent: Path, name: str) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise RuntimeError("unsafe directory basename")
    _require_root_chain(parent)
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        _require_directory_fd(parent_fd, mode=0o755)
        try:
            os.mkdir(name, 0o755, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        child_fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            _require_directory_fd(child_fd, mode=0o755)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            held = os.fstat(child_fd)
            if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
                raise RuntimeError("root directory identity drifted")
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def _rename_noreplace(source: str, target: str, *, parent_fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("atomic RENAME_NOREPLACE is unavailable") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), target)


def _read_installed(
    parent_fd: int,
    name: str,
    expected_sha256: str,
    *,
    owner_uid: int,
    owner_gid: int,
) -> bytes:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
    )
    try:
        before = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != owner_uid
            or before.st_gid != owner_gid
            or stat.S_IMODE(before.st_mode) != 0o555
            or os.listxattr(descriptor)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError("installed provisioner metadata drifted")
        raw = _read_all(descriptor)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_ctime_ns)
            or len(raw) != before.st_size
            or sha256(raw).hexdigest() != expected_sha256
        ):
            raise RuntimeError("installed provisioner identity or digest drifted")
        return raw
    finally:
        os.close(descriptor)


def _cleanup_owned_file(
    parent_fd: int, name: str, identity: tuple[int, int]
) -> None:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise RuntimeError("refusing to remove an unowned staging inode")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _publish_exact_file(
    parent_fd: int,
    target_name: str,
    raw: bytes,
    expected_sha256: str,
    *,
    owner_uid: int,
    owner_gid: int,
) -> None:
    if "/" in target_name or target_name in {"", ".", ".."}:
        raise RuntimeError("unsafe publication basename")
    if sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("publication payload digest drifted")
    try:
        installed = _read_installed(
            parent_fd,
            target_name,
            expected_sha256,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
    except FileNotFoundError:
        installed = None
    if installed is not None:
        if installed != raw:
            raise RuntimeError("existing provisioner bytes drifted")
        return

    staging = f".{target_name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    descriptor = os.open(
        staging,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o500,
        dir_fd=parent_fd,
    )
    created = os.fstat(descriptor)
    identity = (created.st_dev, created.st_ino)
    staging_present = True
    try:
        _write_all(descriptor, raw)
        os.fchown(descriptor, owner_uid, owner_gid)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        try:
            _rename_noreplace(staging, target_name, parent_fd=parent_fd)
            staging_present = False
            os.fsync(parent_fd)
        except FileExistsError:
            _cleanup_owned_file(parent_fd, staging, identity)
            staging_present = False
    except BaseException:
        if staging_present:
            _cleanup_owned_file(parent_fd, staging, identity)
        raise
    finally:
        os.close(descriptor)
    installed = _read_installed(
        parent_fd,
        target_name,
        expected_sha256,
        owner_uid=owner_uid,
        owner_gid=owner_gid,
    )
    if installed != raw:
        raise RuntimeError("published provisioner bytes drifted")


def _publish_provisioner(raw: bytes, expected_sha256: str) -> None:
    target = PROVISIONER_TARGET
    parent = target.parent
    _require_root_chain(parent)
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        _require_directory_fd(parent_fd, mode=0o755)
        _publish_exact_file(
            parent_fd,
            target.name,
            raw,
            expected_sha256,
            owner_uid=0,
            owner_gid=0,
        )
    finally:
        os.close(parent_fd)


def _git_blob(pin: str, path: str) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "/bin/false",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(WORKTREE),
            "show",
            f"{pin}:{path}",
        ],
        check=False,
        env=environment,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("could not read the exact committed provisioner blob")
    return result.stdout


def _root_main(expected_source_sha256: str, payload_path: Path) -> int:
    if (
        os.geteuid() != 0
        or sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or not sys.flags.dont_write_bytecode
        or dict(os.environ) != ROOT_ENVIRONMENT
    ):
        raise RuntimeError("root bootstrap requires the exact isolated environment")
    _read_sealed_path(Path(sys.argv[0]), expected_source_sha256)
    payload = _read_sealed_path(payload_path, PROVISIONER_SHA256)
    for parent, name in DIRECTORIES:
        _ensure_root_directory(parent, name)
    _publish_provisioner(payload, PROVISIONER_SHA256)
    print(
        json.dumps(
            {
                "artifact_kind": "gate_d_provisioner_bootstrapped",
                "provisioner_sha256": PROVISIONER_SHA256,
                "provisioner_target": str(PROVISIONER_TARGET),
                "source_descriptor_sealed": True,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


def _unprivileged_main(expected_source_sha256: str, pin: str) -> int:
    if (
        os.geteuid() == 0
        or sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or not sys.flags.dont_write_bytecode
        or dict(os.environ) != UNPRIVILEGED_ENVIRONMENT
        or re.fullmatch(r"[0-9a-f]{40}", pin) is None
    ):
        raise RuntimeError("bootstrap requires the exact unprivileged environment")
    source_path = Path(sys.argv[0])
    _read_sealed_path(source_path, expected_source_sha256)
    provisioner = _git_blob(pin, PROVISIONER_REPO_PATH)
    if sha256(provisioner).hexdigest() != PROVISIONER_SHA256:
        raise RuntimeError("committed provisioner digest drifted")
    payload_fd = _sealed_memfd("gate-d-provisioner-payload", provisioner)
    try:
        match = _PROC_FD.fullmatch(str(source_path))
        if match is None:
            raise RuntimeError("bootstrap source descriptor path drifted")
        source_fd = int(match.group(1))
        source_for_root = Path(f"/proc/{os.getpid()}/fd/{source_fd}")
        payload_for_root = Path(f"/proc/{os.getpid()}/fd/{payload_fd}")
        command = [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/env",
            "-i",
            *(f"{name}={value}" for name, value in ROOT_ENVIRONMENT.items()),
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            str(source_for_root),
            "--root-bootstrap",
            expected_source_sha256,
            str(payload_for_root),
        ]
        result = subprocess.run(
            command,
            check=False,
            env=UNPRIVILEGED_ENVIRONMENT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("root provisioner bootstrap failed")
    finally:
        os.close(payload_fd)
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--root-bootstrap":
        return _root_main(sys.argv[2], Path(sys.argv[3]))
    if len(sys.argv) == 4 and sys.argv[1] == "--bootstrap":
        return _unprivileged_main(sys.argv[2], sys.argv[3])
    raise RuntimeError("invalid Gate-D provisioner bootstrap invocation")


if __name__ == "__main__":
    raise SystemExit(main())
