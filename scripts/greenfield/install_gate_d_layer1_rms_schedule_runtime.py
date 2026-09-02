#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Atomically install the reviewed layer-1 RMS schedule launcher and runtime."""

from __future__ import annotations

import ctypes
import json
import os
import stat
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

SOURCE_ROOT = Path("/opt/glm-tpu/gate-d-layer1-rms-schedule-install-v3")
INSTALLER_PATH = (
    SOURCE_ROOT / "install_gate_d_layer1_rms_schedule_runtime.py"
)
LAUNCHER_PARENT = Path("/opt/glm-tpu/bin")
LAUNCHER_TARGET = (
    LAUNCHER_PARENT / "launch_gate_d_layer1_rms_schedule_v3.py"
)
CAPSULE_PARENT = Path("/usr/local/libexec/glm-tpu")
CAPSULE_TARGET = CAPSULE_PARENT / "gate-d-layer1-rms-schedule-v3"
EXPECTED_ENVIRONMENT = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
PAYLOADS = {
    "run_gate_d_layer1_rms_schedule.py": (
        "5f5fa802652487a9c3c4ef070f70baa15050338e2799d20d060633587230f972"
    ),
    "launch_gate_d_layer1_rms_schedule.py": (
        "33c0a696e02fbf7fc11fd6e0c812ea79a0b37d9bae1a2faa252fdb1a7052899b"
    ),
    "publish_gate_d_layer1_rms_schedule.py": (
        "a53f3fc22d8a507994b470ce2453c521db218221c66aa02e3fd35a80ef15ea9c"
    ),
    "verify_gate_d_same_region_git_mirror.py": (
        "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b"
    ),
}
CAPSULE_NAMES = (
    "run_gate_d_layer1_rms_schedule.py",
    "publish_gate_d_layer1_rms_schedule.py",
    "verify_gate_d_same_region_git_mirror.py",
)
SOURCE_NAMES = tuple(sorted((*PAYLOADS, INSTALLER_PATH.name)))
_RENAME_NOREPLACE = 1


def _require_root_chain(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe root-owned path: {path}")
    for component in reversed((path, *path.parents)):
        metadata = os.stat(component, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"unsafe root-owned path component: {component}")


def _require_directory(path: Path, *, mode: int, uid: int = 0, gid: int = 0) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != uid
        or metadata.st_gid != gid
        or stat.S_IMODE(metadata.st_mode) != mode
        or os.listxattr(path, follow_symlinks=False)
    ):
        raise RuntimeError(f"unsafe immutable directory: {path}")


def _read_regular(
    path: Path,
    *,
    expected_sha256: str | None,
    mode: int,
    uid: int = 0,
    gid: int = 0,
) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != mode
            or os.listxattr(path, follow_symlinks=False)
        ):
            raise RuntimeError(f"unsafe immutable file: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
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
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
            or (
                expected_sha256 is not None
                and sha256(raw).hexdigest() != expected_sha256
            )
        ):
            raise RuntimeError(f"immutable file identity drifted: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _rename_noreplace(
    source: str, target: str, *, source_dir_fd: int, target_dir_fd: int
) -> None:
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
            source_dir_fd,
            os.fsencode(source),
            target_dir_fd,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("short immutable payload write")
        view = view[written:]


def _create_file(
    parent_fd: int,
    name: str,
    raw: bytes,
    *,
    uid: int = 0,
    gid: int = 0,
) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o500,
        dir_fd=parent_fd,
    )
    try:
        _write_all(descriptor, raw)
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def _unlink_owned_file(parent_fd: int, name: str, identity: tuple[int, int]) -> None:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise RuntimeError(f"refusing cleanup of unowned staging file: {name}")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _remove_owned_capsule_staging(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
    allowed_children: frozenset[str],
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        held = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (held.st_dev, held.st_ino) != identity
            or (named.st_dev, named.st_ino) != identity
        ):
            raise RuntimeError(f"refusing cleanup of unowned staging directory: {name}")
        children = os.listdir(descriptor)
        if len(children) != len(set(children)) or not set(children) <= allowed_children:
            raise RuntimeError(f"refusing cleanup of foreign staging contents: {name}")
        for child in children:
            metadata = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"refusing cleanup of unsafe staging child: {child}")
            os.unlink(child, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != identity:
        raise RuntimeError(f"staging directory changed before cleanup: {name}")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _verify_capsule(
    target: Path,
    payloads: Mapping[str, bytes],
    *,
    uid: int = 0,
    gid: int = 0,
) -> None:
    _require_directory(target, mode=0o555, uid=uid, gid=gid)
    if tuple(sorted(item.name for item in target.iterdir())) != tuple(sorted(payloads)):
        raise RuntimeError("immutable runtime capsule membership drifted")
    for name, raw in payloads.items():
        observed = _read_regular(
            target / name,
            expected_sha256=sha256(raw).hexdigest(),
            mode=0o555,
            uid=uid,
            gid=gid,
        )
        if observed != raw:
            raise RuntimeError(f"immutable runtime payload drifted: {name}")


def _publish_capsule(
    parent: Path,
    target: Path,
    payloads: Mapping[str, bytes],
    *,
    uid: int = 0,
    gid: int = 0,
) -> None:
    _require_directory(parent, mode=0o755, uid=uid, gid=gid)
    if target.exists() or target.is_symlink():
        _verify_capsule(target, payloads, uid=uid, gid=gid)
        return
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    staging_name = f".{target.name}.tmp-{os.getpid()}"
    staging_identity: tuple[int, int] | None = None
    published = False
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        created = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        staging_identity = (created.st_dev, created.st_ino)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            for name, raw in sorted(payloads.items()):
                _create_file(staging_fd, name, raw, uid=uid, gid=gid)
            os.fchown(staging_fd, uid, gid)
            os.fchmod(staging_fd, 0o555)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        _rename_noreplace(
            staging_name,
            target.name,
            source_dir_fd=parent_fd,
            target_dir_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
    except BaseException:
        if staging_identity is not None and not published:
            _remove_owned_capsule_staging(
                parent_fd,
                staging_name,
                staging_identity,
                frozenset(payloads),
            )
        raise
    finally:
        os.close(parent_fd)
    _verify_capsule(target, payloads, uid=uid, gid=gid)


def _publish_launcher(
    parent: Path,
    target: Path,
    raw: bytes,
    *,
    uid: int = 0,
    gid: int = 0,
) -> None:
    _require_directory(parent, mode=0o755, uid=uid, gid=gid)
    expected = sha256(raw).hexdigest()
    if target.exists() or target.is_symlink():
        if (
            _read_regular(
                target,
                expected_sha256=expected,
                mode=0o555,
                uid=uid,
                gid=gid,
            )
            != raw
        ):
            raise RuntimeError("immutable launcher payload drifted")
        return
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    staging_name = f".{target.name}.tmp-{os.getpid()}"
    staging_identity: tuple[int, int] | None = None
    published = False
    try:
        staging_identity = _create_file(parent_fd, staging_name, raw, uid=uid, gid=gid)
        _rename_noreplace(
            staging_name,
            target.name,
            source_dir_fd=parent_fd,
            target_dir_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
    except BaseException:
        if staging_identity is not None and not published:
            _unlink_owned_file(parent_fd, staging_name, staging_identity)
        raise
    finally:
        os.close(parent_fd)
    observed = _read_regular(
        target,
        expected_sha256=expected,
        mode=0o555,
        uid=uid,
        gid=gid,
    )
    if observed != raw:
        raise RuntimeError("installed launcher payload drifted")


def _load_payloads() -> tuple[bytes, dict[str, bytes]]:
    _require_root_chain(SOURCE_ROOT)
    _require_directory(SOURCE_ROOT, mode=0o755)
    if tuple(sorted(item.name for item in SOURCE_ROOT.iterdir())) != SOURCE_NAMES:
        raise RuntimeError("root-owned installation source membership drifted")
    installer_raw = _read_regular(
        INSTALLER_PATH,
        expected_sha256=None,
        mode=0o555,
    )
    payloads = {
        name: _read_regular(
            SOURCE_ROOT / name,
            expected_sha256=expected_sha256,
            mode=0o555,
        )
        for name, expected_sha256 in PAYLOADS.items()
    }
    return installer_raw, payloads


def main() -> int:
    if (
        os.geteuid() != 0
        or Path(__file__) != INSTALLER_PATH
        or sys.executable != "/usr/bin/python3"
        or sys.argv != [str(INSTALLER_PATH)]
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or not sys.flags.dont_write_bytecode
        or dict(os.environ) != EXPECTED_ENVIRONMENT
    ):
        raise RuntimeError("invoke only the exact root-owned isolated installer")
    _require_root_chain(LAUNCHER_PARENT)
    _require_root_chain(CAPSULE_PARENT)
    installer_raw, payloads = _load_payloads()
    capsule_payloads = {name: payloads[name] for name in CAPSULE_NAMES}
    _publish_capsule(CAPSULE_PARENT, CAPSULE_TARGET, capsule_payloads)
    _publish_launcher(
        LAUNCHER_PARENT,
        LAUNCHER_TARGET,
        payloads["launch_gate_d_layer1_rms_schedule.py"],
    )
    report = {
        "artifact_kind": "gate_d_layer1_rms_schedule_runtime_installed",
        "installer_sha256": sha256(installer_raw).hexdigest(),
        "capsule": str(CAPSULE_TARGET),
        "capsule_payload_sha256s": {
            name: sha256(raw).hexdigest() for name, raw in capsule_payloads.items()
        },
        "launcher": str(LAUNCHER_TARGET),
        "launcher_sha256": PAYLOADS[
            "launch_gate_d_layer1_rms_schedule.py"
        ],
        "launcher_invoked": False,
    }
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
