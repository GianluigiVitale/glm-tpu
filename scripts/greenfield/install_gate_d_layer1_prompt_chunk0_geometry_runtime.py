#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Atomically install the reviewed layer-1 prompt chunk-0 probe capsule."""

from __future__ import annotations

import ctypes
import json
import os
import stat
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

SOURCE_ROOT = Path("/opt/glm-tpu/gate-d-layer1-prompt-chunk0-geometry-install-v8")
INSTALLER_PATH = SOURCE_ROOT / "install_gate_d_layer1_prompt_chunk0_geometry_runtime.py"
LAUNCHER_PARENT = Path("/opt/glm-tpu/bin")
LAUNCHER_TARGET = LAUNCHER_PARENT / "launch_gate_d_layer1_prompt_chunk0_geometry_v8.py"
CAPSULE_PARENT = Path("/usr/local/libexec/glm-tpu")
CAPSULE_TARGET = CAPSULE_PARENT / "gate-d-layer1-prompt-chunk0-geometry-v7"
EXPECTED_ENVIRONMENT = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
PAYLOADS = {
    "chunk0_embedding_hlo.py": "e239c20b1a206061c9116726421343d81d5ff989be5e8f8440d5c59106eb9757",
    "original_db518_normalized_boundary_hlo.py": "35757aab4a616a2f1073f78503c29075c3e43cb684e4e1faf7d835630876809e",
    "chunk0_real_layer_consumer_hlo.py": "4ef7bbb0dbd74e5317cc653e67ef72dc5fd6ea472b9e486fc99c9ed410422aec",
    "probe_layer1_prompt_chunk0_geometry.py": "4eb5ee0c11530f3e2ed7be4f25de92f729d39694ae50d531fd57968e948e691c",
    "publish_gate_d_layer1_prompt_chunk0_geometry.py": "09808a13a2c5d19538abf6da59d4225cde149637f5a404a5c00052ef4d2f84e8",
    "launch_gate_d_layer1_prompt_chunk0_geometry.py": "6ba25afc0597fdfaa932317e171189ca5f82b80159ed7e77cad04bef6f31b24f",
    "verify_gate_d_rewrite_same_region_git_mirror.py": (
        "0f61a5a6108b1d931cc19ce247e1d88b7db76ed32029260bd6273cb4d4ca8d4e"
    ),
}
CAPSULE_NAMES = (
    "chunk0_embedding_hlo.py",
    "original_db518_normalized_boundary_hlo.py",
    "chunk0_real_layer_consumer_hlo.py",
    "probe_layer1_prompt_chunk0_geometry.py",
    "publish_gate_d_layer1_prompt_chunk0_geometry.py",
    "verify_gate_d_rewrite_same_region_git_mirror.py",
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


def _require_directory(path: Path, *, mode: int) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != mode
        or os.listxattr(path, follow_symlinks=False)
    ):
        raise RuntimeError(f"unsafe immutable directory: {path}")


def _read_regular(path: Path, *, expected_sha256: str | None, mode: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode) != mode
            or os.listxattr(path, follow_symlinks=False)
        ):
            raise RuntimeError(f"unsafe immutable file: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
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
        return raw
    finally:
        os.close(descriptor)


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


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("short immutable payload write")
        view = view[written:]


def _create_file(parent_fd: int, name: str, raw: bytes) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o500,
        dir_fd=parent_fd,
    )
    try:
        _write_all(descriptor, raw)
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def _cleanup_file(parent_fd: int, name: str, identity: tuple[int, int]) -> None:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise RuntimeError("refusing cleanup of unowned staging file")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _cleanup_directory(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
    allowed: frozenset[str],
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
            or not set(os.listdir(descriptor)) <= allowed
        ):
            raise RuntimeError("refusing cleanup of unowned staging directory")
        for child in os.listdir(descriptor):
            metadata = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError("refusing cleanup of unsafe staging child")
            os.unlink(child, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != identity:
        raise RuntimeError("staging directory changed before cleanup")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _verify_capsule(target: Path, payloads: Mapping[str, bytes]) -> None:
    _require_directory(target, mode=0o555)
    if tuple(sorted(item.name for item in target.iterdir())) != tuple(sorted(payloads)):
        raise RuntimeError("immutable capsule membership drifted")
    for name, raw in payloads.items():
        if (
            _read_regular(
                target / name, expected_sha256=sha256(raw).hexdigest(), mode=0o555
            )
            != raw
        ):
            raise RuntimeError(f"immutable capsule payload drifted: {name}")


def _publish_capsule(parent: Path, target: Path, payloads: Mapping[str, bytes]) -> None:
    _require_directory(parent, mode=0o755)
    if target.exists() or target.is_symlink():
        _verify_capsule(target, payloads)
        return
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    staging = f".{target.name}.tmp-{os.getpid()}"
    identity: tuple[int, int] | None = None
    published = False
    try:
        os.mkdir(staging, 0o700, dir_fd=parent_fd)
        created = os.stat(staging, dir_fd=parent_fd, follow_symlinks=False)
        identity = (created.st_dev, created.st_ino)
        staging_fd = os.open(
            staging,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            for name, raw in sorted(payloads.items()):
                _create_file(staging_fd, name, raw)
            os.fchown(staging_fd, 0, 0)
            os.fchmod(staging_fd, 0o555)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        _rename_noreplace(staging, target.name, parent_fd=parent_fd)
        published = True
        os.fsync(parent_fd)
    except BaseException:
        if identity is not None and not published:
            _cleanup_directory(parent_fd, staging, identity, frozenset(payloads))
        raise
    finally:
        os.close(parent_fd)
    _verify_capsule(target, payloads)


def _publish_launcher(parent: Path, target: Path, raw: bytes) -> None:
    _require_directory(parent, mode=0o755)
    if target.exists() or target.is_symlink():
        if (
            _read_regular(target, expected_sha256=sha256(raw).hexdigest(), mode=0o555)
            != raw
        ):
            raise RuntimeError("immutable launcher payload drifted")
        return
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    staging = f".{target.name}.tmp-{os.getpid()}"
    identity: tuple[int, int] | None = None
    published = False
    try:
        identity = _create_file(parent_fd, staging, raw)
        _rename_noreplace(staging, target.name, parent_fd=parent_fd)
        published = True
        os.fsync(parent_fd)
    except BaseException:
        if identity is not None and not published:
            _cleanup_file(parent_fd, staging, identity)
        raise
    finally:
        os.close(parent_fd)
    if (
        _read_regular(target, expected_sha256=sha256(raw).hexdigest(), mode=0o555)
        != raw
    ):
        raise RuntimeError("installed launcher payload drifted")


def _load_payloads() -> tuple[bytes, dict[str, bytes]]:
    _require_root_chain(SOURCE_ROOT)
    _require_directory(SOURCE_ROOT, mode=0o755)
    if tuple(sorted(item.name for item in SOURCE_ROOT.iterdir())) != SOURCE_NAMES:
        raise RuntimeError("root-owned installation source membership drifted")
    installer = _read_regular(INSTALLER_PATH, expected_sha256=None, mode=0o555)
    payloads = {
        name: _read_regular(SOURCE_ROOT / name, expected_sha256=expected, mode=0o555)
        for name, expected in PAYLOADS.items()
    }
    return installer, payloads


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
    installer, payloads = _load_payloads()
    capsule = {name: payloads[name] for name in CAPSULE_NAMES}
    _publish_capsule(CAPSULE_PARENT, CAPSULE_TARGET, capsule)
    _publish_launcher(
        LAUNCHER_PARENT,
        LAUNCHER_TARGET,
        payloads["launch_gate_d_layer1_prompt_chunk0_geometry.py"],
    )
    print(
        json.dumps(
            {
                "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_runtime_installed",
                "capsule": str(CAPSULE_TARGET),
                "capsule_payload_sha256s": {
                    name: sha256(raw).hexdigest() for name, raw in capsule.items()
                },
                "installer_sha256": sha256(installer).hexdigest(),
                "launcher": str(LAUNCHER_TARGET),
                "launcher_invoked": False,
                "launcher_sha256": PAYLOADS[
                    "launch_gate_d_layer1_prompt_chunk0_geometry.py"
                ],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
