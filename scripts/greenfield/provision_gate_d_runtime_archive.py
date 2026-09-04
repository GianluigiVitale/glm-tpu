#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Create, install, and remove one exact Gate-D runtime archive safely.

This repository file is installation source.  The only privileged entry point is
the exact root-owned mode-0555 copy at
``/opt/glm-tpu/bin/provision_gate_d_runtime_archive.py``.  Archive creation uses
an exclusive no-follow output descriptor in the root-owned runtime directory.
Installation copies the retained transferred-archive descriptor into an
authenticated sealed memfd, parses only that immutable snapshot, validates
every member before extraction, extracts files before symlinks into a root-owned
staging tree, authenticates the complete tree, and publishes with
``RENAME_NOREPLACE``.  Removal unlinks only the exact retained inode.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import subprocess
import sys
import tarfile


INSTALL_PATH = Path("/opt/glm-tpu/bin/provision_gate_d_runtime_archive.py")
RUNTIME_ROOT = Path("/opt/glm-tpu")
_RENAME_NOREPLACE = 1
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_REQUIRED_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 1)
    | getattr(fcntl, "F_SEAL_SHRINK", 2)
    | getattr(fcntl, "F_SEAL_GROW", 4)
    | getattr(fcntl, "F_SEAL_WRITE", 8)
)


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def _tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            file_digest = sha256()
            with entry.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    file_digest.update(block)
            payload = struct.pack(">Q", metadata.st_size) + file_digest.digest()
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            resolved = Path(os.path.realpath(entry))
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"runtime symlink escapes source root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise RuntimeError(f"unsupported runtime entry type: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", _safe_mode(metadata.st_mode)))
        digest.update(payload)
    return digest.hexdigest()


def _fd_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _fd_sha256_exact(descriptor: int, expected_size: int) -> str:
    """Hash exactly one bounded descriptor and reject truncation or growth."""
    _validate_size(expected_size, "archive")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    remaining = expected_size
    while remaining:
        block = os.read(descriptor, min(1024 * 1024, remaining))
        if not block:
            raise RuntimeError("archive descriptor ended before expected size")
        digest.update(block)
        remaining -= len(block)
    if os.read(descriptor, 1):
        raise RuntimeError("archive descriptor exceeds expected size")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("short write while snapshotting archive")
        view = view[written:]


def _copy_descriptor(source: int, target: int, expected_size: int) -> None:
    """Copy exactly the bounded source size and reject early EOF or growth."""
    _validate_size(expected_size, "archive")
    os.lseek(source, 0, os.SEEK_SET)
    os.lseek(target, 0, os.SEEK_SET)
    if os.fstat(target).st_size != 0:
        raise RuntimeError("archive snapshot descriptor is not empty")
    remaining = expected_size
    while remaining:
        block = os.read(source, min(1024 * 1024, remaining))
        if not block:
            raise RuntimeError("archive source ended before expected size")
        _write_all(target, block)
        remaining -= len(block)
    if os.read(source, 1):
        raise RuntimeError("archive source exceeds expected size")
    os.lseek(source, 0, os.SEEK_SET)
    os.lseek(target, 0, os.SEEK_SET)


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_directory(descriptor: int, *, uid: int, gid: int, mode: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != uid
        or metadata.st_gid != gid
        or stat.S_IMODE(metadata.st_mode) != mode
        or os.listxattr(descriptor)
    ):
        raise RuntimeError("unsafe runtime-root directory descriptor")


def _open_runtime_root(root: Path = RUNTIME_ROOT) -> int:
    descriptor = os.open(
        root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        expected_uid = 0 if root == RUNTIME_ROOT else os.getuid()
        expected_gid = 0 if root == RUNTIME_ROOT else os.getgid()
        _require_directory(descriptor, uid=expected_uid, gid=expected_gid, mode=0o755)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"invalid {label} SHA-256")


def _validate_size(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"invalid {label} size")


def _validate_name(value: str, label: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or value.startswith(".")
        or "/" in value
        or "\x00" in value
    ):
        raise RuntimeError(f"unsafe {label} basename")


def _require_sealed_tree(root: Path, *, uid: int, gid: int) -> None:
    for entry in [root, *root.rglob("*")]:
        metadata = entry.lstat()
        if (
            metadata.st_uid != uid
            or metadata.st_gid != gid
            or (
                not stat.S_ISLNK(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) != _safe_mode(metadata.st_mode)
            )
            or os.listxattr(entry, follow_symlinks=False)
        ):
            raise RuntimeError(f"runtime entry is not sealed: {entry}")


def _cleanup_named_inode(
    parent_fd: int, name: str, identity: tuple[int, int]
) -> None:
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != identity:
        raise RuntimeError("refusing to remove a replaced archive inode")
    os.unlink(name, dir_fd=parent_fd)


def _create_archive(
    source: Path,
    target_name: str,
    expected_tree_sha256: str,
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> tuple[str, os.stat_result]:
    _validate_name(target_name, "archive")
    _validate_digest(expected_tree_sha256, "tree")
    source = source.resolve(strict=True)
    if source.parent != runtime_root:
        raise RuntimeError("archive source must be a direct runtime-root child")
    source_metadata = source.lstat()
    expected_uid = 0 if runtime_root == RUNTIME_ROOT else os.getuid()
    expected_gid = 0 if runtime_root == RUNTIME_ROOT else os.getgid()
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise RuntimeError("archive source is not a directory")
    _require_sealed_tree(source, uid=expected_uid, gid=expected_gid)
    if _tree_sha256(source) != expected_tree_sha256:
        raise RuntimeError("archive source tree SHA-256 drifted")

    parent_fd = _open_runtime_root(runtime_root)
    descriptor = -1
    identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            target_name,
            os.O_RDWR | os.O_CLOEXEC | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
        initial = os.fstat(descriptor)
        identity = (initial.st_dev, initial.st_ino)
        result = subprocess.run(
            [
                "/usr/bin/tar",
                "--format=posix",
                "--sort=name",
                "--numeric-owner",
                "-C",
                str(runtime_root),
                "-cf",
                "-",
                source.name,
            ],
            check=False,
            close_fds=True,
            env={"HOME": "/root", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            stderr=subprocess.PIPE,
            stdout=descriptor,
        )
        if result.returncode != 0 or result.stderr:
            raise RuntimeError("GNU tar archive creation failed closed")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        digest = _fd_sha256(descriptor)
        final = os.fstat(descriptor)
        named = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (named.st_dev, named.st_ino) != identity
            or (final.st_dev, final.st_ino) != identity
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_uid != expected_uid
            or final.st_gid != expected_gid
            or stat.S_IMODE(final.st_mode) != 0o444
            or os.listxattr(descriptor)
            or final.st_size <= 0
            or _tree_sha256(source) != expected_tree_sha256
        ):
            raise RuntimeError("created archive identity or source tree drifted")
        return digest, final
    except BaseException:
        if identity is not None:
            _cleanup_named_inode(parent_fd, target_name, identity)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _open_exact_archive(
    path: Path, expected_sha256: str, expected_size: int
) -> int:
    _validate_digest(expected_sha256, "archive")
    _validate_size(expected_size, "archive")
    source = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
    )
    snapshot = -1
    try:
        before = os.fstat(source)
        named_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
            or stat.S_IMODE(before.st_mode) & 0o022
            or os.listxattr(source)
            or _identity(before) != _identity(named_before)
        ):
            raise RuntimeError("transferred archive identity or digest drifted")

        snapshot = os.memfd_create(
            "gate-d-runtime-archive",
            os.MFD_CLOEXEC | getattr(os, "MFD_ALLOW_SEALING", 2),
        )
        _copy_descriptor(source, snapshot, expected_size)
        after = os.fstat(source)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _identity(before) != _identity(after)
            or _identity(after) != _identity(named_after)
            or _fd_sha256_exact(source, expected_size) != expected_sha256
        ):
            raise RuntimeError("transferred archive changed while snapshotting")

        os.fchmod(snapshot, 0o400)
        snapshot_before_seal = os.fstat(snapshot)
        if (
            not stat.S_ISREG(snapshot_before_seal.st_mode)
            or snapshot_before_seal.st_nlink != 0
            or snapshot_before_seal.st_uid != os.geteuid()
            or snapshot_before_seal.st_gid != os.getegid()
            or stat.S_IMODE(snapshot_before_seal.st_mode) != 0o400
            or snapshot_before_seal.st_size != before.st_size
            or snapshot_before_seal.st_size <= 0
            or os.listxattr(snapshot)
            or _fd_sha256_exact(snapshot, expected_size) != expected_sha256
        ):
            raise RuntimeError("immutable archive snapshot identity or digest drifted")
        fcntl.fcntl(snapshot, _F_ADD_SEALS, _REQUIRED_SEALS)
        if fcntl.fcntl(snapshot, _F_GET_SEALS) != _REQUIRED_SEALS:
            raise RuntimeError("archive snapshot seals drifted")
        snapshot_after_seal = os.fstat(snapshot)
        if _identity(snapshot_before_seal) != _identity(snapshot_after_seal):
            raise RuntimeError("archive snapshot changed while sealing")
        os.close(source)
        source = -1
        os.lseek(snapshot, 0, os.SEEK_SET)
        result = snapshot
        snapshot = -1
        return result
    except BaseException:
        raise
    finally:
        if snapshot >= 0:
            os.close(snapshot)
        if source >= 0:
            os.close(source)


def _member_relative(member: tarfile.TarInfo, source_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member.name)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or parts[0] != source_name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"unsafe archive member path: {member.name}")
    return parts[1:]


def _validate_symlink_target(relative: tuple[str, ...], target: str) -> None:
    link = PurePosixPath(target)
    if link.is_absolute() or not target or "\x00" in target:
        raise RuntimeError("unsafe archive symlink target")
    combined = PurePosixPath(*relative[:-1], *link.parts)
    depth = 0
    for part in combined.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                raise RuntimeError("archive symlink escapes runtime root")
        else:
            depth += 1


def _extract_archive(
    descriptor: int, staging: Path, source_name: str
) -> None:
    _validate_name(source_name, "source")
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            members = archive.getmembers()
            relative_by_name: dict[str, tuple[str, ...]] = {}
            seen: set[str] = set()
            for member in members:
                relative = _member_relative(member, source_name)
                canonical = "/".join(relative)
                if canonical in seen:
                    raise RuntimeError("duplicate archive member path")
                seen.add(canonical)
                relative_by_name[member.name] = relative
                if not (member.isdir() or member.isfile() or member.issym()):
                    raise RuntimeError("unsupported archive member type")
                if member.issym():
                    _validate_symlink_target(relative, member.linkname)
            roots = [member for member in members if not relative_by_name[member.name]]
            if len(roots) != 1 or not roots[0].isdir():
                raise RuntimeError("archive lacks one exact directory root")

            directory_members = [
                member for member in members if member.isdir() and relative_by_name[member.name]
            ]
            for member in sorted(
                directory_members,
                key=lambda item: len(relative_by_name[item.name]),
            ):
                path = staging.joinpath(*relative_by_name[member.name])
                if not path.parent.is_dir() or path.parent.is_symlink():
                    raise RuntimeError("archive directory parent is unsafe")
                path.mkdir(mode=0o700)

            for member in members:
                relative = relative_by_name[member.name]
                if not relative or not member.isfile():
                    continue
                path = staging.joinpath(*relative)
                if not path.parent.is_dir() or path.parent.is_symlink():
                    raise RuntimeError("archive file parent is unsafe")
                payload = archive.extractfile(member)
                if payload is None:
                    raise RuntimeError("archive regular file has no payload")
                with payload, path.open("xb") as output:
                    shutil.copyfileobj(payload, output, 1024 * 1024)
                path.chmod(_safe_mode(member.mode))

            for member in members:
                relative = relative_by_name[member.name]
                if not relative or not member.issym():
                    continue
                path = staging.joinpath(*relative)
                if not path.parent.is_dir() or path.parent.is_symlink():
                    raise RuntimeError("archive symlink parent is unsafe")
                path.symlink_to(member.linkname)

            for member in sorted(
                directory_members,
                key=lambda item: len(relative_by_name[item.name]),
                reverse=True,
            ):
                staging.joinpath(*relative_by_name[member.name]).chmod(
                    _safe_mode(member.mode)
                )
            staging.chmod(_safe_mode(roots[0].mode))
    os.lseek(descriptor, 0, os.SEEK_SET)


def _seal_tree(root: Path) -> None:
    for entry in reversed([root, *root.rglob("*")]):
        for attribute in os.listxattr(entry, follow_symlinks=False):
            os.removexattr(entry, attribute, follow_symlinks=False)
        os.chown(entry, 0, 0, follow_symlinks=False)
        if not entry.is_symlink():
            os.chmod(entry, _safe_mode(entry.lstat().st_mode))


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
    if renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    ) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), target)


def _install_archive(
    archive_path: Path,
    archive_sha256: str,
    archive_size: int,
    source_name: str,
    target_name: str,
    expected_tree_sha256: str,
) -> None:
    _validate_name(source_name, "source")
    _validate_name(target_name, "target")
    if source_name != target_name:
        raise RuntimeError("archive source and target names differ")
    _validate_digest(expected_tree_sha256, "tree")
    descriptor = _open_exact_archive(archive_path, archive_sha256, archive_size)
    parent_fd = _open_runtime_root()
    staging = RUNTIME_ROOT / f".{target_name}.archive-tmp-{os.getpid()}"
    try:
        try:
            os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            target_present = False
        else:
            target_present = True
        if staging.exists() or staging.is_symlink():
            raise RuntimeError("refusing pre-existing archive staging path")
        os.mkdir(staging.name, 0o700, dir_fd=parent_fd)
        _extract_archive(descriptor, staging, source_name)
        if _fd_sha256_exact(descriptor, archive_size) != archive_sha256:
            raise RuntimeError("archive changed during extraction")
        if _tree_sha256(staging) != expected_tree_sha256:
            raise RuntimeError("extracted runtime tree SHA-256 drifted")
        _seal_tree(staging)
        _require_sealed_tree(staging, uid=0, gid=0)
        if _tree_sha256(staging) != expected_tree_sha256:
            raise RuntimeError("sealed extracted runtime tree drifted")
        if target_present:
            target = RUNTIME_ROOT / target_name
            _require_sealed_tree(target, uid=0, gid=0)
            if _tree_sha256(target) != expected_tree_sha256:
                raise RuntimeError("existing runtime target SHA-256 drifted")
            shutil.rmtree(staging)
            return
        _rename_noreplace(staging.name, target_name, parent_fd=parent_fd)
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(parent_fd)
        os.close(descriptor)


def _remove_archive(
    target_name: str,
    expected_sha256: str,
    expected_device: int,
    expected_inode: int,
    expected_size: int,
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> None:
    _validate_name(target_name, "archive")
    _validate_digest(expected_sha256, "archive")
    parent_fd = _open_runtime_root(runtime_root)
    descriptor = os.open(
        target_name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != (0 if runtime_root == RUNTIME_ROOT else os.getuid())
            or before.st_gid != (0 if runtime_root == RUNTIME_ROOT else os.getgid())
            or stat.S_IMODE(before.st_mode) != 0o444
            or os.listxattr(descriptor)
            or (before.st_dev, before.st_ino, before.st_size)
            != (expected_device, expected_inode, expected_size)
            or _fd_sha256(descriptor) != expected_sha256
        ):
            raise RuntimeError("archive cleanup identity drifted")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError("archive changed before cleanup")
        _cleanup_named_inode(
            parent_fd, target_name, (expected_device, expected_inode)
        )
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _verify_entrypoint() -> None:
    observed = Path(os.path.abspath(__file__))
    if observed != INSTALL_PATH or Path(os.path.realpath(observed)) != INSTALL_PATH:
        raise RuntimeError("invoke only the fixed root-owned archive provisioner")
    if (
        os.geteuid() != 0
        or sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("invoke with exact isolated root system Python")
    metadata = observed.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or os.listxattr(observed, follow_symlinks=False)
    ):
        raise RuntimeError("archive provisioner entrypoint identity drifted")
    for directory in (Path("/opt"), RUNTIME_ROOT, INSTALL_PATH.parent):
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            _require_directory(descriptor, uid=0, gid=0, mode=0o755)
        finally:
            os.close(descriptor)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--archive-name", required=True)
    create.add_argument("--expected-tree-sha256", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--archive-sha256", required=True)
    install.add_argument("--archive-size", type=int, required=True)
    install.add_argument("--source-name", required=True)
    install.add_argument("--target-name", required=True)
    install.add_argument("--expected-tree-sha256", required=True)
    remove = subparsers.add_parser("remove")
    remove.add_argument("--archive-name", required=True)
    remove.add_argument("--expected-sha256", required=True)
    remove.add_argument("--expected-device", type=int, required=True)
    remove.add_argument("--expected-inode", type=int, required=True)
    remove.add_argument("--expected-size", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    _verify_entrypoint()
    arguments = _arguments()
    if arguments.mode == "create":
        digest, metadata = _create_archive(
            arguments.source,
            arguments.archive_name,
            arguments.expected_tree_sha256,
        )
        print(
            "ARCHIVE_READY",
            digest,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        )
    elif arguments.mode == "install":
        _install_archive(
            arguments.archive,
            arguments.archive_sha256,
            arguments.archive_size,
            arguments.source_name,
            arguments.target_name,
            arguments.expected_tree_sha256,
        )
        print(f"ARCHIVE_RUNTIME_OK {arguments.target_name}")
    else:
        _remove_archive(
            arguments.archive_name,
            arguments.expected_sha256,
            arguments.expected_device,
            arguments.expected_inode,
            arguments.expected_size,
        )
        print(f"ARCHIVE_REMOVED {arguments.archive_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
