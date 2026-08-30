#!/usr/bin/env -S /usr/bin/python3 -I -S
"""Provision one root-owned, read-only Gate-D Python runtime capsule.

This repository file is installation source, not a sudo entry point. A trusted
administrator must install its reviewed exact bytes as root-owned mode 0555 at
``/opt/glm-tpu/bin/provision_gate_d_python_runtime.py``. Invoke only that fixed
root-owned tool as ``sudo -n /usr/bin/python3 -I -S /opt/glm-tpu/bin/``
``provision_gate_d_python_runtime.py ...``. The shebang supplies the same absolute
interpreter and flags for direct execution. That exact privileged invocation is the
security boundary; the in-process check only detects accidental misinvocation after
Python startup. Provisioning is an explicit trusted-admin prerequisite,
outside parser admission authority. The tool copies an already pinned CPython
runtime, authenticates the complete root/content/symlink/safe-mode tree before
and after the copy, strips privilege bits and extended attributes, removes
group/other write authority, and publishes atomically. It never replaces an
existing target.
"""

from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import os
from pathlib import Path
import shutil
import stat
import struct
import sys


_RENAME_NOREPLACE = 1


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def _tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [root, *sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )]
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
                raise SystemExit(f"runtime symlink escapes source root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise SystemExit(f"unsupported runtime entry type: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", _safe_mode(metadata.st_mode)))
        digest.update(payload)
    return digest.hexdigest()


def _seal_ownership(root: Path) -> None:
    entries = [root, *root.rglob("*")]
    for entry in reversed(entries):
        metadata = entry.lstat()
        for attribute in os.listxattr(entry, follow_symlinks=False):
            os.removexattr(entry, attribute, follow_symlinks=False)
        os.chown(entry, 0, 0, follow_symlinks=False)
        if not stat.S_ISLNK(metadata.st_mode):
            os.chmod(entry, _safe_mode(metadata.st_mode))


def _verify_sealed(root: Path) -> None:
    for entry in [root, *root.rglob("*")]:
        metadata = entry.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or (
            not stat.S_ISLNK(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) != _safe_mode(metadata.st_mode)
        ):
            raise SystemExit(f"runtime entry is not root-owned read-only: {entry}")
        if os.listxattr(entry, follow_symlinks=False):
            raise SystemExit(f"runtime entry retains extended attributes: {entry}")


def _verify_directory_descriptor(descriptor: int, path: Path) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or os.listxattr(path, follow_symlinks=False)
    ):
        raise SystemExit(f"{path} is not an exact root-owned 0755 directory")


def _open_or_create_root() -> tuple[Path, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opt_descriptor = os.open("/opt", flags)
    try:
        _verify_directory_descriptor(opt_descriptor, Path("/opt"))
        created = False
        try:
            os.mkdir("glm-tpu", mode=0o755, dir_fd=opt_descriptor)
            created = True
        except FileExistsError:
            pass
        root_descriptor = os.open("glm-tpu", flags, dir_fd=opt_descriptor)
        if created:
            os.fchmod(root_descriptor, 0o755)
    finally:
        os.close(opt_descriptor)
    root = Path("/opt/glm-tpu")
    try:
        _verify_directory_descriptor(root_descriptor, root)
    except BaseException:
        os.close(root_descriptor)
        raise
    return root, root_descriptor


def _verify_installed_entrypoint() -> None:
    expected = Path("/opt/glm-tpu/bin/provision_gate_d_python_runtime.py")
    observed = Path(os.path.abspath(__file__))
    if observed != expected or Path(os.path.realpath(observed)) != expected:
        raise SystemExit("invoke only the fixed root-owned installed provisioner")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for directory in (Path("/opt"), Path("/opt/glm-tpu"), Path("/opt/glm-tpu/bin")):
        descriptor = os.open(directory, flags)
        try:
            _verify_directory_descriptor(descriptor, directory)
        finally:
            os.close(descriptor)
    metadata = observed.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or os.listxattr(observed, follow_symlinks=False)
    ):
        raise SystemExit("installed provisioner identity is unsafe")


def _verify_root_interpreter() -> None:
    if (
        sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise SystemExit(
            "invoke with exact /usr/bin/python3 -I -S root interpreter"
        )


def _rename_noreplace(source: str, target: str, *, directory_fd: int) -> None:
    """Atomically publish one directory without replacing any target type."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise SystemExit("atomic RENAME_NOREPLACE is unavailable") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.geteuid() != 0:
        raise SystemExit("run with sudo and the root-owned system Python")
    _verify_root_interpreter()
    _verify_installed_entrypoint()
    source = arguments.source.resolve(strict=True)
    target = Path(os.path.abspath(arguments.target))
    allowed_root, root_descriptor = _open_or_create_root()
    try:
        target.relative_to(allowed_root)
    except ValueError as error:
        os.close(root_descriptor)
        raise SystemExit("target must be a direct /opt/glm-tpu descendant") from error
    if target.parent != allowed_root or target.name.startswith("."):
        os.close(root_descriptor)
        raise SystemExit("target must be a named direct /opt/glm-tpu descendant")
    expected = arguments.expected_tree_sha256
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        os.close(root_descriptor)
        raise SystemExit("expected tree SHA-256 is invalid")
    if _tree_sha256(source) != expected:
        os.close(root_descriptor)
        raise SystemExit("source runtime tree SHA-256 drifted")
    try:
        os.stat(target.name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        target_present = False
    else:
        target_present = True
    if target_present:
        _verify_sealed(target)
        if _tree_sha256(target) != expected:
            raise SystemExit("existing runtime target SHA-256 drifted")
        print(f"verified existing immutable runtime {target} {expected}")
        os.close(root_descriptor)
        return 0
    staging = allowed_root / f".{target.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise SystemExit(f"refusing pre-existing staging path: {staging}")
    try:
        shutil.copytree(source, staging, symlinks=True)
        if _tree_sha256(source) != expected or _tree_sha256(staging) != expected:
            raise SystemExit("runtime changed during provisioning")
        _seal_ownership(staging)
        _verify_sealed(staging)
        if _tree_sha256(staging) != expected:
            raise SystemExit("sealed runtime tree SHA-256 drifted")
        _rename_noreplace(
            staging.name,
            target.name,
            directory_fd=root_descriptor,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(root_descriptor)
    print(f"provisioned immutable runtime {target} {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
