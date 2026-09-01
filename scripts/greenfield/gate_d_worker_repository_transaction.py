#!/usr/bin/python3
"""Descriptor-bound worker transaction for Gate-D repository recovery.

The controller transports this committed source as ``python -c`` input.  It is
standard-library only, holds a per-worker lock, and never touches TPU state.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_PARENT = Path("/home/gianl")
_LOCK_PARENT = Path("/opt/glm-tpu/locks")
_LOCK_NAME = "gate_d_repo_recovery.lock"
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_GIT_ENV = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_ALLOW_PROTOCOL": "file",
}
_LOCK_FD = -1


class TransactionError(RuntimeError):
    pass


def _fail(reason: str) -> None:
    raise TransactionError(reason)


def _regular_identity(fd: int) -> tuple[int, int, int, int]:
    value = os.fstat(fd)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        _fail("regular_boundary")
    return value.st_dev, value.st_ino, value.st_size, value.st_mode


def _directory_identity(fd: int) -> tuple[int, int]:
    value = os.fstat(fd)
    if not stat.S_ISDIR(value.st_mode):
        _fail("directory_boundary")
    return value.st_dev, value.st_ino


def _path_identity(parent_fd: int, name: str) -> tuple[int, int, int]:
    value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    return value.st_dev, value.st_ino, value.st_mode


def _open_directory(parent_fd: int, name: str) -> int:
    fd = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    _directory_identity(fd)
    return fd


def _sha256_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest(), total


def _run_git(*arguments: str, pass_fds: tuple[int, ...] = ()) -> str:
    inherited_fds = tuple(fd for fd in (_LOCK_FD, *pass_fds) if fd >= 0)
    environment = dict(_GIT_ENV)
    environment["GLM_GATE_D_REPO_RECOVERY_CARRIER"] = os.environ[
        "GLM_GATE_D_REPO_RECOVERY_CARRIER"
    ]
    result = subprocess.run(
        ["/usr/bin/git", *arguments],
        check=False,
        capture_output=True,
        env=environment,
        pass_fds=inherited_fds,
        text=True,
        timeout=120,
    )
    if result.returncode:
        _fail(f"git_{arguments[0]}")
    return result.stdout.strip()


def _verify(
    verifier: str,
    pin: str,
    path: str,
    origin: str,
    linked_common: str,
    linked_git_dir: str,
    expected_layout: str,
    expected_missing_count: int,
    expected_missing_sha: str,
    expected_promisor_count: int,
    expected_promisor_sha: str,
) -> None:
    environment = dict(_GIT_ENV)
    environment["GLM_GATE_D_REPO_RECOVERY_CARRIER"] = os.environ[
        "GLM_GATE_D_REPO_RECOVERY_CARRIER"
    ]
    result = subprocess.run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            verifier,
            "gate-d-repository-transaction",
            pin,
            path,
            origin,
            linked_common,
            linked_git_dir,
            expected_layout,
            str(expected_missing_count),
            expected_missing_sha,
            str(expected_promisor_count),
            expected_promisor_sha,
        ],
        check=False,
        capture_output=True,
        env=environment,
        pass_fds=(_LOCK_FD,) if _LOCK_FD >= 0 else (),
        text=True,
        timeout=120,
    )
    if result.returncode or not result.stdout.startswith("SYNC_OK "):
        reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        _fail("repository_verify:" + reason)


def _verify_old(arguments: argparse.Namespace, pin: str, path: str) -> None:
    _verify(
        arguments.verifier,
        pin,
        path,
        arguments.origin,
        arguments.linked_common,
        arguments.linked_git_dir,
        arguments.old_layout,
        arguments.old_missing_count,
        arguments.old_missing_sha,
        arguments.old_promisor_count,
        arguments.old_promisor_sha,
    )


def _verify_target(arguments: argparse.Namespace, pin: str, path: str) -> None:
    _verify(
        arguments.verifier,
        pin,
        path,
        arguments.origin,
        arguments.linked_common,
        arguments.linked_git_dir,
        arguments.target_layout,
        arguments.target_missing_count,
        arguments.target_missing_sha,
        arguments.target_promisor_count,
        arguments.target_promisor_sha,
    )


def _renameat2(
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
    flags: int,
) -> None:
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise TransactionError("renameat2_unavailable") from error
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        old_dir_fd,
        os.fsencode(old_name),
        new_dir_fd,
        os.fsencode(new_name),
        flags,
    )
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def _acquire_lock() -> int:
    lock_parent_fd = os.open(
        _LOCK_PARENT,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    fd = -1
    try:
        parent_identity = os.fstat(lock_parent_fd)
        if (
            not stat.S_ISDIR(parent_identity.st_mode)
            or parent_identity.st_uid != 0
            or parent_identity.st_gid != 0
            or stat.S_IMODE(parent_identity.st_mode) & 0o022
        ):
            _fail("lock_parent_identity")
        fd = os.open(
            _LOCK_NAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=lock_parent_fd,
        )
        identity = _regular_identity(fd)
        if (
            stat.S_IMODE(identity[3]) != 0o666
            or os.fstat(fd).st_uid != 0
            or os.fstat(fd).st_gid != 0
        ):
            _fail("lock_identity")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TransactionError("worker_lock_held") from error
        named = os.stat(_LOCK_NAME, dir_fd=lock_parent_fd, follow_symlinks=False)
        held = os.fstat(fd)
        if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
            _fail("lock_path_drift")
        return fd
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        os.close(lock_parent_fd)


def _assert_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        _fail("unsafe_name")


def _vacant(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _canonical_path(name: str) -> str:
    return str(_PARENT / name)


def _revalidate(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
    current = _path_identity(parent_fd, name)
    if not stat.S_ISDIR(current[2]) or current[:2] != expected:
        _fail(f"identity_drift_{name}")


def _repository_pin(path: str) -> str:
    pin = _run_git("-C", path, "rev-parse", "--verify", "HEAD")
    if len(pin) != 40 or any(character not in "0123456789abcdef" for character in pin):
        _fail("invalid_old_pin")
    return pin


def preflight(arguments: argparse.Namespace, parent_fd: int) -> dict[str, object]:
    for name in (arguments.worktree, arguments.new, arguments.old, arguments.bundle):
        _assert_name(name)
    canonical_fd = _open_directory(parent_fd, arguments.worktree)
    try:
        canonical_identity = _directory_identity(canonical_fd)
        old_pin = _repository_pin(_canonical_path(arguments.worktree))
        if old_pin == arguments.pin:
            _verify_target(arguments, old_pin, _canonical_path(arguments.worktree))
        else:
            _verify_old(arguments, old_pin, _canonical_path(arguments.worktree))
        _revalidate(parent_fd, arguments.worktree, canonical_identity)
    finally:
        os.close(canonical_fd)
    for name in (arguments.new, arguments.old, arguments.bundle):
        if not _vacant(parent_fd, name):
            _fail(f"occupied_{name}")
    return {
        "marker": "REPO_PREFLIGHT_OK",
        "host": os.uname().nodename,
        "old_dev": canonical_identity[0],
        "old_ino": canonical_identity[1],
        "old_pin": old_pin,
    }


def receive_prepare(
    arguments: argparse.Namespace,
    parent_fd: int,
    bundle_input: BinaryIO | None = None,
) -> dict[str, object]:
    for name in (arguments.worktree, arguments.new, arguments.old, arguments.bundle):
        _assert_name(name)
    for name in (arguments.new, arguments.old, arguments.bundle):
        if not _vacant(parent_fd, name):
            _fail(f"occupied_{name}")

    canonical_fd = _open_directory(parent_fd, arguments.worktree)
    bundle_fd = os.open(
        arguments.bundle,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        digest = hashlib.sha256()
        size = 0
        source = bundle_input if bundle_input is not None else sys.stdin.buffer
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(bundle_fd, view)
                view = view[written:]
            digest.update(chunk)
            size += len(chunk)
        os.fsync(bundle_fd)
        bundle_identity = _regular_identity(bundle_fd)
        if digest.hexdigest() != arguments.bundle_sha or size != arguments.bundle_bytes:
            _fail("bundle_payload")
        if bundle_identity[2] != size:
            _fail("bundle_size")

        old_identity = _directory_identity(canonical_fd)
        if old_identity != (arguments.old_dev, arguments.old_ino):
            _fail("canonical_prestate")
        old_pin = _repository_pin(_canonical_path(arguments.worktree))
        if old_pin != arguments.old_pin:
            _fail("canonical_pin")
        _verify_old(arguments, old_pin, _canonical_path(arguments.worktree))
        _revalidate(parent_fd, arguments.worktree, old_identity)

        os.set_inheritable(bundle_fd, True)
        _run_git(
            "clone",
            "-q",
            "--no-checkout",
            f"/proc/self/fd/{bundle_fd}",
            _canonical_path(arguments.new),
            pass_fds=(bundle_fd,),
        )
        _run_git(
            "-C",
            _canonical_path(arguments.new),
            "checkout",
            "-q",
            "--detach",
            arguments.pin,
        )
        _run_git(
            "-C",
            _canonical_path(arguments.new),
            "remote",
            "set-url",
            "origin",
            arguments.origin,
        )
        new_fd = _open_directory(parent_fd, arguments.new)
        try:
            new_identity = _directory_identity(new_fd)
            _verify_target(arguments, arguments.pin, _canonical_path(arguments.new))
            _revalidate(parent_fd, arguments.new, new_identity)
            if _regular_identity(bundle_fd) != bundle_identity:
                _fail("bundle_identity_drift")
            bundle_sha, bundle_bytes = _sha256_fd(bundle_fd)
            if (
                bundle_sha != arguments.bundle_sha
                or bundle_bytes != arguments.bundle_bytes
            ):
                _fail("bundle_replay")
        finally:
            os.close(new_fd)
        _renameat2(
            parent_fd, arguments.new, parent_fd, arguments.old, _RENAME_NOREPLACE
        )
        os.fsync(parent_fd)
        _verify_target(arguments, arguments.pin, _canonical_path(arguments.old))
        _revalidate(parent_fd, arguments.old, new_identity)
    finally:
        os.close(bundle_fd)
        os.close(canonical_fd)

    return {
        "marker": "REPO_PREPARED",
        "host": os.uname().nodename,
        "old_dev": old_identity[0],
        "old_ino": old_identity[1],
        "old_pin": old_pin,
        "new_dev": new_identity[0],
        "new_ino": new_identity[1],
        "pin": arguments.pin,
        "bundle_sha256": arguments.bundle_sha,
    }


def _rollback_after_exchange(
    arguments: argparse.Namespace,
    parent_fd: int,
    old_identity: tuple[int, int],
    new_identity: tuple[int, int],
) -> None:
    _renameat2(
        parent_fd, arguments.worktree, parent_fd, arguments.old, _RENAME_EXCHANGE
    )
    os.fsync(parent_fd)
    _verify_old(arguments, arguments.old_pin, _canonical_path(arguments.worktree))
    _verify_target(arguments, arguments.pin, _canonical_path(arguments.old))
    _revalidate(parent_fd, arguments.worktree, old_identity)
    _revalidate(parent_fd, arguments.old, new_identity)


def swap(arguments: argparse.Namespace, parent_fd: int) -> dict[str, object]:
    for name in (arguments.worktree, arguments.new, arguments.old, arguments.bundle):
        _assert_name(name)
    if not _vacant(parent_fd, arguments.new):
        _fail("new_path_reappeared")
    old_fd = _open_directory(parent_fd, arguments.worktree)
    new_fd = _open_directory(parent_fd, arguments.old)
    bundle_fd = os.open(
        arguments.bundle,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    exchanged = False
    try:
        old_identity = _directory_identity(old_fd)
        new_identity = _directory_identity(new_fd)
        if old_identity != (arguments.old_dev, arguments.old_ino):
            _fail("canonical_prestate")
        if new_identity != (arguments.new_dev, arguments.new_ino):
            _fail("prepared_prestate")
        _verify_old(arguments, arguments.old_pin, _canonical_path(arguments.worktree))
        _verify_target(arguments, arguments.pin, _canonical_path(arguments.old))
        _revalidate(parent_fd, arguments.worktree, old_identity)
        _revalidate(parent_fd, arguments.old, new_identity)
        bundle_sha, bundle_bytes = _sha256_fd(bundle_fd)
        if bundle_sha != arguments.bundle_sha or bundle_bytes != arguments.bundle_bytes:
            _fail("bundle_replay")

        _renameat2(
            parent_fd, arguments.worktree, parent_fd, arguments.old, _RENAME_EXCHANGE
        )
        exchanged = True
        try:
            os.fsync(parent_fd)
            _verify_target(
                arguments, arguments.pin, _canonical_path(arguments.worktree)
            )
            _verify_old(arguments, arguments.old_pin, _canonical_path(arguments.old))
            _revalidate(parent_fd, arguments.worktree, new_identity)
            _revalidate(parent_fd, arguments.old, old_identity)
        except BaseException:
            try:
                _rollback_after_exchange(
                    arguments, parent_fd, old_identity, new_identity
                )
            finally:
                exchanged = False
            raise

        current_bundle = _path_identity(parent_fd, arguments.bundle)
        descriptor_bundle = os.fstat(bundle_fd)
        if current_bundle[:2] != (descriptor_bundle.st_dev, descriptor_bundle.st_ino):
            _fail("bundle_name_drift")
        os.unlink(arguments.bundle, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if exchanged:
            _rollback_after_exchange(arguments, parent_fd, old_identity, new_identity)
        raise
    finally:
        os.close(bundle_fd)
        os.close(new_fd)
        os.close(old_fd)

    return {
        "marker": "REPO_SWAPPED",
        "host": os.uname().nodename,
        "pin": arguments.pin,
        "old_pin": arguments.old_pin,
        "old_dev": old_identity[0],
        "old_ino": old_identity[1],
        "new_dev": new_identity[0],
        "new_ino": new_identity[1],
    }


def final(arguments: argparse.Namespace, parent_fd: int) -> dict[str, object]:
    current_fd = _open_directory(parent_fd, arguments.worktree)
    old_fd = _open_directory(parent_fd, arguments.old)
    try:
        if _directory_identity(current_fd) != (arguments.new_dev, arguments.new_ino):
            _fail("final_current_identity")
        if _directory_identity(old_fd) != (arguments.old_dev, arguments.old_ino):
            _fail("final_old_identity")
        _verify_target(arguments, arguments.pin, _canonical_path(arguments.worktree))
        _verify_old(arguments, arguments.old_pin, _canonical_path(arguments.old))
        _revalidate(
            parent_fd, arguments.worktree, (arguments.new_dev, arguments.new_ino)
        )
        _revalidate(parent_fd, arguments.old, (arguments.old_dev, arguments.old_ino))
    finally:
        os.close(old_fd)
        os.close(current_fd)
    return {
        "marker": "REPO_FINAL_OK",
        "host": os.uname().nodename,
        "pin": arguments.pin,
        "old_pin": arguments.old_pin,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("preflight", "receive-prepare", "swap", "final")
    )
    parser.add_argument("--pin", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha", default="")
    parser.add_argument("--bundle-bytes", type=int, default=0)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--linked-common", required=True)
    parser.add_argument("--linked-git-dir", required=True)
    parser.add_argument(
        "--old-layout", choices=("standalone", "standalone_promisor"), required=True
    )
    parser.add_argument("--old-missing-count", type=int, required=True)
    parser.add_argument("--old-missing-sha", required=True)
    parser.add_argument("--old-promisor-count", type=int, required=True)
    parser.add_argument("--old-promisor-sha", required=True)
    parser.add_argument("--target-layout", choices=("standalone",), required=True)
    parser.add_argument("--target-missing-count", type=int, choices=(0,), required=True)
    parser.add_argument("--target-missing-sha", choices=(_EMPTY_SHA256,), required=True)
    parser.add_argument(
        "--target-promisor-count", type=int, choices=(0,), required=True
    )
    parser.add_argument(
        "--target-promisor-sha", choices=(_EMPTY_SHA256,), required=True
    )
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--old-pin", default="")
    parser.add_argument("--old-dev", type=int, default=-1)
    parser.add_argument("--old-ino", type=int, default=-1)
    parser.add_argument("--new-dev", type=int, default=-1)
    parser.add_argument("--new-ino", type=int, default=-1)
    return parser


def main() -> int:
    global _LOCK_FD
    arguments = _parser().parse_args()
    if os.environ.get("GLM_GATE_D_REPO_RECOVERY_CARRIER", "") == "":
        _fail("missing_carrier")
    parent_fd = os.open(
        _PARENT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    lock_fd = _acquire_lock()
    _LOCK_FD = lock_fd
    try:
        if arguments.phase == "preflight":
            result = preflight(arguments, parent_fd)
        elif arguments.phase == "receive-prepare":
            result = receive_prepare(arguments, parent_fd)
        elif arguments.phase == "swap":
            result = swap(arguments, parent_fd)
        else:
            result = final(arguments, parent_fd)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - transaction boundary must emit one refusal record
        print(
            json.dumps(
                {
                    "marker": "REPO_TRANSACTION_BAD",
                    "host": os.uname().nodename,
                    "reason": type(error).__name__ + ":" + str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        os.close(lock_fd)
        os.close(parent_fd)


if __name__ == "__main__":
    raise SystemExit(main())
