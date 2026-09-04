#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Refresh one exact clean detached M2048 worker repository.

The fleet controller invokes this committed blob from a sealed memfd and does
so serially for workers 1--7.  It refuses every state except the observed
clean full-repository prestate or the exact requested successor.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from hashlib import sha256


WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
EXPECTED_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
PIN_PATTERN = re.compile(r"[0-9a-f]{40}")
EXPECTED_LOCAL_CONFIG_SHA256 = (
    "bcc0d9c7525ee1afe2c476062aa3b62a88dbf66725e09a631ed6a6bd672bef51"
)
EXPECTED_LOCAL_CONFIG_MODE = 0o664
SAFE_COMMON_DIRECTORY = Path("/opt/glm-tpu/gate-d-m2048-git-common-v1")
SAFE_COMMON_CONFIG = b"""[core]
\trepositoryformatversion = 0
\tfilemode = true
\tbare = false
\tlogallrefupdates = false
"""
SAFE_COMMON_SUBDIRECTORIES = ("hooks", "info", "objects", "refs")
_ROOT_SAFE_COMMON_PROGRAM = f"""
import hashlib
import os
import stat

root = {str(SAFE_COMMON_DIRECTORY)!r}
parent = os.path.dirname(root)
config = bytes.fromhex({SAFE_COMMON_CONFIG.hex()!r})
subdirectories = {SAFE_COMMON_SUBDIRECTORIES!r}
parent_metadata = os.stat(parent, follow_symlinks=False)
assert stat.S_ISDIR(parent_metadata.st_mode)
assert parent_metadata.st_uid == 0 and parent_metadata.st_gid == 0
assert stat.S_IMODE(parent_metadata.st_mode) == 0o755
assert os.path.realpath(parent) == parent
assert not os.listxattr(parent, follow_symlinks=False)
created = False
try:
    os.mkdir(root, 0o700)
    created = True
except FileExistsError:
    pass
if created:
    for name in subdirectories:
        os.mkdir(os.path.join(root, name), 0o555)
    descriptor = os.open(
        os.path.join(root, "config"),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o444,
    )
    try:
        assert os.write(descriptor, config) == len(config)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    os.chmod(root, 0o555)
expected = {{"config", *subdirectories}}
assert set(os.listdir(root)) == expected
root_metadata = os.stat(root, follow_symlinks=False)
assert stat.S_ISDIR(root_metadata.st_mode)
assert root_metadata.st_uid == 0 and root_metadata.st_gid == 0
assert stat.S_IMODE(root_metadata.st_mode) == 0o555
assert os.path.realpath(root) == root
assert not os.listxattr(root, follow_symlinks=False)
for name in subdirectories:
    path = os.path.join(root, name)
    metadata = os.stat(path, follow_symlinks=False)
    assert stat.S_ISDIR(metadata.st_mode)
    assert metadata.st_uid == 0 and metadata.st_gid == 0
    assert stat.S_IMODE(metadata.st_mode) == 0o555
    assert not os.listdir(path)
    assert not os.listxattr(path, follow_symlinks=False)
path = os.path.join(root, "config")
descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    metadata = os.fstat(descriptor)
    raw = b""
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        raw += block
finally:
    os.close(descriptor)
assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
assert metadata.st_uid == 0 and metadata.st_gid == 0
assert stat.S_IMODE(metadata.st_mode) == 0o444
assert raw == config
assert not os.listxattr(path, follow_symlinks=False)
"""
_BASE_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
READ_GIT_ENVIRONMENT = {
    **_BASE_GIT_ENVIRONMENT,
    "GIT_SSH_COMMAND": "/bin/false",
}
FETCH_GIT_ENVIRONMENT = {
    **_BASE_GIT_ENVIRONMENT,
    "GIT_SSH_COMMAND": (
        "/usr/bin/ssh -oBatchMode=yes -oClearAllForwardings=yes "
        "-oForwardAgent=no -oConnectTimeout=30"
    ),
}


def _prepare_safe_common_directory() -> None:
    try:
        result = subprocess.run(
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/env",
                "-i",
                "HOME=/root",
                "LANG=C",
                "LC_ALL=C",
                "PATH=/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE=1",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                _ROOT_SAFE_COMMON_PROGRAM,
            ],
            check=False,
            capture_output=True,
            env=EXPECTED_ENVIRONMENT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("worker safe Git common preparation failed") from error
    if result.returncode or result.stdout or result.stderr:
        raise RuntimeError("worker safe Git common preparation failed closed")


def _require_safe_common_directory(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> None:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError("worker safe Git common path is unsafe")
    parent = path.parent
    parent_metadata = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != expected_uid
        or parent_metadata.st_gid != expected_gid
        or stat.S_IMODE(parent_metadata.st_mode) != 0o755
        or parent.resolve() != parent
        or os.listxattr(parent, follow_symlinks=False)
    ):
        raise RuntimeError("worker safe Git common parent drifted")
    if set(item.name for item in os.scandir(path)) != {
        "config",
        *SAFE_COMMON_SUBDIRECTORIES,
    }:
        raise RuntimeError("worker safe Git common entries drifted")
    root_metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != expected_uid
        or root_metadata.st_gid != expected_gid
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
        or os.listxattr(path, follow_symlinks=False)
    ):
        raise RuntimeError("worker safe Git common root drifted")
    for name in SAFE_COMMON_SUBDIRECTORIES:
        child = path / name
        metadata = os.stat(child, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or metadata.st_gid != expected_gid
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or any(os.scandir(child))
            or os.listxattr(child, follow_symlinks=False)
        ):
            raise RuntimeError("worker safe Git common directory drifted")
    config = _snapshot_regular_file(
        path / "config",
        expected_mode=0o444,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    if config != SAFE_COMMON_CONFIG:
        raise RuntimeError("worker safe Git common config drifted")


def _safe_git_environment(
    worktree: Path,
    common: Path,
    *,
    ssh_command: str,
    allow_file_transport: bool = False,
) -> dict[str, str]:
    environment = {
        **_BASE_GIT_ENVIRONMENT,
        "GIT_COMMON_DIR": str(common),
        "GIT_DIR": str(worktree / ".git"),
        "GIT_OBJECT_DIRECTORY": str(worktree / ".git" / "objects"),
        "GIT_SSH_COMMAND": ssh_command,
        "GIT_WORK_TREE": str(worktree),
    }
    if allow_file_transport:
        environment["GIT_ALLOW_PROTOCOL"] = "file"
    return environment


def _git(
    worktree: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
    timeout: int = 180,
) -> bytes:
    command = [
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
        str(worktree),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment or READ_GIT_ENVIRONMENT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"worker repository Git command failed: {arguments[0]}") from error
    if result.returncode:
        raise RuntimeError(
            f"worker repository Git command failed: {arguments[0]}: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return result.stdout


def _optional_git_config(worktree: Path, key: str) -> bytes:
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(worktree),
            "config",
            "--local",
            "--get",
            key,
        ],
        check=False,
        capture_output=True,
        env=READ_GIT_ENVIRONMENT,
        timeout=30,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"worker repository config read failed: {key}")
    return result.stdout.strip()


def _snapshot_regular_file(
    path: Path,
    *,
    expected_mode: int,
    expected_uid: int,
    expected_gid: int,
) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
            or stat.S_IMODE(before.st_mode) != expected_mode
        ):
            raise RuntimeError("worker repository regular-file boundary is unsafe")
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
            or identity(before) != identity(named)
        ):
            raise RuntimeError("worker repository regular file changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _snapshot_local_config(worktree: Path, *, expected_mode: int) -> bytes:
    return _snapshot_regular_file(
        worktree / ".git" / "config",
        expected_mode=expected_mode,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )


def _require_repository_boundary(
    worktree: Path,
    *,
    expected_config_sha256: str,
    expected_config_mode: int,
) -> None:
    if not worktree.is_absolute() or worktree.is_symlink() or worktree.resolve() != worktree:
        raise RuntimeError("worker repository path is unsafe")
    metadata = os.stat(worktree, follow_symlinks=False)
    git_metadata = os.stat(worktree / ".git", follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or not stat.S_ISDIR(git_metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or git_metadata.st_uid != os.getuid()
        or git_metadata.st_gid != os.getgid()
    ):
        raise RuntimeError("worker repository boundary is unsafe")
    config = _snapshot_local_config(worktree, expected_mode=expected_config_mode)
    if sha256(config).hexdigest() != expected_config_sha256:
        raise RuntimeError("worker repository local config is outside the sealed schema")
    info_attributes = worktree / ".git" / "info" / "attributes"
    if info_attributes.exists() or info_attributes.is_symlink():
        raise RuntimeError("worker repository info attributes are not permitted")


def _require_no_tracked_attributes(worktree: Path, revision: str) -> None:
    raw = _git(worktree, "ls-tree", "-r", "--name-only", "-z", revision)
    try:
        paths = [item.decode("utf-8", "strict") for item in raw.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise RuntimeError("worker repository tree path is not UTF-8") from error
    if any(Path(path).name == ".gitattributes" for path in paths):
        raise RuntimeError("worker repository target tree contains .gitattributes")


def _validate_state(
    worktree: Path,
    *,
    target_pin: str,
    prestate_pin: str,
    origin: str,
    expected_config_sha256: str,
    expected_config_mode: int,
) -> str:
    _require_repository_boundary(
        worktree,
        expected_config_sha256=expected_config_sha256,
        expected_config_mode=expected_config_mode,
    )
    head = _git(worktree, "rev-parse", "HEAD").decode("ascii").strip()
    if head not in (prestate_pin, target_pin):
        raise RuntimeError("worker repository pin is outside the sealed prestate set")
    if (
        _git(worktree, "branch", "--show-current")
        or _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(worktree, "for-each-ref", "--format=%(refname)", "refs/replace")
        or _git(worktree, "remote", "get-url", "origin").decode("utf-8").strip()
        != origin
        or _git(worktree, "rev-parse", "--is-shallow-repository").strip() != b"false"
        or _optional_git_config(worktree, "remote.origin.promisor")
        or _optional_git_config(worktree, "remote.origin.partialclonefilter")
    ):
        raise RuntimeError("worker repository state is not the exact clean detached contract")
    _require_no_tracked_attributes(worktree, head)
    return head


def refresh_repository(
    worktree: Path,
    *,
    target_pin: str,
    prestate_pin: str,
    branch: str = BRANCH,
    origin: str = ORIGIN,
    allow_file_transport: bool = False,
    expected_config_sha256: str = EXPECTED_LOCAL_CONFIG_SHA256,
    expected_config_mode: int = EXPECTED_LOCAL_CONFIG_MODE,
    safe_common_directory: Path = SAFE_COMMON_DIRECTORY,
    prepare_safe_common_directory: bool = True,
    expected_safe_common_uid: int = 0,
    expected_safe_common_gid: int = 0,
) -> None:
    if PIN_PATTERN.fullmatch(target_pin) is None or PIN_PATTERN.fullmatch(prestate_pin) is None:
        raise RuntimeError("worker repository pin is malformed")
    if re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", branch) is None:
        raise RuntimeError("worker repository branch is malformed")
    if prepare_safe_common_directory:
        if (
            safe_common_directory != SAFE_COMMON_DIRECTORY
            or expected_safe_common_uid != 0
            or expected_safe_common_gid != 0
        ):
            raise RuntimeError("worker safe Git common preparation contract drifted")
        _prepare_safe_common_directory()
    _require_safe_common_directory(
        safe_common_directory,
        expected_uid=expected_safe_common_uid,
        expected_gid=expected_safe_common_gid,
    )
    head = _validate_state(
        worktree,
        target_pin=target_pin,
        prestate_pin=prestate_pin,
        origin=origin,
        expected_config_sha256=expected_config_sha256,
        expected_config_mode=expected_config_mode,
    )
    if head != target_pin:
        _git(
            worktree,
            "fetch",
            "--no-tags",
            "--force",
            origin,
            f"refs/heads/{branch}",
            environment=_safe_git_environment(
                worktree,
                safe_common_directory,
                ssh_command=FETCH_GIT_ENVIRONMENT["GIT_SSH_COMMAND"],
                allow_file_transport=allow_file_transport,
            ),
            timeout=600,
        )
        if _git(worktree, "rev-parse", "FETCH_HEAD").decode("ascii").strip() != target_pin:
            raise RuntimeError("worker repository fetch did not resolve to the exact target pin")
        _require_repository_boundary(
            worktree,
            expected_config_sha256=expected_config_sha256,
            expected_config_mode=expected_config_mode,
        )
        _require_no_tracked_attributes(worktree, target_pin)
        _require_safe_common_directory(
            safe_common_directory,
            expected_uid=expected_safe_common_uid,
            expected_gid=expected_safe_common_gid,
        )
        _git(
            worktree,
            "switch",
            "--detach",
            target_pin,
            environment=_safe_git_environment(
                worktree,
                safe_common_directory,
                ssh_command="/bin/false",
            ),
            timeout=180,
        )
    final = _validate_state(
        worktree,
        target_pin=target_pin,
        prestate_pin=target_pin,
        origin=origin,
        expected_config_sha256=expected_config_sha256,
        expected_config_mode=expected_config_mode,
    )
    if final != target_pin or _git(worktree, "cat-file", "-t", target_pin).strip() != b"commit":
        raise RuntimeError("worker repository refresh did not close at the exact commit")


def main() -> int:
    if dict(os.environ) != EXPECTED_ENVIRONMENT:
        raise RuntimeError("worker repository refresher environment drifted")
    arguments = sys.argv[1:]
    if (
        len(arguments) != 4
        or arguments[0] != "--target-pin"
        or arguments[2] != "--prestate-pin"
    ):
        raise RuntimeError("worker repository refresher invocation drifted")
    target_pin, prestate_pin = arguments[1], arguments[3]
    refresh_repository(
        WORKTREE,
        target_pin=target_pin,
        prestate_pin=prestate_pin,
    )
    print(f"REPO_REFRESH_OK {os.uname().nodename} {target_pin}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
