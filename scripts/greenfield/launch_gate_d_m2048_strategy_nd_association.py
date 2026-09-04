#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Root-owned launcher for one protected exact-M2048 fleet fingerprint."""

from __future__ import annotations

import fcntl
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


INSTALL_PATH = Path("/opt/glm-tpu/bin/launch_gate_d_m2048_strategy_nd_v3.py")
SYSTEM_PYTHON = Path("/usr/bin/python3.10")
SYSTEM_PYTHON_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
SOURCE_PATH = (
    "scripts/greenfield/launch_gate_d_m2048_strategy_nd_association.py"
)
WRAPPER_PATH = WORKTREE / (
    "scripts/greenfield/run_gate_d_m2048_strategy_nd_association.sh"
)
WRAPPER_SOURCE_PATH = (
    "scripts/greenfield/run_gate_d_m2048_strategy_nd_association.sh"
)
WRAPPER_SHA256 = "f09f523995ad1ff6c34f7709931026ff2a9e48be23d8a55eab137c8c1947bf68"
CAPSULE_ROOT = Path(
    "/usr/local/libexec/glm-tpu/gate-d-m2048-strategy-nd-v3"
)
PROBE_PATH = CAPSULE_ROOT / "probe_m2048_strategy_nd_association.py"
PROBE_SOURCE_PATH = "scripts/greenfield/probe_m2048_strategy_nd_association.py"
PROBE_SHA256 = "0eb543cb85919f5541f2f8e1cf206eeb07630293ea4ba707d1728376fa6a1502"
PUBLISHER_PATH = (
    CAPSULE_ROOT / "publish_gate_d_m2048_strategy_nd_association.py"
)
PUBLISHER_SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_m2048_strategy_nd_association.py"
)
PUBLISHER_SHA256 = (
    "f6472b2cb43318fad9ece2396c0b079e73ad9840b7bf5b3dd7c131586a52479a"
)
MIRROR_VERIFIER_PATH = (
    CAPSULE_ROOT / "verify_gate_d_rewrite_same_region_git_mirror.py"
)
MIRROR_VERIFIER_SOURCE_PATH = (
    "scripts/greenfield/verify_gate_d_rewrite_same_region_git_mirror.py"
)
MIRROR_VERIFIER_SHA256 = (
    "d49bff51355d88e353d81f0c8c3aafa0457068567f2bb845ae165efc70a7e465"
)
LOCK_ROOT = Path("/opt/glm-tpu/locks")
LOCK_NAMES = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")
LOCK_FDS = (11, 12)
WRAPPER_FD = 10
RUN_FD = 7
CHILDREN = (
    ("PROBE", PROBE_PATH, PROBE_SOURCE_PATH, PROBE_SHA256),
    ("PUBLISHER", PUBLISHER_PATH, PUBLISHER_SOURCE_PATH, PUBLISHER_SHA256),
    (
        "MIRROR_VERIFIER",
        MIRROR_VERIFIER_PATH,
        MIRROR_VERIFIER_SOURCE_PATH,
        MIRROR_VERIFIER_SHA256,
    ),
)
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
REQUIRED_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
PIN_PATTERN = re.compile(r"[0-9a-f]{40}")
TAG_PATTERN = re.compile(
    r"greenfield_m2048_strategy_nd_[0-9]{8}T[0-9]{15}Z"
)
INPUT_ENVIRONMENT_KEYS = {
    "GLM_GATE_D_M2048",
    "GLM_GATE_D_M2048_MODE",
    "GLM_GATE_D_M2048_TAG",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
}
EXPECTED_FIXED_ENVIRONMENT = {
    "GLM_GATE_D_M2048": "1",
    "GLM_GATE_D_M2048_MODE": "execute_once",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _read_stable_regular(
    path: Path,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (expected_uid is not None and before.st_uid != expected_uid)
            or (expected_gid is not None and before.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
            or os.listxattr(path, follow_symlinks=False)
        ):
            raise RuntimeError(f"unsafe protected source identity: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"protected source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
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
                *arguments,
            ],
            env=GIT_ENVIRONMENT,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("launcher Git identity verification failed") from error


def _require_root_boundary(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe root-owned launcher path: {path}")
    for parent in (path.parent, *path.parent.parents):
        value = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != 0
            or value.st_gid != 0
            or stat.S_IMODE(value.st_mode) & 0o022
        ):
            raise RuntimeError(f"unsafe root-owned launcher parent: {parent}")


def _create_sealed_wrapper(raw: bytes) -> int:
    descriptor = os.memfd_create(
        "glm-gate-d-m2048-wrapper",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("failed to snapshot protected wrapper")
            view = view[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, F_ADD_SEALS, REQUIRED_SEALS)
        if fcntl.fcntl(descriptor, F_GET_SEALS) != REQUIRED_SEALS:
            raise RuntimeError("protected wrapper memfd is not fully sealed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_locked_fds() -> list[int]:
    parent_fd = os.open(
        LOCK_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    descriptors: list[int] = []
    try:
        parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != 0
            or parent.st_gid != 0
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise RuntimeError("unsafe immutable lock parent")
        for name in LOCK_NAMES:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            held = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(held.st_mode)
                or held.st_nlink != 1
                or held.st_uid != 0
                or held.st_gid != 0
                or stat.S_IMODE(held.st_mode) != 0o666
                or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
            ):
                os.close(descriptor)
                raise RuntimeError(f"unsafe immutable lock identity: {name}")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors.append(descriptor)
        return descriptors
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


def _create_retained_run_fd(tag: str) -> int:
    root_fd = os.open(
        RUN_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    descriptor = -1
    try:
        root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root.st_mode)
            or stat.S_IMODE(root.st_mode) != 0o700
            or root.st_uid != os.geteuid()
            or root.st_gid != os.getegid()
            or os.listxattr(RUN_ROOT, follow_symlinks=False)
        ):
            raise RuntimeError("unsafe private M2048 run root")
        os.mkdir(tag, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        descriptor = os.open(
            tag,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        held = os.fstat(descriptor)
        named = os.stat(tag, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or held.st_uid != os.geteuid()
            or held.st_gid != os.getegid()
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError("new M2048 run-directory identity drifted")
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _bind_inherited_fd(descriptor: int, target: int) -> None:
    if descriptor != target:
        os.dup2(descriptor, target, inheritable=True)
        os.close(descriptor)
    else:
        os.set_inheritable(target, True)


def main() -> int:
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError("launcher must run from its root-owned installation")
    if (
        Path(sys.executable).resolve() != SYSTEM_PYTHON
        or sys.argv != [str(INSTALL_PATH)]
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or not sys.flags.dont_write_bytecode
        or sha256(
            _read_stable_regular(
                SYSTEM_PYTHON,
                expected_uid=0,
                expected_gid=0,
                expected_mode=0o755,
            )
        ).hexdigest()
        != SYSTEM_PYTHON_SHA256
    ):
        raise RuntimeError("launcher Python runtime drifted")
    if set(os.environ) != INPUT_ENVIRONMENT_KEYS:
        raise RuntimeError("launcher environment names are not exact")
    if any(
        os.environ.get(key) != value
        for key, value in EXPECTED_FIXED_ENVIRONMENT.items()
    ):
        raise RuntimeError("launcher environment values drifted")
    tag = os.environ["GLM_GATE_D_M2048_TAG"]
    if TAG_PATTERN.fullmatch(tag) is None:
        raise RuntimeError("launcher tag is invalid")

    _require_root_boundary(INSTALL_PATH)
    launcher_raw = _read_stable_regular(
        INSTALL_PATH, expected_uid=0, expected_gid=0, expected_mode=0o555
    )
    pin = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if PIN_PATTERN.fullmatch(pin) is None:
        raise RuntimeError("launcher code pin is invalid")
    if (
        _git("rev-parse", "--show-toplevel") != f"{WORKTREE}\n".encode()
        or _git("branch", "--show-current") != f"{BRANCH}\n".encode()
        or _git("status", "--porcelain=v1", "--untracked-files=all")
        or _git("for-each-ref", "--format=%(refname)", "refs/replace")
        or _git("remote", "get-url", "origin") != f"{ORIGIN}\n".encode()
        or launcher_raw != _git("show", f"{pin}:{SOURCE_PATH}")
    ):
        raise RuntimeError("launcher repository authority drifted")
    wrapper_raw = _read_stable_regular(WRAPPER_PATH)
    if sha256(wrapper_raw).hexdigest() != WRAPPER_SHA256 or wrapper_raw != _git(
        "show", f"{pin}:{WRAPPER_SOURCE_PATH}"
    ):
        raise RuntimeError("protected wrapper is not the expected committed blob")
    for label, path, repository_path, expected_sha256 in CHILDREN:
        _require_root_boundary(path)
        raw = _read_stable_regular(
            path, expected_uid=0, expected_gid=0, expected_mode=0o555
        )
        if sha256(raw).hexdigest() != expected_sha256 or raw != _git(
            "show", f"{pin}:{repository_path}"
        ):
            raise RuntimeError(
                f"protected {label.lower()} is not the expected committed blob"
            )
    wrapper_fd = _create_sealed_wrapper(wrapper_raw)
    lock_fds = _open_locked_fds()
    run_fd = _create_retained_run_fd(tag)
    _bind_inherited_fd(run_fd, RUN_FD)
    _bind_inherited_fd(wrapper_fd, WRAPPER_FD)
    for descriptor, target in zip(lock_fds, LOCK_FDS, strict=True):
        _bind_inherited_fd(descriptor, target)
    environment = dict(os.environ)
    environment.update(
        {
            "GLM_GATE_D_IMMUTABLE_LOCKS_HELD": "1",
            "GLM_GATE_D_LAUNCHER_PIN": pin,
            "GLM_GATE_D_LAUNCHER_SHA256": sha256(launcher_raw).hexdigest(),
            "GLM_GATE_D_M2048_WRAPPER_SANITIZED": "1",
            "GLM_GATE_D_WRAPPER_MEMFD": str(WRAPPER_FD),
            "GLM_GATE_D_WRAPPER_SHA256": WRAPPER_SHA256,
        }
    )
    os.execve(
        "/usr/bin/bash",
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            f"/proc/self/fd/{WRAPPER_FD}",
        ],
        environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
