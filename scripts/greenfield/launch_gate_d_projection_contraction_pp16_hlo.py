#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Root-owned, descriptor-bound launcher for projection-contraction PP16 HLO acquisition."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

INSTALL_PATH = Path(
    "/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v3.py"
)
PYTHON = Path("/usr/bin/python3.10")
PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
SOURCE_PATH = "scripts/greenfield/launch_gate_d_projection_contraction_pp16_hlo.py"
WRAPPER_PATH = (
    WORKTREE / "scripts/greenfield/run_gate_d_projection_contraction_pp16_hlo.sh"
)
WRAPPER_SOURCE_PATH = "scripts/greenfield/run_gate_d_projection_contraction_pp16_hlo.sh"
WRAPPER_SHA256 = "4dd06dcb57fdb5a43d0eec0f7a6754724d7534637bda539eb36831359e1b1c70"
IMMUTABLE_CAPSULE_ROOT = Path(
    "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v3"
)
DRIVER_PATH = (
    IMMUTABLE_CAPSULE_ROOT / "acquire_gate_d_projection_contraction_pp16_hlo.py"
)
DRIVER_SOURCE_PATH = (
    "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
)
DRIVER_SHA256 = "c660d50eb60054c9b267840230fb69bbf7259104416dc96c7b2ee01a2a14934a"
PUBLISHER_PATH = (
    IMMUTABLE_CAPSULE_ROOT / "publish_gate_d_projection_contraction_pp16_hlo.py"
)
PUBLISHER_SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
)
PUBLISHER_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
MIRROR_VERIFIER_PATH = (
    IMMUTABLE_CAPSULE_ROOT / "verify_gate_d_same_region_git_mirror.py"
)
MIRROR_VERIFIER_SOURCE_PATH = (
    "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
)
MIRROR_VERIFIER_SHA256 = (
    "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b"
)
LOCK_ROOT = Path("/opt/glm-tpu/locks")
LOCK_NAMES = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")
LOCK_FDS = (11, 12)
WRAPPER_FD = 10
IMMUTABLE_CHILDREN = (
    ("DRIVER", DRIVER_PATH, DRIVER_SOURCE_PATH, DRIVER_SHA256),
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
TAG_PATTERN = re.compile(r"gate_d_projection_contraction_pp16_hlo_[0-9]{8}T[0-9]{15}Z")
INPUT_ENVIRONMENT_KEYS = {
    "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE",
    "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE",
    "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
}
EXPECTED_FIXED_ENVIRONMENT = {
    "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE": "1",
    "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE": "compile_only",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin",
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
    "HOME": "/nonexistent",
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
        ):
            raise RuntimeError(f"unsafe protected source identity: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or not stable
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"protected source changed while reading: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
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
        "glm-gate-d-projection-contraction-hlo-wrapper",
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
    try:
        parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != 0
            or parent.st_gid != 0
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise RuntimeError("unsafe immutable lock parent")
        descriptors = []
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
        for descriptor in locals().get("descriptors", []):
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


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
        Path(sys.executable).resolve() != PYTHON
        or sys.argv != [str(INSTALL_PATH)]
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or not sys.flags.dont_write_bytecode
        or sha256(
            _read_stable_regular(
                PYTHON, expected_uid=0, expected_gid=0, expected_mode=0o755
            )
        ).hexdigest()
        != PYTHON_SHA256
    ):
        raise RuntimeError("launcher Python runtime drifted")
    if set(os.environ) != INPUT_ENVIRONMENT_KEYS:
        raise RuntimeError("launcher environment names are not exact")
    if any(
        os.environ.get(key) != value
        for key, value in EXPECTED_FIXED_ENVIRONMENT.items()
    ):
        raise RuntimeError("launcher environment values drifted")
    tag = os.environ["GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG"]
    if TAG_PATTERN.fullmatch(tag) is None:
        raise RuntimeError("launcher tag is invalid")

    _require_root_boundary(INSTALL_PATH)
    launcher_raw = _read_stable_regular(
        INSTALL_PATH, expected_uid=0, expected_gid=0, expected_mode=0o555
    )
    if _git("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("launcher repository has replacement refs")
    pin = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if PIN_PATTERN.fullmatch(pin) is None:
        raise RuntimeError("launcher code pin is invalid")
    if launcher_raw != _git("show", f"{pin}:{SOURCE_PATH}"):
        raise RuntimeError("installed launcher is not the committed blob")

    wrapper_raw = _read_stable_regular(WRAPPER_PATH)
    if sha256(wrapper_raw).hexdigest() != WRAPPER_SHA256 or wrapper_raw != _git(
        "show", f"{pin}:{WRAPPER_SOURCE_PATH}"
    ):
        raise RuntimeError("protected wrapper is not the expected committed blob")
    for label, path, repository_path, expected_sha256 in IMMUTABLE_CHILDREN:
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
    _bind_inherited_fd(wrapper_fd, WRAPPER_FD)
    for descriptor, target in zip(lock_fds, LOCK_FDS, strict=True):
        _bind_inherited_fd(descriptor, target)

    environment = dict(os.environ)
    environment.update(
        {
            "GLM_GATE_D_PROJECTION_CONTRACTION_HLO_WRAPPER_SANITIZED": "1",
            "GLM_GATE_D_IMMUTABLE_LOCKS_HELD": "1",
            "GLM_GATE_D_LAUNCHER_PIN": pin,
            "GLM_GATE_D_LAUNCHER_SHA256": sha256(launcher_raw).hexdigest(),
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
