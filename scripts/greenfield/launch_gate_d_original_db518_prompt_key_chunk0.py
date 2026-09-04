#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Root-owned launcher for one bounded original DB518 prompt-key chunk-0 probe."""

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
    "/opt/glm-tpu/bin/launch_gate_d_original_db518_prompt_key_chunk0_v10.py")
PYTHON = Path("/usr/bin/python3.10")
PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
SEALED_PYTHON = Path(
    "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12")
SEALED_PYTHON_SHA256 = (
    "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7")
WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
SOURCE_PATH = "scripts/greenfield/launch_gate_d_original_db518_prompt_key_chunk0.py"
WRAPPER_PATH = WORKTREE / "scripts/greenfield/run_original_db518_prompt_key_chunk0.sh"
WRAPPER_SOURCE_PATH = "scripts/greenfield/run_original_db518_prompt_key_chunk0.sh"
WRAPPER_SHA256 = "9d14c4e5d838908b9af43c79f0f329b639e2177d00357b5ca587ea1367837046"
CAPSULE_ROOT = Path(
    "/usr/local/libexec/glm-tpu/gate-d-original-db518-prompt-key-v10")
HLO_CONTRACT_PATH = CAPSULE_ROOT / "original_db518_prompt_key.py"
HLO_CONTRACT_SOURCE_PATH = "glm_tpu/greenfield/validation/original_db518_prompt_key.py"
HLO_CONTRACT_SHA256 = "e86196b39acd075deca1abbfb645ff3264307158b50ce89fcfe60e26c75be95c"
PARSER_CONTRACT_PATH = CAPSULE_ROOT / "chunk0_embedding_hlo.py"
PARSER_CONTRACT_SOURCE_PATH = "glm_tpu/greenfield/validation/chunk0_embedding_hlo.py"
PARSER_CONTRACT_SHA256 = "e239c20b1a206061c9116726421343d81d5ff989be5e8f8440d5c59106eb9757"
BOUNDARY_CONTRACT_PATH = CAPSULE_ROOT / "original_db518_normalized_boundary_hlo.py"
BOUNDARY_CONTRACT_SOURCE_PATH = (
    "glm_tpu/greenfield/validation/original_db518_normalized_boundary_hlo.py")
BOUNDARY_CONTRACT_SHA256 = "35757aab4a616a2f1073f78503c29075c3e43cb684e4e1faf7d835630876809e"
PROBE_PATH = CAPSULE_ROOT / "probe_original_db518_prompt_key_chunk0.py"
PROBE_SOURCE_PATH = "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py"
PROBE_SHA256 = "a64b8466e2507ee5376464fff4469f287a7c2575f8e33603a5fcf4e2b267f8ea"
PUBLISHER_PATH = CAPSULE_ROOT / "publish_gate_d_original_db518_prompt_key_chunk0.py"
PUBLISHER_SOURCE_PATH = "scripts/greenfield/publish_gate_d_original_db518_prompt_key_chunk0.py"
PUBLISHER_SHA256 = "ff4f5a49ed47018a8bbbd4fed58ac18c7a824ced41a73476779e51cfac511ed5"
MIRROR_VERIFIER_PATH = CAPSULE_ROOT / "verify_gate_d_original_db518_same_region_git_mirror.py"
MIRROR_VERIFIER_SOURCE_PATH = "scripts/greenfield/verify_gate_d_original_db518_same_region_git_mirror.py"
MIRROR_VERIFIER_SHA256 = "c8fccad494bf732824973606687b2c6c77dbaa8c3868b356aefe96982a71d114"
LOCK_ROOT = Path("/opt/glm-tpu/locks")
LOCK_NAMES = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")
LOCK_FDS = (11, 12)
WRAPPER_FD = 10
RUN_FD = 7
CHILDREN = (
    (
        "PARSER_CONTRACT",
        PARSER_CONTRACT_PATH,
        PARSER_CONTRACT_SOURCE_PATH,
        PARSER_CONTRACT_SHA256,
    ),
    (
        "BOUNDARY_CONTRACT",
        BOUNDARY_CONTRACT_PATH,
        BOUNDARY_CONTRACT_SOURCE_PATH,
        BOUNDARY_CONTRACT_SHA256,
    ),
    (
        "HLO_CONTRACT",
        HLO_CONTRACT_PATH,
        HLO_CONTRACT_SOURCE_PATH,
        HLO_CONTRACT_SHA256,
    ),
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
    r"greenfield_original_db518_prompt_key_chunk0_[0-9]{8}T[0-9]{15}Z")
INPUT_ENVIRONMENT_KEYS = {
    "GLM_GATE_D_ORIGINAL_DB518_CHUNK0",
    "GLM_GATE_D_ORIGINAL_DB518_CHUNK0_MODE",
    "GLM_GATE_D_ORIGINAL_DB518_CHUNK0_TAG",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
}
EXPECTED_FIXED_ENVIRONMENT = {
    "GLM_GATE_D_ORIGINAL_DB518_CHUNK0": "1",
    "GLM_GATE_D_ORIGINAL_DB518_CHUNK0_MODE": "execute_once",
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
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or (expected_uid is not None and before.st_uid != expected_uid)
                or (expected_gid is not None and before.st_gid != expected_gid)
                or (expected_mode is not None
                    and stat.S_IMODE(before.st_mode) != expected_mode)
                or os.listxattr(path, follow_symlinks=False)):
            raise RuntimeError(f"unsafe protected source identity: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (len(raw) != before.st_size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                     after.st_mtime_ns, after.st_ctime_ns) or
            (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)):
            raise RuntimeError(
                f"protected source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["/usr/bin/git", "-C",
             str(WORKTREE), *arguments],
            env=GIT_ENVIRONMENT,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "launcher Git identity verification failed") from error


def _require_root_boundary(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe root-owned launcher path: {path}")
    for parent in (path.parent, *path.parent.parents):
        value = os.stat(parent, follow_symlinks=False)
        if (not stat.S_ISDIR(value.st_mode) or value.st_uid != 0
                or value.st_gid != 0 or stat.S_IMODE(value.st_mode) & 0o022):
            raise RuntimeError(f"unsafe root-owned launcher parent: {parent}")


def _create_sealed_wrapper(raw: bytes) -> int:
    descriptor = os.memfd_create(
        "glm-gate-d-original-db518-chunk0-wrapper",
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


def _preflight_publisher(pin: str) -> None:
    """Exercise the exact isolated publisher before a tag directory exists."""

    _require_root_boundary(SEALED_PYTHON)
    if sha256(
            _read_stable_regular(
                SEALED_PYTHON,
                expected_uid=0,
                expected_gid=0,
                expected_mode=0o755)).hexdigest() != SEALED_PYTHON_SHA256:
        raise RuntimeError("publisher preflight Python runtime drifted")
    result = subprocess.run(
        [
            str(SEALED_PYTHON),
            "-I",
            "-S",
            "-B",
            str(PUBLISHER_PATH),
            "--expected-code-hash",
            pin,
            "--expected-source-sha256",
            PUBLISHER_SHA256,
            "preflight",
        ],
        cwd="/",
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if (result.returncode != 0 or result.stdout != b"PUBLISHER_PREFLIGHT_OK\n"
            or result.stderr):
        raise RuntimeError(
            "publisher isolated-runtime preflight failed: " +
            result.stderr.decode("utf-8", errors="replace"))


def _preflight_probe(pin: str) -> None:
    """Run the probe's own immutable-source check before creating its tag."""

    program = (
        "import runpy\n"
        "from pathlib import Path\n"
        f"module = runpy.run_path({str(PROBE_PATH)!r})\n"
        "observed = module['_verify_running_source']("
        f"Path({str(WORKTREE)!r}), {pin!r}, {PROBE_SHA256!r})\n"
        f"if observed != {PROBE_SHA256!r}:\n"
        "    raise RuntimeError('probe preflight digest drifted')\n"
        "print('PROBE_PREFLIGHT_OK')\n"
    )
    result = subprocess.run(
        [str(SEALED_PYTHON), "-I", "-S", "-B", "-c", program],
        cwd="/",
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if (result.returncode != 0 or result.stdout != b"PROBE_PREFLIGHT_OK\n"
            or result.stderr):
        raise RuntimeError(
            "probe immutable-source preflight failed: " +
            result.stderr.decode("utf-8", errors="replace"))


def _open_locked_fds() -> list[int]:
    parent_fd = os.open(
        LOCK_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptors: list[int] = []
    try:
        parent = os.fstat(parent_fd)
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0
                or parent.st_gid != 0 or stat.S_IMODE(parent.st_mode) & 0o022):
            raise RuntimeError("unsafe immutable lock parent")
        for name in LOCK_NAMES:
            descriptor = os.open(name,
                                 os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                 dir_fd=parent_fd)
            held = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (not stat.S_ISREG(held.st_mode) or held.st_nlink != 1
                    or held.st_uid != 0 or held.st_gid != 0
                    or stat.S_IMODE(held.st_mode) != 0o666 or
                (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)):
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
        RUN_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = -1
    try:
        root = os.fstat(root_fd)
        if (not stat.S_ISDIR(root.st_mode)
                or stat.S_IMODE(root.st_mode) != 0o700
                or root.st_uid != os.geteuid() or root.st_gid != os.getegid()
                or os.listxattr(RUN_ROOT, follow_symlinks=False)):
            raise RuntimeError("unsafe private chunk-0 run root")
        os.mkdir(tag, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        descriptor = os.open(
            tag,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        held = os.fstat(descriptor)
        named = os.stat(tag, dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(held.st_mode) or held.st_uid != os.geteuid()
                or held.st_gid != os.getegid()
                or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)):
            raise RuntimeError("new chunk-0 run-directory identity drifted")
        os.fchmod(descriptor, 0o700)
        os.mkdir("hlo", mode=0o700, dir_fd=descriptor)
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
        raise RuntimeError(
            "launcher must run from its root-owned installation")
    if (Path(sys.executable).resolve() != PYTHON
            or sys.argv != [str(INSTALL_PATH)] or sys.flags.isolated != 1
            or sys.flags.no_site != 1 or not sys.flags.ignore_environment
            or not sys.flags.dont_write_bytecode or sha256(
                _read_stable_regular(
                    PYTHON,
                    expected_uid=0,
                    expected_gid=0,
                    expected_mode=0o755)).hexdigest() != PYTHON_SHA256):
        raise RuntimeError("launcher Python runtime drifted")
    if set(os.environ) != INPUT_ENVIRONMENT_KEYS:
        raise RuntimeError("launcher environment names are not exact")
    if any(
            os.environ.get(key) != value
            for key, value in EXPECTED_FIXED_ENVIRONMENT.items()):
        raise RuntimeError("launcher environment values drifted")
    tag = os.environ["GLM_GATE_D_ORIGINAL_DB518_CHUNK0_TAG"]
    if TAG_PATTERN.fullmatch(tag) is None:
        raise RuntimeError("launcher tag is invalid")

    _require_root_boundary(INSTALL_PATH)
    launcher_raw = _read_stable_regular(INSTALL_PATH,
                                        expected_uid=0,
                                        expected_gid=0,
                                        expected_mode=0o555)
    pin = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if PIN_PATTERN.fullmatch(pin) is None:
        raise RuntimeError("launcher code pin is invalid")
    if (_git("rev-parse", "--show-toplevel") != f"{WORKTREE}\n".encode()
            or _git("branch", "--show-current") != f"{BRANCH}\n".encode()
            or _git("status", "--porcelain=v1", "--untracked-files=all")
            or _git("for-each-ref", "--format=%(refname)", "refs/replace")
            or _git("remote", "get-url", "origin") != f"{ORIGIN}\n".encode()
            or launcher_raw != _git("show", f"{pin}:{SOURCE_PATH}")):
        raise RuntimeError("launcher repository authority drifted")
    wrapper_raw = _read_stable_regular(WRAPPER_PATH)
    if sha256(
            wrapper_raw).hexdigest() != WRAPPER_SHA256 or wrapper_raw != _git(
                "show", f"{pin}:{WRAPPER_SOURCE_PATH}"):
        raise RuntimeError(
            "protected wrapper is not the expected committed blob")
    for label, path, repository_path, expected_sha256 in CHILDREN:
        _require_root_boundary(path)
        raw = _read_stable_regular(path,
                                   expected_uid=0,
                                   expected_gid=0,
                                   expected_mode=0o555)
        if sha256(raw).hexdigest() != expected_sha256 or raw != _git(
                "show", f"{pin}:{repository_path}"):
            raise RuntimeError(
                f"protected {label.lower()} is not the expected committed blob"
            )
    _preflight_publisher(pin)
    _preflight_probe(pin)
    wrapper_fd = _create_sealed_wrapper(wrapper_raw)
    lock_fds = _open_locked_fds()
    run_fd = _create_retained_run_fd(tag)
    _bind_inherited_fd(run_fd, RUN_FD)
    _bind_inherited_fd(wrapper_fd, WRAPPER_FD)
    for descriptor, target in zip(lock_fds, LOCK_FDS, strict=True):
        _bind_inherited_fd(descriptor, target)
    environment = dict(os.environ)
    environment.update({
        "GLM_GATE_D_IMMUTABLE_LOCKS_HELD":
        "1",
        "GLM_GATE_D_LAUNCHER_PIN":
        pin,
        "GLM_GATE_D_LAUNCHER_SHA256":
        sha256(launcher_raw).hexdigest(),
        "GLM_GATE_D_PROBE_WRAPPER_SANITIZED":
        "1",
        "GLM_GATE_D_WRAPPER_MEMFD":
        str(WRAPPER_FD),
        "GLM_GATE_D_WRAPPER_SHA256":
        WRAPPER_SHA256,
    })
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
