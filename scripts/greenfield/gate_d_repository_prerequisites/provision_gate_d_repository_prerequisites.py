#!/usr/bin/python3
"""Provision only the immutable prerequisites for Gate-D repository recovery.

This default-off root-owned installed controller authenticates its committed source, holds both
legacy leases, creates and acquires the replacement immutable leases, installs
the exact recovery controller without replacement, provisions one distinct
recovery lock per worker, and atomically replaces the exact known cron text.
It never mutates a repository, imports JAX, or touches TPU state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
SELF_RELATIVE = (
    "scripts/greenfield/gate_d_repository_prerequisites/"
    "provision_gate_d_repository_prerequisites.py"
)
ROOT_HELPER_RELATIVE = (
    "scripts/greenfield/gate_d_repository_prerequisites/"
    "gate_d_repository_prerequisite_root.py"
)
CONTROLLER_RELATIVE = "scripts/greenfield/recover_gate_d_worker_repositories.py"
INSTALLED_ROOT = Path("/opt/glm-tpu/gate-d-repository-prerequisites-v3")
INSTALLED_SELF = INSTALLED_ROOT / "provision_gate_d_repository_prerequisites.py"
INSTALLED_ROOT_HELPER = INSTALLED_ROOT / "gate_d_repository_prerequisite_root.py"
EVIDENCE_ROOT = Path("/home/gianl/gate-d-runs")
RECOVERY_CONTROLLER_SHA256 = (
    "fd8950de504df483a3d86420b489406dbf93c3f68de04faf230bc85f402b2225"
)
RECOVERY_CONTROLLER_BYTES = 44013
POD = "db-v4-64-od"
ZONE = "us-central2-b"
GCLOUD = "/snap/bin/gcloud"
LOCK_ROOT = Path("/opt/glm-tpu/locks")
TAG_PATTERN = re.compile(r"gate_d_repo_prerequisite_[0-9]{8}T[0-9]{15}Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_OID_PATTERN = rb"[0-9a-f]{40}"
OLD_CRONTAB = (
    "*/5 * * * * flock -n /home/gianl/.glm-tpu-rsync.lock "
    "/home/gianl/bin/sync-glm.sh >> /home/gianl/sync-glm-tpu.log 2>&1\n"
)
NEW_CRONTAB = (
    "*/5 * * * * /usr/bin/flock -n /opt/glm-tpu/locks/glm_tpu_rsync.lock "
    "/usr/bin/flock -n /home/gianl/.glm-tpu-rsync.lock /home/gianl/bin/sync-glm.sh "
    ">> /home/gianl/sync-glm-tpu.log 2>&1\n"
)
_BASE_ENV = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin",
}
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
}


class ProvisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeldLock:
    parent_path: Path
    name: str
    parent_fd: int
    lock_fd: int
    parent_dev: int
    parent_ino: int
    lock_dev: int
    lock_ino: int
    uid: int
    gid: int
    mode: int
    parent_uid: int
    parent_gid: int
    forbid_parent_write: bool

    def close(self) -> None:
        os.close(self.lock_fd)
        os.close(self.parent_fd)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _validate_exact_directory(path: Path) -> tuple[int, os.stat_result]:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    value = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != 0
        or value.st_gid != 0
        or stat.S_IMODE(value.st_mode) != 0o755
        or os.listxattr(descriptor)
    ):
        os.close(descriptor)
        raise ProvisionError(f"unsafe installed directory: {path}")
    return descriptor, value


def _open_installed_file(parent_fd: int, name: str) -> tuple[int, bytes]:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
    )
    try:
        held = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or held.st_uid != 0
            or held.st_gid != 0
            or stat.S_IMODE(held.st_mode) != 0o755
            or os.listxattr(descriptor)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ProvisionError(f"unsafe installed file: {name}")
        return descriptor, _read_fd(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _open_installed_sources() -> tuple[int, bytes, int, bytes]:
    if (
        Path(os.path.abspath(__file__)) != INSTALLED_SELF
        or Path(os.path.realpath(__file__)) != INSTALLED_SELF
        or sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or sys.dont_write_bytecode is not True
    ):
        raise ProvisionError("invalid installed provisioner entry boundary")
    directory_fds: list[int] = []
    try:
        for directory in (
            Path("/opt"),
            Path("/opt/glm-tpu"),
            INSTALLED_ROOT,
        ):
            descriptor, _ = _validate_exact_directory(directory)
            directory_fds.append(descriptor)
        installed_value = os.fstat(directory_fds[-1])
        named_value = os.stat(INSTALLED_ROOT, follow_symlinks=False)
        if (installed_value.st_dev, installed_value.st_ino) != (
            named_value.st_dev,
            named_value.st_ino,
        ):
            raise ProvisionError("installed root pathname drift")
        cwd_fd = os.open(
            ".", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            cwd = os.fstat(cwd_fd)
            if (cwd.st_dev, cwd.st_ino) != (
                installed_value.st_dev,
                installed_value.st_ino,
            ):
                raise ProvisionError("working directory is not the installed root")
        finally:
            os.close(cwd_fd)
        self_fd, self_raw = _open_installed_file(directory_fds[-1], INSTALLED_SELF.name)
        try:
            helper_fd, helper_raw = _open_installed_file(
                directory_fds[-1], INSTALLED_ROOT_HELPER.name
            )
        except BaseException:
            os.close(self_fd)
            raise
        return self_fd, self_raw, helper_fd, helper_raw
    finally:
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=_BASE_ENV if environment is None else environment,
        input=input_bytes,
        timeout=timeout,
    )
    if result.returncode:
        rendered = []
        for argument in command:
            if len(argument) > 512:
                identity = hashlib.sha256(argument.encode()).hexdigest()
                rendered.append(
                    f"<redacted-long-argument bytes={len(argument.encode())} sha256={identity}>"
                )
            else:
                rendered.append(argument)
        raise ProvisionError(
            f"command failed rc={result.returncode}: {shlex.join(rendered)}\n"
            + result.stderr.decode("utf-8", "replace")[-2000:]
        )
    return result


def _git(*arguments: str, timeout: int = 120) -> bytes:
    return _run(
        ["/usr/bin/git", *arguments], environment=_GIT_ENV, timeout=timeout
    ).stdout


def _validate_environment() -> tuple[str, str, str, str]:
    expected_names = {
        "GLM_GATE_D_PREREQUISITE_PROVISION",
        "GLM_GATE_D_PREREQUISITE_TAG",
        "GLM_GATE_D_PREREQUISITE_SCRIPT_SHA256",
        "GLM_GATE_D_PREREQUISITE_ROOT_SHA256",
        "GLM_GATE_D_PREREQUISITE_CONTROLLER_SHA256",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    if set(os.environ) != expected_names:
        raise ProvisionError("unexpected provisioner environment")
    if os.environ["GLM_GATE_D_PREREQUISITE_PROVISION"] != "1":
        raise ProvisionError("prerequisite provisioning is default-off")
    for name, expected in _BASE_ENV.items():
        if os.environ[name] != expected:
            raise ProvisionError(f"unexpected {name}")
    tag = os.environ["GLM_GATE_D_PREREQUISITE_TAG"]
    script_sha = os.environ["GLM_GATE_D_PREREQUISITE_SCRIPT_SHA256"]
    root_sha = os.environ["GLM_GATE_D_PREREQUISITE_ROOT_SHA256"]
    controller_sha = os.environ["GLM_GATE_D_PREREQUISITE_CONTROLLER_SHA256"]
    if TAG_PATTERN.fullmatch(tag) is None or any(
        SHA256_PATTERN.fullmatch(value) is None
        for value in (script_sha, root_sha, controller_sha)
    ):
        raise ProvisionError("invalid tag or SHA-256")
    if controller_sha != RECOVERY_CONTROLLER_SHA256:
        raise ProvisionError("recovery controller SHA-256 is not allowlisted")
    return tag, script_sha, root_sha, controller_sha


def _open_lock(
    parent: Path,
    name: str,
    *,
    uid: int,
    gid: int,
    mode: int,
    parent_uid: int,
    parent_gid: int,
    forbid_parent_write: bool,
) -> HeldLock:
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    fd = -1
    try:
        parent_value = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_value.st_mode)
            or parent_value.st_uid != parent_uid
            or parent_value.st_gid != parent_gid
            or (forbid_parent_write and stat.S_IMODE(parent_value.st_mode) & 0o022)
        ):
            raise ProvisionError(f"unsafe lock parent: {parent}")
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        held = os.fstat(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or held.st_uid != uid
            or held.st_gid != gid
            or stat.S_IMODE(held.st_mode) != mode
            or held.st_size != 0
            or os.listxattr(fd)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ProvisionError(f"unsafe lock identity: {parent / name}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProvisionError(f"lock held: {parent / name}") from error
        lock = HeldLock(
            parent_path=parent,
            name=name,
            parent_fd=parent_fd,
            lock_fd=fd,
            parent_dev=parent_value.st_dev,
            parent_ino=parent_value.st_ino,
            lock_dev=held.st_dev,
            lock_ino=held.st_ino,
            uid=uid,
            gid=gid,
            mode=mode,
            parent_uid=parent_uid,
            parent_gid=parent_gid,
            forbid_parent_write=forbid_parent_write,
        )
        _revalidate_lock(lock)
        parent_fd = -1
        fd = -1
        return lock
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def _revalidate_lock(lock: HeldLock) -> None:
    parent = os.fstat(lock.parent_fd)
    held = os.fstat(lock.lock_fd)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or (parent.st_dev, parent.st_ino) != (lock.parent_dev, lock.parent_ino)
        or parent.st_uid != lock.parent_uid
        or parent.st_gid != lock.parent_gid
        or (lock.forbid_parent_write and stat.S_IMODE(parent.st_mode) & 0o022)
        or not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or (held.st_dev, held.st_ino) != (lock.lock_dev, lock.lock_ino)
        or held.st_uid != lock.uid
        or held.st_gid != lock.gid
        or stat.S_IMODE(held.st_mode) != lock.mode
        or held.st_size != 0
        or os.listxattr(lock.lock_fd)
    ):
        raise ProvisionError(
            f"held lock identity drift: {lock.parent_path / lock.name}"
        )
    path_parent_fd = os.open(
        lock.parent_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    path_lock_fd = -1
    try:
        path_parent = os.fstat(path_parent_fd)
        if (path_parent.st_dev, path_parent.st_ino) != (
            lock.parent_dev,
            lock.parent_ino,
        ):
            raise ProvisionError(f"lock parent pathname drift: {lock.parent_path}")
        path_lock_fd = os.open(
            lock.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=path_parent_fd,
        )
        path_lock = os.fstat(path_lock_fd)
        named = os.stat(lock.name, dir_fd=lock.parent_fd, follow_symlinks=False)
        if (path_lock.st_dev, path_lock.st_ino) != (lock.lock_dev, lock.lock_ino) or (
            named.st_dev,
            named.st_ino,
        ) != (lock.lock_dev, lock.lock_ino):
            raise ProvisionError(f"lock pathname drift: {lock.parent_path / lock.name}")
        fcntl.flock(lock.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if path_lock_fd >= 0:
            os.close(path_lock_fd)
        os.close(path_parent_fd)


def _revalidate_locks(locks: tuple[HeldLock, ...]) -> None:
    for lock in locks:
        _revalidate_lock(lock)


def _parse_record(raw: bytes, marker: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("marker") == marker:
            records.append(record)
    if len(records) != 1:
        raise ProvisionError(f"expected one {marker}, found {len(records)}")
    return records[0]


def _is_exact_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_exact_int_value(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _validate_lock_receipt(
    receipt: dict[str, object], expected_names: tuple[str, ...]
) -> str:
    if set(receipt) != {"host", "locks", "marker"}:
        raise ProvisionError("unexpected lock-receipt fields")
    host = receipt.get("host")
    locks = receipt.get("locks")
    if (
        receipt.get("marker") != "PREREQUISITE_LOCKS_OK"
        or not isinstance(host, str)
        or not host
        or not isinstance(locks, list)
        or len(locks) != len(expected_names)
    ):
        raise ProvisionError("invalid lock receipt")
    for record, expected_name in zip(locks, expected_names, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "dev",
            "gid",
            "ino",
            "mode",
            "name",
            "nlink",
            "size",
            "uid",
        }:
            raise ProvisionError("invalid lock record")
        if (
            record.get("name") != expected_name
            or record.get("mode") != "0666"
            or not _is_exact_int_value(record.get("uid"), 0)
            or not _is_exact_int_value(record.get("gid"), 0)
            or not _is_exact_int_value(record.get("nlink"), 1)
            or not _is_exact_int_value(record.get("size"), 0)
            or not _is_exact_int(record.get("dev"), minimum=1)
            or not _is_exact_int(record.get("ino"), minimum=1)
        ):
            raise ProvisionError("unsafe lock record")
    return host


def _validate_controller_receipt(
    receipt: dict[str, object], *, expected_sha: str, expected_bytes: int
) -> str:
    if set(receipt) != {"controller", "host", "marker"}:
        raise ProvisionError("unexpected controller-receipt fields")
    host = receipt.get("host")
    record = receipt.get("controller")
    if (
        receipt.get("marker") != "PREREQUISITE_CONTROLLER_OK"
        or not isinstance(host, str)
        or not host
        or not isinstance(record, dict)
        or set(record)
        != {"bytes", "dev", "gid", "ino", "mode", "name", "sha256", "uid"}
    ):
        raise ProvisionError("invalid controller receipt")
    if (
        not _is_exact_int_value(record.get("bytes"), expected_bytes)
        or not _is_exact_int_value(record.get("gid"), 0)
        or record.get("mode") != "0555"
        or record.get("name") != "recover_gate_d_worker_repositories.py"
        or record.get("sha256") != expected_sha
        or not _is_exact_int_value(record.get("uid"), 0)
        or not _is_exact_int(record.get("dev"), minimum=1)
        or not _is_exact_int(record.get("ino"), minimum=1)
    ):
        raise ProvisionError("unsafe controller record")
    return host


def _validate_fleet_receipts(
    remote_records: dict[int, dict[str, object]], local_host: str
) -> None:
    if set(remote_records) != set(range(8)):
        raise ProvisionError("remote lock provisioning is not authenticated 8/8")
    hosts = {
        _validate_lock_receipt(record, ("gate_d_repo_recovery.lock",))
        for record in remote_records.values()
    }
    if len(hosts) != 8 or remote_records[0].get("host") != local_host:
        raise ProvisionError("remote lock provisioning is not authenticated 8/8")


def _root_command(root_source: str, tag: str, arguments: list[str]) -> list[str]:
    return [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/env",
        "-i",
        f"GLM_GATE_D_PREREQUISITE_CARRIER={tag}",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        root_source,
        *arguments,
    ]


def _local_root(
    root_source: str,
    tag: str,
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
) -> tuple[dict[str, object], bytes]:
    command = _root_command(root_source, tag, arguments)
    result = _run(command, input_bytes=input_bytes, timeout=120)
    marker = (
        "PREREQUISITE_CONTROLLER_OK"
        if arguments[0] == "install-controller"
        else "PREREQUISITE_LOCKS_OK"
    )
    return _parse_record(result.stdout, marker), result.stdout + result.stderr


def _remote_locks(
    root_source: str, tag: str, worker: int
) -> tuple[dict[str, object], bytes]:
    remote = shlex.join(
        _root_command(
            root_source,
            tag,
            ["locks", "--lock", "gate_d_repo_recovery.lock"],
        )
    )
    result = _run(
        [
            GCLOUD,
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            POD,
            "--zone",
            ZONE,
            f"--worker={worker}",
            f"--command={remote}",
        ],
        timeout=180,
    )
    return (
        _parse_record(result.stdout, "PREREQUISITE_LOCKS_OK"),
        result.stdout + result.stderr,
    )


def _set_crontab(locks: tuple[HeldLock, ...]) -> tuple[str, str, str]:
    _revalidate_locks(locks)
    before = _run(["/usr/bin/crontab", "-l"], timeout=30).stdout.decode()
    _revalidate_locks(locks)
    if before == NEW_CRONTAB:
        disposition = "existing"
    elif before == OLD_CRONTAB:
        _run(["/usr/bin/crontab", "-"], input_bytes=NEW_CRONTAB.encode(), timeout=30)
        disposition = "updated"
    else:
        raise ProvisionError("crontab is not the exact reviewed old/new state")
    _revalidate_locks(locks)
    after = _run(["/usr/bin/crontab", "-l"], timeout=30).stdout.decode()
    _revalidate_locks(locks)
    if after != NEW_CRONTAB:
        raise ProvisionError("crontab replay mismatch")
    return before, after, disposition


def _write_evidence(
    tag: str,
    evidence: dict[str, object],
    *,
    artifact_root: Path | None = None,
) -> tuple[Path, dict[str, int | str]]:
    if TAG_PATTERN.fullmatch(tag) is None:
        raise ProvisionError("invalid evidence tag")
    exact_private_root = artifact_root is None
    root = EVIDENCE_ROOT if exact_private_root else artifact_root
    if Path(os.path.abspath(root)) != root or Path(os.path.realpath(root)) != root:
        raise ProvisionError("unsafe evidence root path")
    parent_fd = os.open(
        root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    parent = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_gid != os.getgid()
        or (exact_private_root and stat.S_IMODE(parent.st_mode) != 0o700)
        or stat.S_IMODE(parent.st_mode) & 0o022
        or os.listxattr(parent_fd)
    ):
        os.close(parent_fd)
        raise ProvisionError("unsafe evidence root identity")
    name = f"{tag}.json"
    path = root / name
    raw = (json.dumps(evidence, sort_keys=True, indent=2) + "\n").encode()
    fd = -1
    try:
        fd = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        held = os.fstat(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        replay = _read_fd(fd)
        digest = _sha256(replay)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or held.st_uid != os.getuid()
            or held.st_gid != os.getgid()
            or stat.S_IMODE(held.st_mode) != 0o600
            or held.st_size != len(raw)
            or digest != _sha256(raw)
            or os.listxattr(fd)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ProvisionError("evidence file identity drift")
        os.fsync(parent_fd)
        reopened_parent = os.open(
            root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            reopened = os.fstat(reopened_parent)
            if (reopened.st_dev, reopened.st_ino) != (
                parent.st_dev,
                parent.st_ino,
            ):
                raise ProvisionError("evidence root pathname drift")
        finally:
            os.close(reopened_parent)
        receipt: dict[str, int | str] = {
            "bytes": held.st_size,
            "dev": held.st_dev,
            "ino": held.st_ino,
            "mode": f"{stat.S_IMODE(held.st_mode):04o}",
            "sha256": digest,
        }
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)
    return path, receipt


def _validate_repository_sources(
    pin: str,
    *,
    installed_self: bytes,
    installed_helper: bytes,
    script_sha: str,
    root_sha: str,
    controller_sha: str,
) -> tuple[bytes, bytes, bytes]:
    if (
        _git("-C", str(WORKTREE), "rev-parse", "--verify", "HEAD").decode().strip()
        != pin
    ):
        raise ProvisionError("local pin drift")
    if _git("-C", str(WORKTREE), "branch", "--show-current").decode().strip() != BRANCH:
        raise ProvisionError("wrong branch")
    _run(
        ["/usr/bin/git", "-C", str(WORKTREE), "diff-index", "--quiet", "HEAD", "--"],
        environment=_GIT_ENV,
    )
    if _git("-C", str(WORKTREE), "ls-files", "--others", "--exclude-standard").strip():
        raise ProvisionError("untracked worktree state")
    script_blob = _git("-C", str(WORKTREE), "show", f"{pin}:{SELF_RELATIVE}")
    root_blob = _git("-C", str(WORKTREE), "show", f"{pin}:{ROOT_HELPER_RELATIVE}")
    controller_blob = _git("-C", str(WORKTREE), "show", f"{pin}:{CONTROLLER_RELATIVE}")
    if (
        script_blob != installed_self
        or root_blob != installed_helper
        or _sha256(script_blob) != script_sha
        or _sha256(root_blob) != root_sha
        or len(controller_blob) != RECOVERY_CONTROLLER_BYTES
        or _sha256(controller_blob) != controller_sha
    ):
        raise ProvisionError("committed/installed source identity mismatch")
    return script_blob, root_blob, controller_blob


def _remote_branch_pin() -> str:
    expected_ref = f"refs/heads/{BRANCH}"
    raw = _run(
        [
            "/usr/bin/env",
            "-i",
            "HOME=/home/gianl",
            "LANG=C",
            "LC_ALL=C",
            "PATH=/usr/bin:/bin",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_TERMINAL_PROMPT=0",
            "GIT_SSH_COMMAND=/usr/bin/ssh",
            "/usr/bin/git",
            "ls-remote",
            ORIGIN,
            expected_ref,
        ],
        timeout=60,
    ).stdout
    exact_record = re.compile(
        b"("
        + GIT_OID_PATTERN
        + b")\t"
        + re.escape(expected_ref.encode("ascii"))
        + b"\n"
    )
    match = exact_record.fullmatch(raw)
    if match is None:
        raise ProvisionError(
            f"invalid origin pin response bytes={len(raw)} sha256={_sha256(raw)}"
        )
    return match.group(1).decode("ascii")


def _lock_evidence(lock: HeldLock) -> dict[str, int | str]:
    _revalidate_lock(lock)
    return {
        "dev": lock.lock_dev,
        "ino": lock.lock_ino,
        "mode": f"{lock.mode:04o}",
        "name": str(lock.parent_path / lock.name),
        "parent_dev": lock.parent_dev,
        "parent_ino": lock.parent_ino,
    }


def main() -> int:
    installed_self_fd = -1
    installed_helper_fd = -1
    held_locks: list[HeldLock] = []
    try:
        (
            installed_self_fd,
            installed_self,
            installed_helper_fd,
            installed_helper,
        ) = _open_installed_sources()
        tag, script_sha, root_sha, controller_sha = _validate_environment()
        pin = (
            _git("-C", str(WORKTREE), "rev-parse", "--verify", "HEAD").decode().strip()
        )
        _, _, controller_blob = _validate_repository_sources(
            pin,
            installed_self=installed_self,
            installed_helper=installed_helper,
            script_sha=script_sha,
            root_sha=root_sha,
            controller_sha=controller_sha,
        )
        if _remote_branch_pin() != pin:
            raise ProvisionError("origin pin mismatch")
        pod_state = (
            _run(
                [
                    GCLOUD,
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "describe",
                    POD,
                    "--zone",
                    ZONE,
                    "--format=value(state)",
                ],
                timeout=60,
            )
            .stdout.decode()
            .strip()
        )
        if pod_state != "READY":
            raise ProvisionError("pod is not READY")

        held_locks.append(
            _open_lock(
                Path("/home/gianl/glm-run"),
                ".glm_pod_workload.lock",
                uid=os.getuid(),
                gid=os.getgid(),
                mode=0o664,
                parent_uid=os.getuid(),
                parent_gid=os.getgid(),
                forbid_parent_write=False,
            )
        )
        held_locks.append(
            _open_lock(
                Path("/home/gianl"),
                ".glm-tpu-rsync.lock",
                uid=os.getuid(),
                gid=os.getgid(),
                mode=0o664,
                parent_uid=os.getuid(),
                parent_gid=os.getgid(),
                forbid_parent_write=False,
            )
        )
        root_source = installed_helper.decode()
        local_lock_receipt, local_lock_output = _local_root(
            root_source,
            tag,
            [
                "locks",
                "--lock",
                "glm_pod_workload.lock",
                "--lock",
                "glm_tpu_rsync.lock",
                "--lock",
                "gate_d_repo_recovery.lock",
            ],
        )
        local_host = _validate_lock_receipt(
            local_lock_receipt,
            (
                "glm_pod_workload.lock",
                "glm_tpu_rsync.lock",
                "gate_d_repo_recovery.lock",
            ),
        )
        held_locks.append(
            _open_lock(
                LOCK_ROOT,
                "glm_pod_workload.lock",
                uid=0,
                gid=0,
                mode=0o666,
                parent_uid=0,
                parent_gid=0,
                forbid_parent_write=True,
            )
        )
        held_locks.append(
            _open_lock(
                LOCK_ROOT,
                "glm_tpu_rsync.lock",
                uid=0,
                gid=0,
                mode=0o666,
                parent_uid=0,
                parent_gid=0,
                forbid_parent_write=True,
            )
        )
        locks = tuple(held_locks)
        _revalidate_locks(locks)
        controller_record, controller_output = _local_root(
            root_source,
            tag,
            ["install-controller"],
            input_bytes=controller_blob,
        )
        if (
            _validate_controller_receipt(
                controller_record,
                expected_sha=controller_sha,
                expected_bytes=len(controller_blob),
            )
            != local_host
        ):
            raise ProvisionError("local privileged host identity drift")
        remote_records: dict[int, dict[str, object]] = {}
        remote_output = bytearray()
        for worker in range(8):
            record, raw = _remote_locks(root_source, tag, worker)
            remote_records[worker] = record
            remote_output.extend(raw)
        _validate_fleet_receipts(remote_records, local_host)
        _validate_repository_sources(
            pin,
            installed_self=_read_fd(installed_self_fd),
            installed_helper=_read_fd(installed_helper_fd),
            script_sha=script_sha,
            root_sha=root_sha,
            controller_sha=controller_sha,
        )
        if _remote_branch_pin() != pin:
            raise ProvisionError("origin pin drift before cron adoption")
        cron_before, cron_after, cron_disposition = _set_crontab(locks)
        evidence = {
            "controller": controller_record,
            "controller_output_sha256": _sha256(controller_output),
            "controller_sha256": controller_sha,
            "cron_after": cron_after,
            "cron_before_sha256": _sha256(cron_before.encode()),
            "cron_disposition": cron_disposition,
            "hlo_work": False,
            "local_lock_output_sha256": _sha256(local_lock_output),
            "local_locks": local_lock_receipt,
            "held_lock_identities": [_lock_evidence(lock) for lock in locks],
            "installed_root": str(INSTALLED_ROOT),
            "installed_root_helper_dev": os.fstat(installed_helper_fd).st_dev,
            "installed_root_helper_ino": os.fstat(installed_helper_fd).st_ino,
            "installed_self_dev": os.fstat(installed_self_fd).st_dev,
            "installed_self_ino": os.fstat(installed_self_fd).st_ino,
            "pin": pin,
            "remote_lock_output_sha256": _sha256(bytes(remote_output)),
            "remote_locks": remote_records,
            "root_helper_sha256": root_sha,
            "script_sha256": script_sha,
            "status": "PREREQUISITES_PROVISIONED",
            "tag": tag,
            "threat_boundary": "same_uid_is_trusted_passwordless_sudo_administrator",
            "tpu_work": False,
            "workers": 8,
        }
        evidence_path, evidence_receipt = _write_evidence(tag, evidence)
        _revalidate_locks(locks)
        print(
            "PREREQUISITES_PROVISIONED "
            f"pin={pin} evidence={evidence_path} "
            f"evidence_sha256={evidence_receipt['sha256']}",
            flush=True,
        )
        return 0
    finally:
        for lock in reversed(held_locks):
            lock.close()
        if installed_helper_fd >= 0:
            os.close(installed_helper_fd)
        if installed_self_fd >= 0:
            os.close(installed_self_fd)


if __name__ == "__main__":
    raise SystemExit(main())
