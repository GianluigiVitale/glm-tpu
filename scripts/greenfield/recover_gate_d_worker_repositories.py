#!/usr/bin/python3
"""Repository-only, descriptor-bound recovery for the Gate-D PP16 fleet.

This controller must be installed root-owned and invoked through a separately
reviewed empty-environment command.  It never imports JAX or touches TPU state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
CONTROLLER_RELATIVE = "scripts/greenfield/recover_gate_d_worker_repositories.py"
WORKER_RELATIVE = "scripts/greenfield/gate_d_worker_repository_transaction.py"
WRAPPER_RELATIVE = "scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh"
INVENTORY_RELATIVE = (
    "docs/artifacts/gate-d-worker-repository-recovered-prestate-inventory.json"
)
INVENTORY_SHA256 = "c4c3fd86111be9e7e0f5afd52e2cc5dab462fa9bf7b0cd710e75a4cd79a950cf"
PRESTATE_AUTHORITY_RELATIVE = (
    "docs/artifacts/gate-d-worker-repository-recovery-v4-complete.json"
)
PRESTATE_AUTHORITY_SHA256 = (
    "5f9e7ffc37e59a2d0d14879071c55b24ab1abceb7fb8f7fdb3caff364e5150d2"
)
PRESTATE_PIN = "dc9ec468ad1529b09c4dd06005849ee70e0fa7c7"
INSTALLED_CONTROLLER = Path(
    "/opt/glm-tpu/gate-d-worker-recovery-v5/recover_gate_d_worker_repositories.py"
)
LINKED_COMMON = Path("/home/gianl/glm-tpu/.git")
LINKED_GIT_DIR = LINKED_COMMON / "worktrees/glm-tpu-topology-rewrite"
POD = "db-v4-64-od"
ZONE = "us-central2-b"
BUCKET = "gs://driftbench-dsv4-uc"
LOCATION = "US-CENTRAL2"
GCLOUD = "/snap/bin/gcloud"
TAG_PATTERN = re.compile(r"gate_d_repo_recovery_[0-9]{8}T[0-9]{15}Z")
PIN_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
REMOTE_LOCK = "/opt/glm-tpu/locks/gate_d_repo_recovery.lock"
IMMUTABLE_LOCK_ROOT = Path("/opt/glm-tpu/locks")
IMMUTABLE_WORKLOAD_LOCK = "glm_pod_workload.lock"
IMMUTABLE_RSYNC_LOCK = "glm_tpu_rsync.lock"
LEGACY_WORKLOAD_LOCK_ROOT = Path("/home/gianl/glm-run")
LEGACY_WORKLOAD_LOCK = ".glm_pod_workload.lock"
LEGACY_RSYNC_LOCK_ROOT = Path("/home/gianl")
LEGACY_RSYNC_LOCK = ".glm-tpu-rsync.lock"
REMOTE_TIMEOUT_SECONDS = 180
LOCAL_REMOTE_TIMEOUT_SECONDS = 240
ALL_WORKERS = tuple(range(8))
RECOVERY_WORKERS = tuple(range(1, 8))
UNTOUCHED_WORKERS = (0,)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
RECOVERY_PRESTATE_PINS = {
    1: PRESTATE_PIN,
    2: PRESTATE_PIN,
    3: PRESTATE_PIN,
    4: PRESTATE_PIN,
    5: PRESTATE_PIN,
    6: PRESTATE_PIN,
    7: PRESTATE_PIN,
}
RECOVERY_PRESTATE_CONTRACTS = {
    1: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    2: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    3: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    4: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    5: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    6: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
    7: ("standalone", 0, EMPTY_SHA256, 0, EMPTY_SHA256),
}

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


class RecoveryError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_inventory_artifact(raw: bytes, authority_raw: bytes) -> None:
    if _sha256(raw) != INVENTORY_SHA256:
        raise RecoveryError("repository inventory hash mismatch")
    if _sha256(authority_raw) != PRESTATE_AUTHORITY_SHA256:
        raise RecoveryError("repository prestate authority hash mismatch")
    try:
        artifact = json.loads(raw)
        worker_records = artifact["workers"]
        if not isinstance(worker_records, list) or len(worker_records) != len(
            ALL_WORKERS
        ):
            raise RecoveryError("repository inventory worker count mismatch")
        workers = {int(record["worker"]): record for record in worker_records}
        authority = artifact["recovery_v4_authority"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RecoveryError("repository inventory schema mismatch") from error
    if (
        set(workers) != set(ALL_WORKERS)
        or workers[0].get("kind") != "linked_worktree"
        or artifact.get("promisor_record_format")
        != "git_object_dir_relative_path_nul_size_nul_sorted_v1"
        or artifact.get("mutation") is not False
        or artifact.get("hlo_work") is not False
        or artifact.get("tpu_work") is not False
        or authority.get("path") != PRESTATE_AUTHORITY_RELATIVE
        or authority.get("sha256") != PRESTATE_AUTHORITY_SHA256
        or authority.get("code_pin") != PRESTATE_PIN
    ):
        raise RecoveryError("repository inventory contract mismatch")
    for worker, expected_pin in RECOVERY_PRESTATE_PINS.items():
        record = workers[worker]
        layout, missing_count, missing_sha, promisor_count, promisor_sha = (
            RECOVERY_PRESTATE_CONTRACTS[worker]
        )
        expected_format = 1 if layout == "standalone_promisor" else 0
        if (
            record.get("kind") != layout
            or record.get("pin") != expected_pin
            or record.get("repository_format") != expected_format
            or record.get("missing_count") != missing_count
            or record.get("missing_sha256") != missing_sha
            or record.get("promisor_count") != promisor_count
            or record.get("promisor_sha256") != promisor_sha
        ):
            raise RecoveryError("repository inventory prestate mismatch")
        if layout == "standalone_promisor" and (
            record.get("promisor") != "true" or record.get("filter") != "blob:none"
        ):
            raise RecoveryError("repository inventory promisor mismatch")


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    pass_fds: tuple[int, ...] = (),
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=environment or _BASE_ENV,
        input=input_bytes,
        pass_fds=pass_fds,
        timeout=timeout,
    )
    if check and result.returncode:
        raise RecoveryError(
            f"command failed rc={result.returncode}: {shlex.join(command)}\n"
            + result.stderr.decode("utf-8", "replace")[-2000:]
        )
    return result


def _git(*arguments: str, timeout: int = 120) -> bytes:
    return _run(
        ["/usr/bin/git", *arguments], environment=_GIT_ENV, timeout=timeout
    ).stdout


def _open_regular(
    path: Path, *, required_parent_uid: int, required_parent_gid: int
) -> int:
    parent_fd = os.open(
        path.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    fd = -1
    try:
        parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != required_parent_uid
            or parent.st_gid != required_parent_gid
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise RecoveryError(f"unsafe regular-file parent: {path.parent}")
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        value = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or (value.st_dev, value.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RecoveryError(f"unsafe regular file: {path}")
        return fd
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        os.close(parent_fd)


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _validate_installed_source(
    expected_sha: str,
    *,
    running_path: Path | None = None,
    required_uid: int = 0,
    required_gid: int = 0,
) -> bytes:
    source_path = Path(__file__) if running_path is None else running_path
    if source_path.resolve() != INSTALLED_CONTROLLER:
        raise RecoveryError("controller must run from the reviewed /opt path")
    fd = _open_regular(
        INSTALLED_CONTROLLER,
        required_parent_uid=required_uid,
        required_parent_gid=required_gid,
    )
    try:
        value = os.fstat(fd)
        if (
            value.st_uid != required_uid
            or value.st_gid != required_gid
            or stat.S_IMODE(value.st_mode) != 0o555
        ):
            raise RecoveryError("installed controller is not root-owned mode 0555")
        raw = _read_fd(fd)
    finally:
        os.close(fd)
    if _sha256(raw) != expected_sha:
        raise RecoveryError("installed controller hash mismatch")
    return raw


def _validate_environment() -> tuple[str, str]:
    expected_names = {
        "GLM_GATE_D_REPO_RECOVERY",
        "GLM_GATE_D_REPO_RECOVERY_SANITIZED",
        "GLM_GATE_D_REPO_RECOVERY_SCRIPT_SHA256",
        "GLM_GATE_D_REPO_RECOVERY_TAG",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    if set(os.environ) != expected_names:
        raise RecoveryError(
            f"unexpected environment names: {sorted(set(os.environ) - expected_names)}"
        )
    if os.environ["GLM_GATE_D_REPO_RECOVERY"] != "1":
        raise RecoveryError("recovery is default-off")
    if os.environ["GLM_GATE_D_REPO_RECOVERY_SANITIZED"] != "1":
        raise RecoveryError("sanitized entry marker is absent")
    for name, expected in _BASE_ENV.items():
        if os.environ[name] != expected:
            raise RecoveryError(f"unexpected {name}")
    tag = os.environ["GLM_GATE_D_REPO_RECOVERY_TAG"]
    expected_sha = os.environ["GLM_GATE_D_REPO_RECOVERY_SCRIPT_SHA256"]
    if (
        TAG_PATTERN.fullmatch(tag) is None
        or SHA_PATTERN.fullmatch(expected_sha) is None
    ):
        raise RecoveryError("invalid tag or source hash")
    return tag, expected_sha


def _corroborate_lock(parent_fd: int, name: str, fd: int) -> None:
    held = os.fstat(fd)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise RecoveryError(f"lock pathname identity drift: {name}")


def _open_lock(
    parent: Path,
    name: str,
    *,
    required_uid: int,
    required_gid: int,
    required_mode: int,
    required_parent_uid: int,
    required_parent_gid: int,
    forbid_parent_group_other_write: bool,
) -> int:
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        parent_value = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_value.st_mode)
            or parent_value.st_uid != required_parent_uid
            or parent_value.st_gid != required_parent_gid
            or (
                forbid_parent_group_other_write
                and stat.S_IMODE(parent_value.st_mode) & 0o022
            )
        ):
            raise RecoveryError(f"unsafe lock parent identity: {parent}")
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        value = os.fstat(fd)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or value.st_uid != required_uid
            or value.st_gid != required_gid
            or stat.S_IMODE(value.st_mode) != required_mode
        ):
            raise RecoveryError(f"unsafe lock identity: {parent / name}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryError(f"lock held: {parent / name}") from error
        _corroborate_lock(parent_fd, name, fd)
        return fd
    except BaseException:
        if "fd" in locals():
            os.close(fd)
        raise
    finally:
        os.close(parent_fd)


class EvidenceDirectory:
    def __init__(self, tag: str, home: Path = Path("/home/gianl")):
        home_fd = os.open(
            home, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            try:
                os.mkdir("gate-d-repo-recovery", 0o700, dir_fd=home_fd)
            except FileExistsError:
                pass
            root_fd = os.open(
                "gate-d-repo-recovery",
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=home_fd,
            )
        finally:
            os.close(home_fd)
        root_stat = os.fstat(root_fd)
        if root_stat.st_uid != os.getuid() or stat.S_IMODE(root_stat.st_mode) != 0o700:
            os.close(root_fd)
            raise RecoveryError("unsafe recovery root")
        try:
            os.mkdir(tag, 0o700, dir_fd=root_fd)
            self.fd = os.open(
                tag,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
        finally:
            os.close(root_fd)
        self.closed = False
        self._directory_closed = False
        self.names: list[str] = []
        self._files: dict[str, int] = {}
        self._identities: dict[str, tuple[int, int, int, str]] = {}
        self._log_sealed = False
        self.log_fd = self._create("orchestrator.log")

    def _create(self, name: str, mode: int = 0o600) -> int:
        if self.closed or not name or "/" in name or name in {".", ".."}:
            raise RecoveryError("evidence writes are closed or unsafe")
        fd = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=self.fd,
        )
        self.names.append(name)
        self._files[name] = fd
        return fd

    def _seal(self, name: str) -> None:
        if name in self._identities:
            return
        fd = self._files[name]
        os.fsync(fd)
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RecoveryError(f"unsafe evidence identity at seal: {name}")
        raw = _read_fd(fd)
        self._identities[name] = (
            value.st_dev,
            value.st_ino,
            value.st_size,
            _sha256(raw),
        )

    def write(self, name: str, raw: bytes) -> None:
        fd = self._create(name)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        self._seal(name)

    def log(self, message: str) -> None:
        line = f"[gate-d-repo-recovery {time.strftime('%H:%M:%S', time.gmtime())}] {message}\n"
        print(line, end="", flush=True)
        if self.closed or self._log_sealed:
            return
        os.write(self.log_fd, line.encode())
        os.fsync(self.log_fd)

    def seal_log(self) -> None:
        if not self._log_sealed:
            self._seal("orchestrator.log")
            self._log_sealed = True

    def open_read(self, name: str) -> int:
        if name not in self._files or name not in self._identities:
            raise RecoveryError(f"evidence file is absent or unsealed: {name}")
        retained_fd = self._files[name]
        expected = self._identities[name]
        held = os.fstat(retained_fd)
        named = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or (held.st_dev, held.st_ino, held.st_size) != expected[:3]
            or (named.st_dev, named.st_ino) != expected[:2]
        ):
            raise RecoveryError(f"evidence pathname identity drift: {name}")
        fd = os.dup(retained_fd)
        try:
            if _sha256(_read_fd(fd)) != expected[3]:
                raise RecoveryError(f"evidence content identity drift: {name}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def close_writes(self) -> None:
        if not self.closed:
            self.seal_log()
            os.fsync(self.fd)
            self.closed = True

    def close(self) -> None:
        if self._directory_closed:
            return
        self.close_writes()
        for fd in self._files.values():
            os.close(fd)
        os.close(self.fd)
        self._directory_closed = True


def _extract_verifier(wrapper: bytes) -> str:
    text = wrapper.decode()
    start = (
        "read -r -d '' WORKER_REPO_VERIFY_SCRIPT <<'WORKER_REPO_VERIFY_EOF' || true\n"
    )
    if text.count(start) != 1 or text.count("\nWORKER_REPO_VERIFY_EOF") != 1:
        raise RecoveryError("worker verifier boundary mismatch")
    verifier = text.split(start, 1)[1].split("\nWORKER_REPO_VERIFY_EOF", 1)[0]
    if "fsck --connectivity-only --strict --no-dangling" not in verifier:
        raise RecoveryError("strict verifier primitive missing")
    return verifier


def _verify_local_repository(pin: str, verifier: str) -> bytes:
    result = _run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            verifier,
            "gate-d-recovery-local",
            pin,
            str(WORKTREE),
            ORIGIN,
            str(LINKED_COMMON),
            str(LINKED_GIT_DIR),
            "linked",
            "0",
            EMPTY_SHA256,
            "0",
            EMPTY_SHA256,
        ],
        environment=_GIT_ENV,
        timeout=180,
    )
    if result.stdout.count(b"SYNC_OK ") != 1:
        raise RecoveryError("local repository marker mismatch")
    return result.stdout


def _strict_census(tag: str, label: str) -> bytes:
    carrier = f"{tag}_{label}"
    ray_enum = (
        f"GLM_CENSUS_CARRIER={shlex.quote(carrier)} /home/gianl/vllm-env/bin/python -c "
        + shlex.quote(
            "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; "
            'carrier=os.environ["GLM_CENSUS_CARRIER"]; '
            'marked={p.pid for p in psutil.process_iter(["environ"]) if (p.info["environ"] or {}).get("GLM_CENSUS_CARRIER")==carrier}; '
            "me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; "
            'out={p.pid for p in psutil.process_iter(["name","cmdline"]) if p.pid not in skip and '
            'any(k in ((p.info["name"] or "") if f else subprocess.list2cmdline(p.info["cmdline"] or [])) for k,f in RAY_PROCESSES)}; '
            'print(" ".join(map(str,sorted(out))))'
        )
    )
    command = (
        "tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; "
        "command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; "
        f"ray_pids=$({ray_enum} 2>/dev/null); ray_rc=$?; "
        'generic=$(pgrep -af "[a]cquire_gate_d_compensated_pp16_hlo[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head" 2>/dev/null || true); '
        "holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); "
        'containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; '
        'if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; '
        'elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; '
        'then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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
            "--worker=all",
            f"--command={command}",
        ],
        timeout=200,
    )
    hosts = {
        line.split()[1]
        for line in result.stdout.decode("utf-8", "replace").splitlines()
        if line.startswith("CENSUS_OK ") and len(line.split()) == 2
    }
    if len(hosts) != 8:
        raise RecoveryError(f"census {label} is not authenticated 8/8")
    return result.stdout + result.stderr


def _remote_command(
    worker_source: str,
    phase: str,
    tag: str,
    worker: int,
    arguments: list[str],
) -> str:
    carrier = f"{tag}_{phase}_{worker}"
    command = [
        "/usr/bin/env",
        "-i",
        "HOME=/home/gianl",
        "LANG=C",
        "LC_ALL=C",
        "PATH=/usr/bin:/bin",
        f"GLM_GATE_D_REPO_RECOVERY_CARRIER={carrier}",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=15",
        str(REMOTE_TIMEOUT_SECONDS),
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        worker_source,
        phase,
        *arguments,
    ]
    return shlex.join(command)


def _parse_remote_record(raw: bytes, marker: str) -> dict[str, object]:
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
        raise RecoveryError(f"expected one {marker} record, found {len(records)}")
    return records[0]


def _run_worker(
    worker_source: str,
    phase: str,
    tag: str,
    worker: int,
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
) -> tuple[dict[str, object], bytes]:
    command = _remote_command(worker_source, phase, tag, worker, arguments)
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
            f"--command={command}",
        ],
        input_bytes=input_bytes,
        timeout=LOCAL_REMOTE_TIMEOUT_SECONDS,
    )
    marker = {
        "preflight": "REPO_PREFLIGHT_OK",
        "receive-prepare": "REPO_PREPARED",
        "swap": "REPO_SWAPPED",
        "final": "REPO_FINAL_OK",
    }[phase]
    record = _parse_remote_record(result.stdout, marker)
    return record, result.stdout + result.stderr


def _verify_remote_workers(
    verifier: str,
    pin: str,
    tag: str,
    workers: tuple[int, ...],
    label: str,
) -> bytes:
    records: dict[int, dict[str, object]] = {}
    output = bytearray()
    for worker in workers:
        carrier = f"{tag}_verify_{label}_{worker}"
        expected_layout = "linked" if worker in UNTOUCHED_WORKERS else "standalone"
        command = shlex.join(
            [
                "/usr/bin/env",
                "-i",
                "HOME=/home/gianl",
                "LANG=C",
                "LC_ALL=C",
                "PATH=/usr/bin:/bin",
                "GIT_CONFIG_GLOBAL=/dev/null",
                "GIT_CONFIG_NOSYSTEM=1",
                "GIT_NO_LAZY_FETCH=1",
                "GIT_NO_REPLACE_OBJECTS=1",
                "GIT_OPTIONAL_LOCKS=0",
                "GIT_PROTOCOL_FROM_USER=0",
                "GIT_TERMINAL_PROMPT=0",
                "GIT_SSH_COMMAND=/bin/false",
                f"GLM_GATE_D_REPO_RECOVERY_CARRIER={carrier}",
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=15",
                str(REMOTE_TIMEOUT_SECONDS),
                "/usr/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                verifier,
                "gate-d-recovery-fleet-verify",
                pin,
                str(WORKTREE),
                ORIGIN,
                str(LINKED_COMMON),
                str(LINKED_GIT_DIR),
                expected_layout,
                "0",
                EMPTY_SHA256,
                "0",
                EMPTY_SHA256,
            ]
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
                f"--command={command}",
            ],
            timeout=LOCAL_REMOTE_TIMEOUT_SECONDS,
        )
        markers = [
            line.split()
            for line in result.stdout.decode("utf-8", "replace").splitlines()
            if line.startswith("SYNC_OK ")
        ]
        if len(markers) != 1 or len(markers[0]) != 3 or markers[0][2] != pin:
            raise RecoveryError(f"{label}: worker {worker} verifier marker mismatch")
        records[worker] = {"host": markers[0][1], "pin": markers[0][2]}
        output.extend(result.stdout)
        output.extend(result.stderr)
    _ensure_worker_hosts(records, label, workers)
    return bytes(output)


def _quiescence_audit_code() -> str:
    return r"""import os
import stat
import sys


def fail(reason):
    print("RECOVERY_AUDIT_BAD " + os.uname().nodename + " " + reason, file=sys.stderr)
    raise SystemExit(2)


try:
    prefix = os.environ["PREFIX"].encode()
    lock_path = os.environ["LOCK"]
    required_uid = int(os.environ["REQUIRED_UID"])
    required_gid = int(os.environ["REQUIRED_GID"])
    required_mode = int(os.environ["REQUIRED_MODE"], 8)
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        held = os.fstat(lock_fd)
        named = os.lstat(lock_path)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or held.st_uid != required_uid
            or held.st_gid != required_gid
            or stat.S_IMODE(held.st_mode) != required_mode
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            fail("lock_identity")
        lock_identity = (os.major(held.st_dev), os.minor(held.st_dev), held.st_ino)
    finally:
        os.close(lock_fd)

    carrier = b"GLM_GATE_D_REPO_RECOVERY_CARRIER="
    pids = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open("/proc/" + entry.name + "/environ", "rb", buffering=0) as stream:
                environ = stream.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        values = [item[len(carrier):] for item in environ.split(b"\0") if item.startswith(carrier)]
        if any(value.startswith(prefix) for value in values):
            pids.append(int(entry.name))

    holders = []
    with open("/proc/locks", "r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            identity_index = next(
                (index for index, field in enumerate(fields) if field.count(":") == 2), None
            )
            if identity_index is None or identity_index == 0:
                fail("locks_parse_" + str(line_number))
            pieces = fields[identity_index].split(":")
            try:
                identity = (int(pieces[0], 16), int(pieces[1], 16), int(pieces[2], 10))
                holder = int(fields[identity_index - 1], 10)
            except (ValueError, IndexError):
                fail("locks_parse_" + str(line_number))
            if identity == lock_identity:
                holders.append(holder)

    host = os.uname().nodename
    if pids or holders:
        print(
            "RECOVERY_BUSY "
            + host
            + " pids="
            + ",".join(map(str, sorted(pids)))
            + " holders="
            + ",".join(map(str, sorted(holders)))
        )
        raise SystemExit(3)
    print("RECOVERY_QUIESCENT " + host)
except SystemExit:
    raise
except Exception as error:
    fail(type(error).__name__ + ":" + str(error))
"""


def _remote_quiescence_command(tag: str) -> str:
    return shlex.join(
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/env",
            "-i",
            f"PREFIX={tag}",
            f"LOCK={REMOTE_LOCK}",
            "REQUIRED_UID=0",
            "REQUIRED_GID=0",
            "REQUIRED_MODE=0666",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            _quiescence_audit_code(),
        ]
    )


def _remote_quiescence_once(tag: str) -> bytes:
    command = _remote_quiescence_command(tag)
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
            "--worker=all",
            f"--command={command}",
        ],
        timeout=200,
    )
    hosts = {
        line.split()[1]
        for line in result.stdout.decode("utf-8", "replace").splitlines()
        if line.startswith("RECOVERY_QUIESCENT ") and len(line.split()) == 2
    }
    if len(hosts) != 8:
        raise RecoveryError("remote recovery is not quiescent 8/8")
    return result.stdout + result.stderr


def _wait_for_remote_quiescence(tag: str) -> bytes:
    # This loop deliberately has no automatic fail-open deadline: local leases remain held until
    # every remote timeout/descendant and per-worker lock is privileged-audited as absent.
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, signal.SIG_IGN)
    try:
        failures = 0
        while True:
            try:
                return _remote_quiescence_once(tag)
            except (RecoveryError, subprocess.TimeoutExpired) as error:
                failures += 1
                print(
                    f"waiting for remote recovery quiescence attempt={failures}: {error}",
                    flush=True,
                )
                time.sleep(10)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _storage_location() -> str:
    return (
        _run(
            [
                GCLOUD,
                "storage",
                "buckets",
                "describe",
                BUCKET,
                "--format=value(location)",
            ],
            timeout=60,
        )
        .stdout.decode()
        .strip()
    )


def _object_generation_and_size(remote: str) -> tuple[str, int]:
    fields = (
        _run(
            [
                GCLOUD,
                "storage",
                "objects",
                "describe",
                remote,
                "--format=value(generation,size)",
            ],
            timeout=60,
        )
        .stdout.decode()
        .split()
    )
    if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
        raise RecoveryError(f"invalid object identity: {remote}")
    return fields[0], int(fields[1])


def _upload_fd(fd: int, remote: str) -> tuple[str, int, str]:
    raw = _read_fd(fd)
    _run(
        [
            GCLOUD,
            "storage",
            "cp",
            "--if-generation-match=0",
            f"/proc/self/fd/{fd}",
            remote,
        ],
        pass_fds=(fd,),
        timeout=180,
    )
    generation, size = _object_generation_and_size(remote)
    replay = _run(
        [GCLOUD, "storage", "cat", f"{remote}#{generation}"], timeout=180
    ).stdout
    if size != len(raw) or replay != raw:
        raise RecoveryError(f"remote replay mismatch: {remote}")
    return generation, size, _sha256(raw)


def _assert_exact_object_set(
    actual: set[str], expected: set[str], *, label: str
) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RecoveryError(
            f"{label} exact object-set mismatch missing={missing} unexpected={unexpected}"
        )


def _remote_object_set(remote_prefix: str) -> set[str]:
    lines = (
        _run([GCLOUD, "storage", "ls", f"{remote_prefix}/**"], timeout=60)
        .stdout.decode()
        .splitlines()
    )
    objects = {line for line in lines if line}
    if len(objects) != len([line for line in lines if line]):
        raise RecoveryError("remote object listing contains duplicate identities")
    return objects


def _upload_record(fd: int, remote: str) -> dict[str, object]:
    generation, size, digest = _upload_fd(fd, remote)
    return {
        "generation": generation,
        "remote": remote,
        "sha256": digest,
        "size": size,
    }


def _publish_diagnostic(evidence: EvidenceDirectory, remote_prefix: str) -> None:
    evidence.seal_log()
    for name in sorted(set(evidence.names)):
        fd = evidence.open_read(name)
        try:
            _upload_fd(fd, f"{remote_prefix}/diagnostic/{name}")
        finally:
            os.close(fd)


def _common_worker_arguments(
    pin: str,
    worktree_name: str,
    new_name: str,
    old_name: str,
    bundle_name: str,
    verifier: str,
) -> list[str]:
    return [
        "--pin",
        pin,
        "--worktree",
        worktree_name,
        "--new",
        new_name,
        "--old",
        old_name,
        "--bundle",
        bundle_name,
        "--origin",
        ORIGIN,
        "--linked-common",
        str(LINKED_COMMON),
        "--linked-git-dir",
        str(LINKED_GIT_DIR),
        "--target-layout",
        "standalone",
        "--target-missing-count",
        "0",
        "--target-missing-sha",
        EMPTY_SHA256,
        "--target-promisor-count",
        "0",
        "--target-promisor-sha",
        EMPTY_SHA256,
        "--verifier",
        verifier,
    ]


def _prestate_arguments(worker: int) -> list[str]:
    layout, missing_count, missing_sha, promisor_count, promisor_sha = (
        RECOVERY_PRESTATE_CONTRACTS[worker]
    )
    return [
        "--old-layout",
        layout,
        "--old-missing-count",
        str(missing_count),
        "--old-missing-sha",
        missing_sha,
        "--old-promisor-count",
        str(promisor_count),
        "--old-promisor-sha",
        promisor_sha,
    ]


def _record_arguments(record: dict[str, object]) -> list[str]:
    return [
        "--old-pin",
        str(record["old_pin"]),
        "--old-dev",
        str(record["old_dev"]),
        "--old-ino",
        str(record["old_ino"]),
        "--new-dev",
        str(record.get("new_dev", -1)),
        "--new-ino",
        str(record.get("new_ino", -1)),
    ]


def _ensure_worker_hosts(
    records: dict[int, dict[str, object]],
    marker: str,
    expected_workers: tuple[int, ...],
) -> None:
    if set(records) != set(expected_workers):
        raise RecoveryError(f"{marker}: worker set mismatch")
    hosts = {str(record.get("host", "")) for record in records.values()}
    if len(hosts) != len(expected_workers) or "" in hosts:
        raise RecoveryError(f"{marker}: host identity mismatch")


def _select_recovery_workers(
    preflight_records: dict[int, dict[str, object]], pin: str
) -> tuple[int, ...]:
    workers: list[int] = []
    for worker in RECOVERY_WORKERS:
        old_pin = preflight_records[worker].get("old_pin")
        if old_pin not in {RECOVERY_PRESTATE_PINS[worker], pin}:
            raise RecoveryError(f"worker {worker} has unauthorized repository prestate")
        if old_pin != pin:
            workers.append(worker)
    if not workers:
        raise RecoveryError("no stale recovery targets")
    return tuple(workers)


def _install_signal_handlers() -> None:
    def stop(signum: int, _frame: object) -> NoReturn:
        raise RecoveryError(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def main() -> int:
    tag, expected_source_sha = _validate_environment()
    installed_source = _validate_installed_source(expected_source_sha)
    _install_signal_handlers()

    immutable_workload_lock = _open_lock(
        IMMUTABLE_LOCK_ROOT,
        IMMUTABLE_WORKLOAD_LOCK,
        required_uid=0,
        required_gid=0,
        required_mode=0o666,
        required_parent_uid=0,
        required_parent_gid=0,
        forbid_parent_group_other_write=True,
    )
    immutable_rsync_lock = _open_lock(
        IMMUTABLE_LOCK_ROOT,
        IMMUTABLE_RSYNC_LOCK,
        required_uid=0,
        required_gid=0,
        required_mode=0o666,
        required_parent_uid=0,
        required_parent_gid=0,
        forbid_parent_group_other_write=True,
    )
    legacy_workload_lock = _open_lock(
        LEGACY_WORKLOAD_LOCK_ROOT,
        LEGACY_WORKLOAD_LOCK,
        required_uid=os.getuid(),
        required_gid=os.getgid(),
        required_mode=0o664,
        required_parent_uid=os.getuid(),
        required_parent_gid=os.getgid(),
        forbid_parent_group_other_write=False,
    )
    legacy_rsync_lock = _open_lock(
        LEGACY_RSYNC_LOCK_ROOT,
        LEGACY_RSYNC_LOCK,
        required_uid=os.getuid(),
        required_gid=os.getgid(),
        required_mode=0o664,
        required_parent_uid=os.getuid(),
        required_parent_gid=os.getgid(),
        forbid_parent_group_other_write=False,
    )
    evidence: EvidenceDirectory | None = None
    remote_started = False
    remote_owned = False
    terminal_started = False
    quiescence_done = False
    remote_prefix = f"{BUCKET}/results/greenfield/glm52/gate_d_repo_recovery/{tag}"
    try:
        evidence = EvidenceDirectory(tag)
        evidence.log("repository-only recovery source authentication")
        pin = (
            _git("-C", str(WORKTREE), "rev-parse", "--verify", "HEAD").decode().strip()
        )
        if PIN_PATTERN.fullmatch(pin) is None:
            raise RecoveryError("invalid local pin")
        if (
            _git("-C", str(WORKTREE), "branch", "--show-current").decode().strip()
            != BRANCH
        ):
            raise RecoveryError("wrong branch")
        _run(
            [
                "/usr/bin/git",
                "-C",
                str(WORKTREE),
                "diff-index",
                "--quiet",
                "HEAD",
                "--",
            ],
            environment=_GIT_ENV,
        )
        if _git(
            "-C", str(WORKTREE), "ls-files", "--others", "--exclude-standard"
        ).strip():
            raise RecoveryError("untracked worktree state")

        controller_blob = _git(
            "-C", str(WORKTREE), "show", f"{pin}:{CONTROLLER_RELATIVE}"
        )
        worker_blob = _git("-C", str(WORKTREE), "show", f"{pin}:{WORKER_RELATIVE}")
        wrapper_blob = _git("-C", str(WORKTREE), "show", f"{pin}:{WRAPPER_RELATIVE}")
        inventory_blob = _git(
            "-C", str(WORKTREE), "show", f"{pin}:{INVENTORY_RELATIVE}"
        )
        prestate_authority_blob = _git(
            "-C", str(WORKTREE), "show", f"{pin}:{PRESTATE_AUTHORITY_RELATIVE}"
        )
        if (
            controller_blob != installed_source
            or _sha256(controller_blob) != expected_source_sha
        ):
            raise RecoveryError("installed controller does not match committed blob")
        verifier = _extract_verifier(wrapper_blob)
        _validate_inventory_artifact(inventory_blob, prestate_authority_blob)
        local_verify = _verify_local_repository(pin, verifier)
        evidence.write("local_repo_verify.txt", local_verify)
        source_identity = {
            "controller_sha256": _sha256(controller_blob),
            "inventory_sha256": _sha256(inventory_blob),
            "pin": pin,
            "prestate_authority_sha256": _sha256(prestate_authority_blob),
            "recovery_workers": list(RECOVERY_WORKERS),
            "protected_linked_workers": list(UNTOUCHED_WORKERS),
            "verifier_sha256": _sha256(verifier.encode()),
            "worker_sha256": _sha256(worker_blob),
            "wrapper_sha256": _sha256(wrapper_blob),
        }
        evidence.write(
            "source_identity.json",
            (
                json.dumps(source_identity, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode(),
        )
        worker_source = worker_blob.decode()

        remote_pin = (
            _run(
                [
                    "/usr/bin/env",
                    "-i",
                    "HOME=/home/gianl",
                    "LANG=C",
                    "LC_ALL=C",
                    "PATH=/usr/bin:/bin",
                    "GIT_CONFIG_GLOBAL=/dev/null",
                    "GIT_CONFIG_NOSYSTEM=1",
                    "GIT_NO_LAZY_FETCH=1",
                    "GIT_NO_REPLACE_OBJECTS=1",
                    "GIT_OPTIONAL_LOCKS=0",
                    "GIT_PROTOCOL_FROM_USER=0",
                    "GIT_TERMINAL_PROMPT=0",
                    "GIT_SSH_COMMAND=/usr/bin/ssh",
                    "/usr/bin/git",
                    "ls-remote",
                    ORIGIN,
                    f"refs/heads/{BRANCH}",
                ],
                timeout=60,
            )
            .stdout.decode()
            .split()[0]
        )
        if remote_pin != pin:
            raise RecoveryError("origin pin mismatch")
        if _storage_location() != LOCATION:
            raise RecoveryError("bucket region mismatch")
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
            raise RecoveryError("pod is not READY")

        vacancy = _run(
            [GCLOUD, "storage", "ls", f"{remote_prefix}/**"], timeout=60, check=False
        )
        vacancy_raw = vacancy.stdout + vacancy.stderr
        evidence.write("remote_vacancy.txt", vacancy_raw)
        if vacancy.returncode != 1 or b"matched no objects" not in vacancy_raw:
            raise RecoveryError("remote prefix is not provably vacant")
        remote_owned = True

        census_pre = _strict_census(tag, "pre")
        evidence.write("census_pre.txt", census_pre)
        remote_started = True
        untouched_verify_pre = _verify_remote_workers(
            verifier,
            pin,
            tag,
            UNTOUCHED_WORKERS,
            "untouched_pre",
        )
        evidence.write("untouched_verify_pre.txt", untouched_verify_pre)

        evidence.log(
            "building one complete bundle into a descriptor-bound evidence file"
        )
        bundle_raw = _git(
            "-C", str(WORKTREE), "bundle", "create", "-", f"refs/heads/{BRANCH}"
        )
        bundle_sha = _sha256(bundle_raw)
        bundle_bytes = len(bundle_raw)
        evidence.write("repository.bundle", bundle_raw)
        bundle_fd = evidence.open_read("repository.bundle")
        try:
            bundle_verify = _run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(WORKTREE),
                    "bundle",
                    "verify",
                    f"/proc/self/fd/{bundle_fd}",
                ],
                environment=_GIT_ENV,
                pass_fds=(bundle_fd,),
                timeout=120,
            )
            evidence.write(
                "bundle_verify.txt", bundle_verify.stdout + bundle_verify.stderr
            )
            bundle_record = _upload_record(
                bundle_fd, f"{remote_prefix}/repository.bundle"
            )
        finally:
            os.close(bundle_fd)
        bundle_identity = {
            "bytes": bundle_bytes,
            "generation": bundle_record["generation"],
            "pin": pin,
            "remote": f"{remote_prefix}/repository.bundle",
            "sha256": bundle_sha,
        }
        evidence.write(
            "bundle_identity.json",
            (
                json.dumps(bundle_identity, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode(),
        )

        worktree_name = WORKTREE.name
        new_name = f"{worktree_name}.recovery-{tag}.new"
        old_name = f"{worktree_name}.pre-recovery-{tag}"
        bundle_name = f".{tag}.bundle"
        common = _common_worker_arguments(
            pin, worktree_name, new_name, old_name, bundle_name, verifier
        )
        preflight_records: dict[int, dict[str, object]] = {}
        preflight_output = bytearray()
        for worker in RECOVERY_WORKERS:
            arguments = [*common, *_prestate_arguments(worker)]
            record, raw = _run_worker(
                worker_source, "preflight", tag, worker, arguments
            )
            preflight_records[worker] = record
            preflight_output.extend(raw)
        _ensure_worker_hosts(preflight_records, "preflight", RECOVERY_WORKERS)
        recovery_workers = _select_recovery_workers(preflight_records, pin)
        evidence.write("preflight.txt", bytes(preflight_output))

        prepare_records: dict[int, dict[str, object]] = {}
        prepare_output = bytearray()
        for worker in recovery_workers:
            previous = preflight_records[worker]
            arguments = [
                *common,
                *_prestate_arguments(worker),
                "--bundle-sha",
                bundle_sha,
                "--bundle-bytes",
                str(bundle_bytes),
                *_record_arguments(previous),
            ]
            record, raw = _run_worker(
                worker_source,
                "receive-prepare",
                tag,
                worker,
                arguments,
                input_bytes=bundle_raw,
            )
            prepare_records[worker] = record
            prepare_output.extend(raw)
        _ensure_worker_hosts(prepare_records, "prepare", recovery_workers)
        evidence.write("prepare.txt", bytes(prepare_output))

        census_pre_swap = _strict_census(tag, "pre_swap")
        evidence.write("census_pre_swap.txt", census_pre_swap)

        swap_records: dict[int, dict[str, object]] = {}
        swap_output = bytearray()
        for worker in recovery_workers:
            previous = prepare_records[worker]
            arguments = [
                *common,
                *_prestate_arguments(worker),
                "--bundle-sha",
                bundle_sha,
                "--bundle-bytes",
                str(bundle_bytes),
                *_record_arguments(previous),
            ]
            record, raw = _run_worker(worker_source, "swap", tag, worker, arguments)
            swap_records[worker] = record
            swap_output.extend(raw)
        _ensure_worker_hosts(swap_records, "swap", recovery_workers)
        evidence.write("swap.txt", bytes(swap_output))

        final_records: dict[int, dict[str, object]] = {}
        final_output = bytearray()
        for worker in recovery_workers:
            previous = swap_records[worker]
            arguments = [
                *common,
                *_prestate_arguments(worker),
                *_record_arguments(previous),
            ]
            record, raw = _run_worker(worker_source, "final", tag, worker, arguments)
            final_records[worker] = record
            final_output.extend(raw)
        _ensure_worker_hosts(final_records, "final", recovery_workers)
        evidence.write("final.txt", bytes(final_output))

        fleet_verify_post = _verify_remote_workers(
            verifier,
            pin,
            tag,
            ALL_WORKERS,
            "fleet_post",
        )
        evidence.write("fleet_verify_post.txt", fleet_verify_post)

        census_post = _strict_census(tag, "post")
        evidence.write("census_post.txt", census_post)
        quiescence = _wait_for_remote_quiescence(tag)
        evidence.write("recovery_quiescence.txt", quiescence)
        quiescence_done = True

        evidence.log(
            "publishing generation-bound evidence; terminal is the final write"
        )
        evidence.seal_log()
        evidence_names = sorted(
            name
            for name in evidence.names
            if name
            not in {"repository.bundle", "remote_ledger.tsv", "RECOVERY_COMPLETE"}
        )
        uploaded_records: list[dict[str, object]] = [bundle_record]
        ledger_rows: list[str] = []
        for name in evidence_names:
            fd = evidence.open_read(name)
            try:
                record = _upload_record(fd, f"{remote_prefix}/{name}")
            finally:
                os.close(fd)
            uploaded_records.append(record)
            ledger_rows.append(
                f"{name}\t{record['sha256']}\t{record['size']}\t{record['generation']}\n"
            )
        ledger_raw = "".join(ledger_rows).encode()
        evidence.write("remote_ledger.tsv", ledger_raw)
        ledger_fd = evidence.open_read("remote_ledger.tsv")
        try:
            ledger_record = _upload_record(
                ledger_fd,
                f"{remote_prefix}/remote_ledger.tsv",
            )
        finally:
            os.close(ledger_fd)
        uploaded_records.append(ledger_record)
        preterminal_objects = sorted(
            uploaded_records, key=lambda item: str(item["remote"])
        )
        expected_preterminal = {str(item["remote"]) for item in preterminal_objects}
        _assert_exact_object_set(
            _remote_object_set(remote_prefix),
            expected_preterminal,
            label="preterminal",
        )
        terminal = {
            "bundle_sha256": bundle_sha,
            "hlo_work": False,
            "ledger_generation": ledger_record["generation"],
            "ledger_sha256": ledger_record["sha256"],
            "pin": pin,
            "preterminal_objects": preterminal_objects,
            "recovered_workers": list(recovery_workers),
            "recovery_candidates": list(RECOVERY_WORKERS),
            "status": "RECOVERY_COMPLETE",
            "tag": tag,
            "tpu_work": False,
            "untouched_workers": sorted(set(ALL_WORKERS) - set(recovery_workers)),
            "workers": 8,
        }
        terminal_raw = (
            json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        evidence.write("RECOVERY_COMPLETE", terminal_raw)
        terminal_fd = evidence.open_read("RECOVERY_COMPLETE")
        evidence.close_writes()
        terminal_started = True
        try:
            terminal_generation, terminal_size, terminal_sha = _upload_fd(
                terminal_fd, f"{remote_prefix}/RECOVERY_COMPLETE"
            )
        finally:
            os.close(terminal_fd)
        if terminal_size != len(terminal_raw) or terminal_sha != _sha256(terminal_raw):
            raise RecoveryError("terminal replay mismatch")
        expected_objects = expected_preterminal | {f"{remote_prefix}/RECOVERY_COMPLETE"}
        _assert_exact_object_set(
            _remote_object_set(remote_prefix),
            expected_objects,
            label="post-terminal",
        )
        if not terminal_generation.isdigit():
            raise RecoveryError("post-terminal generation identity is invalid")
        print(
            f"REPOSITORY_RECOVERY_COMPLETE pin={pin} archive={remote_prefix} old={old_name}",
            flush=True,
        )
        return 0
    except Exception as error:  # noqa: BLE001 - outer boundary seals all preterminal failures
        print(
            f"repository recovery failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        if evidence is not None and not terminal_started and not evidence.closed:
            if remote_started:
                quiescence = _wait_for_remote_quiescence(tag)
                evidence.write("recovery_quiescence_failure.txt", quiescence)
                quiescence_done = True
            evidence.write(
                "FAILURE.json",
                (
                    json.dumps(
                        {
                            "error": type(error).__name__ + ":" + str(error),
                            "status": "RECOVERY_FAILED",
                            "tag": tag,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode(),
            )
            if remote_owned:
                _publish_diagnostic(evidence, remote_prefix)
        return 1
    finally:
        if remote_started and not terminal_started and not quiescence_done:
            _wait_for_remote_quiescence(tag)
        if evidence is not None:
            evidence.close()
        os.close(legacy_rsync_lock)
        os.close(legacy_workload_lock)
        os.close(immutable_rsync_lock)
        os.close(immutable_workload_lock)


if __name__ == "__main__":
    raise SystemExit(main())
