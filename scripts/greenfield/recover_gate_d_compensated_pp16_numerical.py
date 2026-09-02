#!/usr/bin/env python3
"""Seal the completed compensated PP16 rejection without rerunning a TPU graph."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import google_crc32c

from glm_tpu.greenfield.benchmarking.gate_d_compensated_pp16_recovery import (
    SOURCE_REMOTE,
    SOURCE_TAG,
    authenticate_compensated_pp16_rejection,
    canonical_json,
    require_eight_host_census_bytes,
    validate_recovery_remote_vacancy,
)

POD = "db-v4-64-od"
ZONE = "us-central2-b"
WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
BRANCH = "tooling/gate-d-compensated-pp16-numerical"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
BUCKET = "gs://driftbench-dsv4-uc"
LOCATION = "US-CENTRAL2"
RUN_ROOT = Path("/home/gianl/gate-d-runs")
SOURCE_RUN_DIR = RUN_ROOT / SOURCE_TAG
CAPSULE_ROOT = RUN_ROOT / "greenfield_gate_d_compensated_capsule_20260831T124838Z"
CAPSULE_AUTHORITY = RUN_ROOT / (
    "greenfield_gate_d_compensated_capsule_20260831T124838Z-execution-authority.json"
)
SOURCE_PATH = "scripts/greenfield/recover_gate_d_compensated_pp16_numerical.py"
MODULE_PATH = "glm_tpu/greenfield/benchmarking/gate_d_compensated_pp16_recovery.py"
MIRROR_VERIFIER_PATH = "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
REMOTE_ROOT = f"{BUCKET}/results/greenfield/glm52/gate_d_pp16_numerical_recovery"
TAG_PATTERN = re.compile(r"gate_d_compensated_pp16_recovery_[0-9]{8}T[0-9]{15}Z")
VACANT_STDERR = b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
GCLOUD_ENV = {
    "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin",
    "PYTHONWARNINGS": "ignore",
}
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/usr/bin/ssh -oBatchMode=yes -oConnectTimeout=30",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}

HOME_AUTHORITY = (2049, 1_591_959, 2001, 2001, 0o750)
RUN_ROOT_AUTHORITY = (2049, 2_583_086, 2001, 2001, 0o700)
LOCK_AUTHORITIES = (
    (
        "/opt/glm-tpu/locks",
        2049,
        49_168,
        0,
        0,
        0o755,
        "glm_pod_workload.lock",
        2049,
        49_169,
        0,
        0,
        0o666,
    ),
    (
        "/opt/glm-tpu/locks",
        2049,
        49_168,
        0,
        0,
        0o755,
        "glm_tpu_rsync.lock",
        2049,
        49_170,
        0,
        0,
        0o666,
    ),
    (
        "/home/gianl/glm-run",
        2049,
        1_868_609,
        2001,
        2001,
        0o775,
        ".glm_pod_workload.lock",
        2049,
        1_868_606,
        2001,
        2001,
        0o664,
    ),
    (
        "/home/gianl",
        2049,
        1_591_959,
        2001,
        2001,
        0o750,
        ".glm-tpu-rsync.lock",
        2049,
        1_580_034,
        2001,
        2001,
        0o664,
    ),
)


class RecoveryError(RuntimeError):
    """Raised when append-only recovery cannot be authenticated."""


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout: int = 300,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            env=dict(environment) if environment is not None else None,
            timeout=timeout,
            input=input_bytes,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RecoveryError(f"command failed to execute: {command[0]}") from error
    if check and result.returncode:
        raise RecoveryError(
            f"command failed ({result.returncode}): {' '.join(command[:4])}: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return result


def _git(*arguments: str, timeout: int = 180) -> bytes:
    return _run(
        ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
        environment=GIT_ENV,
        timeout=timeout,
    ).stdout


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


def _snapshot_regular(path: Path, *, limit: int = 64 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise RecoveryError(f"unsafe recovery member: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RecoveryError(f"recovery member changed while reading: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _authority(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
    )


def _require_empty_xattrs(descriptor: int, label: str) -> None:
    if os.listxattr(descriptor):
        raise RecoveryError(f"unexpected xattrs on {label}")


class BoundLock:
    """A flock whose parent, name, descriptor, and immutable authority stay bound."""

    def __init__(self, authority: tuple[Any, ...]) -> None:
        (
            parent,
            parent_dev,
            parent_ino,
            parent_uid,
            parent_gid,
            parent_mode,
            name,
            file_dev,
            file_ino,
            file_uid,
            file_gid,
            file_mode,
        ) = authority
        self.parent_path = Path(parent)
        self.name = str(name)
        self.parent_expected = (
            parent_dev,
            parent_ino,
            parent_uid,
            parent_gid,
            parent_mode,
        )
        self.file_expected = (file_dev, file_ino, file_uid, file_gid, file_mode)
        self.parent_fd = os.open(
            self.parent_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        self.file_fd = -1
        try:
            self.file_fd = os.open(
                self.name,
                os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_fd,
            )
            self.revalidate()
            try:
                fcntl.flock(self.file_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RecoveryError(
                    f"protected workflow lock is busy: {self.name}"
                ) from error
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def revalidate(self) -> None:
        parent = os.fstat(self.parent_fd)
        parent_named = os.stat(self.parent_path, follow_symlinks=False)
        file_metadata = os.fstat(self.file_fd)
        file_named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or _authority(parent) != self.parent_expected
            or (parent.st_dev, parent.st_ino)
            != (parent_named.st_dev, parent_named.st_ino)
            or not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_nlink != 1
            or file_metadata.st_size != 0
            or _authority(file_metadata) != self.file_expected
            or (file_metadata.st_dev, file_metadata.st_ino)
            != (file_named.st_dev, file_named.st_ino)
        ):
            raise RecoveryError(f"descriptor-bound lock authority drifted: {self.name}")
        _require_empty_xattrs(self.parent_fd, str(self.parent_path))
        _require_empty_xattrs(self.file_fd, self.name)

    def close(self) -> None:
        if self.file_fd >= 0:
            os.close(self.file_fd)
            self.file_fd = -1
        if getattr(self, "parent_fd", -1) >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


class EvidenceStore:
    """Append-only flat evidence directory retained and replayed through exact FDs."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.members: dict[str, int] = {}
        self.home_fd = os.open(
            "/home/gianl", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        self.root_fd = -1
        self.run_fd = -1
        try:
            self.root_fd = os.open(
                "gate-d-runs",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.home_fd,
            )
            self._revalidate_parents()
            os.mkdir(tag, mode=0o700, dir_fd=self.root_fd)
            os.fsync(self.root_fd)
            self.run_fd = os.open(
                tag,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.root_fd,
            )
            run = os.fstat(self.run_fd)
            self.run_identity = (run.st_dev, run.st_ino, run.st_uid, run.st_gid, 0o700)
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def _revalidate_parents(self) -> None:
        home = os.fstat(self.home_fd)
        home_named = os.stat("/home/gianl", follow_symlinks=False)
        root = os.fstat(self.root_fd)
        root_named = os.stat("gate-d-runs", dir_fd=self.home_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(home.st_mode)
            or _authority(home) != HOME_AUTHORITY
            or (home.st_dev, home.st_ino) != (home_named.st_dev, home_named.st_ino)
            or not stat.S_ISDIR(root.st_mode)
            or _authority(root) != RUN_ROOT_AUTHORITY
            or (root.st_dev, root.st_ino) != (root_named.st_dev, root_named.st_ino)
        ):
            raise RecoveryError("recovery evidence parent authority drifted")
        _require_empty_xattrs(self.home_fd, "/home/gianl")
        _require_empty_xattrs(self.root_fd, "gate-d-runs")

    def revalidate(self) -> None:
        self._revalidate_parents()
        run = os.fstat(self.run_fd)
        run_named = os.stat(self.tag, dir_fd=self.root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(run.st_mode)
            or _authority(run) != self.run_identity
            or (run.st_dev, run.st_ino) != (run_named.st_dev, run_named.st_ino)
        ):
            raise RecoveryError("recovery run-directory authority drifted")
        _require_empty_xattrs(self.run_fd, self.tag)
        for name, descriptor in self.members.items():
            metadata = os.fstat(descriptor)
            named = os.stat(name, dir_fd=self.run_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
                or stat.S_IMODE(metadata.st_mode) != 0o400
                or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise RecoveryError(f"recovery member authority drifted: {name}")
            _require_empty_xattrs(descriptor, name)

    def write(self, name: str, raw: bytes) -> None:
        if name in self.members or Path(name).name != name:
            raise RecoveryError(f"unsafe or duplicate recovery member: {name}")
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.run_fd,
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            os.fsync(self.run_fd)
            self.members[name] = descriptor
            self.revalidate()
        except BaseException:
            if name not in self.members:
                os.close(descriptor)
            raise

    def snapshot(self, name: str, *, limit: int = 64 << 20) -> bytes:
        self.revalidate()
        descriptor = self.members[name]
        before = os.fstat(descriptor)
        if before.st_size > limit:
            raise RecoveryError(f"oversized recovery member: {name}")
        raw = bytearray()
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(8 << 20, before.st_size - offset), offset)
            if not block:
                raise RecoveryError(f"short recovery member: {name}")
            raw.extend(block)
            offset += len(block)
        after = os.fstat(descriptor)
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if stable_before != stable_after:
            raise RecoveryError(f"recovery member changed while reading: {name}")
        self.revalidate()
        return bytes(raw)

    def close(self) -> None:
        for descriptor in self.members.values():
            os.close(descriptor)
        self.members.clear()
        for name in ("run_fd", "root_fd", "home_fd"):
            descriptor = getattr(self, name, -1)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, name, -1)


def _list_prefix(
    prefix: str,
    *,
    soft_deleted: bool,
    allowed_prefixes: set[str] | None = None,
) -> list[dict[str, Any]]:
    command = [
        "/snap/bin/gcloud",
        "storage",
        "ls",
        "--json",
        "--recursive",
        "--soft-deleted" if soft_deleted else "--all-versions",
        prefix.rstrip("/") + "/",
    ]
    result = _run(command, environment=GCLOUD_ENV, timeout=180, check=False)
    if result.returncode:
        if result.stdout == b"" and result.stderr == VACANT_STDERR:
            return []
        raise RecoveryError(
            "remote prefix listing failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError("remote prefix listing is not JSON") from error
    if not isinstance(payload, list):
        raise RecoveryError("remote prefix listing schema drifted")
    objects: list[dict[str, Any]] = []
    expected_prefix = prefix.rstrip("/") + "/"
    permitted_prefixes = allowed_prefixes or {expected_prefix}
    synthetic_prefixes: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise RecoveryError("remote prefix listing record is not an object")
        record_type = item.get("type")
        if record_type == "prefix":
            url = item.get("url")
            if (
                set(item) != {"type", "url"}
                or not isinstance(url, str)
                or url not in permitted_prefixes
                or url in synthetic_prefixes
            ):
                raise RecoveryError("remote synthetic-prefix record drifted")
            synthetic_prefixes.add(url)
        elif record_type == "cloud_object":
            if set(item) != {"type", "url", "metadata"}:
                raise RecoveryError("remote cloud-object record schema drifted")
            if not isinstance(item.get("url"), str) or not isinstance(
                item.get("metadata"), dict
            ):
                raise RecoveryError("remote cloud-object record type drifted")
            objects.append(item)
        else:
            raise RecoveryError("remote prefix listing contains an unknown record")
    if synthetic_prefixes != (permitted_prefixes if objects else set()):
        raise RecoveryError("remote synthetic-prefix/object relationship drifted")
    return objects


def _recovery_prefix_vacancy(remote: str) -> dict[str, Any]:
    all_versions = _list_prefix(remote, soft_deleted=False)
    soft_deleted = _list_prefix(remote, soft_deleted=True)
    validate_recovery_remote_vacancy([], all_versions, soft_deleted)
    return {
        "all_versions": [],
        "artifact_kind": "gate_d_compensated_pp16_recovery_remote_vacancy",
        "checked_before_upload": True,
        "live": [],
        "remote": remote,
        "soft_deleted": [],
        "status": "HISTORICALLY_VACANT",
    }


def _source_remote_manifest() -> dict[str, Any]:
    source_prefix = SOURCE_REMOTE.rstrip("/") + "/"
    source_prefixes = {source_prefix, f"{source_prefix}hlo/"}
    listing = _list_prefix(
        SOURCE_REMOTE, soft_deleted=False, allowed_prefixes=source_prefixes
    )
    if _list_prefix(SOURCE_REMOTE, soft_deleted=True, allowed_prefixes=source_prefixes):
        raise RecoveryError("source diagnostic has soft-deleted ambiguity")
    ledger = json.loads(
        _snapshot_regular(SOURCE_RUN_DIR / "diagnostic_objects.json", limit=1 << 20)
    )
    receipt = json.loads(
        _snapshot_regular(
            SOURCE_RUN_DIR / "diagnostic_upload_receipt.json", limit=1 << 20
        )
    )
    expected = {record["path"]: record for record in ledger["objects"]}
    expected["diagnostic_objects.json"] = receipt["terminal"]
    prefix = SOURCE_REMOTE.rstrip("/") + "/"
    if len(listing) != len(expected):
        raise RecoveryError("source diagnostic all-version count drifted")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in listing:
        metadata = item.get("metadata")
        url = item.get("url")
        if not isinstance(metadata, Mapping) or not isinstance(url, str):
            raise RecoveryError("source diagnostic listing record drifted")
        generation = str(metadata.get("generation", ""))
        suffix = f"#{generation}"
        if not url.startswith(prefix) or not url.endswith(suffix):
            raise RecoveryError("source diagnostic generation URL drifted")
        path = url[len(prefix) : -len(suffix)]
        if path in seen or path not in expected:
            raise RecoveryError("source diagnostic remote catalogue drifted")
        raw = _run(
            ["/snap/bin/gcloud", "storage", "cat", url],
            environment=GCLOUD_ENV,
            timeout=180,
        ).stdout
        record = {
            "crc32c": str(metadata.get("crc32c", "")),
            "generation": generation,
            "path": path,
            "sha256": _sha256(raw),
            "size": len(raw),
            "uri": f"{SOURCE_REMOTE}/{path}",
        }
        wanted = expected[path]
        if (
            _crc32c(raw) != record["crc32c"]
            or record["generation"] != wanted["generation"]
            or record["crc32c"] != wanted["crc32c"]
            or record["sha256"] != wanted["sha256"]
            or record["size"] != wanted["size"]
        ):
            raise RecoveryError(f"source diagnostic remote bytes drifted: {path}")
        records.append(record)
        seen.add(path)
    if seen != set(expected):
        raise RecoveryError("source diagnostic remote catalogue is incomplete")
    return {
        "artifact_kind": "gate_d_compensated_pp16_source_remote_manifest",
        "exact_object_count": len(records),
        "objects": sorted(records, key=lambda value: value["path"]),
        "source_remote": SOURCE_REMOTE,
        "status": "SOURCE_DIAGNOSTIC_AUTHENTICATED",
    }


def _strict_census(tag: str, label: str) -> bytes:
    carrier = f"{tag}_{label}"
    ray_enum = (
        "GLM_CENSUS_CARRIER="
        + carrier
        + " /home/gianl/vllm-env/bin/python -c 'import os,psutil,subprocess; "
        "from ray.autoscaler._private.constants import RAY_PROCESSES; "
        'carrier=os.environ["GLM_CENSUS_CARRIER"]; '
        'marked={p.pid for p in psutil.process_iter(["environ"]) if '
        '(p.info["environ"] or {}).get("GLM_CENSUS_CARRIER")==carrier}; '
        "me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; "
        'out={p.pid for p in psutil.process_iter(["name","cmdline"]) if '
        'p.pid not in skip and any(k in ((p.info["name"] or "") if f else '
        'subprocess.list2cmdline(p.info["cmdline"] or [])) for k,f in RAY_PROCESSES)}; '
        'print(" ".join(map(str,sorted(out))))\''
    )
    command = (
        "tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; "
        "command -v fuser >/dev/null 2>&1 || tools_ok=0; "
        "sudo -n true >/dev/null 2>&1 || tools_ok=0; "
        f"ray_pids=$({ray_enum} 2>/dev/null); ray_rc=$?; "
        "generic=$(pgrep -af '[r]un_gate_d_compensated_pp16_numerical[.]py|"
        "[V]LLM::EngineCore|"
        "[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head' 2>/dev/null "
        "|| true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); "
        "containers=$(sudo -n docker ps --format '{{.ID}} {{.Image}} {{.Names}} "
        "{{.Command}}' 2>/dev/null); docker_rc=$?; "
        "if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then "
        "echo CENSUS_BAD $(hostname); elif [[ -n $ray_pids || -n $generic || "
        '-n $holders ]] || echo "$containers" | grep -Eqi '
        "'[v]llm|[g]emma|[q]wen|[r]erank|[a]spt'; then echo CENSUS_BUSY "
        "$(hostname); else echo CENSUS_OK $(hostname); fi"
    )
    result = _run(
        [
            "/snap/bin/gcloud",
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
        environment={**GCLOUD_ENV, "GLM_CENSUS_CARRIER": carrier},
        timeout=180,
    )
    return result.stdout + result.stderr


def _verify_worktree(pin: str) -> dict[str, Any]:
    if _git("rev-parse", "--show-toplevel").decode().strip() != str(WORKTREE):
        raise RecoveryError("recovery worktree root drifted")
    if _git("branch", "--show-current").decode().strip() != BRANCH:
        raise RecoveryError("recovery branch drifted")
    if _git("remote", "get-url", "origin").decode().strip() != ORIGIN:
        raise RecoveryError("recovery origin drifted")
    if _git("status", "--porcelain=v1").strip():
        raise RecoveryError("recovery worktree is dirty")
    if _git("rev-parse", "HEAD").decode().strip() != pin:
        raise RecoveryError("recovery pin changed")
    remote = _git("ls-remote", "--refs", "origin", f"refs/heads/{BRANCH}")
    if remote != f"{pin}\trefs/heads/{BRANCH}\n".encode():
        raise RecoveryError("recovery origin pin is not exact")
    blobs = []
    for relative in (SOURCE_PATH, MODULE_PATH, MIRROR_VERIFIER_PATH):
        raw = _git("show", f"{pin}:{relative}")
        if raw != _snapshot_regular(WORKTREE / relative, limit=2 << 20):
            raise RecoveryError(f"recovery committed source drifted: {relative}")
        blobs.append({"path": relative, "sha256": _sha256(raw)})
    return {
        "artifact_kind": "gate_d_compensated_pp16_recovery_code_authority",
        "branch": BRANCH,
        "origin": ORIGIN,
        "origin_pin_exact": True,
        "pin": pin,
        "runtime_blobs": blobs,
    }


def _mirror_replay(pin: str) -> dict[str, Any]:
    verifier = WORKTREE / MIRROR_VERIFIER_PATH
    verifier_sha = _sha256(_snapshot_regular(verifier, limit=2 << 20))
    environment = {
        "HOME": "/home/gianl",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = _run(
        [
            "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12",
            "-I",
            "-S",
            "-B",
            str(verifier),
            "--expected-code-hash",
            pin,
            "--expected-source-sha256",
            verifier_sha,
        ],
        environment=environment,
        timeout=900,
    )
    report = json.loads(result.stdout)
    local_archive_sha256 = _sha256(_git("archive", "--format=tar", pin, timeout=600))
    if (
        report.get("commit") != pin
        or report.get("mirror_uri") != "gs://driftbench-dsv4-uc/repos/glm-tpu/.git"
        or report.get("checkout_archive_sha256") != local_archive_sha256
    ):
        raise RecoveryError("same-region recovery Git mirror closure drifted")
    recovery_blobs = []
    for relative in (SOURCE_PATH, MODULE_PATH, MIRROR_VERIFIER_PATH):
        raw = _git("show", f"{pin}:{relative}")
        recovery_blobs.append(
            {
                "git_object": _git("rev-parse", f"{pin}:{relative}").decode().strip(),
                "path": relative,
                "sha256": _sha256(raw),
            }
        )
    report["recovery_binding_basis"] = (
        "The historical verifier proved byte-identical complete checkout archives at "
        "this pin; these recovery blobs are exact members of that archive."
    )
    report["recovery_bound_blobs"] = recovery_blobs
    return report


def _ledger(store: EvidenceStore, members: Sequence[str]) -> bytes:
    lines = []
    for relative in sorted(members):
        raw = store.snapshot(relative)
        lines.append(f"{_sha256(raw)}  {relative}\n")
    return "".join(lines).encode("ascii")


def _upload_one(raw: bytes, name: str, remote: str) -> dict[str, Any]:
    result = _run(
        [
            "/snap/bin/gcloud",
            "storage",
            "cp",
            "--if-generation-match=0",
            "--print-created-message",
            "-",
            remote,
        ],
        environment=GCLOUD_ENV,
        timeout=180,
        input_bytes=raw,
    )
    metadata_raw = _run(
        ["/snap/bin/gcloud", "storage", "objects", "describe", remote, "--format=json"],
        environment=GCLOUD_ENV,
        timeout=60,
    ).stdout
    metadata = json.loads(metadata_raw)
    generation = str(metadata.get("generation", ""))
    if (
        not generation.isdecimal()
        or int(metadata.get("size", -1)) != len(raw)
        or str(metadata.get("crc32c", metadata.get("crc32c_hash", ""))) != _crc32c(raw)
        or f"{remote}#{generation}".encode() not in result.stderr + result.stdout
    ):
        raise RecoveryError(f"remote upload identity drifted: {remote}")
    downloaded = _run(
        ["/snap/bin/gcloud", "storage", "cat", f"{remote}#{generation}"],
        environment=GCLOUD_ENV,
        timeout=180,
    ).stdout
    if downloaded != raw:
        raise RecoveryError(f"remote generation replay drifted: {remote}")
    return {
        "crc32c": _crc32c(raw),
        "generation": generation,
        "path": name,
        "sha256": _sha256(raw),
        "size": len(raw),
        "uri": remote,
    }


def _verify_remote_catalogue(
    remote: str, expected_paths: set[str]
) -> list[dict[str, Any]]:
    listing = _list_prefix(remote, soft_deleted=False)
    if _list_prefix(remote, soft_deleted=True):
        raise RecoveryError("recovery prefix gained soft-deleted objects")
    observed = set()
    for item in listing:
        url = item.get("url", "")
        metadata = item.get("metadata", {})
        generation = str(metadata.get("generation", ""))
        suffix = f"#{generation}"
        prefix = remote.rstrip("/") + "/"
        if not url.startswith(prefix) or not url.endswith(suffix):
            raise RecoveryError("recovery remote catalogue URL drifted")
        observed.add(url[len(prefix) : -len(suffix)])
    if observed != expected_paths or len(listing) != len(expected_paths):
        raise RecoveryError("recovery remote catalogue is partial or has history")
    return listing


def _summary(
    authentication: Mapping[str, Any],
    code: Mapping[str, Any],
    mirror: Mapping[str, Any],
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
    tag: str,
    remote: str,
) -> dict[str, Any]:
    return {
        "artifact_kind": "gate_d_compensated_pp16_recovery_summary",
        "claim_scope": (
            "CPU-only append-only recovery of one already-completed protected TPU "
            "numerical rejection; no TPU rerun, DB, performance, decoder, token, or "
            "Gate-D closure claim."
        ),
        "compiled_executable_invocation_count": 1,
        "first_divergence": authentication["first_divergence"],
        "fresh_census_post": post,
        "fresh_census_pre": pre,
        "gate_d_closed": False,
        "historical_publisher_accepted": False,
        "mirror_replay_sha256": _sha256(canonical_json(mirror)),
        "next_action": (
            "Tombstone compensated_auxiliary_dependency; continue Gate-D scalar RMS "
            "normalization first-divergence adjudication with offline T1/T2/T3/T4 "
            "evidence before any further TPU compilation."
        ),
        "performance_claim": False,
        "recovery_code_hash": code["pin"],
        "recovery_remote": remote,
        "recovery_tag": tag,
        "source_authentication_sha256": _sha256(canonical_json(authentication)),
        "source_code_hash": authentication["source_code_hash"],
        "source_tag": authentication["source_tag"],
        "status": "NUMERICAL_REJECTED_RECOVERED",
        "tpu_rerun_performed": False,
    }


def _terminal(
    store: EvidenceStore,
    remote_objects: bytes,
    remote_objects_receipt: Mapping[str, Any],
) -> bytes:
    evidence = store.snapshot("recovery.evidence.sha256", limit=1 << 20)
    authentication = store.snapshot("source_authentication.json", limit=1 << 20)
    source_remote = store.snapshot("source_remote_objects.json", limit=1 << 20)
    summary = store.snapshot("summary.json", limit=1 << 20)
    record = {
        "artifact_kind": "gate_d_compensated_pp16_recovery_terminal",
        "evidence_sha256": _sha256(evidence),
        "source_authentication_sha256": _sha256(authentication),
        "preterminal_remote_ledger": dict(remote_objects_receipt),
        "preterminal_remote_ledger_sha256": _sha256(remote_objects),
        "source_remote_manifest_sha256": _sha256(source_remote),
        "status": "NUMERICAL_REJECTED_RECOVERED",
        "summary_sha256": _sha256(summary),
    }
    marker = _sha256(
        json.dumps(
            record, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    record["marker_self_sha256"] = marker
    return canonical_json(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validate-only", "seal-rejected"), required=True
    )
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    if os.environ.get("GLM_GATE_D_COMPENSATED_PP16_RECOVERY") != "1":
        raise RecoveryError("compensated PP16 recovery is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RecoveryError("recovery must be explicitly pinned to CPU")
    if TAG_PATTERN.fullmatch(arguments.tag) is None:
        raise RecoveryError("unsafe recovery tag")
    if shutil.which("gcloud") != "/snap/bin/gcloud":
        raise RecoveryError("unexpected gcloud executable")
    remote = f"{REMOTE_ROOT}/{arguments.tag}"
    locks: list[BoundLock] = []
    store: EvidenceStore | None = None

    def guard() -> None:
        for lock in locks:
            lock.revalidate()
        if store is not None:
            store.revalidate()

    try:
        for authority in LOCK_AUTHORITIES:
            locks.append(BoundLock(authority))
        guard()
        pin = _git("rev-parse", "HEAD").decode().strip()
        code = _verify_worktree(pin)
        guard()
        if (
            _run(
                [
                    "/snap/bin/gcloud",
                    "storage",
                    "buckets",
                    "describe",
                    BUCKET,
                    "--format=value(location)",
                ],
                environment=GCLOUD_ENV,
                timeout=60,
            )
            .stdout.decode()
            .strip()
            != LOCATION
        ):
            raise RecoveryError("approved bucket location drifted")
        guard()
        store = EvidenceStore(arguments.tag)
        guard()
        vacancy = _recovery_prefix_vacancy(remote)
        store.write("remote_vacancy.json", canonical_json(vacancy))
        store.write("code_authority.json", canonical_json(code))
        pre_raw = _strict_census(arguments.tag, "recovery_pre")
        store.write("census_recovery_pre.txt", pre_raw)
        pre = require_eight_host_census_bytes(pre_raw)
        guard()

        source_remote = _source_remote_manifest()
        source_remote_raw = canonical_json(source_remote)
        store.write("source_remote_objects.json", source_remote_raw)
        authentication = authenticate_compensated_pp16_rejection(
            SOURCE_RUN_DIR,
            worktree=WORKTREE,
            capsule=CAPSULE_ROOT / "capsule.json",
            capsule_inputs=CAPSULE_ROOT / "candidate-inputs.npz",
            capsule_state=CAPSULE_ROOT / "candidate-state.npz",
            capsule_execution_authority=CAPSULE_AUTHORITY,
            source_remote_manifest=source_remote_raw,
            verify_live_dependencies=True,
        )
        store.write("source_authentication.json", canonical_json(authentication))
        mirror = _mirror_replay(pin)
        store.write("mirror_replay.json", canonical_json(mirror))
        guard()

        post_raw = _strict_census(arguments.tag, "recovery_post")
        store.write("census_recovery_post.txt", post_raw)
        post = require_eight_host_census_bytes(post_raw)
        summary = _summary(
            authentication, code, mirror, pre, post, arguments.tag, remote
        )
        store.write("summary.json", canonical_json(summary))
        log = (
            f"NUMERICAL_REJECTED_RECOVERED source={SOURCE_TAG} recovery={arguments.tag} "
            "tpu_rerun=false gate_d_open=true performance_claim=false "
            "historical_publisher_accepted=false\n"
        ).encode("ascii")
        store.write("orchestrator.sealed.log", log)
        evidence_members = (
            "census_recovery_post.txt",
            "census_recovery_pre.txt",
            "code_authority.json",
            "mirror_replay.json",
            "orchestrator.sealed.log",
            "remote_vacancy.json",
            "source_authentication.json",
            "source_remote_objects.json",
            "summary.json",
        )
        evidence = _ledger(store, evidence_members)
        store.write("recovery.evidence.sha256", evidence)
        for line in evidence.decode().splitlines():
            digest, relative = line.split(maxsplit=1)
            if _sha256(store.snapshot(relative)) != digest:
                raise RecoveryError("local recovery evidence ledger drifted")
        guard()
        if arguments.mode == "validate-only":
            print(
                f"VALIDATED_ONLY tag={arguments.tag} status=NUMERICAL_REJECTED_RECOVERED "
                "tpu_rerun=false remote_writes=0"
            )
            return 0

        preterminal = set(evidence_members) | {"recovery.evidence.sha256"}
        receipts = []
        for relative in sorted(preterminal):
            guard()
            receipts.append(
                _upload_one(store.snapshot(relative), relative, f"{remote}/{relative}")
            )
            guard()
        _verify_remote_catalogue(remote, preterminal)
        remote_objects = canonical_json(
            {
                "artifact_kind": "gate_d_compensated_pp16_recovery_remote_objects",
                "exact_object_count": len(receipts),
                "objects": receipts,
                "remote": remote,
                "status": "PRETERMINAL_OBJECTS_AUTHENTICATED",
            }
        )
        remote_objects_name = "remote_objects.json"
        store.write(remote_objects_name, remote_objects)
        guard()
        remote_objects_receipt = _upload_one(
            store.snapshot(remote_objects_name),
            remote_objects_name,
            f"{remote}/{remote_objects_name}",
        )
        preterminal_with_ledger = preterminal | {remote_objects_name}
        _verify_remote_catalogue(remote, preterminal_with_ledger)
        guard()

        terminal_name = "NUMERICAL_REJECTED_RECOVERED.json"
        terminal_raw = _terminal(store, remote_objects, remote_objects_receipt)
        store.write(terminal_name, terminal_raw)
        guard()
        terminal_remote = f"{remote}/{terminal_name}"

        # From this publication attempt onward the evidence is immutable.  A failure
        # preserves the partial append-only state for diagnosis; it is never rolled back.
        terminal_receipt = _upload_one(terminal_raw, terminal_name, terminal_remote)
        final_paths = preterminal_with_ledger | {terminal_name}
        final_listing = _verify_remote_catalogue(remote, final_paths)
        terminal_generation = int(terminal_receipt["generation"])
        if any(
            int(item["metadata"]["generation"]) >= terminal_generation
            for item in final_listing
            if not item["url"].endswith(
                f"/{terminal_name}#{terminal_receipt['generation']}"
            )
        ):
            raise RecoveryError("recovery terminal was not the final mutation")
        guard()
        print(
            f"NUMERICAL_REJECTED_RECOVERED sealed={remote} source_untouched=true "
            "tpu_rerun=false no_DB_Gate-D_or_performance_claim=true "
            f"terminal_generation={terminal_receipt['generation']} "
            f"terminal_crc32c={terminal_receipt['crc32c']} "
            f"terminal_sha256={terminal_receipt['sha256']}"
        )
        return 0
    finally:
        if store is not None:
            store.close()
        for lock in reversed(locks):
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
