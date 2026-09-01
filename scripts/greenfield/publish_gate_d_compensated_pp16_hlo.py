#!/usr/bin/env python3
"""Append-only local and generation-bound remote PP16 HLO publication.

The success terminal upload is intentionally the final remote mutation.  A
successful return means every object was generation-zero published, then
generation-qualified downloaded and bound by generation/size/CRC32C/SHA; only
read-only verification and an append-only local receipt follow the terminal.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import stat
import struct
import subprocess
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
PYTHON_RUNTIME_ROOT = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
PYTHON = PYTHON_RUNTIME_ROOT / "bin/python3.12"
PYTHON_SHA256 = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
PYTHON_RUNTIME_TREE_SHA256 = (
    "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
)
JAX_SITE_ROOT = Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e")
JAX_SITE_TREE_SHA256 = (
    "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
)
JAX_SITE_MANIFEST_SHA256 = (
    "ef454caafd2e4ba5da4bc7f7f73bef4e8f157afd6c795a91b09e319961941eff"
)
LIBTPU_SITE_ROOT = Path("/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3")
LIBTPU_SITE_TREE_SHA256 = (
    "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca"
)
LIBTPU_SITE_MANIFEST_SHA256 = (
    "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa"
)
STORAGE_SITE_ROOT = Path("/opt/glm-tpu/gate-d-storage-site-d94fd4c3e0ff")
STORAGE_SITE_TREE_SHA256 = (
    "d94fd4c3e0ffbf024a0b53faff4565d28900b64fc5dd9944bff316f47db1b510"
)
STORAGE_SITE_MANIFEST_SHA256 = (
    "67b485a54ac0eaab8202053d07d58a26f0ea5fc44c2e555e66dfdb982a7e6e3a"
)
STORAGE_SITE_BUILDER_SOURCE_PATH = (
    "scripts/greenfield/build_gate_d_storage_site_capsule.py"
)
STORAGE_SITE_BUILDER_SOURCE_SHA256 = (
    "b7f4f869ae9b98edf5195185bf49fffcf61ef6800a2b707127694b66424c5989"
)
SOURCE_PATH = "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py"
COMPILER_DRIVER_PATH = (
    REPO / "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
)
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/gate_d_pp16_hlo/"
TAG_PATTERN = re.compile(r"gate_d_compensated_pp16_hlo_[0-9]{8}T[0-9]{15}Z")
_ACCELERATOR_DEVICE_PATTERN = re.compile(r"/dev/accel([0-3])")
_ACCELERATOR_DEVICE_DIRECTORY_MODE = 0o755
_ACCELERATOR_DEVICE_MODE = 0o666
_ACCELERATOR_RDEV_MAJOR = 121
_ACCELERATOR_DEVICE_OBSERVATION_SCOPE = (
    "Unique canonical /dev/accel node paths observed mapped in the compiler "
    "process, deduplicated by path; not a complete VMA catalogue."
)
SUCCESS_PAYLOAD = (
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "evidence.json",
    "hlo/compensated_pp16_stage0.optimized_hlo.txt",
    "hlo/compensated_pp16_stage0.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "summary.json",
    "sync.txt",
)
_WRAPPER_WRITE_MEMBERS = {
    "census_failure_exit.txt",
    "census_post.txt",
    "census_pre.txt",
    "mirror.sha256",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.log",
    "sync.txt",
}
_EXPECTED_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_EXPECTED_RUNTIME_PATH = (
    str(PYTHON_RUNTIME_ROOT / "lib/python312.zip"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12/lib-dynload"),
)
_EXPECTED_COMPILER_ENVIRONMENT = {
    **_EXPECTED_ENVIRONMENT,
    "JAX_ENABLE_COMPILATION_CACHE": "0",
    "JAX_PLATFORMS": "tpu",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_VISIBLE_DEVICES": "0,1,2,3",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(REPO), *arguments],
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _snapshot_path(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"unsafe publication source: {path}")
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size:
            raise RuntimeError(f"publication source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path, *, require_sealed: bool) -> str:
    digest = sha256()
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if require_sealed and (
            metadata.st_uid != 0
            or metadata.st_gid != 0
            or os.listxattr(entry, follow_symlinks=False)
            or (
                not stat.S_ISLNK(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) != _safe_mode(metadata.st_mode)
            )
        ):
            raise RuntimeError(f"publication runtime is not root-owned sealed: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(
                _sha_file(entry)
            )
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise RuntimeError(
                    f"publication runtime symlink escapes root: {entry}"
                ) from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise RuntimeError(f"unsupported publication runtime entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", _safe_mode(metadata.st_mode)))
        digest.update(payload)
    return digest.hexdigest()


def _validate_storage_site(
    root: Path,
    *,
    expected_tree_sha256: str,
    expected_manifest_sha256: str,
    require_sealed: bool,
) -> dict[str, Any]:
    if _tree_sha256(root, require_sealed=require_sealed) != expected_tree_sha256:
        raise RuntimeError("PP16 HLO publisher storage dependency tree drifted")
    manifest_raw = _snapshot_path(root / "CAPSULE_MANIFEST.json")
    if sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        raise RuntimeError("PP16 HLO publisher storage dependency manifest drifted")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("allowlist"), list)
        or not manifest["allowlist"]
        or not isinstance(manifest.get("source_entries"), Mapping)
        or sorted(manifest["allowlist"]) != sorted(manifest["source_entries"])
    ):
        raise RuntimeError("PP16 HLO publisher storage dependency manifest is invalid")
    return manifest


def validate_publication_runtime() -> dict[str, Any]:
    if (
        Path(sys.executable) != PYTHON
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or tuple(sys.path) != _EXPECTED_RUNTIME_PATH
    ):
        raise RuntimeError("PP16 HLO publisher interpreter boundary drifted")
    if _sha_file(PYTHON) != PYTHON_SHA256:
        raise RuntimeError("PP16 HLO publisher interpreter bytes drifted")
    if (
        _tree_sha256(PYTHON_RUNTIME_ROOT, require_sealed=True)
        != PYTHON_RUNTIME_TREE_SHA256
    ):
        raise RuntimeError("PP16 HLO publisher Python runtime tree drifted")
    manifest = _validate_storage_site(
        STORAGE_SITE_ROOT,
        expected_tree_sha256=STORAGE_SITE_TREE_SHA256,
        expected_manifest_sha256=STORAGE_SITE_MANIFEST_SHA256,
        require_sealed=True,
    )
    return {
        "artifact_kind": "gate_d_compensated_pp16_publisher_runtime",
        "dependency_allowlist_count": len(manifest["allowlist"]),
        "dependency_builder_source_path": STORAGE_SITE_BUILDER_SOURCE_PATH,
        "dependency_builder_source_sha256": STORAGE_SITE_BUILDER_SOURCE_SHA256,
        "dependency_manifest_sha256": STORAGE_SITE_MANIFEST_SHA256,
        "dependency_tree_sha256": STORAGE_SITE_TREE_SHA256,
        "dependency_root": str(STORAGE_SITE_ROOT),
        "python_executable": str(PYTHON),
        "python_sha256": PYTHON_SHA256,
        "python_runtime_root": str(PYTHON_RUNTIME_ROOT),
        "python_runtime_tree_sha256": PYTHON_RUNTIME_TREE_SHA256,
    }


def verify_running_source(code_pin: str, expected_sha256: str) -> None:
    raw = _snapshot_path(Path(__file__))
    committed = _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
    head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    resolved = _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode("ascii").strip()
    if (
        head != code_pin
        or resolved != code_pin
        or raw != committed
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("PP16 HLO publisher is not the exact committed blob")


def validate_environment() -> dict[str, str]:
    observed = dict(os.environ)
    if observed != _EXPECTED_ENVIRONMENT:
        raise RuntimeError("PP16 HLO publisher environment is not the exact allowlist")
    return dict(sorted(observed.items()))


def _canonical_path_key(path: str) -> bytes:
    return os.fsencode(path)


def _open_directory_chain(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in Path(os.path.abspath(path)).parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            writable = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            protected_sticky_root = (
                metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
            )
            if not stat.S_ISDIR(metadata.st_mode) or (
                writable and not protected_sticky_root
            ):
                raise RuntimeError(f"unsafe publication directory chain: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validate_run_dir(run_dir: Path) -> str:
    absolute = Path(os.path.abspath(run_dir))
    if absolute.parent != RUN_ROOT or not TAG_PATTERN.fullmatch(absolute.name):
        raise RuntimeError("PP16 HLO run directory is outside its fixed root")
    return absolute.name


def _open_run_root() -> int:
    descriptor = _open_directory_chain(RUN_ROOT)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or os.listxattr(RUN_ROOT, follow_symlinks=False)
    ):
        os.close(descriptor)
        raise RuntimeError("PP16 HLO run root is not exact private 0700 authority")
    return descriptor


def initialize_run_dir(
    run_dir: Path, *, publication_runtime_raw: bytes | None = None
) -> str:
    tag = validate_run_dir(run_dir)
    parent_fd = _open_run_root()
    run_fd = -1
    try:
        os.mkdir(tag, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        run_fd = os.open(
            tag,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        os.fchmod(run_fd, 0o700)
        os.mkdir("hlo", mode=0o700, dir_fd=run_fd)
        if publication_runtime_raw is not None:
            write_member_exclusive(
                run_fd, "publisher_runtime.json", publication_runtime_raw
            )
        os.fsync(run_fd)
        metadata = os.fstat(run_fd)
        return f"{metadata.st_dev}:{metadata.st_ino}"
    finally:
        if run_fd >= 0:
            os.close(run_fd)
        os.close(parent_fd)


def _run_fd(run_dir: Path, inherited_fd: int | None = None) -> int:
    tag = validate_run_dir(run_dir)
    root_fd = _open_run_root()
    descriptor = -1
    try:
        descriptor = (
            os.dup(inherited_fd)
            if inherited_fd is not None
            else os.open(
                tag,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
        )
        metadata = os.fstat(descriptor)
        observed = os.stat(tag, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
            or metadata.st_gid != os.getegid()
            or os.listxattr(descriptor)
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError("PP16 HLO run directory authority drifted")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "PP16 HLO run directory supervisor lock is absent"
            ) from error
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _member_parts(relative: str) -> tuple[str, ...]:
    parts = Path(relative).parts
    if (
        not parts
        or relative.startswith("/")
        or any(item in {"", ".", ".."} for item in parts)
        or len(parts) > 2
    ):
        raise RuntimeError(f"unsafe publication member: {relative}")
    return parts


def snapshot_member(run_fd: int, relative: str, *, limit: int = 1 << 30) -> bytes:
    parts = _member_parts(relative)
    parent_fd = run_fd
    owned_parent = False
    try:
        if len(parts) == 2:
            parent_fd = os.open(
                parts[0],
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=run_fd,
            )
            owned_parent = True
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > limit
            ):
                raise RuntimeError(f"unsafe publication member identity: {relative}")
            chunks = []
            total = 0
            while block := os.read(descriptor, 8 * 1024 * 1024):
                total += len(block)
                if total > limit:
                    raise RuntimeError(f"publication member exceeds limit: {relative}")
                chunks.append(block)
            raw = b"".join(chunks)
            final_metadata = os.fstat(descriptor)
            observed = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if (
                len(raw) != metadata.st_size
                or identity
                != (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                    final_metadata.st_mode,
                    final_metadata.st_nlink,
                    final_metadata.st_size,
                    final_metadata.st_mtime_ns,
                    final_metadata.st_ctime_ns,
                )
                or (observed.st_dev, observed.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise RuntimeError(f"publication member changed: {relative}")
            return raw
        finally:
            os.close(descriptor)
    finally:
        if owned_parent:
            os.close(parent_fd)


def write_member_exclusive(run_fd: int, relative: str, payload: bytes) -> None:
    parts = _member_parts(relative)
    if len(parts) != 1:
        raise RuntimeError("generated publication members must be top-level")
    try:
        descriptor = os.open(
            parts[0],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=run_fd,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"append-only publication member exists: {relative}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"unsafe generated publication member: {relative}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"generated publication write stalled: {relative}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        observed = os.stat(parts[0], dir_fd=run_fd, follow_symlinks=False)
        if (
            (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size != len(payload)
        ):
            raise RuntimeError(f"generated publication member changed: {relative}")
        os.fsync(run_fd)
    finally:
        os.close(descriptor)


def write_member_stream_exclusive(
    run_fd: int, relative: str, stream: Any, *, limit: int = 1 << 30
) -> None:
    parts = _member_parts(relative)
    if len(parts) != 1 or relative not in _WRAPPER_WRITE_MEMBERS:
        raise RuntimeError("stream publication member is not allowlisted")
    try:
        descriptor = os.open(
            relative,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=run_fd,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"append-only publication member exists: {relative}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"unsafe stream publication member: {relative}")
        total = 0
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise RuntimeError(
                    f"stream publication member exceeds limit: {relative}"
                )
            view = memoryview(block)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError(f"stream publication write stalled: {relative}")
                view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        observed = os.stat(relative, dir_fd=run_fd, follow_symlinks=False)
        if (
            (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size != total
        ):
            raise RuntimeError(f"stream publication member changed: {relative}")
        os.fsync(run_fd)
    finally:
        os.close(descriptor)


def append_orchestrator_log(run_fd: int, payload: bytes) -> None:
    descriptor = os.open(
        "orchestrator.log",
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=run_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("unsafe orchestrator log identity")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("orchestrator log write stalled")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        observed = os.stat("orchestrator.log", dir_fd=run_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError("orchestrator log identity changed")
        os.fsync(run_fd)
    finally:
        os.close(descriptor)


def require_preterminal(run_fd: int) -> None:
    for name in (
        "HLO_ACQUIRED",
        "terminal_upload_receipt.json",
        "failure_status.json",
        "orchestrator.failure.log",
        "diagnostic_objects.json",
        "diagnostic_upload_receipt.json",
    ):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("local terminal publication has already begun")


def require_publication_runtime(run_fd: int, expected_raw: bytes) -> None:
    observed = snapshot_member(run_fd, "publisher_runtime.json", limit=1 << 20)
    if observed != expected_raw:
        raise RuntimeError("PP16 HLO publisher runtime record drifted")


def write_preterminal_member(
    run_fd: int, relative: str, stream: Any, *, limit: int = 1 << 30
) -> None:
    require_preterminal(run_fd)
    write_member_stream_exclusive(run_fd, relative, stream, limit=limit)


def append_preterminal_log(run_fd: int, payload: bytes) -> None:
    require_preterminal(run_fd)
    append_orchestrator_log(run_fd, payload)


def local_members(run_fd: int) -> set[str]:
    members: set[str] = set()

    def visit(directory_fd: int, prefix: str, depth: int) -> None:
        if depth > 1:
            raise RuntimeError("publication directory depth is unsafe")
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if entry.name in {"", ".", ".."} or "/" in entry.name:
                    raise RuntimeError("publication member name is unsafe")
                relative = f"{prefix}{entry.name}"
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    if depth != 0 or relative != "hlo":
                        raise RuntimeError(
                            f"unexpected local publication directory: {relative}"
                        )
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        visit(child_fd, relative + "/", depth + 1)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    members.add(relative)
                else:
                    raise RuntimeError(f"unsafe local publication member: {relative}")

    visit(run_fd, "", 0)
    return members


def _crc32c(raw: bytes) -> str:
    import google_crc32c

    return base64.b64encode(google_crc32c.value(raw).to_bytes(4, "big")).decode()


def _bucket_and_prefix(remote: str, storage_bucket: Any | None) -> tuple[Any, str]:
    prefix = f"gs://{BUCKET_NAME}/{REMOTE_ROOT}"
    suffix = remote.removeprefix(prefix)
    if not remote.startswith(prefix) or not TAG_PATTERN.fullmatch(suffix):
        raise RuntimeError("PP16 HLO remote prefix is invalid")
    if storage_bucket is None:
        if tuple(sys.path) != _EXPECTED_RUNTIME_PATH:
            raise RuntimeError("PP16 HLO publisher import path drifted")
        sys.path.insert(0, str(STORAGE_SITE_ROOT))
        from google.cloud import storage

        storage_bucket = storage.Client().bucket(BUCKET_NAME)
    if storage_bucket.name != BUCKET_NAME:
        raise RuntimeError("PP16 HLO publication bucket drifted")
    return storage_bucket, REMOTE_ROOT + suffix + "/"


def _observed_names(bucket: Any, prefix: str) -> set[str]:
    return {blob.name.removeprefix(prefix) for blob in bucket.list_blobs(prefix=prefix)}


def _upload_bound(bucket: Any, name: str, raw: bytes) -> dict[str, Any]:
    crc = _crc32c(raw)
    blob = bucket.blob(name)
    blob.upload_from_string(raw, if_generation_match=0, checksum="crc32c")
    generation = str(blob.generation or "")
    if not generation or not generation.isdigit():
        raise RuntimeError(f"remote upload returned no generation: {name}")
    blob.reload(if_generation_match=int(generation))
    if (
        str(blob.generation) != generation
        or int(blob.size) != len(raw)
        or blob.crc32c != crc
    ):
        raise RuntimeError(f"remote upload identity drifted: {name}")
    return {
        "crc32c": crc,
        "generation": generation,
        "path": name.rsplit("/", 1)[-1],
        "sha256": sha256(raw).hexdigest(),
        "size": len(raw),
    }


def _replay_bound(bucket: Any, object_name: str, record: Mapping[str, Any]) -> None:
    blob = bucket.blob(object_name)
    blob.reload(if_generation_match=int(record["generation"]))
    if (
        str(blob.generation) != record["generation"]
        or int(blob.size) != record["size"]
        or blob.crc32c != record["crc32c"]
    ):
        raise RuntimeError(f"remote generation replay drifted: {object_name}")
    downloaded = blob.download_as_bytes(
        if_generation_match=int(record["generation"]), checksum="crc32c"
    )
    if (
        len(downloaded) != record["size"]
        or _crc32c(downloaded) != record["crc32c"]
        or sha256(downloaded).hexdigest() != record["sha256"]
    ):
        raise RuntimeError(f"remote generation-qualified bytes drifted: {object_name}")


def _record(relative: str, raw: bytes) -> dict[str, Any]:
    return {
        "byte_count": len(raw),
        "path": relative,
        "sha256": sha256(raw).hexdigest(),
    }


def _require_census(raw: bytes, marker: str) -> None:
    hosts = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == marker:
            hosts.append(fields[1])
    if len(hosts) != 8 or len(set(hosts)) != 8:
        raise RuntimeError(f"authenticated eight-host marker is absent: {marker}")


def _verify_dependency_record_live(record: Mapping[str, Any]) -> None:
    path = record["path"]
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != record["bytes"]
            or metadata.st_dev != record["device"]
            or metadata.st_ino != record["inode"]
        ):
            raise RuntimeError(
                f"PP16 HLO compiler dependency live identity drifted: {path}"
            )
        digest = sha256()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        final = os.fstat(descriptor)
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) or digest.hexdigest() != record["sha256"]:
            raise RuntimeError(
                f"PP16 HLO compiler dependency live bytes drifted: {path}"
            )
    finally:
        os.close(descriptor)


def _verify_accelerator_device_directory_named(
    *, root_directory_fd: int, device_directory_fd: int
) -> None:
    held = os.fstat(device_directory_fd)
    try:
        named = os.stat("dev", dir_fd=root_directory_fd, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("accelerator device directory vanished") from error
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or held.st_uid != 0
        or held.st_gid != 0
        or stat.S_IMODE(held.st_mode) != _ACCELERATOR_DEVICE_DIRECTORY_MODE
        or (
            held.st_mode,
            held.st_uid,
            held.st_gid,
            held.st_dev,
            held.st_ino,
        )
        != (
            named.st_mode,
            named.st_uid,
            named.st_gid,
            named.st_dev,
            named.st_ino,
        )
    ):
        raise RuntimeError("accelerator device directory identity drifted")


def _open_accelerator_device_directory() -> tuple[int, int]:
    root_directory_fd = os.open(
        "/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        device_directory_fd = os.open(
            "dev",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_directory_fd,
        )
    except BaseException:
        os.close(root_directory_fd)
        raise
    try:
        _verify_accelerator_device_directory_named(
            root_directory_fd=root_directory_fd,
            device_directory_fd=device_directory_fd,
        )
    except BaseException:
        os.close(device_directory_fd)
        os.close(root_directory_fd)
        raise
    return root_directory_fd, device_directory_fd


def _verify_accelerator_device_mapping_record_live(
    record: Mapping[str, Any], *, root_directory_fd: int, device_directory_fd: int
) -> None:
    path = record["path"]
    suffix = int(_ACCELERATOR_DEVICE_PATTERN.fullmatch(path).group(1))
    _verify_accelerator_device_directory_named(
        root_directory_fd=root_directory_fd,
        device_directory_fd=device_directory_fd,
    )

    def named_metadata() -> os.stat_result:
        try:
            return os.stat(
                f"accel{suffix}",
                dir_fd=device_directory_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"PP16 HLO accelerator device mapping vanished: {path}"
            ) from error

    before = named_metadata()
    expected = {
        "gid": int(before.st_gid),
        "mapped_device_major": os.major(before.st_dev),
        "mapped_device_minor": os.minor(before.st_dev),
        "mapped_inode": int(before.st_ino),
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": int(before.st_nlink),
        "node_device": int(before.st_dev),
        "node_inode": int(before.st_ino),
        "path": path,
        "rdev_major": os.major(before.st_rdev),
        "rdev_minor": os.minor(before.st_rdev),
        "uid": int(before.st_uid),
    }
    if (
        not stat.S_ISCHR(before.st_mode)
        or before.st_nlink != 1
        or dict(record) != expected
    ):
        raise RuntimeError(
            f"PP16 HLO accelerator device mapping live identity drifted: {path}"
        )
    after = named_metadata()
    if (
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_dev,
        after.st_ino,
        after.st_rdev,
    ) != (
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_dev,
        before.st_ino,
        before.st_rdev,
    ):
        raise RuntimeError(f"PP16 HLO accelerator device mapping changed: {path}")
    _verify_accelerator_device_directory_named(
        root_directory_fd=root_directory_fd,
        device_directory_fd=device_directory_fd,
    )


def _validate_accelerator_device_mapping_records(
    records: Any, *, verify_live: bool = False
) -> None:
    if not isinstance(records, list) or not records:
        raise RuntimeError("PP16 HLO accelerator device mapping records are absent")
    expected_keys = {
        "gid",
        "mapped_device_major",
        "mapped_device_minor",
        "mapped_inode",
        "mode",
        "nlink",
        "node_device",
        "node_inode",
        "path",
        "rdev_major",
        "rdev_minor",
        "uid",
    }
    paths: list[str] = []
    identities: set[tuple[int, int, int]] = set()
    directory_fds = _open_accelerator_device_directory() if verify_live else None
    root_directory_fd = directory_fds[0] if directory_fds is not None else None
    device_directory_fd = directory_fds[1] if directory_fds is not None else None
    try:
        for record in records:
            if not isinstance(record, Mapping) or set(record) != expected_keys:
                raise RuntimeError("PP16 HLO accelerator device mapping record drifted")
            path = record["path"]
            match = (
                _ACCELERATOR_DEVICE_PATTERN.fullmatch(path)
                if type(path) is str
                else None
            )
            integers = tuple(record[key] for key in expected_keys - {"path"})
            if (
                match is None
                or os.path.realpath(path) != path
                or any(type(value) is not int or value < 0 for value in integers)
            ):
                raise RuntimeError(
                    "PP16 HLO accelerator device mapping identity drifted"
                )
            suffix = int(match.group(1))
            if (
                record["uid"] != 0
                or record["gid"] != 0
                or record["mode"] != _ACCELERATOR_DEVICE_MODE
                or record["nlink"] != 1
                or record["rdev_major"] != _ACCELERATOR_RDEV_MAJOR
                or record["rdev_minor"] != suffix
                or record["mapped_inode"] <= 0
                or record["node_inode"] != record["mapped_inode"]
                or record["mapped_device_major"] != os.major(record["node_device"])
                or record["mapped_device_minor"] != os.minor(record["node_device"])
            ):
                raise RuntimeError(
                    "PP16 HLO accelerator device mapping identity drifted"
                )
            if verify_live:
                _verify_accelerator_device_mapping_record_live(
                    record,
                    root_directory_fd=root_directory_fd,
                    device_directory_fd=device_directory_fd,
                )
            paths.append(path)
            identity = (
                record["mapped_device_major"],
                record["mapped_device_minor"],
                record["mapped_inode"],
            )
            if identity in identities:
                raise RuntimeError(
                    "PP16 HLO accelerator device mapping identity is duplicated"
                )
            identities.add(identity)
        if verify_live:
            _verify_accelerator_device_directory_named(
                root_directory_fd=root_directory_fd,
                device_directory_fd=device_directory_fd,
            )
    finally:
        if directory_fds is not None:
            os.close(device_directory_fd)
            os.close(root_directory_fd)
    if paths != sorted(set(paths), key=_canonical_path_key):
        raise RuntimeError("PP16 HLO accelerator device mapping order drifted")


def _accelerator_device_nodes_sha256(records: Any) -> str:
    return sha256(_canonical(records)).hexdigest()


def _validate_dependency_records(
    records: Any, category: str, *, verify_live: bool = False
) -> None:
    if not isinstance(records, list) or not records:
        raise RuntimeError(
            f"PP16 HLO compiler dependency records are absent: {category}"
        )
    paths = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "bytes",
            "device",
            "inode",
            "path",
            "sha256",
        }:
            raise RuntimeError(
                f"PP16 HLO compiler dependency record drifted: {category}"
            )
        path = record["path"]
        integers = (record["bytes"], record["device"], record["inode"])
        if (
            type(path) is not str
            or not path.startswith("/")
            or os.path.normpath(path) != path
            or os.path.realpath(path) != path
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in integers
            )
            or record["bytes"] < 0
            or record["device"] < 0
            or record["inode"] <= 0
            or type(record["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise RuntimeError(
                f"PP16 HLO compiler dependency identity drifted: {category}"
            )
        resolved = Path(path)
        allowed_roots = (
            PYTHON_RUNTIME_ROOT,
            JAX_SITE_ROOT,
            LIBTPU_SITE_ROOT,
            Path("/lib"),
            Path("/lib64"),
            Path("/usr"),
        )
        if not any(
            resolved == root or root in resolved.parents for root in allowed_roots
        ) and not (category == "python_modules" and resolved == COMPILER_DRIVER_PATH):
            raise RuntimeError(
                f"PP16 HLO compiler dependency escaped allowed roots: {category}"
            )
        if verify_live:
            _verify_dependency_record_live(record)
        paths.append(path)
    if paths != sorted(set(paths), key=_canonical_path_key):
        raise RuntimeError(f"PP16 HLO compiler dependency order drifted: {category}")


def _prepare_success(
    run_fd: int, *, code_pin: str, run_tag: str, remote: str, elapsed: int
) -> dict[str, bytes]:
    runner_raw = snapshot_member(run_fd, "runner.json")
    dependencies_raw = snapshot_member(run_fd, "dependencies.json")
    publisher_runtime_raw = snapshot_member(
        run_fd, "publisher_runtime.json", limit=1 << 20
    )
    runner = json.loads(runner_raw)
    dependencies = json.loads(dependencies_raw)
    if (
        runner.get("artifact_kind")
        != "gate_d_compensated_pp16_optimized_hlo_acquisition"
        or runner.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or runner.get("code_hash") != code_pin
        or runner.get("compile_only") is not True
        or runner.get("compiled_executable_invocation_count") != 0
        or runner.get("tpu_numerical_execution_performed") is not False
        or runner.get("persistent_compilation_cache_enabled") is not False
        or runner.get("gate_d_closed") is not False
        or runner.get("numerical_claim") is not False
        or runner.get("performance_claim") is not False
    ):
        raise RuntimeError("PP16 HLO runner claim boundary drifted")
    if not isinstance(dependencies, Mapping):
        raise RuntimeError("PP16 HLO compiler dependency manifest drifted")
    dependency_identity = runner.get("compiler_dependency_manifest", {})
    accelerator_device_nodes = dependencies.get(
        "accelerator_device_nodes_observed_mapped"
    )
    native_mappings = dependencies.get("native_mappings")
    python_modules = dependencies.get("python_modules")
    _validate_accelerator_device_mapping_records(
        accelerator_device_nodes, verify_live=True
    )
    _validate_dependency_records(native_mappings, "native_mappings", verify_live=True)
    _validate_dependency_records(python_modules, "python_modules", verify_live=True)
    dependency_source = dependencies.get("sealed_project_source")
    compiler_python_runtime = dependencies.get("python_runtime")
    compiler_dependency_sites = dependencies.get("dependency_sites")
    runner_source = runner.get("sealed_project_source")
    dependency_identity_keys = {
        "accelerator_device_node_count",
        "accelerator_device_nodes_sha256",
        "byte_count",
        "filename",
        "native_mapping_count",
        "python_module_count",
        "sha256",
    }
    dependency_count_fields = (
        "accelerator_device_node_count",
        "byte_count",
        "native_mapping_count",
        "python_module_count",
    )
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies)
        != {
            "accelerator_device_observation_scope",
            "accelerator_device_nodes_observed_mapped",
            "artifact_kind",
            "code_hash",
            "dependency_sites",
            "environment",
            "native_mappings",
            "python_modules",
            "python_runtime",
            "sealed_project_source",
        }
        or not isinstance(dependency_identity, Mapping)
        or set(dependency_identity) != dependency_identity_keys
        or any(
            type(dependency_identity.get(name)) is not int
            or dependency_identity[name] < 0
            for name in dependency_count_fields
        )
        or type(dependency_identity.get("accelerator_device_nodes_sha256")) is not str
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            dependency_identity["accelerator_device_nodes_sha256"],
        )
        or type(dependency_identity.get("sha256")) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", dependency_identity["sha256"])
        or dependencies.get("artifact_kind")
        != "gate_d_compensated_pp16_compiler_dependencies"
        or dependencies.get("accelerator_device_observation_scope")
        != _ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        or dependencies.get("code_hash") != code_pin
        or dependencies.get("environment") != _EXPECTED_COMPILER_ENVIRONMENT
        or compiler_python_runtime
        != {
            "python_executable": str(PYTHON),
            "python_runtime_root": str(PYTHON_RUNTIME_ROOT),
            "python_runtime_tree_sha256": PYTHON_RUNTIME_TREE_SHA256,
            "python_sha256": PYTHON_SHA256,
        }
        or compiler_dependency_sites
        != {
            "jax": {
                "manifest_sha256": JAX_SITE_MANIFEST_SHA256,
                "root": str(JAX_SITE_ROOT),
                "tree_sha256": JAX_SITE_TREE_SHA256,
            },
            "libtpu": {
                "manifest_sha256": LIBTPU_SITE_MANIFEST_SHA256,
                "root": str(LIBTPU_SITE_ROOT),
                "tree_sha256": LIBTPU_SITE_TREE_SHA256,
            },
        }
        or not isinstance(dependency_source, Mapping)
        or set(dependency_source)
        != {"archive_sha256", "file_manifest_count", "file_manifest_sha256"}
        or not isinstance(runner_source, Mapping)
        or any(
            runner_source.get(name) != value
            for name, value in dependency_source.items()
        )
        or dependency_identity.get("filename") != "dependencies.json"
        or dependency_identity.get("byte_count") != len(dependencies_raw)
        or dependency_identity.get("sha256") != sha256(dependencies_raw).hexdigest()
        or dependency_identity.get("accelerator_device_node_count")
        != len(accelerator_device_nodes)
        or dependency_identity.get("accelerator_device_nodes_sha256")
        != _accelerator_device_nodes_sha256(accelerator_device_nodes)
        or dependency_identity.get("native_mapping_count") != len(native_mappings)
        or dependency_identity.get("python_module_count") != len(python_modules)
    ):
        raise RuntimeError("PP16 HLO compiler dependency manifest drifted")
    expected_group = {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "local_device_count_visible": 4,
        "mesh_device_count": 2,
        "process_index": 0,
        "stage_id": 0,
    }
    if runner.get("physical_group") != expected_group:
        raise RuntimeError("PP16 HLO physical group drifted")
    hlo_raw: dict[str, bytes] = {}
    for kind, relative in (
        ("optimized", "hlo/compensated_pp16_stage0.optimized_hlo.txt"),
        ("stablehlo", "hlo/compensated_pp16_stage0.stablehlo.mlir"),
    ):
        raw = snapshot_member(run_fd, relative)
        identity = runner.get("hlo", {}).get(kind, {})
        if (
            identity.get("filename") != Path(relative).name
            or identity.get("byte_count") != len(raw)
            or identity.get("sha256") != sha256(raw).hexdigest()
        ):
            raise RuntimeError(f"PP16 {kind} identity drifted")
        hlo_raw[relative] = raw
    census_pre = snapshot_member(run_fd, "census_pre.txt")
    census_post = snapshot_member(run_fd, "census_post.txt")
    _require_census(census_pre, "CENSUS_OK")
    _require_census(census_post, "CENSUS_OK")
    orchestrator = snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
    write_member_exclusive(run_fd, "orchestrator.sealed.log", orchestrator)
    summary = {
        "artifact_kind": "gate_d_compensated_pp16_hlo_acquisition_summary",
        "claim_scope": runner["claim_scope"],
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "numerical_claim": False,
        "optimized_hlo_sha256": runner["hlo"]["optimized"]["sha256"],
        "performance_claim": False,
        "remote_prefix": remote,
        "run_tag": run_tag,
        "stablehlo_sha256": runner["hlo"]["stablehlo"]["sha256"],
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "tpu_numerical_execution_performed": False,
    }
    summary_raw = _canonical(summary)
    write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "runner.json": runner_raw,
        "dependencies.json": dependencies_raw,
        "publisher_runtime.json": publisher_runtime_raw,
        **hlo_raw,
        "census_pre.txt": census_pre,
        "census_post.txt": census_post,
        "orchestrator.sealed.log": orchestrator,
        "summary.json": summary_raw,
    }
    for relative in (
        "mirror.sha256",
        "remote_vacancy.raw.txt",
        "remote_vacancy.txt",
        "runner.log",
        "sync.txt",
    ):
        payload[relative] = snapshot_member(run_fd, relative, limit=256 << 20)
    evidence = {
        "artifact_kind": "gate_d_compensated_pp16_hlo_local_evidence",
        "code_hash": code_pin,
        "files": [_record(name, payload[name]) for name in sorted(payload)],
        "gate_d_closed": False,
        "numerical_claim": False,
        "performance_claim": False,
        "run_tag": run_tag,
        "status": "HLO_ACQUIRED_UNADJUDICATED",
    }
    evidence_raw = _canonical(evidence)
    write_member_exclusive(run_fd, "evidence.json", evidence_raw)
    payload["evidence.json"] = evidence_raw
    if tuple(sorted(payload)) != tuple(sorted(SUCCESS_PAYLOAD)):
        raise RuntimeError("PP16 HLO success payload inventory drifted")
    expected_local = set(SUCCESS_PAYLOAD) | {"orchestrator.log"}
    if local_members(run_fd) != expected_local:
        raise RuntimeError("PP16 HLO local preterminal inventory drifted")
    return payload


def publish_success(
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    elapsed: int,
    publication_runtime_raw: bytes,
    storage_bucket: Any | None = None,
    run_dir_fd: int | None = None,
) -> None:
    run_tag = validate_run_dir(run_dir)
    run_fd = _run_fd(run_dir, run_dir_fd)
    try:
        require_preterminal(run_fd)
        require_publication_runtime(run_fd, publication_runtime_raw)
        payload = _prepare_success(
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        bucket, prefix = _bucket_and_prefix(remote, storage_bucket)
        if _observed_names(bucket, prefix):
            raise RuntimeError("PP16 HLO remote prefix is not vacant")
        records = []
        for relative in sorted(payload):
            record = _upload_bound(bucket, prefix + relative, payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = _canonical(
            {
                "artifact_kind": "gate_d_compensated_pp16_hlo_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger_record = _upload_bound(
            bucket, prefix + "remote_objects.json", ledger_raw
        )
        ledger_record["path"] = "remote_objects.json"
        expected_names = set(payload) | {"remote_objects.json"}
        for record in records:
            _replay_bound(bucket, prefix + record["path"], record)
        _replay_bound(bucket, prefix + "remote_objects.json", ledger_record)
        if _observed_names(bucket, prefix) != expected_names:
            raise RuntimeError("PP16 HLO preterminal remote object set drifted")
        marker_base = {
            "adjudicated": False,
            "artifact_kind": "gate_d_compensated_pp16_hlo_acquired",
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "numerical_claim": False,
            "performance_claim": False,
            "remote_ledger": ledger_record,
            "run_tag": run_tag,
            "status": "HLO_ACQUIRED_UNADJUDICATED",
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
            "tpu_numerical_execution_performed": False,
        }
        marker_base["marker_payload_sha256"] = sha256(
            _canonical(marker_base)
        ).hexdigest()
        terminal_raw = _canonical(marker_base)
        write_member_exclusive(run_fd, "HLO_ACQUIRED", terminal_raw)
        expected_terminal_local = set(SUCCESS_PAYLOAD) | {
            "orchestrator.log",
            "remote_objects.json",
            "HLO_ACQUIRED",
        }
        if local_members(run_fd) != expected_terminal_local:
            raise RuntimeError("PP16 HLO local terminal inventory drifted")
        # This upload must remain the final remote mutation in the success path.
        terminal_record = _upload_bound(bucket, prefix + "HLO_ACQUIRED", terminal_raw)
        terminal_record["path"] = "HLO_ACQUIRED"
        _replay_bound(bucket, prefix + "HLO_ACQUIRED", terminal_record)
        if _observed_names(bucket, prefix) != expected_names | {"HLO_ACQUIRED"}:
            raise RuntimeError("PP16 HLO terminal remote object set drifted")
        write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            _canonical(
                {
                    "artifact_kind": "gate_d_compensated_pp16_hlo_terminal_receipt",
                    "remote": remote + "/HLO_ACQUIRED",
                    "terminal": terminal_record,
                }
            ),
        )
        if local_members(run_fd) != expected_terminal_local | {
            "terminal_upload_receipt.json"
        }:
            raise RuntimeError("PP16 HLO local terminal receipt inventory drifted")
    finally:
        os.close(run_fd)


def _walk_diagnostic_members(run_fd: int) -> list[str]:
    members = [
        relative
        for relative in sorted(local_members(run_fd))
        if relative
        not in {
            "orchestrator.log",
            "HLO_ACQUIRED",
            "diagnostic_objects.json",
        }
    ]
    if not members or len(members) > 64:
        raise RuntimeError("diagnostic member count is unsafe")
    return members


def publish_diagnostic(
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    status: int,
    publication_runtime_raw: bytes,
    storage_bucket: Any | None = None,
    run_dir_fd: int | None = None,
) -> None:
    run_tag = validate_run_dir(run_dir)
    run_fd = _run_fd(run_dir, run_dir_fd)
    try:
        require_preterminal(run_fd)
        require_publication_runtime(run_fd, publication_runtime_raw)
        failure_raw = _canonical(
            {
                "artifact_kind": "gate_d_compensated_pp16_hlo_failure",
                "code_hash": code_pin,
                "exit_status": status,
                "gate_d_closed": False,
                "run_tag": run_tag,
                "terminal_payload_present": False,
            }
        )
        write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
        write_member_exclusive(run_fd, "orchestrator.failure.log", orchestrator)
        members = _walk_diagnostic_members(run_fd)
        payload = {name: snapshot_member(run_fd, name) for name in members}
        bucket, prefix = _bucket_and_prefix(remote, storage_bucket)
        diagnostic_prefix = prefix + "diagnostic/"
        if _observed_names(bucket, diagnostic_prefix):
            raise RuntimeError("PP16 HLO diagnostic prefix is not vacant")
        records = []
        for relative in sorted(payload):
            record = _upload_bound(
                bucket, diagnostic_prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        for record in records:
            _replay_bound(bucket, diagnostic_prefix + record["path"], record)
        if _observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("PP16 HLO diagnostic object set drifted")
        ledger_raw = _canonical(
            {
                "artifact_kind": "gate_d_compensated_pp16_hlo_diagnostic_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        # No success terminal exists; the diagnostic ledger is the final mutation.
        ledger_record = _upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        ledger_record["path"] = "diagnostic_objects.json"
        _replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_record
        )
        if _observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("PP16 HLO diagnostic terminal object set drifted")
        write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            _canonical(
                {
                    "artifact_kind": (
                        "gate_d_compensated_pp16_hlo_diagnostic_terminal_receipt"
                    ),
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": ledger_record,
                }
            ),
        )
    finally:
        os.close(run_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-dir-fd", type=int)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--run-dir", type=Path, required=True)
    success = subparsers.add_parser("success")
    success.add_argument("--run-dir", type=Path, required=True)
    success.add_argument("--remote-prefix", required=True)
    success.add_argument("--elapsed", type=int, required=True)
    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--run-dir", type=Path, required=True)
    diagnostic.add_argument("--remote-prefix", required=True)
    diagnostic.add_argument("--status", type=int, required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument(
        "--member", choices=sorted(_WRAPPER_WRITE_MEMBERS), required=True
    )
    append = subparsers.add_parser("append-log")
    append.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    validate_environment()
    publication_runtime_raw = _canonical(validate_publication_runtime())
    verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    if arguments.mode != "init" and arguments.run_dir_fd != 7:
        raise RuntimeError("PP16 HLO publisher requires inherited run-directory fd 7")
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + initialize_run_dir(
                arguments.run_dir,
                publication_runtime_raw=publication_runtime_raw,
            ),
            flush=True,
        )
    elif arguments.mode == "success":
        publish_success(
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            elapsed=arguments.elapsed,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    elif arguments.mode == "diagnostic":
        publish_diagnostic(
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            status=arguments.status,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    elif arguments.mode == "write":
        run_fd = _run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            require_publication_runtime(run_fd, publication_runtime_raw)
            write_preterminal_member(run_fd, arguments.member, sys.stdin.buffer)
        finally:
            os.close(run_fd)
    else:
        run_fd = _run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            require_publication_runtime(run_fd, publication_runtime_raw)
            payload = sys.stdin.buffer.read((1 << 20) + 1)
            if len(payload) > 1 << 20:
                raise RuntimeError("orchestrator log append exceeds limit")
            append_preterminal_log(run_fd, payload)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
