#!/usr/bin/env python3
"""Acquire optimized TPU HLO for the admitted compensated Gate-D replay.

This process lowers and compiles exactly one two-chip PP16 stage-zero graph.
It creates only abstract arguments and never invokes the compiled executable.
The resulting HLO is deliberately unadjudicated until a separate offline
locality and numerical-policy inspection accepts it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
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
DRIVER_REPOSITORY_PATH = "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
ADMITTED_ID = "compensated_auxiliary_dependency"
ADMITTED_CLASSIFICATION = (
    "PRECOMPILE_LOGICAL_POLICY_PASSED;COMPILE_ONLY_REVIEW_ELIGIBLE;"
    "TPU_NUMERICAL_UNPROVEN;NO_TPU_EXECUTION_AUTHORIZED;GATE_D_OPEN"
)
CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
INPUT_SPEC = (
    ("rms_hidden_update_bf16", (1, 6144), "bfloat16"),
    ("rms_residual_bf16", (1, 6144), "bfloat16"),
    ("rms_weight_bf16", (6144,), "bfloat16"),
    ("prompt_cache_bf16", (2, 16, 256, 128), "bfloat16"),
    ("qkv_a_weight_bits", (32, 6144, 82), "uint8"),
    ("qkv_a_scale_inv", (32, 48, 82), "float32"),
    ("q_a_norm_bf16", (2048,), "bfloat16"),
    ("wq_b_weight_bits", (2, 2048, 2048), "uint8"),
    ("wq_b_scale_inv", (2, 16, 16), "float32"),
    ("wq_b_weight_fp32", (2, 2048, 2048), "float32"),
    ("wk_weight_bits", (2, 128, 6144), "uint8"),
    ("wk_scale_inv", (2, 1, 48), "float32"),
    ("wk_weight_fp32", (2, 128, 6144), "float32"),
    ("key_norm_weight_bf16", (2, 128), "bfloat16"),
    ("key_norm_bias_bf16", (2, 128), "bfloat16"),
    ("head_weight_bf16", (2, 16, 6144), "bfloat16"),
)
CAPSULE_INPUT_MAP = {
    "head_weight_bf16": "head_weight_bf16_bits",
    "key_norm_bias_bf16": "key_norm_bias_bf16_bits",
    "key_norm_weight_bf16": "key_norm_weight_bf16_bits",
    "prompt_cache_bf16": "prompt_cache_bf16_bits",
    "q_a_norm_bf16": "q_a_norm_bf16_bits",
    "qkv_a_scale_inv": "qkv_a_scale_inv",
    "qkv_a_weight_bits": "qkv_a_weight_bits",
    "rms_hidden_update_bf16": "rms_hidden_update_bf16_bits",
    "rms_residual_bf16": "rms_residual_bf16_bits",
    "rms_weight_bf16": "rms_weight_bf16_bits",
    "wk_scale_inv": "wk_scale_inv",
    "wk_weight_bits": "wk_weight_bits",
    "wq_b_scale_inv": "wq_b_scale_inv",
    "wq_b_weight_bits": "wq_b_weight_bits",
}
CAPSULE_SINGLE_ROW_INPUTS = {
    "rms_hidden_update_bf16",
    "rms_residual_bf16",
}
OUTPUT_SPEC = (
    ("rms_input_fp32_owners", (2, 1, 6144), "float32"),
    ("normalized_hidden_owners", (2, 1, 6144), "bfloat16"),
    ("query_owners", (2, 1, 32, 128), "float32"),
    ("head_weights_owners", (2, 1, 32), "float32"),
    ("current_key_owners", (2, 1, 128), "float32"),
    ("index_cache_owners", (2, 16, 256, 128), "bfloat16"),
    ("selected_positions_owners", (2, 1, 2048), "int32"),
    ("valid_counts_owners", (2, 1), "int32"),
    ("selected_scores_owners", (2, 1, 2048), "float32"),
    ("contract_valid_owners", (2,), "uint8"),
)
_COLLECTIVE_NAMES = (
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-permute",
    "reduce-scatter",
)
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
TAG_PATTERN = r"gate_d_compensated_pp16_hlo_[0-9]{8}T[0-9]{15}Z"
_EXPECTED_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "JAX_ENABLE_COMPILATION_CACHE": "0",
    "JAX_PLATFORMS": "tpu",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_VISIBLE_DEVICES": "0,1,2,3",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
}
_EXPECTED_RUNTIME_PATH = (
    str(PYTHON_RUNTIME_ROOT / "lib/python312.zip"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12/lib-dynload"),
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def _sha_file(path: Path) -> str:
    digest = sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"compiler runtime dependency is unsafe: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise RuntimeError(f"compiler runtime dependency changed: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _runtime_tree_sha256(root: Path, *, require_sealed: bool) -> str:
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
            raise RuntimeError(
                f"compiler Python runtime is not root-owned sealed: {entry}"
            )
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
                    f"compiler Python runtime symlink escapes root: {entry}"
                ) from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise RuntimeError(f"unsupported compiler Python runtime entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", _safe_mode(metadata.st_mode)))
        digest.update(payload)
    return digest.hexdigest()


def _validate_python_runtime_storage() -> dict[str, str]:
    if Path(sys.executable) != PYTHON:
        raise RuntimeError("Gate-D compiler Python executable boundary drifted")
    if _sha_file(PYTHON) != PYTHON_SHA256:
        raise RuntimeError("Gate-D compiler Python executable bytes drifted")
    if (
        _runtime_tree_sha256(PYTHON_RUNTIME_ROOT, require_sealed=True)
        != PYTHON_RUNTIME_TREE_SHA256
    ):
        raise RuntimeError("Gate-D compiler Python runtime tree drifted")
    return {
        "python_executable": str(PYTHON),
        "python_runtime_root": str(PYTHON_RUNTIME_ROOT),
        "python_runtime_tree_sha256": PYTHON_RUNTIME_TREE_SHA256,
        "python_sha256": PYTHON_SHA256,
    }


def _validate_python_runtime() -> dict[str, str]:
    if tuple(sys.path) != _EXPECTED_RUNTIME_PATH:
        raise RuntimeError("Gate-D compiler Python runtime boundary drifted")
    return _validate_python_runtime_storage()


def _validate_dependency_sites() -> dict[str, dict[str, str]]:
    result = {}
    for label, root, expected_tree, expected_manifest in (
        (
            "jax",
            JAX_SITE_ROOT,
            JAX_SITE_TREE_SHA256,
            JAX_SITE_MANIFEST_SHA256,
        ),
        (
            "libtpu",
            LIBTPU_SITE_ROOT,
            LIBTPU_SITE_TREE_SHA256,
            LIBTPU_SITE_MANIFEST_SHA256,
        ),
    ):
        if (
            _runtime_tree_sha256(root, require_sealed=True) != expected_tree
            or _sha_file(root / "CAPSULE_MANIFEST.json") != expected_manifest
        ):
            raise RuntimeError(f"Gate-D sealed {label} dependency site drifted")
        result[label] = {
            "manifest_sha256": expected_manifest,
            "root": str(root),
            "tree_sha256": expected_tree,
        }
    return result


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
                raise RuntimeError(f"unsafe output parent chain: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_exclusive(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish one file without following or replacing any pathname component."""

    if not path.name or path.name in {".", ".."}:
        raise RuntimeError("append-only output name is invalid")
    directory_fd = _open_directory_chain(path.parent)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
                dir_fd=directory_fd,
            )
        except FileExistsError as error:
            raise RuntimeError(f"append-only output already exists: {path}") from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"append-only output identity is unsafe: {path}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"append-only output write stalled: {path}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        observed = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
            or observed.st_nlink != 1
            or observed.st_size != len(payload)
        ):
            raise RuntimeError(f"append-only output identity changed: {path}")
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    return {"byte_count": len(payload), "sha256": sha256(payload).hexdigest()}


def _exclusive_text(path: Path, text: str) -> dict[str, Any]:
    return _write_exclusive(path, text.encode("utf-8"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _exclusive_text(path, _canonical(value) + "\n")


def _open_inherited_run_dir(run_dir: Path, inherited_fd: int) -> int:
    absolute = Path(os.path.abspath(run_dir))
    if absolute.parent != RUN_ROOT or not re.fullmatch(TAG_PATTERN, absolute.name):
        raise RuntimeError("Gate-D HLO run directory is outside its fixed root")
    root_fd = _open_directory_chain(RUN_ROOT)
    descriptor = -1
    try:
        root_metadata = os.fstat(root_fd)
        if (
            stat.S_IMODE(root_metadata.st_mode) != 0o700
            or root_metadata.st_uid != os.geteuid()
            or root_metadata.st_gid != os.getegid()
            or os.listxattr(root_fd)
        ):
            raise RuntimeError("Gate-D HLO run root authority drifted")
        descriptor = os.dup(inherited_fd)
        metadata = os.fstat(descriptor)
        observed = os.stat(absolute.name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
            or metadata.st_gid != os.getegid()
            or os.listxattr(descriptor)
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError("Gate-D HLO inherited run-directory identity drifted")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Gate-D HLO run-directory supervisor lock is absent"
            ) from error
        hlo_fd = os.open(
            "hlo",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            hlo_metadata = os.fstat(hlo_fd)
            if (
                stat.S_IMODE(hlo_metadata.st_mode) != 0o700
                or hlo_metadata.st_uid != os.geteuid()
                or hlo_metadata.st_gid != os.getegid()
                or os.listxattr(hlo_fd)
            ):
                raise RuntimeError("Gate-D HLO output directory authority drifted")
        finally:
            os.close(hlo_fd)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _write_run_member_exclusive(
    run_fd: int, relative: str, payload: bytes
) -> dict[str, Any]:
    parts = Path(relative).parts
    if (
        not parts
        or len(parts) > 2
        or any(part in {"", ".", ".."} for part in parts)
        or (len(parts) == 2 and parts[0] != "hlo")
    ):
        raise RuntimeError(f"Gate-D HLO output member is invalid: {relative}")
    parent_fd = run_fd
    owned_parent = False
    if len(parts) == 2:
        parent_fd = os.open(
            "hlo",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=run_fd,
        )
        owned_parent = True
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"append-only output already exists: {relative}"
            ) from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"append-only output identity is unsafe: {relative}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"append-only output write stalled: {relative}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        observed = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
            or observed.st_nlink != 1
            or observed.st_size != len(payload)
        ):
            raise RuntimeError(f"append-only output identity changed: {relative}")
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if owned_parent:
            os.close(parent_fd)
    return {"byte_count": len(payload), "sha256": sha256(payload).hexdigest()}


def _git_bytes(*arguments: str, repo: Path = REPO) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _git_text(*arguments: str, repo: Path = REPO) -> str:
    return _git_bytes(*arguments, repo=repo).decode("ascii", errors="strict").strip()


def _git_head() -> str:
    return _git_text("rev-parse", "HEAD")


def _snapshot_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"source file identity is unsafe: {path}")
        chunks = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        observed = os.stat(path, follow_symlinks=False)
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
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError(f"source file changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    raw = _snapshot_regular(Path(__file__))
    committed = _git_bytes("show", f"{code_pin}:{DRIVER_REPOSITORY_PATH}")
    if (
        raw != committed
        or sha256(raw).hexdigest() != expected_sha256
        or _git_text("rev-parse", f"{code_pin}^{{commit}}") != code_pin
    ):
        raise RuntimeError("Gate-D HLO acquisition driver is not the committed blob")


def _sealed_git_source_archive(
    code_pin: str, *, repo: Path = REPO
) -> tuple[int, str, dict[str, Any]]:
    """Create a sealed zipimport archive from exact committed Git blobs."""

    tree = _git_bytes("ls-tree", "-r", "-z", code_pin, "--", "glm_tpu", repo=repo)
    records = []
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for raw_entry in tree.split(b"\0"):
            if not raw_entry:
                continue
            try:
                metadata, raw_path = raw_entry.split(b"\t", 1)
                mode, kind, object_id = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8", errors="strict")
            except (UnicodeDecodeError, ValueError) as error:
                raise RuntimeError("committed source tree is invalid") from error
            if (
                kind != b"blob"
                or mode not in {b"100644", b"100755"}
                or not path.startswith("glm_tpu/")
            ):
                raise RuntimeError(f"unsupported committed source entry: {path}")
            payload = _git_bytes(
                "cat-file", "blob", object_id.decode("ascii"), repo=repo
            )
            record = {
                "git_object_id": object_id.decode("ascii"),
                "mode": mode.decode("ascii"),
                "path": path,
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
            }
            records.append(record)
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100555 if mode == b"100755" else 0o100444) << 16
            archive.writestr(info, payload)
    if not records or records != sorted(records, key=lambda item: item["path"]):
        raise RuntimeError("committed source tree order drifted")
    raw_archive = archive_buffer.getvalue()
    descriptor = os.memfd_create(
        "gate-d-pp16-hlo-source.zip", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    try:
        view = memoryview(raw_archive)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("committed source archive write stalled")
            view = view[written:]
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS:
            raise RuntimeError("committed source archive seal drifted")
        if os.pread(descriptor, len(raw_archive) + 1, 0) != raw_archive:
            raise RuntimeError("committed source archive revalidation drifted")
    except BaseException:
        os.close(descriptor)
        raise
    manifest = _canonical(records).encode("ascii")
    return (
        descriptor,
        f"/proc/self/fd/{descriptor}",
        {
            "archive_sha256": sha256(raw_archive).hexdigest(),
            "file_manifest_count": len(records),
            "file_manifest_sha256": sha256(manifest).hexdigest(),
        },
    )


def _install_sealed_source_path(source_archive_path: str) -> None:
    sys.path[:] = [
        source_archive_path,
        str(JAX_SITE_ROOT),
        str(LIBTPU_SITE_ROOT),
        *_EXPECTED_RUNTIME_PATH,
    ]


def _validate_compiler_import_path(source_archive_path: str) -> None:
    expected = (
        source_archive_path,
        str(JAX_SITE_ROOT),
        str(LIBTPU_SITE_ROOT),
        *_EXPECTED_RUNTIME_PATH,
    )
    if tuple(sys.path) != expected:
        raise RuntimeError("Gate-D compiler import path drifted")


def _verify_project_imports(source_archive_path: str) -> list[dict[str, str]]:
    records = []
    prefix = source_archive_path + "/"
    for name, module in sorted(sys.modules.items()):
        if name != "glm_tpu" and not name.startswith("glm_tpu."):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path.startswith(prefix):
            raise RuntimeError(f"mutable project import escaped sealed archive: {name}")
        records.append({"module": name, "path": raw_path})
    if not records:
        raise RuntimeError("sealed project import closure is empty")
    return records


def _validate_environment() -> dict[str, str]:
    for name, value in _EXPECTED_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Gate-D compiler environment drifted: {name}")
    extras = sorted(set(os.environ) - set(_EXPECTED_ENVIRONMENT))
    if extras:
        raise RuntimeError(
            f"Gate-D compiler environment has unexpected extras: {extras}"
        )
    return dict(sorted(_EXPECTED_ENVIRONMENT.items()))


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = _snapshot_regular(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"Gate-D {label} bytes drifted")
    return json.loads(raw.decode("ascii", errors="strict"))


def _canonical_path_key(path: Path) -> bytes:
    return os.fsencode(path.as_posix())


def _dependency_file_record(path: Path) -> dict[str, Any]:
    resolved = Path(os.path.realpath(path))
    descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"compiler dependency is not regular: {resolved}")
        digest = sha256()
        total = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            total += len(block)
            digest.update(block)
        observed = os.stat(resolved, follow_symlinks=False)
        if total != metadata.st_size or (observed.st_dev, observed.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise RuntimeError(f"compiler dependency changed: {resolved}")
        return {
            "bytes": total,
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "path": str(resolved),
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _allowed_compiler_dependency_roots() -> tuple[Path, ...]:
    return (
        PYTHON_RUNTIME_ROOT,
        JAX_SITE_ROOT,
        LIBTPU_SITE_ROOT,
        Path("/lib"),
        Path("/lib64"),
        Path("/usr"),
    )


def _validate_native_mapping_root(path: Path) -> None:
    if not any(
        path == root or root in path.parents
        for root in _allowed_compiler_dependency_roots()
    ):
        raise RuntimeError(f"compiler native mapping escaped allowed roots: {path}")


def _compiler_dependency_records(source_archive_path: str) -> dict[str, Any]:
    python_paths = {Path(__file__), Path(sys.executable)}
    source_prefix = source_archive_path + "/"
    allowed_roots = _allowed_compiler_dependency_roots()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or raw_path.startswith(source_prefix):
            continue
        if not raw_path.startswith("/"):
            raise RuntimeError(
                f"compiler Python module has relative origin: {raw_path}"
            )
        resolved = Path(os.path.realpath(raw_path))
        if resolved != Path(__file__).resolve() and not any(
            resolved == root or root in resolved.parents for root in allowed_roots
        ):
            raise RuntimeError(
                f"compiler Python module escaped allowed roots: {resolved}"
            )
        python_paths.add(resolved)
    native_mappings: dict[Path, tuple[int, int, int]] = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError(f"loaded compiler dependency was deleted: {raw_path}")
        path = Path(os.path.realpath(raw_path))
        _validate_native_mapping_root(path)
        try:
            metadata = path.stat()
        except OSError as error:
            raise RuntimeError(
                f"loaded compiler dependency vanished: {path}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            continue
        try:
            major_hex, minor_hex = fields[3].split(":", 1)
            mapped_identity = (
                int(major_hex, 16),
                int(minor_hex, 16),
                int(fields[4]),
            )
        except ValueError as error:
            raise RuntimeError("compiler native mapping identity is invalid") from error
        path_identity = (
            os.major(metadata.st_dev),
            os.minor(metadata.st_dev),
            metadata.st_ino,
        )
        if mapped_identity != path_identity:
            raise RuntimeError(f"compiler native mapping path was replaced: {path}")
        previous = native_mappings.setdefault(path, mapped_identity)
        if previous != mapped_identity:
            raise RuntimeError(f"compiler native mapping identity split: {path}")
    return {
        "native_mappings": [
            _dependency_file_record(path)
            for path in sorted(native_mappings, key=_canonical_path_key)
        ],
        "python_modules": [
            _dependency_file_record(path)
            for path in sorted(python_paths, key=_canonical_path_key)
        ],
    }


def _require_dependency_prefix_stable(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    for category in ("native_mappings", "python_modules"):
        before_by_path = {item["path"]: item for item in before[category]}
        after_by_path = {item["path"]: item for item in after[category]}
        if any(
            after_by_path.get(path) != record for path, record in before_by_path.items()
        ):
            raise RuntimeError(
                f"compiler dependency changed during compile: {category}"
            )


def validate_admission_and_topology(
    admission: Mapping[str, Any], topology: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact offline predecessor and PP16 stage-zero authority."""

    if (
        admission.get("admitted_candidate_ids") != [ADMITTED_ID]
        or admission.get("classification") != ADMITTED_CLASSIFICATION
        or admission.get("gate_d_closed") is not False
        or admission.get("jax_compile_or_tpu_work_performed") is not False
        or admission.get("tpu_successor_authorized") is not False
        or admission.get("compile_only_review_required") is not True
    ):
        raise RuntimeError("Gate-D compile-only admission predecessor drifted")
    physical = admission.get("physical_locality_authority")
    if not isinstance(physical, Mapping):
        raise TypeError("Gate-D physical-locality predecessor is absent")
    if topology.get("artifact_kind") != "gate_d_runtime_physical_locality_authority":
        raise RuntimeError("Gate-D topology authority kind drifted")
    if topology.get("topology_hash") != physical.get("topology_hash"):
        raise RuntimeError("Gate-D topology hash drifted")
    pp16 = topology.get("pp16_lp2")
    groups = pp16.get("groups") if isinstance(pp16, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 16:
        raise RuntimeError("Gate-D PP16 group authority drifted")
    stage_zero = groups[0]
    expected = {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    if stage_zero != expected:
        raise RuntimeError("Gate-D PP16 stage-zero authority drifted")
    if topology.get("tpu_successor_authorized") is not False:
        raise RuntimeError("topology authority unexpectedly authorizes TPU work")
    return expected


def validate_capsule_input_spec(admission: Mapping[str, Any]) -> dict[str, Any]:
    """Bind abstract compilation arguments to the admitted capsule inputs."""

    candidates = admission.get("candidate_results")
    if not isinstance(candidates, list):
        raise TypeError("Gate-D admission candidate catalogue is absent")
    matching = [item for item in candidates if item.get("id") == ADMITTED_ID]
    if len(matching) != 1:
        raise RuntimeError("Gate-D admitted candidate record is ambiguous")
    if (
        matching[0].get("admitted_precompile") is not True
        or matching[0].get("reasons") != []
    ):
        raise RuntimeError("Gate-D admitted candidate disposition drifted")
    capsule = matching[0].get("capsule")
    producer = capsule.get("producer") if isinstance(capsule, Mapping) else None
    if (
        not isinstance(producer, Mapping)
        or producer.get("backend")
        != {"device_count": 2, "device_ids": [0, 1], "platform": "cpu"}
        or capsule.get("producer_provenance_verified") is not True
    ):
        raise RuntimeError("Gate-D capsule producer authority drifted")
    arrays = producer.get("input_arrays") if isinstance(producer, Mapping) else None
    if not isinstance(arrays, Mapping) or set(arrays) != set(
        CAPSULE_INPUT_MAP.values()
    ):
        raise RuntimeError("Gate-D capsule input array catalogue drifted")
    specs = {name: (shape, dtype) for name, shape, dtype in INPUT_SPEC}
    storage_dtypes = {"bfloat16": "<u2", "float32": "<f4", "uint8": "|u1"}
    for compile_name, capsule_name in CAPSULE_INPUT_MAP.items():
        record = arrays.get(capsule_name)
        shape, dtype = specs[compile_name]
        capsule_shape = (
            shape[1:] if compile_name in CAPSULE_SINGLE_ROW_INPUTS else shape
        )
        if (
            not isinstance(record, Mapping)
            or record.get("shape") != list(capsule_shape)
            or record.get("storage_dtype") != storage_dtypes[dtype]
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("array_sha256", "")))
        ):
            raise RuntimeError(f"Gate-D capsule input spec drifted: {capsule_name}")
    for derived, source in (
        ("wq_b_weight_fp32", "wq_b_weight_bits"),
        ("wk_weight_fp32", "wk_weight_bits"),
    ):
        if specs[derived][0] != specs[source][0] or specs[derived][1] != "float32":
            raise RuntimeError(f"Gate-D materialized input spec drifted: {derived}")
    artifact_sha = producer.get("input_artifact_sha256")
    capsule_sha = capsule.get("capsule_sha256")
    if artifact_sha != CAPSULE_INPUT_SHA256 or capsule_sha != CAPSULE_SHA256:
        raise RuntimeError("Gate-D capsule input authority hashes drifted")
    return {
        "capsule_sha256": capsule_sha,
        "input_artifact_sha256": artifact_sha,
        "raw_input_array_count": len(arrays),
    }


def _memory_stats(device: Any) -> dict[str, int] | None:
    values = device.memory_stats()
    if values is None:
        return None
    return {
        str(name): int(value)
        for name, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _memory_analysis(compiled: Any) -> dict[str, int]:
    analysis = compiled.memory_analysis()
    fields = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        name: int(value)
        for name in fields
        if (value := getattr(analysis, name, None)) is not None
    }


def _abstract_record(tree: Any, jax: Any) -> list[dict[str, Any]]:
    leaves = jax.tree_util.tree_leaves(tree)
    names = [name for name, _, _ in OUTPUT_SPEC]
    if len(leaves) != len(names):
        raise RuntimeError("Gate-D TPU abstract output arity drifted")
    records = []
    for name, leaf, (_, expected_shape, expected_dtype) in zip(
        names, leaves, OUTPUT_SPEC, strict=True
    ):
        shape = tuple(int(item) for item in leaf.shape)
        dtype = str(leaf.dtype)
        if shape != expected_shape or dtype != expected_dtype:
            raise RuntimeError(
                f"Gate-D TPU abstract output drifted: {name} {shape} {dtype}"
            )
        records.append({"dtype": dtype, "name": name, "shape": list(shape)})
    return records


def _hlo_surface(optimized_hlo: str) -> dict[str, Any]:
    lines = optimized_hlo.splitlines()
    collectives = [
        line.strip()
        for line in lines
        if any(f" {name}(" in line or f"={name}(" in line for name in _COLLECTIVE_NAMES)
    ]
    replica_group_ids = sorted(
        {
            int(item)
            for line in collectives
            for group in re.findall(r"replica_groups=\{([^}]*)\}", line)
            for item in re.findall(r"\d+", group)
        }
    )
    return {
        "collective_count": len(collectives),
        "collective_lines": collectives,
        "forbidden_host_token_counts": {
            token: optimized_hlo.lower().count(token)
            for token in (
                "host_callback",
                "infeed",
                "outfeed",
                "recv-done",
                "send-done",
                "xla_python_cpu_callback",
            )
        },
        "module_header": lines[0] if lines else "",
        "replica_group_ids": replica_group_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--admission-report-sha256", required=True)
    parser.add_argument("--topology-authority", type=Path, required=True)
    parser.add_argument("--topology-authority-sha256", required=True)
    parser.add_argument("--compile-only", type=int, choices=(1,), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7,), required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("Gate-D HLO acquisition requires Python -I -S")
    python_runtime = _validate_python_runtime()
    dependency_sites = _validate_dependency_sites()
    if _git_head() != args.expected_code_hash:
        raise RuntimeError("Gate-D HLO acquisition code pin drifted")
    _verify_running_source(args.expected_code_hash, args.expected_driver_sha256)
    compiler_environment = _validate_environment()
    admission = _load_bound_json(
        args.admission_report, args.admission_report_sha256, "admission report"
    )
    topology = _load_bound_json(
        args.topology_authority, args.topology_authority_sha256, "topology authority"
    )
    stage_zero = validate_admission_and_topology(admission, topology)
    if (
        admission["physical_locality_authority"].get("authority_sha256")
        != args.topology_authority_sha256
    ):
        raise RuntimeError("Gate-D admission/topology file cross-binding drifted")
    capsule_input_authority = validate_capsule_input_spec(admission)
    run_fd = _open_inherited_run_dir(args.run_dir, args.run_dir_fd)

    source_archive_fd, source_archive_path, source_snapshot = (
        _sealed_git_source_archive(args.expected_code_hash)
    )
    _install_sealed_source_path(source_archive_path)
    _validate_compiler_import_path(source_archive_path)

    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.benchmarking.gate_d_compensated_pp16_hlo import (
        build_gate_d_compensated_pp16_hlo_replay,
    )

    loaded_project_modules = _verify_project_imports(source_archive_path)

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("Gate-D HLO acquisition requires one four-chip TPU-v4 host")
    if bool(jax.config.jax_enable_compilation_cache):
        raise RuntimeError("Gate-D HLO acquisition requires a disabled JAX cache")
    local_devices = tuple(jax.local_devices())
    devices = local_devices[:2]
    observed_group = {
        "coordinates": [list(device.coords) for device in devices],
        "device_ids": [int(device.id) for device in devices],
        "process_index": int(devices[0].process_index),
        "stage_id": 0,
    }
    if observed_group != stage_zero or any(
        int(device.process_index) != 0 or str(device.device_kind) != "TPU v4"
        for device in devices
    ):
        raise RuntimeError(
            f"Gate-D live PP16 stage-zero topology drifted: {observed_group}"
        )
    dtypes = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "int32": jnp.int32,
        "uint8": jnp.uint8,
    }
    arguments = tuple(
        jax.ShapeDtypeStruct(shape, dtypes[dtype]) for _, shape, dtype in INPUT_SPEC
    )
    replay = build_gate_d_compensated_pp16_hlo_replay(devices=devices)
    abstract_outputs = _abstract_record(jax.eval_shape(replay, *arguments), jax)
    dependencies_before = _compiler_dependency_records(source_archive_path)
    memory_before = [_memory_stats(device) for device in devices]
    lowering_started = time.monotonic()
    lowered = replay.lower(*arguments)
    lowering_seconds = time.monotonic() - lowering_started
    stablehlo = lowered.as_text()
    stablehlo_name = "compensated_pp16_stage0.stablehlo.mlir"
    stablehlo_identity = _write_run_member_exclusive(
        run_fd, f"hlo/{stablehlo_name}", stablehlo.encode("utf-8")
    )
    compile_started = time.monotonic()
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - compile_started
    optimized_hlo = compiled.as_text()
    optimized_name = "compensated_pp16_stage0.optimized_hlo.txt"
    optimized_identity = _write_run_member_exclusive(
        run_fd, f"hlo/{optimized_name}", optimized_hlo.encode("utf-8")
    )
    memory_after = [_memory_stats(device) for device in devices]
    if not optimized_hlo.startswith("HloModule "):
        raise RuntimeError("Gate-D optimized TPU HLO is absent or malformed")
    if (
        _verify_project_imports(source_archive_path) != loaded_project_modules
        or fcntl.fcntl(source_archive_fd, _F_GET_SEALS) != _MEMFD_SEALS
    ):
        raise RuntimeError("Gate-D sealed project import closure drifted")
    _validate_compiler_import_path(source_archive_path)
    if _validate_python_runtime_storage() != python_runtime:
        raise RuntimeError("Gate-D sealed Python runtime changed")
    if _validate_dependency_sites() != dependency_sites:
        raise RuntimeError("Gate-D sealed compiler dependency sites changed")
    dependencies_after = _compiler_dependency_records(source_archive_path)
    _require_dependency_prefix_stable(dependencies_before, dependencies_after)
    dependency_manifest = {
        "artifact_kind": "gate_d_compensated_pp16_compiler_dependencies",
        "code_hash": args.expected_code_hash,
        "dependency_sites": dependency_sites,
        "environment": compiler_environment,
        "native_mappings": dependencies_after["native_mappings"],
        "python_runtime": python_runtime,
        "python_modules": dependencies_after["python_modules"],
        "sealed_project_source": source_snapshot,
    }
    dependency_raw = (_canonical(dependency_manifest) + "\n").encode("ascii")
    dependency_identity = _write_run_member_exclusive(
        run_fd, "dependencies.json", dependency_raw
    )

    runtime = {
        "backend_platform": jax.default_backend(),
        "device_kind": str(devices[0].device_kind),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "libtpu": version("libtpu"),
        "platform_version": str(
            getattr(devices[0].client, "platform_version", "UNAVAILABLE")
        ),
    }
    result = {
        "admission_report_sha256": args.admission_report_sha256,
        "artifact_kind": "gate_d_compensated_pp16_optimized_hlo_acquisition",
        "claim_scope": (
            "One abstract-input optimized-HLO acquisition on exact adjacent PP16 "
            "stage-zero TPU-v4 devices. The compiled executable was never invoked; "
            "HLO is unadjudicated and proves no numerical, performance or Gate-D claim."
        ),
        "capsule_input_authority": capsule_input_authority,
        "code_hash": args.expected_code_hash,
        "compile_only": True,
        "compile_seconds": compile_seconds,
        "compiler_dependency_manifest": {
            **dependency_identity,
            "filename": "dependencies.json",
            "native_mapping_count": len(dependencies_after["native_mappings"]),
            "python_module_count": len(dependencies_after["python_modules"]),
        },
        "compiled_executable_invocation_count": 0,
        "gate_d_closed": False,
        "hlo": {
            "optimized": {
                "byte_count": optimized_identity["byte_count"],
                "filename": optimized_name,
                "sha256": optimized_identity["sha256"],
            },
            "stablehlo": {
                "byte_count": stablehlo_identity["byte_count"],
                "filename": stablehlo_name,
                "sha256": stablehlo_identity["sha256"],
            },
            "surface": _hlo_surface(optimized_hlo),
        },
        "input_spec": [
            {"dtype": dtype, "name": name, "shape": list(shape)}
            for name, shape, dtype in INPUT_SPEC
        ],
        "lowering_seconds": lowering_seconds,
        "memory_after_compile": memory_after,
        "memory_analysis": _memory_analysis(compiled),
        "memory_before_compile": memory_before,
        "numerical_claim": False,
        "output_spec": abstract_outputs,
        "performance_claim": False,
        "physical_group": {
            **observed_group,
            "local_device_count_visible": len(local_devices),
            "mesh_device_count": len(devices),
        },
        "runtime": runtime,
        "sealed_project_source": {
            **source_snapshot,
            "loaded_module_count": len(loaded_project_modules),
        },
        "persistent_compilation_cache_enabled": False,
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "topology_authority_sha256": args.topology_authority_sha256,
        "tpu_numerical_execution_performed": False,
    }
    _write_run_member_exclusive(
        run_fd, "runner.json", (_canonical(result) + "\n").encode("ascii")
    )
    os.close(run_fd)
    os.close(source_archive_fd)
    print(
        "GATE_D_COMPENSATED_PP16_HLO_ACQUIRED "
        f"optimized_sha256={result['hlo']['optimized']['sha256']} "
        "execution_count=0 adjudicated=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
