#!/usr/bin/env -S /opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S
"""Produce one replayable Gate-D tuple-auxiliary eight-watchpoint capsule.

The tool is default-off and forced to exactly two CPU devices.  It packs only
the real, SHA-bound inputs needed for one layer-1/position-8155 replay, invokes
the committed tuple candidate and existing stage-local DSA composition, and
writes an append-only input NPZ, state NPZ, receipt, capsule, and SUCCESS seal.
It never creates infrastructure, loads the full model, runs a decoder, or uses
a TPU/cloud workflow.  Its output is precompile admission evidence only.
"""

from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import importlib.metadata
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any
import zipfile


REPO_ROOT = Path("/home/gianl/glm-tpu-topology-rewrite")
PRODUCER_INSTALLED = Path(
    "/opt/glm-tpu/bin/produce_gate_d_tuple_auxiliary_capsule.py"
)
PRODUCER_SOURCE = (
    REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py"
)
SITE_ROOT = Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e")
PYTHON_RUNTIME_ROOT = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
PYTHON = PYTHON_RUNTIME_ROOT / "bin/python3.12"
RUNTIME_ROOT = Path(
    "/home/gianl/glm-run/"
    "greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z/final"
)
DB518_RESULT = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/"
    "result.npz"
)
DB518_COMPARISON = DB518_RESULT.with_name("comparison.json")
DB550_BOUNDARY = Path(
    "/home/gianl/gcs-models/results/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
SOURCE_AUTHORITY = REPO_ROOT / "docs/artifacts/gate-d-tuple-auxiliary-source-authority.json"
PLAN_AUTHORITY = REPO_ROOT / "docs/artifacts/gate-d-tuple-auxiliary-pp16-plan-authority.json"
STABLEHLO_AUTHORITY = REPO_ROOT / "docs/artifacts/gate-d-tuple-auxiliary-stablehlo-authority.json"

RUNTIME_MANIFEST_FILE_SHA256 = (
    "e13ccefb7341756cd68d85e51209eaa8516ea506eac7ace3b4fd0a1d56828032"
)
RUNTIME_MANIFEST_SELF_SHA256 = (
    "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab"
)
RUNTIME_SUCCESS_SHA256 = (
    "dbef7e366e2fdf2a4815b0b58d1645667580d55fc5926133d48c838929f7ee7e"
)
DB518_RESULT_SHA256 = "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0"
DB518_COMPARISON_SHA256 = (
    "06ee82b9d487e3fdf8f9f19d4e824e33f1f9d453738a0090ace5e2cac7272a4d"
)
DB550_BOUNDARY_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
SOURCE_AUTHORITY_FILE_SHA256 = (
    "c95c8aa188e2eda77270022128d121dc3693a899607b1ab2f9977ddb45def0e1"
)
SOURCE_AUTHORITY_SHA256 = (
    "c1d84042ae98c3fb9d28794d7cdec757261d5d72e8600333563d72548570bded"
)
SOURCE_CODE_PIN = "c8b220067577004ddfb824667eadc746642baaca"
CANDIDATE_SOURCE_FILES = {
    "glm_tpu/greenfield/kernels/layer.py": (
        "bfaa79070b9f80825f3365e12e3ce95f67b4ef7b0d0a1a9d15621ce1c474b0ec"
    ),
    "glm_tpu/greenfield/kernels/reference/rmsnorm.py": (
        "751d946079f168257d63887bc2a1e4a0bd6b92785fbf3e5cd3635bfba59dd2d7"
    ),
}
PLAN_AUTHORITY_FILE_SHA256 = (
    "98b4fa21272e0bb019af1ad99abedee35593f9a1c8067a15983e9023c01f7880"
)
PLAN_SHA256 = "d824c19c4393e3b54767ea3a4a228bc865bac1b218fbce63eb04df7a05bc5833"
STABLEHLO_AUTHORITY_FILE_SHA256 = (
    "8cc45b81c3a85ad6131790e82eb73f0839cab7eefd13a634db7ff5a697b0338b"
)
STABLEHLO_AUTHORITY_SHA256 = (
    "19f43f3999bfab3bbe7b6f1e60aed6d643c7a14bf44a29726159726096516d88"
)
CLAIM_SCOPE = (
    "Bounded forced-two-CPU candidate-coherent layer-1/event-1 replay only; "
    "sealed real inputs are explicit, no full decoder or model load runs, and no "
    "TPU, cloud, performance or Gate-D claim is made. Local SUCCESS-last "
    "publication is provisional until an independent protected archive/seal."
)
POSITION = 8155
CURRENT_OWNER = 1
CURRENT_PAGE = 15
CURRENT_LOCAL_ROW = 219
_TENSOR_NAMES = (
    "attention.slot_01.input_norm",
    "attention.slot_01.qkv_a.weight_bits",
    "attention.slot_01.qkv_a.scale_inv",
    "attention.slot_01.q_a_norm",
    "indexer.slot_01.wq_b.weight_bits",
    "indexer.slot_01.wq_b.scale_inv",
    "indexer.slot_01.wk.weight_bits",
    "indexer.slot_01.wk.scale_inv",
    "indexer.slot_01.key_norm_weight",
    "indexer.slot_01.key_norm_bias",
    "indexer.slot_01.head_weight",
)
EXPECTED_PYTHON_SHA256 = (
    "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
)
EXPECTED_PYTHON_RUNTIME_TREE_SHA256 = (
    "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
)
EXPECTED_SITE_TREE_SHA256 = (
    "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
)
RISKY_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "HIP_",
    "HSA_",
    "JAX_",
    "LD_",
    "LIBTPU_",
    "NCCL_",
    "PJRT_",
    "PYTHON",
    "ROCM_",
    "TF_XLA_",
    "TPU_",
    "XLA_",
)
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_MAX_UPSTREAM_SNAPSHOT_BYTES = 512 << 20


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or (
            not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_mode
            & (stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
        ):
            raise RuntimeError(f"sealed runtime entry drifted: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(
                _file_sha256(entry)
            )
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"sealed runtime symlink escapes root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise RuntimeError(f"unsupported sealed runtime entry: {entry}")
        if os.listxattr(entry, follow_symlinks=False):
            raise RuntimeError(f"sealed runtime entry has extended attributes: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", stat.S_IMODE(metadata.st_mode) & ~0o7022))
        digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    return completed.stdout


def _sealed_git_source_archive(code_pin: str) -> tuple[int, str, dict[str, Any]]:
    tree = _git_bytes("ls-tree", "-r", "-z", code_pin, "--", "glm_tpu")
    records = []
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("committed source tree is invalid") from error
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise RuntimeError(f"committed source tree entry is unsupported: {path}")
        records.append(
            {
                "git_object_id": object_id.decode("ascii", errors="strict"),
                "mode": mode.decode("ascii", errors="strict"),
                "path": path,
            }
        )
    if not records or records != sorted(records, key=lambda item: item["path"]):
        raise RuntimeError("committed source tree order drifted")
    archive = _git_bytes("archive", "--format=zip", code_pin, "glm_tpu")
    if not archive:
        raise RuntimeError("committed source archive is empty")
    descriptor = os.memfd_create(
        "gate-d-capsule-source.zip", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    view = memoryview(archive)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
    if fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS:
        os.close(descriptor)
        raise RuntimeError("committed source archive seal drifted")
    observed = os.pread(descriptor, len(archive) + 1, 0)
    if observed != archive:
        os.close(descriptor)
        raise RuntimeError("committed source archive revalidation drifted")
    archive_path = f"/proc/self/fd/{descriptor}"
    manifest_payload = _canonical(records).encode("ascii")
    return descriptor, archive_path, {
        "archive_sha256": sha256(archive).hexdigest(),
        "file_manifest": {
            "count": len(records),
            "sha256": sha256(manifest_payload).hexdigest(),
        },
        "repository": {"commit": code_pin, "root": str(REPO_ROOT)},
    }


def _beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _immutable_file_record(path: Path) -> dict[str, Any]:
    resolved = Path(os.path.realpath(path))
    current = Path("/")
    for component in resolved.parts[1:]:
        current /= component
        metadata = current.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or metadata.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RuntimeError(f"loaded dependency is not immutable root:root: {resolved}")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & (
        stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    ):
        raise RuntimeError(f"loaded dependency type/mode is unsafe: {resolved}")
    if os.listxattr(resolved, follow_symlinks=False):
        raise RuntimeError(f"loaded dependency has extended attributes: {resolved}")
    return {
        "bytes": metadata.st_size,
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
    }


def _loaded_dependency_records(
    producer: Path, source_archive_path: str
) -> dict[str, list[dict[str, Any]]]:
    python_paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        if raw_path.startswith(source_archive_path + "/"):
            continue
        if not raw_path.startswith("/"):
            raise RuntimeError(f"loaded Python module has relative origin: {raw_path}")
        resolved = Path(os.path.realpath(raw_path))
        if not (
            _beneath(resolved, PYTHON_RUNTIME_ROOT)
            or _beneath(resolved, SITE_ROOT)
            or resolved == producer
        ):
            raise RuntimeError(f"loaded Python module escaped immutable roots: {resolved}")
        python_paths.add(resolved)
    native_paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError(f"loaded native dependency was deleted: {raw_path}")
        native_paths.add(Path(raw_path))
    return {
        "native_mappings": [
            _immutable_file_record(path) for path in sorted(native_paths)
        ],
        "python_modules": [
            _immutable_file_record(path) for path in sorted(python_paths)
        ],
    }


def _require_regular(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be one regular file")
    return metadata


def _open_regular_componentwise(path: Path, label: str) -> tuple[int, os.stat_result]:
    """Open one regular file without following any pathname component."""

    normalized = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", directory_flags)
    try:
        for component in normalized.parent.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            normalized.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(file_descriptor)
        raise RuntimeError(f"{label} must be one regular file")
    return file_descriptor, metadata


def _snapshot_regular(
    path: Path, expected_sha256: str, label: str, *, limit: int
) -> dict[str, Any]:
    descriptor, metadata = _open_regular_componentwise(path, label)
    try:
        raw = bytearray()
        digest = sha256()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            digest.update(block)
            if len(raw) > limit:
                raise RuntimeError(f"{label} exceeds the snapshot limit")
    finally:
        os.close(descriptor)
    if len(raw) != metadata.st_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"{label} snapshot identity drifted")
    return {
        "identity": (metadata.st_dev, metadata.st_ino),
        "raw": bytes(raw),
        "record": {
            "bytes": metadata.st_size,
            "path": str(path),
            "sha256": expected_sha256,
        },
    }


def _revalidate_snapshot(snapshot: dict[str, Any], label: str) -> None:
    record = snapshot["record"]
    descriptor, metadata = _open_regular_componentwise(Path(record["path"]), label)
    try:
        digest = sha256()
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            digest.update(block)
    finally:
        os.close(descriptor)
    if (
        (metadata.st_dev, metadata.st_ino) != snapshot["identity"]
        or total != record["bytes"]
        or digest.hexdigest() != record["sha256"]
    ):
        raise RuntimeError(f"{label} changed before publication")


def _write_exclusive(directory_fd: int, name: str, payload: bytes) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise RuntimeError("output member name is invalid")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_chain(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in Path(os.path.abspath(path)).parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise RuntimeError(
                    f"output parent chain is group/world writable: {path}"
                )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_output_directory(
    output: Path,
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    parent_fd = _open_directory_chain(output.parent)
    parent_metadata = os.fstat(parent_fd)
    parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    metadata = os.fstat(output_fd)
    return parent_fd, output_fd, parent_identity, (metadata.st_dev, metadata.st_ino)


def _verify_output_identity(
    parent_fd: int,
    output_fd: int,
    output_parent: Path,
    output_name: str,
    parent_identity: tuple[int, int],
    output_identity: tuple[int, int],
) -> None:
    parent_descriptor = os.fstat(parent_fd)
    parent_path = os.stat(output_parent, follow_symlinks=False)
    output_descriptor = os.fstat(output_fd)
    output_path = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        (parent_descriptor.st_dev, parent_descriptor.st_ino) != parent_identity
        or (parent_path.st_dev, parent_path.st_ino) != parent_identity
        or not stat.S_ISDIR(parent_path.st_mode)
        or (output_descriptor.st_dev, output_descriptor.st_ino) != output_identity
        or (output_path.st_dev, output_path.st_ino) != output_identity
        or not stat.S_ISDIR(output_path.st_mode)
    ):
        raise RuntimeError("append-only output directory identity drifted")


def _verify_output_member(
    directory_fd: int, name: str, expected_payload: bytes
) -> None:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd
    )
    try:
        metadata = os.fstat(descriptor)
        observed = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            observed.extend(block)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or bytes(observed) != expected_payload
        ):
            raise RuntimeError(f"published output member drifted: {name}")
    finally:
        os.close(descriptor)


def _deterministic_npz(values: dict[str, Any]) -> bytes:
    import numpy as np

    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in sorted(values.items()):
            if not name or "/" in name:
                raise RuntimeError("NPZ array name is invalid")
            array = np.ascontiguousarray(value)
            member = BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o400 << 16
            archive.writestr(
                info,
                member.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def _read_runtime(
    np: Any,
    ml_dtypes: Any,
    manifest_raw: bytes,
    success_raw: bytes,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, tuple[int, int]],
]:
    manifest = json.loads(manifest_raw)
    success = json.loads(success_raw)
    if manifest.get("manifest_sha256") != RUNTIME_MANIFEST_SELF_SHA256:
        raise RuntimeError("runtime manifest self identity drifted")
    if not isinstance(success, dict) or not success:
        raise RuntimeError("runtime SUCCESS contract drifted")
    records = {
        int(record["device_slot"]): record
        for record in manifest.get("files", ())
        if record.get("stage_id") == 0 and record.get("device_slot") in (0, 1)
    }
    if set(records) != {0, 1}:
        raise RuntimeError("runtime lacks exact stage-zero LP2 owners")
    dtypes = {
        "BF16": np.dtype(ml_dtypes.bfloat16),
        "F32": np.dtype("<f4"),
        "U8": np.dtype("u1"),
    }
    owners: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    owner_identities: dict[int, tuple[int, int]] = {}
    for slot in (0, 1):
        record = records[slot]
        path = RUNTIME_ROOT / str(record["destination_filename"])
        descriptor, metadata = _open_regular_componentwise(
            path, f"runtime owner {slot}"
        )
        owner_identities[slot] = (metadata.st_dev, metadata.st_ino)
        try:
            if metadata.st_size != int(record["file_bytes"]):
                raise RuntimeError(f"runtime owner {slot} file size drifted")
            header_bytes = int(record["header_bytes"])
            raw_header = os.pread(descriptor, header_bytes, 0)
            if sha256(raw_header).hexdigest() != record["header_sha256"]:
                raise RuntimeError(f"runtime owner {slot} header drifted")
            if struct.unpack("<Q", raw_header[:8])[0] + 8 != header_bytes:
                raise RuntimeError(f"runtime owner {slot} header length drifted")
            header = json.loads(raw_header[8:])
            tensor_records = {item["name"]: item for item in record["tensors"]}
            values: dict[str, Any] = {}
            for name in _TENSOR_NAMES:
                layout = header.get(name)
                tensor = tensor_records.get(name)
                if not isinstance(layout, dict) or not isinstance(tensor, dict):
                    raise RuntimeError(f"runtime owner {slot} lacks {name}")
                dtype = dtypes.get(layout.get("dtype"))
                if dtype is None:
                    raise RuntimeError(f"runtime owner {slot} {name} dtype drifted")
                start, end = (int(item) for item in layout["data_offsets"])
                shape = tuple(int(item) for item in layout["shape"])
                byte_count = end - start
                if (
                    byte_count != int(np.prod(shape)) * dtype.itemsize
                    or byte_count != tensor["byte_count"]
                ):
                    raise RuntimeError(f"runtime owner {slot} {name} bytes drifted")
                payload = os.pread(descriptor, byte_count, header_bytes + start)
                observed_sha = sha256(payload).hexdigest()
                if len(payload) != byte_count or observed_sha != tensor["sha256"]:
                    raise RuntimeError(f"runtime owner {slot} {name} payload drifted")
                values[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
                receipts.append(
                    {
                        "byte_count": byte_count,
                        "device_slot": slot,
                        "file_bytes": metadata.st_size,
                        "file_path": str(path),
                        "name": name,
                        "offset": header_bytes + start,
                        "sha256": observed_sha,
                        "shape": list(shape),
                    }
                )
        finally:
            os.close(descriptor)
        owners.append(values)
    for name in (
        "attention.slot_01.input_norm",
        "attention.slot_01.qkv_a.weight_bits",
        "attention.slot_01.qkv_a.scale_inv",
        "attention.slot_01.q_a_norm",
        "indexer.slot_01.wk.weight_bits",
        "indexer.slot_01.wk.scale_inv",
        "indexer.slot_01.key_norm_weight",
        "indexer.slot_01.key_norm_bias",
    ):
        if not np.array_equal(owners[0][name], owners[1][name]):
            raise RuntimeError(f"replicated runtime tensor {name} differs by owner")
    return owners, receipts, owner_identities


def _revalidate_tensor_receipts(
    receipts: list[dict[str, Any]], owner_identities: dict[int, tuple[int, int]]
) -> None:
    for receipt in receipts:
        path = Path(receipt["file_path"])
        descriptor, metadata = _open_regular_componentwise(
            path, f"runtime tensor {receipt['name']}"
        )
        try:
            payload = os.pread(descriptor, receipt["byte_count"], receipt["offset"])
        finally:
            os.close(descriptor)
        if (
            (metadata.st_dev, metadata.st_ino)
            != owner_identities[receipt["device_slot"]]
            or metadata.st_size != receipt["file_bytes"]
            or len(payload) != receipt["byte_count"]
            or sha256(payload).hexdigest() != receipt["sha256"]
        ):
            raise RuntimeError(f"runtime tensor payload drifted: {receipt['name']}")


def _input_values(
    np: Any,
    ml_dtypes: Any,
    owners: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
        derive_expected_dense_boundary_bits,
    )
    from glm_tpu.greenfield.kernels.stage_local import (
        STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
    )

    with np.load(
        BytesIO(snapshots["db550_boundary"]["raw"]), allow_pickle=False
    ) as values:
        model_axis_ids = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        hidden_update, _ = derive_expected_dense_boundary_bits(
            values["dense_virtual_partials_bfloat16_bits"],
            values["post_attention_residual_bfloat16_bits"],
            model_axis_ids,
        )
        residual = np.ascontiguousarray(values["post_attention_residual_bfloat16_bits"])
        rms_weight = np.ascontiguousarray(values["layer1_input_norm_bfloat16_bits"])
    comparison = json.loads(snapshots["db518_comparison"]["raw"])
    full_width = comparison.get("full_width_comparison", {})
    if (
        comparison.get("capture_sha256") != DB518_RESULT_SHA256
        or full_width.get("status") != "NUMERICAL_REJECTED"
        or full_width.get("mismatch_counts", {}).get("layer1_current_key_bfloat16_bits") != 0
    ):
        raise RuntimeError("DB518 cache/current-key provenance drifted")
    with np.load(
        BytesIO(snapshots["db518_result"]["raw"]), allow_pickle=False
    ) as values:
        prompt_cache = np.ascontiguousarray(
            values["layer1_index_cache_owners_bfloat16_bits"]
        ).copy()
    prompt_cache[CURRENT_OWNER, CURRENT_PAGE, CURRENT_LOCAL_ROW] = np.uint16(0)

    def stack(name: str) -> Any:
        return np.stack([owner[name] for owner in owners])

    return {
        "head_weight_bf16_bits": stack("indexer.slot_01.head_weight").view(np.uint16),
        "key_norm_bias_bf16_bits": stack("indexer.slot_01.key_norm_bias").view(np.uint16),
        "key_norm_weight_bf16_bits": stack("indexer.slot_01.key_norm_weight").view(np.uint16),
        "prompt_cache_bf16_bits": prompt_cache,
        "q_a_norm_bf16_bits": np.ascontiguousarray(
            owners[0]["attention.slot_01.q_a_norm"]
        ).view(np.uint16),
        "qkv_a_scale_inv": np.ascontiguousarray(
            owners[0]["attention.slot_01.qkv_a.scale_inv"], dtype=np.float32
        ),
        "qkv_a_weight_bits": np.ascontiguousarray(
            owners[0]["attention.slot_01.qkv_a.weight_bits"], dtype=np.uint8
        ),
        "rms_hidden_update_bf16_bits": hidden_update.reshape(6144),
        "rms_residual_bf16_bits": residual.reshape(6144),
        "rms_weight_bf16_bits": rms_weight.reshape(6144),
        "wk_scale_inv": np.ascontiguousarray(
            stack("indexer.slot_01.wk.scale_inv"), dtype=np.float32
        ),
        "wk_weight_bits": np.ascontiguousarray(
            stack("indexer.slot_01.wk.weight_bits"), dtype=np.uint8
        ),
        "wq_b_scale_inv": np.ascontiguousarray(
            stack("indexer.slot_01.wq_b.scale_inv"), dtype=np.float32
        ),
        "wq_b_weight_bits": np.ascontiguousarray(
            stack("indexer.slot_01.wq_b.weight_bits"), dtype=np.uint8
        ),
    }


def _require_finite_inputs(np: Any, inputs: dict[str, Any]) -> None:
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.dtype == np.dtype(np.uint16):
            if np.any((array & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                raise RuntimeError(f"capsule BF16 input is non-finite: {name}")
        elif array.dtype == np.dtype(np.float32) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"capsule FP32 input is non-finite: {name}")


def _require_finite_state(np: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        array = np.asarray(value)
        if array.dtype == np.dtype(np.uint16):
            if np.any((array & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                raise RuntimeError(f"capsule BF16 state is non-finite: {name}")
        elif array.dtype == np.dtype(np.float32) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"capsule FP32 state is non-finite: {name}")


def _execute(
    np: Any, ml_dtypes: Any, inputs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.benchmarking.gate_d_tuple_capsule import (
        build_gate_d_tuple_capsule_cpu_replay,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )

    if jax.default_backend() != "cpu" or len(jax.devices()) != 2:
        raise RuntimeError("capsule replay requires exactly two CPU devices")
    materialized_query = []
    materialized_wk = []
    for slot in range(2):
        materialized_query.append(
            np.asarray(
                dequantize_fp8_bits_block_weight(
                    jnp.asarray(inputs["wq_b_weight_bits"][slot]),
                    jnp.asarray(inputs["wq_b_scale_inv"][slot]),
                    output_dtype=jnp.float32,
                )
            )
        )
        wk_bf16 = decode_stage_local_prefill_index_wk_bf16(
            jnp.asarray(inputs["wk_weight_bits"][slot]),
            jnp.asarray(inputs["wk_scale_inv"][slot]),
        )
        materialized_wk.append(np.asarray(promote_stage_local_prefill_index_wk(wk_bf16)))
    replay = build_gate_d_tuple_capsule_cpu_replay(devices=jax.devices())
    result = replay(
        jnp.asarray(inputs["rms_hidden_update_bf16_bits"].view(ml_dtypes.bfloat16)[None, :]),
        jnp.asarray(inputs["rms_residual_bf16_bits"].view(ml_dtypes.bfloat16)[None, :]),
        jnp.asarray(inputs["rms_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["prompt_cache_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["qkv_a_weight_bits"]),
        jnp.asarray(inputs["qkv_a_scale_inv"]),
        jnp.asarray(inputs["q_a_norm_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["wq_b_weight_bits"]),
        jnp.asarray(inputs["wq_b_scale_inv"]),
        jnp.asarray(np.stack(materialized_query)),
        jnp.asarray(inputs["wk_weight_bits"]),
        jnp.asarray(inputs["wk_scale_inv"]),
        jnp.asarray(np.stack(materialized_wk)),
        jnp.asarray(inputs["key_norm_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["key_norm_bias_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["head_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
    )
    jax.block_until_ready(result)
    host = {field: np.asarray(getattr(result, field)) for field in result._fields}
    evidence_keys = (
        "rms_input_fp32_owners",
        "current_key_owners",
        "selected_positions_owners",
        "valid_counts_owners",
        "selected_scores_owners",
    )
    for key in evidence_keys:
        value = np.ascontiguousarray(host[key])
        if value.shape[0] != 2 or value[0].tobytes(order="C") != value[1].tobytes(
            order="C"
        ):
            raise RuntimeError(f"candidate capsule replicated owners disagree: {key}")
    contract_valid = np.ascontiguousarray(host["contract_valid_owners"], dtype=np.uint8)
    if contract_valid.shape != (2,) or contract_valid.tobytes(order="C") != b"\x01\x01":
        raise RuntimeError("candidate capsule replay failed its device contract")
    device_evidence = {
        key: np.ascontiguousarray(host[key]) for key in evidence_keys
    }
    device_evidence["contract_valid_owners"] = contract_valid
    state = {
        "cache_history": np.ascontiguousarray(host["index_cache_owners"]).view(np.uint16),
        "current_key": np.ascontiguousarray(
            host["current_key_owners"][0, 0], dtype=np.float32
        ),
        "event1_positions": np.ascontiguousarray(
            host["selected_positions_owners"][0], dtype=np.int32
        ),
        "event1_scores": np.ascontiguousarray(
            host["selected_scores_owners"][0], dtype=np.float32
        ),
        "event1_valid_count": np.ascontiguousarray(
            host["valid_counts_owners"][0], dtype=np.int32
        ),
        "head_weights": np.ascontiguousarray(host["head_weights_owners"], dtype=np.float32),
        "normalized": np.ascontiguousarray(host["normalized_hidden_owners"]).view(np.uint16),
        "query": np.ascontiguousarray(host["query_owners"], dtype=np.float32),
        "rms_hidden_update": inputs["rms_hidden_update_bf16_bits"],
        "rms_input": np.ascontiguousarray(
            host["rms_input_fp32_owners"][0, 0], dtype=np.float32
        ),
        "rms_residual": inputs["rms_residual_bf16_bits"],
    }
    return state, device_evidence


def _watchpoints(np: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    descriptions = {
        "layer1.rms_operands_bf16": [
            ("hidden_update", "rms_hidden_update", "bf16_bits", [], None, None, []),
            ("residual", "rms_residual", "bf16_bits", [], None, None, []),
        ],
        "layer1.rms_input_fp32": [("value", "rms_input", "float32", [], None, None, [])],
        "layer1.normalized": [
            (f"owner{slot}", "normalized", "bf16_bits", [slot, 0], slot, None, [])
            for slot in range(2)
        ],
        "layer1.cache_history": [
            ("value", "cache_history", "bf16_bits", [], None, 0, [0, 1])
        ],
        "layer1.query": [
            (f"owner{slot}", "query", "float32", [slot, 0], slot, None, [])
            for slot in range(2)
        ],
        "layer1.head_weights": [
            (f"owner{slot}", "head_weights", "float32", [slot, 0], slot, None, [])
            for slot in range(2)
        ],
        "layer1.current_key": [("value", "current_key", "float32", [], None, None, [])],
        "layer1.scorer_event1": [
            ("positions", "event1_positions", "int32", [], None, None, []),
            ("scores", "event1_scores", "float32", [], None, None, []),
            ("valid_count", "event1_valid_count", "int32", [], None, None, []),
        ],
    }
    storage = {
        np.dtype(np.float32): "<f4",
        np.dtype(np.int32): "<i4",
        np.dtype(np.uint16): "<u2",
    }
    result = []
    for watchpoint_id, specifications in descriptions.items():
        arrays = []
        for role, key, semantic_dtype, prefix, owner_id, owner_axis, axis_ids in specifications:
            selected = values[key][tuple(prefix)] if prefix else values[key]
            arrays.append(
                {
                    "array_key": key,
                    "array_sha256": _array_sha256(selected),
                    "index_prefix": prefix,
                    "owner_axis": owner_axis,
                    "owner_axis_ids": axis_ids,
                    "owner_id": owner_id,
                    "role": role,
                    "semantic_dtype": semantic_dtype,
                    "shape": list(selected.shape),
                    "storage_dtype": storage[selected.dtype],
                }
            )
        result.append(
            {
                "arrays": arrays,
                "id": watchpoint_id,
                "layer": 1,
                "layout": "lp2.local",
                "owner_ids": [0, 1],
                "position": POSITION,
            }
        )
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-pin", required=True)
    parser.add_argument("--expected-producer-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output = Path(os.path.abspath(args.output))
    if output.name in {"", ".", ".."}:
        raise RuntimeError("capsule output directory name is invalid")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"capsule output path is occupied: {output}")
    producer = Path(__file__).resolve(strict=True)
    if producer != PRODUCER_INSTALLED or Path(sys.executable).resolve() != PYTHON:
        raise RuntimeError("capsule producer is not using its sealed installed runtime")
    producer_metadata = producer.lstat()
    if (
        producer_metadata.st_uid != 0
        or producer_metadata.st_gid != 0
        or stat.S_IMODE(producer_metadata.st_mode) != 0o555
        or os.listxattr(producer, follow_symlinks=False)
    ):
        raise RuntimeError("installed capsule producer ownership/mode drifted")
    if (
        _file_sha256(PYTHON) != EXPECTED_PYTHON_SHA256
        or _runtime_tree_sha256(PYTHON_RUNTIME_ROOT)
        != EXPECTED_PYTHON_RUNTIME_TREE_SHA256
        or _runtime_tree_sha256(SITE_ROOT) != EXPECTED_SITE_TREE_SHA256
    ):
        raise RuntimeError("capsule sealed Python/JAX runtime drifted")
    code_pin = _git("rev-parse", "HEAD")
    if code_pin != args.expected_code_pin or _git("status", "--porcelain"):
        raise RuntimeError("capsule producer Git pin is dirty or unexpected")
    if (
        _file_sha256(producer) != args.expected_producer_sha256
        or _file_sha256(PRODUCER_SOURCE) != args.expected_producer_sha256
    ):
        raise RuntimeError("capsule producer bytes drifted")
    for repo_path, expected_sha in CANDIDATE_SOURCE_FILES.items():
        committed_source = _git_bytes("cat-file", "blob", f"{code_pin}:{repo_path}")
        if sha256(committed_source).hexdigest() != expected_sha:
            raise RuntimeError(f"reviewed candidate source drifted: {repo_path}")
    snapshot_specs = {
        "db518_comparison": (DB518_COMPARISON, DB518_COMPARISON_SHA256),
        "db518_result": (DB518_RESULT, DB518_RESULT_SHA256),
        "db550_boundary": (DB550_BOUNDARY, DB550_BOUNDARY_SHA256),
        "plan_authority": (PLAN_AUTHORITY, PLAN_AUTHORITY_FILE_SHA256),
        "runtime_manifest": (
            RUNTIME_ROOT / "runtime_manifest.json",
            RUNTIME_MANIFEST_FILE_SHA256,
        ),
        "runtime_success": (RUNTIME_ROOT / "SUCCESS", RUNTIME_SUCCESS_SHA256),
        "source_authority": (SOURCE_AUTHORITY, SOURCE_AUTHORITY_FILE_SHA256),
        "stablehlo_authority": (
            STABLEHLO_AUTHORITY,
            STABLEHLO_AUTHORITY_FILE_SHA256,
        ),
    }
    snapshots = {
        name: _snapshot_regular(
            path,
            expected_sha,
            f"upstream input {name}",
            limit=_MAX_UPSTREAM_SNAPSHOT_BYTES,
        )
        for name, (path, expected_sha) in snapshot_specs.items()
    }

    source_archive_fd, source_archive_path, source_snapshot = (
        _sealed_git_source_archive(code_pin)
    )
    sys.dont_write_bytecode = True
    for name in tuple(os.environ):
        if name.startswith(RISKY_ENVIRONMENT_PREFIXES):
            del os.environ[name]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(SITE_ROOT))
    sys.path.insert(0, source_archive_path)
    if importlib.util.find_spec("libtpu") is not None:
        raise RuntimeError("sealed capsule runtime unexpectedly exposes libtpu")
    if importlib.util.find_spec("jax_plugins") is not None:
        raise RuntimeError("sealed capsule runtime unexpectedly exposes jax_plugins")
    if importlib.metadata.entry_points(group="jax_plugins"):
        raise RuntimeError("sealed capsule runtime exposes a JAX plugin entry point")

    import jax
    import jaxlib
    import ml_dtypes
    import numpy as np

    if jax.__version__ != "0.10.1" or jaxlib.__version__ != "0.10.1":
        raise RuntimeError("capsule JAX toolchain drifted")
    if ml_dtypes.__version__ != "0.5.4" or np.__version__ != "2.3.5":
        raise RuntimeError("capsule numerical toolchain drifted")
    if "libtpu" in sys.modules or any(
        name == "jax_plugins" or name.startswith("jax_plugins.") for name in sys.modules
    ):
        raise RuntimeError("capsule producer loaded a forbidden TPU plugin")

    owners, tensor_receipts, runtime_owner_identities = _read_runtime(
        np,
        ml_dtypes,
        snapshots["runtime_manifest"]["raw"],
        snapshots["runtime_success"]["raw"],
    )
    inputs = _input_values(np, ml_dtypes, owners, snapshots)
    _require_finite_inputs(np, inputs)
    state, device_evidence = _execute(np, ml_dtypes, inputs)
    _require_finite_state(np, state)
    loaded_dependencies = _loaded_dependency_records(producer, source_archive_path)
    input_payload = _deterministic_npz(inputs)
    state_payload = _deterministic_npz(state)
    device_evidence_payload = _deterministic_npz(device_evidence)
    input_sha = sha256(input_payload).hexdigest()
    state_sha = sha256(state_payload).hexdigest()
    device_evidence_sha = sha256(device_evidence_payload).hexdigest()
    watchpoints = _watchpoints(np, state)
    manifest_records = sorted(
        (
            {**item, "arrays": sorted(item["arrays"], key=lambda value: value["role"])}
            for item in watchpoints
        ),
        key=lambda value: value["id"],
    )
    coherence_id = output.name
    producer_repo_path = PRODUCER_SOURCE.relative_to(REPO_ROOT).as_posix()
    producer_object = _git("rev-parse", f"{code_pin}:{producer_repo_path}")
    input_records = {
        name: {
            "array_sha256": _array_sha256(value),
            "shape": list(value.shape),
            "storage_dtype": {
                np.dtype(np.float32): "<f4",
                np.dtype(np.uint16): "<u2",
                np.dtype(np.uint8): "|u1",
            }[value.dtype],
        }
        for name, value in sorted(inputs.items())
    }
    tensor_receipts = sorted(
        tensor_receipts, key=lambda item: (item["device_slot"], item["name"])
    )
    upstream_inputs = {
        name: snapshot["record"] for name, snapshot in sorted(snapshots.items())
    }
    receipt = {
        "artifact": {"bytes": len(state_payload), "sha256": state_sha},
        "backend": {"device_count": 2, "device_ids": [0, 1], "platform": "cpu"},
        "candidate": {
            "code_pin": SOURCE_CODE_PIN,
            "id": "auxiliary_device_tuple_dependency",
            "plan_sha256": PLAN_SHA256,
            "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
            "stablehlo_authority_sha256": STABLEHLO_AUTHORITY_SHA256,
        },
        "claim_scope": CLAIM_SCOPE,
        "coherence_id": coherence_id,
        "environment": {
            "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
            "XLA_FLAGS": os.environ["XLA_FLAGS"],
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "ml_dtypes_version": ml_dtypes.__version__,
            "numpy_version": np.__version__,
            "python_executable": str(PYTHON),
            "python_runtime_root": str(PYTHON_RUNTIME_ROOT),
            "python_runtime_tree_sha256": EXPECTED_PYTHON_RUNTIME_TREE_SHA256,
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "python_version": sys.version,
            "site_root": str(SITE_ROOT),
            "site_tree_sha256": EXPECTED_SITE_TREE_SHA256,
        },
        "execution": {
            "cloud_workflow": False,
            "contract_valid": True,
            "decoder_executed": False,
            "jax_plugins_loaded": False,
            "libtpu_loaded": False,
            "model_loaded": False,
            "scope": "bounded.layer1.event1.candidate.replay",
            "tpus_used": 0,
        },
        "device_evidence": {
            "bytes": len(device_evidence_payload),
            "sha256": device_evidence_sha,
        },
        "input_arrays": input_records,
        "installed_producer": {
            "bytes": producer_metadata.st_size,
            "gid": producer_metadata.st_gid,
            "mode": stat.S_IMODE(producer_metadata.st_mode),
            "path": str(producer),
            "sha256": args.expected_producer_sha256,
            "uid": producer_metadata.st_uid,
        },
        "loaded_dependencies": loaded_dependencies,
        "producer": {
            "git_object_id": producer_object,
            "repo_path": producer_repo_path,
            "repository": {"commit": code_pin, "root": str(REPO_ROOT)},
            "sha256": args.expected_producer_sha256,
        },
        "schema_version": 1,
        "source_snapshot": source_snapshot,
        "tensor_receipts": tensor_receipts,
        "upstream_inputs": upstream_inputs,
        "watchpoint_manifest_sha256": sha256(
            _canonical({"watchpoints": manifest_records}).encode("ascii")
        ).hexdigest(),
    }
    receipt_payload = (_canonical(receipt) + "\n").encode("ascii")
    receipt_sha = sha256(receipt_payload).hexdigest()
    success = {
        "artifact_sha256": state_sha,
        "device_evidence_sha256": device_evidence_sha,
        "input_artifact_sha256": input_sha,
        "producer_receipt_sha256": receipt_sha,
        "schema_version": 1,
    }
    success_payload = (_canonical(success) + "\n").encode("ascii")
    success_sha = sha256(success_payload).hexdigest()
    capsule = {
        "artifact": {"path": "candidate-state.npz", "sha256": state_sha},
        "candidate_id": "auxiliary_device_tuple_dependency",
        "claim_scope": (
            "One bounded candidate-coherent forced-CPU replay capsule only; no TPU, "
            "full decoder, numerical acceptance, performance or Gate-D claim."
        ),
        "code_pin": SOURCE_CODE_PIN,
        "coherence_id": coherence_id,
        "plan_sha256": PLAN_SHA256,
        "producer_input_artifact": {"path": "candidate-inputs.npz", "sha256": input_sha},
        "producer_device_evidence": {
            "path": "candidate-device-evidence.npz",
            "sha256": device_evidence_sha,
        },
        "producer_receipt": {"path": "producer_receipt.json", "sha256": receipt_sha},
        "producer_success": {"path": "SUCCESS", "sha256": success_sha},
        "schema_version": 2,
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
        "stablehlo_authority_sha256": STABLEHLO_AUTHORITY_SHA256,
        "watchpoints": watchpoints,
    }
    capsule_payload = (_canonical(capsule) + "\n").encode("ascii")

    parent_fd, output_fd, parent_identity, output_identity = (
        _create_output_directory(output)
    )
    published = {
        "candidate-device-evidence.npz": device_evidence_payload,
        "candidate-inputs.npz": input_payload,
        "candidate-state.npz": state_payload,
        "capsule.json": capsule_payload,
        "producer_receipt.json": receipt_payload,
    }
    try:
        for name, payload in published.items():
            _write_exclusive(output_fd, name, payload)
        os.fsync(output_fd)
        _verify_output_identity(
            parent_fd,
            output_fd,
            output.parent,
            output.name,
            parent_identity,
            output_identity,
        )
        for name, payload in published.items():
            _verify_output_member(output_fd, name, payload)
        for name, snapshot in snapshots.items():
            _revalidate_snapshot(snapshot, f"upstream input {name}")
        _revalidate_tensor_receipts(tensor_receipts, runtime_owner_identities)
        if (
            _git("rev-parse", "HEAD") != code_pin
            or _git("status", "--porcelain")
            or _file_sha256(producer) != args.expected_producer_sha256
            or _file_sha256(PRODUCER_SOURCE) != args.expected_producer_sha256
            or _loaded_dependency_records(producer, source_archive_path)
            != loaded_dependencies
        ):
            raise RuntimeError("capsule execution identity changed before publication")
        _verify_output_identity(
            parent_fd,
            output_fd,
            output.parent,
            output.name,
            parent_identity,
            output_identity,
        )
        _write_exclusive(output_fd, "SUCCESS", success_payload)
        os.fsync(output_fd)
        os.fsync(parent_fd)
    finally:
        os.close(output_fd)
        os.close(parent_fd)
        os.close(source_archive_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
