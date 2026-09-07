"""Atomic final-layout pack and direct-load metadata for the WS32 decoder.

The source checkpoint is never assembled as a model-sized host tree.  Each
source safetensors leaf is memory-mapped, sliced by the declarative WS32
placement ledger, and written directly into one of 32 final-owner files.
The manifest is the structural commit marker and is published only after every
source, destination file, and destination tensor checksum has been verified.
Protected direct load additionally requires its separately published,
self-hashed ``SUCCESS`` seal.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import base64
import binascii
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..partitioning.source_inventory import SourceFile, SourceInventory
from ..types import ModelGeometry
from .ws32_runtime import (
    Ws32RuntimePlacementReport,
    Ws32SourcePlacement,
    build_ws32_runtime_placement_report,
    placements_for_ws32_source_tensor,
)


WS32_RUNTIME_FORMAT_VERSION = 1
WS32_RUNTIME_ARTIFACT_KIND = "greenfield_ws32_runtime_checkpoint"
WS32_RUNTIME_SLOT_RECORD_KIND = "greenfield_ws32_runtime_slot_records"
WS32_RUNTIME_PLAN_ID = "WS32_2D"
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "U8": 1}
_MANIFEST_KEYS = frozenset(
    {
        "artifact_kind",
        "code_hash",
        "files",
        "format_version",
        "geometry",
        "geometry_sha256",
        "manifest_sha256",
        "mesh_hash",
        "packed_file_bytes",
        "packed_payload_bytes",
        "placement_report",
        "plan_id",
        "source",
        "tensor_schema",
    }
)
_FILE_RECORD_KEYS = frozenset(
    {
        "device_slot",
        "crc32c",
        "expert_coordinate",
        "feature_coordinate",
        "file_bytes",
        "filename",
        "header_bytes",
        "header_sha256",
        "payload_bytes",
        "sha256",
        "tensor_sha256",
    }
)
_TENSOR_SCHEMA_KEYS = frozenset(
    {"byte_count", "dtype", "global_shape", "local_shape", "name", "partition_spec"}
)
_SUCCESS_KEYS = frozenset(
    {
        "artifact_kind",
        "code_hash",
        "file_count",
        "format_version",
        "manifest_file_sha256",
        "manifest_sha256",
        "mesh_hash",
        "packed_payload_bytes",
        "performance_claim",
        "post_census_sha256",
        "remote_preflight_sha256",
        "remote_terminal_sha256",
        "source_file_count",
        "source_inventory_sha256",
        "success_sha256",
        "tag",
        "topology_hash",
        "tpu_initialized",
    }
)
_SUCCESS_ARTIFACT_KIND = "greenfield_ws32_runtime_checkpoint_success"
_SUCCESS_TAG = re.compile(
    r"greenfield_ws32_runtime_pack_[0-9]{8}T[0-9]{15}Z"
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: Mapping[str, Any], *, field: str) -> str:
    copy = dict(value)
    copy.pop(field, None)
    return sha256(_canonical_json(copy).encode("utf-8")).hexdigest()


def _digest(value: object, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase digest")
    return value


def _crc32c(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a base64 CRC32C")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{field} must be a base64 CRC32C") from exc
    if len(decoded) != 4:
        raise ValueError(f"{field} must encode four CRC32C bytes")
    return value


def _sha256_file(path: Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Ws32RuntimeTensorPlan:
    name: str
    dtype: str
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]
    partition_spec: tuple[str | None, ...]
    data_offset_start: int
    data_offset_end: int

    @property
    def byte_count(self) -> int:
        return self.data_offset_end - self.data_offset_start

    def schema_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "dtype": self.dtype,
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "name": self.name,
            "partition_spec": list(self.partition_spec),
        }


@dataclass(frozen=True, slots=True)
class Ws32RuntimeFilePlan:
    device_slot: int
    expert_coordinate: int
    feature_coordinate: int
    filename: str
    header: bytes
    tensors: tuple[Ws32RuntimeTensorPlan, ...]
    payload_bytes: int

    @property
    def file_bytes(self) -> int:
        return len(self.header) + self.payload_bytes


@dataclass(frozen=True, slots=True)
class Ws32RuntimePackConfig:
    source_root: Path
    source_uri: str
    output_dir: Path
    code_hash: str
    mesh_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.source_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("WS32 source URI must use the approved bucket")
        _digest(self.code_hash, field="code_hash", lengths=(40, 64))
        _digest(self.mesh_hash, field="mesh_hash")


@dataclass(frozen=True, slots=True)
class Ws32RuntimeMetadata:
    """Authenticated layout ledger; no claim about on-disk tensor payloads."""

    root: Path
    manifest: Mapping[str, Any]
    success: Mapping[str, Any]
    plans: tuple[Ws32RuntimeFilePlan, ...]
    records_by_slot: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class VerifiedWs32RuntimeCheckpoint(Ws32RuntimeMetadata):
    """Checkpoint admitted by the existing full-runtime verification policy."""


@dataclass(frozen=True, slots=True)
class LoadedWs32RuntimeCheckpoint:
    arrays: Mapping[str, Any]
    local_device_slots: tuple[Mapping[str, Any], ...]
    device_memory_before: tuple[Mapping[str, int] | None, ...]
    device_memory_after: tuple[Mapping[str, int] | None, ...]


def _header_bytes(
    *,
    slot: int,
    tensors: Sequence[Ws32RuntimeTensorPlan],
    source_inventory_sha256: str,
    geometry_sha256: str,
    placement_sha256: str,
    mesh_hash: str,
) -> bytes:
    value: dict[str, Any] = {
        "__metadata__": {
            "artifact_kind": WS32_RUNTIME_ARTIFACT_KIND,
            "device_slot": str(slot),
            "expert_coordinate": str(slot // 4),
            "feature_coordinate": str(slot % 4),
            "format_version": str(WS32_RUNTIME_FORMAT_VERSION),
            "geometry_sha256": geometry_sha256,
            "mesh_hash": mesh_hash,
            "placement_sha256": placement_sha256,
            "plan_id": WS32_RUNTIME_PLAN_ID,
            "source_inventory_sha256": source_inventory_sha256,
        }
    }
    for tensor in tensors:
        value[tensor.name] = {
            "data_offsets": [tensor.data_offset_start, tensor.data_offset_end],
            "dtype": tensor.dtype,
            "shape": list(tensor.local_shape),
        }
    raw = _canonical_json(value).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def build_ws32_runtime_file_plans(
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    mesh_hash: str,
) -> tuple[Ws32RuntimePlacementReport, tuple[Ws32RuntimeFilePlan, ...]]:
    """Build 32 deterministic final-owner safetensors layouts."""

    _digest(mesh_hash, field="mesh_hash")
    report = build_ws32_runtime_placement_report(inventory, geometry)
    schemas: dict[tuple[int, str], Ws32SourcePlacement] = {}
    for source in inventory.tensors:
        if source.layer_id is not None and source.layer_id >= geometry.num_layers:
            continue
        for placement in placements_for_ws32_source_tensor(source, geometry):
            key = (placement.slot, placement.destination_name)
            previous = schemas.setdefault(key, placement)
            if (
                previous.destination_dtype,
                previous.destination_shape,
                previous.global_shape,
                previous.partition_spec,
            ) != (
                placement.destination_dtype,
                placement.destination_shape,
                placement.global_shape,
                placement.partition_spec,
            ):
                raise CheckpointValidationError(
                    f"WS32 destination schema drifted for {key!r}"
                )
    plans = []
    for slot in range(32):
        offset = 0
        tensors = []
        for (record_slot, name), placement in sorted(schemas.items()):
            if record_slot != slot:
                continue
            byte_count = prod(placement.destination_shape) * _DTYPE_BYTES[
                placement.destination_dtype
            ]
            tensors.append(
                Ws32RuntimeTensorPlan(
                    name=name,
                    dtype=placement.destination_dtype,
                    local_shape=placement.destination_shape,
                    global_shape=placement.global_shape,
                    partition_spec=placement.partition_spec,
                    data_offset_start=offset,
                    data_offset_end=offset + byte_count,
                )
            )
            offset += byte_count
        if not tensors:
            raise CheckpointValidationError(f"WS32 slot {slot} has no tensors")
        header = _header_bytes(
            slot=slot,
            tensors=tensors,
            source_inventory_sha256=report.source_inventory_sha256,
            geometry_sha256=report.geometry_sha256,
            placement_sha256=report.placement_sha256,
            mesh_hash=mesh_hash,
        )
        plans.append(
            Ws32RuntimeFilePlan(
                device_slot=slot,
                expert_coordinate=slot // 4,
                feature_coordinate=slot % 4,
                filename=f"device_slot_{slot:02d}.safetensors",
                header=header,
                tensors=tuple(tensors),
                payload_bytes=offset,
            )
        )
    reference = tuple(item.schema_dict() for item in plans[0].tensors)
    if any(
        tuple(item.schema_dict() for item in plan.tensors) != reference
        for plan in plans[1:]
    ):
        raise CheckpointValidationError(
            "WS32 final-owner tensor schemas must be identical on all slots"
        )
    if any(plan.payload_bytes != report.bytes_by_slot[plan.device_slot] for plan in plans):
        raise CheckpointValidationError("WS32 file plans disagree with placement bytes")
    return report, tuple(plans)


def _verify_source_header(root: Path, record: SourceFile) -> None:
    path = root / record.filename
    if not path.is_file() or path.stat().st_size != record.file_bytes:
        raise CheckpointValidationError(
            f"WS32 source file size drifted for {record.filename!r}"
        )
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise CheckpointValidationError("WS32 source safetensors header is truncated")
        length = struct.unpack("<Q", prefix)[0]
        raw = stream.read(length)
    if 8 + length != record.header_bytes or sha256(raw).hexdigest() != record.header_sha256:
        raise CheckpointValidationError(
            f"WS32 source header identity drifted for {record.filename!r}"
        )


def _flat_contiguous_offset(
    shape: tuple[int, ...],
    starts: tuple[int, ...],
    stops: tuple[int, ...],
) -> tuple[int, int]:
    extents = tuple(stop - start for start, stop in zip(starts, stops, strict=True))
    partial = [
        index
        for index, (dimension, start, stop) in enumerate(
            zip(shape, starts, stops, strict=True)
        )
        if start != 0 or stop != dimension
    ]
    if partial:
        first = partial[0]
        if prod(extents[:first]) != 1 or any(
            starts[index] != 0 or stops[index] != shape[index]
            for index in range(first + 1, len(shape))
        ):
            raise CheckpointValidationError(
                "WS32 destination placement is not one contiguous interval"
            )
    strides = tuple(prod(shape[index + 1 :]) for index in range(len(shape)))
    flat_start = sum(
        start * stride for start, stride in zip(starts, strides, strict=True)
    )
    return flat_start, prod(extents)


def _pwrite_all(fd: int, value: bytes, offset: int) -> None:
    view = memoryview(value)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise CheckpointValidationError("WS32 destination write made no progress")
        written += count


def _validate_finite_chunk(raw: bytes, dtype: str) -> None:
    import numpy as np

    if dtype == "U8":
        values = np.frombuffer(raw, dtype=np.uint8)
        if bool(np.any((values == 0x7F) | (values == 0xFF))):
            raise CheckpointValidationError("WS32 FP8 payload contains non-finite bits")
        return
    if dtype == "F32":
        values = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "BF16":
        import ml_dtypes

        values = np.frombuffer(raw, dtype=ml_dtypes.bfloat16)
    else:
        raise CheckpointValidationError(f"unsupported WS32 dtype {dtype!r}")
    if not bool(np.all(np.isfinite(values))):
        raise CheckpointValidationError(f"WS32 {dtype} payload is non-finite")


def _destination_record(
    path: Path,
    plan: Ws32RuntimeFilePlan,
    *,
    chunk_bytes: int,
) -> dict[str, Any]:
    import google_crc32c

    file_digest = sha256()
    file_crc32c = google_crc32c.Checksum()
    tensor_hashes = []
    with path.open("rb", buffering=0) as stream:
        header = stream.read(len(plan.header))
        if header != plan.header:
            raise CheckpointValidationError(
                f"WS32 destination header drifted for {plan.filename!r}"
            )
        file_digest.update(header)
        file_crc32c.update(header)
        for tensor in plan.tensors:
            digest = sha256()
            remaining = tensor.byte_count
            alignment = _DTYPE_BYTES[tensor.dtype]
            while remaining:
                requested = min(remaining, chunk_bytes)
                requested -= requested % alignment
                if requested == 0:
                    requested = alignment
                raw = stream.read(requested)
                if len(raw) != requested:
                    raise CheckpointValidationError(
                        f"WS32 tensor {tensor.name!r} is truncated"
                    )
                _validate_finite_chunk(raw, tensor.dtype)
                digest.update(raw)
                file_digest.update(raw)
                file_crc32c.update(raw)
                remaining -= len(raw)
            tensor_hashes.append(digest.hexdigest())
        if stream.read(1):
            raise CheckpointValidationError(
                f"WS32 destination {plan.filename!r} has trailing bytes"
            )
    if path.stat().st_size != plan.file_bytes:
        raise CheckpointValidationError(
            f"WS32 destination size drifted for {plan.filename!r}"
        )
    return {
        "device_slot": plan.device_slot,
        "crc32c": base64.b64encode(file_crc32c.digest()).decode("ascii"),
        "expert_coordinate": plan.expert_coordinate,
        "feature_coordinate": plan.feature_coordinate,
        "file_bytes": plan.file_bytes,
        "filename": plan.filename,
        "header_bytes": len(plan.header),
        "header_sha256": sha256(plan.header).hexdigest(),
        "payload_bytes": plan.payload_bytes,
        "sha256": file_digest.hexdigest(),
        "tensor_sha256": tensor_hashes,
    }


def _write_ws32_slot_files(
    *,
    config: Ws32RuntimePackConfig,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    plans: Sequence[Ws32RuntimeFilePlan],
    chunk_bytes: int,
) -> list[dict[str, Any]]:
    selected_slots = {plan.device_slot for plan in plans}
    if not plans or len(selected_slots) != len(plans):
        raise CheckpointValidationError("WS32 slot pack selection is invalid")
    plan_by_slot = {plan.device_slot: plan for plan in plans}
    source_files = {record.filename: record for record in inventory.files}
    partial_paths = {
        plan.device_slot: config.output_dir / f".{plan.filename}.partial"
        for plan in plans
    }
    handles: dict[int, Any] = {}
    try:
        for plan in plans:
            handle = partial_paths[plan.device_slot].open("xb+", buffering=0)
            handles[plan.device_slot] = handle
            handle.write(plan.header)
            handle.truncate(plan.file_bytes)
        tensor_by_slot_name = {
            (plan.device_slot, tensor.name): tensor
            for plan in plans
            for tensor in plan.tensors
        }
        bytes_written = {key: 0 for key in tensor_by_slot_name}
        for source in inventory.tensors:
            if source.layer_id is not None and source.layer_id >= geometry.num_layers:
                continue
            selected_placements = tuple(
                placement
                for placement in placements_for_ws32_source_tensor(source, geometry)
                if placement.slot in selected_slots
            )
            if not selected_placements:
                continue
            import numpy as np

            source_file = source_files[source.filename]
            element_bytes = source.byte_count // prod(source.shape)
            mapped = np.memmap(
                config.source_root / source.filename,
                dtype=np.dtype(f"V{element_bytes}"),
                mode="r",
                offset=source_file.header_bytes + source.data_offset_start,
                shape=source.shape,
                order="C",
            )
            try:
                for placement in selected_placements:
                    slices = tuple(
                        slice(start, stop)
                        for start, stop in zip(
                            placement.source_starts,
                            placement.source_stops,
                            strict=True,
                        )
                    )
                    raw = mapped[slices].tobytes(order="C")
                    if len(raw) != placement.byte_count:
                        raise CheckpointValidationError(
                            f"WS32 source slice size drifted for {source.name!r}"
                        )
                    tensor = tensor_by_slot_name[
                        (placement.slot, placement.destination_name)
                    ]
                    flat_start, element_count = _flat_contiguous_offset(
                        placement.destination_shape,
                        placement.destination_starts,
                        placement.destination_stops,
                    )
                    if element_count * element_bytes != len(raw):
                        raise CheckpointValidationError(
                            "WS32 source/destination element size drifted"
                        )
                    plan = plan_by_slot[placement.slot]
                    _pwrite_all(
                        handles[placement.slot].fileno(),
                        raw,
                        len(plan.header)
                        + tensor.data_offset_start
                        + flat_start * element_bytes,
                    )
                    key = (placement.slot, placement.destination_name)
                    bytes_written[key] += len(raw)
            finally:
                del mapped
        if any(
            bytes_written[key] != tensor.byte_count
            for key, tensor in tensor_by_slot_name.items()
        ):
            raise CheckpointValidationError(
                "WS32 destination tensor coverage has a gap or overlap"
            )
        for plan in plans:
            handle = handles.pop(plan.device_slot)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            partial_paths[plan.device_slot].replace(
                config.output_dir / plan.filename
            )
    finally:
        for handle in handles.values():
            handle.close()
    return [
        _destination_record(
            config.output_dir / plan.filename,
            plan,
            chunk_bytes=chunk_bytes,
        )
        for plan in plans
    ]


def pack_ws32_runtime_slots(
    config: Ws32RuntimePackConfig,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    device_slots: Sequence[int],
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Pack an exact disjoint slot subset for an eight-host offline workflow."""

    if chunk_bytes <= 0:
        raise ValueError("WS32 pack chunk size must be positive")
    slots = tuple(device_slots)
    if (
        not slots
        or len(set(slots)) != len(slots)
        or any(
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < 32
            for slot in slots
        )
    ):
        raise ValueError("WS32 slot subset is invalid")
    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only WS32 slot destination exists: {config.output_dir}"
        )
    report, all_plans = build_ws32_runtime_file_plans(
        inventory, geometry, mesh_hash=config.mesh_hash
    )
    for record in inventory.files:
        _verify_source_header(config.source_root, record)
    config.output_dir.mkdir(parents=True)
    selected = tuple(all_plans[slot] for slot in sorted(slots))
    files = _write_ws32_slot_files(
        config=config,
        inventory=inventory,
        geometry=geometry,
        plans=selected,
        chunk_bytes=chunk_bytes,
    )
    result: dict[str, Any] = {
        "artifact_kind": WS32_RUNTIME_SLOT_RECORD_KIND,
        "code_hash": config.code_hash,
        "files": files,
        "format_version": WS32_RUNTIME_FORMAT_VERSION,
        "geometry_sha256": geometry.geometry_hash,
        "mesh_hash": config.mesh_hash,
        "placement_sha256": report.placement_sha256,
        "plan_id": WS32_RUNTIME_PLAN_ID,
        "slots": list(sorted(slots)),
        "source_inventory_sha256": inventory.inventory_sha256,
    }
    result["record_sha256"] = _mapping_hash(result, field="record_sha256")
    partial = config.output_dir / ".slot_records.json.partial"
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(config.output_dir / "slot_records.json")
    directory_fd = os.open(config.output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return result


def _build_ws32_runtime_manifest(
    *,
    config: Ws32RuntimePackConfig,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    report: Ws32RuntimePlacementReport,
    plans: Sequence[Ws32RuntimeFilePlan],
    files: Sequence[Mapping[str, Any]],
    source_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if set(source_file_sha256) != {record.filename for record in inventory.files}:
        raise CheckpointValidationError(
            "WS32 source SHA-256 ledger must cover every source file"
        )
    source_records = []
    for record in inventory.files:
        digest = _digest(
            source_file_sha256[record.filename],
            field=f"source:{record.filename}",
        )
        source_records.append({**record.to_dict(), "sha256": digest})
    manifest: dict[str, Any] = {
        "artifact_kind": WS32_RUNTIME_ARTIFACT_KIND,
        "code_hash": config.code_hash,
        "files": [dict(record) for record in files],
        "format_version": WS32_RUNTIME_FORMAT_VERSION,
        "geometry": geometry.to_dict(),
        "geometry_sha256": geometry.geometry_hash,
        "mesh_hash": config.mesh_hash,
        "packed_file_bytes": sum(plan.file_bytes for plan in plans),
        "packed_payload_bytes": report.packed_bytes,
        "placement_report": report.to_dict(),
        "plan_id": WS32_RUNTIME_PLAN_ID,
        "source": {
            "files": source_records,
            "inventory_sha256": inventory.inventory_sha256,
            "revision": inventory.source_revision,
            "uri": config.source_uri.rstrip("/"),
        },
        "tensor_schema": [item.schema_dict() for item in plans[0].tensors],
    }
    manifest["manifest_sha256"] = _mapping_hash(
        manifest, field="manifest_sha256"
    )
    by_slot = _verify_ws32_runtime_value(
        config.output_dir,
        manifest,
        plans,
    )
    _verify_ws32_runtime_files(
        Ws32RuntimeMetadata(config.output_dir, manifest, {}, tuple(plans), by_slot),
        verify_file_hashes=False,
        verify_file_hash_slots=None,
        local_slot_layout=False,
    )
    return manifest


def _commit_ws32_runtime_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    path = root / "manifest.json"
    if path.exists():
        raise FileExistsError(f"append-only WS32 manifest exists: {path}")
    partial = root / ".manifest.json.partial"
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def finalize_ws32_runtime_checkpoint(
    config: Ws32RuntimePackConfig,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    file_records: Sequence[Mapping[str, Any]],
    source_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate 8 disjoint host results and commit the structural manifest."""

    if not config.output_dir.is_dir():
        raise FileNotFoundError("WS32 packed slot directory is unavailable")
    report, plans = build_ws32_runtime_file_plans(
        inventory, geometry, mesh_hash=config.mesh_hash
    )
    manifest = _build_ws32_runtime_manifest(
        config=config,
        inventory=inventory,
        geometry=geometry,
        report=report,
        plans=plans,
        files=file_records,
        source_file_sha256=source_file_sha256,
    )
    _commit_ws32_runtime_manifest(config.output_dir, manifest)
    return manifest


def pack_ws32_runtime_checkpoint(
    config: Ws32RuntimePackConfig,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Pack the complete base model and publish ``manifest.json`` last."""

    if chunk_bytes <= 0:
        raise ValueError("WS32 pack chunk size must be positive")
    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only WS32 destination exists: {config.output_dir}"
        )
    report, plans = build_ws32_runtime_file_plans(
        inventory, geometry, mesh_hash=config.mesh_hash
    )
    for record in inventory.files:
        _verify_source_header(config.source_root, record)
    source_file_sha256 = {
        record.filename: _sha256_file(config.source_root / record.filename)
        for record in inventory.files
    }
    config.output_dir.mkdir(parents=True)
    files = _write_ws32_slot_files(
        config=config,
        inventory=inventory,
        geometry=geometry,
        plans=plans,
        chunk_bytes=chunk_bytes,
    )
    manifest = _build_ws32_runtime_manifest(
        config=config,
        inventory=inventory,
        geometry=geometry,
        report=report,
        plans=plans,
        files=files,
        source_file_sha256=source_file_sha256,
    )
    _commit_ws32_runtime_manifest(config.output_dir, manifest)
    return manifest


def _verify_ws32_runtime_value(
    root: Path,
    manifest: Mapping[str, Any],
    plans: Sequence[Ws32RuntimeFilePlan],
) -> Mapping[int, Mapping[str, Any]]:
    if set(manifest) != _MANIFEST_KEYS:
        raise CheckpointValidationError("WS32 runtime manifest schema drifted")
    if manifest.get("artifact_kind") != WS32_RUNTIME_ARTIFACT_KIND or (
        manifest.get("format_version") != WS32_RUNTIME_FORMAT_VERSION
    ) or manifest.get("plan_id") != WS32_RUNTIME_PLAN_ID:
        raise CheckpointValidationError("unsupported WS32 runtime checkpoint")
    if manifest.get("manifest_sha256") != _mapping_hash(
        manifest, field="manifest_sha256"
    ):
        raise CheckpointValidationError("WS32 runtime manifest checksum mismatch")
    tensor_schema = manifest.get("tensor_schema")
    if not isinstance(tensor_schema, list) or any(
        not isinstance(item, Mapping) or set(item) != _TENSOR_SCHEMA_KEYS
        for item in tensor_schema
    ) or tensor_schema != [
        item.schema_dict() for item in plans[0].tensors
    ]:
        raise CheckpointValidationError("WS32 runtime tensor schema drifted")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 32:
        raise CheckpointValidationError("WS32 runtime requires 32 file records")
    by_slot = {record.get("device_slot"): record for record in records}
    if set(by_slot) != set(range(32)) or len(by_slot) != len(records):
        raise CheckpointValidationError("WS32 runtime file slots are incomplete")
    for plan in plans:
        record = by_slot[plan.device_slot]
        if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_KEYS:
            raise CheckpointValidationError("WS32 runtime file record schema drifted")
        expected = {
            "device_slot": plan.device_slot,
            "expert_coordinate": plan.expert_coordinate,
            "feature_coordinate": plan.feature_coordinate,
            "file_bytes": plan.file_bytes,
            "filename": plan.filename,
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "payload_bytes": plan.payload_bytes,
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise CheckpointValidationError(
                f"WS32 runtime file metadata drifted for slot {plan.device_slot}"
            )
        _digest(record.get("sha256"), field=f"slot{plan.device_slot}.sha256")
        _crc32c(record.get("crc32c"), field=f"slot{plan.device_slot}.crc32c")
        hashes = record.get("tensor_sha256")
        if not isinstance(hashes, list) or len(hashes) != len(plan.tensors):
            raise CheckpointValidationError("WS32 runtime tensor hash ledger drifted")
        for index, digest in enumerate(hashes):
            _digest(digest, field=f"slot{plan.device_slot}.tensor{index}")
    if manifest.get("packed_payload_bytes") != sum(
        plan.payload_bytes for plan in plans
    ) or manifest.get("packed_file_bytes") != sum(
        plan.file_bytes for plan in plans
    ):
        raise CheckpointValidationError("WS32 runtime byte totals drifted")
    return by_slot


def _verify_ws32_runtime_files(
    metadata: Ws32RuntimeMetadata,
    *,
    verify_file_hashes: bool,
    verify_file_hash_slots: frozenset[int] | None,
    local_slot_layout: bool,
) -> None:
    if local_slot_layout and (
        not verify_file_hashes or verify_file_hash_slots is None
    ):
        raise ValueError(
            "WS32 local slot layout requires hash verification of the owned slots"
        )
    root = metadata.root
    for plan in metadata.plans:
        record = metadata.records_by_slot[plan.device_slot]
        path = root / plan.filename
        if local_slot_layout and plan.device_slot not in verify_file_hash_slots:
            # Streaming tmpfs layout: only this host's owned slots are materialized.
            # The manifest record was fully checked above; a non-owned file that is
            # nevertheless present is refused so a root cannot mix layouts.
            if path.exists():
                raise CheckpointValidationError(
                    f"WS32 local slot layout must not contain foreign slot {plan.filename!r}"
                )
            continue
        if not path.is_file() or path.stat().st_size != plan.file_bytes:
            raise CheckpointValidationError(
                f"WS32 runtime file is missing or truncated: {plan.filename!r}"
            )
        if verify_file_hashes and (
            verify_file_hash_slots is None
            or plan.device_slot in verify_file_hash_slots
        ):
            observed = _destination_record(
                path,
                plan,
                chunk_bytes=64 * 1024 * 1024,
            )
            if observed != record:
                raise CheckpointValidationError(
                    f"WS32 runtime file/tensor checksum drifted: {plan.filename!r}"
                )


def verify_ws32_runtime_checkpoint(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_success_sha256: str,
    expected_mesh_hash: str,
    expected_topology_hash: str,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    verify_file_hashes: bool = True,
    verify_file_hash_slots: Sequence[int] | None = None,
    local_slot_layout: bool = False,
) -> VerifiedWs32RuntimeCheckpoint:
    """Re-derive every layout field and verify a protected sealed artifact."""

    _digest(expected_manifest_sha256, field="expected_manifest_sha256")
    _digest(expected_success_sha256, field="expected_success_sha256")
    _digest(expected_mesh_hash, field="expected_mesh_hash")
    _digest(expected_topology_hash, field="expected_topology_hash")
    selected_hash_slots = None
    if verify_file_hash_slots is not None:
        selected_hash_slots = frozenset(verify_file_hash_slots)
        if (
            not selected_hash_slots
            or len(selected_hash_slots) != len(verify_file_hash_slots)
            or any(
                not isinstance(slot, int)
                or isinstance(slot, bool)
                or not 0 <= slot < 32
                for slot in selected_hash_slots
            )
        ):
            raise ValueError("WS32 runtime verification slot subset is invalid")
    metadata = _read_ws32_runtime_metadata(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_success_sha256=expected_success_sha256,
        expected_mesh_hash=expected_mesh_hash,
        expected_topology_hash=expected_topology_hash,
        inventory=inventory,
        geometry=geometry,
    )
    _verify_ws32_runtime_files(
        metadata,
        verify_file_hashes=verify_file_hashes,
        verify_file_hash_slots=selected_hash_slots,
        local_slot_layout=local_slot_layout,
    )
    return VerifiedWs32RuntimeCheckpoint(
        root=metadata.root,
        manifest=metadata.manifest,
        success=metadata.success,
        plans=metadata.plans,
        records_by_slot=metadata.records_by_slot,
    )


def _read_ws32_runtime_metadata(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_success_sha256: str,
    expected_mesh_hash: str,
    expected_topology_hash: str,
    inventory: SourceInventory,
    geometry: ModelGeometry,
) -> Ws32RuntimeMetadata:
    """Re-derive/authenticate all metadata without opening any owner payload."""

    _digest(expected_manifest_sha256, field="expected_manifest_sha256")
    _digest(expected_success_sha256, field="expected_success_sha256")
    _digest(expected_mesh_hash, field="expected_mesh_hash")
    _digest(expected_topology_hash, field="expected_topology_hash")
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise CheckpointValidationError("WS32 runtime checkpoint lacks manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise CheckpointValidationError("WS32 runtime manifest identity drifted")
    if manifest.get("mesh_hash") != expected_mesh_hash:
        raise CheckpointValidationError("WS32 runtime mesh identity drifted")
    report, plans = build_ws32_runtime_file_plans(
        inventory, geometry, mesh_hash=expected_mesh_hash
    )
    if manifest.get("geometry") != geometry.to_dict() or (
        manifest.get("geometry_sha256") != geometry.geometry_hash
    ) or manifest.get("placement_report") != report.to_dict():
        raise CheckpointValidationError("WS32 runtime geometry/placement drifted")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "files",
        "inventory_sha256",
        "revision",
        "uri",
    } or source.get(
        "inventory_sha256"
    ) != inventory.inventory_sha256 or source.get("revision") != inventory.source_revision:
        raise CheckpointValidationError("WS32 runtime source identity drifted")
    if not isinstance(source.get("uri"), str) or not source["uri"].startswith(
        "gs://driftbench-dsv4-uc/"
    ):
        raise CheckpointValidationError("WS32 runtime source URI drifted")
    _digest(manifest.get("code_hash"), field="code_hash", lengths=(40, 64))
    source_records = source.get("files")
    if not isinstance(source_records, list) or len(source_records) != len(inventory.files):
        raise CheckpointValidationError("WS32 runtime source file ledger drifted")
    for expected, observed in zip(inventory.files, source_records, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != {
            *expected.to_dict(),
            "sha256",
        }:
            raise CheckpointValidationError("WS32 source file record schema drifted")
        for field, value in expected.to_dict().items():
            if observed.get(field) != value:
                raise CheckpointValidationError(
                    f"WS32 source ledger drifted for {expected.filename!r}"
                )
        _digest(observed.get("sha256"), field=f"source:{expected.filename}")
    success_path = root / "SUCCESS"
    if not success_path.is_file():
        raise CheckpointValidationError("WS32 runtime checkpoint lacks SUCCESS")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(success, Mapping) or set(success) != _SUCCESS_KEYS:
        raise CheckpointValidationError("WS32 runtime SUCCESS schema drifted")
    if success.get("success_sha256") != expected_success_sha256 or (
        success.get("success_sha256")
        != _mapping_hash(success, field="success_sha256")
    ):
        raise CheckpointValidationError("WS32 runtime SUCCESS checksum mismatch")
    exact_success = {
        "artifact_kind": _SUCCESS_ARTIFACT_KIND,
        "code_hash": manifest["code_hash"],
        "file_count": 32,
        "format_version": WS32_RUNTIME_FORMAT_VERSION,
        "manifest_file_sha256": _sha256_file(path),
        "manifest_sha256": expected_manifest_sha256,
        "mesh_hash": expected_mesh_hash,
        "packed_payload_bytes": manifest["packed_payload_bytes"],
        "performance_claim": False,
        "source_file_count": len(inventory.files),
        "source_inventory_sha256": inventory.inventory_sha256,
        "topology_hash": expected_topology_hash,
        "tpu_initialized": False,
    }
    if any(success.get(field) != value for field, value in exact_success.items()):
        raise CheckpointValidationError("WS32 runtime SUCCESS identity drifted")
    for field in (
        "post_census_sha256",
        "remote_preflight_sha256",
        "remote_terminal_sha256",
    ):
        try:
            _digest(success.get(field), field=f"SUCCESS.{field}")
        except ValueError as exc:
            raise CheckpointValidationError(str(exc)) from exc
    tag = success.get("tag")
    if not isinstance(tag, str) or _SUCCESS_TAG.fullmatch(tag) is None:
        raise CheckpointValidationError("WS32 runtime SUCCESS tag drifted")
    by_slot = _verify_ws32_runtime_value(
        root,
        manifest,
        plans,
    )
    return Ws32RuntimeMetadata(
        root=root,
        manifest=manifest,
        success=success,
        plans=plans,
        records_by_slot=by_slot,
    )


def _memory_stats(device: object) -> Mapping[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _host_tensor(tensor: Any, *, dtype: str, name: str) -> tuple[Any, str]:
    import ml_dtypes
    import numpy as np
    import torch

    contiguous = tensor.contiguous()
    raw = contiguous.view(torch.uint8).numpy()
    if dtype == "U8" and contiguous.dtype == torch.uint8:
        host = np.asarray(contiguous.numpy())
        storage_dtype = "U8"
    elif dtype == "F32" and contiguous.dtype == torch.float32:
        if not bool(torch.isfinite(contiguous).all()):
            raise CheckpointValidationError(f"WS32 loader found non-finite {name!r}")
        host = np.asarray(contiguous.numpy())
        storage_dtype = "F32"
    elif dtype == "BF16" and contiguous.dtype == torch.bfloat16:
        if not bool(torch.isfinite(contiguous).all()):
            raise CheckpointValidationError(f"WS32 loader found non-finite {name!r}")
        host = contiguous.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        storage_dtype = "BF16"
    else:
        raise CheckpointValidationError(
            f"WS32 loader dtype drifted for {name!r}: {contiguous.dtype}"
        )
    if dtype == "U8" and bool(np.any((host == 0x7F) | (host == 0xFF))):
        raise CheckpointValidationError(
            f"WS32 loader found non-finite FP8 bits in {name!r}"
        )
    return host, sha256(memoryview(raw).cast("B")).hexdigest()


def load_ws32_runtime_checkpoint(
    checkpoint: VerifiedWs32RuntimeCheckpoint,
    *,
    mesh: object,
    physical_mesh: object,
) -> LoadedWs32RuntimeCheckpoint:
    """Direct-load only this host's final-owner files onto its four chips."""

    if not isinstance(checkpoint, VerifiedWs32RuntimeCheckpoint):
        raise CheckpointValidationError("WS32 full loader requires full-runtime verification")

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from safetensors import safe_open

    if checkpoint.manifest.get("mesh_hash") != physical_mesh.mesh_hash:
        raise CheckpointValidationError("WS32 loader physical mesh identity drifted")
    mesh_ids = tuple(
        tuple(int(device.id) for device in row)
        for row in np.asarray(mesh.devices, dtype=object).tolist()
    )
    if mesh_ids != physical_mesh.device_ids:
        raise CheckpointValidationError(
            "WS32 loader JAX mesh order differs from physical slots"
        )
    addressable = tuple(
        device
        for row in np.asarray(mesh.devices, dtype=object).tolist()
        for device in row
        if int(device.process_index) == jax.process_index()
    )
    expected_addressable = (
        32
        if jax.default_backend() == "cpu" and jax.process_count() == 1
        else 4
    )
    if len(addressable) != expected_addressable or set(addressable) != set(
        jax.local_devices()
    ):
        raise CheckpointValidationError(
            "WS32 loader addressable-device geometry drifted"
        )
    slot_by_device_id = {
        device_id: slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
    }
    plan_by_slot = {plan.device_slot: plan for plan in checkpoint.plans}
    before = tuple(_memory_stats(device) for device in addressable)
    arrays: dict[str, Any] = {}
    with ExitStack() as stack:
        handles = {}
        for device in addressable:
            slot = slot_by_device_id[int(device.id)]
            plan = plan_by_slot[slot]
            handles[int(device.id)] = stack.enter_context(
                safe_open(
                    checkpoint.root / plan.filename,
                    framework="pt",
                    device="cpu",
                )
            )
        for tensor_index, tensor_plan in enumerate(checkpoint.plans[0].tensors):
            sharding = NamedSharding(
                mesh, P(*tensor_plan.partition_spec)
            )
            shard_devices = tuple(
                sharding.addressable_devices_indices_map(
                    tensor_plan.global_shape
                )
            )
            if set(shard_devices) != set(addressable):
                raise CheckpointValidationError(
                    f"WS32 {tensor_plan.name!r} addressable owners drifted"
                )
            expected_local_shape = sharding.shard_shape(
                tensor_plan.global_shape
            )
            local_arrays = []
            for device in shard_devices:
                slot = slot_by_device_id[int(device.id)]
                record = checkpoint.records_by_slot[slot]
                tensor = handles[int(device.id)].get_tensor(tensor_plan.name)
                host, digest = _host_tensor(
                    tensor,
                    dtype=tensor_plan.dtype,
                    name=tensor_plan.name,
                )
                if digest != record["tensor_sha256"][tensor_index]:
                    raise CheckpointValidationError(
                        f"WS32 tensor checksum drifted while loading "
                        f"{tensor_plan.name!r} from slot {slot}"
                    )
                if tuple(host.shape) != expected_local_shape:
                    raise CheckpointValidationError(
                        f"WS32 local shape drifted for {tensor_plan.name!r}"
                    )
                local = jax.device_put(host, device)
                local.block_until_ready()
                if tuple(local.devices()) != (device,):
                    local.delete()
                    raise CheckpointValidationError(
                        f"WS32 {tensor_plan.name!r} missed its final owner"
                    )
                local_arrays.append(local)
            arrays[tensor_plan.name] = jax.make_array_from_single_device_arrays(
                tensor_plan.global_shape,
                sharding,
                tuple(local_arrays),
            )
    jax.block_until_ready(tuple(arrays.values()))
    after = tuple(_memory_stats(device) for device in addressable)
    local_records = tuple(
        {
            "device_id": int(device.id),
            "device_slot": slot_by_device_id[int(device.id)],
            "expert_coordinate": slot_by_device_id[int(device.id)] // 4,
            "feature_coordinate": slot_by_device_id[int(device.id)] % 4,
            "file_sha256": checkpoint.records_by_slot[
                slot_by_device_id[int(device.id)]
            ]["sha256"],
        }
        for device in addressable
    )
    return LoadedWs32RuntimeCheckpoint(
        arrays=arrays,
        local_device_slots=local_records,
        device_memory_before=before,
        device_memory_after=after,
    )
