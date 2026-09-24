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
import base64
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

from glm_tpu.config.site import approved_source_uri

from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader.source_inventory import SourceInventory
from glm_tpu.config.model import ModelGeometry
from glm_tpu.model_loader.placement import (
    RuntimePlacementReport,
    SourcePlacement,
    build_runtime_placement_report,
    placements_for_source_tensor,
)


RUNTIME_FORMAT_VERSION = 1
RUNTIME_ARTIFACT_KIND = "greenfield_ws32_runtime_checkpoint"
RUNTIME_SLOT_RECORD_KIND = "greenfield_ws32_runtime_slot_records"
RUNTIME_PLAN_ID = "WS32_2D"
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


def _sha256_file(path: Path, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeTensorPlan:
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
class RuntimeFilePlan:
    device_slot: int
    expert_coordinate: int
    feature_coordinate: int
    filename: str
    header: bytes
    tensors: tuple[RuntimeTensorPlan, ...]
    payload_bytes: int

    @property
    def file_bytes(self) -> int:
        return len(self.header) + self.payload_bytes


@dataclass(frozen=True, slots=True)
class RuntimePackConfig:
    source_root: Path
    source_uri: str
    output_dir: Path
    code_hash: str
    mesh_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        # The approved buckets are the site's storage.allowed_source_uri_prefixes.
        if not approved_source_uri(self.source_uri):
            raise ValueError("WS32 source URI must use the approved bucket")
        _digest(self.code_hash, field="code_hash", lengths=(40, 64))
        _digest(self.mesh_hash, field="mesh_hash")


def _header_bytes(
    *,
    slot: int,
    tensors: Sequence[RuntimeTensorPlan],
    source_inventory_sha256: str,
    geometry_sha256: str,
    placement_sha256: str,
    mesh_hash: str,
) -> bytes:
    value: dict[str, Any] = {
        "__metadata__": {
            "artifact_kind": RUNTIME_ARTIFACT_KIND,
            "device_slot": str(slot),
            "expert_coordinate": str(slot // 4),
            "feature_coordinate": str(slot % 4),
            "format_version": str(RUNTIME_FORMAT_VERSION),
            "geometry_sha256": geometry_sha256,
            "mesh_hash": mesh_hash,
            "placement_sha256": placement_sha256,
            "plan_id": RUNTIME_PLAN_ID,
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


def build_runtime_file_plans(
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    mesh_hash: str,
) -> tuple[RuntimePlacementReport, tuple[RuntimeFilePlan, ...]]:
    """Build 32 deterministic final-owner safetensors layouts."""

    _digest(mesh_hash, field="mesh_hash")
    report = build_runtime_placement_report(inventory, geometry)
    schemas: dict[tuple[int, str], SourcePlacement] = {}
    for source in inventory.tensors:
        if source.layer_id is not None and source.layer_id >= geometry.num_layers:
            continue
        for placement in placements_for_source_tensor(source, geometry):
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
                RuntimeTensorPlan(
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
            RuntimeFilePlan(
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
    plan: RuntimeFilePlan,
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
