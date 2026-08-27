"""Streaming final-layout safetensors writer for the complete checkpoint.

The packer consumes the content-addressed layout manifest.  It processes one
topology-local stage at a time, opens all local destination files together,
and reads each source tensor at most once for that stage.  Axis-1 shards are
split in bounded row batches, so neither a whole source tensor nor a whole
destination file is buffered in host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..partitioning.manifest import validate_layout_manifest


_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True, slots=True)
class DestinationTensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offset_start: int
    data_offset_end: int

    @property
    def byte_count(self) -> int:
        return self.data_offset_end - self.data_offset_start


@dataclass(frozen=True, slots=True)
class DestinationFilePlan:
    filename: str
    load_set: str
    stage_id: int
    device_slot: int
    device_id: int
    header: bytes
    payload_bytes: int
    tensors: tuple[DestinationTensorPlan, ...]

    @property
    def file_bytes(self) -> int:
        return len(self.header) + self.payload_bytes


@dataclass(frozen=True, slots=True)
class StreamedFileEvidence:
    filename: str
    file_bytes: int
    sha256: str


class _HashingWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = sha256()
        self.byte_count = 0

    def write(self, value: bytes | bytearray | memoryview) -> None:
        if not value:
            return
        written = self.stream.write(value)
        if written is not None and written != len(value):
            raise CheckpointValidationError(
                f"short destination write: expected {len(value)}, wrote {written}"
            )
        self.digest.update(value)
        self.byte_count += len(value)


def _header_bytes(
    *,
    filename: str,
    tensors: Sequence[DestinationTensorPlan],
    layout_manifest_sha256: str,
) -> bytes:
    value: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "greenfield_layout_manifest_sha256": layout_manifest_sha256,
            "greenfield_destination_filename": filename,
        }
    }
    for tensor in tensors:
        if tensor.name in value:
            raise CheckpointValidationError(
                f"duplicate destination tensor {tensor.name!r} in {filename!r}"
            )
        value[tensor.name] = {
            "data_offsets": [tensor.data_offset_start, tensor.data_offset_end],
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
        }
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def build_destination_file_plans(
    layout: Mapping[str, Any],
    *,
    validate_layout_contract: bool = True,
) -> tuple[DestinationFilePlan, ...]:
    """Precompute exact safetensors headers and final file sizes."""

    if validate_layout_contract:
        validate_layout_manifest(layout)
    by_file: dict[str, list[tuple[str, str, tuple[int, ...], int]]] = {}
    file_records = {
        record["filename"]: record for record in layout["destination_files"]
    }
    for placement in sorted(layout["placements"], key=lambda item: item["source"]["name"]):
        source = placement["source"]
        for destination in placement["destinations"]:
            by_file.setdefault(destination["filename"], []).append(
                (
                    source["name"],
                    source["dtype"],
                    tuple(destination["shape"]),
                    destination["byte_count"],
                )
            )
    if set(by_file) != set(file_records):
        raise CheckpointValidationError(
            "destination header file set disagrees with layout ledger"
        )
    plans = []
    for filename in sorted(by_file):
        offset = 0
        tensors = []
        for name, dtype, shape, byte_count in by_file[filename]:
            tensors.append(
                DestinationTensorPlan(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    data_offset_start=offset,
                    data_offset_end=offset + byte_count,
                )
            )
            offset += byte_count
        record = file_records[filename]
        if offset != record["planned_payload_bytes"]:
            raise CheckpointValidationError(
                f"destination {filename!r} header payload does not reconcile"
            )
        plans.append(
            DestinationFilePlan(
                filename=filename,
                load_set=record["load_set"],
                stage_id=record["stage_id"],
                device_slot=record["device_slot"],
                device_id=record["device_id"],
                header=_header_bytes(
                    filename=filename,
                    tensors=tensors,
                    layout_manifest_sha256=layout["manifest_sha256"],
                ),
                payload_bytes=offset,
                tensors=tuple(tensors),
            )
        )
    return tuple(plans)


def build_destination_probe_plans(
    layout: Mapping[str, Any],
    *,
    source_names: Sequence[str],
) -> tuple[DestinationFilePlan, ...]:
    """Build small, full-owner plans derived from one complete stage group."""

    validate_layout_manifest(layout)
    requested = tuple(source_names)
    if not requested or len(requested) != len(set(requested)):
        raise CheckpointValidationError(
            "checkpoint probe source names must be nonempty and unique"
        )
    placements = {
        placement["source"]["name"]: placement
        for placement in layout["placements"]
    }
    if len(placements) != len(layout["placements"]):
        raise CheckpointValidationError("layout contains duplicate source placements")
    unknown = set(requested) - set(placements)
    if unknown:
        raise CheckpointValidationError(
            f"checkpoint probe contains unknown source names: {sorted(unknown)!r}"
        )
    selected = [placements[name] for name in requested]
    groups = {
        (placement["load_set"], destination["stage_id"])
        for placement in selected
        for destination in placement["destinations"]
    }
    if len(groups) != 1:
        raise CheckpointValidationError(
            "checkpoint probe must stay within one load-set/stage group"
        )
    load_set, stage_id = next(iter(groups))
    file_records = {
        record["filename"]: record
        for record in layout["destination_files"]
        if record["load_set"] == load_set and record["stage_id"] == stage_id
    }
    by_file: dict[str, list[tuple[str, str, tuple[int, ...], int]]] = {}
    for placement in sorted(selected, key=lambda item: item["source"]["name"]):
        source = placement["source"]
        for destination in placement["destinations"]:
            by_file.setdefault(destination["filename"], []).append(
                (
                    source["name"],
                    source["dtype"],
                    tuple(destination["shape"]),
                    destination["byte_count"],
                )
            )
    if set(by_file) != set(file_records):
        raise CheckpointValidationError(
            "checkpoint probe must cover every owner in its stage group"
        )
    plans = []
    for filename in sorted(by_file):
        offset = 0
        tensors = []
        for name, dtype, shape, byte_count in by_file[filename]:
            tensors.append(
                DestinationTensorPlan(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    data_offset_start=offset,
                    data_offset_end=offset + byte_count,
                )
            )
            offset += byte_count
        record = file_records[filename]
        plans.append(
            DestinationFilePlan(
                filename=filename,
                load_set=load_set,
                stage_id=stage_id,
                device_slot=record["device_slot"],
                device_id=record["device_id"],
                header=_header_bytes(
                    filename=filename,
                    tensors=tensors,
                    layout_manifest_sha256=layout["manifest_sha256"],
                ),
                payload_bytes=offset,
                tensors=tuple(tensors),
            )
        )
    return tuple(plans)


def destination_groups(
    plans: Sequence[DestinationFilePlan],
) -> tuple[tuple[DestinationFilePlan, ...], ...]:
    """Return deterministic load-set/stage groups for one-pass source reads."""

    groups: dict[tuple[str, int], list[DestinationFilePlan]] = {}
    for plan in plans:
        groups.setdefault((plan.load_set, plan.stage_id), []).append(plan)
    return tuple(
        tuple(sorted(group, key=lambda item: item.device_slot))
        for _, group in sorted(groups.items())
    )


def _copy_range(
    source: BinaryIO,
    *,
    offset: int,
    byte_count: int,
    writers: Sequence[_HashingWriter],
    chunk_bytes: int,
    source_digest: Any | None = None,
) -> None:
    source.seek(offset)
    remaining = byte_count
    while remaining:
        value = source.read(min(remaining, chunk_bytes))
        if not value:
            raise CheckpointValidationError(
                f"source range truncated with {remaining} bytes remaining"
            )
        if source_digest is not None:
            source_digest.update(value)
        for writer in writers:
            writer.write(value)
        remaining -= len(value)


def _validate_axis_destinations(
    *,
    source_shape: tuple[int, ...],
    axis: int,
    destinations: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    ordered = tuple(sorted(destinations, key=lambda item: item["axis_start"]))
    previous_end = 0
    for destination in ordered:
        start = destination.get("axis_start")
        end = destination.get("axis_end_exclusive")
        if (
            destination.get("axis") != axis
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < previous_end
            or end <= start
            or end > source_shape[axis]
        ):
            raise CheckpointValidationError(
                "axis-sharded destinations overlap, exceed bounds, or drift axes"
            )
        previous_end = end
    return ordered


def _copy_axis_shards(
    source: BinaryIO,
    *,
    source_offset: int,
    source_shape: tuple[int, ...],
    dtype: str,
    destinations: Sequence[Mapping[str, Any]],
    writers: Mapping[str, _HashingWriter],
    chunk_bytes: int,
    source_digest: Any | None = None,
) -> None:
    axes = {destination.get("axis") for destination in destinations}
    if len(axes) != 1 or None in axes:
        raise CheckpointValidationError(
            "axis-sharded placement must declare one common axis"
        )
    axis = next(iter(axes))
    if not isinstance(axis, int) or not 0 <= axis < len(source_shape):
        raise CheckpointValidationError("axis-sharded placement axis is invalid")
    ordered = _validate_axis_destinations(
        source_shape=source_shape,
        axis=axis,
        destinations=destinations,
    )
    element_bytes = _DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise CheckpointValidationError(f"unsupported packed dtype {dtype!r}")
    if axis == 0:
        stride = prod(source_shape[1:]) * element_bytes
        for destination in ordered:
            start = destination["axis_start"] * stride
            _copy_range(
                source,
                offset=source_offset + start,
                byte_count=destination["byte_count"],
                writers=(writers[destination["filename"]],),
                chunk_bytes=chunk_bytes,
                source_digest=source_digest,
            )
        return

    outer_count = prod(source_shape[:axis])
    inner_bytes = prod(source_shape[axis + 1 :]) * element_bytes
    full_axis_bytes = source_shape[axis] * inner_bytes
    rows_per_batch = max(1, chunk_bytes // max(1, full_axis_bytes))
    source.seek(source_offset)
    for outer_start in range(0, outer_count, rows_per_batch):
        batch_rows = min(rows_per_batch, outer_count - outer_start)
        input_bytes = batch_rows * full_axis_bytes
        raw = source.read(input_bytes)
        if len(raw) != input_bytes:
            raise CheckpointValidationError(
                "source tensor truncated during strided axis split"
            )
        if source_digest is not None:
            source_digest.update(raw)
        for destination in ordered:
            begin = destination["axis_start"] * inner_bytes
            width = (
                destination["axis_end_exclusive"] - destination["axis_start"]
            ) * inner_bytes
            output = bytearray(batch_rows * width)
            for row in range(batch_rows):
                source_begin = row * full_axis_bytes + begin
                target_begin = row * width
                output[target_begin : target_begin + width] = raw[
                    source_begin : source_begin + width
                ]
            writers[destination["filename"]].write(output)


def stream_pack_group(
    *,
    layout: Mapping[str, Any],
    plans: Sequence[DestinationFilePlan],
    source_root: Path,
    outputs: Mapping[str, BinaryIO],
    chunk_bytes: int = 8 * 1024 * 1024,
    validate_layout_contract: bool = True,
    expected_source_sha256: Mapping[str, str] | None = None,
) -> tuple[StreamedFileEvidence, ...]:
    """Write one complete local stage group with bounded host memory."""

    if chunk_bytes <= 0:
        raise CheckpointValidationError("stream chunk size must be positive")
    if not plans:
        raise CheckpointValidationError("destination group must not be empty")
    filenames = {plan.filename for plan in plans}
    if filenames != set(outputs):
        raise CheckpointValidationError(
            "output streams must match the exact destination group"
        )
    groups = {(plan.load_set, plan.stage_id) for plan in plans}
    if len(groups) != 1:
        raise CheckpointValidationError(
            "one streaming group cannot span load sets or stages"
        )
    if validate_layout_contract:
        validate_layout_manifest(layout)
    tensor_names_by_file: dict[str, set[str]] = {}
    for plan in plans:
        names = {tensor.name for tensor in plan.tensors}
        if len(names) != len(plan.tensors):
            raise CheckpointValidationError(
                f"destination plan {plan.filename!r} contains duplicate tensors"
            )
        tensor_names_by_file[plan.filename] = names
    expected_hashes = dict(expected_source_sha256 or {})
    if expected_source_sha256 is not None:
        relevant = {
            placement["source"]["name"]
            for placement in layout["placements"]
            if any(
                destination["filename"] in filenames
                and placement["source"]["name"]
                in tensor_names_by_file[destination["filename"]]
                for destination in placement["destinations"]
            )
        }
        if set(expected_hashes) != relevant:
            raise CheckpointValidationError(
                "expected source SHA-256 keys must match the exact streamed leaf set"
            )
        for placement in layout["placements"]:
            destinations = placement["destinations"]
            selected = [
                destination
                for destination in destinations
                if destination["filename"] in filenames
                and placement["source"]["name"]
                in tensor_names_by_file[destination["filename"]]
            ]
            if selected and len(selected) != len(destinations):
                raise CheckpointValidationError(
                    "source SHA-256 validation requires every destination shard "
                    "for each streamed leaf"
                )
        for name, digest in expected_hashes.items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CheckpointValidationError(
                    f"expected source SHA-256 is invalid for {name!r}"
                )
    writers = {name: _HashingWriter(stream) for name, stream in outputs.items()}
    for plan in plans:
        writers[plan.filename].write(plan.header)

    source_files = {
        record["filename"]: record for record in layout["source"]["files"]
    }
    written_names = {filename: set() for filename in filenames}
    handles: dict[str, BinaryIO] = {}
    try:
        for placement in sorted(
            layout["placements"], key=lambda item: item["source"]["name"]
        ):
            all_destinations = placement["destinations"]
            source_record = placement["source"]
            source_name = source_record["name"]
            current_destinations = [
                destination
                for destination in all_destinations
                if destination["filename"] in filenames
            ]
            destinations = [
                destination
                for destination in current_destinations
                if source_name
                in tensor_names_by_file[destination["filename"]]
            ]
            if not destinations:
                continue
            if len(destinations) != len(current_destinations):
                raise CheckpointValidationError(
                    f"destination plans select only part of {source_name!r}"
                )
            for destination in destinations:
                written_names[destination["filename"]].add(source_name)
            source_digest = sha256() if expected_source_sha256 is not None else None
            filename = source_record["filename"]
            file_record = source_files.get(filename)
            if file_record is None:
                raise CheckpointValidationError(
                    f"source file ledger missing {filename!r}"
                )
            if filename not in handles:
                path = Path(source_root) / filename
                if path.stat().st_size != file_record["file_bytes"]:
                    raise CheckpointValidationError(
                        f"source file {filename!r} size changed after inventory"
                    )
                handles[filename] = path.open("rb")
            source = handles[filename]
            source_offset = (
                file_record["header_bytes"]
                + source_record["data_offsets"][0]
            )
            layout_kind = placement["layout"]
            if layout_kind == "axis_sharded":
                _copy_axis_shards(
                    source,
                    source_offset=source_offset,
                    source_shape=tuple(source_record["shape"]),
                    dtype=source_record["dtype"],
                    destinations=destinations,
                    writers=writers,
                    chunk_bytes=chunk_bytes,
                    source_digest=source_digest,
                )
            elif layout_kind in ("replicated", "expert_identity"):
                if any(destination.get("axis") is not None for destination in destinations):
                    raise CheckpointValidationError(
                        "unsharded placement unexpectedly declares a slice axis"
                    )
                _copy_range(
                    source,
                    offset=source_offset,
                    byte_count=source_record["byte_count"],
                    writers=tuple(
                        writers[destination["filename"]]
                        for destination in destinations
                    ),
                    chunk_bytes=chunk_bytes,
                    source_digest=source_digest,
                )
            else:
                raise CheckpointValidationError(
                    f"unsupported streaming layout {layout_kind!r}"
                )
            if (
                source_digest is not None
                and source_digest.hexdigest() != expected_hashes[source_name]
            ):
                raise CheckpointValidationError(
                    f"source tensor SHA-256 mismatch for {source_name!r}"
                )
    finally:
        for handle in handles.values():
            handle.close()
    if written_names != tensor_names_by_file:
        raise CheckpointValidationError(
            "destination plan tensors do not match streamed layout placements"
        )
    evidence = []
    for plan in plans:
        writer = writers[plan.filename]
        if writer.byte_count != plan.file_bytes:
            raise CheckpointValidationError(
                f"destination {plan.filename!r} wrote {writer.byte_count} bytes, "
                f"expected {plan.file_bytes}"
            )
        evidence.append(
            StreamedFileEvidence(
                filename=plan.filename,
                file_bytes=writer.byte_count,
                sha256=writer.digest.hexdigest(),
            )
        )
    return tuple(evidence)
