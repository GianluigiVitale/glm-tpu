"""Payload-free, fail-closed inventory of a sharded safetensors checkpoint.

Gate B begins from source metadata, not from constructing the 753B model or
reading its tensor payloads.  This module reads the index plus each
safetensors header, reconciles every name, file assignment, dtype, shape,
offset, and byte count, and emits a content-addressed append-only inventory.

Safetensors data offsets are relative to the payload immediately following
the eight-byte header length and padded JSON header.  Header parsing therefore
touches only metadata (about 15 MB for the target checkpoint), never the
755.6 GB tensor payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePath
import re
import struct
from typing import Any, Mapping, Sequence

from .errors import CheckpointValidationError


FORMAT_VERSION = 1
DEFAULT_INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_CONFIG_FILENAME = "config.json"
_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
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


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointValidationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except CheckpointValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"{label} is not valid JSON") from exc


def _require_nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointValidationError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointValidationError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _safe_source_filename(value: object) -> str:
    filename = _require_nonempty(value, field="source filename")
    pure = PurePath(filename)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != filename
        or pure.suffix != ".safetensors"
    ):
        raise CheckpointValidationError(
            f"unsafe or unsupported source filename {filename!r}"
        )
    return filename


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 0
        for dimension in value
    ):
        raise CheckpointValidationError(
            f"tensor {name!r} has invalid safetensors shape"
        )
    return tuple(value)


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


@dataclass(frozen=True, slots=True)
class SourceTensor:
    """One source leaf described entirely by safetensors metadata."""

    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    data_offset_start: int
    data_offset_end: int

    def __post_init__(self) -> None:
        _require_nonempty(self.name, field="tensor name")
        _safe_source_filename(self.filename)
        if self.dtype not in _DTYPE_BYTES:
            raise CheckpointValidationError(
                f"tensor {self.name!r} has unsupported dtype {self.dtype!r}"
            )
        object.__setattr__(self, "shape", tuple(self.shape))
        _shape(list(self.shape), name=self.name)
        start = _require_nonnegative(
            self.data_offset_start, field=f"{self.name}.data_offset_start"
        )
        end = _require_nonnegative(
            self.data_offset_end, field=f"{self.name}.data_offset_end"
        )
        if end < start:
            raise CheckpointValidationError(
                f"tensor {self.name!r} has descending data offsets"
            )
        logical_bytes = _element_count(self.shape) * _DTYPE_BYTES[self.dtype]
        if end - start != logical_bytes:
            raise CheckpointValidationError(
                f"tensor {self.name!r} byte mismatch: offsets={end - start}, "
                f"shape/dtype={logical_bytes}"
            )

    @property
    def byte_count(self) -> int:
        return self.data_offset_end - self.data_offset_start

    @property
    def layer_id(self) -> int | None:
        match = _LAYER_PATTERN.match(self.name)
        return None if match is None else int(match.group(1))

    @property
    def is_fp8_scale(self) -> bool:
        return self.name.endswith(".weight_scale_inv")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "data_offsets": [self.data_offset_start, self.data_offset_end],
            "dtype": self.dtype,
            "filename": self.filename,
            "name": self.name,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceTensor":
        offsets = value.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise CheckpointValidationError("source tensor requires two data offsets")
        tensor = cls(
            name=value.get("name"),
            filename=value.get("filename"),
            dtype=value.get("dtype"),
            shape=tuple(value.get("shape", ())),
            data_offset_start=offsets[0],
            data_offset_end=offsets[1],
        )
        if value.get("byte_count") != tensor.byte_count:
            raise CheckpointValidationError(
                f"tensor {tensor.name!r} recorded byte_count is inconsistent"
            )
        return tensor


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Header and payload accounting for one immutable source shard."""

    filename: str
    file_bytes: int
    header_bytes: int
    payload_bytes: int
    tensor_count: int
    header_sha256: str

    def __post_init__(self) -> None:
        _safe_source_filename(self.filename)
        for field in (
            "file_bytes",
            "header_bytes",
            "payload_bytes",
            "tensor_count",
        ):
            _require_nonnegative(getattr(self, field), field=field)
        _require_digest(self.header_sha256, field="header_sha256")
        if self.file_bytes != self.header_bytes + self.payload_bytes:
            raise CheckpointValidationError(
                f"source file {self.filename!r} does not reconcile header and payload"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_bytes": self.file_bytes,
            "filename": self.filename,
            "header_bytes": self.header_bytes,
            "header_sha256": self.header_sha256,
            "payload_bytes": self.payload_bytes,
            "tensor_count": self.tensor_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceFile":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """Complete, deterministic source leaf set and byte ledger."""

    model_id: str
    source_revision: str
    index_filename: str
    index_sha256: str
    config_filename: str | None
    config_sha256: str | None
    declared_payload_bytes: int
    files: tuple[SourceFile, ...]
    tensors: tuple[SourceTensor, ...]
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty(self.model_id, field="model_id")
        _require_nonempty(self.source_revision, field="source_revision")
        _require_nonempty(self.index_filename, field="index_filename")
        _require_digest(self.index_sha256, field="index_sha256")
        if (self.config_filename is None) != (self.config_sha256 is None):
            raise CheckpointValidationError(
                "config filename and digest must either both exist or both be null"
            )
        if self.config_filename is not None:
            _require_nonempty(self.config_filename, field="config_filename")
            _require_digest(self.config_sha256, field="config_sha256")
        _require_nonnegative(
            self.declared_payload_bytes, field="declared_payload_bytes"
        )
        if self.format_version != FORMAT_VERSION:
            raise CheckpointValidationError(
                f"unsupported source inventory format {self.format_version}"
            )
        object.__setattr__(
            self, "files", tuple(sorted(self.files, key=lambda item: item.filename))
        )
        object.__setattr__(
            self, "tensors", tuple(sorted(self.tensors, key=lambda item: item.name))
        )
        if not self.files or not self.tensors:
            raise CheckpointValidationError("source inventory must not be empty")
        file_names = [item.filename for item in self.files]
        tensor_names = [item.name for item in self.tensors]
        if len(file_names) != len(set(file_names)):
            raise CheckpointValidationError("source inventory has duplicate files")
        if len(tensor_names) != len(set(tensor_names)):
            raise CheckpointValidationError("source inventory has duplicate tensor names")
        known_files = set(file_names)
        if any(tensor.filename not in known_files for tensor in self.tensors):
            raise CheckpointValidationError(
                "source tensor references a file absent from the inventory"
            )
        payload = sum(item.payload_bytes for item in self.files)
        tensor_bytes = sum(item.byte_count for item in self.tensors)
        if payload != tensor_bytes or payload != self.declared_payload_bytes:
            raise CheckpointValidationError(
                "source inventory payload totals do not reconcile: "
                f"files={payload}, tensors={tensor_bytes}, "
                f"declared={self.declared_payload_bytes}"
            )
        counts: dict[str, int] = {filename: 0 for filename in known_files}
        for tensor in self.tensors:
            counts[tensor.filename] += 1
        if any(item.tensor_count != counts[item.filename] for item in self.files):
            raise CheckpointValidationError(
                "source file tensor counts do not reconcile"
            )

    @property
    def payload_bytes(self) -> int:
        return self.declared_payload_bytes

    @property
    def file_bytes(self) -> int:
        return sum(item.file_bytes for item in self.files)

    @property
    def header_bytes(self) -> int:
        return sum(item.header_bytes for item in self.files)

    @property
    def inventory_sha256(self) -> str:
        return _fingerprint(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "config_filename": self.config_filename,
            "config_sha256": self.config_sha256,
            "declared_payload_bytes": self.declared_payload_bytes,
            "files": [item.to_dict() for item in self.files],
            "format_version": self.format_version,
            "index_filename": self.index_filename,
            "index_sha256": self.index_sha256,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "tensors": [item.to_dict() for item in self.tensors],
        }
        if include_hash:
            value["inventory_sha256"] = self.inventory_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceInventory":
        inventory = cls(
            model_id=value.get("model_id"),
            source_revision=value.get("source_revision"),
            index_filename=value.get("index_filename"),
            index_sha256=value.get("index_sha256"),
            config_filename=value.get("config_filename"),
            config_sha256=value.get("config_sha256"),
            declared_payload_bytes=value.get("declared_payload_bytes"),
            files=tuple(SourceFile.from_dict(item) for item in value.get("files", ())),
            tensors=tuple(
                SourceTensor.from_dict(item) for item in value.get("tensors", ())
            ),
            format_version=value.get("format_version"),
        )
        if value.get("inventory_sha256") != inventory.inventory_sha256:
            raise CheckpointValidationError("source inventory SHA-256 mismatch")
        return inventory

    def summary_dict(self) -> dict[str, Any]:
        dtype_bytes: dict[str, int] = {}
        dtype_leaves: dict[str, int] = {}
        layer_bytes: dict[int, int] = {}
        non_layer_bytes = 0
        for tensor in self.tensors:
            dtype_bytes[tensor.dtype] = dtype_bytes.get(tensor.dtype, 0) + tensor.byte_count
            dtype_leaves[tensor.dtype] = dtype_leaves.get(tensor.dtype, 0) + 1
            if tensor.layer_id is None:
                non_layer_bytes += tensor.byte_count
            else:
                layer_bytes[tensor.layer_id] = (
                    layer_bytes.get(tensor.layer_id, 0) + tensor.byte_count
                )
        return {
            "dtype_bytes": dict(sorted(dtype_bytes.items())),
            "dtype_leaves": dict(sorted(dtype_leaves.items())),
            "file_bytes": self.file_bytes,
            "file_count": len(self.files),
            "header_bytes": self.header_bytes,
            "inventory_sha256": self.inventory_sha256,
            "layer_bytes": {
                str(layer): layer_bytes[layer] for layer in sorted(layer_bytes)
            },
            "leaf_count": len(self.tensors),
            "non_layer_bytes": non_layer_bytes,
            "payload_bytes": self.payload_bytes,
        }


def _header_entries(
    *, filename: str, raw_header: bytes, file_bytes: int, header_bytes: int
) -> tuple[SourceFile, tuple[SourceTensor, ...]]:
    decoded = _strict_json(raw_header, label=f"safetensors header {filename}")
    if not isinstance(decoded, Mapping):
        raise CheckpointValidationError(
            f"safetensors header {filename!r} must contain an object"
        )
    entries = []
    for name, metadata in decoded.items():
        if name == "__metadata__":
            if not isinstance(metadata, Mapping):
                raise CheckpointValidationError(
                    f"safetensors metadata in {filename!r} must be an object"
                )
            continue
        if not isinstance(metadata, Mapping):
            raise CheckpointValidationError(
                f"tensor metadata for {name!r} must be an object"
            )
        offsets = metadata.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise CheckpointValidationError(
                f"tensor {name!r} requires two data offsets"
            )
        entries.append(
            SourceTensor(
                name=name,
                filename=filename,
                dtype=metadata.get("dtype"),
                shape=_shape(metadata.get("shape"), name=name),
                data_offset_start=offsets[0],
                data_offset_end=offsets[1],
            )
        )
    entries.sort(
        key=lambda tensor: (
            tensor.data_offset_start,
            tensor.data_offset_end,
            tensor.name,
        )
    )
    expected_offset = 0
    for tensor in entries:
        if tensor.data_offset_start != expected_offset:
            raise CheckpointValidationError(
                f"source file {filename!r} has a gap or overlap before "
                f"tensor {tensor.name!r}: expected {expected_offset}, "
                f"found {tensor.data_offset_start}"
            )
        expected_offset = tensor.data_offset_end
    if file_bytes != header_bytes + expected_offset:
        raise CheckpointValidationError(
            f"source file {filename!r} size does not match header payload end: "
            f"file={file_bytes}, expected={header_bytes + expected_offset}"
        )
    source_file = SourceFile(
        filename=filename,
        file_bytes=file_bytes,
        header_bytes=header_bytes,
        payload_bytes=expected_offset,
        tensor_count=len(entries),
        header_sha256=_sha256_bytes(raw_header),
    )
    return source_file, tuple(entries)


def read_source_inventory(
    source_root: Path,
    *,
    model_id: str,
    source_revision: str,
    index_filename: str = DEFAULT_INDEX_FILENAME,
    config_filename: str | None = DEFAULT_CONFIG_FILENAME,
) -> SourceInventory:
    """Read and reconcile checkpoint metadata without touching tensor payloads."""

    root = Path(source_root)
    index_path = root / index_filename
    try:
        raw_index = index_path.read_bytes()
    except OSError as exc:
        raise CheckpointValidationError(
            f"cannot read source index {index_path}"
        ) from exc
    decoded = _strict_json(raw_index, label="safetensors index")
    if not isinstance(decoded, Mapping):
        raise CheckpointValidationError("safetensors index must contain an object")
    metadata = decoded.get("metadata")
    weight_map = decoded.get("weight_map")
    if not isinstance(metadata, Mapping) or not isinstance(weight_map, Mapping):
        raise CheckpointValidationError(
            "safetensors index requires metadata and weight_map objects"
        )
    declared = _require_nonnegative(
        metadata.get("total_size"), field="index metadata.total_size"
    )
    if not weight_map:
        raise CheckpointValidationError("safetensors weight_map must not be empty")
    normalized_map: dict[str, str] = {}
    for name, filename in weight_map.items():
        normalized_name = _require_nonempty(name, field="weight_map tensor name")
        normalized_map[normalized_name] = _safe_source_filename(filename)

    files = []
    tensors = []
    for filename in sorted(set(normalized_map.values())):
        path = root / filename
        try:
            file_bytes = path.stat().st_size
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise CheckpointValidationError(
                        f"source file {filename!r} lacks a safetensors header length"
                    )
                header_length = struct.unpack("<Q", prefix)[0]
                if header_length > file_bytes - 8:
                    raise CheckpointValidationError(
                        f"source file {filename!r} declares an oversized header"
                    )
                raw_header = stream.read(header_length)
                if len(raw_header) != header_length:
                    raise CheckpointValidationError(
                        f"source file {filename!r} has a truncated header"
                    )
        except CheckpointValidationError:
            raise
        except OSError as exc:
            raise CheckpointValidationError(
                f"cannot inspect source file {path}"
            ) from exc
        source_file, source_tensors = _header_entries(
            filename=filename,
            raw_header=raw_header,
            file_bytes=file_bytes,
            header_bytes=8 + header_length,
        )
        expected_names = {
            name for name, mapped_filename in normalized_map.items()
            if mapped_filename == filename
        }
        actual_names = {tensor.name for tensor in source_tensors}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:5]
            unexpected = sorted(actual_names - expected_names)[:5]
            raise CheckpointValidationError(
                f"source file {filename!r} disagrees with weight_map: "
                f"missing={missing}, unexpected={unexpected}"
            )
        files.append(source_file)
        tensors.extend(source_tensors)

    actual_names = {tensor.name for tensor in tensors}
    if actual_names != set(normalized_map):
        raise CheckpointValidationError(
            "source header leaf set does not equal index weight_map"
        )
    config_sha256 = None
    if config_filename is not None:
        config_path = root / config_filename
        try:
            raw_config = config_path.read_bytes()
        except OSError as exc:
            raise CheckpointValidationError(
                f"cannot read source config {config_path}"
            ) from exc
        config = _strict_json(raw_config, label="model config")
        if not isinstance(config, Mapping):
            raise CheckpointValidationError("model config must contain an object")
        config_sha256 = _sha256_bytes(raw_config)

    return SourceInventory(
        model_id=model_id,
        source_revision=source_revision,
        index_filename=index_filename,
        index_sha256=_sha256_bytes(raw_index),
        config_filename=config_filename,
        config_sha256=config_sha256,
        declared_payload_bytes=declared,
        files=tuple(files),
        tensors=tuple(tensors),
    )


def write_source_inventory(inventory: SourceInventory, output: Path) -> None:
    """Write a manifest append-only and commit it by atomic rename."""

    path = Path(output)
    if path.exists():
        raise CheckpointValidationError(
            f"refusing to overwrite source inventory {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CheckpointValidationError(
            f"stale source inventory temporary exists: {temporary}"
        )
    try:
        temporary.write_text(
            json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_source_inventory(path: Path) -> SourceInventory:
    """Parse and re-hash an inventory without consulting its source mount."""

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CheckpointValidationError(
            f"cannot read source inventory {path}"
        ) from exc
    decoded = _strict_json(raw, label="source inventory")
    if not isinstance(decoded, Mapping):
        raise CheckpointValidationError("source inventory must contain an object")
    return SourceInventory.from_dict(decoded)


def authenticated_inventory(path: Path, expected_sha256: str) -> SourceInventory:
    """Validate the inventory and its pinned canonical digest, not JSON file bytes."""
    inventory = inspect_source_inventory(path)
    if inventory.inventory_sha256 != expected_sha256:
        raise ValueError("layer source inventory canonical hash drifted")
    return inventory
