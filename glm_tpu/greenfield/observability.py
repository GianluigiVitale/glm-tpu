"""Fail-closed observability over immutable numerical artifacts.

The protected TPU programs deliberately do not print or call back into the
host.  This module turns their sealed NPZ outputs into an ordered numerical
state-machine view without importing JAX or executing a model.  Every source
file and every selected array is SHA-bound before comparison.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import zipfile

from .errors import BenchmarkValidationError


__all__ = (
    "ArrayObservation",
    "OBSERVABILITY_SCHEMA_VERSION",
    "audit_observability_contract",
    "compare_array_observations",
    "inspect_npz_artifact",
    "write_observability_report",
)


OBSERVABILITY_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_PIN = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ROLES = frozenset(("accepted", "candidate", "supporting", "rejected"))
_TRUST = frozenset(
    (
        "accepted_protected",
        "candidate_protected",
        "supporting_protected",
        "rejected_perturbed",
    )
)
_OBSERVATION_METHODS = {
    "accepted": frozenset(("protected_oracle",)),
    "candidate": frozenset(("device_output",)),
    "supporting": frozenset(("device_output", "offline_derived")),
    "rejected": frozenset(("rejected_callback", "rejected_device_output")),
}
_DTYPES: dict[str, tuple[str, int, str]] = {
    "|b1": ("?", 1, "bool"),
    "<u2": ("H", 2, "integer"),
    "<u4": ("I", 4, "integer"),
    "<i4": ("i", 4, "integer"),
    "<f4": ("f", 4, "float"),
}
_SEMANTIC_DTYPES = {
    "bf16": "<u2",
    "bool": "|b1",
    "float32": "<f4",
    "int32": "<i4",
    "uint16": "<u2",
    "uint32": "<u4",
}
_MAX_NPZ_MEMBERS = 4096
_MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
_MAX_CONTRACT_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BenchmarkValidationError(f"non-finite JSON value: {value}")


@contextmanager
def _open_regular_file(path: Path, label: str) -> Iterator[BinaryIO]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkValidationError(f"cannot open {label}: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkValidationError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _snapshot_regular_file(
    path: Path,
    label: str,
    expected_sha256: str,
) -> Iterator[tuple[BinaryIO, str]]:
    """Copy and hash one bounded regular file, then parse only the snapshot."""

    digest = sha256()
    total = 0
    with _open_regular_file(path, label) as source, tempfile.TemporaryFile() as copy:
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > _MAX_ARTIFACT_BYTES:
                raise BenchmarkValidationError(f"{label} is too large: {path}")
            digest.update(block)
            copy.write(block)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise BenchmarkValidationError(f"{label} SHA-256 drifted: {path}")
        copy.seek(0)
        yield copy, observed_sha256


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_code_pin(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CODE_PIN.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a full lowercase Git SHA")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise BenchmarkValidationError(
            f"{label} keys drifted: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_npy_header(stream: BinaryIO, label: str) -> tuple[str, tuple[int, ...]]:
    if stream.read(6) != b"\x93NUMPY":
        raise BenchmarkValidationError(f"{label} has invalid NPY magic")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise BenchmarkValidationError(f"{label} has unsupported NPY version")
    length_size = 2 if version[0] == 1 else 4
    length_raw = stream.read(length_size)
    if len(length_raw) != length_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY header")
    header_size = struct.unpack("<H" if length_size == 2 else "<I", length_raw)[0]
    header_raw = stream.read(header_size)
    if len(header_raw) != header_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY metadata")
    try:
        header = ast.literal_eval(header_raw.decode("latin1").strip())
    except (SyntaxError, ValueError) as error:
        raise BenchmarkValidationError(f"{label} has invalid NPY metadata") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise BenchmarkValidationError(f"{label} NPY metadata schema drifted")
    if header["fortran_order"] is not False:
        raise BenchmarkValidationError(f"{label} must be C-contiguous")
    dtype = header["descr"]
    shape = header["shape"]
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise BenchmarkValidationError(f"{label} dtype is unsupported: {dtype!r}")
    if not isinstance(shape, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape
    ):
        raise BenchmarkValidationError(f"{label} shape is invalid")
    return dtype, shape


@dataclass(frozen=True, slots=True)
class ArrayObservation:
    """One authenticated, optionally prefix-selected NPZ array."""

    source_id: str
    watchpoint_id: str
    semantic_dtype: str
    dtype: str
    shape: tuple[int, ...]
    raw: bytes
    raw_sha256: str


def _read_npz_observation(
    archive: zipfile.ZipFile,
    *,
    source_id: str,
    watchpoint_id: str,
    semantic_dtype: str,
    key: str,
    index_prefix: tuple[int, ...],
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    expected_sha256: str,
) -> ArrayObservation:
    member = f"{key}.npy"
    try:
        info = archive.getinfo(member)
    except KeyError as error:
        raise BenchmarkValidationError(
            f"NPZ key is absent for {watchpoint_id}: {key}"
        ) from error
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise BenchmarkValidationError(
            f"cannot inspect NPZ member: {member}"
        ) from error
    if info.flag_bits & 0x1:
        raise BenchmarkValidationError(f"NPZ member is encrypted: {member}")
    if info.file_size > _MAX_OBSERVATION_BYTES + 64 * 1024:
        raise BenchmarkValidationError(f"NPZ member is too large: {member}")
    try:
        with archive.open(info) as stream:
            dtype, source_shape = _read_npy_header(stream, member)
            item_size = _DTYPES[dtype][1]
            expected_bytes = _product(source_shape) * item_size
            if expected_bytes > _MAX_OBSERVATION_BYTES:
                raise BenchmarkValidationError(f"NPZ array is too large: {member}")
            raw = stream.read(expected_bytes + 1)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise BenchmarkValidationError(f"cannot read NPZ member: {member}") from error
    if len(raw) != expected_bytes:
        raise BenchmarkValidationError(
            f"NPZ member byte count drifted for {member}: "
            f"{len(raw)} != {expected_bytes}"
        )
    if len(index_prefix) > len(source_shape):
        raise BenchmarkValidationError(
            f"index prefix rank exceeds source rank for {watchpoint_id}"
        )
    flat_offset = 0
    for axis, index in enumerate(index_prefix):
        if index < 0 or index >= source_shape[axis]:
            raise BenchmarkValidationError(
                f"index prefix is out of bounds for {watchpoint_id}"
            )
        flat_offset += index * _product(source_shape[axis + 1 :])
    selected_shape = source_shape[len(index_prefix) :]
    selected_count = _product(selected_shape)
    start = flat_offset * item_size
    selected = raw[start : start + selected_count * item_size]
    observed_sha256 = sha256(selected).hexdigest()
    if dtype != expected_dtype:
        raise BenchmarkValidationError(
            f"storage dtype drifted for {watchpoint_id}: {dtype} != {expected_dtype}"
        )
    if selected_shape != expected_shape:
        raise BenchmarkValidationError(
            f"shape drifted for {watchpoint_id}: "
            f"{selected_shape} != {expected_shape}"
        )
    if observed_sha256 != expected_sha256:
        raise BenchmarkValidationError(
            f"array SHA-256 drifted for {watchpoint_id}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    return ArrayObservation(
        source_id=source_id,
        watchpoint_id=watchpoint_id,
        semantic_dtype=semantic_dtype,
        dtype=dtype,
        shape=selected_shape,
        raw=selected,
        raw_sha256=observed_sha256,
    )


def _npz_names(archive: zipfile.ZipFile) -> list[str]:
    try:
        return archive.namelist()
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise BenchmarkValidationError("cannot inspect NPZ directory") from error


def _unravel(index: int, shape: tuple[int, ...]) -> list[int]:
    if not shape:
        return []
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = index % shape[axis]
        index //= shape[axis]
    return result


def _numeric_value(observation: ArrayObservation, raw: bytes) -> Any:
    fmt, item_size, _ = _DTYPES[observation.dtype]
    value = struct.unpack("<" + fmt, raw[:item_size])[0]
    if observation.semantic_dtype == "bf16":
        value = struct.unpack("<f", struct.pack("<I", int(value) << 16))[0]
    return value


def _value_record(observation: ArrayObservation, raw: bytes) -> dict[str, Any]:
    item_size = _DTYPES[observation.dtype][1]
    value = _numeric_value(observation, raw)
    kind = _DTYPES[observation.dtype][2]
    if observation.semantic_dtype == "bf16":
        kind = "float"
    if kind == "float" and not math.isfinite(value):
        rendered: Any = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    else:
        rendered = value
    return {"raw_hex": raw[:item_size].hex(), "value": rendered}


def compare_array_observations(
    accepted: ArrayObservation, candidate: ArrayObservation
) -> dict[str, Any]:
    """Return an exact bit-level comparison for two authenticated arrays."""

    for label, observation in (("accepted", accepted), ("candidate", candidate)):
        if not isinstance(observation.dtype, str) or observation.dtype not in _DTYPES:
            raise BenchmarkValidationError(f"{label} observation dtype is invalid")
        if (
            not isinstance(observation.semantic_dtype, str)
            or observation.semantic_dtype not in _SEMANTIC_DTYPES
            or _SEMANTIC_DTYPES[observation.semantic_dtype] != observation.dtype
        ):
            raise BenchmarkValidationError(
                f"{label} observation semantic/storage dtype disagrees"
            )
        if not isinstance(observation.shape, tuple) or not observation.shape or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in observation.shape
        ):
            raise BenchmarkValidationError(f"{label} observation shape is invalid")
        expected_bytes = _product(observation.shape) * _DTYPES[observation.dtype][1]
        if not isinstance(observation.raw, bytes):
            raise BenchmarkValidationError(
                f"{label} observation raw value must be bytes"
            )
        if len(observation.raw) != expected_bytes:
            raise BenchmarkValidationError(
                f"{label} observation raw byte count is invalid"
            )
        if (
            not isinstance(observation.raw_sha256, str)
            or _SHA256.fullmatch(observation.raw_sha256) is None
            or sha256(observation.raw).hexdigest() != observation.raw_sha256
        ):
            raise BenchmarkValidationError(
                f"{label} observation raw SHA-256 is invalid"
            )
    if accepted.watchpoint_id != candidate.watchpoint_id:
        raise BenchmarkValidationError("watchpoint identities do not match")
    if (
        accepted.dtype != candidate.dtype
        or accepted.semantic_dtype != candidate.semantic_dtype
        or accepted.shape != candidate.shape
    ):
        raise BenchmarkValidationError("watchpoint dtype/shape does not match")
    item_size = _DTYPES[accepted.dtype][1]
    count = len(accepted.raw) // item_size
    first = None
    mismatches = 0
    numeric = accepted.semantic_dtype in ("bf16", "float32")
    max_abs_error: float | None = None
    nonfinite_mismatch_count = 0
    for index in range(count):
        start = index * item_size
        left = accepted.raw[start : start + item_size]
        right = candidate.raw[start : start + item_size]
        if left == right:
            continue
        mismatches += 1
        if first is None:
            first = {
                "accepted": _value_record(accepted, left),
                "candidate": _value_record(candidate, right),
                "index": _unravel(index, accepted.shape),
            }
        if numeric:
            left_value = _numeric_value(accepted, left)
            right_value = _numeric_value(candidate, right)
            if math.isfinite(left_value) and math.isfinite(right_value):
                difference = abs(left_value - right_value)
                max_abs_error = (
                    difference
                    if max_abs_error is None
                    else max(max_abs_error, difference)
                )
            else:
                nonfinite_mismatch_count += 1
    if numeric and mismatches == 0:
        max_abs_error = 0.0
    elif nonfinite_mismatch_count:
        max_abs_error = None
    return {
        "accepted_array_sha256": accepted.raw_sha256,
        "candidate_array_sha256": candidate.raw_sha256,
        "dtype": accepted.dtype,
        "exact": mismatches == 0,
        "first_mismatch": first,
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatches,
        "nonfinite_mismatch_count": nonfinite_mismatch_count,
        "semantic_dtype": accepted.semantic_dtype,
        "shape": list(accepted.shape),
        "value_count": count,
        "watchpoint_id": accepted.watchpoint_id,
    }


def _load_json(
    path: Path,
    expected_sha256: str,
    *,
    label: str = "observability contract",
) -> tuple[dict[str, Any], str]:
    expected_sha256 = _require_sha256(expected_sha256, f"{label} SHA-256")
    try:
        with _snapshot_regular_file(
            path, label, expected_sha256
        ) as (stream, observed_sha256):
            raw = stream.read(_MAX_CONTRACT_BYTES + 1)
        if len(raw) > _MAX_CONTRACT_BYTES:
            raise BenchmarkValidationError(f"{label} is too large")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(f"cannot parse {label}: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    return value, observed_sha256


def _read_json_path(value: Any, path: Any, label: str) -> Any:
    if not isinstance(path, list) or not path or any(
        not isinstance(item, str) or not item for item in path
    ):
        raise BenchmarkValidationError(f"{label} JSON path is invalid")
    current = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise BenchmarkValidationError(f"{label} JSON path is absent")
        current = current[component]
    return current


def _verify_source_identity_evidence(
    source: Mapping[str, Any], source_index: int
) -> list[dict[str, Any]]:
    evidence = source["identity_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise BenchmarkValidationError("source identity_evidence must be non-empty")
    allowed_fields = {
        "artifact_sha256",
        "code_pin",
        "coherence_id",
        "executable_identity_sha256",
        "plan_sha256",
    }
    bound_fields: set[str] = set()
    records: list[dict[str, Any]] = []
    for evidence_index, item in enumerate(evidence):
        label = f"source[{source_index}].identity_evidence[{evidence_index}]"
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"{label} must be an object")
        _require_exact_keys(
            item,
            {"artifact_path", "artifact_sha256", "bindings"},
            label,
        )
        if not isinstance(item["artifact_path"], str):
            raise BenchmarkValidationError(f"{label} artifact_path is invalid")
        evidence_path = Path(item["artifact_path"])
        if not evidence_path.is_absolute():
            raise BenchmarkValidationError(f"{label} artifact_path must be absolute")
        evidence_sha256 = _require_sha256(
            item["artifact_sha256"], f"{label} artifact_sha256"
        )
        document, observed_sha256 = _load_json(
            evidence_path,
            evidence_sha256,
            label="source identity evidence",
        )
        bindings = item["bindings"]
        if not isinstance(bindings, list) or not bindings:
            raise BenchmarkValidationError(f"{label} bindings must be non-empty")
        binding_records: list[dict[str, Any]] = []
        for binding_index, binding in enumerate(bindings):
            binding_label = f"{label}.bindings[{binding_index}]"
            if not isinstance(binding, dict):
                raise BenchmarkValidationError(f"{binding_label} must be an object")
            _require_exact_keys(
                binding, {"json_path", "source_field"}, binding_label
            )
            source_field = binding["source_field"]
            if source_field not in allowed_fields or source_field in bound_fields:
                raise BenchmarkValidationError(
                    f"{binding_label} source_field is invalid or duplicated"
                )
            observed = _read_json_path(
                document, binding["json_path"], binding_label
            )
            if observed != source[source_field]:
                raise BenchmarkValidationError(
                    f"{binding_label} disagrees with source {source_field}"
                )
            bound_fields.add(source_field)
            binding_records.append(
                {
                    "json_path": binding["json_path"],
                    "source_field": source_field,
                }
            )
        records.append(
            {
                "artifact_path": str(evidence_path),
                "artifact_sha256": observed_sha256,
                "bindings": binding_records,
            }
        )
    required_fields = {"artifact_sha256", "code_pin", "coherence_id"}
    if source["executable_identity_sha256"] is not None:
        required_fields.add("executable_identity_sha256")
    if not required_fields.issubset(bound_fields):
        raise BenchmarkValidationError(
            "source identity evidence does not bind every required field"
        )
    return records


def _read_source_bindings(
    *,
    artifact_path: Path,
    expected_artifact_sha256: str,
    source_id: str,
    source_index: int,
    bindings: Any,
    watchpoints: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[ArrayObservation]]:
    if not artifact_path.is_absolute():
        raise BenchmarkValidationError("source artifact path must be absolute")
    if not isinstance(bindings, list) or not bindings:
        raise BenchmarkValidationError("source observations must be a non-empty list")
    with _snapshot_regular_file(
        artifact_path, "source artifact", expected_artifact_sha256
    ) as (stream, observed_artifact_sha256):
        try:
            archive = zipfile.ZipFile(stream)
        except (OSError, zipfile.BadZipFile) as error:
            raise BenchmarkValidationError(
                f"cannot open NPZ artifact: {artifact_path}"
            ) from error
        with archive:
            names = _npz_names(archive)
            if len(names) > _MAX_NPZ_MEMBERS:
                raise BenchmarkValidationError("NPZ contains too many members")
            if len(names) != len(set(names)):
                raise BenchmarkValidationError("NPZ contains duplicate members")
            bound_ids: set[str] = set()
            result: list[ArrayObservation] = []
            for binding_index, binding in enumerate(bindings):
                if not isinstance(binding, dict):
                    raise BenchmarkValidationError(
                        "observation binding must be an object"
                    )
                _require_exact_keys(
                    binding,
                    {
                        "array_sha256",
                        "index_prefix",
                        "key",
                        "shape",
                        "storage_dtype",
                        "watchpoint_id",
                    },
                    f"source[{source_index}].observations[{binding_index}]",
                )
                watchpoint_id = _require_identifier(
                    binding["watchpoint_id"], "binding watchpoint_id"
                )
                if watchpoint_id not in watchpoints or watchpoint_id in bound_ids:
                    raise BenchmarkValidationError(
                        "binding watchpoint is absent or duplicated"
                    )
                bound_ids.add(watchpoint_id)
                if not isinstance(binding["key"], str) or not binding["key"]:
                    raise BenchmarkValidationError("binding key is invalid")
                index_prefix = binding["index_prefix"]
                if not isinstance(index_prefix, list) or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in index_prefix
                ):
                    raise BenchmarkValidationError("binding index_prefix is invalid")
                shape = binding["shape"]
                if not isinstance(shape, list) or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in shape
                ):
                    raise BenchmarkValidationError("binding shape is invalid")
                watchpoint = watchpoints[watchpoint_id]
                if shape != watchpoint["shape"]:
                    raise BenchmarkValidationError(
                        "binding/watchpoint shape disagrees"
                    )
                semantic_dtype = watchpoint["semantic_dtype"]
                storage_dtype = binding["storage_dtype"]
                if _SEMANTIC_DTYPES[semantic_dtype] != storage_dtype:
                    raise BenchmarkValidationError(
                        "binding semantic/storage dtype disagrees"
                    )
                result.append(
                    _read_npz_observation(
                        archive,
                        source_id=source_id,
                        watchpoint_id=watchpoint_id,
                        semantic_dtype=semantic_dtype,
                        key=binding["key"],
                        index_prefix=tuple(index_prefix),
                        expected_dtype=storage_dtype,
                        expected_shape=tuple(shape),
                        expected_sha256=_require_sha256(
                            binding["array_sha256"], "binding array_sha256"
                        ),
                    )
                )
    return observed_artifact_sha256, result


def inspect_npz_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Return a bounded typed inventory of one authenticated numerical NPZ."""

    expected_sha256 = _require_sha256(expected_sha256, "artifact SHA-256")
    with _snapshot_regular_file(
        path, "NPZ artifact", expected_sha256
    ) as (stream, artifact_sha256):
        try:
            archive = zipfile.ZipFile(stream)
        except (OSError, zipfile.BadZipFile) as error:
            raise BenchmarkValidationError(
                f"cannot open NPZ artifact: {path}"
            ) from error
        with archive:
            names = _npz_names(archive)
            if not names or len(names) > _MAX_NPZ_MEMBERS:
                raise BenchmarkValidationError("NPZ member count is invalid")
            if len(names) != len(set(names)):
                raise BenchmarkValidationError("NPZ contains duplicate members")
            members: list[dict[str, Any]] = []
            for name in sorted(names):
                if "/" in name or not name.endswith(".npy"):
                    raise BenchmarkValidationError(
                        f"NPZ contains a noncanonical member: {name}"
                    )
                try:
                    info = archive.getinfo(name)
                except (OSError, zipfile.BadZipFile, RuntimeError) as error:
                    raise BenchmarkValidationError(
                        f"cannot inspect NPZ member: {name}"
                    ) from error
                if info.flag_bits & 0x1:
                    raise BenchmarkValidationError(f"NPZ member is encrypted: {name}")
                if info.file_size > _MAX_OBSERVATION_BYTES + 64 * 1024:
                    raise BenchmarkValidationError(f"NPZ member is too large: {name}")
                try:
                    with archive.open(info) as member_stream:
                        dtype, shape = _read_npy_header(member_stream, name)
                        expected_bytes = _product(shape) * _DTYPES[dtype][1]
                        if expected_bytes > _MAX_OBSERVATION_BYTES:
                            raise BenchmarkValidationError(
                                f"NPZ array is too large: {name}"
                            )
                        raw = member_stream.read(expected_bytes + 1)
                except BenchmarkValidationError:
                    raise
                except (OSError, zipfile.BadZipFile, RuntimeError) as error:
                    raise BenchmarkValidationError(
                        f"cannot read NPZ member: {name}"
                    ) from error
                if len(raw) != expected_bytes:
                    raise BenchmarkValidationError(
                        f"NPZ member byte count drifted: {name}"
                    )
                members.append(
                    {
                        "key": name[:-4],
                        "raw_bytes": len(raw),
                        "raw_sha256": sha256(raw).hexdigest(),
                        "shape": list(shape),
                        "storage_dtype": dtype,
                    }
                )
    return {
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "member_count": len(members),
        "members": members,
    }


def audit_observability_contract(
    path: Path, expected_contract_sha256: str
) -> dict[str, Any]:
    """Authenticate a contract and report coverage plus first divergence."""

    contract, contract_sha256 = _load_json(path, expected_contract_sha256)
    _require_exact_keys(
        contract,
        {"contract_id", "plans", "schema_version", "sources", "watchpoints"},
        "contract",
    )
    if contract["schema_version"] != OBSERVABILITY_SCHEMA_VERSION:
        raise BenchmarkValidationError("observability schema version drifted")
    contract_id = _require_identifier(contract["contract_id"], "contract_id")
    watchpoints_raw = contract["watchpoints"]
    sources_raw = contract["sources"]
    plans_raw = contract["plans"]
    if not isinstance(watchpoints_raw, list) or not watchpoints_raw:
        raise BenchmarkValidationError("watchpoints must be a non-empty list")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise BenchmarkValidationError("sources must be a non-empty list")
    if not isinstance(plans_raw, list) or not plans_raw:
        raise BenchmarkValidationError("plans must be a non-empty list")

    plans: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(plans_raw):
        if not isinstance(item, dict):
            raise BenchmarkValidationError("plan must be an object")
        _require_exact_keys(
            item,
            {
                "authority_file_sha256",
                "authority_json_path",
                "authority_kind",
                "authority_path",
                "id",
                "name",
                "plan_sha256",
            },
            f"plan[{index}]",
        )
        identifier = _require_identifier(item["id"], f"plan[{index}].id")
        if identifier in plans:
            raise BenchmarkValidationError("plan ids must be unique")
        if not isinstance(item["name"], str) or not item["name"]:
            raise BenchmarkValidationError("plan name is invalid")
        authority_kind = _require_identifier(
            item["authority_kind"], f"plan[{index}].authority_kind"
        )
        plan_sha256 = _require_sha256(
            item["plan_sha256"], f"plan[{index}].plan_sha256"
        )
        authority_path = item["authority_path"]
        authority_file_sha256 = item["authority_file_sha256"]
        authority_json_path = item["authority_json_path"]
        if authority_kind == "canonical_name_sha256":
            if (
                authority_path is not None
                or authority_file_sha256 is not None
                or authority_json_path != []
                or plan_sha256 != sha256(item["name"].encode("utf-8")).hexdigest()
            ):
                raise BenchmarkValidationError(
                    "canonical plan authority does not match its name"
                )
        elif authority_kind == "sealed_json_field":
            if not isinstance(authority_path, str):
                raise BenchmarkValidationError("sealed plan authority path is invalid")
            authority = Path(authority_path)
            if not authority.is_absolute():
                raise BenchmarkValidationError(
                    "sealed plan authority path must be absolute"
                )
            authority_file_sha256 = _require_sha256(
                authority_file_sha256, "sealed plan authority file SHA-256"
            )
            document, _ = _load_json(
                authority,
                authority_file_sha256,
                label="sealed plan authority",
            )
            if (
                _read_json_path(document, authority_json_path, "plan authority")
                != plan_sha256
            ):
                raise BenchmarkValidationError("sealed plan authority value drifted")
        else:
            raise BenchmarkValidationError("plan authority kind is unsupported")
        plans[identifier] = {
            "authority_file_sha256": authority_file_sha256,
            "authority_json_path": authority_json_path,
            "authority_kind": authority_kind,
            "authority_path": authority_path,
            "id": identifier,
            "name": item["name"],
            "plan_sha256": plan_sha256,
        }

    watchpoints: dict[str, dict[str, Any]] = {}
    orders: set[int] = set()
    for index, item in enumerate(watchpoints_raw):
        if not isinstance(item, dict):
            raise BenchmarkValidationError("watchpoint must be an object")
        _require_exact_keys(
            item,
            {"id", "layer", "order", "position", "semantic_dtype", "shape"},
            f"watchpoint[{index}]",
        )
        identifier = _require_identifier(item["id"], f"watchpoint[{index}].id")
        if identifier in watchpoints:
            raise BenchmarkValidationError("watchpoint ids must be unique")
        order = item["order"]
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise BenchmarkValidationError("watchpoint order must be non-negative")
        if order in orders:
            raise BenchmarkValidationError("watchpoint order must be unique")
        for label in ("layer", "position"):
            value = item[label]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BenchmarkValidationError(
                    f"watchpoint {label} must be non-negative"
                )
        shape = item["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        ):
            raise BenchmarkValidationError("watchpoint shape is invalid")
        if item["semantic_dtype"] not in _SEMANTIC_DTYPES:
            raise BenchmarkValidationError("watchpoint semantic dtype is invalid")
        watchpoints[identifier] = item
        orders.add(order)
    if orders != set(range(len(watchpoints))):
        raise BenchmarkValidationError("watchpoint order must be contiguous")

    observations: dict[str, dict[str, list[ArrayObservation]]] = {
        identifier: {role: [] for role in _ROLES} for identifier in watchpoints
    }
    source_records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for source_index, source in enumerate(sources_raw):
        if not isinstance(source, dict):
            raise BenchmarkValidationError("source must be an object")
        _require_exact_keys(
            source,
            {
                "artifact_path",
                "artifact_sha256",
                "code_pin",
                "coherence_id",
                "executable_identity_kind",
                "executable_identity_sha256",
                "id",
                "identity_evidence",
                "observation_method",
                "observations",
                "plan_id",
                "plan_sha256",
                "role",
                "trust",
            },
            f"source[{source_index}]",
        )
        source_id = _require_identifier(source["id"], f"source[{source_index}].id")
        if source_id in source_ids:
            raise BenchmarkValidationError("source ids must be unique")
        source_ids.add(source_id)
        role = source["role"]
        trust = source["trust"]
        if role not in _ROLES or trust not in _TRUST:
            raise BenchmarkValidationError("source role/trust is invalid")
        expected_trust_prefix = {
            "accepted": "accepted_",
            "candidate": "candidate_",
            "supporting": "supporting_",
            "rejected": "rejected_",
        }[role]
        if not trust.startswith(expected_trust_prefix):
            raise BenchmarkValidationError("source role and trust disagree")
        method = source["observation_method"]
        if method not in _OBSERVATION_METHODS[role]:
            raise BenchmarkValidationError(
                "source role and observation method disagree"
            )
        coherence_id = _require_nonempty_string(
            source["coherence_id"], "source coherence_id"
        )
        code_pin = _require_code_pin(source["code_pin"], "source code_pin")
        plan_id = _require_identifier(source["plan_id"], "source plan_id")
        plan_sha256 = _require_sha256(source["plan_sha256"], "source plan_sha256")
        if plan_id not in plans or plans[plan_id]["plan_sha256"] != plan_sha256:
            raise BenchmarkValidationError("source plan authority drifted")
        executable_identity_kind = _require_identifier(
            source["executable_identity_kind"],
            "source executable_identity_kind",
        )
        if executable_identity_kind not in {
            "unavailable",
            "optimized_hlo",
            "executable_fingerprint",
            "executable_triple",
        }:
            raise BenchmarkValidationError(
                "source executable identity kind is unsupported"
            )
        executable_identity_sha256 = source["executable_identity_sha256"]
        if executable_identity_sha256 is not None:
            executable_identity_sha256 = _require_sha256(
                executable_identity_sha256,
                "source executable_identity_sha256",
            )
        if (executable_identity_kind == "unavailable") != (
            executable_identity_sha256 is None
        ):
            raise BenchmarkValidationError(
                "source executable identity kind/value disagree"
            )
        if role == "candidate" and executable_identity_sha256 is None:
            raise BenchmarkValidationError(
                "candidate source requires an executable SHA-256"
            )
        artifact_sha256 = _require_sha256(
            source["artifact_sha256"], "source artifact_sha256"
        )
        if not isinstance(source["artifact_path"], str):
            raise BenchmarkValidationError("source artifact_path must be a string")
        identity_evidence = _verify_source_identity_evidence(source, source_index)
        artifact_path = Path(source["artifact_path"])
        bindings = source["observations"]
        observed_artifact_sha256, bound_observations = _read_source_bindings(
            artifact_path=artifact_path,
            expected_artifact_sha256=artifact_sha256,
            source_id=source_id,
            source_index=source_index,
            bindings=bindings,
            watchpoints=watchpoints,
        )
        for observation in bound_observations:
            watchpoint_id = observation.watchpoint_id
            observations[watchpoint_id][role].append(observation)
        source_records.append(
            {
                "artifact_path": str(artifact_path),
                "artifact_sha256": observed_artifact_sha256,
                "code_pin": code_pin,
                "coherence_id": coherence_id,
                "executable_identity_kind": executable_identity_kind,
                "executable_identity_sha256": executable_identity_sha256,
                "id": source_id,
                "identity_evidence": identity_evidence,
                "observation_count": len(bindings),
                "observation_method": method,
                "plan_id": plan_id,
                "plan_sha256": plan_sha256,
                "role": role,
                "trust": trust,
            }
        )

    authority: dict[str, dict[str, Any]] = {}
    for role in ("accepted", "candidate"):
        identities = {
            (
                record["coherence_id"],
                record["code_pin"],
                record["plan_id"],
                record["plan_sha256"],
                record["executable_identity_kind"],
                record["executable_identity_sha256"],
            )
            for record in source_records
            if record["role"] == role
        }
        if len(identities) != 1:
            raise BenchmarkValidationError(
                f"{role} sources do not form one coherent authority"
            )
        (
            coherence_id,
            code_pin,
            plan_id,
            plan_sha256,
            executable_identity_kind,
            executable_identity_sha256,
        ) = identities.pop()
        authority[role] = {
            "code_pin": code_pin,
            "coherence_id": coherence_id,
            "executable_identity_kind": executable_identity_kind,
            "executable_identity_sha256": executable_identity_sha256,
            "plan_id": plan_id,
            "plan_sha256": plan_sha256,
        }

    coverage: list[dict[str, Any]] = []
    first_unobservable = None
    first_divergence = None
    for watchpoint_id, watchpoint in sorted(
        watchpoints.items(), key=lambda pair: pair[1]["order"]
    ):
        grouped = observations[watchpoint_id]
        if len(grouped["accepted"]) > 1 or len(grouped["candidate"]) > 1:
            raise BenchmarkValidationError(
                "watchpoint has ambiguous accepted/candidate authority: "
                f"{watchpoint_id}"
            )
        accepted = grouped["accepted"]
        candidate = grouped["candidate"]
        complete = len(accepted) == 1 and len(candidate) == 1
        comparison = (
            compare_array_observations(accepted[0], candidate[0])
            if complete
            else None
        )
        record = {
            "accepted_sources": [item.source_id for item in accepted],
            "authority_complete": complete,
            "candidate_sources": [item.source_id for item in candidate],
            "comparison": comparison,
            "id": watchpoint_id,
            "layer": watchpoint["layer"],
            "order": watchpoint["order"],
            "position": watchpoint["position"],
            "rejected_sources": [item.source_id for item in grouped["rejected"]],
            "semantic_dtype": watchpoint["semantic_dtype"],
            "shape": watchpoint["shape"],
            "supporting_sources": [item.source_id for item in grouped["supporting"]],
        }
        coverage.append(record)
        if first_unobservable is None and not complete:
            first_unobservable = {
                "id": watchpoint_id,
                "missing_accepted": len(accepted) == 0,
                "missing_candidate": len(candidate) == 0,
                "order": watchpoint["order"],
            }
        if (
            first_divergence is None
            and comparison is not None
            and not comparison["exact"]
        ):
            first_divergence = {
                **comparison,
                "layer": watchpoint["layer"],
                "order": watchpoint["order"],
                "position": watchpoint["position"],
            }

    gap_precedes_divergence = first_unobservable is not None and (
        first_divergence is None
        or first_unobservable["order"] < first_divergence["order"]
    )
    if gap_precedes_divergence:
        classification = "OBSERVABILITY_GAP"
        causal_frontier = {"kind": "unobservable", **first_unobservable}
    elif first_divergence is not None:
        classification = "DIVERGENCE_LOCALIZED"
        causal_frontier = {
            "id": first_divergence["watchpoint_id"],
            "kind": "divergence",
            "order": first_divergence["order"],
        }
    else:
        classification = "EXACT"
        causal_frontier = None

    return {
        "authority": authority,
        "causal_frontier": causal_frontier,
        "classification": classification,
        "contract_id": contract_id,
        "contract_sha256": contract_sha256,
        "coverage": coverage,
        "first_divergence": first_divergence,
        "first_unobservable": first_unobservable,
        "plans": sorted(plans.values(), key=lambda item: item["id"]),
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "sources": source_records,
    }


@contextmanager
def _open_directory_no_symlinks(path: Path) -> Iterator[int]:
    if ".." in path.parts:
        raise BenchmarkValidationError("output parent cannot contain '..'")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            "/" if path.is_absolute() else ".", directory_flags
        )
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot safely open output parent: {path}"
        ) from error
    try:
        for component in path.parts:
            if component in ("", ".", "/"):
                continue
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except OSError as error:
                raise BenchmarkValidationError(
                    f"cannot safely open output parent: {path}"
                ) from error
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def write_observability_report(path: Path, report: Mapping[str, Any]) -> None:
    """Create one append-only canonical report, refusing every occupied path."""

    if not path.name or path.name in (".", ".."):
        raise BenchmarkValidationError(f"output path is invalid: {path}")
    payload = (_canonical_json(report) + "\n").encode("ascii")
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    with _open_directory_no_symlinks(path.parent) as parent_descriptor:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise BenchmarkValidationError(
                f"cannot inspect output path: {path}"
            ) from error
        else:
            raise BenchmarkValidationError(f"output path is occupied: {path}")
        try:
            descriptor = os.open(
                path.name, file_flags, 0o644, dir_fd=parent_descriptor
            )
        except OSError as error:
            raise BenchmarkValidationError(f"cannot create output: {path}") from error
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent_descriptor)
        except BaseException:
            try:
                current = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except OSError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (
                created.st_dev,
                created.st_ino,
            ):
                os.unlink(path.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            raise
