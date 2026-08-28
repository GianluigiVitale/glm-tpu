"""Selective final-owner loader for the bounded PP16 feature2 discriminator.

The production runtime files remain immutable.  This loader authenticates the
sealed manifest and safetensors headers, reads only the exact layer-0 plus
layer-1-query/indexer ranges admitted by the feature2 scaffold, and places each
range directly on its final LP2 owner.  It never constructs a source checkpoint
tree or reads layer-1 dense/later-layer payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from ..checkpoint.full_loader import (
    _array_bytes_sha256,
    _canonical_json,
    _memory_stats,
    _rss_peak_bytes,
)
from ..errors import BenchmarkValidationError, CheckpointValidationError
from .pp16_feature_sharded_state import (
    PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
    PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE,
    Feature2TensorRead,
    derive_feature2_tensor_allowlist,
    read_feature2_owner_headers,
    validate_feature2_acquisition_reads,
    validate_feature2_n82_manifest,
)


_DTYPE_BY_HEADER = {
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F32": np.dtype("<f4"),
    "U8": np.dtype("u1"),
}


@dataclass(slots=True)
class LoadedFeature2SelectiveCheckpoint:
    """Selected global arrays backed by the two exact final-owner buffers."""

    weights: dict[str, Any]
    state_manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        for array in self.weights.values():
            try:
                array.delete()
            except (AttributeError, RuntimeError):
                pass
        self.weights.clear()
        self.closed = True


def _read_manifest(root: Path) -> Mapping[str, Any]:
    manifest_path = root / "runtime_manifest.json"
    success_path = root / "SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise CheckpointValidationError(
            "feature2 selective runtime is incomplete"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(
            "feature2 selective runtime manifest is unreadable"
        ) from error
    if not isinstance(manifest, Mapping):
        raise CheckpointValidationError(
            "feature2 selective runtime manifest is not an object"
        )
    expected_success = (
        f"{PP16_FEATURE2_RUNTIME_MANIFEST_SHA256}  runtime_manifest.json\n"
    )
    if success_path.read_text() != expected_success:
        raise CheckpointValidationError(
            "feature2 selective runtime SUCCESS marker drifted"
        )
    try:
        validate_feature2_n82_manifest(manifest)
    except BenchmarkValidationError as error:
        raise CheckpointValidationError(
            "feature2 selective runtime manifest authentication failed"
        ) from error
    return manifest


def _manifest_file_records(
    manifest: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    records = [
        record
        for record in manifest.get("files", ())
        if isinstance(record, Mapping)
        and record.get("stage_id") == 0
        and record.get("device_slot") in (0, 1)
    ]
    by_slot = {int(record["device_slot"]): record for record in records}
    if len(records) != 2 or set(by_slot) != {0, 1}:
        raise CheckpointValidationError(
            "feature2 selective runtime lacks its exact two stage-0 owners"
        )
    return by_slot


def _validate_file_envelopes(
    root: Path,
    records: Mapping[int, Mapping[str, Any]],
) -> None:
    for slot, record in records.items():
        filename = record.get("destination_filename")
        file_bytes = record.get("file_bytes")
        if not isinstance(filename, str) or not isinstance(file_bytes, int):
            raise CheckpointValidationError(
                f"feature2 owner {slot} file envelope is invalid"
            )
        path = root / filename
        try:
            stat = path.stat()
        except OSError as error:
            raise CheckpointValidationError(
                f"feature2 owner {slot} payload is missing"
            ) from error
        if not path.is_file() or stat.st_size != file_bytes:
            raise CheckpointValidationError(
                f"feature2 owner {slot} payload size drifted"
            )


def _read_authenticated_range(
    path: Path,
    tensor: Feature2TensorRead,
    *,
    chunk_bytes: int,
) -> tuple[np.ndarray, str]:
    """Read, authenticate and retain the exact buffer passed to JAX."""

    if chunk_bytes <= 0:
        raise ValueError("feature2 selective chunk_bytes must be positive")
    dtype = _DTYPE_BY_HEADER.get(tensor.dtype)
    if dtype is None:
        raise CheckpointValidationError(
            f"feature2 selected tensor {tensor.name!r} has unsupported dtype"
        )
    expected_bytes = prod(tensor.shape) * dtype.itemsize
    if expected_bytes != tensor.byte_count:
        raise CheckpointValidationError(
            f"feature2 selected tensor {tensor.name!r} shape bytes drifted"
        )
    aligned_chunk = max(dtype.itemsize, chunk_bytes - chunk_bytes % dtype.itemsize)
    raw = bytearray(tensor.byte_count)
    view = memoryview(raw)
    digest = sha256()
    copied = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        while copied < tensor.byte_count:
            count = min(aligned_chunk, tensor.byte_count - copied)
            part = view[copied : copied + count]
            observed_count = os.preadv(
                fd, (part,), tensor.offset + copied
            )
            if observed_count != count:
                raise CheckpointValidationError(
                    f"feature2 selected tensor {tensor.name!r} is truncated"
                )
            digest.update(part)
            values = np.frombuffer(part, dtype=dtype)
            if tensor.dtype in ("BF16", "F32"):
                finite = bool(np.all(np.isfinite(values)))
            else:
                if not tensor.name.endswith(".weight_bits"):
                    raise CheckpointValidationError(
                        f"feature2 U8 tensor {tensor.name!r} is not FP8 state"
                    )
                finite = not bool(np.any((values & np.uint8(0x7F)) == 0x7F))
            if not finite:
                raise CheckpointValidationError(
                    f"feature2 selected tensor {tensor.name!r} is non-finite"
                )
            copied += count
    finally:
        os.close(fd)
    observed = digest.hexdigest()
    if observed != tensor.sha256:
        raise CheckpointValidationError(
            f"feature2 selected tensor {tensor.name!r} SHA-256 drifted"
        )
    host = np.frombuffer(raw, dtype=dtype).reshape(
        (1, *tensor.shape),
        order="C",
    )
    return host, observed


def _load_ranges_to_final_owners(
    root: Path,
    allowed: Sequence[Feature2TensorRead],
    devices: Sequence[Any],
    *,
    axis_name: str,
    chunk_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Execute an already-authenticated plan; production derives it internally."""

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(devices)
    if len(runtime_devices) != 2 or len(
        {int(device.id) for device in runtime_devices}
    ) != 2:
        raise CheckpointValidationError(
            "feature2 selective load requires two distinct final-owner devices"
        )
    by_slot: dict[int, list[Feature2TensorRead]] = {0: [], 1: []}
    for tensor in allowed:
        if tensor.device_slot not in by_slot:
            raise CheckpointValidationError(
                "feature2 selective plan contains an out-of-stage owner"
            )
        by_slot[tensor.device_slot].append(tensor)
    names_by_slot = {
        slot: {tensor.name for tensor in tensors}
        for slot, tensors in by_slot.items()
    }
    if not names_by_slot[0] or names_by_slot[0] != names_by_slot[1]:
        raise CheckpointValidationError(
            "feature2 selective owners have different tensor sets"
        )

    mesh = Mesh(np.asarray(runtime_devices, dtype=object), (axis_name,))
    local_by_name: dict[str, dict[int, Any]] = {
        name: {} for name in names_by_slot[0]
    }
    global_weights: dict[str, Any] = {}
    receipts: list[dict[str, Any]] = []
    loaded_local_arrays: list[Any] = []
    roundtrip_bytes = 0
    try:
        for slot, device in enumerate(runtime_devices):
            for tensor in sorted(
                by_slot[slot], key=lambda item: (item.offset, item.name)
            ):
                path = root / tensor.filename
                host, observed_sha = _read_authenticated_range(
                    path, tensor, chunk_bytes=chunk_bytes
                )
                array = jax.device_put(host, device)
                array.block_until_ready()
                if tuple(array.shape) != (1, *tensor.shape) or (
                    array.nbytes != tensor.byte_count
                ) or tuple(array.devices()) != (device,):
                    array.delete()
                    raise CheckpointValidationError(
                        f"feature2 tensor {tensor.name!r} missed its final owner"
                    )
                if _array_bytes_sha256(jax.device_get(array)) != observed_sha:
                    array.delete()
                    raise CheckpointValidationError(
                        f"feature2 tensor {tensor.name!r} device bytes drifted"
                    )
                roundtrip_bytes += tensor.byte_count
                local_by_name[tensor.name][slot] = array
                loaded_local_arrays.append(array)
                receipts.append(tensor.to_dict())
                del host

        for name in sorted(names_by_slot[0]):
            first = next(item for item in by_slot[0] if item.name == name)
            sharding = NamedSharding(
                mesh,
                P(axis_name, *(None for _ in first.shape)),
            )
            global_weights[name] = jax.make_array_from_single_device_arrays(
                (2, *first.shape),
                sharding,
                (local_by_name[name][0], local_by_name[name][1]),
            )
        return global_weights, receipts, {
            "device_roundtrip_bytes": roundtrip_bytes,
            "mesh_axis": axis_name,
            "owner_device_ids": [int(device.id) for device in runtime_devices],
        }
    except Exception:
        for array in global_weights.values():
            try:
                array.delete()
            except (AttributeError, RuntimeError):
                pass
        for array in loaded_local_arrays:
            try:
                array.delete()
            except (AttributeError, RuntimeError):
                pass
        raise


def load_feature2_selective_checkpoint(
    runtime_root: Path,
    devices: Sequence[Any],
    *,
    axis_name: str = "feature",
    chunk_bytes: int = 64 * 1024 * 1024,
) -> LoadedFeature2SelectiveCheckpoint:
    """Load exactly 39 selected tensors on each PP16 stage-0 owner."""

    root = Path(runtime_root)
    manifest = _read_manifest(root)
    headers = read_feature2_owner_headers(root, manifest)
    allowed = derive_feature2_tensor_allowlist(manifest, headers)
    records = _manifest_file_records(manifest)
    _validate_file_envelopes(root, records)
    runtime_devices = tuple(devices)
    if tuple(int(device.id) for device in runtime_devices) != tuple(
        int(records[slot]["device_id"]) for slot in (0, 1)
    ):
        raise CheckpointValidationError(
            "feature2 selective devices disagree with manifest final owners"
        )
    before = [_memory_stats(device) for device in runtime_devices]
    host_before = _rss_peak_bytes()
    weights, receipts, placement = _load_ranges_to_final_owners(
        root,
        allowed,
        runtime_devices,
        axis_name=axis_name,
        chunk_bytes=chunk_bytes,
    )
    try:
        read_report = validate_feature2_acquisition_reads(allowed, receipts)
        after = [_memory_stats(device) for device in runtime_devices]
        bytes_by_slot = read_report["bytes_by_slot"]
        if bytes_by_slot != {
            0: PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE,
            1: PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE,
        }:
            raise CheckpointValidationError(
                "feature2 selective loaded-byte totals drifted"
            )
        state_manifest: dict[str, Any] = {
            "artifact_kind": "greenfield_pp16_feature2_selective_state",
            "files": [
                {
                    "device_id": int(records[slot]["device_id"]),
                    "device_slot": slot,
                    "filename": records[slot]["destination_filename"],
                    "read_bytes": bytes_by_slot[slot],
                    "read_count": sum(
                        item.device_slot == slot for item in allowed
                    ),
                }
                for slot in (0, 1)
            ],
            "global_tensor_count": len(weights),
            "manifest_sha256": PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
            "plan_id": "PP16_LP2",
            "selected_read_count": len(receipts),
            "selected_weight_bytes_per_owner": (
                PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE
            ),
            **placement,
        }
        state_manifest["state_sha256"] = sha256(
            _canonical_json(state_manifest).encode("utf-8")
        ).hexdigest()
        load_record = {
            "device_memory_after": after,
            "device_memory_before": before,
            "device_roundtrip_verified": True,
            "host_peak_rss_after": _rss_peak_bytes(),
            "host_peak_rss_before": host_before,
            "receipts_sha256": sha256(
                _canonical_json(receipts).encode("utf-8")
            ).hexdigest(),
            "selected_reads": receipts,
        }
        return LoadedFeature2SelectiveCheckpoint(
            weights=weights,
            state_manifest=state_manifest,
            load_record=load_record,
        )
    except Exception:
        for array in weights.values():
            try:
                array.delete()
            except (AttributeError, RuntimeError):
                pass
        raise


def inspect_feature2_selective_plan(runtime_root: Path) -> dict[str, Any]:
    """Authenticate and summarize the exact read plan without payload reads."""

    root = Path(runtime_root)
    manifest = _read_manifest(root)
    headers = read_feature2_owner_headers(root, manifest)
    allowed = derive_feature2_tensor_allowlist(manifest, headers)
    records = _manifest_file_records(manifest)
    _validate_file_envelopes(root, records)
    report = validate_feature2_acquisition_reads(
        allowed, [item.to_dict() for item in allowed]
    )
    return {
        "bytes_by_slot": report["bytes_by_slot"],
        "manifest_sha256": PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
        "read_count": len(allowed),
        "tensor_names": sorted({item.name for item in allowed}),
    }
