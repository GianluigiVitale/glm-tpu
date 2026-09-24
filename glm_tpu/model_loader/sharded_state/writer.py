"""Packing owner shards: slot files, destination records and the manifest of a packed checkpoint."""

from __future__ import annotations

from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from glm_tpu.config.model import ModelGeometry
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader.placement import Ws32RuntimePlacementReport, placements_for_ws32_source_tensor
from glm_tpu.model_loader.sharded_state.format import WS32_RUNTIME_ARTIFACT_KIND, WS32_RUNTIME_FORMAT_VERSION, WS32_RUNTIME_PLAN_ID, WS32_RUNTIME_SLOT_RECORD_KIND, Ws32RuntimeFilePlan, Ws32RuntimePackConfig, _destination_record, _digest, _mapping_hash, _sha256_file, build_ws32_runtime_file_plans
from glm_tpu.model_loader.sharded_state.verify import Ws32RuntimeMetadata, _verify_ws32_runtime_files, _verify_ws32_runtime_value
from glm_tpu.model_loader.source_inventory import SourceFile, SourceInventory


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
