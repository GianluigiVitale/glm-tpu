"""Verification of a packed checkpoint: metadata, manifest and SUCCESS seals, file hashes."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence  # noqa: UP035 (isinstance(x, typing.Mapping) checks type(x) only)

from glm_tpu.config.model import ModelGeometry
from glm_tpu.config.site import approved_source_uri
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader.sharded_state.format import (
    RUNTIME_ARTIFACT_KIND,
    RUNTIME_FORMAT_VERSION,
    RUNTIME_PLAN_ID,
    RuntimeFilePlan,
    FILE_RECORD_KEYS,
    MANIFEST_KEYS,
    SUCCESS_ARTIFACT_KIND,
    SUCCESS_KEYS,
    SUCCESS_TAG,
    TENSOR_SCHEMA_KEYS,
    destination_record,
    require_digest,
    mapping_hash,
    sha256_file,
    build_runtime_file_plans,
)
from glm_tpu.model_loader.source_inventory import SourceInventory


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


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Authenticated layout ledger; no claim about on-disk tensor payloads."""

    root: Path
    manifest: Mapping[str, Any]
    success: Mapping[str, Any]
    plans: tuple[RuntimeFilePlan, ...]
    records_by_slot: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeCheckpoint(RuntimeMetadata):
    """Checkpoint admitted by the existing full-runtime verification policy."""


def _verify_runtime_value(
    root: Path,
    manifest: Mapping[str, Any],
    plans: Sequence[RuntimeFilePlan],
) -> Mapping[int, Mapping[str, Any]]:
    if set(manifest) != MANIFEST_KEYS:
        raise CheckpointValidationError("WS32 runtime manifest schema drifted")
    if (
        manifest.get("artifact_kind") != RUNTIME_ARTIFACT_KIND
        or (manifest.get("format_version") != RUNTIME_FORMAT_VERSION)
        or manifest.get("plan_id") != RUNTIME_PLAN_ID
    ):
        raise CheckpointValidationError("unsupported WS32 runtime checkpoint")
    if manifest.get("manifest_sha256") != mapping_hash(manifest, field="manifest_sha256"):
        raise CheckpointValidationError("WS32 runtime manifest checksum mismatch")
    tensor_schema = manifest.get("tensor_schema")
    if (
        not isinstance(tensor_schema, list)
        or any(not isinstance(item, Mapping) or set(item) != TENSOR_SCHEMA_KEYS for item in tensor_schema)
        or tensor_schema != [item.schema_dict() for item in plans[0].tensors]
    ):
        raise CheckpointValidationError("WS32 runtime tensor schema drifted")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 32:
        raise CheckpointValidationError("WS32 runtime requires 32 file records")
    by_slot = {record.get("device_slot"): record for record in records}
    if set(by_slot) != set(range(32)) or len(by_slot) != len(records):
        raise CheckpointValidationError("WS32 runtime file slots are incomplete")
    for plan in plans:
        record = by_slot[plan.device_slot]
        if not isinstance(record, Mapping) or set(record) != FILE_RECORD_KEYS:
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
            raise CheckpointValidationError(f"WS32 runtime file metadata drifted for slot {plan.device_slot}")
        require_digest(record.get("sha256"), field=f"slot{plan.device_slot}.sha256")
        _crc32c(record.get("crc32c"), field=f"slot{plan.device_slot}.crc32c")
        hashes = record.get("tensor_sha256")
        if not isinstance(hashes, list) or len(hashes) != len(plan.tensors):
            raise CheckpointValidationError("WS32 runtime tensor hash ledger drifted")
        for index, digest in enumerate(hashes):
            require_digest(digest, field=f"slot{plan.device_slot}.tensor{index}")
    if manifest.get("packed_payload_bytes") != sum(plan.payload_bytes for plan in plans) or manifest.get(
        "packed_file_bytes"
    ) != sum(plan.file_bytes for plan in plans):
        raise CheckpointValidationError("WS32 runtime byte totals drifted")
    return by_slot


def _verify_runtime_files(
    metadata: RuntimeMetadata,
    *,
    verify_file_hashes: bool,
    verify_file_hash_slots: frozenset[int] | None,
    local_slot_layout: bool,
) -> None:
    if local_slot_layout and (not verify_file_hashes or verify_file_hash_slots is None):
        raise ValueError("WS32 local slot layout requires hash verification of the owned slots")
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
            raise CheckpointValidationError(f"WS32 runtime file is missing or truncated: {plan.filename!r}")
        if verify_file_hashes and (verify_file_hash_slots is None or plan.device_slot in verify_file_hash_slots):
            observed = destination_record(
                path,
                plan,
                chunk_bytes=64 * 1024 * 1024,
            )
            if observed != record:
                raise CheckpointValidationError(f"WS32 runtime file/tensor checksum drifted: {plan.filename!r}")


def verify_runtime_checkpoint(
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
) -> VerifiedRuntimeCheckpoint:
    """Re-derive every layout field and verify a protected sealed artifact."""

    require_digest(expected_manifest_sha256, field="expected_manifest_sha256")
    require_digest(expected_success_sha256, field="expected_success_sha256")
    require_digest(expected_mesh_hash, field="expected_mesh_hash")
    require_digest(expected_topology_hash, field="expected_topology_hash")
    selected_hash_slots = None
    if verify_file_hash_slots is not None:
        selected_hash_slots = frozenset(verify_file_hash_slots)
        if (
            not selected_hash_slots
            or len(selected_hash_slots) != len(verify_file_hash_slots)
            or any(
                not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 32
                for slot in selected_hash_slots
            )
        ):
            raise ValueError("WS32 runtime verification slot subset is invalid")
    metadata = _read_runtime_metadata(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_success_sha256=expected_success_sha256,
        expected_mesh_hash=expected_mesh_hash,
        expected_topology_hash=expected_topology_hash,
        inventory=inventory,
        geometry=geometry,
    )
    _verify_runtime_files(
        metadata,
        verify_file_hashes=verify_file_hashes,
        verify_file_hash_slots=selected_hash_slots,
        local_slot_layout=local_slot_layout,
    )
    return VerifiedRuntimeCheckpoint(
        root=metadata.root,
        manifest=metadata.manifest,
        success=metadata.success,
        plans=metadata.plans,
        records_by_slot=metadata.records_by_slot,
    )


def _read_runtime_metadata(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_success_sha256: str,
    expected_mesh_hash: str,
    expected_topology_hash: str,
    inventory: SourceInventory,
    geometry: ModelGeometry,
) -> RuntimeMetadata:
    """Re-derive/authenticate all metadata without opening any owner payload."""

    require_digest(expected_manifest_sha256, field="expected_manifest_sha256")
    require_digest(expected_success_sha256, field="expected_success_sha256")
    require_digest(expected_mesh_hash, field="expected_mesh_hash")
    require_digest(expected_topology_hash, field="expected_topology_hash")
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise CheckpointValidationError("WS32 runtime checkpoint lacks manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise CheckpointValidationError("WS32 runtime manifest identity drifted")
    if manifest.get("mesh_hash") != expected_mesh_hash:
        raise CheckpointValidationError("WS32 runtime mesh identity drifted")
    report, plans = build_runtime_file_plans(inventory, geometry, mesh_hash=expected_mesh_hash)
    if (
        manifest.get("geometry") != geometry.to_dict()
        or (manifest.get("geometry_sha256") != geometry.geometry_hash)
        or manifest.get("placement_report") != report.to_dict()
    ):
        raise CheckpointValidationError("WS32 runtime geometry/placement drifted")
    source = manifest.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "files",
            "inventory_sha256",
            "revision",
            "uri",
        }
        or source.get("inventory_sha256") != inventory.inventory_sha256
        or source.get("revision") != inventory.source_revision
    ):
        raise CheckpointValidationError("WS32 runtime source identity drifted")
    if not isinstance(source.get("uri"), str) or not approved_source_uri(source["uri"]):
        raise CheckpointValidationError("WS32 runtime source URI drifted")
    require_digest(manifest.get("code_hash"), field="code_hash", lengths=(40, 64))
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
                raise CheckpointValidationError(f"WS32 source ledger drifted for {expected.filename!r}")
        require_digest(observed.get("sha256"), field=f"source:{expected.filename}")
    success_path = root / "SUCCESS"
    if not success_path.is_file():
        raise CheckpointValidationError("WS32 runtime checkpoint lacks SUCCESS")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(success, Mapping) or set(success) != SUCCESS_KEYS:
        raise CheckpointValidationError("WS32 runtime SUCCESS schema drifted")
    if success.get("success_sha256") != expected_success_sha256 or (
        success.get("success_sha256") != mapping_hash(success, field="success_sha256")
    ):
        raise CheckpointValidationError("WS32 runtime SUCCESS checksum mismatch")
    exact_success = {
        "artifact_kind": SUCCESS_ARTIFACT_KIND,
        "code_hash": manifest["code_hash"],
        "file_count": 32,
        "format_version": RUNTIME_FORMAT_VERSION,
        "manifest_file_sha256": sha256_file(path),
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
            require_digest(success.get(field), field=f"SUCCESS.{field}")
        except ValueError as exc:
            raise CheckpointValidationError(str(exc)) from exc
    tag = success.get("tag")
    if not isinstance(tag, str) or SUCCESS_TAG.fullmatch(tag) is None:
        raise CheckpointValidationError("WS32 runtime SUCCESS tag drifted")
    by_slot = _verify_runtime_value(
        root,
        manifest,
        plans,
    )
    return RuntimeMetadata(
        root=root,
        manifest=manifest,
        success=success,
        plans=plans,
        records_by_slot=by_slot,
    )
