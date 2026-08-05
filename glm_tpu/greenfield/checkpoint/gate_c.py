"""Bounded final-owner checkpoint for the Gate C dense/DSA/IndexShare proof.

The artifact is a content-addressed subset of the protected complete PP8
layout.  It preserves every selected placement byte-for-byte and validates
the raw source tensor hashes captured by the independent Gate C oracle while
streaming all four stage-local owners together.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..partitioning.manifest import validate_layout_manifest
from .stream_pack import (
    DestinationFilePlan,
    build_destination_file_plans,
    destination_groups,
    stream_pack_group,
)


FORMAT_VERSION = 1
LAYOUT_ARTIFACT_KIND = "greenfield_gate_c_checkpoint_layout"
PACKED_ARTIFACT_KIND = "greenfield_gate_c_packed_checkpoint"
ORACLE_ARTIFACT_KIND = "greenfield_gate_c_oracle"
PLAN_ID = "PP8_LP4"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: Mapping[str, Any], *, field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(field, None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(
    value: object,
    *,
    field: str,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointValidationError(f"{field} is not a lowercase digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"cannot parse JSON artifact {path}") from exc
    if not isinstance(decoded, dict):
        raise CheckpointValidationError(f"JSON artifact is not an object: {path}")
    return decoded


def _validate_oracle_manifest(value: Mapping[str, Any]) -> None:
    if value.get("artifact_kind") != ORACLE_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong Gate C oracle artifact kind")
    if value.get("format_version") != FORMAT_VERSION:
        raise CheckpointValidationError("unsupported Gate C oracle format")
    if value.get("manifest_sha256") != _mapping_hash(
        value, field="manifest_sha256"
    ):
        raise CheckpointValidationError("Gate C oracle manifest SHA-256 mismatch")
    _digest(value.get("manifest_sha256"), field="oracle_manifest_sha256")
    source_tensors = value.get("source_tensors")
    if not isinstance(source_tensors, list) or not source_tensors:
        raise CheckpointValidationError("Gate C oracle source tensors are absent")
    names = []
    for record in source_tensors:
        if not isinstance(record, Mapping):
            raise CheckpointValidationError("Gate C oracle source record is invalid")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise CheckpointValidationError("Gate C oracle source name is invalid")
        names.append(name)
        _digest(record.get("sha256"), field=f"source tensor {name} SHA-256")
        if (
            not isinstance(record.get("shape"), list)
            or not isinstance(record.get("dtype"), str)
            or not isinstance(record.get("byte_count"), int)
            or record["byte_count"] <= 0
            or not isinstance(record.get("source_shard"), str)
        ):
            raise CheckpointValidationError(
                f"Gate C oracle source metadata is invalid for {name!r}"
            )
    if len(names) != len(set(names)):
        raise CheckpointValidationError("Gate C oracle source names are duplicate")


def _destination_records(
    placements: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parent_files = {
        record["filename"]: record for record in parent["destination_files"]
    }
    aggregates: dict[str, tuple[int, int]] = {}
    for placement in placements:
        for destination in placement["destinations"]:
            filename = destination["filename"]
            byte_count, tensor_count = aggregates.get(filename, (0, 0))
            aggregates[filename] = (
                byte_count + destination["byte_count"],
                tensor_count + 1,
            )
    records = []
    for filename in sorted(aggregates):
        if filename not in parent_files:
            raise CheckpointValidationError(
                f"Gate C destination {filename!r} is absent from parent layout"
            )
        record = deepcopy(parent_files[filename])
        record["planned_payload_bytes"], record["tensor_count"] = aggregates[
            filename
        ]
        records.append(record)
    return records


def build_gate_c_layout(
    *,
    parent_layout: Mapping[str, Any],
    oracle_manifest: Mapping[str, Any],
    code_hash: str,
) -> dict[str, Any]:
    """Derive the exact Gate C subset from the protected complete PP8 layout."""

    validate_layout_manifest(parent_layout)
    _validate_oracle_manifest(oracle_manifest)
    _digest(code_hash, field="code_hash", lengths=(40, 64))
    if parent_layout.get("plan_id") != PLAN_ID:
        raise CheckpointValidationError("Gate C requires the protected PP8 layout")
    parent_source = parent_layout["source"]
    if (
        oracle_manifest.get("model_id") != parent_source.get("model_id")
        or oracle_manifest.get("source_revision") != parent_source.get("revision")
        or oracle_manifest.get("source_uri", "").rstrip("/")
        != parent_source.get("uri", "").rstrip("/")
    ):
        raise CheckpointValidationError(
            "Gate C oracle and parent layout source identity disagree"
        )
    oracle_records = {
        record["name"]: dict(record)
        for record in oracle_manifest["source_tensors"]
    }
    parent_placements = {
        placement["source"]["name"]: placement
        for placement in parent_layout["placements"]
    }
    if not set(oracle_records) <= set(parent_placements):
        missing = sorted(set(oracle_records) - set(parent_placements))
        raise CheckpointValidationError(
            f"Gate C tensors are absent from the parent layout: {missing}"
        )
    placements = []
    for name in sorted(oracle_records):
        oracle_record = oracle_records[name]
        placement = deepcopy(parent_placements[name])
        source = placement["source"]
        if (
            source.get("shape") != oracle_record.get("shape")
            or source.get("dtype") != oracle_record.get("dtype")
            or source.get("byte_count") != oracle_record.get("byte_count")
            or source.get("filename") != oracle_record.get("source_shard")
        ):
            raise CheckpointValidationError(
                f"Gate C oracle metadata disagrees with parent placement for {name!r}"
            )
        if placement.get("load_set") != "base_decoder" or {
            destination.get("stage_id")
            for destination in placement["destinations"]
        } != {0}:
            raise CheckpointValidationError(
                f"Gate C tensor {name!r} is not wholly owned by PP8 stage 0"
            )
        placements.append(placement)
    source_filenames = {
        placement["source"]["filename"] for placement in placements
    }
    source_files = [
        deepcopy(record)
        for record in parent_source["files"]
        if record["filename"] in source_filenames
    ]
    if {record["filename"] for record in source_files} != source_filenames:
        raise CheckpointValidationError("Gate C parent source file ledger is incomplete")
    source_tensors = [deepcopy(oracle_records[name]) for name in sorted(oracle_records)]
    value: dict[str, Any] = {
        "artifact_kind": LAYOUT_ARTIFACT_KIND,
        "code_hash": code_hash,
        "consumer_layer": oracle_manifest["consumer_layer"],
        "destination_files": _destination_records(placements, parent_layout),
        "format_version": FORMAT_VERSION,
        "oracle_manifest_sha256": oracle_manifest["manifest_sha256"],
        "packed_payload_bytes": sum(
            placement["packed_byte_count"] for placement in placements
        ),
        "parent_layout_manifest_sha256": parent_layout["manifest_sha256"],
        "placements": placements,
        "plan_group_hash": parent_layout["plan_group_hash"],
        "plan_id": parent_layout["plan_id"],
        "plan_manifest_sha256": parent_layout["plan_manifest_sha256"],
        "producer_layer": oracle_manifest["producer_layer"],
        "source": {
            "files": source_files,
            "inventory_sha256": parent_source["inventory_sha256"],
            "model_id": parent_source["model_id"],
            "payload_bytes": sum(
                record["byte_count"] for record in source_tensors
            ),
            "revision": parent_source["revision"],
            "tensors": source_tensors,
            "uri": parent_source["uri"],
        },
        "topology_hash": parent_layout["topology_hash"],
    }
    value["manifest_sha256"] = _mapping_hash(value, field="manifest_sha256")
    validate_gate_c_layout(value)
    return value


def validate_gate_c_layout(value: Mapping[str, Any]) -> None:
    """Fail closed on subset hash, ownership, source, or byte drift."""

    if value.get("artifact_kind") != LAYOUT_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong Gate C checkpoint layout kind")
    if value.get("format_version") != FORMAT_VERSION:
        raise CheckpointValidationError("unsupported Gate C checkpoint layout")
    if value.get("manifest_sha256") != _mapping_hash(
        value, field="manifest_sha256"
    ):
        raise CheckpointValidationError("Gate C checkpoint layout SHA-256 mismatch")
    for field, lengths in (
        ("manifest_sha256", (64,)),
        ("code_hash", (40, 64)),
        ("oracle_manifest_sha256", (64,)),
        ("parent_layout_manifest_sha256", (64,)),
        ("plan_group_hash", (64,)),
        ("plan_manifest_sha256", (64,)),
        ("topology_hash", (64,)),
    ):
        _digest(value.get(field), field=field, lengths=lengths)
    if value.get("plan_id") != PLAN_ID:
        raise CheckpointValidationError("Gate C layout is not PP8_LP4")
    source = value.get("source")
    placements = value.get("placements")
    files = value.get("destination_files")
    if (
        not isinstance(source, Mapping)
        or not isinstance(placements, list)
        or not placements
        or not isinstance(files, list)
        or len(files) != 4
    ):
        raise CheckpointValidationError("Gate C layout ledgers are incomplete")
    source_tensors = source.get("tensors")
    source_files = source.get("files")
    if not isinstance(source_tensors, list) or not isinstance(source_files, list):
        raise CheckpointValidationError("Gate C source ledgers are incomplete")
    oracle_by_name: dict[str, Mapping[str, Any]] = {}
    for record in source_tensors:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise CheckpointValidationError("Gate C source tensor record is invalid")
        name = record["name"]
        if name in oracle_by_name:
            raise CheckpointValidationError("Gate C source tensor names are duplicate")
        oracle_by_name[name] = record
        _digest(record.get("sha256"), field=f"source tensor {name} SHA-256")
    placement_by_name: dict[str, Mapping[str, Any]] = {}
    packed_bytes = 0
    aggregates: dict[str, tuple[int, int]] = {}
    scale_locality: dict[str, set[tuple[int, int, str]]] = {}
    weight_locality: dict[str, set[tuple[int, int, str]]] = {}
    for placement in placements:
        if not isinstance(placement, Mapping):
            raise CheckpointValidationError("Gate C placement is invalid")
        tensor = placement.get("source")
        destinations = placement.get("destinations")
        if not isinstance(tensor, Mapping) or not isinstance(destinations, list):
            raise CheckpointValidationError("Gate C placement ledgers are invalid")
        name = tensor.get("name")
        if not isinstance(name, str) or name in placement_by_name:
            raise CheckpointValidationError("Gate C placement names are invalid")
        placement_by_name[name] = placement
        oracle_record = oracle_by_name.get(name)
        if oracle_record is None or (
            tensor.get("shape") != oracle_record.get("shape")
            or tensor.get("dtype") != oracle_record.get("dtype")
            or tensor.get("byte_count") != oracle_record.get("byte_count")
            or tensor.get("filename") != oracle_record.get("source_shard")
        ):
            raise CheckpointValidationError(
                f"Gate C source metadata drifted for {name!r}"
            )
        locality = set()
        placement_bytes = 0
        for destination in destinations:
            if not isinstance(destination, Mapping):
                raise CheckpointValidationError(
                    f"Gate C destination identity drifted for {name!r}"
                )
            slot = destination.get("device_slot")
            expected_filename = (
                f"base_decoder/stage_00/device_slot_{slot:02d}.safetensors"
                if isinstance(slot, int) and not isinstance(slot, bool)
                else None
            )
            if (
                destination.get("stage_id") != 0
                or slot not in (0, 1, 2, 3)
                or destination.get("filename") != expected_filename
                or not isinstance(destination.get("byte_count"), int)
            ):
                raise CheckpointValidationError(
                    f"Gate C destination identity drifted for {name!r}"
                )
            filename = destination["filename"]
            previous_bytes, previous_count = aggregates.get(filename, (0, 0))
            aggregates[filename] = (
                previous_bytes + destination["byte_count"],
                previous_count + 1,
            )
            placement_bytes += destination["byte_count"]
            locality.add((0, slot, filename))
        if placement.get("load_set") != "base_decoder" or placement.get(
            "packed_byte_count"
        ) != placement_bytes:
            raise CheckpointValidationError(
                f"Gate C placement bytes drifted for {name!r}"
            )
        packed_bytes += placement_bytes
        target = (
            scale_locality
            if placement.get("value_class") == "fp8_scale"
            else weight_locality
        )
        target[name] = locality
    if set(placement_by_name) != set(oracle_by_name):
        raise CheckpointValidationError("Gate C source and placement sets disagree")
    if source.get("payload_bytes") != sum(
        record["byte_count"] for record in source_tensors
    ):
        raise CheckpointValidationError("Gate C source bytes do not reconcile")
    if value.get("packed_payload_bytes") != packed_bytes:
        raise CheckpointValidationError("Gate C packed bytes do not reconcile")
    file_names = []
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("filename"), str
        ):
            raise CheckpointValidationError("Gate C destination file is invalid")
        filename = record["filename"]
        file_names.append(filename)
        if aggregates.get(filename) != (
            record.get("planned_payload_bytes"),
            record.get("tensor_count"),
        ):
            raise CheckpointValidationError(
                f"Gate C destination file does not reconcile: {filename!r}"
            )
    if len(file_names) != len(set(file_names)) or set(file_names) != set(aggregates):
        raise CheckpointValidationError("Gate C destination file set is invalid")
    if any(not isinstance(record, Mapping) for record in source_files):
        raise CheckpointValidationError("Gate C source file record is invalid")
    if {record.get("filename") for record in source_files} != {
        record.get("source_shard") for record in source_tensors
    }:
        raise CheckpointValidationError("Gate C source file set does not reconcile")
    for scale_name, locality in scale_locality.items():
        if weight_locality.get(scale_name.removesuffix("_scale_inv")) != locality:
            raise CheckpointValidationError(
                f"Gate C FP8 scale is not colocated with its weight: {scale_name!r}"
            )


def validate_gate_c_layout_bindings(
    value: Mapping[str, Any],
    *,
    parent_layout: Mapping[str, Any],
    oracle_manifest: Mapping[str, Any],
) -> None:
    """Prove every subset record is copied from the named parent and oracle."""

    validate_gate_c_layout(value)
    expected = build_gate_c_layout(
        parent_layout=parent_layout,
        oracle_manifest=oracle_manifest,
        code_hash=value["code_hash"],
    )
    if dict(value) != expected:
        raise CheckpointValidationError(
            "Gate C layout is not the exact protected parent/oracle derivation"
        )


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise CheckpointValidationError(f"refusing to overwrite artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise CheckpointValidationError(f"stale artifact temporary exists: {temporary}")
    temporary.write_text(encoded)
    temporary.replace(path)


def pack_gate_c_checkpoint(
    *,
    layout: Mapping[str, Any],
    source_root: Path,
    output_dir: Path,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream the four final PP8 owners and commit their manifest last."""

    validate_gate_c_layout(layout)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise CheckpointValidationError(
            f"refusing to overwrite Gate C checkpoint {output_dir}"
        )
    output_dir.mkdir(parents=True)
    _write_json_once(output_dir / "layout_manifest.json", layout)
    plans = build_destination_file_plans(
        layout, validate_layout_contract=False
    )
    groups = destination_groups(plans)
    if len(groups) != 1 or len(groups[0]) != 4:
        raise CheckpointValidationError(
            "Gate C checkpoint must contain one complete four-owner PP8 stage"
        )
    handles = []
    outputs = {}
    partials: dict[str, Path] = {}
    try:
        for plan in groups[0]:
            final = output_dir / plan.filename
            final.parent.mkdir(parents=True, exist_ok=True)
            partial = final.with_name(f".{final.name}.partial")
            handle = partial.open("xb")
            handles.append(handle)
            outputs[plan.filename] = handle
            partials[plan.filename] = partial
        evidence = stream_pack_group(
            layout=layout,
            plans=groups[0],
            source_root=source_root,
            outputs=outputs,
            chunk_bytes=chunk_bytes,
            validate_layout_contract=False,
            expected_source_sha256={
                record["name"]: record["sha256"]
                for record in layout["source"]["tensors"]
            },
        )
    finally:
        for handle in handles:
            handle.close()
    evidence_by_name = {record.filename: record for record in evidence}
    for plan in groups[0]:
        partials[plan.filename].replace(output_dir / plan.filename)
    files = []
    for plan in plans:
        record = evidence_by_name[plan.filename]
        files.append(
            {
                "device_id": plan.device_id,
                "device_slot": plan.device_slot,
                "file_bytes": record.file_bytes,
                "filename": plan.filename,
                "header_bytes": len(plan.header),
                "header_sha256": sha256(plan.header).hexdigest(),
                "payload_bytes": plan.payload_bytes,
                "sha256": record.sha256,
                "stage_id": plan.stage_id,
                "tensor_count": len(plan.tensors),
            }
        )
    manifest: dict[str, Any] = {
        "artifact_kind": PACKED_ARTIFACT_KIND,
        "code_hash": layout["code_hash"],
        "file_count": len(files),
        "files": files,
        "format_version": FORMAT_VERSION,
        "layout_manifest_sha256": layout["manifest_sha256"],
        "oracle_manifest_sha256": layout["oracle_manifest_sha256"],
        "packed_file_bytes": sum(record["file_bytes"] for record in files),
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "parent_layout_manifest_sha256": layout[
            "parent_layout_manifest_sha256"
        ],
        "plan_id": layout["plan_id"],
        "source_payload_bytes": layout["source"]["payload_bytes"],
        "source_revision": layout["source"]["revision"],
        "source_tensor_count": len(layout["source"]["tensors"]),
    }
    manifest["manifest_sha256"] = _mapping_hash(
        manifest, field="manifest_sha256"
    )
    _write_json_once(output_dir / "manifest.json", manifest)
    inspect_gate_c_checkpoint(output_dir)
    return manifest


def _expected_file_record(plan: DestinationFilePlan) -> dict[str, Any]:
    return {
        "device_id": plan.device_id,
        "device_slot": plan.device_slot,
        "file_bytes": plan.file_bytes,
        "filename": plan.filename,
        "header_bytes": len(plan.header),
        "header_sha256": sha256(plan.header).hexdigest(),
        "payload_bytes": plan.payload_bytes,
        "stage_id": plan.stage_id,
        "tensor_count": len(plan.tensors),
    }


def inspect_gate_c_checkpoint(
    output_dir: Path,
    *,
    parent_layout: Mapping[str, Any] | None = None,
    oracle_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the bounded layout, all files, metadata, and optional bindings."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    layout = _read_json(output_dir / "layout_manifest.json")
    validate_gate_c_layout(layout)
    if (parent_layout is None) != (oracle_manifest is None):
        raise CheckpointValidationError(
            "parent layout and oracle manifest must be supplied together"
        )
    if parent_layout is not None and oracle_manifest is not None:
        validate_gate_c_layout_bindings(
            layout,
            parent_layout=parent_layout,
            oracle_manifest=oracle_manifest,
        )
    manifest = _read_json(output_dir / "manifest.json")
    if manifest.get("artifact_kind") != PACKED_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong Gate C packed artifact kind")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointValidationError("unsupported Gate C packed artifact")
    if manifest.get("manifest_sha256") != _mapping_hash(
        manifest, field="manifest_sha256"
    ):
        raise CheckpointValidationError("Gate C packed manifest SHA-256 mismatch")
    plans = build_destination_file_plans(
        layout, validate_layout_contract=False
    )
    records = manifest.get("files")
    if not isinstance(records, list):
        raise CheckpointValidationError("Gate C packed file ledger is absent")
    if any(not isinstance(record, Mapping) for record in records):
        raise CheckpointValidationError("Gate C packed file record is invalid")
    by_name = {record.get("filename"): record for record in records}
    if len(by_name) != len(records) or set(by_name) != {
        plan.filename for plan in plans
    }:
        raise CheckpointValidationError("Gate C packed file set is invalid")
    for plan in plans:
        record = by_name[plan.filename]
        expected = _expected_file_record(plan)
        for field, value in expected.items():
            if record.get(field) != value:
                raise CheckpointValidationError(
                    f"Gate C packed file field drifted for {plan.filename!r}: {field}"
                )
        _digest(record.get("sha256"), field=f"{plan.filename} SHA-256")
        path = output_dir / plan.filename
        if path.stat().st_size != plan.file_bytes:
            raise CheckpointValidationError(
                f"Gate C packed file size mismatch: {plan.filename!r}"
            )
        if _sha256_file(path) != record["sha256"]:
            raise CheckpointValidationError(
                f"Gate C packed file SHA-256 mismatch: {plan.filename!r}"
            )
        expected_tensors = {tensor.name: tensor for tensor in plan.tensors}
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected_tensors):
                raise CheckpointValidationError(
                    f"Gate C packed tensor set mismatch: {plan.filename!r}"
                )
            metadata = handle.metadata() or {}
            if (
                metadata.get("greenfield_layout_manifest_sha256")
                != layout["manifest_sha256"]
                or metadata.get("greenfield_destination_filename")
                != plan.filename
            ):
                raise CheckpointValidationError(
                    f"Gate C packed header metadata mismatch: {plan.filename!r}"
                )
            for name, tensor in expected_tensors.items():
                tensor_slice = handle.get_slice(name)
                if (
                    tuple(tensor_slice.get_shape()) != tensor.shape
                    or tensor_slice.get_dtype() != tensor.dtype
                ):
                    raise CheckpointValidationError(
                        f"Gate C packed tensor metadata mismatch: {name!r}"
                    )
    expected_manifest = {
        "code_hash": layout["code_hash"],
        "file_count": len(plans),
        "layout_manifest_sha256": layout["manifest_sha256"],
        "oracle_manifest_sha256": layout["oracle_manifest_sha256"],
        "packed_file_bytes": sum(plan.file_bytes for plan in plans),
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "parent_layout_manifest_sha256": layout[
            "parent_layout_manifest_sha256"
        ],
        "plan_id": layout["plan_id"],
        "source_payload_bytes": layout["source"]["payload_bytes"],
        "source_revision": layout["source"]["revision"],
        "source_tensor_count": len(layout["source"]["tensors"]),
    }
    for field, value in expected_manifest.items():
        if manifest.get(field) != value:
            raise CheckpointValidationError(
                f"Gate C packed manifest field drifted: {field}"
            )
    return manifest


def read_gate_c_layout(path: Path) -> dict[str, Any]:
    """Read and validate a Gate C subset layout."""

    value = _read_json(path)
    validate_gate_c_layout(value)
    return value
