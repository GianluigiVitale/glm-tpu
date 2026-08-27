#!/usr/bin/env python3
"""Bounded real-byte proof for canonical-to-PP8/WS32 reproduction paths."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any


SOURCE_INVENTORY_SHA256 = (
    "a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4"
)
PP8_MANIFEST_SHA256 = (
    "0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1"
)
WS32_MANIFEST_SHA256 = (
    "c04f800edf15651198ab2c5183fff8a8ae9a427609b59cc61ccabdab32f5ee08"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: dict[str, Any], hash_field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(hash_field, None)
    return sha256(_canonical_json(unhashed).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _tensor_bytes(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as stream:
        header_length_raw = stream.read(8)
        if len(header_length_raw) != 8:
            raise RuntimeError(f"truncated safetensor header: {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header = json.loads(stream.read(header_length))
        record = header.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"missing tensor {name!r}: {path}")
        offsets = record.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) for item in offsets)
        ):
            raise RuntimeError(f"invalid tensor offsets for {name!r}: {path}")
        start, stop = offsets
        stream.seek(8 + header_length + start)
        payload = stream.read(stop - start)
        if len(payload) != stop - start:
            raise RuntimeError(f"truncated tensor {name!r}: {path}")
    return record, payload


def _matrix_slice(
    raw: bytes,
    *,
    rows: int,
    columns: int,
    item_bytes: int,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> bytes:
    if len(raw) != rows * columns * item_bytes:
        raise RuntimeError("source matrix byte count drifted")
    row_bytes = columns * item_bytes
    return b"".join(
        raw[
            row * row_bytes + column_start * item_bytes :
            row * row_bytes + column_stop * item_bytes
        ]
        for row in range(row_start, row_stop)
    )


def _exact_check(
    *, name: str, expected: bytes, observed: bytes, transform: str
) -> dict[str, Any]:
    expected_hash = sha256(expected).hexdigest()
    observed_hash = sha256(observed).hexdigest()
    if expected != observed:
        raise RuntimeError(f"bounded recreation mismatch: {name}")
    return {
        "byte_count": len(expected),
        "name": name,
        "recreated_sha256": expected_hash,
        "retained_sha256": observed_hash,
        "transform": transform,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--pp8-plan-layout", type=Path, required=True)
    parser.add_argument("--pp8-root", type=Path, required=True)
    parser.add_argument("--ws32-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = _read_json(args.source_inventory)
    if inventory.get("inventory_sha256") != SOURCE_INVENTORY_SHA256:
        raise RuntimeError("canonical source inventory identity drifted")
    pp8_manifest = _read_json(args.pp8_root / "packed_manifest.json")
    ws32_manifest = _read_json(args.ws32_root / "manifest.json")
    if (
        pp8_manifest.get("manifest_sha256") != PP8_MANIFEST_SHA256
        or _mapping_hash(pp8_manifest, "manifest_sha256") != PP8_MANIFEST_SHA256
    ):
        raise RuntimeError("PP8 packed manifest identity drifted")
    if (
        ws32_manifest.get("manifest_sha256") != WS32_MANIFEST_SHA256
        or _mapping_hash(ws32_manifest, "manifest_sha256") != WS32_MANIFEST_SHA256
    ):
        raise RuntimeError("WS32 runtime manifest identity drifted")
    pp8_layout = args.pp8_root / "layout_manifest.json"
    if sha256(pp8_layout.read_bytes()).hexdigest() != sha256(
        args.pp8_plan_layout.read_bytes()
    ).hexdigest():
        raise RuntimeError("preserved PP8 plan layout differs from packed layout")

    source_file = args.source_root / "model-00001-of-00141.safetensors"
    pp8_file = (
        args.pp8_root
        / "base_decoder/stage_00/device_slot_00.safetensors"
    )
    ws32_file = args.ws32_root / "device_slot_00.safetensors"
    norm_name = "model.layers.0.input_layernorm.weight"
    scale_source_name = "model.layers.0.mlp.down_proj.weight_scale_inv"

    norm_record, norm_source = _tensor_bytes(source_file, norm_name)
    if norm_record.get("dtype") != "BF16" or norm_record.get("shape") != [6144]:
        raise RuntimeError("canonical norm tensor schema drifted")
    _, pp8_norm = _tensor_bytes(pp8_file, norm_name)
    _, ws32_norm = _tensor_bytes(ws32_file, norm_name)

    scale_record, scale_source = _tensor_bytes(source_file, scale_source_name)
    if scale_record.get("dtype") != "F32" or scale_record.get("shape") != [48, 96]:
        raise RuntimeError("canonical FP8 scale tensor schema drifted")
    _, pp8_scale = _tensor_bytes(pp8_file, scale_source_name)
    _, ws32_scale = _tensor_bytes(
        ws32_file, "model.layers.0.mlp.down_proj.scale_inv"
    )

    checks = [
        _exact_check(
            name="pp8_replicated_norm",
            expected=norm_source,
            observed=pp8_norm,
            transform="replicated_identity",
        ),
        _exact_check(
            name="pp8_axis1_scale_shard_slot0",
            expected=_matrix_slice(
                scale_source,
                rows=48,
                columns=96,
                item_bytes=4,
                row_start=0,
                row_stop=48,
                column_start=0,
                column_stop=24,
            ),
            observed=pp8_scale,
            transform="axis1_slice_0_24",
        ),
        _exact_check(
            name="ws32_feature_norm_shard_slot0",
            expected=norm_source[: 1536 * 2],
            observed=ws32_norm,
            transform="feature_slice_0_1536",
        ),
        _exact_check(
            name="ws32_feature_expert_scale_tile_slot0",
            expected=_matrix_slice(
                scale_source,
                rows=48,
                columns=96,
                item_bytes=4,
                row_start=0,
                row_stop=12,
                column_start=0,
                column_stop=12,
            ),
            observed=ws32_scale,
            transform="feature_expert_tile_rows_0_12_columns_0_12",
        ),
    ]
    schema = ws32_manifest.get("tensor_schema")
    files = ws32_manifest.get("files")
    if not isinstance(schema, list) or not isinstance(files, list):
        raise RuntimeError("WS32 tensor hash ledger is missing")
    slot0 = next(
        (record for record in files if record.get("device_slot") == 0), None
    )
    if not isinstance(slot0, dict) or not isinstance(slot0.get("tensor_sha256"), list):
        raise RuntimeError("WS32 slot-0 tensor hash ledger is missing")
    hash_by_name = {
        record["name"]: digest
        for record, digest in zip(schema, slot0["tensor_sha256"], strict=True)
    }
    if (
        hash_by_name[norm_name] != sha256(ws32_norm).hexdigest()
        or hash_by_name["model.layers.0.mlp.down_proj.scale_inv"]
        != sha256(ws32_scale).hexdigest()
    ):
        raise RuntimeError("WS32 manifest tensor hashes disagree with retained bytes")

    proof: dict[str, Any] = {
        "artifact_kind": "greenfield_checkpoint_bounded_recreation_proof",
        "checks": checks,
        "format_version": 1,
        "pp8_manifest_sha256": PP8_MANIFEST_SHA256,
        "pp8_plan_layout_file_sha256": sha256(
            args.pp8_plan_layout.read_bytes()
        ).hexdigest(),
        "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
        "status": "passed",
        "total_recreated_bytes": sum(check["byte_count"] for check in checks),
        "ws32_manifest_sha256": WS32_MANIFEST_SHA256,
    }
    proof["proof_sha256"] = _mapping_hash(proof, "proof_sha256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(proof["proof_sha256"])


if __name__ == "__main__":
    main()
