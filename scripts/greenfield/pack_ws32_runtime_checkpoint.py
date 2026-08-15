#!/usr/bin/env python3
"""Distributed offline packer for the complete WS32 final-owner checkpoint."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (  # noqa: E402
    WS32_RUNTIME_PLAN_ID,
    WS32_RUNTIME_SLOT_RECORD_KIND,
    Ws32RuntimePackConfig,
    finalize_ws32_runtime_checkpoint,
    pack_ws32_runtime_slots,
)
from glm_tpu.greenfield.partitioning.source_inventory import (  # noqa: E402
    SourceInventory,
    inspect_source_inventory,
)
from glm_tpu.greenfield.types import ModelGeometry  # noqa: E402


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


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only WS32 evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_code(expected: str) -> str:
    observed = _head()
    if observed != expected:
        raise RuntimeError(
            f"WS32 code pin drifted: expected={expected}, observed={observed}"
        )
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("WS32 protected pack requires a clean worktree")
    return observed


def _geometry() -> ModelGeometry:
    value = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(value)


def _inventory(path: Path) -> SourceInventory:
    return inspect_source_inventory(path)


def _config(args: argparse.Namespace, code_hash: str) -> Ws32RuntimePackConfig:
    return Ws32RuntimePackConfig(
        source_root=args.source_root,
        source_uri=args.source_uri,
        output_dir=args.output,
        code_hash=code_hash,
        mesh_hash=args.mesh_hash,
    )


def _sha_crc_file(path: Path) -> tuple[str, str]:
    import google_crc32c

    digest = sha256()
    crc = google_crc32c.Checksum()
    with path.open("rb", buffering=0) as stream:
        for raw in iter(lambda: stream.read(64 * 1024 * 1024), b""):
            digest.update(raw)
            crc.update(raw)
    return digest.hexdigest(), base64.b64encode(crc.digest()).decode("ascii")


def _hash_sources(args: argparse.Namespace, code_hash: str) -> None:
    inventory = _inventory(args.source_inventory)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("source-hash shard index is out of range")
    selected = tuple(
        record
        for index, record in enumerate(inventory.files)
        if index % args.shard_count == args.shard_index
    )
    files = []
    for record in selected:
        path = args.source_root / record.filename
        if not path.is_file() or path.stat().st_size != record.file_bytes:
            raise RuntimeError(f"source file size drifted: {record.filename}")
        digest, crc32c = _sha_crc_file(path)
        files.append(
            {
                "crc32c": crc32c,
                "file_bytes": record.file_bytes,
                "filename": record.filename,
                "sha256": digest,
            }
        )
    result: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_source_file_hashes",
        "code_hash": code_hash,
        "files": files,
        "format_version": 1,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "source_inventory_sha256": inventory.inventory_sha256,
    }
    result["record_sha256"] = _mapping_hash(result, field="record_sha256")
    _write_json_once(args.output, result)
    print(json.dumps(result, sort_keys=True))


def _pack_slots(args: argparse.Namespace, code_hash: str) -> None:
    inventory = _inventory(args.source_inventory)
    records = pack_ws32_runtime_slots(
        _config(args, code_hash),
        inventory,
        _geometry(),
        device_slots=tuple(args.slots),
    )
    print(json.dumps(records, sort_keys=True))


def _read_exact_records(paths: list[Path], kind: str) -> list[dict[str, Any]]:
    values = []
    for path in paths:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("artifact_kind") != kind:
            raise RuntimeError(f"wrong WS32 record kind at {path}")
        if value.get("record_sha256") != _mapping_hash(
            value, field="record_sha256"
        ):
            raise RuntimeError(f"WS32 record checksum drifted at {path}")
        values.append(value)
    return values


def _finalize(args: argparse.Namespace, code_hash: str) -> None:
    inventory = _inventory(args.source_inventory)
    geometry = _geometry()
    slot_records = _read_exact_records(
        args.slot_records, WS32_RUNTIME_SLOT_RECORD_KIND
    )
    source_records = _read_exact_records(
        args.source_records, "greenfield_ws32_source_file_hashes"
    )
    common = {
        "code_hash": code_hash,
        "format_version": 1,
        "source_inventory_sha256": inventory.inventory_sha256,
    }
    files = []
    slots = []
    for record in slot_records:
        for field, expected in {
            **common,
            "geometry_sha256": geometry.geometry_hash,
            "mesh_hash": args.mesh_hash,
            "plan_id": WS32_RUNTIME_PLAN_ID,
        }.items():
            if record.get(field) != expected:
                raise RuntimeError(f"WS32 slot record drifted field {field!r}")
        slots.extend(record["slots"])
        files.extend(record["files"])
    if sorted(slots) != list(range(32)) or len(set(slots)) != 32:
        raise RuntimeError("WS32 slot records do not cover 32 unique owners")
    if len(files) != 32:
        raise RuntimeError("WS32 slot file record count drifted")

    source_sha256: dict[str, str] = {}
    source_names = []
    source_shards = []
    for record in source_records:
        for field, expected in common.items():
            if record.get(field) != expected:
                raise RuntimeError(f"WS32 source record drifted field {field!r}")
        if record.get("shard_count") != len(source_records):
            raise RuntimeError("WS32 source hash shard count drifted")
        source_shards.append(record.get("shard_index"))
        for item in record["files"]:
            name = item["filename"]
            source_names.append(name)
            source_sha256[name] = item["sha256"]
    expected_source_names = [record.filename for record in inventory.files]
    if sorted(source_shards) != list(range(len(source_records))):
        raise RuntimeError("WS32 source hash shard indices are incomplete")
    if sorted(source_names) != sorted(expected_source_names) or len(
        source_names
    ) != len(set(source_names)):
        raise RuntimeError("WS32 source hash records are incomplete or duplicate")

    manifest = finalize_ws32_runtime_checkpoint(
        _config(args, code_hash),
        inventory,
        geometry,
        file_records=files,
        source_file_sha256=source_sha256,
    )
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "manifest_sha256": manifest["manifest_sha256"],
                "packed_file_bytes": manifest["packed_file_bytes"],
                "packed_payload_bytes": manifest["packed_payload_bytes"],
                "source_files": len(manifest["source"]["files"]),
            },
            sort_keys=True,
        )
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--mesh-hash", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    pack = subparsers.add_parser("pack-slots")
    _common(pack)
    pack.add_argument("--slots", type=int, nargs="+", required=True)

    hashes = subparsers.add_parser("hash-sources")
    hashes.add_argument("--source-inventory", type=Path, required=True)
    hashes.add_argument("--source-root", type=Path, required=True)
    hashes.add_argument("--output", type=Path, required=True)
    hashes.add_argument("--expected-code-hash", required=True)
    hashes.add_argument("--shard-index", type=int, required=True)
    hashes.add_argument("--shard-count", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    _common(finalize)
    finalize.add_argument("--slot-records", type=Path, nargs="+", required=True)
    finalize.add_argument("--source-records", type=Path, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_hash = _require_clean_code(args.expected_code_hash)
    if args.mode == "pack-slots":
        _pack_slots(args, code_hash)
    elif args.mode == "hash-sources":
        _hash_sources(args, code_hash)
    else:
        _finalize(args, code_hash)


if __name__ == "__main__":
    main()
