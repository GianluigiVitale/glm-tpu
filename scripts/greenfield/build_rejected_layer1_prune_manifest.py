#!/usr/bin/env python3
"""Build an exact generation-bound prune manifest for the rejected run."""

from __future__ import annotations

import argparse
import base64
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

TAG = "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z"
BUCKET = "driftbench-dsv4-uc"
PREFIX = f"oracles/greenfield/glm52/layer1_rms_input/8k/{TAG}"
PAYLOAD_PREFIX = f"{PREFIX}/diagnostic_local/{TAG}/"
EXPECTED = {
    "delete": {"file_count": 485, "byte_count": 3_079_571_533},
    "keep": {"file_count": 45, "byte_count": 14_629_391},
}
EXPECTED_MANIFEST_SHA256 = (
    "2cdcbc5caca23580bd792dad2228a1abaf29b5baf3612ebee20a9ff0327fd6ed"
)
EXPECTED_FILE_SHA256 = (
    "b582f02d8008572d26e8f133175608d646941b222586942f580bc376bb2503cb"
)


def _digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _serialized_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _record(item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("failure-object metadata is not a mapping")
    if metadata.get("bucket") != BUCKET:
        raise ValueError("unexpected failure-object bucket")
    name = metadata.get("name", "")
    if not isinstance(name, str) or not name.startswith(PAYLOAD_PREFIX):
        raise ValueError(f"unexpected failure-object name: {name}")
    relative = name.removeprefix(PAYLOAD_PREFIX)
    if not relative:
        raise ValueError("failure-object relative path is empty")
    raw_size = metadata.get("size")
    raw_generation = metadata.get("generation")
    if not isinstance(raw_size, str) or not raw_size.isdigit():
        raise ValueError(f"invalid failure-object size for {name}")
    if not isinstance(raw_generation, str) or not raw_generation.isdigit():
        raise ValueError(f"invalid failure-object generation for {name}")
    byte_count = int(raw_size)
    generation = raw_generation
    if int(generation) <= 0:
        raise ValueError(f"invalid failure-object generation for {name}")
    crc32c = metadata.get("crc32c")
    if not isinstance(crc32c, str):
        raise TypeError(f"invalid failure-object CRC32C for {name}")
    try:
        decoded_crc32c = base64.b64decode(crc32c, validate=True)
    except ValueError as error:
        raise ValueError(f"invalid failure-object CRC32C for {name}") from error
    if len(decoded_crc32c) != 4:
        raise ValueError(f"invalid failure-object CRC32C for {name}")
    url = f"gs://{BUCKET}/{name}#{generation}"
    record = {
        "byte_count": byte_count,
        "crc32c": crc32c,
        "generation": generation,
        "name": name,
        "relative_path": relative,
        "url": url,
    }
    if relative.startswith("source_dumps/") and Path(relative).name.startswith("topk."):
        return record, "delete"
    if Path(relative).name.startswith("vllm_") and relative.endswith(".tar.gz"):
        return record, "delete"
    if Path(relative).name.startswith("observer_") and relative.endswith(".bundle"):
        return record, "delete"
    return record, "keep"


def build_manifest(listing: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"delete": [], "keep": []}
    names: set[str] = set()
    for item in listing:
        record, disposition = _record(item)
        name = str(record["name"])
        if name in names:
            raise ValueError(f"duplicate failure-object name: {name}")
        names.add(name)
        groups[disposition].append(record)
    for records in groups.values():
        records.sort(key=lambda item: str(item["name"]))
    summary = {
        key: {
            "byte_count": sum(int(item["byte_count"]) for item in records),
            "file_count": len(records),
        }
        for key, records in groups.items()
    }
    if summary != EXPECTED:
        raise ValueError(f"rejected-run object inventory drifted: {summary!r}")
    manifest: dict[str, Any] = {
        "artifact_kind": "greenfield_rejected_layer1_generation_bound_prune_manifest",
        "bucket": BUCKET,
        "delete": groups["delete"],
        "format_version": 1,
        "keep": groups["keep"],
        "prefix": PREFIX,
        "reason": (
            "callback observer reproduced frozen DB551 perturbation; delete only "
            "reproducible topk dumps and source transports after this exact "
            "generation-bound manifest is archived"
        ),
        "run_tag": TAG,
        "summary": summary,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    if manifest["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "rejected-run generation-bound semantic identity drifted: "
            f"{manifest['manifest_sha256']}"
        )
    file_sha256 = sha256(_serialized_bytes(manifest)).hexdigest()
    if file_sha256 != EXPECTED_FILE_SHA256:
        raise ValueError(
            f"rejected-run generation-bound file identity drifted: {file_sha256}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only prune manifest exists: {args.output}")
    listing = json.loads(args.listing.read_text(encoding="utf-8"))
    manifest = build_manifest(listing)
    args.output.write_bytes(_serialized_bytes(manifest))
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
