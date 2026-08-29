#!/usr/bin/env python3
"""Stage bounded failure diagnostics without reproducible bulk payloads."""

from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path

MAX_KEPT_FILE_BYTES = 64 * 1024 * 1024
MAX_KEPT_TOTAL_BYTES = 256 * 1024 * 1024
REJECTED_RUN_TAG = "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusion_reason(relative: Path) -> str | None:
    value = relative.as_posix()
    if value.startswith("source_dumps/") and relative.name.startswith("topk."):
        return "reproducible_dsa_source_dump"
    if relative.name.startswith("vllm_") and relative.name.endswith(".tar.gz"):
        return "reproducible_vllm_source_archive"
    if relative.name.startswith("observer_") and relative.suffix == ".bundle":
        return "reproducible_observer_git_bundle"
    return None


def prepare_compact_failure_archive(source: Path, output: Path) -> dict[str, object]:
    if source.is_symlink():
        raise ValueError("failure archive source root must not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("failure archive source is not a directory")
    if source.name != REJECTED_RUN_TAG:
        raise ValueError("compact policy is bound to the rejected layer-1 run")
    if output.is_symlink():
        raise ValueError("failure archive output root must not be a symlink")
    if output.exists():
        output_resolved = output.resolve(strict=True)
    else:
        output_resolved = output.parent.resolve(strict=True) / output.name
    if (
        source == output_resolved
        or source.is_relative_to(output_resolved)
        or output_resolved.is_relative_to(source)
    ):
        raise ValueError("failure archive source and output must be disjoint")

    entries = sorted(source.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"compact failure archive refuses symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"compact failure archive refuses special entry: {path}")
        if path.relative_to(source).as_posix() == "failure_archive_manifest.json":
            raise ValueError("source contains reserved failure archive manifest path")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"append-only failure archive exists: {output}")
    else:
        output.mkdir(parents=True)

    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    snapshots: dict[Path, tuple[int, str]] = {}
    kept_bytes = 0
    for path in entries:
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        byte_count = path.stat().st_size
        record = {
            "byte_count": byte_count,
            "path": relative.as_posix(),
            "sha256": _sha256(path),
        }
        snapshots[relative] = (byte_count, str(record["sha256"]))
        reason = _exclusion_reason(relative)
        if reason is not None:
            record["reason"] = reason
            excluded.append(record)
            continue
        if byte_count > MAX_KEPT_FILE_BYTES:
            raise ValueError(f"unclassified file exceeds 64 MiB cap: {relative}")
        kept_bytes += byte_count
        if kept_bytes > MAX_KEPT_TOTAL_BYTES:
            raise ValueError("compact failure archive exceeds 256 MiB kept-byte cap")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if (
            destination.stat().st_size != byte_count
            or _sha256(destination) != record["sha256"]
            or path.stat().st_size != byte_count
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"failure archive copy drifted: {relative}")
        kept.append(record)

    final_entries = sorted(source.rglob("*"))
    if [path.relative_to(source) for path in final_entries] != [
        path.relative_to(source) for path in entries
    ]:
        raise RuntimeError("failure archive source inventory changed during copy")
    for path in final_entries:
        if path.is_symlink():
            raise RuntimeError("failure archive source became a symlink during copy")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                "failure archive source became a special entry during copy"
            )
        relative = path.relative_to(source)
        expected = snapshots.get(relative)
        if (
            expected is None
            or path.stat().st_size != expected[0]
            or _sha256(path) != expected[1]
        ):
            raise RuntimeError(f"failure archive source entry drifted: {relative}")

    report: dict[str, object] = {
        "artifact_kind": "greenfield_compact_failure_archive_manifest",
        "excluded": excluded,
        "excluded_byte_count": sum(int(item["byte_count"]) for item in excluded),
        "excluded_file_count": len(excluded),
        "format_version": 1,
        "kept": kept,
        "kept_byte_count": kept_bytes,
        "kept_file_count": len(kept),
        "policy": {
            "max_kept_file_bytes": MAX_KEPT_FILE_BYTES,
            "max_kept_total_bytes": MAX_KEPT_TOTAL_BYTES,
        },
        "source_directory_name": source.name,
    }
    encoded = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    report["manifest_sha256"] = sha256(encoded).hexdigest()
    (output / "failure_archive_manifest.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_compact_failure_archive(args.source, args.output)
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
