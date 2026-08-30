#!/usr/bin/env python3
"""Seal a DB485-identity compile-only log and seven scheduled TPU HLOs."""

from __future__ import annotations

import argparse
import json
import re
import stat
from hashlib import sha256
from pathlib import Path

from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import (
    ACCEPTED_FULL_CODE_PIN,
    ACCEPTED_VLLM_PIN,
    ACCEPTED_VLLM_VERSION,
    ACCEPTED_VLLM_VERSION_FILE_SHA256,
    validate_compile_only_log,
    validate_hlo_owner_receipts,
    validate_scheduled_hlo_files,
)

HARNESS_PIN = "3409bf58a758e7bfa398eb92051db701d623e6b9"
CALLBACK_CERTIFICATE_SHA256 = (
    "6e58bca961c0629799683ccfb75efda195206885e36975e497630ec379076e48"
)


def validate_regular_tree(run_dir: Path) -> list[Path]:
    """Return regular evidence files after refusing links and special entries."""

    try:
        root_stat = run_dir.lstat()
        root = run_dir.resolve(strict=True)
    except OSError as exc:
        raise SystemExit("compile-only run root is unavailable") from exc
    if run_dir.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise SystemExit("compile-only run root is not a real directory")
    files = []
    for path in sorted(run_dir.rglob("*")):
        try:
            path_stat = path.lstat()
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"unsafe compile-only tree entry: {path}") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not (
            stat.S_ISDIR(path_stat.st_mode) or stat.S_ISREG(path_stat.st_mode)
        ):
            raise SystemExit(f"unsafe compile-only tree entry: {path}")
        if stat.S_ISREG(path_stat.st_mode):
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--hlo-dir", required=True, type=Path)
    parser.add_argument("--hlo-owner-receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--greenfield-pin", required=True)
    parser.add_argument("--accepted-bundle-sha256", required=True)
    parser.add_argument("--vllm-archive-sha256", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--vllm-version-file-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--network-validator-sha256", required=True)
    parser.add_argument("--callback-certificate-sha256", required=True)
    parser.add_argument("--accepted-source-pin", required=True)
    parser.add_argument("--vllm-source-pin", required=True)
    parser.add_argument("--harness-pin", required=True)
    parser.add_argument("--remote-prefix", required=True)
    args = parser.parse_args()

    files_before_manifest = validate_regular_tree(args.run_dir)
    run_root = args.run_dir.resolve()
    expected_output = args.run_dir / "manifest.json"
    if args.output != expected_output:
        raise SystemExit("compile-only manifest output path is invalid")
    reserved = (
        args.output,
        args.run_dir / "SUCCESS",
        args.run_dir / "remote_objects.json",
    )
    if any(path.exists() or path.is_symlink() for path in reserved):
        raise SystemExit("compile-only manifest output already exists")
    for path in (args.log, args.hlo_dir, args.hlo_owner_receipts):
        try:
            path.resolve(strict=True).relative_to(run_root)
        except ValueError as exc:
            raise SystemExit(f"evidence path escapes run directory: {path}") from exc
        except OSError as exc:
            raise SystemExit(f"evidence path is unavailable: {path}") from exc
    try:
        args.output.parent.resolve(strict=True).relative_to(run_root)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"manifest output escapes run directory: {args.output}"
        ) from exc
    sha_fields = (
        args.accepted_bundle_sha256,
        args.vllm_archive_sha256,
        args.vllm_version_file_sha256,
        args.launcher_sha256,
        args.network_validator_sha256,
        args.callback_certificate_sha256,
    )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", args.greenfield_pin)
        or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in sha_fields)
        or args.accepted_source_pin != ACCEPTED_FULL_CODE_PIN
        or args.vllm_source_pin != ACCEPTED_VLLM_PIN
        or args.vllm_version != ACCEPTED_VLLM_VERSION
        or args.vllm_version_file_sha256 != ACCEPTED_VLLM_VERSION_FILE_SHA256
        or args.harness_pin != HARNESS_PIN
        or args.callback_certificate_sha256 != CALLBACK_CERTIFICATE_SHA256
        or args.remote_prefix
        != (
            "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
            f"accepted_compile_only_hlo/{args.run_tag}"
        )
    ):
        raise SystemExit("compile-only provenance identity is invalid")

    report = validate_compile_only_log(args.log.read_bytes(), run_tag=args.run_tag)
    report["hlo_owner"] = validate_hlo_owner_receipts(
        args.hlo_owner_receipts.read_bytes()
    )
    report["hlo"] = validate_scheduled_hlo_files(
        [
            path
            for path in args.hlo_dir.iterdir()
            if path.is_file() and path.name.endswith(".txt.gz")
        ]
    )
    raw_inventory = args.hlo_dir / "raw_hlo_inventory.txt"
    raw_inventory_bytes = raw_inventory.read_bytes()
    inventory_paths: dict[str, int] = {}
    for raw_line in raw_inventory_bytes.decode("utf-8", errors="strict").splitlines():
        name, separator, raw_size = raw_line.rpartition("\t")
        if (
            not separator
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or not raw_size.isdigit()
            or int(raw_size) <= 0
        ):
            raise SystemExit("raw HLO inventory entry is invalid")
        if name in inventory_paths:
            raise SystemExit("raw HLO inventory contains a duplicate path")
        inventory_paths[name] = int(raw_size)
    compressed_raw_names = {item["raw_name"] for item in report["hlo"]["files"]}
    inventoried_after_codegen = {
        Path(name).name
        for name in inventory_paths
        if name.endswith(".after_codegen.txt")
    }
    if len(inventory_paths) < 7 or inventoried_after_codegen != compressed_raw_names:
        raise SystemExit("raw HLO inventory is incomplete")
    for item in report["hlo"]["files"]:
        matching = [
            size
            for name, size in inventory_paths.items()
            if Path(name).name == item["raw_name"]
        ]
        if matching != [item["raw_bytes"]]:
            raise SystemExit("raw HLO inventory size differs from sealed bytes")
    bucket_map_path = args.hlo_dir / "bucket_hlo_map.tsv"
    bucket_map_bytes = bucket_map_path.read_bytes()
    bucket_map = []
    for raw_line in bucket_map_bytes.decode("utf-8", errors="strict").splitlines():
        fields = raw_line.split("\t")
        if len(fields) != 4 or not fields[0].isdigit() or not fields[2].isdigit():
            raise SystemExit("bucket/HLO map entry is invalid")
        bucket_map.append(
            {
                "num_tokens": int(fields[0]),
                "raw_path": fields[1],
                "raw_bytes": int(fields[2]),
                "sealed_name": fields[3],
            }
        )
    expected_map = [
        {
            "num_tokens": item["num_tokens"],
            "raw_path": next(
                name for name in inventory_paths if Path(name).name == item["raw_name"]
            ),
            "raw_bytes": item["raw_bytes"],
            "sealed_name": item["name"],
        }
        for item in report["hlo"]["files"]
    ]
    if bucket_map != expected_map:
        raise SystemExit("bucket/HLO map differs from sealed HLO contents")
    report["hlo"]["raw_inventory"] = {
        "byte_count": len(raw_inventory_bytes),
        "line_count": len(inventory_paths),
        "sha256": sha256(raw_inventory_bytes).hexdigest(),
    }
    report["hlo"]["bucket_map"] = {
        "records": bucket_map,
        "sha256": sha256(bucket_map_bytes).hexdigest(),
    }
    report["provenance"] = {
        "accepted_bundle_sha256": args.accepted_bundle_sha256,
        "accepted_source_pin": args.accepted_source_pin,
        "callback_certificate_sha256": args.callback_certificate_sha256,
        "greenfield_pin": args.greenfield_pin,
        "harness_pin": args.harness_pin,
        "launcher_sha256": args.launcher_sha256,
        "network_validator_sha256": args.network_validator_sha256,
        "remote_prefix": args.remote_prefix,
        "vllm_archive_sha256": args.vllm_archive_sha256,
        "vllm_source_pin": args.vllm_source_pin,
        "vllm_version": args.vllm_version,
        "vllm_version_file_sha256": args.vllm_version_file_sha256,
    }
    output = args.output.resolve()
    files = []
    for path in files_before_manifest:
        relative = path.relative_to(args.run_dir).as_posix()
        if path.resolve() == output:
            continue
        raw = path.read_bytes()
        files.append(
            {
                "byte_count": len(raw),
                "path": relative,
                "sha256": sha256(raw).hexdigest(),
            }
        )
    report["archive_files_before_manifest"] = files
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(payload)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
