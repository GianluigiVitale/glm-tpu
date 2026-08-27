#!/usr/bin/env python3
"""Seal the preserved PP16 compile acquisition without TPU execution.

The protected workload completed before its outer validator rejected the
replacement pod's non-identity launcher/JAX process mapping.  This recovery
copies only preserved bytes, replays the corrected validator, pins every
source object generation/CRC/SHA, and publishes to a new create-only prefix.
It never imports JAX or initializes a TPU.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any

import google_crc32c
from google.cloud import storage

from scripts.greenfield.validate_short_decoder_compile_pp16 import (
    validate_records,
)


APPROVED_BUCKET = "driftbench-dsv4-uc"
APPROVED_RESULTS_PREFIX = "results/"
SOURCE_FOLDERS = {"hlo": 3, "host_logs": 8, "host_records": 8}
SOURCE_TOP_LEVEL = (
    "census_post.txt",
    "census_pre.txt",
    "execute.txt",
    "preflight.json",
    "remote_vacancy.txt",
    "sync.txt",
)
RECOVERY_TOP_LEVEL = (
    *SOURCE_TOP_LEVEL,
    "census_recovery.txt",
    "evidence.sha256",
    "recovery.json",
    "source_orchestrator.log",
    "summary.json",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32c_file(path: Path) -> str:
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _parse_results_uri(uri: str) -> tuple[str, str]:
    approved = f"gs://{APPROVED_BUCKET}/{APPROVED_RESULTS_PREFIX}"
    if not uri.startswith(approved):
        raise ValueError("recovery must use the approved results bucket")
    prefix = uri.removeprefix(f"gs://{APPROVED_BUCKET}/").rstrip("/") + "/"
    if prefix == APPROVED_RESULTS_PREFIX:
        raise ValueError("recovery prefix must be below results/")
    return APPROVED_BUCKET, prefix


def _require_census(path: Path) -> None:
    workers = re.findall(
        r"^CENSUS_OK\s+(\S*?-w-[0-7])$", path.read_text(), re.MULTILINE
    )
    if len(workers) != 8 or len(set(workers)) != 8:
        raise ValueError(f"census is not authenticated 8/8 zero work: {path}")


def _verify_evidence_manifest(run_dir: Path) -> None:
    manifest = run_dir / "evidence.sha256"
    seen: set[str] = set()
    for line in manifest.read_text().splitlines():
        expected, separator, relative_name = line.partition("  ")
        if (
            not separator
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or relative_name in seen
        ):
            raise ValueError("recovery evidence manifest is malformed")
        path = run_dir / relative_name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"recovery evidence hash drifted: {relative_name}")
        seen.add(relative_name)
    expected_names = set(
        _recovery_files(run_dir, require_evidence_manifest=False)
    )
    if seen != expected_names:
        raise ValueError("recovery evidence manifest set drifted")


def _source_local_files(source_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for folder, count in SOURCE_FOLDERS.items():
        paths = sorted(path for path in (source_dir / folder).iterdir() if path.is_file())
        if len(paths) != count:
            raise ValueError(f"source {folder} expected {count} files, found {len(paths)}")
        for path in paths:
            files[f"{folder}/{path.name}"] = path
    for name in SOURCE_TOP_LEVEL:
        path = source_dir / name
        if not path.is_file():
            raise ValueError(f"source acquisition is missing {name}")
        files[name] = path
    orchestrator = source_dir / "orchestrator.log"
    if not orchestrator.is_file():
        raise ValueError("source acquisition is missing orchestrator.log")
    files["source_orchestrator.log"] = orchestrator
    return files


def _source_blob_name(
    relative_name: str, *, source_prefix: str, source_run_tag: str
) -> str:
    if relative_name.startswith(("hlo/", "host_logs/", "host_records/")):
        return source_prefix + relative_name
    original = (
        "orchestrator.log"
        if relative_name == "source_orchestrator.log"
        else relative_name
    )
    return (
        source_prefix
        + f"diagnostic_local/{source_run_tag}/"
        + original
    )


def _verify_source_objects(
    *,
    client: storage.Client,
    bucket_name: str,
    source_prefix: str,
    source_run_tag: str,
    local_files: dict[str, Path],
) -> list[dict[str, object]]:
    bucket = client.bucket(bucket_name)
    result: list[dict[str, object]] = []
    for relative_name, path in sorted(local_files.items()):
        blob_name = _source_blob_name(
            relative_name,
            source_prefix=source_prefix,
            source_run_tag=source_run_tag,
        )
        blob = bucket.get_blob(blob_name)
        crc = _crc32c_file(path)
        if blob is None:
            raise ValueError(f"preserved source object is missing: {blob_name}")
        if (
            int(blob.size) != path.stat().st_size
            or blob.crc32c != crc
            or not blob.generation
        ):
            raise ValueError(f"preserved source object drifted: {blob_name}")
        result.append(
            {
                "crc32c": crc,
                "generation": int(blob.generation),
                "name": blob_name,
                "recovered_as": relative_name,
                "sha256": _sha256_file(path),
                "size": int(blob.size),
            }
        )
    return result


def _recovery_files(
    run_dir: Path, *, require_evidence_manifest: bool = True
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for folder, count in SOURCE_FOLDERS.items():
        paths = sorted(path for path in (run_dir / folder).iterdir() if path.is_file())
        if len(paths) != count:
            raise ValueError(f"recovery {folder} expected {count} files, found {len(paths)}")
        for path in paths:
            files[f"{folder}/{path.name}"] = path
    for name in RECOVERY_TOP_LEVEL:
        if name == "evidence.sha256" and not require_evidence_manifest:
            continue
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"recovery evidence is missing {name}")
        files[name] = path
    return files


def prepare(
    *,
    source_dir: Path,
    run_dir: Path,
    recovery_census: Path,
    source_remote_prefix: str,
    remote_prefix: str,
    source_run_tag: str,
    recovery_run_tag: str,
    workload_code_hash: str,
    recovery_code_hash: str,
    client: storage.Client,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"recovery run directory already exists: {run_dir}")
    if len(workload_code_hash) != 40 or len(recovery_code_hash) != 40:
        raise ValueError("workload and recovery code hashes must be full Git SHAs")
    source_bucket, source_prefix = _parse_results_uri(source_remote_prefix)
    recovery_bucket, recovery_prefix = _parse_results_uri(remote_prefix)
    if source_bucket != recovery_bucket or source_prefix == recovery_prefix:
        raise ValueError("recovery requires a distinct prefix in the approved bucket")
    if not source_prefix.endswith(f"/{source_run_tag}/"):
        raise ValueError("source run tag and remote prefix disagree")
    if not recovery_prefix.endswith(f"/{recovery_run_tag}/"):
        raise ValueError("recovery run tag and remote prefix disagree")
    local_files = _source_local_files(source_dir)
    for name in ("census_pre.txt", "census_post.txt"):
        _require_census(local_files[name])
    _require_census(recovery_census)
    source_objects = _verify_source_objects(
        client=client,
        bucket_name=source_bucket,
        source_prefix=source_prefix,
        source_run_tag=source_run_tag,
        local_files=local_files,
    )

    run_dir.mkdir(parents=True)
    for relative_name, source in local_files.items():
        destination = run_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(recovery_census, run_dir / "census_recovery.txt")
    summary = validate_records(run_dir, workload_code_hash)
    recovery: dict[str, Any] = {
        "artifact_kind": "greenfield_pp16_short_decoder_compile_recovery",
        "gate_d_passed": False,
        "model_workload_rerun": False,
        "numerical_claim": False,
        "performance_claim": False,
        "recovery_code_hash": recovery_code_hash,
        "recovery_run_tag": recovery_run_tag,
        "remote_prefix": remote_prefix,
        "results_db_run_id": None,
        "source_objects": source_objects,
        "source_remote_prefix": source_remote_prefix,
        "source_run_tag": source_run_tag,
        "tpu_initialized_by_recovery": False,
        "workload_code_hash": workload_code_hash,
    }
    recovery["recovery_sha256"] = sha256(_canonical(recovery)).hexdigest()
    (run_dir / "recovery.json").write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n"
    )
    evidence = _recovery_files(run_dir, require_evidence_manifest=False)
    lines = [
        f"{_sha256_file(path)}  {name}\n"
        for name, path in sorted(evidence.items())
    ]
    (run_dir / "evidence.sha256").write_text("".join(lines))
    _recovery_files(run_dir)
    _verify_evidence_manifest(run_dir)
    return {"recovery": recovery, "summary": summary}


def publish(
    *,
    run_dir: Path,
    remote_prefix: str,
    recovery: dict[str, Any],
    summary: dict[str, Any],
    client: storage.Client,
) -> dict[str, Any]:
    bucket_name, prefix = _parse_results_uri(remote_prefix)
    bucket = client.bucket(bucket_name)
    files = _recovery_files(run_dir)
    _verify_evidence_manifest(run_dir)
    if list(client.list_blobs(bucket_name, prefix=prefix)):
        raise ValueError("recovery remote prefix is not vacant")
    records: list[dict[str, object]] = []
    for name, path in sorted(files.items()):
        blob = bucket.blob(prefix + name)
        blob.upload_from_filename(
            path, if_generation_match=0, checksum="crc32c", timeout=300
        )
        blob.reload()
        crc = _crc32c_file(path)
        if (
            int(blob.size) != path.stat().st_size
            or blob.crc32c != crc
            or not blob.generation
        ):
            raise ValueError(f"recovery remote identity drifted: {name}")
        records.append(
            {
                "crc32c": crc,
                "generation": int(blob.generation),
                "name": name,
                "sha256": _sha256_file(path),
                "size": int(blob.size),
            }
        )
    ledger: dict[str, Any] = {
        "objects": records,
        "remote_prefix": remote_prefix,
    }
    ledger["ledger_sha256"] = sha256(_canonical(ledger)).hexdigest()
    ledger_path = run_dir / "remote_objects.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger_blob = bucket.blob(prefix + "remote_objects.json")
    ledger_blob.upload_from_filename(
        ledger_path, if_generation_match=0, checksum="crc32c", timeout=300
    )
    ledger_blob.reload()
    ledger_crc = _crc32c_file(ledger_path)
    if (
        int(ledger_blob.size) != ledger_path.stat().st_size
        or ledger_blob.crc32c != ledger_crc
        or not ledger_blob.generation
    ):
        raise ValueError("recovery remote ledger identity drifted")
    success: dict[str, Any] = {
        "artifact_kind": "greenfield_pp16_short_decoder_compile_recovery_SUCCESS",
        "gate_d_passed": False,
        "model_workload_rerun": False,
        "numerical_claim": False,
        "optimized_hlo_sha256": summary["optimized_hlo_sha256"],
        "performance_claim": False,
        "recovery_code_hash": recovery["recovery_code_hash"],
        "recovery_run_tag": recovery["recovery_run_tag"],
        "remote_ledger_crc32c": ledger_crc,
        "remote_ledger_generation": int(ledger_blob.generation),
        "remote_ledger_sha256": ledger["ledger_sha256"],
        "runtime_manifest_sha256": summary["runtime_manifest_sha256"],
        "source_run_tag": recovery["source_run_tag"],
        "workload_code_hash": recovery["workload_code_hash"],
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    success_path = run_dir / "SUCCESS"
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
    success_blob = bucket.blob(prefix + "SUCCESS")
    success_blob.upload_from_filename(
        success_path, if_generation_match=0, checksum="crc32c", timeout=300
    )
    success_blob.reload()
    success_crc = _crc32c_file(success_path)
    if (
        int(success_blob.size) != success_path.stat().st_size
        or success_blob.crc32c != success_crc
        or not success_blob.generation
    ):
        raise ValueError("recovery terminal SUCCESS identity drifted")
    terminal = {
        blob.name.removeprefix(prefix): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    expected = set(files) | {"remote_objects.json", "SUCCESS"}
    if set(terminal) != expected:
        raise ValueError("recovery terminal remote set drifted")
    return {
        "ledger_sha256": ledger["ledger_sha256"],
        "object_count": len(expected),
        "success_sha256": success["success_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recovery-census", type=Path, required=True)
    parser.add_argument("--source-remote-prefix", required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--source-run-tag", required=True)
    parser.add_argument("--recovery-run-tag", required=True)
    parser.add_argument("--workload-code-hash", required=True)
    parser.add_argument("--recovery-code-hash", required=True)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = storage.Client()
    prepared = prepare(
        source_dir=args.source_dir,
        run_dir=args.run_dir,
        recovery_census=args.recovery_census,
        source_remote_prefix=args.source_remote_prefix,
        remote_prefix=args.remote_prefix,
        source_run_tag=args.source_run_tag,
        recovery_run_tag=args.recovery_run_tag,
        workload_code_hash=args.workload_code_hash,
        recovery_code_hash=args.recovery_code_hash,
        client=client,
    )
    if not args.publish:
        print(
            "PP16_COMPILE_RECOVERY_PREPARED "
            f"sha={prepared['recovery']['recovery_sha256']}"
        )
        return 0
    result = publish(
        run_dir=args.run_dir,
        remote_prefix=args.remote_prefix,
        recovery=prepared["recovery"],
        summary=prepared["summary"],
        client=client,
    )
    print(
        "PP16_COMPILE_RECOVERY_SEALED "
        f"objects={result['object_count']} ledger={result['ledger_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
