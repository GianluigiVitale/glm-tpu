#!/usr/bin/env python3
"""Generation/CRC seal a successful non-performance PP16 compile proof."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import google_crc32c
from google.cloud import storage


TOP_LEVEL_EVIDENCE = (
    "census_post.txt",
    "census_pre.txt",
    "evidence.sha256",
    "execute.txt",
    "orchestrator.sealed.log",
    "preflight.json",
    "remote_vacancy.txt",
    "summary.json",
    "sync.txt",
)

DECODER_HLO_FILES = frozenset(
    {
        "decoder_78layer_2k_token.hlo_contract.json",
        "decoder_78layer_2k_token.optimized_hlo.txt.gz",
        "decoder_78layer_2k_token.stablehlo.mlir.gz",
    }
)
EXACT_QUERY_HLO_FILES = frozenset(
    {
        "dsa_query_weight_materializer.hlo_contract.json",
        "dsa_query_weight_materializer.optimized_hlo.txt.gz",
    }
)
PP16_COMPILE_HLO_FILES = DECODER_HLO_FILES | EXACT_QUERY_HLO_FILES


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _crc(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")


def _parse_remote(remote: str) -> tuple[str, str]:
    expected = "gs://driftbench-dsv4-uc/results/"
    if not remote.startswith(expected):
        raise ValueError("PP16 compile proof must use the approved results bucket")
    bucket, prefix = remote[5:].split("/", 1)
    return bucket, prefix.rstrip("/") + "/"


def _local_evidence(run_dir: Path) -> dict[str, Path]:
    local: dict[str, Path] = {}
    expected_counts = {"host_records": 8, "host_logs": 8}
    for folder, expected_count in expected_counts.items():
        paths = sorted((run_dir / folder).iterdir())
        paths = [path for path in paths if path.is_file()]
        if len(paths) != expected_count:
            raise ValueError(
                f"PP16 compile {folder} expected {expected_count} files, "
                f"found {len(paths)}"
            )
        for path in paths:
            local[f"{folder}/{path.name}"] = path
    hlo_paths = sorted(
        path for path in (run_dir / "hlo").iterdir() if path.is_file()
    )
    observed_hlo = {path.name for path in hlo_paths}
    if observed_hlo != PP16_COMPILE_HLO_FILES:
        raise ValueError(
            "PP16 compile hlo set drifted: "
            f"missing={sorted(PP16_COMPILE_HLO_FILES - observed_hlo)} "
            f"extra={sorted(observed_hlo - PP16_COMPILE_HLO_FILES)}"
        )
    for path in hlo_paths:
        local[f"hlo/{path.name}"] = path
    for name in TOP_LEVEL_EVIDENCE:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"PP16 compile evidence is missing {name}")
        local[name] = path
    return local


def seal(
    run_dir: Path,
    remote_prefix: str,
    code_hash: str,
    run_tag: str,
) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text())
    if (
        summary.get("status") != "SUCCESS"
        or summary.get("code_hash") != code_hash
        or summary.get("plan_id") != "PP16_LP2"
        or summary.get("performance_claim") is not False
        or summary.get("numerical_claim") is not False
        or summary.get("gate_d_passed") is not False
        or summary.get("results_db_run_id") is not None
    ):
        raise ValueError("PP16 compile summary identity failed")
    local = _local_evidence(run_dir)
    bucket_name, prefix = _parse_remote(remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = {
        blob.name.removeprefix(prefix): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    if set(blobs) != set(local):
        raise ValueError(
            "PP16 compile preterminal remote set drifted: "
            f"missing={sorted(set(local) - set(blobs))} "
            f"extra={sorted(set(blobs) - set(local))}"
        )
    records = []
    for name, path in sorted(local.items()):
        crc = _crc(path)
        blob = blobs[name]
        if (
            int(blob.size) != path.stat().st_size
            or blob.crc32c != crc
            or not blob.generation
        ):
            raise ValueError(f"PP16 compile remote identity drifted: {name}")
        records.append(
            {
                "crc32c": crc,
                "generation": int(blob.generation),
                "name": name,
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
        ledger_path,
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    ledger_blob.reload()
    ledger_crc = _crc(ledger_path)
    if (
        int(ledger_blob.size) != ledger_path.stat().st_size
        or ledger_blob.crc32c != ledger_crc
        or not ledger_blob.generation
    ):
        raise ValueError("PP16 compile remote ledger identity failed")
    success: dict[str, Any] = {
        "artifact_kind": "greenfield_pp16_short_decoder_compile_SUCCESS",
        "code_hash": code_hash,
        "gate_d_passed": False,
        "numerical_claim": False,
        "optimized_hlo_sha256": summary["optimized_hlo_sha256"],
        "performance_claim": False,
        "remote_ledger_crc32c": ledger_crc,
        "remote_ledger_generation": int(ledger_blob.generation),
        "remote_ledger_sha256": ledger["ledger_sha256"],
        "run_tag": run_tag,
        "runtime_manifest_sha256": summary["runtime_manifest_sha256"],
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    success_path = run_dir / "SUCCESS"
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
    success_blob = bucket.blob(prefix + "SUCCESS")
    success_blob.upload_from_filename(
        success_path,
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    success_blob.reload()
    success_crc = _crc(success_path)
    if (
        int(success_blob.size) != success_path.stat().st_size
        or success_blob.crc32c != success_crc
        or not success_blob.generation
    ):
        raise ValueError("PP16 compile terminal SUCCESS identity failed")

    terminal = {
        blob.name.removeprefix(prefix): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    expected = set(local) | {"remote_objects.json", "SUCCESS"}
    if set(terminal) != expected:
        raise ValueError("PP16 compile terminal remote set drifted")
    for item in ledger["objects"]:
        blob = terminal[item["name"]]
        if (
            int(blob.size) != item["size"]
            or blob.crc32c != item["crc32c"]
            or int(blob.generation) != item["generation"]
        ):
            raise ValueError(
                f"PP16 compile terminal payload drifted: {item['name']}"
            )
    return {
        "object_count": len(expected),
        "ledger_sha256": ledger["ledger_sha256"],
        "success_sha256": success["success_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = seal(
        args.run_dir,
        args.remote_prefix,
        args.code_hash,
        args.run_tag,
    )
    print(
        f"PP16_COMPILE_ACQUISITION_SEALED objects={result['object_count']} "
        f"ledger={result['ledger_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
