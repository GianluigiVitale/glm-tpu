#!/usr/bin/env python3
"""Generation/CRC seal a successful protected PP16 full-load proof."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import google_crc32c
from google.cloud import storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    return parser.parse_args()


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
        raise ValueError("PP16 load proof must use the approved results bucket")
    bucket, prefix = remote[5:].split("/", 1)
    return bucket, prefix.rstrip("/") + "/"


def main() -> int:
    args = parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    if (
        summary.get("status") != "SUCCESS"
        or summary.get("code_hash") != args.code_hash
        or summary.get("plan_id") != "PP16_LP2"
        or summary.get("performance_claim")
    ):
        raise SystemExit("PP16 load summary identity failed")
    local: dict[str, Path] = {}
    for folder in ("probe", "host_records"):
        for path in sorted((args.run_dir / folder).iterdir()):
            if path.is_file():
                local[f"{folder}/{path.name}"] = path
    for name in (
        "summary.json",
        "results_ckpt.db",
        "sync.txt",
        "probe_dispatch.txt",
        "load_dispatch.txt",
        "census_pre.txt",
        "census_post_probe.txt",
        "census_post.txt",
        "orchestrator.sealed.log",
        "evidence.sha256",
    ):
        local[name] = args.run_dir / name
    bucket_name, prefix = _parse_remote(args.remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = {
        blob.name.removeprefix(prefix): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    if set(blobs) != set(local):
        raise SystemExit(
            "PP16 load preterminal remote set drifted: "
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
            raise SystemExit(f"PP16 load remote identity drifted: {name}")
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
        "remote_prefix": args.remote_prefix,
    }
    ledger["ledger_sha256"] = sha256(_canonical(ledger)).hexdigest()
    ledger_path = args.run_dir / "remote_objects.json"
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
        raise SystemExit("PP16 load remote ledger identity failed")
    success: dict[str, Any] = {
        "artifact_kind": "greenfield_full_checkpoint_load_SUCCESS",
        "code_hash": args.code_hash,
        "packed_manifest_sha256": summary["packed_manifest_sha256"],
        "performance_claim": False,
        "remote_ledger_crc32c": ledger_crc,
        "remote_ledger_generation": int(ledger_blob.generation),
        "remote_ledger_sha256": ledger["ledger_sha256"],
        "results_db_run_id": summary["results_db_run_id"],
        "run_tag": args.run_tag,
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    success_path = args.run_dir / "SUCCESS"
    success_path.write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n"
    )
    # Terminal marker is intentionally the final remote write.
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
        raise SystemExit("PP16 load terminal SUCCESS identity failed")

    terminal = {
        blob.name.removeprefix(prefix): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    expected = set(local) | {"remote_objects.json", "SUCCESS"}
    if set(terminal) != expected:
        raise SystemExit("PP16 load terminal remote set drifted")
    if (
        success["remote_ledger_sha256"] != ledger["ledger_sha256"]
        or success["remote_ledger_crc32c"] != ledger_crc
        or success["remote_ledger_generation"] != int(
            terminal["remote_objects.json"].generation
        )
    ):
        raise SystemExit("PP16 load SUCCESS-to-ledger binding failed")
    for item in ledger["objects"]:
        blob = terminal[item["name"]]
        if (
            int(blob.size) != item["size"]
            or blob.crc32c != item["crc32c"]
            or int(blob.generation) != item["generation"]
        ):
            raise SystemExit(
                f"PP16 load terminal payload drifted: {item['name']}"
            )
    print(
        f"PP16_FULL_LOAD_SEALED objects={len(expected)} "
        f"ledger={ledger['ledger_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
