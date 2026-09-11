#!/usr/bin/env python3
"""Two exact archived DB snapshots; reuse the unchanged default-dry-run engine."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield import evict_reviewed_local_copies as engine
from scripts.greenfield.evict_history_local_copies import _digest_fields

ROOT = Path("/home/gianl/glm-run")
BUCKET = "driftbench-dsv4-uc"
TARGETS = {
    "greenfield_fp8_ws32_prefill_completed_window_acquisition_l6_20260908T164531308007030Z": 82_169_856,
    "greenfield_fp8_ws32_prefill_window_boundary_diagnostic_l6_20260908T155153878825613Z": 79_384_576,
}
TOTAL_BYTES = 161_554_432


def _validate_manifest(manifest: dict) -> list[dict]:
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind") != "ws32_history_two_archived_db_headroom_review_v1"
            or manifest.get("status") != "VERIFIED_NOT_DELETED" or manifest.get("problems") != []
            or manifest.get("bucket") != BUCKET or manifest.get("bucket_location") != "US-CENTRAL2"
            or any(type(manifest.get(key)) is not int or manifest[key] != value for key, value in
                   (("local_files_deleted", 0), ("cloud_objects_deleted", 0), ("file_count", 2), ("total_bytes", TOTAL_BYTES)))):
        raise ValueError("two-copy headroom requires the exact verify-only review")
    entries, ledgers = manifest.get("files"), manifest.get("ledger_sources")
    expected_paths = {str(ROOT / tag / "results_ckpt.db") for tag in TARGETS}
    expected_ledgers = {str(ROOT / tag / "archive_receipts.json") for tag in TARGETS}
    if (not isinstance(entries, list) or len(entries) != 2 or any(not isinstance(row, dict) for row in entries)
            or {row.get("path") for row in entries} != expected_paths
            or not isinstance(ledgers, list) or len(ledgers) != 2 or any(not isinstance(row, dict) for row in ledgers)
            or {row.get("path") for row in ledgers} != expected_ledgers):
        raise ValueError("headroom may select only the two archived per-run DB snapshots")
    by_path = {row["path"]: row for row in ledgers}
    for row in entries:
        _digest_fields(row)
        path = Path(row["path"])
        tag = path.parent.name
        name = f"results/{tag}/results_ckpt.db"
        ledger_path = str(ROOT / tag / "archive_receipts.json")
        ledger = by_path[ledger_path]
        _digest_fields(ledger)
        if (row.get("tag") != tag or row["size"] != TARGETS[tag] or row.get("name") != name
                or row.get("restore_uri") != f"gs://{BUCKET}/{name}#{row['generation']}"
                or row.get("ledger_path") != ledger_path or row.get("ledger_source_generation") != ledger["generation"]
                or ledger.get("name") != f"results/{tag}/archive_receipts.json" or ledger["size"] > 2 << 20
                or ledger.get("remote_exact_generation_download_byte_equal") is not True):
            raise ValueError("headroom original ledger/cloud restore identity differs")
        if (any(type(row.get(key)) is not int or row[key] <= 0 for key in
                ("device", "inode", "mtime_ns", "ctime_ns", "nlink", "allocated_bytes"))
                or row["nlink"] != 1 or row.get("is_symlink") is not False
                or any(row.get(key) is not True for key in
                    ("no_symlink_ancestors", "no_open_holders", "local_matches_receipt", "cloud_generation_verified"))
                or type(row.get("receipt_size")) is not int
                or (row["size"], row["sha256"], row["crc32c"])
                != (row["receipt_size"], row.get("receipt_sha256"), row.get("receipt_crc32c"))):
            raise ValueError("headroom reviewed local stat/byte identity differs")
    if (sum(row["size"] for row in entries) != TOTAL_BYTES
            or type(manifest.get("total_allocated_bytes")) is not int
            or sum(row["allocated_bytes"] for row in entries) != manifest["total_allocated_bytes"]):
        raise ValueError("headroom reviewed byte total differs")
    return entries


def main(argv: list[str] | None = None) -> int:
    return engine.main(argv, validate_manifest=_validate_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
