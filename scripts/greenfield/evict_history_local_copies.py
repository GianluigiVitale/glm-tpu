#!/usr/bin/env python3
"""Exact reviewed 111 local copies only; the existing leased engine defaults dry-run.

This adapter changes no unlink, cloud verification, lease or receipt mechanism.
Its storage cleanup is not numerical launch admission: remeasure launch space
after any separately authorized application. All cloud originals remain.
"""

from __future__ import annotations

import base64
from pathlib import Path
import re
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield import evict_reviewed_local_copies as engine

LOCAL_ROOT = Path("/home/gianl/glm-run")
BUCKET = "driftbench-dsv4-uc"
TAGS = tuple("greenfield_fp8_ws32_" + suffix for suffix in (
    "history_frontier_compile_20260911T183647058550669Z",
    "prefill_canonical_model_compile_20260909T202846089848089Z",
    "dense_canonical_d01_20260909T194344477028376Z",
    "dense_canonical_compile_20260909T185544706417041Z",
    "dense_norm_d01_20260909T175425536328904Z",
    "dense_frontier_d01_20260909T153537693051589Z",
    "prefill_rolled_model_compile_20260909T054150449575877Z",
    "prefill_rolled_layer_l6_20260909T044037994639176Z",
    "prefill_expert_panel_phase_l6_20260909T033031628458338Z",
    "prefill_sorted_merge_20260909T030011185253029Z",
    "prefill_budget_baseline_20260909T021350314569852Z",
    "prefill_paired_sort_phase_l6_20260908T205932416058403Z",
    "prefill_completed_phase_baseline_l6_20260908T200408874805057Z",
))
PROGRAMS = ("candidate_b128", "candidate_b114", "control_b128", "control_b114",
            "exact_decode", "exact_promote", "observer")
FILENAMES = tuple(f"{program}.{form}" for program in PROGRAMS
                  for form in ("stablehlo.mlir", "optimized_hlo.txt"))
TOTAL_BYTES = 2_624_409_673
FILE_COUNT = 111
_SHA = re.compile(r"[0-9a-f]{64}")
_GENERATION = re.compile(r"[1-9][0-9]*")


def _digest_fields(row: dict[str, Any], *, size: str = "size", digest: str = "sha256") -> None:
    if (type(row.get(size)) is not int or row[size] <= 0
            or not isinstance(row.get(digest), str) or not _SHA.fullmatch(row[digest])
            or not isinstance(row.get("generation"), str) or not _GENERATION.fullmatch(row["generation"])
            or not isinstance(row.get("crc32c"), str)
            or len(base64.b64decode(row["crc32c"], validate=True)) != 4):
        raise ValueError("history eviction reviewed byte/generation identity differs")


def _scope() -> dict[str, tuple[str, str, str, int | None]]:
    result = {str(LOCAL_ROOT / tag / "results_ckpt.db"):
        (tag, f"results/{tag}/results_ckpt.db", "ARCHIVED_PER_RUN_DATABASE_SNAPSHOT", None) for tag in TAGS}
    for rank in range(1, 8):
        for name in FILENAMES:
            result[str(LOCAL_ROOT / TAGS[0] / "fleet" / f"rank{rank}" / name)] = (
                TAGS[0], f"results/{TAGS[0]}/workers/rank{rank}/{name}",
                "DB611_COLLECTED_NONZERO_RANK_COMPILER_COPY", rank)
    return result


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind") != "ws32_history_local_copy_eviction_review_v1"
            or manifest.get("status") != "VERIFIED_NOT_DELETED" or manifest.get("problems") != []
            or manifest.get("excluded_candidates") != [] or manifest.get("bucket") != BUCKET
            or manifest.get("bucket_location") != "US-CENTRAL2"
            or any(type(manifest.get(key)) is not int or manifest[key] != 0
                   for key in ("cloud_objects_deleted", "local_files_deleted"))
            or type(manifest.get("file_count")) is not int or manifest["file_count"] != FILE_COUNT
            or type(manifest.get("total_bytes")) is not int or manifest["total_bytes"] != TOTAL_BYTES):
        raise ValueError("history eviction requires the clean verify-only review")
    entries, ledgers = manifest.get("files"), manifest.get("ledger_sources")
    scope = _scope()
    if (not isinstance(entries, list) or len(entries) != FILE_COUNT
            or any(not isinstance(row, dict) for row in entries)
            or {row.get("path") for row in entries} != set(scope)):
        raise ValueError("history eviction must name exactly the 111 reviewed local paths")
    expected_ledgers = {str(LOCAL_ROOT / tag / "archive_receipts.json"):
                        f"results/{tag}/archive_receipts.json" for tag in TAGS}
    expected_ledgers.update({str(LOCAL_ROOT / TAGS[0] / "fleet" / f"rank{rank}" / "worker_receipts.json"):
        f"results/{TAGS[0]}/workers/rank{rank}/worker_receipts.json" for rank in range(1, 8)})
    if (not isinstance(ledgers, list) or len(ledgers) != 20
            or any(not isinstance(row, dict) for row in ledgers)
            or {row.get("path") for row in ledgers} != set(expected_ledgers)
            or any(not isinstance(row.get("id"), str) for row in ledgers)
            or len({row["id"] for row in ledgers}) != len(ledgers)):
        raise ValueError("history eviction archive ledger inventory differs")
    by_id = {row["id"]: row for row in ledgers}
    for row in ledgers:
        _digest_fields(row)
        if (row["name"] != expected_ledgers[row["path"]]
                or row["remote_exact_generation_download_byte_equal"] is not True
                or row["size"] > 2 << 20):
            raise ValueError("history eviction original ledger binding differs")
    for row in entries:
        tag, name, category, rank = scope[row["path"]]
        _digest_fields(row)
        path = Path(row["path"])
        if (str(path) != row["path"] or row.get("tag") != tag or row.get("name") != name
                or row.get("category") != category
                or row.get("restore_uri") != f"gs://{BUCKET}/{name}#{row['generation']}"):
            raise ValueError("history eviction local/cloud restore scope differs")
        if rank is not None and (type(row.get("rank")) is not int or row["rank"] != rank
                                 or type(row.get("db")) is not int or row["db"] != 611):
            raise ValueError("history eviction nonzero rank/DB611 binding differs")
        if rank is None and "rank" in row:
            raise ValueError("history eviction database snapshot is not a rank artifact")
        for key in ("device", "inode", "mtime_ns", "ctime_ns", "nlink", "allocated_bytes"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError("history eviction local stat identity differs")
        if (row["nlink"] != 1 or row.get("is_symlink") is not False
                or any(row.get(key) is not True for key in ("no_symlink_ancestors", "no_open_holders",
                    "local_matches_receipt", "cloud_generation_verified"))
                or type(row.get("receipt_size")) is not int
                or (row["size"], row["sha256"], row["crc32c"])
                != (row["receipt_size"], row.get("receipt_sha256"), row.get("receipt_crc32c"))):
            raise ValueError("history eviction reviewed local/receipt identity differs")
        expected_path = (LOCAL_ROOT / tag / "archive_receipts.json" if rank is None else
                         LOCAL_ROOT / tag / "fleet" / f"rank{rank}" / "worker_receipts.json")
        ledger = by_id.get(row.get("ledger_id"), {})
        if ledger.get("path") != str(expected_path) or row.get("ledger_source_generation") != ledger.get("generation"):
            raise ValueError("history eviction file is bound to a different source ledger")
    if (sum(row["size"] for row in entries) != TOTAL_BYTES
            or type(manifest.get("total_allocated_bytes")) is not int
            or sum(row["allocated_bytes"] for row in entries) != manifest["total_allocated_bytes"]):
        raise ValueError("history eviction reviewed total bytes differ")
    return entries


def main(argv: list[str] | None = None) -> int:
    return engine.main(argv, validate_manifest=_validate_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
