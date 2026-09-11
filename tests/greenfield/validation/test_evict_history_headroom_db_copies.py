"""Exact two-copy manifest; local stat and mocked delegation only."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.greenfield import evict_history_headroom_db_copies as adapter

MANIFEST = adapter.REPO / "docs/artifacts/history-local-db-headroom-eviction-plan-20260911.json"


@pytest.fixture
def manifest():
    return json.loads(MANIFEST.read_bytes())


def test_exact_pair_and_unrounded_local_stat(manifest):
    rows = adapter._validate_manifest(manifest)
    assert len(rows) == 2 and sum(row["size"] for row in rows) == 161_554_432
    assert {row["tag"] for row in rows} == set(adapter.TARGETS)
    assert not {row["path"] for row in rows} & {manifest["primary_database_preserved"]["path"]}
    for row in rows:
        try:
            st = Path(row["path"]).lstat()
        except FileNotFoundError:
            continue  # A later separately reviewed application may remove it.
        adapter.engine._require_local(st, row)
        assert (st.st_dev, st.st_ino, st.st_size, st.st_nlink, st.st_mtime_ns, st.st_ctime_ns) == (
            row["device"], row["inode"], row["size"], row["nlink"], row["mtime_ns"], row["ctime_ns"])


@pytest.mark.parametrize("mutation", [
    "status", "kind", "problems", "bucket", "region", "deleted", "bool_zero", "count", "total",
    "allocated", "duplicate", "primary", "rank0", "other_run", "other_file", "generation",
    "size", "sha", "crc", "inode", "mtime", "hardlink", "symlink", "ancestor", "holder",
    "local_bytes", "cloud", "receipt", "restore", "ledger_generation", "ledger_path",
    "ledger_object", "ledger_count", "ledger_sha", "ledger_download",
])
def test_mutated_scope_and_identity_refuse(manifest, mutation):
    row, ledger = manifest["files"][0], manifest["ledger_sources"][0]
    if mutation == "status": manifest["status"] = "DELETED"
    elif mutation == "kind": manifest["artifact_kind"] = "other"
    elif mutation == "problems": manifest["problems"] = ["unverified"]
    elif mutation == "bucket": manifest["bucket"] = "other"
    elif mutation == "region": manifest["bucket_location"] = "US-CENTRAL1"
    elif mutation == "deleted": manifest["local_files_deleted"] = 1
    elif mutation == "bool_zero": manifest["cloud_objects_deleted"] = False
    elif mutation == "count": manifest["file_count"] = 3
    elif mutation == "total": manifest["total_bytes"] += 1
    elif mutation == "allocated": manifest["total_allocated_bytes"] += 1
    elif mutation == "duplicate": manifest["files"][1] = deepcopy(row)
    elif mutation == "primary": row["path"] = manifest["primary_database_preserved"]["path"]
    elif mutation == "rank0": row["path"] = str(Path(row["path"]).parent / "rank0" / "results_ckpt.db")
    elif mutation == "other_run": row["path"] = row["path"].replace("20260908", "20260911")
    elif mutation == "other_file": row["path"] = str(Path(row["path"]).with_name("runner.json"))
    elif mutation == "generation": row["generation"] = "0"
    elif mutation == "size": row["size"] = True
    elif mutation == "sha": row["sha256"] = "z" * 64
    elif mutation == "crc": row["crc32c"] = "AA=="
    elif mutation == "inode": row["inode"] = 0
    elif mutation == "mtime": row["mtime_ns"] = False
    elif mutation == "hardlink": row["nlink"] = 2
    elif mutation == "symlink": row["is_symlink"] = True
    elif mutation == "ancestor": row["no_symlink_ancestors"] = False
    elif mutation == "holder": row["no_open_holders"] = False
    elif mutation == "local_bytes": row["local_matches_receipt"] = False
    elif mutation == "cloud": row["cloud_generation_verified"] = 1
    elif mutation == "receipt": row["receipt_sha256"] = "0" * 64
    elif mutation == "restore": row["restore_uri"] = row["restore_uri"].replace(adapter.BUCKET, "other")
    elif mutation == "ledger_generation": row["ledger_source_generation"] = "1"
    elif mutation == "ledger_path": row["ledger_path"] = manifest["ledger_sources"][1]["path"]
    elif mutation == "ledger_object": ledger["name"] = "results/other/archive_receipts.json"
    elif mutation == "ledger_count": manifest["ledger_sources"].pop()
    elif mutation == "ledger_sha": ledger["sha256"] = "bad"
    else: ledger["remote_exact_generation_download_byte_equal"] = False
    with pytest.raises(ValueError): adapter._validate_manifest(manifest)


def test_reuses_engine_callback_without_changing_old_default(monkeypatch, manifest):
    original = adapter.engine._validate_manifest
    calls = []

    def main(argv, *, validate_manifest=None):
        assert validate_manifest(manifest) is manifest["files"]
        calls.append((argv, validate_manifest))
        return 7

    monkeypatch.setattr(adapter.engine, "main", main)
    assert adapter.main(["--manifest", str(MANIFEST)]) == 7
    assert calls == [(["--manifest", str(MANIFEST)], adapter._validate_manifest)]
    assert adapter.engine._validate_manifest is original
