"""Manifest-only mutations and engine callback wiring; no payload/cloud/unlink."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.greenfield import evict_history_local_copies as adapter

MANIFEST = adapter.REPO / "docs/artifacts/history-local-copy-eviction-plan-20260911.json"


@pytest.fixture
def manifest():
    return json.loads(MANIFEST.read_bytes())


def test_exact_111_copy_scope_and_honest_nonlaunch_status(manifest):
    rows = adapter._validate_manifest(manifest)
    assert len(rows) == 111
    assert sum(row["size"] for row in rows) == 2_624_409_673
    databases = [row for row in rows if row["path"].endswith("/results_ckpt.db")]
    assert len(databases) == 13
    assert {row["tag"] for row in databases} == set(adapter.TAGS)
    hlo = [row for row in rows if row not in databases]
    assert len(hlo) == 98
    assert {row["rank"] for row in hlo} == set(range(1, 8))
    assert all(Path(row["path"]).name in adapter.FILENAMES for row in hlo)
    assert not any("/rank0/" in row["path"] for row in rows)
    assert manifest["primary_database_preserved"]["path"] not in {row["path"] for row in rows}
    assert len(manifest["preserved_rank0_originals"]) == 28
    # Cleanup can be reviewed while launch remains unavailable. Never rewrite
    # the observed free-space evidence to pretend the requested floor was met.
    assert manifest["launch_headroom_sufficient"] is False
    assert manifest["projected_launch_headroom_bytes"] < 0
    assert manifest["launch_blockers"][0]["kind"] == "PROJECTED_FREE_BELOW_LAUNCH_FLOOR"


def test_retained_manifest_has_exact_filesystem_nanosecond_identity(manifest):
    checked = 0
    for row in manifest["files"]:
        path = Path(row["path"])
        try:
            observed = path.lstat()
        except FileNotFoundError:
            continue  # A later separately reviewed cleanup may remove a copy.
        adapter.engine._require_local(observed, row)
        assert (observed.st_dev, observed.st_ino, observed.st_size, observed.st_nlink,
                observed.st_mtime_ns, observed.st_ctime_ns) == (
                    row["device"], row["inode"], row["size"], row["nlink"],
                    row["mtime_ns"], row["ctime_ns"]), row["path"]
        checked += 1
    if not checked:
        pytest.skip("all separately reviewed local copies have been removed")


@pytest.mark.parametrize("mutation", [
    "status", "kind", "problems", "excluded", "bucket", "region", "deleted", "bool_deleted",
    "count", "total", "allocated_total", "duplicate", "missing", "outside", "primary",
    "rank0", "compact_metadata", "new_tag", "name", "category", "restore", "generation",
    "sha", "crc", "inode", "bool_size", "mtime", "hardlink", "symlink", "ancestor",
    "holder", "local_match", "cloud_match", "receipt_size", "receipt_sha", "receipt_crc",
    "ledger_generation", "ledger_id", "rank_bool", "db", "database_rank",
])
def test_manifest_mutations_refuse(manifest, mutation):
    row = manifest["files"][13]  # First DB611 nonzero-rank HLO copy.
    if mutation == "status": manifest["status"] = "DELETED"
    elif mutation == "kind": manifest["artifact_kind"] = "old"
    elif mutation == "problems": manifest["problems"] = ["unverified"]
    elif mutation == "excluded": manifest["excluded_candidates"] = [{"path": "ambiguous"}]
    elif mutation == "bucket": manifest["bucket"] = "other"
    elif mutation == "region": manifest["bucket_location"] = "US-CENTRAL1"
    elif mutation == "deleted": manifest["cloud_objects_deleted"] = 1
    elif mutation == "bool_deleted": manifest["local_files_deleted"] = False
    elif mutation == "count": manifest["file_count"] = 110
    elif mutation == "total": manifest["total_bytes"] -= 1
    elif mutation == "allocated_total": manifest["total_allocated_bytes"] -= 1
    elif mutation == "duplicate": manifest["files"][14] = deepcopy(row)
    elif mutation == "missing": manifest["files"].pop()
    elif mutation == "outside": row["path"] = "/tmp/results_ckpt.db"
    elif mutation == "primary": manifest["files"][0]["path"] = "/home/gianl/glm-tpu/bench/results.db"
    elif mutation == "rank0": row["path"] = row["path"].replace("rank1", "rank0")
    elif mutation == "compact_metadata": row["path"] = str(Path(row["path"]).with_name("runner.json"))
    elif mutation == "new_tag": row["tag"] = adapter.TAGS[1]
    elif mutation == "name": row["name"] = row["name"].replace("workers/rank1", "fleet/rank1")
    elif mutation == "category": row["category"] = "ARCHIVED_PER_RUN_DATABASE_SNAPSHOT"
    elif mutation == "restore": row["restore_uri"] = row["restore_uri"].replace(adapter.BUCKET, "other")
    elif mutation == "generation": row["generation"] = "0"
    elif mutation == "sha": row["sha256"] = "x" * 64
    elif mutation == "crc": row["crc32c"] = "AA=="
    elif mutation == "inode": row["inode"] = 0
    elif mutation == "bool_size": row["size"] = True
    elif mutation == "mtime": row["mtime_ns"] = -1
    elif mutation == "hardlink": row["nlink"] = 2
    elif mutation == "symlink": row["is_symlink"] = True
    elif mutation == "ancestor": row["no_symlink_ancestors"] = 1
    elif mutation == "holder": row["no_open_holders"] = False
    elif mutation == "local_match": row["local_matches_receipt"] = False
    elif mutation == "cloud_match": row["cloud_generation_verified"] = 1
    elif mutation == "receipt_size": row["receipt_size"] += 1
    elif mutation == "receipt_sha": row["receipt_sha256"] = "0" * 64
    elif mutation == "receipt_crc": row["receipt_crc32c"] = "AAAAAA=="
    elif mutation == "ledger_generation": row["ledger_source_generation"] = "1"
    elif mutation == "ledger_id": row["ledger_id"] = manifest["ledger_sources"][0]["id"]
    elif mutation == "rank_bool": row["rank"] = True
    elif mutation == "db": row["db"] = 609
    else: manifest["files"][0]["rank"] = 0
    with pytest.raises(ValueError): adapter._validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "path", "name", "size", "sha", "crc", "generation", "download"])
def test_original_ledger_binding_mutations_refuse(manifest, mutation):
    row = manifest["ledger_sources"][0]
    if mutation == "missing": manifest["ledger_sources"].pop()
    elif mutation == "duplicate": manifest["ledger_sources"][1] = deepcopy(row)
    elif mutation == "path": row["path"] = "/tmp/ledger.json"
    elif mutation == "name": row["name"] = "results/other/ledger.json"
    elif mutation == "size": row["size"] = 3 << 20
    elif mutation == "sha": row["sha256"] = "invalid"
    elif mutation == "crc": row["crc32c"] = "AAAA"
    elif mutation == "generation": row["generation"] = "-1"
    else: row["remote_exact_generation_download_byte_equal"] = 1
    with pytest.raises(ValueError): adapter._validate_manifest(manifest)


def test_adapter_passes_validator_without_mutating_engine_default(monkeypatch, manifest):
    original_validator = adapter.engine._validate_manifest
    seen = []

    def main(argv, *, validate_manifest=None):
        seen.append((argv, validate_manifest))
        assert validate_manifest(manifest) is manifest["files"]
        return 17

    monkeypatch.setattr(adapter.engine, "main", main)
    argv = ["--manifest", str(MANIFEST)]
    assert adapter.main(argv) == 17
    assert seen == [(argv, adapter._validate_manifest)]
    assert adapter.engine._validate_manifest is original_validator


def test_actual_cli_help_defaults_apply_off():
    result = subprocess.run([sys.executable, "-m", "scripts.greenfield.evict_history_local_copies", "--help"],
                            cwd=adapter.REPO, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0
    assert "default is a dry run" in result.stdout
    assert "--manifest-sha256" in result.stdout and "--apply" in result.stdout
