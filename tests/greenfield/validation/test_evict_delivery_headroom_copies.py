"""Fixed-scope review manifest and delegation; no cloud or candidate deletion."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.greenfield import evict_delivery_headroom_copies as adapter

MANIFEST = adapter.engine.REPO / "docs/artifacts/delivery-local-headroom-review-20260911.json"


@pytest.fixture
def manifest():
    return json.loads(MANIFEST.read_bytes())


def test_exact_scope_and_integer_stat(manifest):
    entries = adapter.validate_manifest(manifest)
    assert len(entries) == 84 + 133 + 5
    assert sum(e["size"] for e in entries) == 4_455_299_027
    assert not any("/rank0/" in e["path"] or "/bench/" in e["path"] for e in entries)
    for e in entries:
        path = Path(e["path"])
        if path.exists():  # The separately authorized application may evict it later.
            st = path.lstat()
            adapter.engine._require_local(st, e)
            assert (st.st_dev, st.st_ctime_ns) == (e["device"], e["ctime_ns"])


@pytest.mark.parametrize("mutation", [
    "kind", "status", "region", "deleted", "bool_zero", "count", "total",
    "duplicate", "primary", "rank0", "cloud_path", "restore", "hardlink",
    "inode", "mtime", "ctime", "device", "size", "generation", "sha", "crc",
])
def test_refuses_mutated_scope_or_identity(manifest, mutation):
    e = manifest["files"][0]
    if mutation == "kind": manifest["artifact_kind"] = "other"
    elif mutation == "status": manifest["status"] = "DELETED"
    elif mutation == "region": manifest["bucket_location"] = "EU"
    elif mutation == "deleted": manifest["cloud_objects_deleted"] = 1
    elif mutation == "bool_zero": manifest["cloud_objects_deleted"] = False
    elif mutation == "count": manifest["file_count"] += 1
    elif mutation == "total": manifest["total_bytes"] += 1
    elif mutation == "duplicate": manifest["files"][1] = deepcopy(e)
    elif mutation == "primary": e["path"] = "/home/gianl/glm-tpu/bench/results.db"
    elif mutation == "rank0": manifest["files"][84]["path"] = manifest["files"][84]["path"].replace("/rank1/", "/rank0/")
    elif mutation == "cloud_path": e["name"] = "results/wrong/file.npz"
    elif mutation == "restore": e["restore_uri"] += "0"
    elif mutation == "hardlink": e["nlink"] = 2
    elif mutation == "inode": e["inode"] = 0
    elif mutation == "mtime": e["mtime_ns"] = float(e["mtime_ns"])
    elif mutation == "ctime": e["ctime_ns"] = False
    elif mutation == "device": e["device"] = -1
    elif mutation == "size": e["size"] = True
    elif mutation == "generation": e["generation"] = "0"
    elif mutation == "sha": e["sha256"] = "z" * 64
    else: e["crc32c"] = "AA=="
    with pytest.raises(ValueError):
        adapter.validate_manifest(manifest)


def test_only_delegates_apply_to_existing_engine(monkeypatch, manifest):
    original = adapter.engine._validate_manifest
    calls = []

    def fake_main(argv, *, validate_manifest):
        assert validate_manifest(manifest) is manifest["files"]
        calls.append(argv)
        return 7

    monkeypatch.setattr(adapter.engine, "main", fake_main)
    assert adapter.main(["--manifest", str(MANIFEST)]) == 7
    assert calls == [["--manifest", str(MANIFEST)]]
    assert adapter.engine._validate_manifest is original


def test_review_refuses_overwrite_before_cloud(tmp_path):
    existing = tmp_path / "manifest.json"
    existing.touch()
    with pytest.raises(FileExistsError):
        adapter.review(existing)
