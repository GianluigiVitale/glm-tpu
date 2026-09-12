"""Fixed local-only scope; reuse shared deletion-engine failure tests."""
import pytest

from scripts.greenfield import evict_db614_collected_copies as adapter
from tests.greenfield.validation import test_evict_delivery_db_snapshots as checks


@pytest.fixture
def manifest():
    files = []
    for path, (name, size, ledger_path, ledger_name) in adapter.scope().items():
        files.append(dict(path=path, name=name, size=size, inode=1, device=1,
            mtime_ns=1789000000000000001, ctime_ns=1789000000000000001, nlink=1,
            generation="123", sha256="a" * 64, crc32c="AAAAAA==",
            ledger_path=str(ledger_path), ledger_name=ledger_name,
            restore_uri=f"gs://{adapter.engine.BUCKET}/{name}#123"))
    return dict(artifact_kind=adapter.KIND, status="VERIFIED_NOT_DELETED",
        bucket_location="US-CENTRAL2", cloud_objects_deleted=0,
        total_bytes=adapter.TOTAL_BYTES, files=files)


def test_scope_retains_rank0_and_primary(manifest):
    assert len(adapter.validate_manifest(manifest)) == 21
    assert adapter.TOTAL_BYTES == 1514991152
    assert all("/fleet/rank0/" not in e["path"] and "/bench/" not in e["path"] for e in manifest["files"])


@pytest.mark.parametrize("kind", ["primary", "duplicate", "cloud", "restore", "size",
    "time", "hardlink", "generation", "sha", "crc", "region", "deleted", "bool", "total"])
def test_mutations(manifest, monkeypatch, kind):
    monkeypatch.setattr(checks, "adapter", adapter)
    checks.test_refuses_mutations(manifest, kind)


def test_delegates_engine(manifest, monkeypatch):
    monkeypatch.setattr(checks, "adapter", adapter)
    checks.test_only_delegates_to_original_engine(monkeypatch, manifest)


def test_no_overwrite(tmp_path):
    path = tmp_path / "existing.json"
    path.touch()
    with pytest.raises(FileExistsError):
        adapter.review(path)
