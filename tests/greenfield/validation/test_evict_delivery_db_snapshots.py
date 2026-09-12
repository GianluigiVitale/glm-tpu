"""Fixed archived DB-copy scope; actual deletion engine tested separately."""
from copy import deepcopy
import pytest
from scripts.greenfield import evict_delivery_db_snapshots as adapter


@pytest.fixture
def manifest():
    files = []
    for tag, size in zip(adapter.TAGS, adapter.SIZES, strict=True):
        name = f"results/{tag}/results_ckpt.db"
        files.append(dict(path=str(adapter.engine.LOCAL_ROOT / tag / "results_ckpt.db"),
            name=name, size=size, inode=1, device=1, mtime_ns=1789000000000000001,
            ctime_ns=1789000000000000001, nlink=1, generation="123", sha256="a"*64,
            crc32c="AAAAAA==", restore_uri=f"gs://{adapter.engine.BUCKET}/{name}#123"))
    return dict(artifact_kind=adapter.KIND, status="VERIFIED_NOT_DELETED",
        bucket_location="US-CENTRAL2", cloud_objects_deleted=0,
        total_bytes=sum(adapter.SIZES), files=files)


def test_exact_scope(manifest):
    assert len(adapter.validate_manifest(manifest)) == 11
    assert sum(adapter.SIZES) == 402063360
    assert all(e["path"].endswith("/results_ckpt.db") and "/bench/" not in e["path"]
               for e in manifest["files"])


@pytest.mark.parametrize("kind", ["primary", "duplicate", "cloud", "restore", "size",
    "time", "hardlink", "generation", "sha", "crc", "region", "deleted", "bool", "total"])
def test_refuses_mutations(manifest, kind):
    e = manifest["files"][0]
    if kind == "primary": e["path"] = "/home/gianl/glm-tpu/bench/results.db"
    elif kind == "duplicate": manifest["files"][1] = deepcopy(e)
    elif kind == "cloud": e["name"] = "models/checkpoint"
    elif kind == "restore": e["restore_uri"] += "0"
    elif kind == "size": e["size"] += 1
    elif kind == "time": e["mtime_ns"] = float(e["mtime_ns"])
    elif kind == "hardlink": e["nlink"] = 2
    elif kind == "generation": e["generation"] = "0"
    elif kind == "sha": e["sha256"] = "z"*64
    elif kind == "crc": e["crc32c"] = "AA=="
    elif kind == "region": manifest["bucket_location"] = "EU"
    elif kind == "deleted": manifest["cloud_objects_deleted"] = 1
    elif kind == "bool": manifest["cloud_objects_deleted"] = False
    else: manifest["total_bytes"] += 1
    with pytest.raises(ValueError):
        adapter.validate_manifest(manifest)


def test_only_delegates_to_original_engine(monkeypatch, manifest):
    def run(argv, *, validate_manifest):
        assert validate_manifest(manifest) is manifest["files"]
        assert argv == ["--apply"]
        return 7
    monkeypatch.setattr(adapter.engine, "main", run)
    assert adapter.main(["--apply"]) == 7


def test_review_refuses_overwrite(tmp_path):
    path = tmp_path / "exists.json"
    path.touch()
    with pytest.raises(FileExistsError):
        adapter.review(path)
