"""Fixed scope and bounded gzip recovery, with no cloud writes."""
from copy import deepcopy
import gzip
from hashlib import sha256
from types import SimpleNamespace

import pytest

from scripts.greenfield import evict_delivery_256k_headroom as c


def manifest():
    return dict(artifact_kind=c.KIND, status="VERIFIED_NOT_DELETED", cloud_objects_deleted=0,
                bucket_location="US-CENTRAL2", total_bytes=sum(v[1] for v in c.scope().values()),
                files=[dict(path=p, name=n, size=s, restore_transform=t, nlink=1,
                            generation="123", compressed_size=100,
                            restore_uri=f"gs://{c.engine.BUCKET}/{n}#123")
                       for p, (n, s, t) in c.scope().items()])


def test_scope_retains_primary_db_weights_active_compiler_and_cloud():
    value = manifest()
    assert len(c.validate_manifest(value)) == 38
    assert value["total_bytes"] > 4_000_000_000
    assert all("/glm-run/" in r["path"] and "/checkpoints/" not in r["path"] for r in value["files"])
    assert not any("trace.rank0" in r["path"] for r in value["files"])
    assert sum(r["restore_transform"] == "gunzip" for r in value["files"]) == 30


@pytest.mark.parametrize("key,new", [
    ("path", "/home/gianl/glm-tpu/bench/results.db"), ("name", "models/weights"),
    ("size", 1), ("nlink", 2), ("restore_transform", "gunzip"), ("restore_uri", "gs://other/file"),
])
def test_modified_target_refuses(key, new):
    value = manifest()
    value["files"][0][key] = new
    with pytest.raises(ValueError):
        c.validate_manifest(value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "region", "total", "cap"])
def test_modified_scope_refuses(mutation):
    value = deepcopy(manifest())
    if mutation == "missing":
        value["files"].pop()
    elif mutation == "duplicate":
        value["files"].append(value["files"][0])
    elif mutation == "region":
        value["bucket_location"] = "EU"
    elif mutation == "total":
        value["total_bytes"] += 1
    else:
        value["files"][-1]["compressed_size"] = c.COMPRESSED_CAP + 1
    with pytest.raises(ValueError):
        c.validate_manifest(value)


@pytest.mark.parametrize("mutation", [None, "generation", "crc", "size", "bytes", "sha", "oversize", "short", "invalid_gzip"])
def test_gzip_exact_bounded_original(mutation):
    original = b"actual compiler original\n" * 100
    raw = gzip.compress(original, mtime=0)
    if mutation == "invalid_gzip":
        raw = b"not gzip"
    entry = dict(generation="123", compressed_size=len(raw), compressed_crc32c=c.crc(raw),
                 size=len(original), sha256=sha256(original).hexdigest())
    blob = SimpleNamespace(generation=123, size=len(raw), crc32c=c.crc(raw))
    def download(**kwargs):
        assert kwargs == dict(start=0, end=entry["compressed_size"] - 1, raw_download=True,
                             if_generation_match=123, checksum=None, timeout=60)
        return raw if mutation != "bytes" else raw[:-1]
    blob.download_as_bytes = download
    if mutation in ("generation", "size"):
        setattr(blob, mutation, getattr(blob, mutation) + 1)
    elif mutation == "crc":
        blob.crc32c = "AAAAAA=="
    elif mutation == "sha":
        entry["sha256"] = "0" * 64
    elif mutation == "oversize":
        entry["size"] -= 1
    elif mutation == "short":
        entry["size"] += 1
    if mutation is None:
        c.verify_gzip(entry, blob)
    else:
        with pytest.raises((ValueError, OSError, EOFError)):
            c.verify_gzip(entry, blob)


def test_gzip_selection_is_explicit(monkeypatch):
    calls = []
    monkeypatch.setattr(c.engine, "_verified_file", lambda e, b, **kw: calls.append(kw))
    c.verified_file({"restore_transform": "identity"}, None)
    c.verified_file({"restore_transform": "gunzip"}, None)
    assert calls == [dict(cloud_verifier=None), dict(cloud_verifier=c.verify_gzip)]


def test_gzip_runs_inside_original_identity_and_holder_checks(tmp_path, monkeypatch):
    original = b"local exact HLO"
    path = tmp_path / "graph.txt"
    path.write_bytes(original)
    st = path.stat()
    raw = gzip.compress(original, mtime=0)
    row = dict(path=str(path), inode=st.st_ino, mtime_ns=st.st_mtime_ns, nlink=1,
               size=len(original), sha256=sha256(original).hexdigest(), crc32c=c.crc(original),
               name="results/graph.gz", generation="123", restore_transform="gunzip",
               compressed_size=len(raw), compressed_crc32c=c.crc(raw))
    def reload(**kw):
        assert kw == {"if_generation_match": 123}
    blob = SimpleNamespace(generation=123, size=len(raw), crc32c=c.crc(raw),
                           reload=reload, download_as_bytes=lambda **kw: raw)
    bucket = SimpleNamespace(blob=lambda name, generation: blob)
    monkeypatch.setattr(c.engine, "_holders", lambda p: False)
    with c.verified_file(row, bucket) as parent:
        assert parent >= 0
    monkeypatch.setattr(c.engine, "_holders", lambda p: True)
    with pytest.raises(ValueError, match="held open"):
        with c.verified_file(row, bucket):
            pytest.fail("holder escaped original engine")
