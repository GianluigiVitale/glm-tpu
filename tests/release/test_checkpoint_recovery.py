"""Metadata refusal tests only; no GCS calls or weight reads."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

from tools import check_checkpoint_recovery as recovery


def sources():
    files = [
        dict(
            filename=f"model-{i:05d}-of-00141.safetensors",
            file_bytes=1234,
            sha256="a" * 64,
        )
        for i in range(1, 142)
    ]
    manifest = dict(
        source=dict(uri="gs://" + recovery.BUCKET + "/" + recovery.MODEL, files=files)
    )
    sealed = dict(
        source_files=[dict(row, generation=123, crc32c="AAAAAA==") for row in files]
    )
    blobs = [
        SimpleNamespace(
            name=recovery.MODEL + "/" + row["filename"],
            size=1234,
            generation=123,
            crc32c="AAAAAA==",
        )
        for row in files
    ]
    return manifest, sealed, blobs


@pytest.mark.parametrize(
    "change",
    [None, "missing", "generation", "crc", "size", "duplicate", "hash", "bucket"],
)
def test_source_requires_complete_exact_sealed_generation_size_crc(change):
    manifest, sealed, blobs = sources()
    if change == "missing":
        blobs.pop()
    elif change == "generation":
        blobs[0].generation += 1
    elif change == "crc":
        blobs[0].crc32c = "BBBBBB=="
    elif change == "size":
        blobs[0].size += 1
    elif change == "duplicate":
        sealed["source_files"][-1] = sealed["source_files"][0]
    elif change == "hash":
        sealed["source_files"][0]["sha256"] = "b" * 64
    elif change == "bucket":
        manifest["source"]["uri"] = "gs://another-bucket/models"
    if change:
        with pytest.raises(ValueError):
            recovery.reconcile_sources(manifest, sealed, blobs)
    else:
        assert len(recovery.reconcile_sources(manifest, sealed, blobs)) == 141


@pytest.mark.parametrize("change", [None, "missing", "size", "duplicate", "escape"])
def test_overlay_is_complete_but_does_not_claim_fresh_payload_hash(change):
    files = [
        dict(filename=f"layer/part{i}.safetensors", byte_count=123, sha256="a" * 64)
        for i in range(96)
    ]
    manifest = dict(files=files)
    blobs = [
        SimpleNamespace(
            name=recovery.OVERLAY + "/" + row["filename"],
            size=123,
            generation=456,
            crc32c="AAAAAA==",
        )
        for row in files
    ]
    if change == "missing":
        blobs.pop()
    elif change == "size":
        blobs[0].size += 1
    elif change == "duplicate":
        files[-1] = files[0]
    elif change == "escape":
        files[0]["filename"] = "../outside"
    if change:
        with pytest.raises(ValueError):
            recovery.reconcile_overlay(manifest, blobs)
    else:
        result = recovery.reconcile_overlay(manifest, blobs)
        assert len(result) == 96 and not any(row["payload_rehashed"] for row in result)


def test_metadata_download_is_bounded_and_generation_pinned(monkeypatch):
    raw = b'{"fixture":true}'
    monkeypatch.setattr(
        recovery, "PINS", {"manifest": ("small.json", sha256(raw).hexdigest(), 32)}
    )
    calls = []
    blob = SimpleNamespace(
        size=len(raw),
        generation=42,
        download_as_bytes=lambda **kwargs: calls.append(kwargs) or raw,
    )
    bucket = SimpleNamespace(
        name=recovery.BUCKET, location="US-CENTRAL2", get_blob=lambda name: blob
    )
    values, receipts = recovery.authenticated_metadata(bucket)
    assert values == {"manifest": {"fixture": True}}
    assert calls == [{"if_generation_match": 42, "timeout": 60}]
    assert receipts[0]["generation"] == 42
    blob.size = 33
    with pytest.raises(ValueError):
        recovery.authenticated_metadata(bucket)
    assert len(calls) == 1
    bucket.location = "EU"
    with pytest.raises(ValueError):
        recovery.authenticated_metadata(bucket)
