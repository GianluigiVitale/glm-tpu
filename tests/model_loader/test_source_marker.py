"""Tests of :mod:`glm_tpu.model_loader.source_marker`: the completion marker of a local source, written only when every
shard and metadata file equals the upstream listing, and read the way the runtime reads it (``site_args``)."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from glm_tpu.config import model
from glm_tpu.config.site import site_args
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader import source_marker
from tests.fixtures.site import example_site

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    """A safetensors file: the little-endian header length, the JSON header, the payloads in order."""
    header, payload = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
        payload += data
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + payload


def write_source(root: Path) -> Path:
    """Two shards named by an index, and every metadata file the marker covers."""
    root.mkdir()
    (root / SHARDS[0]).write_bytes(safetensors({"a.weight": ("BF16", [2, 2], bytes(range(8)))}))
    (root / SHARDS[1]).write_bytes(safetensors({"b.weight": ("F32", [2], bytes(range(8, 16)))}))
    index = dict(metadata=dict(total_size=16), weight_map={"a.weight": SHARDS[0], "b.weight": SHARDS[1]})
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    for name in source_marker.METADATA_FILES:
        if name != "model.safetensors.index.json":
            (root / name).write_text(json.dumps({"file": name}))
    return root


def listing(root: Path) -> dict[str, dict]:
    """The upstream listing of ``root`` as the Hub reports it: LFS SHA-256 for the shards, git blob ids otherwise."""
    result = {}
    for name in [*SHARDS, *source_marker.METADATA_FILES]:
        data = (root / name).read_bytes()
        if name.endswith(".safetensors"):
            result[name] = dict(size=len(data), sha256=sha256(data).hexdigest(), git_blob_sha1=None)
        else:
            blob = sha1(b"blob %d\0" % len(data) + data).hexdigest()
            result[name] = dict(size=len(data), sha256=None, git_blob_sha1=blob)
    return result


def shard_bytes(root: Path) -> int:
    return sum((root / name).stat().st_size for name in SHARDS)


def mark(root: Path, output: Path, upstream: dict, **options):
    times = iter([datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 0, 5, tzinfo=UTC)])
    options = dict(
        dict(expected_shards=2, expected_bytes=shard_bytes(root), workers=2, upstream_kind="huggingface"), **options
    )
    return source_marker.mark_source(root, output, upstream=upstream, now=lambda: next(times), **options)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return write_source(tmp_path / "source")


def test_hash_file_matches_sha256_and_the_git_blob_id(tmp_path: Path):
    path = tmp_path / "f"
    path.write_bytes(b"hello\n")
    # `printf 'hello\n' | git hash-object --stdin`
    assert source_marker.hash_file(path) == dict(
        size=6,
        sha256=sha256(b"hello\n").hexdigest(),
        git_blob_sha1="ce013625030ba8dba906f756967f9e9ca394464a",
    )


def test_marker_records_every_file_and_the_runtime_accepts_it(source: Path, tmp_path: Path):
    output = tmp_path / "SOURCE_COMPLETE.json"
    report = mark(source, output, listing(source))
    raw = output.read_bytes()
    marker = json.loads(raw)
    assert report == dict(
        schema="glm_tpu_source_complete_report_v1",
        output=str(output),
        marker_sha256=sha256(raw).hexdigest(),
        repository=model.MODEL_ID,
        revision=model.REVISION,
        upstream="huggingface",
        verified_shards=2,
        verified_bytes=shard_bytes(source),
        metadata_files=len(source_marker.METADATA_FILES),
        passed=True,
    )
    assert marker["passed"] is True and marker["schema"] == source_marker.SCHEMA
    assert (marker["repository"], marker["revision"]) == (model.MODEL_ID, model.REVISION)
    assert [s["name"] for s in marker["shards"]] == list(SHARDS)
    assert marker["shards"][0]["sha256"] == sha256((source / SHARDS[0]).read_bytes()).hexdigest()
    assert marker["shards"][0]["tensor_count"] == 1
    assert [m["name"] for m in marker["metadata"]] == list(source_marker.METADATA_FILES)
    assert (marker["started_utc"], marker["finished_utc"]) == ("2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00")
    assert raw == (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    assert oct(os.stat(output).st_mode & 0o777) == "0o600"
    # The runtime's own reading (site_args) accepts exactly these fields once the counts are the pinned ones.
    pinned = dict(marker, verified_shards=model.SOURCE_SHARDS, verified_bytes=model.SOURCE_BYTES)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "SOURCE_COMPLETE.json").write_bytes(json.dumps(pinned).encode())
    digest = sha256((model_path / "SOURCE_COMPLETE.json").read_bytes()).hexdigest()
    site = example_site(
        tmp_path, paths=dict(model_path=str(model_path)), checkpoint=dict(source_complete_sha256=digest)
    )
    assert site_args(SimpleNamespace(), site).source_complete_sha256 == digest


def test_a_marker_checks_another_copy_of_the_same_source(source: Path, tmp_path: Path):
    first = tmp_path / "first.json"
    mark(source, first, listing(source))
    report = mark(source, tmp_path / "second.json", source_marker.marker_listing(first), upstream_kind="marker")
    assert report["passed"] is True and report["upstream"] == "marker"


def _flip_shard_byte(root: Path, upstream: dict) -> None:
    data = bytearray((root / SHARDS[1]).read_bytes())
    data[-1] ^= 1
    (root / SHARDS[1]).write_bytes(bytes(data))


def _edit_metadata(root: Path, upstream: dict) -> None:
    (root / "tokenizer.json").write_text(json.dumps({"file": "other"}))


def _missing_metadata(root: Path, upstream: dict) -> None:
    (root / "chat_template.jinja").unlink()


def _upstream_has_another_shard(root: Path, upstream: dict) -> None:
    upstream["model-00003-of-00003.safetensors"] = dict(size=1, sha256="0" * 64, git_blob_sha1=None)


def _upstream_lacks_a_metadata_file(root: Path, upstream: dict) -> None:
    del upstream["generation_config.json"]


def _upstream_without_digest(root: Path, upstream: dict) -> None:
    upstream[SHARDS[0]] = dict(upstream[SHARDS[0]], sha256=None)


def _upstream_size_differs(root: Path, upstream: dict) -> None:
    upstream[SHARDS[0]] = dict(upstream[SHARDS[0]], size=upstream[SHARDS[0]]["size"] + 1)


REFUSALS = {
    "shard byte": (_flip_shard_byte, "SHA-256 differs from upstream"),
    "metadata edited": (_edit_metadata, "tokenizer.json: git blob id differs from upstream"),
    "metadata missing": (_missing_metadata, "chat_template.jinja: missing from the source"),
    "extra upstream shard": (_upstream_has_another_shard, "upstream lists 3"),
    "upstream lacks metadata": (_upstream_lacks_a_metadata_file, "generation_config.json: not in the upstream listing"),
    "no upstream digest": (_upstream_without_digest, "the upstream listing has no digest"),
    "size differs": (_upstream_size_differs, "bytes, upstream"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_any_difference_refuses_and_writes_nothing(source: Path, tmp_path: Path, case: str):
    upstream = listing(source)
    mutate, message = REFUSALS[case]
    mutate(source, upstream)
    output = tmp_path / "SOURCE_COMPLETE.json"
    with pytest.raises(CheckpointValidationError, match=message):
        mark(source, output, upstream)
    assert not output.exists()


def test_the_pinned_shard_count_and_bytes_are_required(source: Path, tmp_path: Path):
    output = tmp_path / "SOURCE_COMPLETE.json"
    with pytest.raises(CheckpointValidationError, match="2 shards, 141 expected"):
        mark(source, output, listing(source), expected_shards=model.SOURCE_SHARDS)
    with pytest.raises(CheckpointValidationError, match="755632050320 expected"):
        mark(source, output, listing(source), expected_bytes=model.SOURCE_BYTES)
    assert not output.exists()


def test_an_existing_marker_is_never_overwritten(source: Path, tmp_path: Path):
    output = tmp_path / "SOURCE_COMPLETE.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        mark(source, output, listing(source))
    assert output.read_text() == "keep"


@pytest.mark.parametrize(
    "change",
    [dict(passed=False), dict(repository="other/model"), dict(revision="0" * 40), dict(shards=None)],
    ids=["not passed", "repository", "revision", "no shards"],
)
def test_marker_listing_refuses_another_source(source: Path, tmp_path: Path, change: dict):
    first = tmp_path / "first.json"
    mark(source, first, listing(source))
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(json.loads(first.read_text()), **change)))
    with pytest.raises(CheckpointValidationError, match="is not a passed completion marker"):
        source_marker.marker_listing(other)


def test_hub_listing_reads_lfs_digests_and_blob_ids(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub

    calls = []

    class Api:
        def list_repo_tree(self, repository, revision, recursive):
            calls.append((repository, revision, recursive))
            return [
                SimpleNamespace(path="folder"),
                SimpleNamespace(path=SHARDS[0], size=10, blob_id="p" * 40, lfs=SimpleNamespace(sha256="a" * 64)),
                SimpleNamespace(path="config.json", size=3, blob_id="b" * 40, lfs=None),
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert source_marker.hub_listing() == {
        SHARDS[0]: dict(size=10, sha256="a" * 64, git_blob_sha1=None),
        "config.json": dict(size=3, sha256=None, git_blob_sha1="b" * 40),
    }
    assert calls == [(model.MODEL_ID, model.REVISION, False)]
