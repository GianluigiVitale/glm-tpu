"""Actual bounded writes/gzip/generation readback with in-memory GCS, not TPU."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_native_benchmark_transport as transport
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.ws32_compile_originals import _write_compiler_original
from scripts.greenfield.ws32_native_benchmark_worker import NativeBenchmarkJournal
from scripts.greenfield.ws32_history_call_evidence import preserve_call
from tests.greenfield.validation.gcs_fixtures import Bucket

TAG = "greenfield_ws32_native_benchmark_20260912T220000000000000Z"
PIN = "a" * 40


def test_caps_cover_retained_production_graph_sizes_without_unbounded_rank_growth():
    limits = transport.file_limits()
    # DB620 exact original sizes, not tiny test strings. Native sampling needs
    # new actual graphs, but must not be rejected by a known-insufficient cap.
    for graph, size in {"prefill_chunk":103234144,"prefill_tail":103234144,
                        "decode":74715008,"observer":76037394}.items():
        assert limits[f"{graph}.optimized_hlo.txt"] >= size
    assert transport.COLD_CAP == 576 << 20
    assert 8 * ((576+512+128+6) << 20) < 10 << 30


def originals(root, rank):
    values = {}
    for relative in transport.file_limits():
        # Graph bytes identical across ranks; every other original is distinct.
        graph = any(relative.endswith("." + form) for form in transport.FORMS)
        raw = f"fixture/{relative}/{0 if graph else rank}".encode()
        path = root / f"native.rank{rank}" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        values[relative] = raw
    return values


def publish(root, bucket, rank=0):
    return transport.publish_rank(run_root=root, tag=TAG, pin=PIN, rank=rank, client=bucket.client())


def collect(destination, bucket, rank=0):
    return transport.collect_rank(destination=destination, tag=TAG, pin=PIN, rank=rank,
        client=bucket.client(), blobs={name: bucket.get_blob(name) for name in bucket.objects})


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    root = tmp_path / TAG
    values = originals(root, 0)
    bucket = Bucket()
    manifest = publish(root, bucket)
    destination = tmp_path / "collected"
    destination.mkdir()
    return NS(root=root, values=values, bucket=bucket, manifest=manifest, destination=destination)


def test_real_originals_roundtrip_idempotence_and_no_local_gzip_copies(case):
    assert len(case.manifest["files"]) == 67 and not case.manifest["missing"]
    assert not case.manifest["numerical_claim"] and not case.manifest["performance_claim"]
    before = deepcopy(case.bucket.objects)
    assert publish(case.root, case.bucket) == case.manifest
    assert case.bucket.objects == before
    result = collect(case.destination, case.bucket)
    assert result["manifest"] == case.manifest
    assert collect(case.destination, case.bucket) == result
    for name, raw in case.values.items():
        path = case.destination / "native.rank0" / name
        assert path.read_bytes() == raw
        remote = transport.object_name(TAG, 0, name)
        assert gzip.decompress(case.bucket.objects[remote][1]) == raw
    assert not list(case.root.rglob("*.gz"))


@pytest.mark.parametrize("mutation", ["rank", "pin", "profile", "missing", "duplicate",
    "traversal", "oversize", "total", "generation", "hash", "crc", "schema", "row_schema"])
def test_manifest_refuses_mutations(case, mutation):
    value = deepcopy(case.manifest)
    if mutation == "rank": value["rank"] = True
    elif mutation == "pin": value["code_hash"] = "b" * 40
    elif mutation == "profile": value["profile"] = "old_long"
    elif mutation == "missing": value["missing"] = [] if value["missing"] else ["missing"]
    elif mutation == "duplicate": value["files"].append(value["files"][0])
    elif mutation == "traversal": value["files"][0]["relative_path"] = "../outside"
    elif mutation == "oversize": value["files"][0]["original_bytes"] = transport.COLD_CAP
    elif mutation == "total": value["original_bytes"] += 1
    elif mutation == "generation": value["files"][0]["generation"] = 1
    elif mutation == "hash": value["files"][0]["original_sha256"] = "A" * 64
    elif mutation == "crc": value["files"][0]["crc32c"] = "not-crc"
    elif mutation == "schema": value["extra"] = True
    else: value["files"][0]["extra"] = True
    with pytest.raises(ValueError):
        transport.validate_manifest(value, tag=TAG, pin=PIN, rank=0)


@pytest.mark.parametrize("mutation", ["generation", "payload", "bomb", "extra", "local", "link", "space", "region"])
def test_collection_refuses_damage_preserving_originals(case, monkeypatch, mutation):
    row = case.manifest["files"][0]
    if mutation in ("generation", "payload", "bomb"):
        generation, data = case.bucket.objects[row["name"]]
        if mutation == "generation": generation += 1
        else: data = gzip.compress(b"x" * (row["original_bytes"] + (100000 if mutation == "bomb" else 0)))
        case.bucket.objects[row["name"]] = generation, data
        if mutation == "bomb":
            blob = case.bucket.get_blob(row["name"])
            row.update(size=blob.size, crc32c=blob.crc32c)
            name = transport.prefix(TAG, 0) + "manifest.json"
            generation, _ = case.bucket.objects[name]
            case.bucket.objects[name] = generation, json.dumps(case.manifest).encode()
    elif mutation == "extra": case.bucket.objects[transport.prefix(TAG, 0) + "extra.json"] = 999, b"extra"
    elif mutation in ("local", "link"):
        target = case.destination / "native.rank0" / row["relative_path"]
        target.parent.mkdir(parents=True)
        if mutation == "local": target.write_bytes(b"keep existing")
        else: target.symlink_to(case.root / "native.rank0" / row["relative_path"])
    elif mutation == "space": monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=1))
    else: case.bucket.location = "EU"
    before = {n: (case.root / "native.rank0" / n).read_bytes() for n in case.values}
    with pytest.raises((ValueError, RuntimeError)):
        collect(case.destination, case.bucket)
    assert {n: (case.root / "native.rank0" / n).read_bytes() for n in case.values} == before


def test_fleet_shared_HLO_one_physical_copy_and_exact_union(case):
    for rank in range(1, 8):
        originals(case.root, rank)
        publish(case.root, case.bucket, rank)
    kwargs = dict(destination=case.destination, tag=TAG, pin=PIN, client=case.bucket.client(),
                  blobs={name: case.bucket.get_blob(name) for name in case.bucket.objects})
    rows = transport.collect_fleet(**kwargs)
    assert len(rows) == 8
    graph_names = {row["name"] for result in rows for row in result["manifest"]["files"]
                   if "/hlo/" in row["name"]}
    assert len(graph_names) == 20
    for graph, folder in transport.GRAPHS.items():
        for form in transport.FORMS:
            paths = [case.destination / f"native.rank{r}" / folder / f"{graph}.{form}" for r in range(8)]
            assert len({p.stat().st_ino for p in paths}) == 1
    name = f"results/{TAG}/native_cold/hlo/unknown.gz"
    case.bucket.objects[name] = 9999, b"extra"
    kwargs["blobs"] = {n: case.bucket.get_blob(n) for n in case.bucket.objects}
    with pytest.raises(ValueError, match="union"):
        transport.collect_fleet(**kwargs)


def test_partial_and_terminal_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    root = tmp_path / TAG
    path = root / "native.rank0" / "runner.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"complete": false}')
    bucket = Bucket()
    value = publish(root, bucket)
    assert len(value["files"]) == 1 and len(value["missing"]) == 66
    bucket.objects[f"results/{TAG}/SUCCESS"] = 999, b"existing terminal"
    before = deepcopy(bucket.objects)
    with pytest.raises(ValueError, match="terminal"):
        publish(root, bucket)
    assert bucket.objects == before


def test_all_actual_writer_routes_enforce_caps_before_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    root = tmp_path / TAG / "native.rank0"
    root.mkdir(parents=True)
    path = root / "runner.json"
    _atomic_json(path, {"fixture": True})
    old = path.read_bytes()
    monkeypatch.setattr(transport, "COLD_CAP", len(old) + 1)
    with pytest.raises(ValueError, match="byte cap"):
        _atomic_json(path, {"fixture": "replacement"})
    assert path.read_bytes() == old
    with pytest.raises(ValueError): _write_compiler_original(root, {}, "decode", "stablehlo.mlir", "x" * 100)
    assert not (root / "decode.stablehlo.mlir").exists()
    (root / "wk").mkdir()
    with pytest.raises(ValueError):
        preserve_call(root / "wk", {"phase": "p", "graph": "g", "completed": True}, index=0, used_bytes=0)
    assert not (root / "wk/call_records/call000.json").exists()
    # Restore cap to create the real journal, then prove future exact appends
    # refuse before changing its previous durable contents.
    monkeypatch.setattr(transport, "COLD_CAP", 1 << 20)
    journal = NativeBenchmarkJournal(root / "journal.jsonl", dict(profile=transport.PROFILE,
        compile_only=False, context_capacity=166912, jax_process_index=0,
        launch_process_id=0, code_hash=PIN))
    before = (root / "journal.jsonl").read_bytes()
    monkeypatch.setattr(transport, "COLD_CAP", 1)
    try:
        with pytest.raises(ValueError): journal.phase("fixture", passed=True)
    finally:
        journal.close()
    assert (root / "journal.jsonl").read_bytes() == before


def test_unrelated_writer_namespaces_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "require_write_size", lambda *a, **kw: pytest.fail("native route"))
    _atomic_json(tmp_path / "runner.json", {"old": True})
    _write_compiler_original(tmp_path, {}, "old", "stablehlo.mlir", "original")
    assert (tmp_path / "old.stablehlo.mlir").read_text() == "original"
