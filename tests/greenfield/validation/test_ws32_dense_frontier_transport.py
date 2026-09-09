"""Real publisher/downloader, in-memory GCS transport; no external writes."""

import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

from google.api_core.exceptions import PreconditionFailed
import google_crc32c
import pytest

from scripts.greenfield import ws32_dense_frontier_transport as transport

TAG = "greenfield_fp8_ws32_dense_frontier_d01_20260909T150000000000000Z"
PIN = "a" * 40


class Blob:
    def __init__(self, bucket, name, generation=None):
        self.bucket, self.name = bucket, name
        self.requested = generation
        self.generation = generation
        if name in bucket.objects:
            self.reload(if_generation_match=generation)

    def reload(self, if_generation_match=None):
        generation, data = self.bucket.objects[self.name]
        if if_generation_match is not None and generation != if_generation_match:
            raise PreconditionFailed("generation changed")
        self.generation, self.size = generation, len(data)
        self.crc32c = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()

    def upload_from_file(self, stream, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "crc32c"
        if self.name in self.bucket.objects:
            raise PreconditionFailed("already exists")
        self.bucket.objects[self.name] = (
            9007199254740993 + len(self.bucket.objects),
            stream.read(),
        )
        self.bucket.events.append(("upload", self.name))

    def download_as_bytes(self, *, if_generation_match, checksum="crc32c"):
        assert checksum == "crc32c"
        self.reload(if_generation_match=if_generation_match)
        self.bucket.events.append(("download", self.name))
        return self.bucket.objects[self.name][1]

    def download_to_file(self, stream, **kwargs):
        stream.write(self.download_as_bytes(**kwargs))

    def download_to_filename(self, name, **kwargs):
        Path(name).write_bytes(self.download_as_bytes(**kwargs))


class Bucket:
    location = "US-CENTRAL2"

    def __init__(self):
        self.objects, self.events = {}, []

    def reload(self):
        self.events.append(("region", self.location))

    def blob(self, name, generation=None):
        return Blob(self, name, generation)

    def get_blob(self, name):
        return self.blob(name) if name in self.objects else None

    def client(self):
        def bucket(name):
            assert name == "driftbench-dsv4-uc"
            return self

        return NS(bucket=bucket)


def originals(root, rank):
    root.mkdir()
    for name in transport.FILES:
        raw = (
            json.dumps(
                dict(
                    launch_rank=rank,
                    code_hash=PIN,
                    programs=dict(dense01=dict(optimized_hlo_sha256="b" * 64)),
                )
            ).encode()
            if name == "runner.json"
            else name.encode()
        )
        (root / name).write_bytes(raw)


def published(tmp_path):
    bucket = Bucket()
    for rank in range(8):
        source = tmp_path / f"source{rank}"
        originals(source, rank)
        transport.publish_rank(tag=TAG, rank=rank, root=source, client=bucket.client())
    bucket.events.clear()
    return bucket


def test_actual_publication_generation_download_and_fleet_reader(tmp_path, monkeypatch):
    bucket = published(tmp_path)
    destination = tmp_path / "fleet"
    calls = []

    def replay(root, records, **kwargs):
        # Numerical/graph replay is tested in the evidence suite. This tests
        # actual transport + exact fleet layout delivered to that same reader.
        assert root == destination and kwargs == dict(
            pin=PIN, tag=TAG, repo=tmp_path, original_root=tmp_path / "original"
        )
        assert [r["launch_rank"] for r in records] == list(range(8))
        for rank in range(8):
            for name in transport.FILES:
                assert (root / f"rank{rank}" / name).read_bytes() == (
                    tmp_path / f"source{rank}" / name
                ).read_bytes()
            assert json.loads((root / f"rank{rank}/ledger_source.json").read_text())[
                "generation"
            ].isdigit()
        calls.append("replay")
        return dict(reproduced=True, numerical_promotion=False, performance_claim=False)

    monkeypatch.setattr(transport.evidence, "validate_fleet", replay)
    result = transport.collect(
        tag=TAG,
        pin=PIN,
        root=destination,
        repo=tmp_path,
        original_root=tmp_path / "original",
        client=bucket.client(),
    )
    assert (
        calls == ["replay"] and result["iterations"] == 0 and result["latency"] is None
    )
    downloads = [name for op, name in bucket.events if op == "download"]
    assert all(name.endswith("worker_receipts.json") for name in downloads[:8])
    assert result["original_bytes"] == sum(
        len(raw) for _, raw in bucket.objects.values()
    )
    transport.validate_record(
        result,
        PIN,
        root=destination,
        repo=tmp_path,
        original_root=tmp_path / "original",
    )
    assert calls == ["replay", "replay"]
    bucket.events.clear()
    recovered = transport.replay_collected(
        tag=TAG,
        pin=PIN,
        root=destination,
        repo=tmp_path,
        original_root=tmp_path / "original",
        client=bucket.client(),
    )
    assert recovered == result
    # Only8small ledger downloads, no original payload copies or writes.
    assert [n for op, n in bucket.events if op == "download"] == [
        f"results/{TAG}/workers/rank{r}/worker_receipts.json" for r in range(8)
    ]
    assert not any(op == "upload" for op, _ in bucket.events)
    changed = deepcopy(result)
    changed["numerical_promotion"] = True
    with pytest.raises(ValueError, match="aggregate"):
        transport.validate_record(
            changed,
            PIN,
            root=destination,
            repo=tmp_path,
            original_root=tmp_path / "original",
        )
    with pytest.raises(FileExistsError):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=destination,
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )


@pytest.mark.parametrize(
    "mutation", ["ledger", "payload", "remote_generation", "source_path", "symlink"]
)
def test_collected_recovery_reauthenticates_originals(tmp_path, monkeypatch, mutation):
    bucket = published(tmp_path)
    destination = tmp_path / "fleet"
    monkeypatch.setattr(
        transport.evidence, "validate_fleet", lambda *a, **k: dict(reproduced=True)
    )
    transport.collect(
        tag=TAG,
        pin=PIN,
        root=destination,
        repo=tmp_path,
        original_root=tmp_path,
        client=bucket.client(),
    )

    def forbidden(*a, **kw):
        raise AssertionError("entered numerical replay before authenticating originals")

    monkeypatch.setattr(transport.evidence, "validate_fleet", forbidden)
    rank = destination / "rank7"
    if mutation == "ledger":
        (rank / "worker_receipts.json").write_text("[]")
    elif mutation == "payload":
        (rank / "wide_final.npz").write_text("tampered")
    elif mutation == "remote_generation":
        key = f"results/{TAG}/workers/rank7/worker_receipts.json"
        gen, raw = bucket.objects[key]
        bucket.objects[key] = (gen + 1, raw)
    elif mutation == "source_path":
        p = rank / "ledger_source.json"
        r = json.loads(p.read_bytes())
        r["name"] = "results/other/worker_receipts.json"
        p.write_text(json.dumps(r))
    else:
        p = rank / "wide_final.npz"
        p.unlink()
        p.symlink_to(tmp_path / "source7/wide_final.npz")
    with pytest.raises((ValueError, PreconditionFailed, SystemExit)):
        transport.replay_collected(
            tag=TAG,
            pin=PIN,
            root=destination,
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "escape",
        "size",
        "generation",
        "sha",
        "crc",
        "aux_cap",
        "npz_cap",
    ],
)
def test_bad_last_ledger_refuses_before_any_payload(tmp_path, mutation):
    bucket = published(tmp_path)
    name = f"results/{TAG}/workers/rank7/worker_receipts.json"
    generation, raw = bucket.objects[name]
    rows = json.loads(raw)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[0] = rows[1]
    elif mutation == "escape":
        rows[0]["name"] = f"results/{TAG}/workers/rank7/../runner.json"
    elif mutation == "size":
        rows[0]["size"] = True
    elif mutation == "generation":
        rows[0]["generation"] = "0"
    elif mutation == "sha":
        rows[0]["original_sha256"] = "a"
    elif mutation == "crc":
        rows[0]["crc32c"] = "invalid"
    elif mutation == "aux_cap":
        rows[0]["size"] = transport.MAX_AUX_BYTES + 1
    else:
        next(r for r in rows if r["name"].endswith("wide_final.npz"))["size"] = (
            128 << 20
        ) + 1
    bucket.objects[name] = (generation, json.dumps(rows).encode())
    with pytest.raises(ValueError):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )
    assert all(
        name.endswith("worker_receipts.json")
        for op, name in bucket.events
        if op == "download"
    )
    assert not (tmp_path / "fleet").exists()


@pytest.mark.parametrize("mutation", ["generation", "bytes", "sha"])
def test_payload_generation_crc_sha_refusal(tmp_path, mutation):
    bucket = published(tmp_path)
    name = f"results/{TAG}/workers/rank0/wide_final.npz"
    generation, raw = bucket.objects[name]
    if mutation == "generation":
        bucket.objects[name] = (generation + 1, raw)
    elif mutation == "bytes":
        bucket.objects[name] = (generation, b"x" * len(raw))
    else:
        ledger = f"results/{TAG}/workers/rank0/worker_receipts.json"
        gen, data = bucket.objects[ledger]
        rows = json.loads(data)
        next(r for r in rows if r["name"] == name)["original_sha256"] = "0" * 64
        bucket.objects[ledger] = (gen, json.dumps(rows).encode())
    with pytest.raises((PreconditionFailed, ValueError, SystemExit)):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )


def test_partial_failure_preserved_and_idempotent_no_overwrite(tmp_path, monkeypatch):
    bucket = Bucket()
    (tmp_path / "runner.json").write_text('{"status":"DIAGNOSTIC_FAILED"}')
    (tmp_path / "dense01.optimized_hlo.txt").write_bytes(b"partial original")
    (tmp_path / "wide_final.npz").symlink_to(tmp_path / "absent")
    (tmp_path / "layer0_wk_decode.npz").write_bytes(b"oversized")
    (tmp_path / "narrow_128.npz.pending").write_bytes(b"partial captured bytes")
    monkeypatch.setattr(transport, "LIMITS", dict(wk=1, model=128 << 20, aux=31 << 20))
    receipts = transport.publish_rank(
        tag=TAG, rank=0, root=tmp_path, client=bucket.client()
    )
    names = [r["name"].rsplit("/", 1)[1] for r in receipts]
    assert names == [
        "runner.json",
        "dense01.optimized_hlo.txt",
        "narrow_128.npz.pending",
        transport.NOTICE,
    ]
    assert (tmp_path / "layer0_wk_decode.npz").read_bytes() == b"oversized"
    before = deepcopy(bucket.objects)
    assert receipts == transport.publish_rank(
        tag=TAG, rank=0, root=tmp_path, client=bucket.client()
    )
    assert bucket.objects == before
    (tmp_path / "runner.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        transport.publish_rank(tag=TAG, rank=0, root=tmp_path, client=bucket.client())
    assert bucket.objects == before


def test_wrong_region_and_disk_refuse_before_payload(tmp_path, monkeypatch):
    bucket = published(tmp_path)
    bucket.location = "EU"
    with pytest.raises(ValueError, match="US-CENTRAL2"):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )
    assert not any(op == "download" for op, _ in bucket.events)
    bucket.location = "US-CENTRAL2"
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=0))
    with pytest.raises(ValueError, match="headroom"):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )
    assert all(
        name.endswith("worker_receipts.json")
        for op, name in bucket.events
        if op == "download"
    )
