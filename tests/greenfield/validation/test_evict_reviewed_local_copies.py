"""Local tiny-file safety checks; all cloud calls and payload unlinks mocked."""

import base64
from copy import deepcopy
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import google_crc32c
import pytest

from scripts.greenfield import evict_reviewed_local_copies as e

MANIFEST = e.REPO / "docs/artifacts/db602-db609-local-compile-copy-eviction-review-20260910.json"


def test_exact_reviewed_manifest_scope():
    manifest = json.loads(MANIFEST.read_text())
    entries = e._validate_manifest(manifest)
    assert len(entries) == 98
    assert sum(row["size"] for row in entries) == 3_411_885_438
    assert {row["db"] for row in entries} == {602, 609}
    assert {row["rank"] for row in entries} == set(range(1, 8))


@pytest.mark.parametrize("mutation", [
    "outside_root", "rank0", "duplicate", "foreign_cloud", "restore_bucket",
    "receipt_sha", "generation", "symlink", "hardlink", "db", "status",
])
def test_manifest_scope_refusals(mutation):
    manifest = json.loads(MANIFEST.read_text())
    row = manifest["files"][0]
    if mutation == "outside_root":
        row["path"] = "/tmp/worker.log"
    elif mutation == "rank0":
        row["path"] = row["path"].replace("rank1", "rank0")
    elif mutation == "duplicate":
        manifest["files"][1] = deepcopy(row)
    elif mutation == "foreign_cloud":
        row["name"] = "results/foreign/worker.log"
    elif mutation == "restore_bucket":
        row["restore_uri"] = row["restore_uri"].replace(e.BUCKET, "other-bucket")
    elif mutation == "receipt_sha":
        row["receipt_sha256"] = "0" * 64
    elif mutation == "generation":
        row["generation"] = "-1"
    elif mutation == "symlink":
        row["is_symlink"] = True
    elif mutation == "hardlink":
        row["nlink"] = 2
    elif mutation == "db":
        row["db"] = 603
    else:
        manifest["status"] = "DELETED"
    with pytest.raises(ValueError):
        e._validate_manifest(manifest)


@pytest.mark.parametrize("code,stdout,stderr,expected", [
    (1, "", "", False), (0, "1234", "", True),
    (1, "", "Permission denied", None), (1, "1234", "", None),
    (2, "", "fatal", None),
])
def test_fuser_fail_closed(monkeypatch, code, stdout, stderr, expected):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(e.subprocess, "run", run)
    if expected is None:
        with pytest.raises(RuntimeError, match="could not prove"):
            e._holders(Path("/reviewed/file"))
    else:
        assert e._holders(Path("/reviewed/file")) is expected
    assert calls[0][0] == ["fuser", "--", "/reviewed/file"]
    assert calls[0][1]["timeout"] == 30


@pytest.fixture
def operation(tmp_path, monkeypatch):
    """Exercise actual file/lease/receipt implementation on two tiny files."""
    from google.cloud import storage

    entries = []
    for index in range(2):
        path = tmp_path / f"file{index}.txt"
        raw = f"data{index}".encode()
        path.write_bytes(raw)
        st = path.stat()
        entries.append(dict(
            path=str(path), inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns,
            nlink=1, sha256=sha256(raw).hexdigest(),
            crc32c=base64.b64encode(google_crc32c.Checksum(raw).digest()).decode(),
            generation=str(index + 11), name=f"results/test/file{index}.txt",
            restore_uri=f"gs://{e.BUCKET}/results/test/file{index}.txt#{index + 11}",
        ))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(files=entries)))
    receipt = tmp_path / "receipt.json"
    for name in ("WORKLOAD_LEASE", "RSYNC_LEASE"):
        lock = tmp_path / name
        lock.touch()
        monkeypatch.setattr(e, name, lock)
    monkeypatch.setattr(e, "LOCAL_ROOT", tmp_path)
    # Complete real 98-file schema is covered separately, without reading payloads.
    monkeypatch.setattr(e, "_validate_manifest", lambda value: value["files"])
    events = []
    state = SimpleNamespace(entries=entries, receipt=receipt, events=events,
                            on_cloud=lambda row, count: None, on_unlink=lambda index: None,
                            on_holder=lambda path, count: False, cloud_count=0, holders=0)
    class Bucket:
        location = "US-CENTRAL2"
        def reload(self):
            events.append(("bucket",))
        def blob(self, name, *, generation):
            row = next(row for row in entries if row["name"] == name)
            assert generation == int(row["generation"])
            blob = SimpleNamespace(generation=generation, size=row["size"], crc32c=row["crc32c"])
            def reload(*, if_generation_match):
                assert if_generation_match == generation
                state.cloud_count += 1
                events.append(("cloud", row["path"]))
                state.on_cloud(blob, state.cloud_count)
            blob.reload = reload
            return blob
    state.bucket = Bucket()
    def bucket(name):
        assert name == e.BUCKET
        return state.bucket
    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=bucket))
    def holder(path):
        state.holders += 1
        events.append(("holder", str(path)))
        return state.on_holder(path, state.holders)
    monkeypatch.setattr(e, "_holders", holder)
    def unlink(name, *, dir_fd):
        path = Path(f"/proc/self/fd/{dir_fd}").resolve() / name
        assert str(path) in {row["path"] for row in entries}
        durable = json.loads(receipt.read_text())
        row = next(row for row in durable["files"] if row["path"] == str(path))
        assert row["state"] == "UNLINK_PENDING"
        assert events[-1] == ("holder", str(path))
        events.append(("unlink", str(path)))
        state.on_unlink(len([x for x in events if x[0] == "unlink"]))
    monkeypatch.setattr(e.os, "unlink", unlink)
    state.args = ["--manifest", str(manifest), "--manifest-sha256", sha256(manifest.read_bytes()).hexdigest(),
                  "--receipt", str(receipt)]
    return state


def test_default_dry_run_and_repeated_receipt_refusal(operation):
    assert e.main(operation.args) == 0
    record = json.loads(operation.receipt.read_text())
    assert record["status"] == "DRY_RUN_VERIFIED_NOT_DELETED"
    assert record["local_files_deleted"] == 0
    assert operation.cloud_count == operation.holders == 2
    assert not any(item[0] == "unlink" for item in operation.events)
    original = operation.receipt.read_bytes()
    with pytest.raises(FileExistsError):
        e.main(operation.args + ["--apply"])
    assert operation.receipt.read_bytes() == original


def test_apply_rechecks_and_records_every_intent(operation):
    assert e.main(operation.args + ["--apply"]) == 0
    assert operation.cloud_count == operation.holders == 4
    record = json.loads(operation.receipt.read_text())
    assert record["status"] == "LOCAL_COPIES_EVICTED"
    assert record["local_files_deleted"] == 2
    assert record["local_payload_bytes_removed"] == 10
    assert all(row["state"] == "DELETED" for row in record["files"])
    assert record["cloud_objects_deleted"] == 0


@pytest.mark.parametrize("change", ["bytes", "holder", "generation", "crc"])
def test_change_after_global_preflight_blocks_unlink(operation, change):
    if change == "bytes":
        def changed(blob, count):
            if count == 2:
                row = operation.entries[0]
                Path(row["path"]).write_bytes(b"other")
                os.utime(row["path"], ns=(row["mtime_ns"], row["mtime_ns"]))
        operation.on_cloud = changed
    elif change == "holder":
        operation.on_holder = lambda path, count: count == 3
    else:
        def changed(blob, count):
            if count == 3:
                if change == "generation":
                    blob.generation += 1
                else:
                    blob.crc32c = "AAAAAA=="
        operation.on_cloud = changed
    with pytest.raises(ValueError):
        e.main(operation.args + ["--apply"])
    assert not any(item[0] == "unlink" for item in operation.events)
    record = json.loads(operation.receipt.read_text())
    assert record["status"] == "FAILED_UNCERTAIN"
    assert record["files"][0]["state"] == "UNLINK_PENDING"


def test_partial_failure_keeps_exact_durable_outcome(operation):
    def fail(index):
        if index == 2:
            raise OSError("injected second unlink failure")
    operation.on_unlink = fail
    with pytest.raises(OSError, match="second unlink"):
        e.main(operation.args + ["--apply"])
    record = json.loads(operation.receipt.read_text())
    assert record["status"] == "FAILED_PARTIAL_UNCERTAIN"
    assert record["local_files_deleted"] == 1
    assert record["local_payload_bytes_removed"] == 5
    assert [row["state"] for row in record["files"]] == ["DELETED", "UNLINK_PENDING"]


def test_unlink_side_effect_then_exception_remains_uncertain(operation):
    removed_in_mock = []
    def interrupt_after_side_effect(index):
        removed_in_mock.append(operation.entries[index - 1]["path"])
        raise KeyboardInterrupt("injected after unlink side effect")
    operation.on_unlink = interrupt_after_side_effect
    with pytest.raises(KeyboardInterrupt, match="after unlink"):
        e.main(operation.args + ["--apply"])
    record = json.loads(operation.receipt.read_text())
    assert record["status"] == "FAILED_UNCERTAIN"
    assert record["local_files_deleted"] == 0
    assert record["deletion_counts_are_confirmed_only"] is True
    assert record["unconfirmed_unlink_paths"] == removed_in_mock


@pytest.mark.parametrize("lock_name", ["WORKLOAD_LEASE", "RSYNC_LEASE"])
def test_both_leases_refuse_before_receipt_or_cloud(operation, lock_name):
    with getattr(e, lock_name).open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            e.main(operation.args + ["--apply"])
    assert not operation.receipt.exists()
    assert operation.events == []


def test_symlink_ancestors_and_leaf_refuse(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        with e._directory(link):
            pytest.fail("followed a symlink ancestor")
    target = real / "target"
    target.write_bytes(b"small")
    leaf = real / "leaf"
    leaf.symlink_to(target)
    with pytest.raises(OSError):
        with e._verified_file(dict(path=str(leaf)), None):
            pytest.fail("followed a symlink leaf")


def test_receipt_failure_before_intent_prevents_unlink(operation, monkeypatch):
    durable = e._durable_receipt
    def refuse(path, receipt):
        if any(row["state"] == "UNLINK_PENDING" for row in receipt["files"]):
            raise OSError("injected intent persistence failure")
        return durable(path, receipt)
    monkeypatch.setattr(e, "_durable_receipt", refuse)
    with pytest.raises(OSError, match="intent persistence"):
        e.main(operation.args + ["--apply"])
    assert not any(item[0] == "unlink" for item in operation.events)
    assert all(row["state"] == "VERIFIED" for row in json.loads(operation.receipt.read_text())["files"])
