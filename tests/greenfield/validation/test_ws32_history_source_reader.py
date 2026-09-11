"""Tiny local source originals; no cloud, SSH, TPU, or full checkpoint reads."""

import base64
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import Ws32RuntimeFilePlan, Ws32RuntimeTensorPlan
from scripts.greenfield import ws32_history_source_reader as source
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_storage as storage
from tests.greenfield.validation.test_ws32_history_materializer_evidence import runtime_record

REPO = Path(__file__).resolve().parents[3]
TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T190000000000000Z"
PIN = "a" * 40


def forbidden(*args, **kwargs):
    raise AssertionError("live cloud/SSH/device or full-file read forbidden")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import jax
    from scripts.greenfield import ws32_prefill_moe_campaign
    from google.cloud import storage as cloud

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "local_devices", forbidden)
    monkeypatch.setattr(jax.distributed, "initialize", forbidden)
    monkeypatch.setattr(cloud, "Client", forbidden)
    monkeypatch.setattr(ws32_prefill_moe_campaign, "ssh", forbidden)
    monkeypatch.setattr(source.Snapshot, "read_all", forbidden)
    monkeypatch.setattr(source.Snapshot, "sha256", forbidden)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from scripts.greenfield.ws32_prefill_budget_campaign import FLEET_SHA

    root = tmp_path / TAG / "source_tiles"
    root.parent.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    names = ("test.q_bits", "test.q_scale", "test.head")
    dtype_rows = (("U8", np.uint8), ("F32", np.float32), ("BF16", ml_dtypes.bfloat16))
    plans, owners, rows, requests, expected_values = [], {}, {}, [], {}
    # Nonidentity launch/process/device/slot mappings, including all four source
    # owners per host. Tiny tensors preserve the real header/offset/SHA path.
    maps = tuple({(rank * 4 + j + 7) % 32: (rank * 4 + j + 9) % 32 for j in range(4)} for rank in range(8))
    rank_by_slot = {slot: rank for rank, slots in enumerate(maps) for slot in slots.values()}
    for slot in range(32):
        tensors, payload, selected = [], b"", {}
        for name, (dtype, npdtype) in zip(names, dtype_rows, strict=True):
            value = (np.arange(4).reshape(2, 2) + slot / 4).astype(npdtype)
            raw = value.tobytes()
            tensors.append(Ws32RuntimeTensorPlan(name, dtype, (2, 2), (2, 2), (None, None), len(payload), len(payload) + len(raw)))
            payload += raw
            selected[name] = sha256(raw).hexdigest()
            expected_values[slot, name] = value
        header = b"fixture-header-" + str(slot).encode() + b"\0" * 8
        filename = f"device_slot{slot:02d}.safetensors"
        plan = Ws32RuntimeFilePlan(slot, slot // 4, slot % 4, filename, header, tuple(tensors), len(payload))
        plans.append(plan)
        (runtime / filename).write_bytes(header + payload)
        rows[slot] = dict(filename=filename, file_bytes=plan.file_bytes, header_sha256=sha256(header).hexdigest(),
                          sha256=sha256(header + payload).hexdigest())
        owners[slot] = dict(full_file_sha256=rows[slot]["sha256"], selected=selected)
        for tensor in tensors:
            index = len(requests)
            requests.append(dict(index=index, key=f"tile{index:03d}", rank=rank_by_slot[slot], slot=slot,
                tensor=tensor.name, dtype=tensor.dtype, shape=list(tensor.local_shape),
                offset=len(header) + tensor.data_offset_start, bytes=tensor.byte_count, sha256=selected[tensor.name]))
    context = {key: sha256(key.encode()).hexdigest() for key in ("checkpoint_manifest_sha256",
        "checkpoint_success_sha256", "source_inventory_sha256", "mesh_sha256", "topology_sha256")}
    overlay = NS(manifest={"manifest_sha256": "b" * 64}, manifest_file_sha256="c" * 64,
        success_file_sha256="d" * 64, records={(layer, expert, feature): {
            "sha256": sha256(f"overlay{layer}/{expert}/{feature}".encode()).hexdigest()}
            for layer in range(3) for expert in range(8) for feature in range(4)})
    bindings = source.evidence.CheckpointBindings(None, context, owners, overlay)
    captures = tuple(dict(hostname=f"fixture-host-{rank}", jax_process_index=(rank + 3) % 8,
                          local_device_ids=list(maps[rank])) for rank in range(8))
    inventory = source.Inventory(NS(root=runtime, plans=tuple(plans), records_by_slot=rows), bindings,
                                 captures, maps, tuple(requests))
    records = [dict(runtime_record(rank, slots, bindings), tag=TAG, code_hash=PIN, protocol=protocol.PROTOCOL,
        hostname=captures[rank]["hostname"], topology_fleet_sha256=FLEET_SHA, boot_id=boot,
        status="DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION") for rank, slots in enumerate(maps)]
    monkeypatch.setattr(source, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(source, "_inventory", lambda repo: inventory)
    monkeypatch.setattr(source.subprocess, "check_output", lambda *a, **kw: PIN + "\n")
    calls = []

    def fetch(command, *, output, worker, timeout):
        rank = int(worker)
        assert timeout == source.FETCH_TIMEOUT
        args = shlex.split(command)
        assert args[:6] == ["cd", str(REPO), "&&", "JAX_PLATFORMS=cpu", source.PYTHON, "-m"]
        identity = json.loads(base64.b64decode(args[-1]))
        with monkeypatch.context() as patch:
            patch.setattr(source.socket, "gethostname", lambda: captures[rank]["hostname"])
            report = source._capture(inventory, identity, rank)
        output.write_bytes(b"fixture SSH notice\n" + source.MARKER.encode() + source._json(report))
        calls.append(rank)

    def local_fetch(fetch_callback, command, rank, maximum):
        path = tmp_path / "fixture_wire.txt"
        fetch_callback(command, output=path, worker=str(rank), timeout=source.FETCH_TIMEOUT)
        raw = path.read_bytes()
        assert len(raw) <= maximum
        path.unlink()
        return raw

    monkeypatch.setattr(source, "_fetch", local_fetch)
    return NS(root=root, runtime=runtime, inventory=inventory, records=records, fetch=fetch,
              calls=calls, expected=expected_values, monkeypatch=monkeypatch, tmp=tmp_path)


def prepare(f):
    return source.prepare_reader(root=f.root, repo=REPO, records=f.records, fetch=f.fetch)


def report(f, rank=0):
    identity = source._identity(f.inventory, f.records)
    with f.monkeypatch.context() as patch:
        patch.setattr(source.socket, "gethostname", lambda: f.inventory.captures[rank]["hostname"])
        return deepcopy(source._capture(f.inventory, identity, rank))


def save_manifest(f, value):
    (f.root / "manifest.json").write_bytes(source._json(value))


def test_selected_capture_and_network_free_recovery(fixture, monkeypatch):
    f = fixture
    reads = []
    real = source.Snapshot.pread

    def pread(snapshot, length, offset):
        if snapshot.path.parent == f.runtime:
            reads.append((snapshot.path.name, length, offset))
        return real(snapshot, length, offset)

    monkeypatch.setattr(source.Snapshot, "pread", pread)
    reader = prepare(f)
    assert f.calls == list(range(8))
    assert len(reads) == 32 + len(f.inventory.requests)
    assert sum(length for _, length, offset in reads if offset) == sum(row["bytes"] for row in f.inventory.requests)
    assert {path.name for path in f.root.iterdir()} == {"manifest.json", *(f"rank{i}.npz" for i in range(8))}
    assert reader.receipt["files"] == 9
    assert reader.receipt["retained_bytes"] < source.ROOT_LIMIT
    assert "NOT_GCS_GENERATION_OR_FULL_FILE_VERIFICATION" in reader.receipt["identity"]["provenance"]
    assert not reader.receipt["numerical_promotion"] and not reader.receipt["performance_claim"]
    monkeypatch.setattr(source, "_fetch", forbidden)
    loaded = source.load_reader(root=f.root, repo=REPO, records=f.records)
    assert len(reads) == 32 + len(f.inventory.requests)  # Capsule replay opens zero source files.
    assert loaded.receipt == reader.receipt
    for key, value in f.expected.items():
        for actual in (reader(*key), loaded(*key)):
            assert actual.dtype == value.dtype and np.array_equal(actual, value)
            with pytest.raises(ValueError):
                actual.setflags(write=True)
    for args in ((True, "test.q_bits"), (99, "test.q_bits"), (0, "unregistered")):
        with pytest.raises(ValueError, match="unregistered"):
            loaded(*args)


@pytest.mark.parametrize("mutation", ["tag", "pin", "rank", "process", "host", "boot", "context", "owner", "selected", "status", "count"])
def test_worker_identity_refuses_before_fetch_or_creation(fixture, mutation):
    f = fixture
    row = f.records[0]
    if mutation == "tag": row["tag"] = "unregistered"
    elif mutation == "pin": row["code_hash"] = "x" * 40
    elif mutation == "rank": row["launch_rank"] = False
    elif mutation == "process": row["jax_process_index"] = 0
    elif mutation == "host": row["hostname"] = "wrong-host"
    elif mutation == "boot": row["boot_id"] = "not-a-boot-id"
    elif mutation == "context": row["mesh_sha256"] = "0" * 64
    elif mutation == "owner": row["local_device_slots"][0]["device_slot"] = 31
    elif mutation == "selected": row["local_device_slots"][0]["observed_selected_tensor_sha256"] = {}
    elif mutation == "status": row["status"] = "RUNNING"
    else: f.records.pop()
    with pytest.raises(ValueError): prepare(f)
    assert not f.root.exists() and not f.calls


@pytest.mark.parametrize("mutation", ["directory", "existing", "symlink", "repo", "budget"])
def test_capsule_target_refuses_before_fetch(fixture, mutation, monkeypatch):
    f = fixture
    if mutation == "directory": f.root = f.tmp / "source_tiles"
    elif mutation == "existing": f.root.mkdir()
    elif mutation == "symlink": f.root.symlink_to(f.runtime, target_is_directory=True)
    elif mutation == "repo": monkeypatch.setattr(source, "REPO", f.tmp / "wrong-repo")
    else: monkeypatch.setattr(source, "ROOT_LIMIT", 1)
    with pytest.raises((ValueError, FileExistsError)): prepare(f)
    assert not f.calls


@pytest.mark.parametrize("mutation", ["size", "header", "payload", "symlink", "ancestor", "hardlink", "host", "boot", "nan"])
def test_retained_source_refusal_never_claims_valid(fixture, mutation, monkeypatch):
    f = fixture
    request = source._requests(f.inventory, 0)[0]
    plan = next(p for p in f.inventory.metadata.plans if p.device_slot == request["slot"])
    path = f.runtime / plan.filename
    if mutation == "size": path.write_bytes(path.read_bytes() + b"x")
    elif mutation == "header": path.write_bytes(b"X" + path.read_bytes()[1:])
    elif mutation == "payload":
        raw = bytearray(path.read_bytes()); raw[request["offset"]] ^= 1; path.write_bytes(raw)
    elif mutation == "symlink":
        other = f.runtime / "other"; path.rename(other); path.symlink_to(other)
    elif mutation == "ancestor":
        link = f.tmp / "runtime-link"; link.symlink_to(f.runtime, target_is_directory=True)
        f.inventory.metadata.root = link
    elif mutation == "hardlink": os.link(path, f.runtime / "hardlink")
    elif mutation == "host":
        identity = source._identity(f.inventory, f.records)
        monkeypatch.setattr(source.socket, "gethostname", lambda: "wrong-host")
        with pytest.raises(ValueError, match="hostname/boot"):
            source._capture(f.inventory, identity, 0)
        return
    elif mutation == "boot": f.records[0]["boot_id"] = "0" * 8 + "-0000-0000-0000-" + "0" * 12
    else:
        raw = bytearray(path.read_bytes()); raw[request["offset"]] = 0x7f; path.write_bytes(raw)
    with pytest.raises((ValueError, OSError)): report(f)


@pytest.mark.parametrize("mutation", ["replace", "inplace"])
def test_source_mutation_during_pread_is_detected(fixture, mutation, monkeypatch):
    f = fixture
    real = source.Snapshot.pread
    changed = []

    def pread(snapshot, length, offset):
        raw = real(snapshot, length, offset)
        if offset and not changed:
            changed.append(True)
            if mutation == "replace":
                replacement = snapshot.path.with_suffix(".replacement")
                replacement.write_bytes(snapshot.path.read_bytes())
                replacement.replace(snapshot.path)
            else:
                with snapshot.path.open("r+b") as stream:
                    stream.seek(offset); stream.write(b"\x00")
        return raw

    monkeypatch.setattr(source.Snapshot, "pread", pread)
    with pytest.raises(ValueError, match="changed"): report(f)


@pytest.mark.parametrize("mutation", ["rank", "request", "stat", "source", "encoded_short", "encoded_invalid", "sha", "extra"])
def test_returned_reply_is_independently_bound(fixture, mutation):
    f = fixture
    value = report(f)
    if mutation == "rank": value["rank"] = False
    elif mutation == "request": value["requests"][0]["offset"] += 1
    elif mutation == "stat": value["sources"][0]["stat_after"]["st_ino"] += 1
    elif mutation == "source": value["sources"][0]["header_sha256"] = "0" * 64
    elif mutation == "encoded_short": value["payload_base64"] = ""
    elif mutation == "encoded_invalid": value["payload_base64"] = "!" * len(value["payload_base64"])
    elif mutation == "sha":
        raw = bytearray(base64.b64decode(value["payload_base64"])); raw[0] ^= 1
        value["payload_base64"] = base64.b64encode(raw).decode()
    else: value["generation"] = "made-up"
    with pytest.raises(ValueError): source._decode(f.inventory, source._identity(f.inventory, f.records), 0, value)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink", "manifest_size", "identity", "request", "sources", "descriptor", "sha", "shape", "dtype", "tile_sha", "inventory", "expanded"])
def test_recovery_revalidates_all_originals_without_network(fixture, mutation, monkeypatch):
    f = fixture
    prepare(f)
    monkeypatch.setattr(source, "_fetch", forbidden)
    manifest = json.loads((f.root / "manifest.json").read_bytes())
    path = f.root / "rank0.npz"
    if mutation == "missing": path.unlink()
    elif mutation == "extra": (f.root / "unknown").write_bytes(b"x")
    elif mutation == "symlink":
        other = f.tmp / "capsule"; path.rename(other); path.symlink_to(other)
    elif mutation == "manifest_size": monkeypatch.setattr(source, "MANIFEST_LIMIT", 1)
    elif mutation == "identity": manifest["identity"]["code_hash"] = "0" * 40
    elif mutation == "request": manifest["requests"][0]["sha256"] = "0" * 64
    elif mutation == "sources": manifest["originals"][0]["sources"][0]["stat_after"]["st_size"] += 1
    elif mutation == "descriptor": manifest["originals"][0]["rank"] = False
    elif mutation == "sha": path.write_bytes(b"wrong")
    else:
        with np.load(path, allow_pickle=False) as saved: arrays = dict(saved)
        key = next(iter(arrays))
        if mutation == "shape": arrays[key] = arrays[key].reshape(-1)
        elif mutation == "dtype": arrays[key] = arrays[key].astype(np.int32)
        elif mutation == "tile_sha": arrays[key] = arrays[key] ^ np.uint8(1)
        elif mutation == "inventory": arrays["extra"] = arrays.pop(key)
        else: arrays[key] = np.zeros((100_000,), np.uint8)
        stream = BytesIO(); np.savez_compressed(stream, **arrays); raw = stream.getvalue()
        path.write_bytes(raw)
        manifest["originals"][0].update(bytes=len(raw), npz_sha256=sha256(raw).hexdigest(),
            raw_array_bytes=sum(v.nbytes for v in arrays.values()))
    save_manifest(f, manifest)
    with pytest.raises(ValueError): source.load_reader(root=f.root, repo=REPO, records=f.records)


def test_late_failure_preserves_bounded_earlier_originals(fixture):
    f = fixture
    fetch = f.fetch

    def failure(command, *, output, worker, timeout):
        if worker == "3": raise RuntimeError("fixture late source failure")
        fetch(command, output=output, worker=worker, timeout=timeout)

    f.fetch = failure
    with pytest.raises(RuntimeError, match="late source failure"): prepare(f)
    assert {p.name for p in f.root.iterdir()} == {"rank0.npz", "rank1.npz", "rank2.npz"}
    assert sum(p.stat().st_size for p in f.root.iterdir()) < source.ROOT_LIMIT
    with pytest.raises(ValueError, match="inventory"):
        source.load_reader(root=f.root, repo=REPO, records=f.records)
    with pytest.raises(FileExistsError): prepare(f)


def test_shared_source_budget_refuses_before_capsule_write(fixture, monkeypatch):
    f = fixture
    monkeypatch.setitem(storage.LIMITS, "sources", 1)
    with pytest.raises(ValueError, match="write budget"): prepare(f)
    assert list(f.root.iterdir()) == []


@pytest.mark.parametrize("kind", ["none", "twice", "noise", "total"])
def test_wire_parser_refuses_unbounded_or_ambiguous_output(kind):
    raw = source.MARKER.encode() + b"{}\n"
    cap = source.NOISE_LIMIT + 100
    if kind == "none": raw = b"SSH failed\n"
    elif kind == "twice": raw *= 2
    elif kind == "noise": raw += b"x" * (source.NOISE_LIMIT + 1)
    else: cap = len(raw) - 1
    with pytest.raises(ValueError): source._report(raw, cap)


def test_actual_forked_fetch_is_capped_and_uses_one_owned_tmpfs_file(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "TMP_ROOT", tmp_path)

    def fetch(command, *, output, worker, timeout):
        assert worker == "4" and command == "fixture-only"
        output.write_bytes(b"small reply")

    assert source._fetch(fetch, "fixture-only", 4, 64) == b"small reply"
    assert list(tmp_path.iterdir()) == []

    def oversized(command, *, output, worker, timeout):
        output.write_bytes(b"x" * 1024)

    with pytest.raises(ValueError, match="SSH failed"):
        source._fetch(oversized, "fixture-only", 4, 64)
    assert list(tmp_path.iterdir()) == []


def test_failed_fetch_kills_only_its_owned_cpu_descendant(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "TMP_ROOT", tmp_path)
    identity_path = tmp_path / "fixture-child.json"

    def fetch(command, *, output, worker, timeout):
        child = subprocess.Popen([source.PYTHON, "-c", "import time; time.sleep(30)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        identity_path.write_text(json.dumps(dict(pid=child.pid, group=os.getpgrp())))
        raise RuntimeError("controlled fetch failure after spawning a CPU child")

    with pytest.raises(ValueError, match="SSH failed"):
        source._fetch(fetch, "fixture-only", 0, 1024)
    child = json.loads(identity_path.read_text())
    path = Path(f"/proc/{child['pid']}/stat")

    def live():
        try:
            return path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    try:
        for _ in range(100):
            if not live(): break
            time.sleep(0.01)
        assert not live(), "failed bounded fetch left its CPU descendant alive"
        assert list(tmp_path.iterdir()) == [identity_path]
    finally:
        if live() and os.getpgid(child["pid"]) == child["group"]:
            os.killpg(child["group"], signal.SIGKILL)


def test_capsule_replacement_during_existing_reader_is_detected(fixture, monkeypatch):
    f = fixture
    prepare(f)
    real = source.read_npz

    def read(path, report, **kwargs):
        result = real(path, report, **kwargs)
        replacement = f.tmp / "replacement.npz"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return result

    monkeypatch.setattr(source, "read_npz", read)
    with pytest.raises(ValueError, match="changed"):
        source.load_reader(root=f.root, repo=REPO, records=f.records)


@pytest.mark.parametrize("kind", ["off", "platform", "rank", "identity", "pin", "hosts", "context", "boot"])
def test_remote_cli_refuses_before_any_selected_read(fixture, kind, monkeypatch):
    f = fixture
    identity = source._identity(f.inventory, f.records)
    args = ["--serve-selected-source", "--rank", "0"]
    if kind == "off": args = []
    elif kind == "platform": monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    elif kind == "rank": args[-1] = "8"
    elif kind == "identity": identity["tag"] = "unregistered"
    elif kind == "pin": identity["code_hash"] = "0" * 40
    elif kind == "hosts": identity["hosts"][0]["slots"] = [0, 1, 2, 3]
    elif kind == "context": identity["context"]["mesh_sha256"] = "0" * 64
    else: identity["hosts"][0]["boot_id"] = "bad"
    args += ["--identity-base64", base64.b64encode(source._json(identity)).decode()]
    monkeypatch.setattr(source, "_capture", forbidden)
    with pytest.raises(ValueError): source.main(args)


def test_remote_cli_reads_only_its_authenticated_host(fixture, monkeypatch, capsys):
    f = fixture
    identity = source._identity(f.inventory, f.records)
    monkeypatch.setattr(source.socket, "gethostname", lambda: f.inventory.captures[4]["hostname"])
    assert source.main(["--serve-selected-source", "--rank", "4", "--identity-base64",
                        base64.b64encode(source._json(identity)).decode()]) == 0
    value = source._report(capsys.readouterr().out.encode(), source._wire_cap(source._requests(f.inventory, 4)))
    arrays = source._decode(f.inventory, identity, 4, value)
    assert set(arrays) == {row["key"] for row in source._requests(f.inventory, 4)}


def test_actual_retained_metadata_derives_exact_fixed_requests_without_payload(monkeypatch):
    # Authentication reads retained small metadata only; any owner-file open or
    # backend/network use is trapped. No synthetic geometry assertion replaces
    # this actual 256-tile / 99,639,040-byte inventory test.
    monkeypatch.setattr(source, "Snapshot", forbidden)
    inventory = source._inventory(REPO)
    assert len(inventory.requests) == 256
    assert sum(row["bytes"] for row in inventory.requests) == 99_639_040
    assert [len(source._requests(inventory, rank)) for rank in range(8)] == [16, 16, 48, 32, 64, 32, 16, 32]
    assert [sum(row["bytes"] for row in source._requests(inventory, rank)) for rank in range(8)] == [
        196608, 196608, 16977920, 16322496, 33103808, 16322496, 196608, 16322496]
    assert sum(source._capsule_cap(source._requests(inventory, rank)) for rank in range(8)) + source.MANIFEST_LIMIT < source.ROOT_LIMIT
    assert max(source._wire_cap(source._requests(inventory, rank)) for rank in range(8)) < 44 << 20
    assert all("indexer.wk." not in row["tensor"] for row in inventory.requests)
    assert inventory.slots[0] == {12: 9, 14: 13, 13: 25, 15: 29}
