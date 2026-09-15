"""Actual bounded/compressed transport with fake GCS; no TPU or external writes."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu import user_request
from scripts.release import ws32_user_transport as transport
from tests.greenfield.validation.gcs_fixtures import Bucket
from tests.greenfield.validation.test_ws32_native_benchmark_transport import originals

TAG = "greenfield_ws32_user_request_20260914T020000000000000Z"
PIN = "a" * 40


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.worker, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(transport, "require_local_idle", lambda: None)
    monkeypatch.setattr(transport.outputs, "require_local_idle", lambda: None)
    monkeypatch.setattr(
        transport.cold.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 << 30)
    )
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    request = user_request.from_token_ids(
        [1, 2, 3], request_id="fixture", seed=7, max_new_tokens=2
    )
    raw = user_request.canonical(request) + b"\n"
    (root / "request.json").write_bytes(raw)
    digest = sha256(raw).hexdigest()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    destination = tmp_path / "collected"
    destination.mkdir(mode=0o700)
    bucket = Bucket()
    bucket.soft_delete_policy = SimpleNamespace(retention_duration_seconds=0)
    return SimpleNamespace(
        root=root,
        digest=digest,
        boot=boot,
        bucket=bucket,
        destination=destination,
        monkeypatch=monkeypatch,
        fleet=[],
    )


def publish(case, rank=0, *, with_cold=True):
    host = f"fixture-w-{rank}"
    case.monkeypatch.setattr("socket.gethostname", lambda: host)
    ended = dict(
        tag=TAG,
        code_hash=PIN,
        rank=rank,
        request_file_sha256=case.digest,
        host=host,
        boot_id=case.boot,
        worker_exit_code=0,
        supervisor_pid=1000 + rank,
    )
    transport.outputs._write_once(
        case.root / f"ended.rank{rank}.json", user_request.canonical(ended)
    )
    transport.outputs._write_once(case.root / f"runner.rank{rank}.log", b"")
    if rank == 0:
        transport.outputs._write_once(
            case.root / "sessions.rank0/item000/tokens.jsonl", b'{"token_id":7}\n'
        )
    if with_cold:
        originals(case.root, rank)
    case.fleet.append(dict(rank=rank, host=host, boot_id=case.boot, tag=TAG, pin=PIN))
    return transport.publish(
        root=case.root,
        tag=TAG,
        pin=PIN,
        rank=rank,
        request_file_sha256=case.digest,
        client=case.bucket.client(),
    )


def collect(case):
    return transport.collect(
        destination=case.destination,
        tag=TAG,
        pin=PIN,
        request_file_sha256=case.digest,
        original_fleet=case.fleet,
        client=case.bucket.client(),
        blobs={name: case.bucket.get_blob(name) for name in case.bucket.objects},
    )


def test_eight_rank_roundtrip_private_outputs_and_separate_schema(case):
    for rank in range(8):
        report = publish(case, rank)
        assert (
            report["benchmark"] is False and report["protected_result_sealed"] is False
        )
    before = deepcopy(case.bucket.objects)
    result = collect(case)
    assert len(result["requests"]) == len(result["cold"]) == 8
    assert result["missing_cold_ranks"] == []
    assert all(r["schema"] == transport.outputs.USER_SCHEMA for r in result["requests"])
    assert all(
        r["manifest"]["schema"] == transport.cold.USER_SCHEMA for r in result["cold"]
    )
    assert collect(case) == result and before == case.bucket.objects
    assert all(
        not name.startswith(f"results/{TAG}/native_requests/") for name in before
    )
    assert all(not name.endswith("/SUCCESS") for name in before)
    assert (
        case.destination / "sessions.rank0/item000/tokens.jsonl"
    ).stat().st_mode & 0o777 == 0o600
    assert all(path.stat().st_mode & 0o077 == 0 for path in case.destination.rglob("*"))
    paths = [
        case.destination / f"native.rank{r}/prefill_chunk.optimized_hlo.txt"
        for r in range(8)
    ]
    assert len({path.stat().st_ino for path in paths}) == 1


def test_partial_before_cold_load_remains_explicit_and_recoverable(case):
    for rank in range(8):
        assert publish(case, rank, with_cold=False)["cold_present"] is False
    result = collect(case)
    assert result["missing_cold_ranks"] == list(range(8))
    assert result["protected_result_sealed"] is False
    assert (
        case.destination / "sessions.rank0/item000/tokens.jsonl"
    ).read_bytes() == b'{"token_id":7}\n'


def test_cold_failure_does_not_hide_the_user_response(case):
    case.monkeypatch.setattr(
        transport.cold,
        "publish_rank",
        lambda **kw: (_ for _ in ()).throw(ValueError("cold fixture")),
    )
    with pytest.raises(RuntimeError, match="cold"):
        publish(case)
    assert (
        transport.outputs.prefix(TAG, 0, user_request=True) + "manifest.json"
        in case.bucket.objects
    )
    assert any("tokens.jsonl" in name for name in case.bucket.objects)


@pytest.mark.parametrize(
    "problem",
    [
        "generation",
        "schema",
        "owner",
        "request_hash",
        "extra",
        "region",
        "soft_delete",
        "permissions",
    ],
)
def test_collector_rejects_wrong_bindings_or_scope(case, problem):
    for rank in range(8):
        publish(case, rank, with_cold=False)
    if problem == "generation":
        name = next(name for name in case.bucket.objects if "tokens.jsonl" in name)
        generation, data = case.bucket.objects[name]
        case.bucket.objects[name] = generation + 1, data
    elif problem == "schema":
        name = transport.outputs.prefix(TAG, 0, user_request=True) + "manifest.json"
        generation, data = case.bucket.objects[name]
        value = json.loads(data)
        value["schema"] = transport.outputs.SCHEMA
        case.bucket.objects[name] = generation, user_request.canonical(value)
    elif problem == "owner":
        case.fleet[0]["boot_id"] = "different-boot"
    elif problem == "request_hash":
        case.digest = "b" * 64
    elif problem == "extra":
        case.bucket.objects[f"results/{TAG}/user_requests/unknown"] = 9999, b"extra"
    elif problem == "region":
        case.bucket.location = "EU"
    elif problem == "soft_delete":
        case.bucket.soft_delete_policy.retention_duration_seconds = 604800
    elif problem == "permissions":
        case.destination.chmod(0o755)
    with pytest.raises(ValueError):
        collect(case)
    assert (
        case.root / "sessions.rank0/item000/tokens.jsonl"
    ).read_bytes() == b'{"token_id":7}\n'


def test_explicit_user_mode_cannot_widen_benchmark_default_or_item_count(case):
    with pytest.raises(ValueError):
        transport.cold._identity(TAG, PIN, 0)
    with pytest.raises(ValueError):
        transport.cold._identity(TAG, PIN, 0, user_request=1)
    with pytest.raises(ValueError):
        transport.cold._identity(
            TAG.replace("user_request", "native_benchmark"), PIN, 0, user_request=True
        )
    with pytest.raises(ValueError):
        transport.outputs.limit(
            "sessions.rank0/item001/result.json.gz", 0, user_request=True
        )
    assert (
        transport.outputs.limit("sessions.rank0/item001/result.json.gz", 0) == 4 << 20
    )


def test_private_mask_is_restored_after_failure():
    previous = os.umask(0o022)
    try:
        with pytest.raises(RuntimeError):
            with transport.private_writes():
                raise RuntimeError("fixture")
        observed = os.umask(0o022)
        assert observed == 0o022
    finally:
        os.umask(previous)
