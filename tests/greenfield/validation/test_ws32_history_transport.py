"""Actual bounded publication/download helpers, in-memory cloud and fixture replay."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

from google.api_core.exceptions import PreconditionFailed
import pytest

from scripts.greenfield import ws32_history_transport as transport
from scripts.greenfield.ws32_history_execution import call_schedule
from tests.greenfield.validation.test_ws32_dense_frontier_transport import Bucket

TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T200000000000000Z"
PIN = "a" * 40


def payloads(rank, *, difference=True):
    result = {name: name.encode() for name in transport.FILES}
    entries = []
    for i, (phase, graph) in enumerate(call_schedule()):
        relative = f"call_records/call{i:03d}.json"
        raw = json.dumps(dict(phase=phase, graph=graph, completed=True)).encode()
        result[relative] = raw
        entries.append(dict(phase=phase, graph=graph, completed=True,
            original=dict(schema=transport.calls.SCHEMA, path=relative, index=i,
                          bytes=len(raw), sha256=sha256(raw).hexdigest())))
    result["runner.json"] = json.dumps(dict(tag=TAG, code_hash=PIN, launch_rank=rank,
        protocol=transport.protocol.PROTOCOL, profile=transport.admission.PROFILE,
        kernel=transport.protocol.KERNEL, call_evidence_layout=transport.calls.SCHEMA,
        call_evidence=entries, programs={name: dict(optimized_hlo_sha256="b" * 64)
                                       for name in transport.protocol.PROGRAMS})).encode()
    if difference:
        result["first_difference_group63.npz"] = b"fixture original difference"
    return result


def write(root, files):
    root.mkdir(parents=True)
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    base = tmp_path_factory.mktemp("history-publication")
    bucket = Bucket()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transport, "RUN_ROOT", base)
        for rank in range(8):
            root = base / TAG / f"rank{rank}"
            write(root, payloads(rank, difference=rank % 2 == 0))
            transport.publish_rank(tag=TAG, rank=rank, root=root, client=bucket.client())
    return bucket.objects


@pytest.fixture
def fixture(tmp_path, monkeypatch, published):
    monkeypatch.setattr(transport, "RUN_ROOT", tmp_path)
    bucket = Bucket()
    bucket.objects = deepcopy(published)
    root = tmp_path / TAG
    local = root / "rank0"
    write(local, payloads(0))
    destination = root / "fleet"
    replay_calls = []

    def validate_fleet(**kwargs):
        assert set(kwargs) == {"root", "records", "repo", "original_root"}
        assert kwargs["root"] == destination
        assert [r["launch_rank"] for r in kwargs["records"]] == list(range(8))
        assert (destination / "rank7/call_records/call330.json").is_file()
        assert (destination / "rank7/materializers/exact_promote.json").is_file()
        assert not (destination / "rank7/call330.json").exists()
        replay_calls.append("replay")
        return dict(reproduced=True, numerical_promotion=False, performance_claim=False)
    arguments = dict(tag=TAG, pin=PIN, root=destination, repo=tmp_path,
        original_root=root / "original_references", client=bucket.client(),
        local_rank0_root=local, validate_fleet=validate_fleet)
    return NS(bucket=bucket, root=root, local=local, destination=destination,
              arguments=arguments, replay_calls=replay_calls, validator=validate_fleet)


def test_complete_inventory_caps_and_originals_not_republished(fixture):
    assert len(transport.FILES) == 366 and len(transport.CALL_FILES) == 331
    assert sum(transport.LIMITS.values()) == transport.MAX_RANK_BYTES == 512 << 20
    assert transport.MAX_LOCAL_RANK_BYTES == 544 << 20
    assert transport.MAX_ARCHIVE_BYTES == 10 << 30
    retained = fixture.local / "retained_reference"
    retained.mkdir()
    (retained / "candidate.json").write_text("old original")
    receipts = transport.publish_rank(tag=TAG, rank=0, root=fixture.local, client=fixture.bucket.client())
    assert len(receipts) == 367
    assert all("retained_reference" not in r["name"] for r in receipts)
    assert (retained / "candidate.json").read_text() == "old original"
    before = deepcopy(fixture.bucket.objects)
    assert receipts == transport.publish_rank(tag=TAG, rank=0, root=fixture.local, client=fixture.bucket.client())
    assert fixture.bucket.objects == before


def test_collect_nested_originals_exact_local_rank0_and_recovery(fixture):
    result = transport.collect(**fixture.arguments)
    assert fixture.replay_calls == ["replay"]
    assert result["comparison"]["passed"] is None and result["latency"] is None
    assert result["iterations"] == 0 and result["numerical_promotion"] is False
    assert result["performance_claim"] is False
    assert result["original_bytes"] == sum(len(raw) for _, raw in fixture.bucket.objects.values())
    for relative in (*transport.FILES, "first_difference_group63.npz"):
        left, right = fixture.local / relative, fixture.destination / "rank0" / relative
        assert os.path.samefile(left, right)
    downloads = [name for action, name in fixture.bucket.events if action == "download"]
    assert all(name.endswith("worker_receipts.json") for name in downloads[:8])
    assert not any("/workers/rank0/" in name and not name.endswith("worker_receipts.json") for name in downloads)
    assert len(result["workers"][0]["call_evidence"]) == 331
    args = {k: v for k, v in fixture.arguments.items() if k != "local_rank0_root"}
    fixture.bucket.events.clear()
    assert transport.replay_collected(**args) == result
    assert len([v for v in fixture.bucket.events if v[0] == "download"]) == 8
    assert not any(v[0] == "upload" for v in fixture.bucket.events)
    local_args = {k: v for k, v in args.items() if k not in ("client", "tag", "pin")}
    transport.validate_record(result, PIN, **local_args)
    assert fixture.replay_calls == ["replay"] * 3
    changed = deepcopy(result)
    changed["numerical_promotion"] = True
    with pytest.raises(ValueError, match="aggregate"):
        transport.validate_record(changed, PIN, **local_args)
    with pytest.raises(FileExistsError):
        transport.collect(**fixture.arguments)


def mutate_ledger(fixture, mutation, *, rank=7):
    key = f"results/{TAG}/workers/rank{rank}/worker_receipts.json"
    generation, raw = fixture.bucket.objects[key]
    rows = json.loads(raw)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[0] = rows[1]
    elif mutation == "escape":
        rows[0]["name"] = f"results/{TAG}/workers/rank{rank}/call_records/../runner.json"
    elif mutation == "flatten":
        next(r for r in rows if r["name"].endswith("call330.json"))["name"] = f"results/{TAG}/workers/rank{rank}/call330.json"
    elif mutation == "two_differences":
        extra = deepcopy(rows[-1])
        extra["name"] = f"results/{TAG}/workers/rank{rank}/first_difference_group0.npz"
        rows.append(extra)
        extra = deepcopy(extra)
        extra["name"] = f"results/{TAG}/workers/rank{rank}/first_difference_group1.npz"
        rows.append(extra)
    elif mutation == "refused":
        extra = deepcopy(rows[-1])
        extra["name"] = f"results/{TAG}/workers/rank{rank}/materializers/refused_exact.npz"
        rows.append(extra)
    elif mutation in ("size", "generation", "sha", "crc"):
        key = {"sha": "original_sha256", "crc": "crc32c"}.get(mutation, mutation)
        rows[0][key] = {"size": True, "generation": "0", "original_sha256": "a", "crc32c": "invalid"}[key]
    elif mutation == "call_cap":
        next(r for r in rows if r["name"].endswith("call330.json"))["size"] = (2 << 20) + 1
    elif mutation == "calls_total":
        for row in rows:
            if "/call_records/" in row["name"]:
                row["size"] = 1 << 20
    elif mutation == "wk_total":
        for row in rows:
            if "_wk_" in row["name"]:
                row["size"] = 15 << 20
    elif mutation == "exact_cap":
        next(r for r in rows if r["name"].endswith("exact_decode.json"))["size"] = (1 << 20) + 1
    elif mutation == "hlo_total":
        for row in rows:
            if row["name"].endswith("optimized_hlo.txt"):
                row["size"] = 8 << 20
    elif mutation == "metadata_cap":
        rows[0]["size"] = (6 << 20) + 1
    else:
        next(r for r in rows if r["name"].endswith("observer_control.npz"))["size"] = 128 << 20
    fixture.bucket.objects[f"results/{TAG}/workers/rank{rank}/worker_receipts.json"] = (generation, json.dumps(rows).encode())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "escape", "flatten", "two_differences", "refused",
    "size", "generation", "sha", "crc", "call_cap", "calls_total", "wk_total", "exact_cap", "hlo_total", "metadata_cap", "history_total"])
def test_last_ledger_caps_and_inventory_refuse_before_any_payload(fixture, mutation):
    mutate_ledger(fixture, mutation)
    with pytest.raises(ValueError):
        transport.collect(**fixture.arguments)
    assert not fixture.destination.exists() and not fixture.replay_calls
    assert all(name.endswith("worker_receipts.json") for action, name in fixture.bucket.events if action == "download")


@pytest.mark.parametrize("mutation", ["remote_generation", "remote_bytes", "local_bytes", "local_symlink", "nested_symlink"])
def test_rank0_reuse_reauthenticates_before_creating_destination(fixture, mutation):
    relative = "materializers/exact_promote.json"
    remote = f"results/{TAG}/workers/rank0/{relative}"
    generation, raw = fixture.bucket.objects[remote]
    if mutation == "remote_generation":
        fixture.bucket.objects[remote] = (generation + 1, raw)
    elif mutation == "remote_bytes":
        fixture.bucket.objects[remote] = (generation, b"x" * len(raw))
    elif mutation == "local_bytes":
        (fixture.local / relative).write_bytes(b"tampered")
    elif mutation == "local_symlink":
        path = fixture.local / relative
        path.rename(path.with_suffix(".original"))
        path.symlink_to(path.with_suffix(".original"))
    else:
        path = fixture.local / "materializers"
        path.rename(fixture.local / "original_materializers")
        path.symlink_to(fixture.local / "original_materializers", target_is_directory=True)
    with pytest.raises((ValueError, PreconditionFailed)):
        transport.collect(**fixture.arguments)
    assert not fixture.destination.exists() and not fixture.replay_calls


def test_downloaded_payload_sha_failure_is_retained_before_replay(fixture):
    name = f"results/{TAG}/workers/rank1/worker_receipts.json"
    generation, raw = fixture.bucket.objects[name]
    rows = json.loads(raw)
    next(r for r in rows if r["name"].endswith("exact_promote.json"))["original_sha256"] = "0" * 64
    fixture.bucket.objects[name] = (generation, json.dumps(rows).encode())
    with pytest.raises((ValueError, SystemExit)):
        transport.collect(**fixture.arguments)
    assert (fixture.destination / "rank1/materializers/exact_promote.json").is_file()
    assert not fixture.replay_calls


def test_changed_rank0_at_link_is_rejected_after_preserving_copy(fixture, monkeypatch):
    link = transport._link_exact
    def changed(source, destination):
        if source.name == "exact_promote.json":
            source.write_bytes(b"changed after the immediate check")
        link(source, destination)
    monkeypatch.setattr(transport, "_link_exact", changed)
    with pytest.raises(SystemExit, match="identity drifted"):
        transport.collect(**fixture.arguments)
    assert (fixture.destination / "rank0/materializers/exact_promote.json").read_bytes() == b"changed after the immediate check"
    assert not fixture.replay_calls


@pytest.mark.parametrize("mutation", ["ledger", "source_generation", "payload", "generation"])
def test_recovery_revalidates_without_replacing_evidence(fixture, mutation):
    transport.collect(**fixture.arguments)
    rank = fixture.destination / "rank7"
    if mutation == "ledger":
        (rank / transport.LEDGER).write_text("[]")
    elif mutation == "source_generation":
        path = rank / "ledger_source.json"
        value = json.loads(path.read_bytes())
        value["generation"] = True
        path.write_text(json.dumps(value))
    elif mutation == "payload":
        (rank / "call_records/call330.json").write_text("tampered")
    else:
        name = f"results/{TAG}/workers/rank7/materializers/exact_promote.json"
        generation, raw = fixture.bucket.objects[name]
        fixture.bucket.objects[name] = (generation + 1, raw)
    fixture.bucket.events.clear()
    args = {k: v for k, v in fixture.arguments.items() if k != "local_rank0_root"}
    with pytest.raises((ValueError, PreconditionFailed)):
        transport.replay_collected(**args)
    assert fixture.replay_calls == ["replay"]
    assert not any(action == "upload" for action, _ in fixture.bucket.events)
    assert all(name.endswith(transport.LEDGER) for action, name in fixture.bucket.events if action == "download")


@pytest.mark.parametrize("failure", ["region", "space", "tag", "rank0_path", "collector_path"])
def test_scope_and_space_fail_closed(fixture, monkeypatch, failure):
    if failure == "region":
        fixture.bucket.location = "EU"
    elif failure == "space":
        monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=0))
    elif failure == "tag":
        fixture.arguments["tag"] = TAG.replace("history_frontier_l06", "history_frontier_compile")
    elif failure == "rank0_path":
        fixture.arguments["local_rank0_root"] = fixture.root
    else:
        fixture.arguments["root"] = fixture.root / "wrong"
    with pytest.raises(ValueError):
        transport.collect(**fixture.arguments)
    assert not fixture.destination.exists() and not fixture.replay_calls
    assert all(name.endswith(transport.LEDGER) for action, name in fixture.bucket.events if action == "download")


def test_space_budget_credits_only_authenticated_rank0_payloads(fixture, monkeypatch):
    total = sum(len(raw) for _, raw in fixture.bucket.objects.values())
    reused = sum(len(raw) for raw in payloads(0).values())
    exact = total - reused + 8 * transport.MAX_LEDGER_BYTES + transport.DISK_RESERVE
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=exact - 1))
    with pytest.raises(ValueError, match="headroom"):
        transport.collect(**fixture.arguments)
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=exact))
    transport.collect(**fixture.arguments)
    assert fixture.replay_calls == ["replay"]


def test_partial_publication_preserves_bounded_originals_and_discloses_omissions(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "RUN_ROOT", tmp_path)
    root = tmp_path / TAG / "rank0"
    root.mkdir(parents=True)
    (root / "observer_candidate.npz.pending").write_bytes(b"preserved pending capsule")
    (root / "unhealthy_step318.npz").write_bytes(b"preserved refused capsule")
    (root / "materializers").mkdir()
    (root / "materializers/refused_exact.npz").write_bytes(b"bounded exact offending pair")
    (root / "worker.log").write_bytes(b"oversized fixture")
    monkeypatch.setitem(transport.METADATA_LIMITS, "worker.log", 1)
    bucket = Bucket()
    receipts = transport.publish_rank(tag=TAG, rank=0, root=root, client=bucket.client())
    names = {r["name"].removeprefix(f"results/{TAG}/workers/rank0/") for r in receipts}
    assert names == {"observer_candidate.npz.pending", "unhealthy_step318.npz",
                     "materializers/refused_exact.npz", transport.NOTICE}
    assert (root / "worker.log").read_bytes() == b"oversized fixture"
    with pytest.raises(ValueError, match="inventory"):
        transport._receipts((root / transport.LEDGER).read_bytes(), tag=TAG, rank=0)


def test_publication_exception_keeps_every_local_original(fixture, monkeypatch):
    bucket, publish, count = Bucket(), transport.publish_exact, []
    def failing(*args, **kwargs):
        count.append(args[1])
        if len(count) == 3:
            raise OSError("fixture publication interrupted")
        return publish(*args, **kwargs)
    monkeypatch.setattr(transport, "publish_exact", failing)
    with pytest.raises(OSError):
        transport.publish_rank(tag=TAG, rank=0, root=fixture.local, client=bucket.client())
    for relative, original in payloads(0).items():
        assert (fixture.local / relative).read_bytes() == original
    assert len(bucket.objects) == 2
    assert not (fixture.local / transport.LEDGER).exists()


@pytest.mark.parametrize("field", ["tag", "launch_rank", "protocol", "profile", "kernel"])
def test_publisher_refuses_mixed_worker_identity_before_upload(fixture, field):
    path = fixture.local / "runner.json"
    value = json.loads(path.read_bytes())
    value[field] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        transport.publish_rank(tag=TAG, rank=0, root=fixture.local, client=fixture.bucket.client())
    assert not fixture.bucket.events


@pytest.mark.parametrize("verdict", [dict(reproduced=1, numerical_promotion=False, performance_claim=False),
    dict(reproduced=True, numerical_promotion=True, performance_claim=False),
    dict(reproduced=True, numerical_promotion=False), {}])
def test_aggregate_requires_literal_validated_reproduction(fixture, verdict):
    records = [json.loads(payloads(rank)["runner.json"]) for rank in range(8)]
    with pytest.raises(ValueError, match="reproduction"):
        transport._aggregate(records, verdict, tag=TAG, pin=PIN, original_bytes=1)


def test_aggregate_refuses_expanded_call_entries(fixture):
    records = [json.loads(payloads(rank)["runner.json"]) for rank in range(8)]
    records[7]["call_evidence"][0] = dict(completed=True, memory={"expanded": "census"})
    with pytest.raises(ValueError, match="expanded"):
        transport._aggregate(records, dict(reproduced=True, numerical_promotion=False,
            performance_claim=False), tag=TAG, pin=PIN, original_bytes=1)


def test_whole_archive_failure_uses_same_paths_excludes_old_originals(fixture):
    # No collection is needed to establish the controller naming/no-duplicate rule.
    old = fixture.local / "retained_reference"
    old.mkdir()
    (old / "candidate.json").write_text("old generation")
    (fixture.root / "orchestrator.log").write_text("FAILED after normal archive")
    files = transport.archive_inventory(fixture.root)
    assert all("retained_reference" not in p.parts for p in files)
    transport.publish_controller_failure(fixture.root, client=fixture.bucket.client())
    assert f"results/{TAG}/rank0/call_records/call330.json" in fixture.bucket.objects
    assert f"results/{TAG}/failure_orchestrator.log" in fixture.bucket.objects
    assert not any("/diagnostic/" in name or "/retained_reference/" in name for name in fixture.bucket.objects)
    before = deepcopy(fixture.bucket.objects)
    transport.publish_controller_failure(fixture.root, client=fixture.bucket.client())
    assert fixture.bucket.objects == before


def test_archive_preserves_only_registered_partial_and_atomic_metadata_paths(fixture):
    partial = fixture.local / "call_records/call330.json.partial"
    partial.write_bytes(b"bounded incomplete download")
    temporary = fixture.local / ".runner.json.tmp.12345"
    temporary.write_text("bounded incomplete record")
    files = transport.archive_inventory(fixture.root)
    assert partial in files and temporary in files
    unexpected = fixture.local / "call_records/unknown.json.partial"
    unexpected.write_text("not a registered call")
    with pytest.raises(ValueError, match="unregistered"):
        transport.archive_inventory(fixture.root)


def test_controller_ledger_cap_checked_before_any_upload(fixture, monkeypatch):
    (fixture.root / "archive_receipts.json").write_text("[]")
    monkeypatch.setattr(transport, "MAX_CONTROLLER_LEDGER_BYTES", 1)
    with pytest.raises(ValueError, match="ledger budget"):
        transport.publish_controller_failure(fixture.root, client=fixture.bucket.client())
    assert not any(action == "upload" for action, _ in fixture.bucket.events)


@pytest.mark.parametrize("failure", ["bytes", "files", "symlink", "metadata", "reference", "extra"])
def test_archive_caps_refuse_without_upload_or_deletion(fixture, monkeypatch, failure):
    if failure == "bytes":
        monkeypatch.setattr(transport, "MAX_ARCHIVE_BYTES", transport.MAX_FLEET_BYTES + transport.CONTROLLER_METADATA_RESERVE)
    elif failure == "files":
        monkeypatch.setattr(transport, "MAX_ARCHIVE_FILES", 1)
    elif failure == "symlink":
        (fixture.root / "linked").symlink_to(fixture.local)
    elif failure == "metadata":
        monkeypatch.setitem(transport.METADATA_LIMITS, "worker.log", 1)
    elif failure == "extra":
        (fixture.root / "summary.json").write_text("bounded fixture")
        monkeypatch.setattr(transport, "MAX_CONTROLLER_EXTRA_BYTES", transport.CONTROLLER_METADATA_RESERVE)
    else:
        old = fixture.local / "retained_reference"
        old.mkdir()
        (old / "candidate.json").write_text("preserved")
        monkeypatch.setattr(transport, "MAX_REFERENCE_BYTES", 1)
    with pytest.raises(ValueError):
        transport.publish_controller_failure(fixture.root, client=fixture.bucket.client())
    assert not any(action == "upload" for action, _ in fixture.bucket.events)
    assert (fixture.local / "runner.json").is_file()
