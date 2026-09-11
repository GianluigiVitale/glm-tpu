"""Compile-only CLI→original publication→fleet replay→SQLite, fixture compiler."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield import ws32_rolled_prefill_evidence as evidence
from tests.greenfield.hlo.test_ws32_rolled_prefill_worker import (
    lifecycle,
    test_actual_probe_selects_compile_only_before_runtime as _run_entry,
)

PIN = "c" * 40
TAG = "greenfield_fp8_" + worker.KERNEL + "_fixture"


@pytest.mark.parametrize("lifecycle", [False, True, "full", "history"], indirect=True)
def test_actual_compiler_cli_publication_fleet_and_database(
    lifecycle, monkeypatch, capsys, tmp_path
):
    case = lifecycle
    tag = "greenfield_fp8_" + case.mode.kernel + "_fixture"
    files = evidence.files(
        case.canonical_dense, full_canonical=case.full_canonical, history=case.history
    )
    case.root = tmp_path / tag
    case.root.mkdir()
    # Reuse the already-tested actual CLI fixture: only runtime/compiler are
    # substituted. It exercises the actual metadata-only continuation and
    # local original replay, never a replacement worker implementation.
    _run_entry(case, monkeypatch, False)
    root = case.root
    original_log = capsys.readouterr().out
    template = json.loads((root / "rank4/runner.json").read_text())
    physical, captures = campaign.topology_bindings()
    device_slots = {d: i for i, d in enumerate(physical.flattened_device_ids)}
    for rank, capture in enumerate(captures):
        destination = root / f"rank{rank}"
        if rank == 4:
            record = template
        else:
            destination.mkdir()
            record = deepcopy(template)
            record.update(
                launch_rank=rank,
                hostname=capture["hostname"],
                jax_process_index=capture["jax_process_index"],
                pid=10000 + rank,
                local_device_slots=[
                    dict(device_id=d, device_slot=device_slots[d])
                    for d in capture["local_device_ids"]
                ],
            )
            rows = [
                json.loads(line)
                for line in (root / "rank4/compile_journal.jsonl")
                .read_text()
                .splitlines()
            ]
            rows[0]["identity"]["launch_rank"] = rank
            for row in rows:
                if row["stage"] == "compiled":
                    row["device_memory"] = [
                        dict(device_id=d, stats={"bytes_in_use": 0})
                        for d in capture["local_device_ids"]
                    ]
            data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
            (destination / "compile_journal.jsonl").write_bytes(data)
            record["compile_journal_sha256"] = sha256(data).hexdigest()
            for name in files:
                if name.endswith((".mlir", ".txt")):
                    os.link(root / "rank4" / name, destination / name)
        (destination / "runner.json").write_text(json.dumps(record))
        (destination / "worker.log").write_text(
            original_log
            if rank == 4
            else "SYNTHETIC CPU fixture peer, NOT TPU evidence\n"
        )

    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication

    blobs, downloads = {}, []

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, data):
            self.name, self.data, self.size = name, data, len(data)

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            downloads.append(self.name)
            return self.data

    def generation_blob(name, generation):
        blob = blobs[name]
        if int(generation) != blob.generation:
            raise ValueError("fixture exact cloud generation is unavailable")
        return blob

    bucket = SimpleNamespace(get_blob=lambda name: blobs.get(name), blob=generation_blob)

    def named_bucket(name):
        assert name == "driftbench-dsv4-uc"
        return bucket

    def publish(bucket, name, path, digest, *, compressed):
        assert not compressed
        blob = Blob(name, path.read_bytes())
        blobs[name] = blob
        return dict(
            name=name,
            generation=str(blob.generation),
            size=blob.size,
            crc32c=blob.crc32c,
            original_sha256=sha256(blob.data).hexdigest(),
        )

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=named_bucket))
    monkeypatch.setattr(publication, "publish_exact", publish)
    monkeypatch.setattr(campaign, "run_root", lambda tag: root)
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda path: SimpleNamespace(free=8 << 30)
    )
    for rank in range(8):
        campaign.publish_rank(tag, rank)
    result = campaign.collect(tag, PIN)
    assert all(name.endswith("worker_receipts.json") for name in downloads[:8])
    assert len(blobs) == 8 * (len(files) + 1)
    campaign.validate_record(result, PIN, root)
    assert result["compile_only"] and result["compiler_acquisition_complete"]
    assert not result["numerical_claim"] and not result["performance_claim"]
    assert result["comparison"]["passed"] is None and result["latency"] is None
    assert (
        result["hlo"]["contract"]["scope"]
        == "COMPILER_ORIGINALS_ONLY_NOT_MODEL_HLO_ADMISSION"
    )
    assert len(result["compiler_reports"]) == 8
    assert {
        owner["device_slot"] for record in result["workers"]
        for owner in record["local_device_slots"]
    } == set(range(32))
    assert all(record["model_executable_calls"] == 0 for record in result["workers"])
    if case.history:
        assert len(files) == 17 and len(case.mode.programs) == 7
        assert all(len(record["acquisition_phases"]) == 12 for record in result["workers"])
    assert "dsa_wall" not in result and "overhead_wall" not in result
    for change in ("owner", "rank", "phase", "kernel", "count", "claim"):
        bad = deepcopy(result)
        if change == "owner":
            bad["workers"][0]["local_device_slots"][0]["device_slot"] = 31
        elif change == "rank":
            bad["workers"][0]["jax_process_index"] = 0
        elif change == "phase":
            del bad["workers"][0]["acquisition_phases"][case.mode.prefix + "_terminal"]
        elif change == "kernel":
            bad["workers"][0]["kernel"] = campaign.KERNEL
        elif change == "count":
            bad["workers"][0]["iterations"] = 5
        else:
            bad["numerical_claim"] = True
        with pytest.raises(ValueError):
            campaign.validate_record(bad, PIN, root)

    (root / "runner.json").write_text(json.dumps(result))
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    db = tmp_path / "compiler.db"
    monkeypatch.setattr(
        sys,
        "argv",
        ["accounting", str(root), PIN, str(db), str(Path.cwd()), "1", case.mode.kernel],
    )
    exec(
        compile(accounting, "<actual-compiler-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            (
                (
                    "history_l06_metadata_seven_graphs_zero_calls_v1"
                    if case.history
                    else (
                        "canonical_dense_b128_b114_metadata_two_graphs_zero_calls_v1"
                        if case.full_canonical
                        else (
                            "dense01_canonical_metadata_one_graph_zero_calls_v1"
                            if case.canonical_dense
                            else "rolled_b128_b114_metadata_two_graphs_zero_calls_v1"
                        )
                    )
                ),
                None,
                None,
                None,
            )
        ]
    assert json.loads((root / "summary.json").read_text())["claim_scope"] == (
        campaign.history_compile.NOTE if case.history else (
            campaign.full_compile.NOTE if case.full_canonical
            else campaign.dense_compile.NOTE if case.canonical_dense else evidence.NOTE
        )
    )

    # A generation declared in the ledger must select that exact object version.
    if case.history:
        downloads.clear()
        ledger_name = f"results/{tag}/workers/rank0/worker_receipts.json"
        saved = blobs[ledger_name]
        ledger = json.loads(saved.data)
        ledger[0]["generation"] = "24"
        blobs[ledger_name] = Blob(ledger_name, json.dumps(ledger).encode())
        with pytest.raises(ValueError, match="generation"):
            campaign.collect(tag, PIN)
        assert not any(name.endswith((".mlir", ".txt")) for name in downloads)
        blobs[ledger_name] = saved

    # Oversized generation-resolved inventory must refuse BEFORE graph downloads.
    downloads.clear()
    ledger_name = f"results/{tag}/workers/rank0/worker_receipts.json"
    ledger = json.loads(blobs[ledger_name].data)
    ledger[0]["size"] = campaign.rank_byte_limit(tag) + 1
    blobs[ledger_name] = Blob(ledger_name, json.dumps(ledger).encode())
    with pytest.raises(ValueError, match="inventory/size"):
        campaign.collect(tag, PIN)
    assert downloads == [ledger_name]


@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "manifest", "inventory", "pin"]
)
@pytest.mark.parametrize("variant", [False, True, "full", "history"])
def test_metadata_preflight_requires_all_captured_hosts(
    tmp_path, monkeypatch, failure, variant
):
    canonical_dense, full_canonical = variant is True, variant == "full"
    history = variant == "history"
    _, captures = campaign.topology_bindings()
    pins = json.loads(
        (
            campaign.REPO
            / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
        ).read_text()
    )
    rows = [
        [
            "ROLLED_METADATA_OK",
            c["hostname"],
            pins["expected_manifest_sha256"],
            pins["source_inventory_sha256"],
            PIN,
        ]
        for c in captures
    ]
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = rows[0]
    elif failure:
        rows[0][{"manifest": 2, "inventory": 3, "pin": 4}[failure]] = "a" * 64

    def ssh(command, *, output, timeout):
        assert "JAX_PLATFORMS=cpu" in command and "read_metadata" in command
        assert "probe_ws32_prefill_budget.py" not in command
        assert ("ws32_dense_canonical_compile" in command) == canonical_dense
        assert ("ws32_canonical_prefill_compile" in command) == full_canonical
        assert ("ws32_history_compile" in command) == history
        output.write_text("\n".join(" ".join(row) for row in rows) + "\n")

    monkeypatch.setattr(campaign, "ssh", ssh)
    if failure:
        with pytest.raises(ValueError, match="eight-host"):
            campaign.metadata_preflight(
                tmp_path,
                PIN,
                canonical_dense=canonical_dense,
                full_canonical=full_canonical,
                history=history,
            )
    else:
        campaign.metadata_preflight(
            tmp_path,
            PIN,
            canonical_dense=canonical_dense,
            full_canonical=full_canonical,
            history=history,
        )


@pytest.mark.parametrize("variant", [False, True, "full", "history"])
def test_campaign_cannot_launch_when_metadata_preflight_fails(
    tmp_path, monkeypatch, variant
):
    canonical_dense, full_canonical = variant is True, variant == "full"
    history = variant == "history"
    events = []
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    monkeypatch.setattr(
        campaign, "deploy_existing_workers", lambda root, pin: events.append("deploy")
    )

    def refuse(root, pin, **kwargs):
        assert kwargs == (
            {"history": True} if history else (
                {"full_canonical": True} if full_canonical
                else {"canonical_dense": True} if canonical_dense else {}
            )
        )
        events.append("metadata")
        raise ValueError("missing metadata")

    monkeypatch.setattr(campaign, "metadata_preflight", refuse)
    monkeypatch.setattr(campaign, "ssh", lambda *a, **k: events.append("LAUNCH"))
    with pytest.raises(ValueError, match="missing metadata"):
        tag = (
            "greenfield_fp8_ws32_history_frontier_compile_fixture"
            if history
            else (
                "greenfield_fp8_ws32_prefill_canonical_model_compile_fixture"
                if full_canonical
                else (
                    "greenfield_fp8_ws32_dense_canonical_compile_fixture"
                    if canonical_dense
                    else TAG
                )
            )
        )
        campaign.campaign(tag, PIN)
    assert events == ["deploy", "metadata"]


@pytest.mark.parametrize("variant", [False, "full", "history"])
def test_compile_mode_keeps_fixed_budget_and_no_sampling(variant):
    history, full_canonical = variant == "history", variant == "full"
    mode = worker.compile_mode(full_canonical=full_canonical, history=history)
    tag = "greenfield_fp8_" + mode.kernel + "_fixture"
    files = evidence.files(full_canonical=full_canonical, history=history)
    assert campaign.evidence_files(tag) == files
    assert len(files) == (17 if history else 7)
    assert campaign.rank_byte_limit(tag) == 256 << 20
    assert campaign.program_names(tag) == mode.programs
    command = campaign.launch_command(tag, PIN, "10.0.0.1:8476")
    assert "timeout --kill-after=30s 900s" in command
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert f"[[ $KERNEL != {mode.kernel} ]] || ROLLED_COMPILE=1" in wrapper
    assert "if [[ $GROUPED_ADMISSION == 1 || $ROLLED_COMPILE == 1 ]]; then" in wrapper


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_full_compiler_mode_is_strict_and_exclusive(invalid):
    with pytest.raises(ValueError, match="static bool"):
        worker.compile_mode(full_canonical=invalid)
    with pytest.raises(ValueError, match="exclusive"):
        worker.compile_mode(True, full_canonical=True)
    with pytest.raises(ValueError, match="static bool"):
        worker.compile_mode(history=invalid)
    for flags in (
        dict(canonical_dense=True, history=True),
        dict(full_canonical=True, history=True),
        dict(canonical_dense=True, full_canonical=True, history=True),
    ):
        with pytest.raises(ValueError, match="exclusive"):
            worker.compile_mode(**flags)


@pytest.mark.parametrize("lifecycle", ["full", "history"], indirect=True)
def test_full_compiler_cannot_inherit_old_identity_or_extra_graph(lifecycle):
    case = lifecycle
    case.run()
    for reduced in (False, True):
        with pytest.raises(ValueError, match="identity"):
            worker.journal_identity(case.record, canonical_dense=reduced)
    bad = deepcopy(case.record)
    if case.history:
        with pytest.raises(ValueError, match="identity"):
            worker.journal_identity(case.record, full_canonical=True)
    bad["programs"]["wk_decode"] = bad["programs"][case.mode.programs[0]]
    with pytest.raises(ValueError, match="registered/preserved graphs"):
        case.mode.preparation.validate_preserved_pair(
            case.root, bad, repo=campaign.REPO
        )
    for name in case.mode.programs:
        path = case.root / f"{name}.optimized_hlo.txt"
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="identity"):
            case.mode.preparation.validate_preserved_pair(
                case.root, case.record, repo=campaign.REPO
            )
        path.write_bytes(original)
