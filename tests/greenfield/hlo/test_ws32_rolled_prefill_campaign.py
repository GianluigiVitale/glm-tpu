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


def test_actual_compiler_cli_publication_fleet_and_database(
    lifecycle, monkeypatch, capsys, tmp_path
):
    case = lifecycle
    case.root = tmp_path / TAG
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
            for name in evidence.FILES:
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

    bucket = SimpleNamespace(
        get_blob=lambda name: blobs.get(name), blob=lambda name, generation: blobs[name]
    )

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
        campaign.publish_rank(TAG, rank)
    result = campaign.collect(TAG, PIN)
    assert all(name.endswith("worker_receipts.json") for name in downloads[:8])
    assert len(blobs) == 8 * (len(evidence.FILES) + 1)
    campaign.validate_record(result, PIN, root)
    assert result["compile_only"] and result["compiler_acquisition_complete"]
    assert not result["numerical_claim"] and not result["performance_claim"]
    assert result["comparison"]["passed"] is None and result["latency"] is None
    assert (
        result["hlo"]["contract"]["scope"]
        == "COMPILER_ORIGINALS_ONLY_NOT_MODEL_HLO_ADMISSION"
    )
    assert len(result["compiler_reports"]) == 8
    assert "dsa_wall" not in result and "overhead_wall" not in result
    for change in ("owner", "rank", "phase", "kernel", "count", "claim"):
        bad = deepcopy(result)
        if change == "owner":
            bad["workers"][0]["local_device_slots"][0]["device_slot"] = 31
        elif change == "rank":
            bad["workers"][0]["jax_process_index"] = 0
        elif change == "phase":
            del bad["workers"][0]["acquisition_phases"]["rolled_compile_terminal"]
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
        ["accounting", str(root), PIN, str(db), str(Path.cwd()), "1", worker.KERNEL],
    )
    exec(
        compile(accounting, "<actual-compiler-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            ("rolled_b128_b114_metadata_two_graphs_zero_calls_v1", None, None, None)
        ]
    assert (
        json.loads((root / "summary.json").read_text())["claim_scope"] == evidence.NOTE
    )

    # Oversized generation-resolved inventory must refuse BEFORE graph downloads.
    downloads.clear()
    ledger_name = f"results/{TAG}/workers/rank0/worker_receipts.json"
    ledger = json.loads(blobs[ledger_name].data)
    ledger[0]["size"] = campaign.rank_byte_limit(TAG) + 1
    blobs[ledger_name] = Blob(ledger_name, json.dumps(ledger).encode())
    with pytest.raises(ValueError, match="inventory/size"):
        campaign.collect(TAG, PIN)
    assert downloads == [ledger_name]


@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "manifest", "inventory", "pin"]
)
def test_metadata_preflight_requires_all_captured_hosts(tmp_path, monkeypatch, failure):
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
        output.write_text("\n".join(" ".join(row) for row in rows) + "\n")

    monkeypatch.setattr(campaign, "ssh", ssh)
    if failure:
        with pytest.raises(ValueError, match="eight-host"):
            campaign.metadata_preflight(tmp_path, PIN)
    else:
        campaign.metadata_preflight(tmp_path, PIN)


def test_campaign_cannot_launch_when_metadata_preflight_fails(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    monkeypatch.setattr(
        campaign, "deploy_existing_workers", lambda root, pin: events.append("deploy")
    )

    def refuse(root, pin):
        events.append("metadata")
        raise ValueError("missing metadata")

    monkeypatch.setattr(campaign, "metadata_preflight", refuse)
    monkeypatch.setattr(campaign, "ssh", lambda *a, **k: events.append("LAUNCH"))
    with pytest.raises(ValueError, match="missing metadata"):
        campaign.campaign(TAG, PIN)
    assert events == ["deploy", "metadata"]


def test_compile_mode_keeps_fixed_budget_and_no_sampling():
    assert campaign.evidence_files(TAG) == evidence.FILES
    assert len(evidence.FILES) == 7
    assert campaign.rank_byte_limit(TAG) == 256 << 20
    assert campaign.program_names(TAG) == worker.PROGRAMS
    command = campaign.launch_command(TAG, PIN, "10.0.0.1:8476")
    assert "timeout --kill-after=30s 900s" in command
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert (
        "[[ $KERNEL != ws32_prefill_rolled_model_compile ]] || ROLLED_COMPILE=1"
        in wrapper
    )
    assert "if [[ $GROUPED_ADMISSION == 1 || $ROLLED_COMPILE == 1 ]]; then" in wrapper
