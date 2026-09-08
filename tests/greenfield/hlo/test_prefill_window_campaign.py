"""Actual eight-rank publication/replay/DB composition, CPU fixtures only."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.probe_ws32_prefill_moe import FLEET_SHA, MESH_SHA, TOPOLOGY_SHA
from tests.greenfield.hlo.test_prefill_window_evidence import completed_worker


@pytest.mark.parametrize(
    "stage",
    [
        "numerical_finalize",
        "close_compile_journal",
        "numerical_snapshot",
        "journal_close",
    ],
)
def test_final_publication_refusal_votes_before_success(tmp_path, stage):
    observed, votes = {}, []

    def configure(patches, record, sequence):
        observed.update(record=record, sequence=sequence, fired=False)
        original = acquisition._atomic_json

        def publish(path, value):
            if (
                not observed["fired"]
                and value.get("acquisition_phases", {}).get(stage, {}).get("status")
                == "RUNNING"
            ):
                observed["fired"] = True
                raise OSError("injected final publication failure")
            return original(path, value)

        patches.setattr(acquisition, "_atomic_json", publish)
        if stage == "journal_close":
            original_close = worker.WindowNumericalJournal.close

            def close(journal):
                original_close(journal)
                if not observed["fired"]:
                    observed["fired"] = True
                    raise OSError("injected journal close failure")

            patches.setattr(worker.WindowNumericalJournal, "close", close)

    with pytest.raises(OSError, match="injected"):
        with completed_worker(
            tmp_path, configure=configure, consensus=lambda ok: votes.append(ok) or ok
        ):
            pytest.fail("final publication failure must not reach success")
    assert observed["fired"] and False in votes
    assert len(observed["sequence"]) == 17  # No successor/retry dispatch.
    assert "terminal" not in observed["record"]["acquisition_phases"]
    assert all((tmp_path / f"{case}.npz").is_file() for case in protocol.CASES)


def test_peer_finalization_refusal_stops_before_terminal(tmp_path):
    observed, votes = {}, []

    def configure(patches, record, sequence):
        observed.update(record=record, sequence=sequence)

    def consensus(ok):
        votes.append(ok)
        return ok and "numerical_snapshot" not in observed["record"].get(
            "acquisition_phases", {}
        )

    with pytest.raises(RuntimeError, match="peer failed: numerical_snapshot"):
        with completed_worker(tmp_path, configure=configure, consensus=consensus):
            pytest.fail("peer refusal must not reach success")
    assert all(votes)  # This host succeeded; the peer's failure still stops it.
    assert len(observed["sequence"]) == 17


def test_actual_eight_worker_collector_and_db(tmp_path, monkeypatch):
    pin, pins = "a" * 40, {"synthetic_metadata_fixture_only": True}
    ledger = {
        i: dict(selected={"tensor": str(i)}, full_sha256="d" * 64) for i in range(32)
    }
    order = [(7 * i) % 32 for i in range(32)]
    processes = [3, 5, 1, 2, 0, 6, 7, 4]
    monkeypatch.setattr(campaign, "checkpoint_ledger", lambda layer: (pins, ledger))
    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_fixture"
    records = []
    for rank in range(8):
        root = tmp_path / f"rank{rank}"
        slots = {order[s]: s for s in range(rank * 4, rank * 4 + 4)}
        with completed_worker(
            root, rank=rank, slots=slots, process=processes[rank]
        ) as (_, record):
            record.update(
                status="SUCCESS",
                hostname=f"fixture-host{rank}",
                pid=100 + rank,
                start_ticks=1234,
                boot_id=f"fixture-boot{rank}",
                layer=6,
                selected_layer_ids=[6],
                rows=128,
                control_rows=32,
                context_capacity=4096,
                key_tile=512,
                iterations=0,
                latency=None,
                admission_only=True,
                diagnostic_only=False,
                numerical_execution_authorized=True,
                reference_scope=protocol.REFERENCE_SCOPE,
                state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
                integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
                checkpoint_pins=pins,
                payload_bytes_per_chip=protocol.PAYLOAD_BYTES_PER_CHIP,
                mesh_sha256=MESH_SHA,
                topology_sha256=TOPOLOGY_SHA,
                topology_fleet_sha256=FLEET_SHA,
                versions={"jax": "0.10.1", "libtpu": "0.0.41"},
                physical_device_ids=np.asarray(order).reshape(8, 4).tolist(),
            )
            for s in record["local_device_slots"]:
                slot = s["device_slot"]
                s.update(
                    observed_selected_tensor_sha256=ledger[slot]["selected"],
                    expected_full_file_sha256_not_verified=ledger[slot]["full_sha256"],
                    selected_payload_bytes=protocol.PAYLOAD_BYTES_PER_CHIP,
                )
            preflight = dict(
                code_hash=pin,
                layer=6,
                launch_rank=rank,
                hostname=record["hostname"],
                headers=[dict(device_slot=s) for s in slots.values()],
            )
            raw = json.dumps(preflight).encode()
            (root / "retained_preflight.json").write_bytes(raw)
            record["retained_preflight_sha256"] = sha256(raw).hexdigest()
            (root / "runner.json").write_text(json.dumps(record))
            (root / "worker.log").write_text(
                "CPU synthetic-output/counter fixture; NOT TPU proof\n"
            )
            records.append(deepcopy(record))
    campaign.validate_workers(
        records, pin, layer=6, pins=pins, ledger=ledger, window_numerical=True
    )
    from google.cloud import storage

    blobs = {}

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, data):
            self.name, self.data, self.size = name, data, len(data)

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return self.data

    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        receipts = []
        for filename in campaign.evidence_files(6, window_numerical=True):
            data = (tmp_path / f"rank{rank}" / filename).read_bytes()
            blob = Blob(prefix + filename, data)
            blobs[blob.name] = blob
            receipts.append(
                dict(
                    name=blob.name,
                    generation=str(blob.generation),
                    size=blob.size,
                    crc32c=blob.crc32c,
                    original_sha256=sha256(data).hexdigest(),
                )
            )
        name = prefix + "worker_receipts.json"
        blobs[name] = Blob(name, json.dumps(receipts).encode())
    bucket = SimpleNamespace(
        get_blob=lambda name: blobs[name], blob=lambda name, generation: blobs[name]
    )

    def named_bucket(name):
        assert name == "driftbench-dsv4-uc"
        return bucket

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=named_bucket))
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path / "collected")
    result = campaign.collect(tag, pin)
    campaign.validate_record(result, pin)
    for mutate in (
        lambda r: r.update(compile_only=True),
        lambda r: r.update(performance_claim=True),
        lambda r: r["workers"][0].update(model_executable_calls=14),
        lambda r: r["workers"][0]["local_device_slots"][0].update(device_slot=4),
        lambda r: r["workers"][0]["call_evidence"][0]["post_memory"][0].update(
            process_index=4
        ),
        lambda r: r["workers"][0]["local_device_slots"][0].update(
            observed_selected_tensor_sha256={}
        ),
    ):
        bad = deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            campaign.validate_record(bad, pin)
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        block.split("\nPY\n", 1)[0]
        for block in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in block
    )
    (tmp_path / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "test.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(tmp_path),
            pin,
            str(db),
            str(Path.cwd()),
            "1",
            protocol.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-wrapper-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "four completed B32" in summary["claim_scope"]
    assert "not independent full-score-row DSA" in summary["claim_scope"]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "select correct, score, latency_ms from items"
        ).fetchall() == [(1, 1.0, None)]


def test_numerical_mode_is_distinct_and_fixed_layer():
    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_test"
    assert acquisition.is_numerical_tag(tag) and acquisition.is_window_tag(tag)
    assert not acquisition.is_acquisition_tag(tag)
    assert campaign.layer_from_tag(tag) == 6
    assert not acquisition.is_numerical_tag(tag.replace("_l6_", "_l3_"))
    with pytest.raises(ValueError):
        campaign.evidence_files(3, window_numerical=True)
    assert len(campaign.evidence_files(6, window_numerical=True)) == 18
