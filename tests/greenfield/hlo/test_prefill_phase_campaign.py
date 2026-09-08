"""Actual eight-rank publication/collector/DB, archived tensors and CPU traces."""

from copy import deepcopy
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_evidence as evidence
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.hlo.test_prefill_phase_evidence import collected


def test_phase_fleet_publication_collection_and_real_db(
    collected, tmp_path, monkeypatch
):
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication
    from scripts.analysis import parse_xplane
    from scripts.analysis.test_parse_xplane import fake_core

    base, template, old_slots = collected
    capsule = originals.load_capsule()
    archive = Path("/home/gianl/glm-run") / capsule["source_tag"] / "fleet"
    tag = f"greenfield_fp8_{phase.KERNEL}_l6_cpu_fixture"
    records, ledger = [], {}
    programs = {n: (base / f"{n}.optimized_hlo.txt").read_text() for n in phase.COUNTS}
    groups = {g["module_regex"]: g for g in phase.trace_groups(programs).values()}
    for rank in range(8):
        src = archive / f"rank{rank}"
        root = tmp_path / f"rank{rank}"
        root.mkdir()
        old = json.loads((src / "runner.json").read_text())
        record = deepcopy(template)
        for k in (
            "launch_rank",
            "jax_process_index",
            "local_device_slots",
            "hostname",
            "pid",
            "start_ticks",
            "boot_id",
            "checkpoint_pins",
            "topology_sha256",
            "topology_fleet_sha256",
            "versions",
            "integrity_scope",
            "payload_bytes_per_chip",
        ):
            record[k] = old[k]
        record.update(
            status="SUCCESS",
            layer=6,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            admission_only=False,
            diagnostic_only=True,
            numerical_execution_authorized=True,
            state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
            hlo=old["hlo"],
        )
        slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
        for slot in record["local_device_slots"]:
            ledger[slot["device_slot"]] = dict(
                selected=slot["observed_selected_tensor_sha256"],
                full_sha256=slot["expected_full_file_sha256_not_verified"],
            )
        for n in evidence.PROGRAMS:
            for form in ("stablehlo.mlir", "optimized_hlo.txt"):
                os.link(base / f"{n}.{form}", root / f"{n}.{form}")
        os.link(src / "competitive.npz", root / "phase_first.npz")
        for n in ("wk_decode", "wk_promote", "wk_boundary"):
            os.link(src / f"{n}.npz", root / f"{n}.npz")
        record["wk_originals"] = old["wk_originals"]
        record["wk_boundary_sha256"] = old["wk_boundary_sha256"]
        record["original_binding"] = originals.bind_originals(record, slots, capsule)
        record["original_authentication"].update(
            binding=record["original_binding"],
            first_npz_sha256=sha256(
                (root / "phase_first.npz").read_bytes()
            ).hexdigest(),
        )
        journal = [
            json.loads(s)
            for s in (base / "compile_journal.jsonl").read_text().splitlines()
        ]
        journal[0]["identity"]["launch_rank"] = rank
        raw = ("\n".join(json.dumps(r) for r in journal) + "\n").encode()
        (root / "compile_journal.jsonl").write_bytes(raw)
        record["compile_journal_sha256"] = sha256(raw).hexdigest()
        index = []
        with (root / "phase_calls.jsonl.gz").open("wb") as stream:
            for entry in phase.read_call_witnesses(
                base / "phase_calls.jsonl.gz", template["phase_call_index"]
            ):
                for rows in (entry["census"]["devices"], entry["post_memory"]):
                    for row, device in zip(rows, slots, strict=True):
                        row.update(
                            device_id=device, process_index=record["jax_process_index"]
                        )
                entry["budget"] = assembly.memory_budget(
                    entry["census"],
                    entry["compiled_memory"],
                    active_graph=entry["graph"],
                )
                raw = (
                    json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                packed = gzip.compress(raw, mtime=0)
                compact = {
                    k: entry[k]
                    for k in ("phase", "graph", "completed", "completed_call_seconds")
                }
                compact.update(
                    offset=stream.tell(),
                    compressed_bytes=len(packed),
                    raw_sha256=sha256(raw).hexdigest(),
                )
                index.append(compact)
                stream.write(packed)
        record["phase_call_index"] = index
        xs = parse_xplane.build_xplane_classes()["XSpace"]()
        xs.hostnames.append(record["hostname"])
        raw = xs.SerializeToString()
        (root / "phase.xplane.pb").write_bytes(raw)
        record["phase_trace_file"].update(
            bytes=len(raw), sha256=sha256(raw).hexdigest()
        )
        preflight = dict(
            code_hash=record["code_hash"],
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
            "CPU fixture: original tensors, synthetic counters/trace events\n"
        )
        records.append(record)
    pin = template["code_hash"]
    monkeypatch.setattr(
        campaign,
        "checkpoint_ledger",
        lambda layer: (records[0]["checkpoint_pins"], ledger),
    )
    blobs = {}

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, path):
            self.name, self.path, self.size = name, path, path.stat().st_size

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return self.path.read_bytes()

    bucket = SimpleNamespace(
        get_blob=lambda n: blobs[n], blob=lambda n, generation: blobs[n]
    )
    monkeypatch.setattr(
        storage, "Client", lambda: SimpleNamespace(bucket=lambda n: bucket)
    )

    def publish(bucket, name, path, digest, *, compressed):
        assert not compressed
        b = blobs[name] = Blob(name, path)
        return dict(
            name=name,
            generation=str(b.generation),
            size=b.size,
            crc32c=b.crc32c,
            original_sha256=digest["sha256"],
        )

    monkeypatch.setattr(publication, "publish_exact", publish)
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    for rank in range(8):
        campaign.publish_rank(tag, rank)

    # Parse real protobuf hostnames; only TPU event contents are fixture data.
    seen = set()

    def aggregate_host(path, regex):
        host = list(parse_xplane.load_xspace(path).hostnames)[0]
        seen.add(path)
        return [
            fake_core(host, i, steps=groups[regex]["expected_calls_per_core"])
            for i in range(8)
        ]

    monkeypatch.setattr(parse_xplane, "aggregate_host", aggregate_host)
    destination = tmp_path / "collected"
    destination.mkdir()
    monkeypatch.setattr(campaign, "run_root", lambda tag: destination)
    result = campaign.collect(tag, pin)
    assert len(seen) == 8 and all("/collected/fleet/" in p for p in seen)
    campaign.validate_record(result, pin)
    assert result["phase_wall"]["scope"] == phase.SCOPE
    for change in ("wall", "owner", "scope"):
        bad = deepcopy(result)
        if change == "wall":
            bad["phase_wall"]["wall"]["shared_prefix_seconds"]["p50"] = -1
        elif change == "owner":
            bad["workers"][0]["local_device_slots"][0]["device_slot"] = 31
        else:
            bad["reference_scope"] = "END_TO_END"
        with pytest.raises(ValueError):
            campaign.validate_record(bad, pin)
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    (destination / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "phase.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(destination),
            pin,
            str(db),
            str(Path.cwd()),
            "1",
            phase.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-phase-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            (
                "layer6_db594_b128_four_b32_phase_sum_estimate_287calls_v1",
                None,
                None,
                None,
            )
        ]
    assert (
        "NOT full-layer latency"
        in json.loads((destination / "summary.json").read_text())["claim_scope"]
    )
