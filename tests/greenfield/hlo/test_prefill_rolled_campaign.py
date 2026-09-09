"""Three-call producer to real collector/SQLite; fixture compute, no TPU use.

One producer invocation; peer records use their OWN retained32-owner originals,
with explicitly synthetic runtime counters. No relabeling another owner's cache.
"""

import ast
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from scripts.greenfield import prefill_rolled_window as protocol
from scripts.greenfield import prefill_rolled_worker as worker
from scripts.greenfield import prefill_rolled_admission as admission
from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.prefill_layer_evidence import encode_arrays
from scripts.greenfield.prefill_window_worker import save_arrays
from tests.greenfield.hlo.test_prefill_rolled_worker import ROOT, setup_worker


def test_distinct_mode_and_actual_early_probe(tmp_path, monkeypatch):
    from scripts.greenfield import probe_ws32_prefill_layer as probe
    from scripts.greenfield import prefill_router_protocol as router

    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_fixture"
    assert campaign.layer_from_tag(tag) == 6 and acquisition.is_window_tag(tag)
    assert not any(
        f(tag)
        for f in (
            acquisition.is_acquisition_tag,
            acquisition.is_numerical_tag,
            acquisition.is_phase_baseline_tag,
            acquisition.is_completed_numerical_tag,
        )
    )
    assert campaign.program_names(6, rolled_window=True) == protocol.PROGRAMS
    assert len(campaign.evidence_files(6, rolled_window=True)) == 13
    for mode in (
        "diagnostic",
        "materialized",
        "window_numerical",
        "phase_baseline",
        "completed_window",
        "completed_numerical",
        "boundary_diagnostic",
    ):
        with pytest.raises(ValueError):
            campaign.evidence_files(6, rolled_window=True, **{mode: True})
    tree = ast.parse(Path(probe.__file__).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    first = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prefix_mlp" for t in n.targets)
    )
    last = next(
        i
        for i, n in enumerate(main.body[first:], first)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "_atomic_json"
    )
    calls = []
    monkeypatch.setattr(admission, "registered_programs", lambda: calls.append("pins"))
    monkeypatch.setattr(
        protocol, "load_reference", lambda root, rank: calls.append((root, rank))
    )
    namespace = dict(
        vars(probe),
        tag=tag,
        layer=6,
        router_protocol=router,
        output=tmp_path / "runner.json",
        args=SimpleNamespace(
            expected_code_hash="a" * 40, process_id=0, output_dir=tmp_path
        ),
    )
    exec(
        compile(
            ast.Module(body=main.body[first : last + 1], type_ignores=[]),
            "<actual-rolled-prejax>",
            "exec",
        ),
        namespace,
    )
    assert calls == ["pins", (tmp_path / "retained_reference", 0)]
    r = namespace["record"]
    assert r["protocol"] == protocol.PROTOCOL and r["profile"] == admission.PROFILE
    assert r["reference_scope"] == protocol.REFERENCE_SCOPE
    assert r["admission_only"] is True and r["diagnostic_only"] is False
    assert r["compile_only"] is False and not r["independent_canonical_dsa_claim"]


def test_three_call_publication_fleet_original_replay_and_db(tmp_path, monkeypatch):
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication

    refs = [protocol.load_reference(ROOT, rank=r) for r in range(8)]
    base = tmp_path / "base"
    base.mkdir()
    template, dispatched, kwargs = setup_worker(base, monkeypatch, refs[0])
    worker.execute(**kwargs)
    assert dispatched == list(protocol.PROGRAMS)
    records = []
    for rank, ref in enumerate(refs):
        root = tmp_path / f"rank{rank}"
        root.mkdir()
        r = deepcopy(template)
        for key in (
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
            r[key] = deepcopy(ref.record[key])
        r.update(
            status="SUCCESS",
            admission_only=True,
            diagnostic_only=False,
            numerical_execution_authorized=True,
            reference_scope=protocol.REFERENCE_SCOPE,
        )
        r["original_binding"] = protocol.originals.bind_originals(
            r, ref.slots, protocol.originals.load_capsule()
        )
        r["retained_sources"] = list(ref.sources)
        r["retained_original_comparison"] = dict(ref.bounded_receipt)
        for name in protocol.PROGRAMS:
            for form in ("stablehlo.mlir", "optimized_hlo.txt"):
                os.link(base / f"{name}.{form}", root / f"{name}.{form}")
        for name in protocol.PROGRAMS[:2]:
            os.link(ROOT / f"fleet/rank{rank}/{name}.npz", root / f"{name}.npz")
            r["wk_sha256"][name] = sha256(
                (root / f"{name}.npz").read_bytes()
            ).hexdigest()
        arrays = encode_arrays("input", ref.inputs)
        for d, fields in ref.controls.items():
            arrays.update(encode_arrays(f"actual_{d}", fields))
        r["candidate_sha256"] = save_arrays(root / "candidate.npz", arrays)
        r["candidate_comparison"] = worker.compact_comparison(
            protocol.compare_observations(ref.controls, ref)
        )
        for call in r["call_evidence"]:
            for owners in (call["census"]["devices"], call["post_memory"]):
                for item, device in zip(owners, ref.slots, strict=True):
                    item.update(device_id=device, process_index=r["jax_process_index"])
            call["budget"] = admission.memory_budget(
                call["census"], call["compiled_memory"], active_graph=call["graph"]
            )
        journal = [
            json.loads(s)
            for s in (base / "compile_journal.jsonl").read_text().splitlines()
        ]
        journal[0]["identity"]["launch_rank"] = rank
        raw = ("\n".join(json.dumps(s) for s in journal) + "\n").encode()
        (root / "compile_journal.jsonl").write_bytes(raw)
        r["compile_journal_sha256"] = sha256(raw).hexdigest()
        preflight = dict(
            code_hash=r["code_hash"],
            layer=6,
            launch_rank=rank,
            hostname=r["hostname"],
            headers=[dict(device_slot=s) for s in ref.slots.values()],
        )
        raw = json.dumps(preflight).encode()
        (root / "retained_preflight.json").write_bytes(raw)
        r["retained_preflight_sha256"] = sha256(raw).hexdigest()
        (root / "runner.json").write_text(json.dumps(r))
        (root / "worker.log").write_text(
            "CPU fixture: retained own-owner arrays; synthetic runtime/compiled counters\n"
        )
        records.append(r)

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
    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_cpu_fixture"
    for rank in range(8):
        campaign.publish_rank(tag, rank)
    # Refuse excessive declared payloads BEFORE reading model/array payloads.
    ledger = tmp_path / "rank0/worker_receipts.json"
    old = ledger.read_bytes()
    bad = json.loads(old)
    bad[-1]["size"] = protocol.MAX_RANK_BYTES + 1
    ledger.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="size refused"):
        campaign.phase_receipt_preflight(bucket, tag, tmp_path)
    ledger.write_bytes(old)
    dest = tmp_path / "collected"
    dest.mkdir()
    monkeypatch.setattr(campaign, "run_root", lambda tag: dest)
    result = campaign.collect(tag, template["code_hash"])
    campaign.validate_record(result, template["code_hash"])
    assert result["latency"] is None and result["admission_only"] is True
    for field, value in (
        ("checksum", "0" * 64),
        ("reference_scope", "FULL_MODEL"),
        ("performance_claim", True),
    ):
        bad = deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError):
            campaign.validate_record(bad, template["code_hash"])
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    (dest / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "rolled.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(dest),
            template["code_hash"],
            str(db),
            str(Path.cwd()),
            "1",
            protocol.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-rolled-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id,correct,score,latency_ms from items"
        ).fetchall() == [
            ("layer6_rolled128_retained_db600_three_calls_v1", 1, 1.0, None)
        ]
    assert (
        "not independent canonical"
        in json.loads((dest / "summary.json").read_text())["claim_scope"]
    )
