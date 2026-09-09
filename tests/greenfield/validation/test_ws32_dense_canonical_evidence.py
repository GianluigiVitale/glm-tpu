"""Retained real DB605 arrays; synthetic candidate outputs, no hardware proof."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_evidence as base
from scripts.greenfield import ws32_dense_frontier_protocol as prior_protocol
from scripts.greenfield import ws32_dense_canonical as canonical
from scripts.greenfield import ws32_dense_canonical_admission as admission
from scripts.greenfield import ws32_dense_canonical_evidence as evidence
from scripts.greenfield import ws32_dense_norm_originals as source
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import save_arrays
from tests.greenfield.validation.test_ws32_dense_canonical_execution import staged
from tests.greenfield.validation.test_ws32_dense_frontier_execution import (
    SLOTS,
    ORIGINAL,
)
from tests.greenfield.validation.test_ws32_dense_canonical_entry_transport import (
    TAG,
    PIN,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path("/home/gianl/glm-run") / source.TAG / "fleet"


def capture(root, old, originals):
    slots = {o["device_id"]: o["device_slot"] for o in old["local_device_slots"]}
    arrays = canonical.narrow_target(originals, tuple(slots.values()))
    report = deepcopy(old["dense_frontier"]["originals"]["wide_final"])
    path = root / (canonical.CAPSULE + ".npz")
    report.update(
        npz_sha256=save_arrays(path, arrays),
        npz_bytes=path.stat().st_size,
        raw_array_bytes=sum(v.nbytes for v in arrays.values()),
    )
    _atomic_json(root / (canonical.CAPSULE + ".json"), report)
    comparison = canonical.compare(arrays, originals, tuple(slots.values()))
    _atomic_json(root / "comparison.json", comparison)
    return (
        dict(
            dense_canonical=dict(
                complete=True,
                numerical_promotion=False,
                performance_claim=False,
                original=report,
                comparison=comparison,
            )
        ),
        slots,
        arrays,
    )


def test_actual_journal_and_five_call_replay(tmp_path, monkeypatch):
    run, record, _ = staged(tmp_path, monkeypatch)
    run()
    monkeypatch.setattr(
        admission, "inspect_program", lambda *a: dict(passed=True, fixture=True)
    )
    record["profile"] = canonical.PROFILE
    original = json.loads(json.dumps(record))
    base.replay_execution(tmp_path, original, SLOTS)
    for change in ("call", "graph", "incomplete", "memory", "scope"):
        bad = deepcopy(original)
        if change == "call":
            bad["call_evidence"].pop()
        elif change == "graph":
            bad["call_evidence"][-1]["graph"] = "dense01"
        elif change == "incomplete":
            bad["call_evidence"][-1]["completed"] = False
        elif change == "memory":
            bad["call_evidence"][-1]["compiled_memory"][canonical.GRAPH][
                "temp_size_in_bytes"
            ] += 1
        else:
            bad["protocol"] = prior_protocol.PROTOCOL
        with pytest.raises((ValueError, KeyError)):
            base.replay_execution(tmp_path, bad, SLOTS)


def test_retained96_fields_and_resigned_mutations(tmp_path):
    old, originals, _ = source.load_bundle(SOURCE / "rank0", repo=REPO, rank=0)
    record, slots, arrays = capture(tmp_path, old, originals)
    result, hashes = evidence.replay_outputs(tmp_path, record, slots, originals)
    assert result["passed"] and result["arrays"] == len(hashes) == 96
    assert result["bytes"] == 40965120 and not result["token11_cause_proven"]
    assert evidence.replay_outputs(
        tmp_path, record, dict(reversed(list(slots.items()))), originals
    ) == (result, hashes)
    path = tmp_path / (canonical.CAPSULE + ".npz")
    for field in ("output", "health", "kv", "positions", "route_weights"):
        changed = dict(arrays)
        key = next(k for k in arrays if k.endswith("__" + field))
        changed[key] = arrays[key].copy()
        changed[key].flat[0] = (
            not changed[key].flat[0] if field == "health" else changed[key].flat[0] + 1
        )
        rec = deepcopy(record)
        report = rec["dense_canonical"]["original"]
        report.update(
            npz_sha256=save_arrays(path, changed), npz_bytes=path.stat().st_size
        )
        _atomic_json(tmp_path / (canonical.CAPSULE + ".json"), report)
        with pytest.raises(ValueError):
            evidence.replay_outputs(tmp_path, rec, slots, originals)


def test_all8_retained_arrays_fleet_owner_join_and_mutations(tmp_path, monkeypatch):
    records, bindings = [], {}
    for rank in range(8):
        old, originals, identity = source.load_bundle(
            SOURCE / f"rank{rank}", repo=REPO, rank=rank
        )
        folder = tmp_path / f"rank{rank}"
        folder.mkdir()
        captured, _, _ = capture(folder, old, originals)
        record = deepcopy(old)
        record.update(captured)
        record.update(
            tag=TAG,
            code_hash=PIN,
            protocol=canonical.PROTOCOL,
            kernel=canonical.KERNEL,
            profile=canonical.PROFILE,
            current_phase="canonical/reproduction",
            norm_originals=identity,
        )
        record["programs"] = {
            name: dict(
                stablehlo_sha256="c" * 64,
                optimized_hlo_sha256="d" * 64,
                compiled_memory={"fixture": 1},
            )
            for name in canonical.PROGRAMS
        }
        for o in record["local_device_slots"]:
            bindings[o["device_slot"]] = dict(
                full_file_sha256=o["expected_full_file_sha256_not_verified"],
                selected=o["observed_selected_tensor_sha256"],
            )
        preflight = {
            k: record[k]
            for k in (
                "tag",
                "code_hash",
                "launch_rank",
                "hostname",
                "original_runner_sha256",
                "checkpoint_pins",
                "protocol",
                "norm_originals",
            )
        }
        preflight["combined_reference_bytes"] = (
            prior_protocol.LEDGER_PIN["size"]
            + sum(
                p["size"]
                for p in prior_protocol.reference_pins(ORIGINAL, rank).values()
            )
            + identity["bytes"]
        )
        _atomic_json(folder / "retained_preflight.json", preflight)
        record["retained_preflight_sha256"] = sha256(
            (folder / "retained_preflight.json").read_bytes()
        ).hexdigest()
        records.append(record)

    def selected(repo, *, canonical_dense=False):
        assert canonical_dense
        return bindings

    monkeypatch.setattr(base, "checkpoint_bindings", selected)
    # Compiler/call-memory and WK replay have their own actual-writer tests;
    # this seam only avoids fabricating numeric hardware counters here.
    monkeypatch.setattr(base, "replay_execution", lambda *a, **kw: None)
    monkeypatch.setattr(
        base,
        "replay_wk",
        lambda root, record, slots, bindings: [
            dict(layer=l, slot=s, bf16_sha256=str(l), fp32_sha256=str(l))
            for l in (0, 1)
            for s in slots.values()
        ],
    )

    def replay(rows, root=SOURCE):
        return base.validate_fleet(
            tmp_path,
            rows,
            pin=PIN,
            tag=TAG,
            repo=REPO,
            original_root=ORIGINAL,
            norm_original_root=root,
        )

    result = replay(records)
    assert result["reproduced"] and result["hosts"] == 8 and result["owners"] == 32
    assert result["model_calls_per_host"] == 1 and len(result["canonical_replays"]) == 8
    assert result["comparisons"] == [] and not result["cause_claim"]
    for mutation in (
        "source",
        "generation",
        "selected",
        "protocol",
        "mesh",
        "root",
        "phase",
    ):
        bad = deepcopy(records)
        if mutation == "source":
            bad[0]["norm_originals"]["code_hash"] = "e" * 40
        elif mutation == "generation":
            bad[0]["norm_originals"]["original_pins"]["runner.json"]["generation"] = "1"
        elif mutation == "selected":
            bad[0]["local_device_slots"][0]["selected_payload_bytes"] += 1
        elif mutation == "protocol":
            bad[0]["protocol"] = prior_protocol.PROTOCOL
        elif mutation == "mesh":
            bad[0]["physical_device_ids"][0][0] = 999
        elif mutation == "phase":
            bad[0]["current_phase"] = "dense/comparison"
        with pytest.raises(ValueError):
            replay(bad, None if mutation == "root" else SOURCE)
