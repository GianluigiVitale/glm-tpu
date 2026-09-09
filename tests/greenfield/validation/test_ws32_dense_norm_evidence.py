"""Original DB605 endpoints plus explicitly fixture norm/suffix observations.

Exercise real capture/NPZ/JSON readers; these fixtures do not establish actual
TPU norm values, boundary arithmetic, optimized HLO or numerical memory.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_evidence as base
from scripts.greenfield import ws32_dense_frontier_protocol as prior_protocol
from scripts.greenfield import ws32_dense_norm_evidence as evidence
from scripts.greenfield import ws32_dense_norm_originals as originals_module
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield import ws32_dense_norm_worker as worker
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import save_arrays
from tests.greenfield.validation import test_ws32_dense_norm_capture as capture_fixture
from tests.greenfield.validation.test_ws32_dense_norm_execution import staged
from tests.greenfield.validation.test_ws32_dense_frontier_execution import (
    SLOTS,
    ORIGINAL,
)


def save(root, label, arrays, report):
    path = root / f"{label}.npz"
    report = {
        **report,
        "npz_sha256": save_arrays(path, arrays),
        "npz_bytes": path.stat().st_size,
        "raw_array_bytes": sum(a.nbytes for a in arrays.values()),
    }
    _atomic_json(root / f"{label}.json", report)
    return json.loads(json.dumps(report))


def setup(root, monkeypatch):
    repo = Path.cwd()
    source = Path("/home/gianl/glm-run") / originals_module.TAG / "fleet/rank0"
    old_runner, originals, identity = originals_module.load_bundle(
        source, repo=repo, rank=0
    )
    prior, witness = prior_protocol.load_reference(ORIGINAL, rank=0)
    slots = {o["device_id"]: o["device_slot"] for o in old_runner["local_device_slots"]}
    monkeypatch.setattr(capture_fixture, "SLOTS", slots)
    reports, saved = {}, {}
    binding = {
        s: {
            "selected": {
                evidence.NORM_WEIGHT: base.digest(
                    np.ones(1536, ml_dtypes.bfloat16).view(np.uint16)
                )
            }
        }
        for s in slots.values()
    }
    for label, count, _ in protocol.CAPTURES:
        reports[label] = save(
            root,
            label,
            originals[label],
            old_runner["dense_frontier"]["originals"][label],
        )
        packet = capture_fixture.packet(count)
        a, r = worker.capture_packet(packet, slots=slots, count=count)
        saved[label + "_norm"] = a
        reports[label + "_norm"] = save(root, label + "_norm", a, r)
        a = {
            f"slot{s}__{f}": originals[label][f"slot{s}_layer0__{f}"]
            for s in slots.values()
            for f in ("output", "health")
        }
        r = dict(valid=True, errors=[], count=count, owner_slots=slots)
        saved["own_" + label] = a
        reports["own_" + label] = save(root, "own_" + label, a, r)
    diffs = []
    for start in (0, 32, 64, 96):
        a = {
            f"slot{s}__output": originals["wide_final"][f"slot{s}_layer0__output"][
                start : start + 32
            ].copy()
            for s in slots.values()
        }
        a.update({f"slot{s}__health": np.ones(128, bool) for s in slots.values()})
        # Deliberately nonidentical placement is an admissible DIAGNOSTIC result.
        if start == 32:
            a[f"slot{next(iter(slots.values()))}__output"][0, 0] ^= 1
        for s in slots.values():
            left = a[f"slot{s}__output"]
            right = originals["wide_final"][f"slot{s}_layer0__output"][
                start : start + 32
            ]
            diffs.append(
                dict(
                    slot=s,
                    start=start,
                    differing_words=int(np.count_nonzero(left != right)),
                    bytes_equal=left.tobytes() == right.tobytes(),
                )
            )
        saved[f"cross_{start}"] = a
        reports[f"cross_{start}"] = save(
            root,
            f"cross_{start}",
            a,
            dict(valid=True, errors=[], count=32, owner_slots=slots),
        )
    owners = [
        base.compare_owner(
            witness, branch=label, slot=s, caches=base.cache_bits(originals[label], s)
        )
        for label in ("wide_final", "narrow_128")
        for s in slots.values()
    ]
    reproduction = dict(
        retained_fields={
            n: originals_module.require_reproduction(a, a) for n, a in originals.items()
        },
        caches=owners,
        reproduced=True,
    )
    own = {
        n: {
            **originals_module.require_reproduction(
                saved["own_" + n], saved["own_" + n]
            ),
            "scope": "OWN_COMPLETED_SUFFIX_OUTPUT_AND_FULL_HEALTH",
        }
        for n in originals
    }
    cross = dict(
        comparisons=diffs,
        numerical_promotion=False,
        cause_claim=False,
        performance_claim=False,
        scope="COMPLETED_SUFFIX_IDENTICAL_ROW_INPUT_PLACEMENT_ONLY",
    )
    for name, value in [
        ("reproduction", reproduction),
        ("own_reproduction", own),
        ("cross_comparison", cross),
    ]:
        _atomic_json(root / f"{name}.json", value)
    record = dict(
        dense_norm=dict(
            complete=True,
            originals=reports,
            reproduction=reproduction,
            own_suffix_reproduction=own,
            cross_comparison=cross,
            numerical_promotion=False,
            performance_claim=False,
            cause_claim=False,
            raw_array_bytes=sum(r["raw_array_bytes"] for r in reports.values()),
        )
    )
    return record, slots, witness, originals, binding, saved


def test_actual_originals_capture_replay_and_resigned_mutations(tmp_path, monkeypatch):
    record, slots, witness, originals, bindings, saved = setup(tmp_path, monkeypatch)

    def replay(rec=record, refs=originals, weights=bindings):
        return evidence.replay_outputs(tmp_path, rec, slots, witness, refs, weights)

    report, hashes = replay()
    assert report["reproduced"] and len(hashes) == 408
    assert len(report["boundaries"]) == 16 and len(report["owners"]) == 8
    assert (
        sum(r["differing_words"] for r in report["cross_comparison"]["comparisons"])
        == 1
    )
    assert not report["cause_claim"] and not report["numerical_promotion"]
    assert report["raw_array_bytes"] == 108752896
    # Actual DB604 and DB605 enumerate local devices differently. Unchanged
    # original reports must replay by owner identity, never insertion order.
    reordered = dict(reversed(list(slots.items())))
    reordered_report, reordered_hashes = evidence.replay_outputs(
        tmp_path, record, reordered, witness, originals, bindings
    )
    assert reordered_hashes == hashes and reordered_report["reproduced"]
    for case in (
        "missing",
        "raw_budget",
        "promotion",
        "false_cross",
        "false_own",
        "reference",
        "checkpoint",
    ):
        rec = deepcopy(record)
        refs = originals
        weights = bindings
        if case == "missing":
            del rec["dense_norm"]["originals"]["cross_96"]
        elif case == "raw_budget":
            rec["dense_norm"]["raw_array_bytes"] += 1
        elif case == "promotion":
            rec["dense_norm"]["cause_claim"] = True
        elif case == "false_cross":
            rec["dense_norm"]["cross_comparison"]["comparisons"][0][
                "differing_words"
            ] = 99
        elif case == "false_own":
            rec["dense_norm"]["own_suffix_reproduction"]["wide_final"]["bytes"] = 0
        elif case == "reference":
            refs = {
                **originals,
                "wide_final": dict(originals["wide_final"], unknown=np.zeros(1)),
            }
        else:
            weights = deepcopy(bindings)
            weights[next(iter(slots.values()))]["selected"][evidence.NORM_WEIGHT] = (
                "0" * 64
            )
        with pytest.raises(ValueError):
            replay(rec, refs, weights)
    # Re-sign altered packet bytes and its sidecar: the independent reader must
    # still reject semantics, not merely detect a stale digest.
    name = "wide_final_norm"
    original_report = record["dense_norm"]["originals"][name]
    for field, kind in [
        ("boundary_live", "mask"),
        ("post_norm_weight", "weight"),
        ("post_norm_summed", "nonfinite"),
        ("post_norm_inverse", "dtype"),
    ]:
        arrays = {k: v.copy() for k, v in saved[name].items()}
        key = f"slot{next(iter(slots.values()))}__{field}"
        if kind == "mask":
            arrays[key][0] = False
        elif kind == "weight":
            arrays[key][0] ^= 1
        elif kind == "nonfinite":
            arrays[key][0, 0] = np.nan
        else:
            arrays[key] = arrays[key].astype(np.uint16)
        changed = save(tmp_path, name, arrays, original_report)
        with pytest.raises(ValueError):
            evidence.read_packet(
                tmp_path, name, changed, slots, 128, packet=True, bindings=bindings
            )
    save(tmp_path, name, saved[name], original_report)
    replay()


@pytest.mark.parametrize("case", ["duplicate", "bad_slot", "bad_start", "value"])
def test_cross_owner_join_keeps_complete_records(case):
    value = dict(
        comparisons=[
            dict(slot=9, start=32, differing_words=1, bytes_equal=False),
            dict(slot=13, start=32, differing_words=0, bytes_equal=True),
        ]
    )
    assert evidence.canonical_cross(value) == evidence.canonical_cross(
        dict(comparisons=list(reversed(value["comparisons"])))
    )
    bad = deepcopy(value)
    if case == "duplicate":
        bad["comparisons"].append(bad["comparisons"][0])
    elif case == "bad_slot":
        bad["comparisons"][0]["slot"] = True
    elif case == "bad_start":
        bad["comparisons"][0]["start"] = 33
    else:
        bad["comparisons"][0]["differing_words"] = 2
        assert evidence.canonical_cross(bad) != evidence.canonical_cross(value)
        return
    with pytest.raises(ValueError):
        evidence.canonical_cross(bad)


def test_actual18call_journal_replay_and_gate_order(tmp_path, monkeypatch):
    run, record, events = staged(tmp_path, monkeypatch)
    run()
    from scripts.greenfield import ws32_dense_norm_admission

    monkeypatch.setattr(
        ws32_dense_norm_admission,
        "inspect_program",
        lambda *a: dict(passed=True, fixture=True),
    )
    # Actual writer/reader/budgets; staged compiler/math/memory are fixtures.
    base.replay_execution(tmp_path, record, SLOTS)
    raw = (tmp_path / "compile_journal.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [r["stage"] for r in rows] == evidence.expected_stages()
    for change in ("reproduction_order", "own_order", "time", "memory", "calls"):
        altered = deepcopy(rows)
        rec = deepcopy(record)
        if change in ("reproduction_order", "own_order"):
            target = (
                "norm/reproduction"
                if change == "reproduction_order"
                else "norm/own_reproduction"
            )
            n = next(i for i, r in enumerate(altered) if r["stage"] == target)
            altered[n], altered[n + 1] = altered[n + 1], altered[n]
        elif change == "time":
            altered[-1]["monotonic_seconds"] = 0
        elif change == "memory":
            rec["call_evidence"][-1]["compiled_memory"]["dense_suffix"][
                "temp_size_in_bytes"
            ] += 1
        else:
            rec["call_evidence"].pop()
        changed = b"".join((json.dumps(row) + "\n").encode() for row in altered)
        (tmp_path / "compile_journal.jsonl").write_bytes(changed)
        rec["compile_journal_sha256"] = sha256(changed).hexdigest()
        with pytest.raises(ValueError):
            base.replay_execution(tmp_path, rec, SLOTS)
    (tmp_path / "compile_journal.jsonl").write_bytes(raw)
