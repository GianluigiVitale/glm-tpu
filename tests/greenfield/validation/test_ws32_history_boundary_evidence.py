"""Original-byte boundary replay; synthetic values, no TPU/math admission."""

from copy import deepcopy
from hashlib import sha256
import json

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_history_boundary_evidence as reader
from scripts.greenfield.prefill_window_worker import save_arrays

SLOTS = dict(zip((12, 14, 13, 15), (9, 13, 25, 29)))


def rows(n, *, health=False):
    return {l: {**{f: np.zeros((n, 1536), dtype=ml_dtypes.bfloat16)
                     for f in reader.worker.FIELDS[:3]},
                "route_ids": np.tile(np.arange(8, dtype=np.int32), (n, 1)),
                "route_weights": np.ones((n, 8), np.float32),
                **(dict(health=np.ones((n, 4), np.bool_)) if health else {})}
            for l in reader.protocol.LAYERS}


def flatten(values, prefix=""):
    return {f"{prefix}layer{l}_{f}": v.view(np.uint16) if v.dtype.name == "bfloat16" else v
            for l, fields in values.items() for f, v in fields.items()}


def persist(root, record):
    (root / "history_report.json").write_text(json.dumps(record["history"], sort_keys=True))


def save(root, report, label, arrays):
    path = root / f"{label}.npz"
    digest = save_arrays(path, arrays)
    raw = sum(v.nbytes for v in arrays.values())
    report["retained"][label] = dict(npz_sha256=digest, bytes=path.stat().st_size, raw_array_bytes=raw)
    report["retained_bytes"] = sum(v["raw_array_bytes"] for v in report["retained"].values())


@pytest.fixture
def bundle(tmp_path):
    originals = {b: dict(positions=np.tile(np.arange(2048, dtype=np.int32), (4, 1, 1)),
                        counts=np.full((4, 1), 2048, np.int32),
                        scores=np.ones((4, 1, 2048), np.float32)) for b in reader.protocol.BRANCHES}
    report = dict(protocol=reader.protocol.PROTOCOL, complete=True, attribution_eligible=True,
        prompt_length=8155, steps=319, numerical_promotion=False, performance_claim=False,
        cause_claim=False, boundary_layout=reader.layout(SLOTS, 3), groups=[], retained={}, retained_bytes=0,
        observer={b: dict(healthy=True, capture_errors=[]) for b in reader.protocol.BRANCHES},
        reproduction={b: reader.worker.reproduce(originals[b], originals[b]) for b in reader.protocol.BRANCHES})
    for step in reader.protocol.plan():
        if step.branch != "candidate":
            continue
        a, b = rows(step.count), rows(step.count)
        if step.group == 1:
            a[3]["update"][4, 0] = 1
            save(tmp_path, report, "first_difference_group1", {**flatten(a, "candidate_"), **flatten(b, "control_")})
        report["groups"].append(dict(reader.worker.compare_rows(a, b, offset=step.offset), group=step.group))
    report["first_difference"] = dict(report["groups"][1]["first"], group=1)
    a, b = rows(1, health=True), rows(1, health=True)
    for branch, value in (("candidate", a), ("control", b)):
        save(tmp_path, report, f"observer_{branch}", {**originals[branch], **flatten(value)})
    report["observer_comparison"] = reader.worker.compare_rows(a, b, offset=8155)
    report["observations_equal"] = True
    record = dict(history=report, jax_process_index=3)
    persist(tmp_path, record)
    return tmp_path, record, originals


def test_replay_retained_first_and_both_original_observers(bundle):
    root, record, originals = bundle
    result = reader.replay(root, record, SLOTS, originals=originals)
    assert result["reproduced"] is True
    assert result["first_difference"] == dict(group=1, layer=3, field="update", position=132)
    assert result["retained_groups_replayed"] == 1
    assert result["cache_values_replayed"] is False and result["cause_claim"] is False
    assert result["remaining_group_comparisons"] == "WORKER_ASSERTIONS_NOT_RETAINED_ARRAY_REPLAY"


@pytest.mark.parametrize("defect", ["report", "scope", "layout", "group_offset", "group_count",
    "group_first", "first", "bytes", "extra_npz", "npz_sha", "observer_value", "observer_shape",
    "observer_health", "observer_nonfinite", "first_array", "array_dtype", "array_inventory"])
def test_rebound_producer_claims_cannot_replace_original_replay(bundle, defect):
    root, record, originals = bundle
    report = record["history"]
    if defect == "report":
        record["history"] = deepcopy(report)
        record["history"]["complete"] = False
    elif defect == "scope":
        report["cause_claim"] = True
    elif defect == "layout":
        report["boundary_layout"]["feature_columns"][0] = 0
    elif defect == "group_offset":
        report["groups"][1]["offset"] += 1
    elif defect == "group_count":
        report["groups"][0]["layers"]["0"]["residual"]["differing_rows"] = True
    elif defect == "group_first":
        report["groups"][1]["first"]["position"] += 1
    elif defect == "first":
        report["first_difference"]["layer"] = 2
    elif defect == "bytes":
        report["retained_bytes"] += 1
    elif defect == "extra_npz":
        (root / "extra.npz").write_bytes(b"x")
    elif defect == "npz_sha":
        report["retained"]["observer_candidate"]["npz_sha256"] = "0" * 64
    else:
        label = "first_difference_group1" if defect == "first_array" else "observer_candidate"
        with np.load(root / f"{label}.npz", allow_pickle=False) as saved:
            arrays = dict(saved)
        if defect == "observer_value":
            arrays["scores"][0, 0, 0] += 1
        elif defect == "observer_shape":
            arrays["positions"] = arrays["positions"].reshape(4, 2048)
        elif defect == "observer_health":
            arrays["layer3_health"][0, 0] = False
        elif defect == "observer_nonfinite":
            arrays["layer3_update"].view(ml_dtypes.bfloat16)[0, 0] = np.nan
        elif defect == "first_array":
            arrays["candidate_layer3_update"][:] = 0
        elif defect == "array_dtype":
            arrays["layer3_update"] = arrays["layer3_update"].astype(np.float32)
        else:
            arrays["unexpected"] = np.zeros(1, np.int32)
        save(root, report, label, arrays)
    if defect != "report":
        persist(root, record)
    with pytest.raises((ValueError, KeyError)):
        reader.replay(root, record, SLOTS, originals=originals)


def fleet():
    reports, records = [], []
    for rank in range(8):
        slots = list(range(rank * 4, rank * 4 + 4))
        geom = reader.layout(dict(zip(slots, slots)), rank)
        fingerprints = {f"observer/{b}/{s}/{l}/{f}": sha256(f.encode()).hexdigest()
                        for b in reader.protocol.BRANCHES for s in slots for l in reader.protocol.LAYERS
                        for f in (*reader.worker.FIELDS, "health")}
        reports.append(dict(fingerprints=fingerprints, observation_sha256={"fixture": "same"}))
        caches = {b: {f: {str(i): {str(s): sha256(f"{f}/{i}".encode()).hexdigest() for s in slots}
                         for i in range(7 if f == "kv" else 4)} for f in reader.worker.CACHE_FAMILIES}
                  for b in reader.protocol.BRANCHES}
        equality = {f: {i: True for i in caches["candidate"][f]} for f in reader.worker.CACHE_FAMILIES}
        records.append(dict(history=dict(boundary_layout=geom, cache_digests=caches,
                                         cache_equality=equality, groups=["fixture same"])))
    return reports, records


def test_join_all32_original_boundary_and_cache_observations():
    result = reader.validate_replicas(*fleet())
    assert result["retained_boundary_replicas_checked"] is True
    assert result["cache_values_replayed"] is False


@pytest.mark.parametrize("defect", ["observation", "boundary", "missing", "duplicate", "cache",
                                  "cache_owner", "cache_equal", "group"])
def test_fleet_join_refuses_conflicting_owners_and_replicas(defect):
    reports, records = fleet()
    if defect == "observation":
        reports[1]["observation_sha256"] = {}
    elif defect in ("boundary", "missing"):
        key = next(iter(reports[0]["fingerprints"]))
        if defect == "boundary":
            reports[0]["fingerprints"][key] = "0" * 64
        else:
            del reports[0]["fingerprints"][key]
    elif defect == "duplicate":
        reports[1]["fingerprints"].update(reports[0]["fingerprints"])
    elif defect == "cache":
        records[0]["history"]["cache_digests"]["candidate"]["kv"]["0"]["0"] = "0" * 64
        records[0]["history"]["cache_equality"]["kv"]["0"] = False
    elif defect == "cache_owner":
        del records[0]["history"]["cache_digests"]["candidate"]["kv"]["0"]["0"]
    elif defect == "cache_equal":
        records[0]["history"]["cache_equality"]["kv"]["0"] = False
    else:
        records[0]["history"]["groups"] = ["changed"]
    with pytest.raises((ValueError, KeyError)):
        reader.validate_replicas(reports, records)
