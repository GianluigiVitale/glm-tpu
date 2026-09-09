"""Actual producer files and fixed original references; fixture TPU math only."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_evidence as evidence
from scripts.greenfield import ws32_dense_frontier_execution as execution
from scripts.greenfield import ws32_dense_frontier_capture as capture
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield.prefill_window_worker import save_arrays
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from tests.greenfield.validation import test_ws32_dense_frontier_capture as captured
from tests.greenfield.validation.test_ws32_dense_frontier_execution import (
    array,
    SLOTS,
    staged,
    ORIGINAL,
)


def test_actual_journal_writer_nine_call_consumer_and_mutations(tmp_path, monkeypatch):
    run, record, _ = staged(tmp_path, monkeypatch)
    run()
    # Compiler/math/memory are fixtures in staged; writer, journal, budgeted
    # sequence and independent phase/memory re-derivation are actual code.
    monkeypatch.setattr(
        evidence.admission,
        "inspect_program",
        lambda *a: dict(passed=True, fixture=True),
    )
    original = json.loads(json.dumps(record))
    evidence.replay_execution(tmp_path, original, SLOTS)
    raw = (tmp_path / "compile_journal.jsonl").read_bytes()
    journal = [json.loads(v) for v in raw.splitlines()]
    for change in ("phase", "time", "capture_report", "memory"):
        rows = deepcopy(journal)
        if change == "phase":
            rows[-1]["stage"] = "fake_completion"
        elif change == "time":
            rows[-1]["monotonic_seconds"] = 0
        elif change == "capture_report":
            next(r for r in rows if r["stage"] == "inspected")["report"] = {
                "passed": True
            }
        else:
            next(r for r in rows if r["stage"] == "compiled")["compiled_memory"][
                "temp_size_in_bytes"
            ] += 1
        changed = b"".join((json.dumps(r) + "\n").encode() for r in rows)
        (tmp_path / "compile_journal.jsonl").write_bytes(changed)
        rec = deepcopy(original)
        rec["compile_journal_sha256"] = sha256(changed).hexdigest()
        with pytest.raises(ValueError):
            evidence.replay_execution(tmp_path, rec, SLOTS)
    (tmp_path / "compile_journal.jsonl").write_bytes(raw)
    for change in ("missing", "memory", "duration", "uncompleted"):
        rec = deepcopy(original)
        if change == "missing":
            rec["call_evidence"].pop()
        elif change == "memory":
            rec["call_evidence"][4]["compiled_memory"]["dense01"][
                "temp_size_in_bytes"
            ] += 1
        elif change == "duration":
            rec["call_evidence"][4]["completed_call_seconds"] = -1
        else:
            rec["call_evidence"][4]["completed"] = False
        with pytest.raises(ValueError):
            evidence.replay_execution(tmp_path, rec, SLOTS)


def wk_capsules(root):
    record = dict(jax_process_index=3)
    bindings = {s: dict(wk={}) for s in SLOTS.values()}
    for layer in (0, 1):
        # Exact representable, distinct per-layer and per-feature weights.
        bits = np.concatenate(
            [np.full((128, 1536), 48 + 4 * f + layer, np.uint8) for f in range(4)],
            axis=1,
        )
        scales = np.asarray([[2.0 ** ((i % 5) - 2) for i in range(48)]], np.float32)
        decoded = (
            bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            * np.repeat(scales, 128, axis=1)
        ).astype(ml_dtypes.bfloat16)
        inputs = (array(bits, sharded=True), array(scales, sharded=True))
        execution.capture_wk(
            root,
            record,
            SLOTS,
            layer=layer,
            name="wk_decode",
            value=array(decoded),
            inputs=inputs,
        )
        execution.capture_wk(
            root,
            record,
            SLOTS,
            layer=layer,
            name="wk_promote",
            value=array(decoded.astype(np.float32)),
            inputs=(array(decoded),),
        )
        for slot in SLOTS.values():
            f = slot % 4
            bindings[slot]["wk"][(layer, "input0")] = evidence.digest(
                bits[:, f * 1536 : (f + 1) * 1536]
            )
            bindings[slot]["wk"][(layer, "input1")] = evidence.digest(
                scales[:, f * 12 : (f + 1) * 12]
            )
    return json.loads(json.dumps(record)), bindings


def test_actual_wk_capture_reconstruction_and_resigned_mutations(tmp_path):
    record, bindings = wk_capsules(tmp_path)
    rows = evidence.replay_wk(tmp_path, record, SLOTS, bindings)
    assert len(rows) == 8
    assert rows[0]["bf16_sha256"] != rows[4]["bf16_sha256"]
    path = tmp_path / "layer1_wk_decode.npz"
    report = record["wk_originals"]["layer1/wk_decode"]
    arrays = evidence.read_npz(path, report, limit=execution.WK_ORIGINALS_LIMIT)
    for key in ("slot0_input0", "slot0_input1", "slot0_result"):
        altered = {k: v.copy() for k, v in arrays.items()}
        altered[key].flat[0] += 1
        rec = deepcopy(record)
        r = rec["wk_originals"]["layer1/wk_decode"]
        r["npz_sha256"] = save_arrays(path, altered)
        r["bytes"] = path.stat().st_size
        with pytest.raises(ValueError, match="checkpoint leaf|reconstruction"):
            evidence.replay_wk(tmp_path, rec, SLOTS, bindings)
    save_arrays(path, arrays)
    bad = deepcopy(bindings)
    bad[0]["wk"][(1, "input0")] = bad[0]["wk"][(0, "input0")]
    with pytest.raises(ValueError, match="checkpoint leaf"):
        evidence.replay_wk(tmp_path, record, SLOTS, bad)


def test_npz_limits_hash_and_symlink(tmp_path):
    path = tmp_path / "x.npz"
    values = {"x": np.zeros((100,), np.uint16)}
    pin = dict(
        npz_sha256=save_arrays(path, values),
        bytes=path.stat().st_size,
        raw_array_bytes=200,
    )
    assert evidence.read_npz(path, pin, limit=2048)["x"].shape == (100,)
    for changes in (
        {"bytes": True},
        {"npz_sha256": "0" * 64},
        {"raw_array_bytes": 201},
    ):
        with pytest.raises(ValueError):
            evidence.read_npz(path, {**pin, **changes}, limit=2048)
    with pytest.raises(ValueError):
        evidence.read_npz(path, pin, limit=100)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        evidence.read_npz(link, pin, limit=2048)


def test_npy_declared_allocation_refuses_before_numpy_load(tmp_path, monkeypatch):
    from io import BytesIO
    import zipfile

    payload = BytesIO()
    np.lib.format.write_array_header_1_0(
        payload, dict(descr="<f4", fortran_order=False, shape=(10**12,))
    )
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("huge.npy", payload.getvalue())
    pin = dict(
        npz_sha256=sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        raw_array_bytes=0,
    )

    def forbidden(*a, **kw):
        raise AssertionError("attempted unbounded numpy allocation")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="declared payload"):
        evidence.read_npz(path, pin, limit=2048)


def test_actual_owner_capture_to_original_cache_consumer(tmp_path, monkeypatch):
    # DB604 controller owners are all outside first128: every original cache
    # byte is zero. Test exact real witness, not merely matching a pass label.
    prior, witness = protocol.load_reference(ORIGINAL, rank=0)
    slots = {v["device_id"]: v["device_slot"] for v in prior["local_device_slots"]}
    assert set(slots.values()) == {9, 13, 25, 29}
    monkeypatch.setattr(captured, "SLOTS", slots)
    originals = {}
    owners = []
    for name, count, endpoint in evidence.CAPSULES:
        arrays, report = capture.capture(
            captured.outputs(),
            local_slots=slots,
            process_index=3,
            count=count,
            keep_caches=endpoint,
        )
        path = tmp_path / f"{name}.npz"
        report.update(
            npz_sha256=save_arrays(path, arrays),
            npz_bytes=path.stat().st_size,
            raw_array_bytes=sum(v.nbytes for v in arrays.values()),
        )
        _atomic_json(tmp_path / f"{name}.json", report)
        originals[name] = report
        if endpoint:
            owners.extend(
                evidence.compare_owner(
                    witness, branch=name, slot=s, caches=capture.cache_bits(arrays, s)
                )
                for s in slots.values()
            )
    comparison = dict(
        owners=owners,
        reproduced=True,
        model_calls=5,
        wk_calls=4,
        numerical_promotion=False,
        performance_claim=False,
    )
    _atomic_json(tmp_path / "comparison.json", comparison)
    record = json.loads(
        json.dumps(
            dict(
                dense_frontier=dict(
                    complete=True,
                    originals=originals,
                    comparison=comparison,
                    numerical_promotion=False,
                    performance_claim=False,
                )
            )
        )
    )
    report, hashes = evidence.replay_outputs(tmp_path, record, slots, witness)
    assert report["reproduced"] and len(report["owners"]) == 8 and len(hashes) == 408
    # Match the real TPU case: runtime device order differs from checkpoint
    # owner order. Only record order changes, not row/DSA/cache byte order.
    reordered = deepcopy(record)
    reordered["dense_frontier"]["comparison"]["owners"].reverse()
    _atomic_json(
        tmp_path / "comparison.json", reordered["dense_frontier"]["comparison"]
    )
    replayed, _ = evidence.replay_outputs(tmp_path, reordered, slots, witness)
    assert replayed == report
    for mode in ("duplicate", "missing", "wrong_slot", "value"):
        bad = deepcopy(reordered)
        values = bad["dense_frontier"]["comparison"]["owners"]
        if mode == "duplicate":
            values[0] = deepcopy(values[1])
        elif mode == "missing":
            values.pop()
        elif mode == "wrong_slot":
            values[0]["slot"] = 31
        else:
            values[0]["reproduced"] = False
        _atomic_json(tmp_path / "comparison.json", bad["dense_frontier"]["comparison"])
        with pytest.raises(ValueError):
            evidence.replay_outputs(tmp_path, bad, slots, witness)
    _atomic_json(tmp_path / "comparison.json", comparison)
    path = tmp_path / "wide_final.npz"
    original_report = record["dense_frontier"]["originals"]["wide_final"]
    values = evidence.read_npz(
        path,
        original_report,
        limit=evidence.worker.ORIGINALS_LIMIT,
        size_key="npz_bytes",
    )
    for key in ("slot9_layer0__health", "slot9_layer1__kv", "slot29_layer0__output"):
        changed = {k: v.copy() for k, v in values.items()}
        changed[key].flat[0] = (
            False if key.endswith("health") else 0x7FC0 if key.endswith("output") else 1
        )
        rec = deepcopy(record)
        r = rec["dense_frontier"]["originals"]["wide_final"]
        r.update(npz_sha256=save_arrays(path, changed), npz_bytes=path.stat().st_size)
        _atomic_json(tmp_path / "wide_final.json", r)
        with pytest.raises(ValueError):
            evidence.replay_outputs(tmp_path, rec, slots, witness)


def test_actual_metadata_binding_without_payload(monkeypatch):
    opened = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix != ".safetensors"
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    bindings = evidence.checkpoint_bindings(Path.cwd())
    assert set(bindings) == set(range(32))
    assert all(
        len(b["selected"]) == 55 and len(b["wk"]) == 4 for b in bindings.values()
    )
    assert bindings[0]["wk"][(0, "input0")] != bindings[0]["wk"][(1, "input0")]
