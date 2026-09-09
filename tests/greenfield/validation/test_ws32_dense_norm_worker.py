"""Real voted calls/NPZ persistence; explicitly fixture compute and device memory."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_dense_norm_worker as worker
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from tests.greenfield.validation.test_ws32_dense_frontier_worker import (
    setup as dense_setup,
)


def setup(tmp_path, monkeypatch, failure=None):
    calls, config, prompt, events = dense_setup(tmp_path, monkeypatch, failure)
    calls.budgeter = worker.memory_budget
    calls.record["protocol"] = protocol.PROTOCOL
    old_program = calls.programs["dense01"]
    old_capture = worker.base.capture
    originals = {}

    def output(count, offset):
        return (
            np.arange(offset, offset + count, dtype=np.uint16)[:, None]
            + np.arange(2, dtype=np.uint16)[None]
        )

    def capture(result, *, count, **kw):
        arrays, report = old_capture(result, count=count, **kw)
        offset = result[0][2].offset - count
        for slot in calls.local_slots.values():
            arrays[f"slot{slot}_layer0__output"] = output(count, offset)
            arrays[f"slot{slot}_layer0__health"] = np.ones(128, bool)
        return arrays, report

    monkeypatch.setattr(worker.base, "capture", capture)
    for label, count, offset in protocol.CAPTURES:
        cache = NS(offset=offset + count)
        original, _ = capture(
            ((0, 0, cache),) * 2,
            count=count,
            keep_caches=label in ("wide_final", "narrow_128"),
        )
        originals[label] = original
    if failure == "retained":
        originals["narrow_64"]["row_output"][0, 0] = 1

    class Capture:
        memory_analysis = old_program.memory_analysis

        def __call__(self, tokens, count, offset, caches):
            out = old_program(tokens, count, offset, caches)
            return out, dict(offset=offset, count=count)

    class Suffix:
        memory_analysis = old_program.memory_analysis

        def __call__(self, packet, start, count, dense):
            events.append(("suffix_dispatch", packet["offset"], start, count))
            value = output(count, packet["offset"] + start)
            if failure == "own":
                value[0, 0] += 1
            return value, np.ones(128, bool)

    calls.programs = {
        n: Capture() if n == "dense01_norm" else Suffix() for n in protocol.PROGRAMS
    }

    def packet_capture(packet, **kw):
        return dict(packet_offset=np.asarray(packet["offset"], np.int32)), dict(
            valid=failure != "packet", count=packet["count"]
        )

    monkeypatch.setattr(worker, "capture_packet", packet_capture)
    monkeypatch.setattr(
        worker,
        "suffix_inputs",
        lambda mesh, packet, start, count, dense: (packet, start, count, dense),
    )

    def suffix_capture(result, **kw):
        return {
            f"slot{s}__{name}": value
            for s in calls.local_slots.values()
            for name, value in zip(("output", "health"), result)
        }, dict(valid=True)

    monkeypatch.setattr(worker, "capture_suffix", suffix_capture)
    return (
        calls,
        dict(
            mesh=None,
            config=config,
            prompt_tokens=prompt,
            embedding=None,
            layers=(NS(dense=None),),
            wk=None,
            rope=None,
            witness={},
            originals=originals,
        ),
        events,
    )


def test_fixed18calls_and_both_reproduction_votes_precede_interventions(
    tmp_path, monkeypatch
):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    phases = []
    old_phase = calls.phase

    def phase(name, action):
        value = old_phase(name, action)
        phases.append(name)
        return value

    calls.phase = phase
    worker.execute_after_wk(calls, **kwargs)
    assert [(e["phase"], e["graph"]) for e in calls.record["call_evidence"]] == list(
        protocol.CALLS
    )
    assert len([e for e in events if e[0] == "dense_dispatch"]) == 5
    assert len([e for e in events if e[0] == "suffix_dispatch"]) == 9
    assert len(list(tmp_path.glob("*.npz"))) == 19
    assert phases.index("norm/reproduction") < phases.index(
        "norm/own/wide_final/execute"
    )
    assert phases.index("norm/own_reproduction") < phases.index("norm/cross/0/execute")
    final = calls.record["dense_norm"]
    assert final["complete"] and final["reproduction"]["reproduced"]
    assert len(final["reproduction"]["caches"]) == 8
    assert all(r["bytes_equal"] for r in final["cross_comparison"]["comparisons"])
    assert not final["cause_claim"] and not final["numerical_promotion"]
    for name, record in final["originals"].items():
        assert json.loads((tmp_path / f"{name}.json").read_text()) == record
        with np.load(tmp_path / f"{name}.npz", allow_pickle=False) as arrays:
            assert record["raw_array_bytes"] == sum(arrays[n].nbytes for n in arrays)


@pytest.mark.parametrize(
    "failure,ncapture,nsuffix",
    [
        ("dispatch", 1, 0),
        ("health", 1, 0),
        ("packet", 1, 0),
        ("post_memory", 1, 0),
        ("retained", 5, 0),
        ("reproduction", 5, 0),
        ("own", 5, 5),
    ],
)
def test_refusal_preserves_completed_originals_and_stops_escalation(
    tmp_path,
    monkeypatch,
    failure,
    ncapture,
    nsuffix,
):
    calls, kwargs, events = setup(tmp_path, monkeypatch, failure)
    with pytest.raises(ValueError):
        worker.execute_after_wk(calls, **kwargs)
    assert len([e for e in events if e[0] == "dense_dispatch"]) == ncapture
    assert len([e for e in events if e[0] == "suffix_dispatch"]) == nsuffix
    assert (tmp_path / "wide_final.npz").exists() == (failure != "dispatch")
    assert (tmp_path / "wide_final_norm.npz").exists() == (failure != "dispatch")
    assert not (tmp_path / "cross_0.npz").exists()
    assert events[-1] == ("vote", False)
    assert not calls.record["dense_norm"]["complete"]


@pytest.mark.parametrize(
    "gate,nsuffix", [("norm/reproduction", 0), ("norm/own_reproduction", 5)]
)
def test_peer_refusal_at_reproduction_prevents_successor(
    tmp_path, monkeypatch, gate, nsuffix
):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    calls.consensus = lambda ok: ok and calls.record.get("current_phase") != gate
    with pytest.raises(RuntimeError, match="peer refused"):
        worker.execute_after_wk(calls, **kwargs)
    assert len([e for e in events if e[0] == "suffix_dispatch"]) == nsuffix
    assert not (tmp_path / "cross_0.npz").exists()


@pytest.mark.parametrize(
    "change", ["protocol", "wk", "program", "prompt", "budget", "originals"]
)
def test_preflight_refuses_before_dispatch(tmp_path, monkeypatch, change):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    if change == "protocol":
        calls.record["protocol"] = "other"
    elif change == "wk":
        calls.record["call_evidence"][0]["completed"] = False
    elif change == "program":
        calls.programs.pop("dense_suffix")
    elif change == "prompt":
        kwargs["prompt_tokens"][0] += 1
    elif change == "budget":
        calls.budgeter = lambda *a, **kw: {}
    else:
        kwargs["originals"].pop("narrow_64")
    with pytest.raises(ValueError, match="inventory"):
        worker.execute_after_wk(calls, **kwargs)
    assert not any(e[0].endswith("dispatch") for e in events)


@pytest.mark.parametrize(
    "name", ["wide_final.npz", "wide_final.json", "wide_final.npz.pending"]
)
def test_existing_original_is_never_overwritten(tmp_path, monkeypatch, name):
    calls, kwargs, _ = setup(tmp_path, monkeypatch)
    original = tmp_path / name
    original.write_bytes(b"prior evidence")
    with pytest.raises(FileExistsError):
        worker.execute_after_wk(calls, **kwargs)
    assert original.read_bytes() == b"prior evidence"


def test_dangling_pending_symlink_is_not_followed(tmp_path, monkeypatch):
    calls, kwargs, _ = setup(tmp_path, monkeypatch)
    outside = tmp_path / "not_a_capsule"
    (tmp_path / "wide_final.npz.pending").symlink_to(outside)
    with pytest.raises(FileExistsError):
        worker.execute_after_wk(calls, **kwargs)
    assert not outside.exists()


def test_budget_refuses_before_payload_write(tmp_path, monkeypatch):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(protocol, "MODEL_ORIGINALS_LIMIT", 1)
    with pytest.raises(ValueError, match="budget"):
        worker.execute_after_wk(calls, **kwargs)
    assert not list(tmp_path.glob("*.npz"))
    assert len([e for e in events if e[0] == "dense_dispatch"]) == 1


@pytest.mark.parametrize(
    "phase",
    [
        "norm/capture/wide_final/execute",
        "norm/own/wide_final/execute",
        "norm/cross/0/execute",
    ],
)
def test_publication_failure_retains_completed_capsules(tmp_path, monkeypatch, phase):
    from scripts.greenfield import prefill_window_worker as budgeted

    calls, kwargs, events = setup(tmp_path, monkeypatch)
    old = budgeted._atomic_json

    def publish(path, record):
        if record.get("current_phase") == phase:
            raise OSError("fixture runner publication failure")
        return old(path, record)

    monkeypatch.setattr(budgeted, "_atomic_json", publish)
    with pytest.raises(OSError, match="publication"):
        worker.execute_after_wk(calls, **kwargs)
    label = {
        "norm/capture/wide_final/execute": "wide_final_norm",
        "norm/own/wide_final/execute": "own_wide_final",
        "norm/cross/0/execute": "cross_0",
    }[phase]
    assert (tmp_path / f"{label}.npz").exists()
    assert calls.record["call_evidence"][-1]["completed"]
    assert not calls.record["dense_norm"]["complete"] and events[-1] == ("vote", False)
