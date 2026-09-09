"""Saved-row assembly and actual voted-call lifecycle; fixture math/memory only."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import ws32_dense_canonical as candidate
from scripts.greenfield.prefill_layer_numerical import FIELDS
from tests.greenfield.validation.test_ws32_dense_frontier_worker import (
    setup as dense_setup,
)


def retained(slots=(0, 4, 16, 20)):
    originals = {name: {} for name in candidate.NARROW}
    expected = {}
    for slot in slots:
        for layer in (0, 1):
            for field in FIELDS:
                key = f"slot{slot}_layer{layer}__{field}"
                if field in ("kv", "index", "repair"):
                    # Small fixture widths; real capture owns production geometry.
                    value = np.full((16, 64, 3), slot + layer, np.uint16)
                    originals[candidate.NARROW[-1]][key] = value
                    expected[key] = value.copy()
                    continue
                if field == "health":
                    expected[key] = np.ones(128, bool)
                    for name in candidate.NARROW:
                        originals[name][key] = np.ones(128, bool)
                    continue
                dtype = (
                    np.float32
                    if field in ("scores", "route_weights")
                    else (
                        np.int32
                        if field in ("positions", "counts", "routes")
                        else np.uint16
                    )
                )
                value = np.arange(128, dtype=dtype) + slot * 128 + layer * 4096
                if field != "counts":
                    value = value[:, None]
                expected[key] = value
                for i, name in enumerate(candidate.NARROW):
                    originals[name][key] = value[i * 32 : (i + 1) * 32].copy()
    return originals, expected


def test_original_rows_full_health_and_only_final_cache_are_compared():
    originals, expected = retained()
    result = candidate.compare(expected, originals, (20, 0, 16, 4))
    assert result["passed"] and result["slots"] == [0, 4, 16, 20]
    assert result["arrays"] == 96
    assert not result["numerical_promotion"] and not result["own8k_pass"]


@pytest.mark.parametrize("field", FIELDS)
def test_candidate_any_field_mismatch_refuses(field):
    originals, expected = retained()
    value = expected[f"slot0_layer1__{field}"]
    value.flat[-1] = not value.flat[-1] if field == "health" else value.flat[-1] + 1
    with pytest.raises(ValueError, match="realization differs"):
        candidate.compare(expected, originals, (0, 4, 16, 20))


@pytest.mark.parametrize(
    "change", ["missing", "extra", "owner", "health_padding", "order", "shape"]
)
def test_retained_metadata_and_padding_health_fail_closed(change):
    originals, expected = retained()
    slots = (0, 4, 16, 20)
    first = originals["narrow_32"]
    if change == "missing":
        first.pop("slot0_layer0__output")
    elif change == "extra":
        first["unknown"] = np.zeros(1)
    elif change == "owner":
        slots = (0, 4, 16, 16)
    elif change == "health_padding":
        first["slot0_layer0__health"][100] = False
    elif change == "order":
        originals["narrow_32"], originals["narrow_64"] = originals["narrow_64"], first
    else:
        first["slot0_layer0__output"] = first["slot0_layer0__output"][:31]
    with pytest.raises(ValueError):
        candidate.compare(expected, originals, slots)


def setup(tmp_path, monkeypatch, failure=None):
    calls, config, prompt, events = dense_setup(tmp_path, monkeypatch, failure)
    calls.budgeter = candidate.memory_budget
    calls.record["protocol"] = candidate.PROTOCOL
    program = calls.programs["dense01"]
    calls.programs = {name: program for name in candidate.PROGRAMS}
    originals, expected = retained(tuple(calls.local_slots.values()))

    def capture(*args, **kwargs):
        arrays = deepcopy(expected)
        if failure == "reproduction":
            next(v for k, v in arrays.items() if k.endswith("__output"))[0, 0] += 1
        return arrays, dict(
            count=128, keep_caches=True, valid=failure != "health", errors=[]
        )

    monkeypatch.setattr(candidate.original, "capture", capture)
    kwargs = dict(
        mesh=None,
        config=config,
        prompt_tokens=prompt,
        embedding=None,
        layers=None,
        wk=None,
        rope=None,
        originals=originals,
    )
    return calls, kwargs, events


def test_exact_five_calls_one_original_and_independent_replay(tmp_path, monkeypatch):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    candidate.execute_after_wk(calls, **kwargs)
    assert [(r["phase"], r["graph"]) for r in calls.record["call_evidence"]] == list(
        candidate.CALLS
    )
    assert len([e for e in events if e[0] == "dense_dispatch"]) == 1
    assert len(list(tmp_path.glob("*.npz"))) == 1
    report = json.loads((tmp_path / "comparison.json").read_text())
    with np.load(tmp_path / (candidate.CAPSULE + ".npz"), allow_pickle=False) as arrays:
        assert report == candidate.compare(
            arrays, kwargs["originals"], tuple(calls.local_slots.values())
        )
    assert calls.record["dense_canonical"]["complete"]


@pytest.mark.parametrize(
    "failure", ["dispatch", "health", "post_memory", "reproduction"]
)
def test_no_successor_completed_originals_survive_refusals(
    tmp_path, monkeypatch, failure
):
    calls, kwargs, events = setup(tmp_path, monkeypatch, failure)
    with pytest.raises(ValueError):
        candidate.execute_after_wk(calls, **kwargs)
    assert len([e for e in events if e[0] == "dense_dispatch"]) == 1
    assert (tmp_path / (candidate.CAPSULE + ".npz")).exists() == (failure != "dispatch")
    assert events[-1] == ("vote", False)
    assert not calls.record["dense_canonical"]["complete"]


@pytest.mark.parametrize("change", ["protocol", "wk", "prompt", "budget", "reference"])
def test_preflight_no_dispatch_for_wrong_identity(tmp_path, monkeypatch, change):
    calls, kwargs, events = setup(tmp_path, monkeypatch)
    if change == "protocol":
        calls.record["protocol"] = "old-protocol"
    elif change == "wk":
        calls.record["call_evidence"].pop()
    elif change == "prompt":
        kwargs["prompt_tokens"][0] += 1
    elif change == "budget":
        calls.budgeter = lambda *args, **kwargs: {}
    else:
        kwargs["originals"].pop("narrow_32")
    with pytest.raises(ValueError):
        candidate.execute_after_wk(calls, **kwargs)
    assert not any(e[0] == "dense_dispatch" for e in events)


@pytest.mark.parametrize("change", [None, "window", "kernel", "other_model"])
def test_fixed_source_scope_and_exclusions(monkeypatch, change):
    repo = Path(__file__).resolve().parents[3]
    original_read = Path.read_bytes
    names = tuple(candidate.MODEL_SOURCE_OVERRIDES)

    def read(path):
        raw = original_read(path)
        if change in ("window", "kernel") and path == repo / names[change == "kernel"]:
            return raw + b"# unreviewed change"
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=1 if change == "other_model" else 0)

    monkeypatch.setattr(candidate.subprocess, "run", run)
    if change is None:
        candidate.require_source(repo)
        assert commands[0][4:6] == [
            "--quiet",
            "7456bf6433e1dce966670deb252f4c64bbc5f432",
        ]
        assert [v for v in commands[0] if v.startswith(":(exclude)")] == [
            ":(exclude)" + n for n in names
        ]
        for name, expected in candidate.MODEL_SOURCE_OVERRIDES.items():
            assert sha256(original_read(repo / name)).hexdigest() == expected
    else:
        with pytest.raises(ValueError, match="source"):
            candidate.require_source(repo)


def test_metadata_defaults_keep_historical_guard(monkeypatch):
    from scripts.greenfield import ws32_rolled_prefill_compile as metadata
    from scripts.greenfield import ws32_dense_frontier_prepare as prepare

    seen = []

    def old(*args, **kwargs):
        seen.append("old")
        raise ValueError("historical_guard")

    def new(*args, **kwargs):
        seen.append("new")
        raise ValueError("canonical_guard")

    monkeypatch.setattr(metadata.admission, "require_acquired_model_source", old)
    monkeypatch.setattr(candidate, "require_source", new)
    with pytest.raises(ValueError, match="historical_guard"):
        metadata.read_metadata(Path("unused"))
    with pytest.raises(ValueError, match="canonical_guard"):
        metadata.read_metadata(Path("unused"), canonical_dense=True)
    with pytest.raises(ValueError, match="static bool"):
        metadata.read_metadata(Path("unused"), canonical_dense=1)
    with pytest.raises(ValueError, match="static bool"):
        prepare.prepare(None, repo=Path("unused"), canonical_dense=1)
    assert seen == ["old", "new"]


@pytest.mark.parametrize(
    "change", ["rows", "dense", "moe", "rolled", "panels", "observe", "bool"]
)
def test_window_rejects_unsupported_canonical_combinations_before_compute(change):
    from glm_tpu.greenfield.kernels.ws32_prefill_window import (
        ws32_prefill_layer_window_mapped,
    )

    rows = 32 if change == "rows" else 128
    args = [None] * 19
    args[0] = np.zeros((rows, 1))
    args[16] = None if change == "dense" else object()
    args[17] = object() if change == "moe" else None
    with pytest.raises(ValueError):
        ws32_prefill_layer_window_mapped(
            *args,
            main_rope_table_rows=None,
            canonical_dense=1 if change == "bool" else True,
            rolled_prefix=change != "rolled",
            expert_panels=change != "panels",
            _observe=(lambda *args: None) if change == "observe" else None,
        )
