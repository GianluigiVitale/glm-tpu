"""Fast driver refusals with post-dispatch faults; no TPU or compilation."""

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_worker as worker


def _rows(count=1):
    return {layer: {**{field: np.zeros((count, 8), np.float32)
                      for field in worker.FIELDS},
                   "health": np.ones((count, 4), np.bool_)}
            for layer in worker.LAYERS}


def test_boundary_comparison_is_byte_exact_including_signed_zero():
    left, right = _rows(), _rows()
    right[3]["route_weights"][0, 2] = np.float32(-0.)
    report = worker.compare_rows(left, right, offset=31)
    assert report["equal"] is False
    assert report["first"] == dict(layer=3, field="route_weights", position=31)
    assert report["layers"]["3"]["route_weights"]["max_abs_difference"] == 0


class Calls:
    def __init__(self, root, error=None):
        self.root = root
        self.programs = {name: object() for name in (*protocol.FRONTIER_PROGRAMS, "observer")}
        self.record = {"jax_process_index": 0}
        self.local_slots = {12: 9, 14: 13, 13: 25, 15: 29}
        self.phases = []
        self.error = error
        self.dispatched = 0

    def phase(self, name, action):
        self.phases.append(name)
        return action()

    def call(self, phase, name, values, *, preserve):
        self.dispatched += 1
        preserve(SimpleNamespace(healthy=np.asarray(True)))
        raise RuntimeError(self.error or "injected postflight failure")


def _kwargs(calls):
    prompt = np.asarray([30], np.int32)
    originals = {branch: {field: np.zeros((1,), np.int32)
                         for field in worker.OBSERVER_FIELDS}
                 for branch in protocol.BRANCHES}
    return dict(calls=calls, mesh=None, config=SimpleNamespace(context_capacity=8192, page_count=16),
                plan=protocol.plan(1), prompt_tokens=prompt, embedding=None,
                layers=(None,) * 7, observer_layers=(None,) * 7, wk=(None,) * 4,
                exact=(None,) * 4, rope=None, originals=originals,
                expected_prompt_sha256=sha256(prompt.tobytes()).hexdigest())


def _patch_host_work(monkeypatch, *, errors=()):
    monkeypatch.setattr(worker, "_replicated", lambda mesh, value: value)
    monkeypatch.setattr(worker, "fresh_caches", lambda *a: object())
    monkeypatch.setattr(worker, "independent", lambda *a: None)
    monkeypatch.setattr(worker, "block_values", lambda *a: ())
    monkeypatch.setattr(worker, "observer_values", lambda *a: ())
    monkeypatch.setattr(worker, "replicated_host", lambda value, **k: value)
    monkeypatch.setattr(worker, "read_boundaries", lambda result, count, **k:
                        (_rows(count), {"feature_columns": [[2, 4]]}, list(errors)))


@pytest.mark.parametrize("error", ["postflight memory refusal", "phase publication failure", "peer consensus failure"])
def test_completed_rows_survive_post_callback_failure(tmp_path, monkeypatch, error):
    _patch_host_work(monkeypatch)
    calls = Calls(tmp_path, error)
    with pytest.raises(RuntimeError, match=error):
        worker.execute_history(**_kwargs(calls))
    report = json.loads((tmp_path / "history_report.json").read_text())
    assert report["complete"] is False
    assert report["refused_step"]["error"] == error
    assert set(report["retained"]) == {"refused_completed_step0"}
    assert "history/initial_health" in calls.phases
    assert calls.dispatched == 1
    with np.load(tmp_path / "refused_completed_step0.npz", allow_pickle=False) as values:
        np.testing.assert_array_equal(values["layer0_update"], _rows()[0]["update"])


@pytest.mark.parametrize("name,symlink", [("history_report.json", False), ("observer_candidate.npz", False),
                                        ("observer_candidate.npz", True), ("observer_candidate.npz.pending", True)])
def test_prior_original_or_dangling_symlink_refuses_before_placement(tmp_path, monkeypatch, name, symlink):
    target = tmp_path / name
    if symlink:
        target.symlink_to(tmp_path / "missing")
    else:
        target.write_bytes(b"prior evidence")
    def forbidden(*args):
        raise AssertionError("must refuse before array placement")
    monkeypatch.setattr(worker, "_replicated", forbidden)
    calls = Calls(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        worker.execute_history(**_kwargs(calls))
    assert calls.dispatched == 0
    assert target.is_symlink() if symlink else target.read_bytes() == b"prior evidence"


def test_nonfinite_capture_is_preserved_as_unhealthy_not_witness(tmp_path, monkeypatch):
    _patch_host_work(monkeypatch, errors=("layer0/slot9/update/nonfinite",))
    calls = Calls(tmp_path)
    with pytest.raises(ValueError, match="unhealthy"):
        worker.execute_history(**_kwargs(calls))
    report = json.loads((tmp_path / "history_report.json").read_text())
    assert report["unhealthy"]["capture_errors"] == ["layer0/slot9/update/nonfinite"]
    assert set(report["retained"]) == {"unhealthy_step0"}
    assert calls.dispatched == 1 and not report["complete"]


def test_default_prompt_pin_cannot_be_omitted(tmp_path, monkeypatch):
    calls = Calls(tmp_path)
    kwargs = _kwargs(calls)
    del kwargs["expected_prompt_sha256"]
    with pytest.raises(ValueError, match="pinned original"):
        worker.execute_history(**kwargs)
    assert calls.dispatched == 0


@pytest.mark.parametrize("stage", ["history", "observer"])
def test_replica_conflict_retains_offending_pair_before_refusal(tmp_path, monkeypatch, stage):
    _patch_host_work(monkeypatch)
    first = np.asarray([[0., 1.]], np.float32)
    other = np.asarray([[-0., 1.]], np.float32)
    def shard(device):
        return SimpleNamespace(device=SimpleNamespace(id=device), index=(slice(None), slice(None)))
    error = worker.ReplicaMismatch(stage, (shard(12), first), (shard(14), other),
                                   global_shape=(1, 2), local_slots={12: 9, 14: 13})
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(worker, "read_boundaries" if stage == "history" else "capture_observation", fail)
    monkeypatch.setattr(worker, "cache_digests", lambda *a, **k: {
        family: {"0": {"9": "same"}} for family in worker.CACHE_FAMILIES})
    class CompletedCalls(Calls):
        def call(self, phase, name, values, *, preserve):
            self.dispatched += 1
            result = SimpleNamespace(healthy=np.asarray(True), caches=object())
            preserve(result)
            return result
    calls = CompletedCalls(tmp_path)
    with pytest.raises(worker.ReplicaMismatch):
        worker.execute_history(**_kwargs(calls))
    report = json.loads((tmp_path / "history_report.json").read_text())
    label = "refused_replica_step0" if stage == "history" else "refused_replica_observer_candidate"
    assert set(report["retained"]) == {label}
    assert report["replica_failure"] == error.metadata
    assert report["complete"] is False
    with np.load(tmp_path / f"{label}.npz", allow_pickle=False) as arrays:
        assert arrays["replica0"].tobytes() == first.tobytes()
        assert arrays["replica1"].tobytes() == other.tobytes()
        assert arrays["replica0"].tobytes() != arrays["replica1"].tobytes()
