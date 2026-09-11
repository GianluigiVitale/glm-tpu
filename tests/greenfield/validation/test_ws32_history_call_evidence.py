"""Append-once evidence preserves parent calls, failure causes and exact bytes."""

import json

import pytest

from scripts.greenfield import ws32_history_call_evidence as evidence


def entry(index=0, completed=True):
    return dict(phase=f"history/{index}/candidate", graph="candidate_b128", completed=completed,
                census=dict(devices=[dict(device_id=12, buffers=[dict(bytes=128)])]),
                compiled_memory=dict(temp_size_in_bytes=256), post_memory=[])


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "MAX_CALLS", 3)
    references = []
    used = 0
    for i in range(3):
        ref = evidence.preserve_call(tmp_path, entry(i), index=i, used_bytes=used)
        references.append(ref)
        used += ref["original"]["bytes"]
    record = dict(call_evidence=references, call_evidence_layout=evidence.SCHEMA, call_original_bytes=used)
    return tmp_path, record


def test_originals_roundtrip_and_no_overwrite(bundle):
    root, record = bundle
    assert evidence.load_calls(root, record) == tuple(entry(i) for i in range(3))
    before = (root / "call_records/call000.json").read_bytes()
    with pytest.raises(FileExistsError):
        evidence.preserve_call(root, entry(), index=0, used_bytes=0)
    assert (root / "call_records/call000.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["hash", "size", "index", "path", "incomplete", "count", "total", "symlink", "missing", "unexpected", "identity", "schema", "bool"])
def test_independent_resolution_refusals(bundle, defect):
    root, record = bundle
    reference = record["call_evidence"][0]
    path = root / reference["original"]["path"]
    if defect == "hash":
        reference["original"]["sha256"] = "0" * 64
    elif defect == "size":
        reference["original"]["bytes"] += 1
    elif defect == "index":
        reference["original"]["index"] = 1
    elif defect == "path":
        reference["original"]["path"] = "../call000.json"
    elif defect == "incomplete":
        reference["completed"] = False
    elif defect == "count":
        record["call_evidence"].pop()
    elif defect == "total":
        record["call_original_bytes"] -= 1
    elif defect == "symlink":
        path.unlink()
        path.symlink_to(root / "call_records/call001.json")
    elif defect == "missing":
        path.unlink()
    elif defect == "unexpected":
        (root / "call_records/unknown.json").write_text("{}")
    elif defect == "identity":
        reference["phase"] = "different"
    elif defect == "schema":
        reference["original"]["extra"] = True
    else:
        reference["original"]["index"] = False
    with pytest.raises(ValueError):
        evidence.load_calls(root, record)


@pytest.mark.parametrize("defect", ["per_call", "total", "index", "nan", "linked", "nested_reference"])
def test_refuse_before_creating_original(tmp_path, monkeypatch, defect):
    value, index, used = entry(), 0, 0
    if defect == "per_call":
        monkeypatch.setattr(evidence, "MAX_CALL_BYTES", 2)
    elif defect == "total":
        used = evidence.MAX_TOTAL_BYTES
    elif defect == "index":
        index = 331
    elif defect == "nan":
        value["bad"] = float("nan")
    elif defect == "linked":
        other = tmp_path / "other"
        other.mkdir()
        (tmp_path / "call_records").symlink_to(other, target_is_directory=True)
    else:
        value["original"] = {}
    with pytest.raises(ValueError):
        evidence.preserve_call(tmp_path, value, index=index, used_bytes=used)
    assert not (tmp_path / "call_records/call000.json").exists()


@pytest.mark.parametrize("failure", [None, "dispatch", "post_memory", "original_write"])
def test_parent_execution_and_refusal_are_preserved(tmp_path, monkeypatch, failure):
    calls = object.__new__(evidence.HistoryCalls)
    calls.root = tmp_path
    calls.record = dict(call_evidence=[])
    phases = []
    callbacks = []

    def phase(name, fn):
        phases.append(name)
        return fn()
    calls.phase = phase

    def parent(self, phase, name, values, *, preserve):
        assert values == ("input",)
        value = entry(len(self.record["call_evidence"]))
        value.update(phase=phase, graph=name)
        self.record["call_evidence"].append(value)
        if failure == "dispatch":
            value["completed"] = False
            raise RuntimeError("dispatch failed")
        preserve("result")
        if failure == "post_memory":
            raise RuntimeError("post memory failed")
        return "result"
    monkeypatch.setattr(evidence.BudgetedCalls, "call", parent)
    if failure == "original_write":
        monkeypatch.setattr(evidence, "preserve_call", lambda *a, **k: (_ for _ in ()).throw(OSError("disk failed")))
    if failure is None:
        assert calls.call("phase", "candidate_b128", ("input",), preserve=callbacks.append) == "result"
    else:
        with pytest.raises((RuntimeError, OSError), match={"dispatch": "dispatch", "post_memory": "post memory", "original_write": "disk"}[failure]):
            calls.call("phase", "candidate_b128", ("input",), preserve=callbacks.append)
    assert callbacks == ([] if failure == "dispatch" else ["result"])
    assert phases == ["phase/original_preflight", "phase/original_retained"]
    value = calls.record["call_evidence"][0]
    if failure == "original_write":
        assert "original" not in value and value["completed"] is True
        assert calls.record["call_original_error"].endswith("disk failed")
    else:
        assert set(value) == {"original", "phase", "graph", "completed"}
        saved = json.loads((tmp_path / value["original"]["path"]).read_bytes())
        assert saved["census"] == entry()["census"]
        assert saved["completed"] == (failure != "dispatch")


def test_compact_runner_growth_is_linear(tmp_path):
    references, used = [], 0
    for i in range(331):
        value = entry(i)
        value["census"]["devices"][0]["buffers"] *= 100
        ref = evidence.preserve_call(tmp_path, value, index=i, used_bytes=used)
        references.append(ref)
        used += ref["original"]["bytes"]
    # Full original buffer rows never leak into the compact runner snapshots.
    assert len(json.dumps(references)) < 120_000
    assert all("census" not in ref for ref in references)
    record = dict(call_evidence=references, call_evidence_layout=evidence.SCHEMA, call_original_bytes=used)
    assert len(evidence.load_calls(tmp_path, record)) == 331


@pytest.mark.parametrize("peer_stage", ["original_preflight", "original_retained"])
def test_peer_evidence_refusal_stops_continuation(tmp_path, monkeypatch, peer_stage):
    calls = object.__new__(evidence.HistoryCalls)
    calls.root, calls.record = tmp_path, dict(call_evidence=[])
    dispatched = []

    def phase(name, action):
        result = action()
        if name.endswith(peer_stage):
            raise RuntimeError("peer refused")
        return result
    calls.phase = phase

    def parent(self, phase, name, values, *, preserve):
        dispatched.append(name)
        self.record["call_evidence"].append(entry())
        preserve("completed")
        return "completed"
    monkeypatch.setattr(evidence.BudgetedCalls, "call", parent)
    retained = []
    with pytest.raises(RuntimeError, match="peer refused"):
        calls.call("history/0", "candidate_b128", (), preserve=retained.append)
    assert bool(dispatched) == bool(retained) == (peer_stage == "original_retained")
    assert (tmp_path / "call_records/call000.json").exists() == (peer_stage == "original_retained")


def test_original_device_failure_survives_independent_publication_failure(tmp_path, monkeypatch):
    calls = object.__new__(evidence.HistoryCalls)
    calls.root, calls.record = tmp_path, dict(call_evidence=[])
    calls.phase = lambda _, action: action()
    def parent(self, *args, **kwargs):
        self.record["call_evidence"].append(entry(completed=False))
        raise RuntimeError("original dispatch cause")
    monkeypatch.setattr(evidence.BudgetedCalls, "call", parent)
    monkeypatch.setattr(evidence, "preserve_call", lambda *a, **k: (_ for _ in ()).throw(OSError("separate disk cause")))
    with pytest.raises(RuntimeError, match="original dispatch cause"):
        calls.call("phase", "candidate_b128", (), preserve=lambda _: None)
    assert calls.record["call_original_error"] == "OSError: separate disk cause"
    assert "original" not in calls.record["call_evidence"][0]
