"""Replay refused numerical originals; no compile, payload read or TPU dispatch."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_history_admission as admission

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def originals():
    raw = (REPO / admission.NUMERICAL_RECEIPT).read_bytes()
    assert sha256(raw).hexdigest() == admission.NUMERICAL_RECEIPT_SHA256
    receipt = json.loads(raw)
    root = Path(receipt["source_local_root"]) / "rank0"
    return {name: ((root / f"{name}.stablehlo.mlir").read_text(),
                   (root / f"{name}.optimized_hlo.txt").read_text(),
                   saved["compiled_memory"])
            for name, saved in receipt["graphs"].items()}


@pytest.mark.parametrize("name", admission.compiler.PROGRAMS)
def test_actual_numerical_original_passes_complete_admission(name, originals):
    stable, optimized, memory = originals[name]
    report = admission.inspect_program(name, stable, optimized, memory, repo=REPO)
    assert report == json.loads(json.dumps(report))
    assert report["passed"] and report["structure"]["helpers"]["passed"]
    assert report["stablehlo_sha256"] == sha256(stable.encode()).hexdigest()
    assert report["optimized_hlo_sha256"] == sha256(optimized.encode()).hexdigest()
    assert not report["numerical_promotion"] and not report["performance_claim"]
    if name.startswith("exact_"):
        assert report["structure"]["materializer"]["passed"]


@pytest.mark.parametrize("defect", ("whitespace", "host_frame", "model_location", "instruction", "raw", "memory"))
def test_any_unregistered_numerical_change_refuses_before_structure(defect, originals, monkeypatch):
    stable, optimized, memory = originals["candidate_b128"]
    memory = dict(memory)
    if defect == "whitespace":
        optimized += "\n"
    elif defect == "host_frame":
        assert "history_entry.py" in optimized
        optimized = optimized.replace("history_entry.py", "other_entry.py", 1)
    elif defect == "model_location":
        assert "ws32_history_frontier.py" in optimized
        optimized = optimized.replace("ws32_history_frontier.py", "other_frontier.py", 1)
    elif defect == "instruction":
        assert "constant(0)" in optimized
        optimized = optimized.replace("constant(0)", "constant(1)", 1)
    elif defect == "raw":
        stable += "\n"
    else:
        memory["temp_size_in_bytes"] += 128
    monkeypatch.setattr(admission, "inspect_structure", lambda *a, **k: pytest.fail("reached structure"))
    with pytest.raises(ValueError):
        admission.inspect_program("candidate_b128", stable, optimized, memory, repo=REPO)


@pytest.mark.parametrize("defect", ("missing", "changed", "linked"))
def test_numerical_receipt_is_mandatory_and_byte_bound(defect, originals, tmp_path, monkeypatch):
    original_registration = admission.registration(REPO)
    monkeypatch.setattr(admission, "registration", lambda repo: original_registration)
    target = tmp_path / admission.NUMERICAL_RECEIPT
    target.parent.mkdir(parents=True)
    if defect == "changed":
        target.write_bytes((REPO / admission.NUMERICAL_RECEIPT).read_bytes() + b"\n")
    elif defect == "linked":
        target.symlink_to(REPO / admission.NUMERICAL_RECEIPT)
    monkeypatch.setattr(admission, "inspect_structure", lambda *a, **k: pytest.fail("reached structure"))
    with pytest.raises(ValueError, match="numerical identity receipt"):
        admission.inspect_program("candidate_b128", *originals["candidate_b128"], repo=tmp_path)
