"""Real four-graph writer/journal/WK and18-call orchestration; fixture compute."""

from hashlib import sha256
import json
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_execution as execution
from scripts.greenfield import ws32_dense_norm_prepare as preparation
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from tests.greenfield.validation.test_ws32_dense_frontier_execution import (
    array,
    wk_inputs,
)
from tests.greenfield.validation.test_ws32_dense_norm_worker import setup


def staged(tmp_path, monkeypatch, failure=None):
    initial, kwargs, events = setup(
        tmp_path,
        monkeypatch,
        "retained" if failure == "retained" else "own" if failure == "own" else None,
    )
    record = initial.record
    record.update(programs={}, diagnostic_only=True, code_hash="a" * 40, launch_rank=0)
    captured_programs = initial.programs
    operands = (wk_inputs(), wk_inputs())
    layers = tuple(
        NS(dense=None, dsa=NS(wk_bits_local=w, wk_scale_local=s)) for w, s in operands
    )

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return captured_programs[self.name].memory_analysis()

        def as_text(self):
            return self.name + " optimized fixture"

        def __call__(self, *args):
            if self.name in ("dense01_norm", "dense_suffix"):
                return captured_programs[self.name](*args)
            events.append(("wk_dispatch", self.name))
            if failure == "wk":
                raise ValueError("fixture WK failure")
            return array(
                np.zeros(
                    (128, 6144),
                    ml_dtypes.bfloat16 if self.name == "wk_decode" else np.float32,
                )
            )

    class Lowered:
        def __init__(self, name):
            self.name = name

        def compiler_ir(self, *, dialect):
            assert dialect == "stablehlo"
            return self.name + " stable fixture"

        def compile(self):
            events.append(("compile", self.name))
            if failure == "compile_" + self.name:
                raise ValueError("fixture compiler failure")
            return Program(self.name)

    jobs = tuple(
        (name, NS(lower=lambda *args, name=name: Lowered(name)), ())
        for name in protocol.PROGRAMS
    )
    monkeypatch.setattr(preparation, "compiler_programs", lambda *args: jobs)

    def inspect(name, stable, optimized, memory):
        assert all(
            (tmp_path / f"{g}.optimized_hlo.txt").is_file() for g in protocol.PROGRAMS
        )
        events.append(("admit", name))
        if failure == "admission":
            raise ValueError("fixture HLO refusal")
        return dict(passed=True, fixture=True)

    def consensus(ok):
        events.append(("vote", ok))
        if (
            failure == "peer_admission"
            and record.get("current_phase") == "dense/admission"
        ):
            return False
        return ok

    if failure == "finalize":
        old_close = execution.NormJournal.close

        def close(self):
            old_close(self)
            raise ValueError("fixture journal close failure")

        monkeypatch.setattr(execution.NormJournal, "close", close)

    if failure == "missing_originals":
        kwargs["originals"] = None
    elif failure == "mixed_originals":
        from scripts.greenfield.ws32_dense_frontier_protocol import PROTOCOL

        record["protocol"] = PROTOCOL

    def run():
        execution.execute(
            root=tmp_path,
            record=record,
            mesh=None,
            prepared=NS(config=kwargs["config"]),
            embedding=None,
            layers=layers,
            tokens=kwargs["prompt_tokens"],
            rope=None,
            witness={},
            local_slots=initial.local_slots,
            consensus=consensus,
            inspect_program=inspect,
            norm_originals=kwargs["originals"],
        )

    return run, record, events


def test_compiler_to_all18calls_and_capsules_actual_composition(tmp_path, monkeypatch):
    run, record, events = staged(tmp_path, monkeypatch)
    run()
    assert [(r["phase"], r["graph"]) for r in record["call_evidence"]] == list(
        protocol.CALLS
    )
    assert [e[0] for e in events if e[0] in ("compile", "admit")] == ["compile"] * 4 + [
        "admit"
    ] * 4
    assert all(r["completed"] for r in record["call_evidence"])
    assert record["dense_norm"]["complete"] and len(list(tmp_path.glob("*.npz"))) == 23
    raw = (tmp_path / "compile_journal.jsonl").read_bytes()
    assert sha256(raw).hexdigest() == record["compile_journal_sha256"]
    journal = [json.loads(line) for line in raw.splitlines()]
    assert {r["artifact_kind"] for r in journal} == {
        execution.NormJournal.artifact_kind
    }
    assert [r["graph"] for r in journal if r["stage"] == "inspected"] == list(
        protocol.PROGRAMS
    )


@pytest.mark.parametrize(
    "failure",
    [
        *(f"compile_{n}" for n in protocol.PROGRAMS),
        "admission",
        "peer_admission",
        "wk",
        "retained",
        "own",
        "finalize",
        "missing_originals",
        "mixed_originals",
    ],
)
def test_failure_preservation_and_no_unreviewed_successor(
    tmp_path, monkeypatch, failure
):
    run, record, events = staged(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert record["status"] == "DIAGNOSTIC_FAILED"
    assert record["acquisition_phases"]["dense/finalize"]["status"] in (
        "COMPLETE",
        "FAILED",
    )
    if failure.startswith("compile") or failure in (
        "admission",
        "peer_admission",
        "missing_originals",
        "mixed_originals",
    ):
        assert not any(e[0].endswith("dispatch") for e in events)
    if failure in ("admission", "peer_admission"):
        assert all(
            (tmp_path / f"{g}.optimized_hlo.txt").exists() for g in protocol.PROGRAMS
        )
    if failure == "retained":
        assert (
            len(record["call_evidence"]) == 9
            and (tmp_path / "narrow_128_norm.npz").exists()
        )
    if failure == "own":
        assert (
            len(record["call_evidence"]) == 14
            and (tmp_path / "own_narrow_128.npz").exists()
        )
    if failure == "finalize":
        assert (
            len(record["call_evidence"]) == 18 and (tmp_path / "cross_96.npz").exists()
        )
