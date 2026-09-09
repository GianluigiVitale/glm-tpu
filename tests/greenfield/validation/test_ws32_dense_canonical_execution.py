"""Actual compiler/journal/five-call composition with fixture math/counters."""

from hashlib import sha256
import json
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_execution as execution
from scripts.greenfield import ws32_dense_canonical as canonical
from tests.greenfield.validation.test_ws32_dense_canonical import setup
from tests.greenfield.validation.test_ws32_dense_frontier_execution import (
    array,
    wk_inputs,
)


@pytest.mark.parametrize(
    "first",
    [
        "ws32_dense_canonical_admission",
        "ws32_dense_canonical",
        "ws32_dense_frontier_execution",
    ],
)
def test_fresh_process_import_order(first):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from scripts.greenfield import {first}; from scripts.greenfield.ws32_dense_frontier_execution import CanonicalJournal; from scripts.greenfield.ws32_dense_canonical import PROTOCOL; assert CanonicalJournal.protocol_id == PROTOCOL",
        ],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def staged(tmp_path, monkeypatch, failure=None):
    initial, kwargs, events = setup(tmp_path, monkeypatch, failure)
    record = initial.record
    record.update(programs={}, diagnostic_only=True, code_hash="a" * 40, launch_rank=0)
    captured = initial.programs
    originals = {**kwargs["originals"], "wide_final": {}}
    operands = (wk_inputs(), wk_inputs())
    layers = tuple(NS(dsa=NS(wk_bits_local=w, wk_scale_local=s)) for w, s in operands)

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return captured[self.name].memory_analysis()

        def as_text(self):
            return self.name + " optimized fixture"

        def __call__(self, *args):
            if self.name == canonical.GRAPH:
                return captured[self.name](*args)
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
        (n, NS(lower=lambda *args, n=n: Lowered(n)), ()) for n in canonical.PROGRAMS
    )
    monkeypatch.setattr(canonical, "compiler_programs", lambda *args: jobs)

    def inspect(name, stable, optimized, memory):
        assert all(
            (tmp_path / f"{g}.optimized_hlo.txt").is_file() for g in canonical.PROGRAMS
        )
        events.append(("admit", name))
        if failure == "admission":
            raise ValueError("fixture HLO refusal")
        return dict(passed=True, fixture=True)

    def consensus(ok):
        events.append(("vote", ok))
        return (
            False
            if failure == "peer_admission"
            and record.get("current_phase") == "dense/admission"
            else ok
        )

    if failure == "finalize":
        old_close = execution.CanonicalJournal.close

        def close(self):
            old_close(self)
            raise ValueError("fixture journal close failure")

        monkeypatch.setattr(execution.CanonicalJournal, "close", close)
    if failure == "missing_originals":
        originals = None
    if failure == "mixed_originals":
        record["protocol"] = execution.protocol.PROTOCOL

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
            canonical_originals=originals,
        )

    return run, record, events


def test_compiler_to_four_wk_one_candidate_and_preserved_capsule(tmp_path, monkeypatch):
    run, record, events = staged(tmp_path, monkeypatch)
    run()
    assert [(r["phase"], r["graph"]) for r in record["call_evidence"]] == list(
        canonical.CALLS
    )
    assert [e[0] for e in events if e[0] in ("compile", "admit")] == ["compile"] * 3 + [
        "admit"
    ] * 3
    assert all(r["completed"] for r in record["call_evidence"])
    assert record["dense_canonical"]["complete"]
    assert len(list(tmp_path.glob("*.npz"))) == 5
    raw = (tmp_path / "compile_journal.jsonl").read_bytes()
    assert sha256(raw).hexdigest() == record["compile_journal_sha256"]
    journal = [json.loads(l) for l in raw.splitlines()]
    assert {r["artifact_kind"] for r in journal} == {
        execution.CanonicalJournal.artifact_kind
    }
    assert [r["graph"] for r in journal if r["stage"] == "inspected"] == list(
        canonical.PROGRAMS
    )
    assert len([e for e in events if e[0] == "dense_dispatch"]) == 1


@pytest.mark.parametrize(
    "failure",
    [
        *("compile_" + n for n in canonical.PROGRAMS),
        "admission",
        "peer_admission",
        "wk",
        "health",
        "reproduction",
        "finalize",
        "missing_originals",
        "mixed_originals",
    ],
)
def test_refusal_preserves_originals_and_prevents_successor(
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
    if failure.startswith("compile_") or failure in (
        "admission",
        "peer_admission",
        "missing_originals",
        "mixed_originals",
    ):
        assert not any(e[0] in ("wk_dispatch", "dense_dispatch") for e in events)
    if failure in ("admission", "peer_admission"):
        assert all(
            (tmp_path / f"{g}.optimized_hlo.txt").exists() for g in canonical.PROGRAMS
        )
    if failure in ("health", "reproduction", "finalize"):
        assert (tmp_path / (canonical.CAPSULE + ".npz")).exists()
        assert len([e for e in events if e[0] == "dense_dispatch"]) == 1
