"""Actual writer/journal/voted lifecycle with fixture compiler, never TPU."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import probe_ws32_prefill_layer as compiler
from scripts.greenfield import ws32_rolled_prefill_worker as worker

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    import jax

    assert jax.default_backend() == "cpu"
    events = []
    controls = dict(fail_compile=None, refuse_memory=False)
    originals = admission.short_acquisition(ROOT)["fleet"][0]["compiled"]
    registration = deepcopy(admission.rolled_registration(ROOT))

    class Compiled:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kwargs):
            raise AssertionError("compiler-only path invoked executable")

        def as_text(self):
            return "optimized " + self.name

    class Lowered:
        def __init__(self, name):
            self.name = name

        def compiler_ir(self, dialect):
            assert dialect == "stablehlo"
            return "stable " + self.name

        def compile(self):
            events.append(self.name)
            if controls["fail_compile"] == self.name:
                raise RuntimeError("fixture compiler failure")
            return Compiled(self.name)

    class Program:
        def __init__(self, name):
            self.name = name

        def lower(self, *args):
            assert all(isinstance(x, jax.ShapeDtypeStruct) for x in args)
            return Lowered(self.name)

        def __call__(self, *args):
            raise AssertionError("compiler-only path called jitted function")

    def memory(compiled):
        result = deepcopy(originals[compiled.name]["memory"])
        if controls["refuse_memory"] and compiled.name == worker.PROGRAMS[0]:
            result["temp_size_in_bytes"] = 1 << 31
        return result

    monkeypatch.setattr(compiler, "_compiled_memory", memory)
    monkeypatch.setattr(jax, "local_devices", lambda: [])
    monkeypatch.setattr(
        worker.preparation, "read_metadata", lambda repo: "authenticated-fixture"
    )
    metadata_pins = json.loads(
        (
            ROOT / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
        ).read_text()
    )
    pair = worker.preparation.AbstractPrefillPair(
        {n: SimpleNamespace(execute=Program(n)) for n in worker.PROGRAMS},
        {n: (jax.ShapeDtypeStruct((1,), "int32"),) for n in worker.PROGRAMS},
        metadata_pins["expected_manifest_sha256"],
        metadata_pins["source_inventory_sha256"],
    )
    monkeypatch.setattr(worker.preparation, "prepare", lambda *a, **k: pair)
    for name in worker.PROGRAMS:
        raw = ("stable " + name).encode()
        registration["graphs"][name].update(
            stablehlo_sha256=sha256(raw).hexdigest(), stablehlo_bytes=len(raw)
        )
    monkeypatch.setattr(admission, "rolled_registration", lambda repo: registration)
    record = dict(
        kernel=worker.KERNEL,
        protocol=worker.PROTOCOL,
        profile=worker.ROLLED_SHORT_PROFILE,
        prefill_mode=worker.PREFILL_MODE,
        compile_only=True,
        weights_loaded=False,
        model_executable_calls=0,
        numerical_claim=False,
        performance_claim=False,
        code_hash="c" * 40,
        launch_rank=0,
        programs={},
        status="RUNNING",
    )
    votes = []

    def consensus(ok):
        votes.append(ok)
        return ok

    def run(vote=consensus):
        worker.execute_pair(tmp_path, record, mesh=None, repo=ROOT, consensus=vote)

    def journal():
        path = tmp_path / "compile_journal.jsonl"
        assert record["compile_journal_sha256"] == sha256(path.read_bytes()).hexdigest()
        return [json.loads(line) for line in path.read_text().splitlines()]

    return SimpleNamespace(
        root=tmp_path,
        record=record,
        events=events,
        controls=controls,
        run=run,
        votes=votes,
        journal=journal,
        pair=pair,
    )


def test_both_actual_writers_and_journal_complete_without_dispatch(lifecycle):
    case = lifecycle
    case.run()
    assert case.events == list(worker.PROGRAMS)
    assert case.record["status"] == "SUCCESS"
    assert case.record["compiler_acquisition_complete"] is True
    assert case.record["model_executable_calls"] == 0
    assert case.votes == [True] * 6
    rows = case.journal()
    assert [r["stage"] for r in rows] == [
        "identity",
        "lower_compile_started",
        "compiled",
        "raw_written",
        "inspected",
        "lower_compile_started",
        "compiled",
        "raw_written",
        "inspected",
        "preserved_pair_verified",
    ]
    assert all(not r["numerical_claim"] and not r["performance_claim"] for r in rows)
    for row in rows:
        if row["stage"] == "inspected":
            assert row["report"] == worker.CAPTURE
    assert json.loads((case.root / "runner.json").read_text()) == case.record


def test_memory_refusal_happens_after_both_originals_are_preserved(lifecycle):
    case = lifecycle
    case.controls["refuse_memory"] = True
    with pytest.raises(ValueError, match="allocation"):
        case.run()
    assert case.events == list(worker.PROGRAMS)
    assert case.record["status"] == "FAILED"
    assert not case.record["compiler_acquisition_complete"]
    for name in worker.PROGRAMS:
        assert (case.root / f"{name}.optimized_hlo.txt").is_file()
    assert case.journal()[-1]["stage"] == "inspected"
    assert case.votes == [True, True, True, False, True]


@pytest.mark.parametrize("failed", worker.PROGRAMS)
def test_compile_failure_retains_partial_originals_and_closes(lifecycle, failed):
    case = lifecycle
    case.controls["fail_compile"] = failed
    with pytest.raises(RuntimeError, match="fixture compiler failure"):
        case.run()
    assert case.record["status"] == "FAILED"
    assert case.journal()[-1]["stage"] == "lower_compile_started"
    assert (case.root / f"{failed}.stablehlo.mlir").exists()
    if failed == "prefill_tail":
        assert (case.root / "prefill_chunk.optimized_hlo.txt").exists()
    assert case.votes[-2:] == [False, True]


@pytest.mark.parametrize("phase", range(6))
def test_peer_refusal_never_advances_or_leaves_success(lifecycle, phase):
    case = lifecycle
    votes = []

    def vote(ok):
        votes.append(ok)
        return ok and len(votes) != phase + 1

    with pytest.raises(RuntimeError, match="peer failed"):
        case.run(vote)
    assert case.record["status"] == "FAILED"
    assert case.record["compiler_acquisition_complete"] is False
    assert len(case.events) == min(phase, 2)
    case.journal()


def test_invalid_identity_refuses_before_metadata_or_compilation(lifecycle):
    case = lifecycle
    case.record["compile_only"] = 1
    with pytest.raises(ValueError, match="identity"):
        case.run()
    assert case.events == []
    assert not (case.root / "compile_journal.jsonl").exists()


def test_primary_failure_survives_finalize_failure(lifecycle, monkeypatch):
    case = lifecycle
    case.controls["fail_compile"] = "prefill_tail"
    original = worker.RolledCompileJournal.close

    def close(self):
        original(self)
        raise OSError("fixture close refusal")

    monkeypatch.setattr(worker.RolledCompileJournal, "close", close)
    with pytest.raises(RuntimeError, match="fixture compiler failure"):
        case.run()
    assert "fixture close refusal" in case.record["finalization_error"]
    assert "fixture compiler failure" in case.record["error"]
    assert (case.root / "prefill_chunk.optimized_hlo.txt").exists()


@pytest.mark.parametrize("metadata_failure", [False, True])
def test_actual_probe_selects_compile_only_before_runtime(
    lifecycle, monkeypatch, metadata_failure
):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils
    from scripts.greenfield import probe_ws32_prefill_budget as entry
    from scripts.greenfield import run_short_decoder_ws32 as runtime
    from scripts.greenfield.ws32_prefill_budget_campaign import topology_bindings

    case = lifecycle
    tag = "greenfield_fp8_" + worker.KERNEL + "_fixture"
    physical, captures = topology_bindings()
    initialized = []
    reads = []

    def read_metadata(repo):
        reads.append(repo)
        if metadata_failure:
            raise ValueError("fixture missing metadata")
        return "authenticated-fixture"

    def initialize(args):
        assert reads, "metadata was not checked before runtime"
        initialized.append(True)
        return (
            jax,
            None,
            physical,
            SimpleNamespace(topology_hash=entry.TOPOLOGY_SHA),
            entry.FLEET_SHA,
        )

    monkeypatch.setattr(worker.preparation, "read_metadata", read_metadata)
    monkeypatch.setattr(entry, "run_root", lambda tag: case.root)
    monkeypatch.setattr(entry, "_git_head", lambda: "c" * 40)
    monkeypatch.setattr(entry.socket, "gethostname", lambda: captures[4]["hostname"])
    monkeypatch.setattr(
        entry, "version", lambda name: {"jax": "0.10.1", "libtpu": "0.0.41"}[name]
    )
    monkeypatch.setattr(runtime, "_initialize_runtime", initialize)
    monkeypatch.setattr(
        jax, "local_devices", lambda: [SimpleNamespace(id=i) for i in range(4)]
    )
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(compiler, "_memory_stats", lambda device: {"bytes_in_use": 0})
    monkeypatch.setattr(
        multihost_utils, "process_allgather", lambda value: np.repeat(value, 8)
    )
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", tag)
    args = [
        "--expected-code-hash",
        "c" * 40,
        "--coordinator-address",
        "127.0.0.1:8476",
        "--process-id",
        "4",
        "--output-dir",
        str(case.root / "rank4"),
    ]
    if metadata_failure:
        with pytest.raises(ValueError, match="missing metadata"):
            entry.main(args)
        assert initialized == case.events == []
        return
    assert entry.main(args) == 0
    record = json.loads((case.root / "rank4/runner.json").read_text())
    assert worker.journal_identity(record)["launch_rank"] == 4
    assert record["compiler_acquisition_complete"] is True
    assert record["warmup"] == record["iterations"] == 0
    assert record["baseline_only"] is False
    assert record["jax_process_index"] == 0
    assert len(record["local_device_slots"]) == 4
    assert set(record["programs"]) == set(worker.PROGRAMS)
    assert case.events == list(worker.PROGRAMS)
    from scripts.greenfield import ws32_rolled_prefill_evidence as evidence

    def validate(value):
        return evidence.validate_local(
            case.root / "rank4", value, repo=ROOT, local_devices=set(range(4))
        )

    assert validate(record) == record["preserved_pair"]
    for change in ("incomplete", "dispatch", "metadata", "phase", "journal_hash"):
        bad = deepcopy(record)
        if change == "incomplete":
            bad["compiler_acquisition_complete"] = False
        elif change == "dispatch":
            bad["model_executable_calls"] = 1
        elif change == "metadata":
            bad["abstract_metadata"]["manifest_sha256"] = "d" * 64
        elif change == "phase":
            bad["acquisition_phases"]["rolled_compile_terminal"]["status"] = "RUNNING"
        else:
            bad["compile_journal_sha256"] = "e" * 64
        with pytest.raises(ValueError):
            validate(bad)
    journal_path = case.root / "rank4/compile_journal.jsonl"
    original = journal_path.read_bytes()
    for change in ("extra_call", "owner", "compiled_memory", "raw", "scope", "time"):
        rows = [json.loads(line) for line in original.splitlines()]
        if change == "extra_call":
            rows.insert(5, dict(rows[4], stage="model_dispatch"))
        elif change == "owner":
            rows[2]["device_memory"][0]["device_id"] = 31
        elif change == "compiled_memory":
            rows[2]["compiled_memory"]["temp_size_in_bytes"] += 1
        elif change == "raw":
            rows[3]["stablehlo_sha256"] = "d" * 64
        elif change == "scope":
            rows[4]["report"]["numerical_execution_authorized"] = True
        else:
            rows[3]["monotonic_seconds"] = -1
        data = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        journal_path.write_bytes(data)
        bad = deepcopy(record)
        bad["compile_journal_sha256"] = sha256(data).hexdigest()
        with pytest.raises(ValueError):
            validate(bad)
    journal_path.write_bytes(original)
    assert validate(record) == record["preserved_pair"]
