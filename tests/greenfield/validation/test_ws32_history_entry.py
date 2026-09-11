"""Protected-entry sequencing only: fixture runtime/math, actual local call writer."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import run_short_decoder_ws32 as runner
from scripts.greenfield import ws32_history_entry as entry
from scripts.greenfield import ws32_history_call_evidence as evidence
from tests.greenfield.validation.test_ws32_history_execution import staged

TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T200000000000000Z"
PIN = "a" * 40
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / TAG / "rank0"
    root.mkdir(parents=True)
    monkeypatch.setattr(entry.preflight_module, "RUN_ROOT", tmp_path)
    pins = {branch: {form: dict(name=f"fixture/{branch}.{form}", generation="123",
                               size=10, crc32c="fixture", sha256="b" * 64)
                     for form in entry.protocol.ORIGINAL_FORMS}
            for branch in entry.protocol.BRANCHES}
    monkeypatch.setattr(entry.protocol, "original_pins", lambda repo, rank: deepcopy(pins))
    retained = dict(
        tag=TAG, code_hash=PIN, launch_rank=0, hostname=entry.socket.gethostname(),
        protocol=entry.protocol.PROTOCOL, selected_layer_ids=list(entry.protocol.LAYERS),
        include_embedding=True, context_capacity=entry.protocol.CAPACITY,
        host_main_rope_table=True, selected_leaf_count=entry.protocol.SELECTED_LEAVES,
        payload_bytes_per_chip=entry.protocol.PAYLOAD_BYTES,
        overlay_tensor_count=entry.protocol.OVERLAY_TENSORS,
        overlay_bytes_per_chip=entry.protocol.OVERLAY_BYTES,
        prompt_ids_sha256=entry.protocol.PROMPT_SHA,
        prompt_hash_source="AUTHENTICATED_ORIGINAL_TOKEN_ORACLE",
        original_tags=entry.protocol.ORIGINAL_TAGS, original_pins=deepcopy(pins),
        receipt_sha256={b: entry.protocol.RECEIPTS[b][1] for b in entry.protocol.BRANCHES},
        bytes=40, scope="HEADERS_ORIGINALS_AND_HOST_INPUTS_ONLY_NOT_SELECTED_PAYLOAD_OR_LIVE_TOPOLOGY",
        numerical_execution_available=False, numerical_admission=False,
        numerical_promotion=False, performance_claim=False,
    )

    def save():
        (root / "retained_preflight.json").write_text(json.dumps(retained))
    save()
    events, votes = [], []
    runtime = (object(), object(), object(), object(), object())
    bound = object()

    def registration(repo):
        assert repo == REPO
        events.append("source_and_db611")
    monkeypatch.setattr(entry.admission, "registration", registration)

    def initialize(args):
        events.append("initialize")
        return runtime
    monkeypatch.setattr(runner, "_initialize_runtime", initialize)

    def prepare_bound(**kwargs):
        assert kwargs["runtime"] is runtime and kwargs["root"] == root and kwargs["repo"] == REPO
        assert kwargs["record"]["profile"] == entry.admission.PROFILE
        assert kwargs["record"]["protocol"] == entry.protocol.PROTOCOL
        events.append("prepare_bound")
        return bound
    monkeypatch.setattr(entry.runtime_module, "prepare_bound", prepare_bound)
    from jax.experimental import multihost_utils

    def vote(value):
        assert value.shape == () and value.dtype == np.int32
        votes.append(bool(value))
        return np.full(8, value, np.int32)
    monkeypatch.setattr(multihost_utils, "process_allgather", vote)

    def execute(**kwargs):
        assert kwargs["bound"] is bound and kwargs["mesh"] is runtime[1]
        assert kwargs["root"] == root and kwargs["repo"] == REPO
        events.append("execute")
        complete_claims(kwargs["record"])
    # A distinct adapter object avoids patching the reused execution fixture's
    # module; it also makes explicit that these terminal mutations use no math.
    monkeypatch.setattr(entry, "execution", NS(execute=execute, call_schedule=entry.execution.call_schedule))
    return NS(root=root, args=NS(expected_code_hash=PIN, process_id=0, output_dir=root),
              retained=retained, save=save, events=events, votes=votes,
              runtime=runtime, bound=bound)


def history_claims():
    return dict(protocol=entry.protocol.PROTOCOL, complete=True, attribution_eligible=True,
                prompt_length=entry.protocol.PROMPT_LENGTH, steps=len(entry.protocol.plan()),
                reproduction={branch: dict(reproduced=True) for branch in entry.protocol.BRANCHES},
                numerical_promotion=False, performance_claim=False, cause_claim=False)


def complete_claims(record):
    """No original validation is asserted: only the worker's completion envelope."""
    entries = [dict(phase=phase, graph=graph, completed=True,
                    original=dict(schema=evidence.SCHEMA, path=f"call_records/call{i:03d}.json",
                                  bytes=10, sha256="c" * 64, index=i))
               for i, (phase, graph) in enumerate(entry.execution.call_schedule())]
    record.update(history=history_claims(), history_execution_complete=True,
                  current_phase="history/execution_complete", planned_call_count=331,
                  programs={name: dict(admission=dict(passed=True)) for name in entry.protocol.PROGRAMS},
                  compile_journal_sha256="d" * 64, call_evidence_layout=evidence.SCHEMA,
                  call_evidence=entries, call_original_bytes=3310,
                  acquisition_phases={"history/finalize": dict(status="COMPLETE", error=None)})


def saved(fixture):
    return json.loads((fixture.root / "runner.json").read_bytes())


def test_entry_orders_guards_bind_execution_and_terminal_vote(fixture):
    assert entry.execute(fixture.args, tag=TAG, repo=REPO) == 0
    assert fixture.events == ["source_and_db611", "initialize", "prepare_bound", "execute"]
    assert fixture.votes == [True]
    record = saved(fixture)
    assert record["status"] == entry.STATUS
    assert record["acquisition_phases"]["history/terminal"]["status"] == "COMPLETE"
    assert record["kernel"] == entry.protocol.KERNEL
    assert record["diagnostic_only"] is True
    assert record["numerical_promotion"] is record["performance_claim"] is False
    assert record["iterations"] == 0 and record["latency"] is None
    assert record["pid"] > 0 and record["start_ticks"] > 0 and record["boot_id"]
    assert fixture.retained["numerical_execution_available"] is False
    assert "profile" not in fixture.retained  # Metadata preflight is not numerical admission.
    assert not (fixture.root / "call_records").exists()  # Collector is deliberately separate.


def test_entry_actual_nine_graphs_and_331_append_once_call_originals(fixture, monkeypatch):
    run, actual, events = staged(fixture.root, monkeypatch)

    def execute(**kwargs):
        assert kwargs["mesh"] is fixture.runtime[1] and kwargs["bound"] is fixture.bound
        actual.update(kwargs["record"])
        try:
            run()
            # staged's worker arithmetic is a fixture, not reproduction evidence.
            actual["history"] = history_claims()
        finally:
            kwargs["record"].update(actual)
    monkeypatch.setattr(entry.execution, "execute", execute)
    assert entry.execute(fixture.args, tag=TAG, repo=REPO) == 0
    record = saved(fixture)
    originals = evidence.load_calls(fixture.root, record)
    assert len(originals) == 331
    assert tuple((v["phase"], v["graph"]) for v in originals) == entry.execution.call_schedule()
    assert len([v for v in events if v[0] == "dispatch"]) == 331
    assert len([v for v in events if v[0] == "compile"]) == 9
    assert sha256((fixture.root / "compile_journal.jsonl").read_bytes()).hexdigest() == record["compile_journal_sha256"]
    assert fixture.votes == [True]
    assert record["status"] == entry.STATUS


@pytest.mark.parametrize("key,value", [
    ("tag", "old"), ("code_hash", "b" * 40), ("launch_rank", True), ("hostname", "other"),
    ("protocol", "compile-only"), ("selected_layer_ids", [0, 1]), ("selected_leaf_count", 200),
    ("include_embedding", False), ("context_capacity", 2048), ("host_main_rope_table", False),
    ("payload_bytes_per_chip", 1), ("overlay_tensor_count", 11), ("overlay_bytes_per_chip", 1),
    ("prompt_ids_sha256", None), ("prompt_hash_source", "ASSUMED"), ("original_tags", {}),
    ("original_pins", {}), ("receipt_sha256", {}), ("bytes", 0), ("scope", "NUMERICAL"),
    ("numerical_execution_available", True), ("numerical_admission", True),
    ("numerical_promotion", True), ("performance_claim", True),
])
def test_retained_mismatch_refuses_before_runtime(fixture, key, value):
    fixture.retained[key] = value
    fixture.save()
    with pytest.raises(ValueError, match="retained identity"):
        entry.execute(fixture.args, tag=TAG, repo=REPO)
    assert fixture.events == ["source_and_db611"] and not fixture.votes
    assert saved(fixture)["status"] == "DIAGNOSTIC_FAILED"
    assert saved(fixture)["programs"] == {}


@pytest.mark.parametrize("failure", ["source", "missing_preflight", "linked_preflight", "json", "initialize", "bind", "execute"])
def test_setup_failures_never_reach_successor(fixture, monkeypatch, failure):
    def refused(*args, **kwargs):
        raise ValueError("fixture refusal")
    if failure == "source":
        monkeypatch.setattr(entry.admission, "registration", refused)
    elif failure in ("missing_preflight", "linked_preflight"):
        path = fixture.root / "retained_preflight.json"
        original = fixture.root / "original_preflight.json"
        path.rename(original)
        if failure == "linked_preflight":
            path.symlink_to(original)
    elif failure == "json":
        (fixture.root / "retained_preflight.json").write_text("invalid json")
    elif failure == "initialize":
        monkeypatch.setattr(runner, "_initialize_runtime", refused)
    elif failure == "bind":
        monkeypatch.setattr(entry.runtime_module, "prepare_bound", refused)
    else:
        monkeypatch.setattr(entry.execution, "execute", refused)
    with pytest.raises(ValueError):
        entry.execute(fixture.args, tag=TAG, repo=REPO)
    assert saved(fixture)["status"] == "DIAGNOSTIC_FAILED"
    assert "execute" not in fixture.events and not fixture.votes
    if failure not in ("bind", "execute"):
        assert "initialize" not in fixture.events


@pytest.mark.parametrize("failure", ["tag", "compile_tag", "rank", "bool_rank", "pin", "path", "relative", "symlink", "prior_runner"])
def test_default_off_identity_and_no_overwrite(fixture, monkeypatch, failure):
    tag = TAG
    if failure == "tag":
        tag = ""
    elif failure == "compile_tag":
        tag = TAG.replace("history_frontier_l06", "history_frontier_compile")
    elif failure == "rank":
        fixture.args.process_id = 8
    elif failure == "bool_rank":
        fixture.args.process_id = False
    elif failure == "pin":
        fixture.args.expected_code_hash = "a" * 39
    elif failure == "path":
        fixture.args.output_dir = fixture.root.parent
    elif failure == "relative":
        fixture.args.output_dir = Path("rank0")
    elif failure == "symlink":
        link = fixture.root.parent / "linked"
        link.symlink_to(fixture.root, target_is_directory=True)
        fixture.args.output_dir = link
    else:
        (fixture.root / "runner.json").write_text("preserved original")
    with pytest.raises((ValueError, FileExistsError)):
        entry.execute(fixture.args, tag=tag, repo=REPO)
    assert fixture.events == [] and fixture.votes == []
    if failure == "prior_runner":
        assert (fixture.root / "runner.json").read_text() == "preserved original"
    else:
        assert not (fixture.root / "runner.json").exists()


@pytest.mark.parametrize("mutation", [
    "profile", "protocol", "kernel", "promotion", "performance", "latency", "iterations", "finalization",
    "missing_branch", "reproduction", "cause", "incomplete", "missing_call", "schedule",
    "unsealed", "not_completed", "reference_path", "reference_hash", "reference_index",
    "reference_size", "numeric_hash", "numeric_journal_hash", "total", "program", "admission", "phase",
])
def test_terminal_mutations_vote_false_and_preserve_failure(fixture, monkeypatch, mutation):
    execute = entry.execution.execute

    def changed(**kwargs):
        execute(**kwargs)
        record = kwargs["record"]
        if mutation in ("profile", "protocol", "kernel"):
            record[mutation] = "old"
        elif mutation in ("promotion", "performance"):
            record[{"promotion": "numerical_promotion", "performance": "performance_claim"}[mutation]] = True
        elif mutation == "latency":
            record["latency"] = 1
        elif mutation == "iterations":
            record["iterations"] = False
        elif mutation == "finalization":
            record["acquisition_phases"]["history/finalize"]["status"] = "FAILED"
        elif mutation == "missing_branch":
            del record["history"]["reproduction"]["control"]
        elif mutation == "reproduction":
            record["history"]["reproduction"]["candidate"]["reproduced"] = False
        elif mutation == "cause":
            record["history"]["cause_claim"] = True
        elif mutation == "incomplete":
            record["history_execution_complete"] = False
        elif mutation == "missing_call":
            record["call_evidence"].pop()
        elif mutation == "schedule":
            record["call_evidence"][10]["phase"] = "history/wrong"
        elif mutation == "unsealed":
            del record["call_evidence"][0]["original"]
        elif mutation == "not_completed":
            record["call_evidence"][0]["completed"] = False
        elif mutation == "numeric_hash":
            record["call_evidence"][0]["original"]["sha256"] = int("1" * 64)
        elif mutation == "numeric_journal_hash":
            record["compile_journal_sha256"] = int("1" * 64)
        elif mutation.startswith("reference_"):
            key = mutation.removeprefix("reference_")
            key = {"hash": "sha256", "size": "bytes"}.get(key, key)
            record["call_evidence"][0]["original"][key] = {
                "path": "../original.json", "sha256": "z" * 64, "index": False,
                "bytes": evidence.MAX_CALL_BYTES + 1}[key]
        elif mutation == "total":
            record["call_original_bytes"] += 1
        elif mutation == "program":
            del record["programs"]["observer"]
        elif mutation == "admission":
            record["programs"]["observer"]["admission"]["passed"] = False
        else:
            record["current_phase"] = "history/conclude"
    monkeypatch.setattr(entry.execution, "execute", changed)
    with pytest.raises(ValueError):
        entry.execute(fixture.args, tag=TAG, repo=REPO)
    assert fixture.votes == [False]
    assert saved(fixture)["status"] == "DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("failure", ["peer", "write", "write_and_failure_record"])
def test_terminal_peer_and_publication_failure_never_return_success(fixture, monkeypatch, failure, capsys):
    if failure == "peer":
        from jax.experimental import multihost_utils
        def vote(value):
            fixture.votes.append(bool(value))
            return np.asarray([1] * 7 + [0], np.int32)
        monkeypatch.setattr(multihost_utils, "process_allgather", vote)
    else:
        atomic = acquisition._atomic_json
        def write(path, record):
            if "history/terminal" in record.get("acquisition_phases", {}):
                raise OSError("fixture terminal publication")
            atomic(path, record)
        monkeypatch.setattr(acquisition, "_atomic_json", write)
        if failure == "write_and_failure_record":
            monkeypatch.setattr(entry, "_atomic_json", write)
    with pytest.raises((RuntimeError, OSError)):
        entry.execute(fixture.args, tag=TAG, repo=REPO)
    assert fixture.votes == [failure == "peer"]
    if failure == "write_and_failure_record":
        assert "history final record publication failed" in capsys.readouterr().out
        assert saved(fixture)["status"] == "RUNNING"
    else:
        assert saved(fixture)["status"] == "DIAGNOSTIC_FAILED"
