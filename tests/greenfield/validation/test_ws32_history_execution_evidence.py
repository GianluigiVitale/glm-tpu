"""Original-file/journal/call reader with synthetic compute and memory corpus."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from scripts.greenfield import ws32_history_execution_evidence as reader
from scripts.greenfield.ws32_history_call_evidence import preserve_call
from glm_tpu.greenfield.validation.ws32_prefill_memory import MEMORY_FIELDS

SLOTS = dict(zip((12, 14, 13, 15), (9, 13, 25, 29)))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("history_execution_corpus")
    identity = dict(protocol=reader.protocol.PROTOCOL, profile=reader.admission.PROFILE,
                    compile_only=False, diagnostic_only=True, code_hash="a" * 40, launch_rank=0)
    record = dict(identity, jax_process_index=3, history_execution_complete=True,
                  planned_call_count=331, programs={}, call_evidence=[], call_original_bytes=0,
                  call_evidence_layout="ws32_history_call_original_v1")
    memory = {key: 0 for key in MEMORY_FIELDS}
    for name in reader.protocol.PROGRAMS:
        stable, optimized = f"{name} stable fixture", f"{name} optimized fixture"
        (root / f"{name}.stablehlo.mlir").write_text(stable)
        (root / f"{name}.optimized_hlo.txt").write_text(optimized)
        record["programs"][name] = dict(stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
            compiled_memory=memory, compile_seconds=1., admission=dict(passed=True, fixture=True))
    stages = reader.expected_stages()
    journal = []
    current = None
    for i, stage in enumerate(stages):
        row = dict(schema_version=1, artifact_kind=reader.HistoryJournal.artifact_kind,
            status=reader.HistoryJournal.status, numerical_claim=False, performance_claim=False,
            stage=stage, graph=current, monotonic_seconds=float(i))
        if i == 0:
            row["identity"] = identity
        elif i <= 36:
            current = reader.protocol.PROGRAMS[(i - 1) // 4]
            row["graph"] = current
            p = record["programs"][current]
            if stage == "compiled":
                row.update(compiled_memory=memory, seconds=1., device_memory=[])
            elif stage == "raw_written":
                row.update({key: p[key] for key in ("stablehlo_sha256", "optimized_hlo_sha256")})
            elif stage == "inspected":
                row["report"] = dict(scope="ORIGINAL_CAPTURE_ONLY_NOT_ADMISSION")
        else:
            row["passed"] = True
        journal.append(row)
    raw = ("\n".join(json.dumps(row, sort_keys=True) for row in journal) + "\n").encode()
    (root / "compile_journal.jsonl").write_bytes(raw)
    record["compile_journal_sha256"] = sha256(raw).hexdigest()
    stats = dict(bytes_limit=32 << 30, bytes_in_use=1 << 20, peak_bytes_in_use=2 << 20)
    census = dict(devices=[dict(device_id=d, platform="tpu", process_index=3,
                               memory_stats=stats) for d in SLOTS])
    for index, (phase, name) in enumerate(reader.call_schedule()):
        entry = dict(phase=phase, graph=name, completed=True, completed_call_seconds=.001,
            census=census, compiled_memory={n: memory for n in reader.protocol.PROGRAMS},
            budget=dict(estimate_fits=True, fixture=True),
            post_memory=[dict(device_id=d, platform="tpu", process_index=3, **stats) for d in SLOTS])
        ref = preserve_call(root, entry, index=index, used_bytes=record["call_original_bytes"])
        record["call_evidence"].append(ref)
        record["call_original_bytes"] += ref["original"]["bytes"]
    return root, record


@pytest.fixture
def bundle(corpus, tmp_path, monkeypatch):
    original, record = corpus
    root = tmp_path / "rank0"
    shutil.copytree(original, root)
    inspections = []
    def inspect(name, stable, optimized, memory, *, repo):
        assert stable == f"{name} stable fixture" and optimized == f"{name} optimized fixture"
        inspections.append(name)
        return dict(passed=True, fixture=True)
    monkeypatch.setattr(reader.admission, "inspect_program", inspect)
    monkeypatch.setattr(reader, "memory_budget", lambda *a, **k: dict(estimate_fits=True, fixture=True))
    return root, deepcopy(record), inspections


def test_replay_all_originals_with_compact_records_and_graph_cache(bundle):
    root, record, inspections = bundle
    before = deepcopy(record)
    cache = {}
    result = reader.replay(root, record, SLOTS, repo=root, graph_cache=cache)
    assert result["compiler_programs"] == 9 and result["completed_calls"] == 331
    assert result["journal_stages"] == 2087
    assert result["materializer_or_boundary_replay"] is False
    assert record == before and all("census" not in ref for ref in record["call_evidence"])
    reader.replay(root, record, SLOTS, repo=root, graph_cache=cache)
    assert inspections == list(reader.protocol.PROGRAMS)


@pytest.mark.parametrize("defect", ["profile", "count", "completed", "programs", "graph_bytes",
    "journal_order", "journal_identity", "journal_time", "journal_phase_failure",
    "call_sequence", "call_budget", "call_owner", "call_peak", "between_call_peak"])
def test_producer_rebound_claims_do_not_replace_independent_replay(bundle, defect):
    root, record, _ = bundle
    if defect == "profile":
        record["profile"] = "another"
    elif defect == "count":
        record["planned_call_count"] = 330
    elif defect == "completed":
        record["history_execution_complete"] = False
    elif defect == "programs":
        record["programs"]["unexpected"] = {}
    elif defect == "graph_bytes":
        (root / "observer.optimized_hlo.txt").write_text("changed")
    elif defect.startswith("journal_"):
        path = root / "compile_journal.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if defect == "journal_order":
            rows[50], rows[51] = rows[51], rows[50]
        elif defect == "journal_identity":
            rows[0]["identity"]["profile"] = "other"
        elif defect == "journal_time":
            rows[50]["monotonic_seconds"] = -1.
        else:
            rows[50]["passed"] = False
        raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        path.write_bytes(raw)
        record["compile_journal_sha256"] = sha256(raw).hexdigest()
    else:
        ref = record["call_evidence"][1]
        path = root / ref["original"]["path"]
        row = json.loads(path.read_bytes())
        if defect == "call_sequence":
            row["phase"] = ref["phase"] = "wrong-phase"
        elif defect == "call_budget":
            row["budget"]["estimate_fits"] = False
        elif defect == "call_owner":
            row["census"]["devices"][0]["device_id"] = 0
        elif defect == "call_peak":
            row["post_memory"][0]["peak_bytes_in_use"] = 32 << 30
        else:
            row["census"]["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 1 << 20
        raw = (json.dumps(row, sort_keys=True) + "\n").encode()
        record["call_original_bytes"] += len(raw) - ref["original"]["bytes"]
        ref["original"].update(bytes=len(raw), sha256=sha256(raw).hexdigest())
        path.write_bytes(raw)
    with pytest.raises((ValueError, AssertionError)):
        reader.replay(root, record, SLOTS, repo=root)
