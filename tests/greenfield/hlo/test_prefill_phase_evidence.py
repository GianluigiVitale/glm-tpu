"""CPU compact collector replay: actual archived graphs/arrays, fixture timings.

This is not a hardware measurement or the still-unwired acquisition/campaign.
"""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_evidence as evidence
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_window_worker as worker
from tests.greenfield.hlo.test_prefill_window_worker import fake_memory


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """Real CompactPhaseCalls/PhaseJournal -> JSON/gzip/NPZ -> actual consumer.

    Nine original DB594 graph pairs and first/WK outputs remain unchanged.
    Executables and counters are CPU fixtures; no original model is rerun.
    """
    root = tmp_path_factory.mktemp("phase-collector")
    capsule = originals.load_capsule()
    source = Path("/home/gianl/glm-run") / capsule["source_tag"] / "fleet/rank0"
    if not source.exists():
        pytest.skip("generation-bound DB594 local archive not present")
    old = json.loads((source / "runner.json").read_text())
    record = {
        k: deepcopy(old[k])
        for k in (
            "code_hash",
            "launch_rank",
            "jax_process_index",
            "local_device_slots",
            "mesh_sha256",
            "physical_device_ids",
            "programs",
            "wk_originals",
            "wk_boundary_sha256",
        )
    }
    record.update(
        protocol=phase.PROTOCOL,
        profile=old["profile"],
        compile_only=False,
        performance_claim=False,
        reference_scope=phase.SCOPE,
        independent_full_layer_admission=False,
        cases={},
        model_executable_calls=135,
        assembly_executable_calls=150,
        wk_executable_calls=2,
    )
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    record["original_binding"] = originals.bind_originals(record, slots, capsule)
    for name in evidence.PROGRAMS:
        for ext in ("stablehlo.mlir", "optimized_hlo.txt"):
            os.link(source / f"{name}.{ext}", root / f"{name}.{ext}")
    for name in ("wk_decode", "wk_promote", "wk_boundary"):
        os.link(source / f"{name}.npz", root / f"{name}.npz")
    os.link(source / "competitive.npz", root / "phase_first.npz")
    record["original_authentication"] = dict(
        binding=record["original_binding"],
        complete=True,
        visits={k: 15 for k in originals.COMPONENTS},
        first_npz_sha256=sha256((root / "phase_first.npz").read_bytes()).hexdigest(),
    )
    journal = phase.PhaseJournal(
        root / "compile_journal.jsonl",
        {
            k: record[k]
            for k in ("protocol", "profile", "compile_only", "code_hash", "launch_rank")
        },
    )
    # Copy the real compile evidence into the distinct phase journal. Model and
    # host-location checks execute below; no synthetic HLO admission reports.
    old_journal = [
        json.loads(s)
        for s in (source / "compile_journal.jsonl").read_text().splitlines()
    ]
    for entry in old_journal[1:]:
        if entry["stage"] == "bind_compiled":
            break
        journal._graph = entry["graph"]
        fields = {
            k: v
            for k, v in entry.items()
            if k
            not in (
                "schema_version",
                "artifact_kind",
                "status",
                "performance_claim",
                "numerical_claim",
                "monotonic_seconds",
                "graph",
                "stage",
            )
        }
        journal._write(entry["stage"], **fields)
    calls = phase.CompactPhaseCalls(
        root=root,
        record=record,
        journal=journal,
        consensus=lambda ok: ok,
        local_slots=slots,
        budgeter=assembly.memory_budget,
    )

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**record["programs"][self.name]["compiled_memory"])

        def __call__(self, *args):
            return np.zeros(1)  # no numerical claim from this fixture

    calls.programs = {n: Program(n) for n in evidence.PROGRAMS}
    census = fake_memory()
    for row, device in zip(census["devices"], slots, strict=True):
        row.update(device_id=device, process_index=record["jax_process_index"])
    post = [
        dict(
            device_id=r["device_id"],
            process_index=r["process_index"],
            platform="tpu",
            **r["memory_stats"],
        )
        for r in census["devices"]
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            worker, "capture_resident_buffers", lambda *a, **k: deepcopy(census)
        )
        patch.setattr(
            worker, "capture_identified_device_memory", lambda *a: deepcopy(post)
        )
        calls.phase("bind_compiled", lambda: None)
        for name in ("wk_decode", "wk_promote"):
            calls.call(name, name, (), preserve=lambda r: None)
        for name in (
            "wk_boundary",
            "input_specs",
            "phase_original_bind",
            "phase_fixture",
            "phase_verifier",
            "phase_inputs",
            "phase_tiles",
            "phase_inputs_ready",
        ):
            calls.phase(name, lambda: None)

        def traversal(calls, *, case, **kwargs):
            # Real budget/stream/phase writer, fixture model dispatches. The
            # separate sampler tests execute the actual device-assembly traversal.
            for label, name in evidence.traversal_calls(case):
                if name in ("prefix", "candidate", "control"):
                    calls.phase(label + "_inputs", lambda: None)
                    calls.phase(label + "_ready", lambda: None)
                calls.call(label, name, (), preserve=lambda r: None)

        patch.setattr(phase, "execute_window", traversal)
        phase.run_samples(
            calls,
            values=(),
            tiles=(),
            verify_component=lambda *a: None,
            verify_assembly=lambda *a: None,
            trace_start=lambda p: None,
            trace_stop=lambda: None,
        )
        calls.phase("phase_original_complete", lambda: None)
        calls.phase("phase_numerical_complete", lambda: None)
    journal.close()
    record["compile_journal_sha256"] = sha256(
        (root / "compile_journal.jsonl").read_bytes()
    ).hexdigest()
    return root, json.loads(json.dumps(record)), slots


def test_collector_replays_original_graphs_arrays_and_real_compact_stream(collected):
    root, record, _ = collected
    assert len(record["phase_call_index"]) == 287
    evidence.validate_files(root, record)


@pytest.mark.parametrize(
    "change",
    [
        "missing_final",
        "partial",
        "sum",
        "bool_wall",
        "traced_as_wall",
        "visits",
        "owner",
        "scope",
        "counts",
    ],
)
def test_phase_collector_refuses_bad_records(collected, change):
    root, original, slots = collected
    record = deepcopy(original)
    if change == "missing_final":
        record["phase_call_index"].pop()
    elif change == "partial":
        record["call_evidence"].append({})
    elif change == "sum":
        record["phase_baseline"]["wall_samples"][0]["wide_suffix_seconds"] += 0.01
    elif change == "bool_wall":
        record["phase_baseline"]["wall_samples"][0][
            "whole_traversal_seconds_including_checks"
        ] = True
    elif change == "traced_as_wall":
        record["phase_baseline"]["wall_samples"][-2:] = record["phase_baseline"][
            "traced_samples"
        ]
    elif change == "visits":
        record["original_authentication"]["visits"]["wide"] = True
    elif change == "owner":
        record["local_device_slots"][0]["device_slot"] = 31
    elif change == "scope":
        record["independent_full_layer_admission"] = True
    else:
        record["model_executable_calls"] = 27
    with pytest.raises(ValueError):
        if change in ("visits", "owner"):
            evidence.validate_originals(root, record, slots)
        elif change in ("scope", "counts"):
            evidence.validate_files(root, record)
        else:
            evidence.validate_samples(root, record, slots)


def test_stream_consumer_refuses_cross_sample_peak_regression(collected):
    root, record, slots = collected
    from scripts.greenfield.prefill_window_evidence import validate_call_sequence

    def entries():
        for i, entry in enumerate(
            phase.read_call_witnesses(
                root / "phase_calls.jsonl.gz", record["phase_call_index"]
            )
        ):
            if i == 20:  # final assembly of warmup0; warmup1 must not reset peak
                entry["post_memory"][0]["peak_bytes_in_use"] += 1
            yield entry

    with pytest.raises(ValueError, match="between-call"):
        validate_call_sequence(
            record,
            entries(),
            expected=evidence.expected_calls(),
            local_slots=slots,
            names=evidence.PROGRAMS,
            budgeter=assembly.memory_budget,
        )


@pytest.mark.parametrize("change", ["failed", "trace_order", "extra"])
def test_phase_journal_mutations_refuse(collected, tmp_path, change):
    root, record, _ = collected
    record = deepcopy(record)
    for item in root.iterdir():
        if item.name != "compile_journal.jsonl":
            os.link(item, tmp_path / item.name)
    path = root / "compile_journal.jsonl"
    entries = [json.loads(s) for s in path.read_text().splitlines()]
    entry = next(r for r in entries if r["stage"] == "phase_trace/start")
    if change == "failed":
        entry["passed"] = False
    elif change == "trace_order":
        entry["stage"] = "phase_trace/stop"
    else:
        entries.append(entries[-1])
    raw = ("\n".join(json.dumps(r) for r in entries) + "\n").encode()
    (tmp_path / path.name).write_bytes(raw)
    record["compile_journal_sha256"] = sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        evidence.validate_files(tmp_path, record)
