"""Compact phase-baseline replay, not full-layer or deployment admission.

The protected collector must still authenticate generations, code/checkpoint
provenance and all eight hosts, and publish/aggregate their actual XPlanes.
"""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_window_evidence as shared
from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS, decode_arrays

PROGRAMS = (*admission.PROGRAMS, *assembly.PROGRAMS)
SAMPLE_GROUPS = (
    ("warmup", phase.WARMUP),
    ("wall", phase.SAMPLES),
    ("traced", phase.TRACED),
)


def traversal_calls(label: str) -> tuple[tuple[str, str], ...]:
    """Reuse the already admitted traversal inventory, including helper calls."""
    return tuple(
        (label + "/" + name.split("/", 1)[1], graph)
        for name, graph in shared.expected_calls(completed_numerical=True)
        if name.startswith("competitive/")
    )


def expected_calls() -> Iterator[tuple[str, str]]:
    yield "wk_decode", "wk_decode"
    yield "wk_promote", "wk_promote"
    for kind, count in SAMPLE_GROUPS:
        for i in range(count):
            yield from traversal_calls(f"{kind}{i}")


def expected_stages() -> list[str]:
    stages = ["identity", "prepare"]
    for name in PROGRAMS:
        stages.extend(
            (
                "lower_compile_started",
                "compiled",
                "raw_written",
                "inspected",
                "compile/" + name,
            )
        )
    stages.append("bind_compiled")

    def append_call(label):
        stages.extend(
            label + "/" + s
            for s in ("memory", "execute", "memory_after", "archive_call")
        )

    append_call("wk_decode")
    append_call("wk_promote")
    stages.extend(
        (
            "wk_boundary",
            "input_specs",
            "phase_original_bind",
            "phase_fixture",
            "phase_verifier",
            "phase_inputs",
            "phase_tiles",
            "phase_inputs_ready",
        )
    )
    for kind, count in SAMPLE_GROUPS:
        if kind == "traced":
            stages.append("phase_trace/start")
        for i in range(count):
            label = f"{kind}{i}"
            for name, graph in traversal_calls(label):
                if graph in ("prefix", "candidate", "control"):
                    stages.extend((name + "_inputs", name + "_ready"))
                append_call(name)
            stages.append(label + "/sample_complete")
        if kind == "traced":
            stages.append("phase_trace/stop")
    stages.extend(
        (
            "phase_baseline/complete",
            "phase_original_complete",
            "phase_trace/finalize",
            "phase_numerical_complete",
        )
    )
    return stages


def validate_samples(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int]
) -> None:
    """Exhaust all287 witnesses, replay budgets and recompute compact timings."""
    if record["call_evidence"] != []:
        raise ValueError("successful phase run retained a partial call")
    index = record["phase_call_index"]
    shared.validate_call_sequence(
        record,
        phase.read_call_witnesses(root / "phase_calls.jsonl.gz", index),
        expected=expected_calls(),
        local_slots=slots,
        names=PROGRAMS,
        budgeter=assembly.memory_budget,
    )
    result = record["phase_baseline"]
    fixed = dict(
        scope=phase.SCOPE,
        order=phase.ORDER,
        warmup=phase.WARMUP,
        iterations=phase.SAMPLES,
        trace_iterations=phase.TRACED,
        case="competitive",
        rows=128,
        initial_offset=2553,
        performance_claim=False,
        complete=True,
    )
    shared.same_json({k: result.get(k) for k in fixed}, fixed, "phase sample scope")
    if set(result) != set(fixed) | {k + "_samples" for k, _ in SAMPLE_GROUPS}:
        raise ValueError("phase sample report inventory differs")
    offset, total_wall = 2, 0.0
    for kind, count in SAMPLE_GROUPS:
        samples = result[kind + "_samples"]
        if type(samples) is not list or len(samples) != count:
            raise ValueError("phase sample count differs")
        for sample in samples:
            calculated = phase.phase_summary(
                index[offset : offset + len(phase.SEQUENCE)]
            )
            wall = sample.get("whole_traversal_seconds_including_checks")
            if (
                type(wall) not in (float, int)
                or not math.isfinite(wall)
                or wall < calculated["all_completed_call_seconds"]
            ):
                raise ValueError("phase whole traversal duration invalid")
            calculated.update(
                whole_traversal_seconds_including_checks=wall,
                excluded_checks_control_seconds=wall
                - calculated["all_completed_call_seconds"],
            )
            shared.same_json(sample, calculated, "phase timing rederivation")
            total_wall += wall
            offset += len(phase.SEQUENCE)
    if total_wall > phase.MAX_SECONDS:
        raise ValueError("phase sampling exceeds bounded wall budget")


def validate_originals(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int]
) -> None:
    """Replay first capture and WK; later exact repeats are worker attestations."""
    original = originals.load_capsule()
    binding = originals.bind_originals(record, slots, original)
    shared.same_json(record["original_binding"], binding, "phase original binding")
    report = record["original_authentication"]
    path = root / "phase_first.npz"
    shared.same_json(
        report,
        dict(
            binding=binding,
            visits={k: sum(n for _, n in SAMPLE_GROUPS) for k in originals.COMPONENTS},
            complete=True,
            first_npz_sha256=sha256(path.read_bytes()).hexdigest(),
        ),
        "phase original authentication",
    )
    if (root / "phase_failure.npz").exists():
        raise ValueError("successful phase run retained failed originals")
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"input__" + n for n in INPUT_FIELDS}
        shared.same_json(
            originals.manifest(decode_arrays(arrays, "input", INPUT_FIELDS)),
            original["inputs"],
            "phase original fixture",
        )
        for device, slot in slots.items():
            for kind, fields in originals.COMPONENTS.items():
                names = {n: f"{kind}_{device}__{n}" for n in fields}
                expected.update(names.values())
                originals.check_observation(
                    original,
                    slot=slot,
                    kind=kind,
                    values={n: arrays[key] for n, key in names.items()},
                )
        if len(arrays.files) != len(expected) or set(arrays.files) != expected:
            raise ValueError("phase first original inventory differs")
    shared.validate_wk(root, record, slots)
    for kind in ("wk_decode", "wk_promote"):
        with np.load(root / f"{kind}.npz", allow_pickle=False) as arrays:
            for device, slot in slots.items():
                originals.check_observation(
                    original, slot=slot, kind=kind, values={"wk": arrays[str(device)]}
                )


def validate_files(root: Path, record: Mapping[str, Any]) -> None:
    """Compose existing graph, journal, memory and first-original replay.

    No fresh trace or all-host claim: the campaign must separately bind those.
    No timed throughput claim: these are partial phase sums, not whole-layer wall.
    """
    fixed = dict(
        protocol=phase.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        performance_claim=False,
        reference_scope=phase.SCOPE,
        independent_full_layer_admission=False,
        current_phase="phase_numerical_complete",
        model_executable_calls=135,
        assembly_executable_calls=150,
        wk_executable_calls=2,
    )
    shared.same_json({k: record.get(k) for k in fixed}, fixed, "phase numerical scope")
    if (
        set(record["programs"]) != set(PROGRAMS)
        or record["cases"] != {}
        or "phase_error" in record
    ):
        raise ValueError("phase program/case/failure inventory differs")
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}

    def inspect(name, p, stable, optimized):
        if name in assembly.PROGRAMS:
            shared.same_json(
                p["admission"],
                assembly.inspect_program(name, stable, optimized, p["compiled_memory"]),
                "phase assembly graph",
            )
        else:
            admission.validate_program_report(
                p["admission"], name, stable, optimized, p["compiled_memory"]
            )

    journal = shared.validate_graph_journal(
        root,
        record,
        protocol_id=phase.PROTOCOL,
        profile=admission.PROFILE,
        names=PROGRAMS,
        journal_type=phase.PhaseJournal,
        inspect=inspect,
    )
    shared.same_json(
        [r["stage"] for r in journal], expected_stages(), "complete phase journal order"
    )
    validate_samples(root, record, slots)
    validate_originals(root, record, slots)
    from scripts.greenfield.collect_ws32_worker_evidence import digest_file

    path = root / "phase.xplane.pb"
    declaration = record["phase_trace_file"]
    source = Path(declaration["source_relative_path"])
    if (
        source.is_absolute()
        or ".." in source.parts
        or source.parts[0] != "phase_trace"
        or not source.name.endswith(".xplane.pb")
    ):
        raise ValueError("phase original trace path differs")
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= phase.MAX_TRACE_BYTES
    ):
        raise ValueError("phase original trace size/type differs")
    shared.same_json(
        declaration,
        dict(
            source_relative_path=str(source),
            bytes=path.stat().st_size,
            sha256=digest_file(path)["sha256"],
        ),
        "phase original trace",
    )


def validate_workers(records, *, pin, pins, ledger, order) -> None:
    shared.validate_workers(
        records, pin=pin, pins=pins, ledger=ledger, order=order, phase_baseline=True
    )


def validate_record(record: Mapping[str, Any], pin: str) -> None:
    from scripts.greenfield.ws32_prefill_layer_campaign import (
        checkpoint_ledger,
        validate_workers as fleet,
    )

    fixed = dict(
        status="SUCCESS",
        kernel=phase.KERNEL,
        protocol=phase.PROTOCOL,
        profile=admission.PROFILE,
        code_hash=pin,
        layer=6,
        latency=None,
        warmup=0,
        iterations=0,
        compile_only=False,
        numerical_execution_authorized=True,
        admission_only=False,
        baseline_only=False,
        diagnostic_only=True,
        performance_claim=False,
        profiler_free_timing=False,
        reference_scope=phase.SCOPE,
        independent_full_layer_admission=False,
        comparison=dict(passed=None, diagnostic_evidence_complete=True),
    )
    shared.same_json({k: record.get(k) for k in fixed}, fixed, "phase aggregate scope")
    pins, ledger = checkpoint_ledger(6)
    fleet(
        record["workers"], pin, layer=6, pins=pins, ledger=ledger, phase_baseline=True
    )
    shared.same_json(record["hlo"], record["workers"][0]["hlo"], "phase aggregate HLO")
    import json

    if (
        record["checksum"]
        != sha256(json.dumps(record["workers"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("phase aggregate worker digest differs")
    if (
        record.get("phase_trace", {}).get("scope")
        != "TRACED_PHASE_ATTRIBUTION_NOT_PROFILER_FREE_THROUGHPUT"
    ):
        raise ValueError("phase aggregate requires collected trace")
    shared.same_json(
        record["phase_wall"], fleet_wall(record["workers"]), "phase fleet timing"
    )


def fleet_wall(records: list[dict]) -> dict:
    """Worst-host sample sums, not a measured independent end-to-end path."""
    fields = (
        "shared_prefix_seconds",
        "wide_suffix_seconds",
        "narrow_suffix_seconds",
        "wide_path_partial_phase_sum_seconds",
        "narrow_path_partial_phase_sum_seconds",
        "comparison_only_assembly_seconds",
        "whole_traversal_seconds_including_checks",
    )
    result = {}
    for field in fields:
        samples = [
            max(r["phase_baseline"]["wall_samples"][i][field] for r in records)
            for i in range(phase.SAMPLES)
        ]
        result[field] = dict(
            samples=samples,
            p50=float(np.percentile(samples, 50)),
            p99=float(np.percentile(samples, 99)),
        )
    return dict(
        scope=phase.SCOPE,
        aggregation="MAX_HOST_PER_UNPROFILED_SAMPLE",
        independent_final_assembly_included=False,
        wall=result,
    )
