"""Independent original/graph/journal/call replay for the three-call trial."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_rolled_admission as admission
from scripts.greenfield import prefill_rolled_window as protocol
from scripts.greenfield.prefill_rolled_worker import RolledJournal, compact_comparison
from scripts.greenfield import prefill_window_evidence as shared

CALLS = tuple((name, name) for name in protocol.PROGRAMS)


def validate_record(record: Mapping[str, Any], pin: str) -> None:
    from scripts.greenfield.ws32_prefill_layer_campaign import (
        checkpoint_ledger,
        validate_workers,
    )

    fixed = dict(
        status="SUCCESS",
        kernel=protocol.KERNEL,
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        code_hash=pin,
        layer=6,
        latency=None,
        warmup=0,
        iterations=0,
        compile_only=False,
        numerical_execution_authorized=True,
        admission_only=True,
        baseline_only=False,
        diagnostic_only=False,
        performance_claim=False,
        profiler_free_timing=False,
        reference_scope=protocol.REFERENCE_SCOPE,
        independent_canonical_dsa_claim=False,
        comparison=dict(passed=True, diagnostic_evidence_complete=False),
    )
    shared.same_json({k: record.get(k) for k in fixed}, fixed, "rolled aggregate scope")
    pins, ledger = checkpoint_ledger(6)
    validate_workers(
        record["workers"], pin, layer=6, pins=pins, ledger=ledger, rolled_window=True
    )
    shared.same_json(record["hlo"], record["workers"][0]["hlo"], "rolled aggregate HLO")
    if (
        record["checksum"]
        != sha256(json.dumps(record["workers"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("rolled aggregate worker digest differs")


def validate_files(
    root: Path, record: Mapping[str, Any], reference: protocol.RetainedReference
) -> None:
    fixed = dict(
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        performance_claim=False,
        independent_canonical_dsa_claim=False,
        integration_complete=True,
        model_executable_calls=1,
        wk_executable_calls=2,
        current_phase="rolled_complete",
    )
    shared.same_json(
        {k: record.get(k) for k in fixed}, fixed, "rolled scope/completion"
    )
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    if slots != dict(reference.slots) or set(record["programs"]) != set(
        protocol.PROGRAMS
    ):
        raise ValueError("rolled current owner/program inventory differs")
    capsule = protocol.originals.load_capsule()
    shared.same_json(
        record["original_binding"],
        protocol.originals.bind_originals(record, slots, capsule),
        "rolled current original binding",
    )
    shared.same_json(
        record["retained_sources"],
        list(reference.sources),
        "rolled retained source generations",
    )
    shared.same_json(
        record["retained_source_seal_sha256"], protocol.SEAL_SHA, "rolled source seal"
    )
    shared.same_json(
        record["retained_original_comparison"],
        dict(reference.bounded_receipt),
        "rolled original receipt",
    )
    shared.validate_call_sequence(
        record,
        record["call_evidence"],
        expected=CALLS,
        local_slots=slots,
        names=protocol.PROGRAMS,
        budgeter=admission.memory_budget,
    )

    def inspect(name, program, stable, optimized):
        admission.validate_program_report(
            program["admission"], name, stable, optimized, program["compiled_memory"]
        )

    journal = shared.validate_graph_journal(
        root,
        record,
        protocol_id=protocol.PROTOCOL,
        profile=admission.PROFILE,
        names=protocol.PROGRAMS,
        journal_type=RolledJournal,
        inspect=inspect,
    )
    stages = ["identity"]
    for _ in protocol.PROGRAMS:
        stages.extend(("lower_compile_started", "compiled", "raw_written", "inspected"))
    for name in protocol.PROGRAMS:
        if name == "candidate":
            stages.extend(
                (
                    "candidate_inputs",
                    "candidate_inputs_ready",
                    "candidate_input_capture",
                )
            )
        stages.extend(
            name + suffix for suffix in ("/memory", "/execute", "/memory_after")
        )
    stages.append("rolled_complete")
    shared.same_json(
        [r["stage"] for r in journal], stages, "rolled complete phase order"
    )
    if set(record["wk_sha256"]) != set(protocol.PROGRAMS[:2]):
        raise ValueError("rolled WK capture inventory differs")
    captures = {}
    for name in protocol.PROGRAMS[:2]:
        path = root / f"{name}.npz"
        if sha256(path.read_bytes()).hexdigest() != record["wk_sha256"][name]:
            raise ValueError("rolled original WK bytes differ")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {str(d) for d in slots}:
                raise ValueError("rolled original WK owner inventory differs")
            captures[name] = dict(saved)
        for device, slot in slots.items():
            value = captures[name][str(device)]
            protocol.originals.check_observation(
                capsule, slot=slot, kind=name, values={"wk": value}
            )
    for device in slots:
        raw = captures["wk_decode"][str(device)]
        value = captures["wk_promote"][str(device)]
        if not np.array_equal(raw.view(protocol.window.BF16).astype(np.float32), value):
            raise ValueError("rolled completed WK conversion differs")
    path = root / "candidate.npz"
    if sha256(path.read_bytes()).hexdigest() != record["candidate_sha256"]:
        raise ValueError("rolled candidate original bytes differ")
    comparison = protocol.replay_candidate(path, reference)
    if not comparison["passed"]:
        raise ValueError("rolled candidate original comparison fails")
    shared.same_json(
        record["candidate_comparison"],
        compact_comparison(comparison),
        "rolled candidate independent comparison",
    )
    shared.same_json(
        record["hlo"],
        dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(passed=True, profile=admission.PROFILE),
        ),
        "rolled HLO binding",
    )
