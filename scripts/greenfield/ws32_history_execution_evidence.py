"""Independent compiler/journal/per-call replay for the fixed history diagnostic.

Reuses existing graph and actual-memory validators. Compact runner references
are expanded only transiently here, never copied into aggregates or SQLite.
This does not validate materializers, boundary arrays, fleet identity or cleanup.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.greenfield import ws32_history_admission as admission
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.ws32_history_execution import HistoryJournal, call_schedule
from scripts.greenfield.ws32_history_call_evidence import load_calls
from scripts.greenfield.ws32_history_runtime import memory_budget
from scripts.greenfield.prefill_window_evidence import (
    same_json, validate_call_sequence, validate_graph_journal,
)


def expected_stages() -> tuple[str, ...]:
    """Exact original journal stages, including all append-once evidence votes."""
    stages = ["identity", *(stage for _ in protocol.PROGRAMS for stage in
               ("lower_compile_started", "compiled", "raw_written", "inspected")),
              "history/admission"]

    def call(phase: str) -> None:
        stages.extend(phase + "/" + suffix for suffix in
                      ("original_preflight", "memory", "execute", "memory_after", "original_retained"))

    for phase, _ in call_schedule()[:10]:
        call(phase)
    stages.extend(("history/preflight", "history/table", "history/candidate/caches",
                   "history/control/caches", "history/independent", "history/initial_health"))
    plan = protocol.plan()
    for i, step in enumerate(plan):
        stages.append(f"history/{step.index}/inputs")
        call(f"history/{step.index}/{step.branch}")
        if i + 1 == len(plan) or plan[i + 1].group != step.group:
            stages.append(f"history/group{step.group}/compare")
    stages.append("history/cache_digests")
    for branch in protocol.BRANCHES:
        stages.append(f"history/observer/{branch}/inputs")
        call(f"history/observer/{branch}")
    stages.extend(("history/conclude", "history/execution_complete"))
    return tuple(stages)


def replay(root: Path, record: Mapping[str, Any], local_slots: Mapping[int, int], *,
           repo: Path, graph_cache: dict | None = None) -> dict[str, Any]:
    """Hash every original; inspect each identical actual graph once per fleet."""
    if (record.get("protocol") != protocol.PROTOCOL
            or record.get("profile") != admission.PROFILE
            or record.get("compile_only") is not False
            or record.get("diagnostic_only") is not True
            or record.get("history_execution_complete") is not True
            or record.get("planned_call_count") != 331
            or set(record.get("programs", {})) != set(protocol.PROGRAMS)):
        raise ValueError("history execution evidence identity differs")

    def inspect(name: str, saved: Mapping, stable: str, optimized: str) -> None:
        key = (admission.PROFILE, name, sha256(stable.encode()).hexdigest(),
               sha256(optimized.encode()).hexdigest(), json.dumps(saved["compiled_memory"], sort_keys=True))
        if graph_cache is None or key not in graph_cache:
            checked = admission.inspect_program(name, stable, optimized,
                                                saved["compiled_memory"], repo=repo)
            if graph_cache is not None:
                graph_cache[key] = checked
        else:
            checked = graph_cache[key]
        if checked.get("passed") is not True:
            raise ValueError("history independently inspected graph refused")
        same_json(saved["admission"], checked, "history actual graph report")

    journal = validate_graph_journal(root, record, protocol_id=protocol.PROTOCOL,
        profile=admission.PROFILE, names=protocol.PROGRAMS, journal_type=HistoryJournal,
        inspect=inspect, identity_fields=dict(protocol=protocol.PROTOCOL, profile=admission.PROFILE,
            compile_only=False, diagnostic_only=True, code_hash=record["code_hash"],
            launch_rank=record["launch_rank"]),
        capture_report=dict(scope="ORIGINAL_CAPTURE_ONLY_NOT_ADMISSION"))
    same_json([r["stage"] for r in journal], list(expected_stages()), "history original journal phase order")
    times = [r["monotonic_seconds"] for r in journal]
    if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
            or times != sorted(times)):
        raise ValueError("history journal monotonic times differ")
    calls = load_calls(root, record)
    validate_call_sequence(record, calls, expected=call_schedule(), local_slots=local_slots,
                           names=protocol.PROGRAMS, budgeter=memory_budget)
    return dict(compiler_programs=len(protocol.PROGRAMS), completed_calls=len(calls),
                journal_stages=len(journal), original_call_bytes=record["call_original_bytes"],
                execution_evidence_replayed=True, numerical_promotion=False,
                materializer_or_boundary_replay=False, fleet_or_cleanup_admission=False)
