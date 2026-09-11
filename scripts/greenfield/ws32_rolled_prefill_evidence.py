"""Replay the compiler-only originals; never authorize model execution."""

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield.prefill_window_evidence import same_json

PHASES = (
    "budget_runtime",
    "rolled_compile_setup",
    "rolled_compile_prefill_chunk",
    "rolled_compile_prefill_tail",
    "rolled_compile_pair_validation",
    "rolled_compile_finalize",
    "rolled_compile_terminal",
)
FILES = (
    "runner.json",
    "worker.log",
    "compile_journal.jsonl",
    *(
        f"{name}.{form}"
        for name in worker.PROGRAMS
        for form in ("stablehlo.mlir", "optimized_hlo.txt")
    ),
)
NOTE = (
    "Production78-layer B128/B114 graphs compiled from authenticated metadata and "
    "abstract inputs only. No checkpoint payload loading, WK/model execution, "
    "numerical correctness, numerical peak HBM, throughput or TTFT claim. "
    "Actual optimized loop/cache/health admission remains separate."
)


def phases(
    canonical_dense: bool = False, *, full_canonical: bool = False,
    history: bool = False,
) -> tuple[str, ...]:
    mode = worker.compile_mode(canonical_dense, full_canonical=full_canonical, history=history)
    return (
        "budget_runtime",
        mode.prefix + "_setup",
        *(mode.prefix + "_" + name for name in mode.programs),
        mode.prefix + "_pair_validation",
        mode.prefix + "_finalize",
        mode.prefix + "_terminal",
    )


def files(
    canonical_dense: bool = False, *, full_canonical: bool = False,
    history: bool = False,
) -> tuple[str, ...]:
    mode = worker.compile_mode(canonical_dense, full_canonical=full_canonical, history=history)
    return (
        "runner.json",
        "worker.log",
        "compile_journal.jsonl",
        *(
            f"{name}.{form}"
            for name in mode.programs
            for form in ("stablehlo.mlir", "optimized_hlo.txt")
        ),
    )


def validate_local(
    root: Path,
    record: Mapping[str, Any],
    *,
    repo: Path,
    local_devices: set[int],
    canonical_dense: bool = False,
    full_canonical: bool = False,
    history: bool = False,
) -> dict[str, Any]:
    """Bind source-reviewed zero-dispatch lifecycle to both original graph files.

    The outer collector authenticates code/runtime/fleet/physical owners and GCS
    generations. A journal is a worker assertion, not an independent trace of
    absent execution; the fixed reviewed worker supplies that scope.
    """
    mode = worker.compile_mode(canonical_dense, full_canonical=full_canonical, history=history)
    identity = worker.journal_identity(
        dict(record), canonical_dense=canonical_dense, full_canonical=full_canonical,
        history=history,
    )
    if (
        record.get("status") != "SUCCESS"
        or record.get("compiler_acquisition_complete") is not True
    ):
        raise ValueError("rolled compiler acquisition incomplete")
    if any(key in record for key in ("error", "finalization_error", "phase_error")):
        raise ValueError("rolled compiler record contains failure")
    recorded_phases = record.get("acquisition_phases", {})
    if set(recorded_phases) != set(
        phases(canonical_dense, full_canonical=full_canonical, history=history)
    ):
        raise ValueError("rolled compiler phase inventory differs")
    for phase in recorded_phases.values():
        if (
            phase.get("status") != "COMPLETE"
            or phase.get("error") is not None
            or type(phase.get("seconds")) not in (float, int)
            or not math.isfinite(phase["seconds"])
            or phase["seconds"] < 0
        ):
            raise ValueError("rolled compiler phase not complete")
    pins = json.loads(
        (
            repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
        ).read_text()
    )
    same_json(
        record.get("abstract_metadata"),
        dict(
            manifest_sha256=pins["expected_manifest_sha256"],
            source_inventory_sha256=pins["source_inventory_sha256"],
            payload_reads=False,
            input_kind="SHAPE_DTYPE_STRUCT_ONLY",
        ),
        "rolled authenticated metadata",
    )
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record.get("compile_journal_sha256"):
        raise ValueError("rolled compiler journal digest differs")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = [(None, "identity")]
    for name in mode.programs:
        expected.extend(
            (name, stage)
            for stage in (
                "lower_compile_started",
                "compiled",
                "raw_written",
                "inspected",
            )
        )
    expected.append((mode.programs[-1], "preserved_pair_verified"))
    if [(r.get("graph"), r.get("stage")) for r in rows] != expected:
        raise ValueError("rolled compiler journal sequence differs")
    same_json(rows[0].get("identity"), identity, "rolled journal identity")
    previous = -math.inf
    for row in rows:
        fixed = dict(
            schema_version=1,
            artifact_kind=mode.journal_type.artifact_kind,
            status=mode.journal_type.status,
            performance_claim=False,
            numerical_claim=False,
        )
        same_json({key: row.get(key) for key in fixed}, fixed, "rolled journal scope")
        now = row.get("monotonic_seconds")
        if type(now) not in (float, int) or not math.isfinite(now) or now < previous:
            raise ValueError("rolled journal timestamps differ")
        previous = now
        name, stage = row["graph"], row["stage"]
        if stage == "compiled":
            saved = record["programs"][name]
            same_json(
                row.get("compiled_memory"),
                saved["compiled_memory"],
                "rolled compile memory",
            )
            if (
                type(row.get("seconds")) not in (int, float)
                or type(saved.get("compile_seconds")) not in (int, float)
                or row.get("seconds") != saved.get("compile_seconds")
                or not math.isfinite(row["seconds"])
                or row["seconds"] < 0
            ):
                raise ValueError("rolled compiler duration differs")
            devices = row.get("device_memory", [])
            if (
                len(devices) != 4
                or {d.get("device_id") for d in devices} != local_devices
                or any(
                    type(d.get("device_id")) is not int
                    or not isinstance(d.get("stats"), dict)
                    for d in devices
                )
            ):
                raise ValueError("rolled compiler memory owners differ")
        elif stage == "raw_written":
            saved = record["programs"][name]
            for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
                if row.get(key) != saved[key]:
                    raise ValueError("rolled journal original graph differs")
        elif stage == "inspected":
            same_json(row.get("report"), worker.CAPTURE, "rolled capture-only scope")
    report = mode.preparation.validate_preserved_pair(root, record, repo=repo)
    same_json(rows[-1].get("report"), report, "rolled journal pair replay")
    same_json(record.get("preserved_pair"), report, "rolled worker pair replay")
    return report
