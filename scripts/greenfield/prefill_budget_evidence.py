"""Independent original-array and timing replay of the missing DSA budget.

Reuses compiled-journal and call-memory replay. The outer collector still must
authenticate code, generations, topology and all32 physical owners. This local
replay never converts synthetic DSA timings into full-model performance.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield import prefill_window_evidence as shared


def expected_calls() -> tuple[tuple[str, str], ...]:
    ties = tuple(
        (c.name + "/tie", f"dsa_c{c.capacity}")
        for c in probe.cases()
        if c.last_valid_length == c.prompt_length
    )
    samples = tuple(
        (c.name + f"/sample{i}", f"dsa_c{c.capacity}")
        for c in probe.cases()
        for i in range(probe.WARMUP + probe.ITERATIONS)
    )
    return ties + samples


def expected_stages(*, include_overhead: bool = False) -> list[str]:
    stages = ["identity"]
    if include_overhead:
        from scripts.greenfield import prefill_budget_overhead

        stages += ["budget/campaign_bind", *prefill_budget_overhead.expected_stages()]
    stages.append("budget/bind")
    stages.extend(c.name + "/prepare" for c in probe.cases())
    for name in worker.PROGRAMS:
        stages.extend(
            (
                "lower_compile_started",
                "compiled",
                "raw_written",
                "inspected",
                name + "/compile_inspect",
            )
        )
    stages.extend(
        c.name + "/tie_prepare"
        for c in probe.cases()
        if c.last_valid_length == c.prompt_length
    )
    stages.append("budget/sampling_bind")
    for c in probe.cases():
        if c.last_valid_length == c.prompt_length:
            stages.extend(
                c.name + "/tie/" + s for s in ("memory", "execute", "memory_after")
            )
            stages.append(c.name + "/tie_check")
    stages.append("budget/sampling_started")
    for c in probe.cases():
        for i in range(probe.WARMUP + probe.ITERATIONS):
            label = c.name + f"/sample{i}"
            stages.append(label + "_budget")
            stages.extend(
                label + "/" + s for s in ("memory", "execute", "memory_after")
            )
            stages.extend((label + "_check", label + "_post_budget"))
        stages.append(c.name + "/complete")
    stages.append("budget/dsa_complete")
    return stages


def reference_check(expected: dict, slots: dict[int, int]) -> dict:
    hashes = {
        name: sha256(value.tobytes()).hexdigest() for name, value in expected.items()
    }
    return dict(
        passed=True,
        owners={str(d): hashes for d in slots},
        reference="ANALYTIC_ONE_HOT_FP32_SCALED_MOD251",
    )


def validate_original(
    path: Path, digest: str, expected: dict, slots: dict[int, int]
) -> None:
    if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("DSA budget original digest/type differs")
    fields = dict(expected, health=np.ones((1, 1), np.bool_))
    names = {f"device{d}_{name}" for d in slots for name in fields}
    with np.load(path, allow_pickle=False) as original:
        if len(original.files) != len(names) or set(original.files) != names:
            raise ValueError("DSA budget original owner/field inventory differs")
        for device in slots:
            for name, reference in fields.items():
                actual = original[f"device{device}_{name}"]
                if (
                    actual.shape != reference.shape
                    or actual.dtype != reference.dtype
                    or actual.tobytes() != reference.tobytes()
                ):
                    raise ValueError(
                        "DSA budget original differs from analytic reference"
                    )


def validate_files(
    root: Path, record: dict, *, slots: dict[int, int], require_overhead: bool = False
) -> dict:
    """Replay all44 calls,14 original arrays and exact sampling/journal order."""
    fixed = dict(
        protocol=probe.PROTOCOL,
        profile=worker.PROFILE,
        compile_only=False,
        current_phase="budget/dsa_complete",
    )
    shared.same_json({k: record.get(k) for k in fixed}, fixed, "DSA budget scope")
    if (
        set(record["programs"]) != set(worker.PROGRAMS)
        or set(record["budget_cases"]) != {c.name for c in probe.cases()}
        or "phase_error" in record
    ):
        raise ValueError("DSA budget inventory/failure differs")

    def inspect(name, entry, stable, optimized):
        shared.same_json(
            entry["admission"],
            worker.inspect_program(name, stable, optimized, entry["compiled_memory"]),
            "DSA actual graph",
        )

    journal = shared.validate_graph_journal(
        root,
        record,
        protocol_id=probe.PROTOCOL,
        profile=worker.PROFILE,
        names=worker.PROGRAMS,
        journal_type=worker.BudgetJournal,
        inspect=inspect,
    )
    include_overhead = "budget_overhead" in record
    if require_overhead and not include_overhead:
        raise ValueError("full budget campaign is missing overhead evidence")
    shared.same_json(
        [r["stage"] for r in journal],
        expected_stages(include_overhead=include_overhead),
        "DSA full phase order",
    )
    if include_overhead:
        from scripts.greenfield import prefill_budget_overhead

        prefill_budget_overhead.validate_record(
            record["budget_overhead"],
            slots=slots,
            process_index=record["jax_process_index"],
        )
        times = {r["stage"]: r["monotonic_seconds"] for r in journal}
        if (
            not 0
            <= times["overhead/complete"] - times["overhead/start"]
            < prefill_budget_overhead.MAX_SECONDS
        ):
            raise ValueError("overhead journal deadline exceeded")
        last = record["budget_overhead"]["initialization"]["262656"][-1]["after_memory"]
        before = {
            r["device_id"]: r["memory_stats"]
            for r in record["call_evidence"][0]["census"]["devices"]
        }
        if any(
            before[r["device_id"]]["bytes_limit"] != r["bytes_limit"]
            or before[r["device_id"]]["peak_bytes_in_use"] < r["peak_bytes_in_use"]
            for r in last
        ):
            raise ValueError("overhead-to-DSA lifetime memory counter drift")
    shared.validate_call_sequence(
        record,
        record["call_evidence"],
        expected=expected_calls(),
        local_slots=slots,
        names=worker.PROGRAMS,
        budgeter=worker.memory_budget,
    )
    originals = {}
    tied = {}
    for case in probe.cases():
        if case.last_valid_length == case.prompt_length:
            expected = probe.expected_selection(case.valid_lengths, tied=True)
            originals[case.name + "_tie.npz"] = expected
            tied[case.name] = reference_check(expected, slots)
    shared.same_json(record["budget_ties"], tied, "DSA tied reference")
    calls = record["call_evidence"][2:]
    for case_index, case in enumerate(probe.cases()):
        expected = probe.expected_selection(case.valid_lengths)
        for i in (0, probe.WARMUP + probe.ITERATIONS - 1):
            originals[case.name + f"_sample{i}.npz"] = expected
        n = probe.WARMUP + probe.ITERATIONS
        case_calls = calls[case_index * n : (case_index + 1) * n]
        calculated = dict(
            warmup=probe.WARMUP,
            iterations=probe.ITERATIONS,
            samples_seconds=[
                c["completed_call_seconds"] for c in case_calls[probe.WARMUP :]
            ],
            checks=[reference_check(expected, slots)] * n,
            scope="SYNTHETIC_DSA_DISPATCH_THROUGH_COMPLETION_ONLY",
            model_prefill_measured=False,
            model_ttft_measured=False,
        )
        shared.same_json(
            record["budget_cases"][case.name], calculated, "DSA sample rederivation"
        )
    if set(record["budget_originals"]) != set(originals):
        raise ValueError("DSA budget original manifest differs")
    if {p.name for p in root.glob("c*_*.npz")} != set(originals):
        raise ValueError("DSA budget retained unexpected/missing originals")
    for name, expected in originals.items():
        validate_original(
            root / name, record["budget_originals"][name], expected, slots
        )
    sample = record["budget_sampling"]
    elapsed = sample.get("elapsed_seconds")
    if (
        type(elapsed) not in (int, float)
        or not np.isfinite(elapsed)
        or not sum(c["completed_call_seconds"] for c in calls)
        <= elapsed
        < probe.SAMPLING_BUDGET_SECONDS
    ):
        raise ValueError("DSA inclusive sampling budget invalid")
    shared.same_json(
        sample,
        dict(
            complete=True,
            elapsed_seconds=elapsed,
            budget_seconds=probe.SAMPLING_BUDGET_SECONDS,
            scope="SYNTHETIC_DSA_NOT_MODEL_PREFILL_OR_TTFT",
        ),
        "DSA inclusive sampling scope",
    )
    start = next(
        r["monotonic_seconds"]
        for r in journal
        if r["stage"] == "budget/sampling_started"
    )
    end = journal[-1]["monotonic_seconds"]
    if not 0 <= end - start < probe.SAMPLING_BUDGET_SECONDS:
        raise ValueError("DSA journal sampling deadline exceeded")
    return dict(
        passed=True,
        scope="LOCAL_DSA_BUDGET_REPLAY_NOT_FLEET_OR_MODEL_PROOF",
        replayed_originals=len(originals),
        replayed_calls=len(calls) + 2,
    )


def fleet_wall(records: list[dict[str, Any]]) -> dict:
    """Caller MUST authenticate/replay all eight hosts before aggregation."""
    if len(records) != 8 or {r["jax_process_index"] for r in records} != set(range(8)):
        raise ValueError("DSA budget timing needs eight unique processes")
    result = {}
    for case in probe.cases():
        samples = [
            max(r["budget_cases"][case.name]["samples_seconds"][i] for r in records)
            for i in range(probe.ITERATIONS)
        ]
        result[case.name] = dict(
            samples_seconds=samples,
            p50_seconds=float(np.percentile(samples, 50)),
            p99_seconds=float(np.percentile(samples, 99)),
        )
    return dict(
        scope="SYNTHETIC_DSA_NOT_MODEL_PREFILL_OR_TTFT",
        aggregation="MAX_HOST_PER_ALIGNED_UNPROFILED_SAMPLE",
        cases=result,
    )
