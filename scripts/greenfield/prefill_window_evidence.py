"""Original-array and runtime-memory replay for the fixed layer6 discriminator.

The existing campaign must separately bind run/source/checkpoint/fleet identities
and generation-qualified files. This module grants no launch or promotion authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield.prefill_window_worker import (
    WindowNumericalJournal,
    validate_memory_owners,
)


def same_json(actual: Any, expected: Any, description: str) -> None:
    """Compare full serialized evidence, including nested bool/int/float types."""
    encoded = json.dumps(actual, sort_keys=True, allow_nan=False)
    if actual != json.loads(encoded) or encoded != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"window {description} differs from original replay")


def expected_calls() -> tuple[tuple[str, str], ...]:
    return (
        ("wk_decode", "wk_decode"),
        ("wk_promote", "wk_promote"),
        *(
            (phase, name)
            for case in protocol.CASES
            for phase, name in (
                (case + "/candidate", "candidate"),
                *((case + f"/control{tile}", "control") for tile in range(4)),
            )
        ),
    )


def validate_calls(
    record: Mapping[str, Any], *, local_slots: Mapping[int, int]
) -> None:
    """Recompute every predispatch budget and bind all post-call lifetime peaks."""
    calls = record["call_evidence"]
    if len(calls) != len(expected_calls()):
        raise ValueError("window requires exactly two WK and fifteen model calls")
    last: dict[int, dict[str, int]] = {}
    for call, (phase, name) in zip(calls, expected_calls()):
        seconds = call.get("completed_call_seconds")
        if (
            call.get("phase") != phase
            or call.get("graph") != name
            or call.get("completed") is not True
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(
                "window completed-call order or diagnostic duration differs"
            )
        census = call["census"]
        validate_memory_owners(
            census["devices"],
            local_slots=local_slots,
            process_index=record["jax_process_index"],
        )
        before = {r["device_id"]: r["memory_stats"] for r in census["devices"]}
        if last and any(
            before[d]["bytes_limit"] != last[d]["bytes_limit"]
            or before[d]["peak_bytes_in_use"] < last[d]["peak_bytes_in_use"]
            for d in last
        ):
            raise ValueError("window between-call lifetime memory counters regressed")
        budget = admission.memory_budget(
            census, call["compiled_memory"], active_graph=name
        )
        same_json(
            call["compiled_memory"],
            {n: record["programs"][n]["compiled_memory"] for n in admission.PROGRAMS},
            "compiled call memory",
        )
        same_json(call["budget"], budget, "memory budget")
        if budget["estimate_fits"] is not True:
            raise ValueError("window predispatch memory reserve failed")
        after = call["post_memory"]
        validate_memory_owners(
            after, local_slots=local_slots, process_index=record["jax_process_index"]
        )
        for row in after:
            prior = before[row["device_id"]]
            if (
                any(
                    type(row[k]) is not int or row[k] < 0
                    for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                )
                or not row["bytes_in_use"]
                <= row["peak_bytes_in_use"]
                <= row["bytes_limit"]
                or row["bytes_limit"] != prior["bytes_limit"]
                or row["peak_bytes_in_use"] < prior["peak_bytes_in_use"]
                or row["bytes_limit"] - row["peak_bytes_in_use"]
                < admission.REQUIRED_RESERVE_BYTES
            ):
                raise ValueError("window post-call memory counters/reserve failed")
        last = {r["device_id"]: r for r in after}


def validate_files(root: Path, record: Mapping[str, Any]) -> None:
    """Replay actual graph/journal/NPZ bytes, never trust the worker's pass label."""
    if (
        record.get("protocol") != protocol.PROTOCOL
        or record.get("profile") != admission.PROFILE
        or record.get("compile_only") is not False
        or record.get("performance_claim") is not False
        or record.get("current_phase") != "numerical_complete"
        or set(record["programs"]) != set(admission.PROGRAMS)
        or set(record["cases"]) != set(protocol.CASES)
    ):
        raise ValueError("window numerical scope or completeness differs")
    same_json(
        [record["model_executable_calls"], record["wk_executable_calls"]],
        [15, 2],
        "completed totals",
    )
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    validate_calls(record, local_slots=slots)
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record["compile_journal_sha256"]:
        raise ValueError("window numerical journal bytes differ")
    journal = [json.loads(line) for line in raw.splitlines()]
    same_json(
        journal[0]["identity"],
        dict(
            protocol=protocol.PROTOCOL,
            profile=admission.PROFILE,
            compile_only=False,
            code_hash=record["code_hash"],
            launch_rank=record["launch_rank"],
        ),
        "journal identity",
    )
    if any(
        r.get("artifact_kind") != WindowNumericalJournal.artifact_kind
        or r.get("status") != WindowNumericalJournal.status
        or r.get("performance_claim") is not False
        or r.get("numerical_claim") is not False
        or r.get("passed", True) is not True
        for r in journal
    ):
        raise ValueError("window journal scope or failed phase")
    # Compile stages must precede any numerical dispatch. Phase events retain
    # the last graph ID; distinguish them by their exact stage, not graph alone.
    compile_stages = ("lower_compile_started", "compiled", "raw_written", "inspected")
    observed = [
        (r["graph"], r["stage"]) for r in journal if r["stage"] in compile_stages
    ]
    same_json(
        [list(v) for v in observed],
        [[n, s] for n in admission.PROGRAMS for s in compile_stages],
        "compile journal order",
    )
    for name in admission.PROGRAMS:
        p = record["programs"][name]
        stable = (root / f"{name}.stablehlo.mlir").read_text()
        optimized = (root / f"{name}.optimized_hlo.txt").read_text()
        for text, key in (
            (stable, "stablehlo_sha256"),
            (optimized, "optimized_hlo_sha256"),
        ):
            if sha256(text.encode()).hexdigest() != p[key]:
                raise ValueError("window original graph bytes differ")
        admission.validate_program_report(
            p["admission"], name, stable, optimized, p["compiled_memory"]
        )
        stages = [
            r for r in journal if r["graph"] == name and r["stage"] in compile_stages
        ]
        same_json(
            stages[1]["compiled_memory"], p["compiled_memory"], "compile allocation"
        )
        same_json(stages[1]["seconds"], p["compile_seconds"], "compile duration")
        for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
            same_json(stages[2][key], p[key], "journal graph SHA")
        same_json(stages[3]["report"], p["admission"], "journal graph report")
    stages = [r["stage"] for r in journal]
    required = ["identity", "prepare"]
    for name in admission.PROGRAMS:
        required.extend((*compile_stages, "compile/" + name))

    def call_phases(phase):
        return (phase + "/memory", phase + "/execute", phase + "/memory_after")

    required.extend(
        (
            *call_phases("wk_decode"),
            *call_phases("wk_promote"),
            "wk_boundary",
            "input_specs",
            "rotary_fixture",
        )
    )
    for case in protocol.CASES:
        required.extend(
            case + "/" + s for s in ("host", "inputs", "inputs_ready", "input_capture")
        )
        required.extend(call_phases(case + "/candidate"))
        for tile in range(4):
            phase = case + f"/control{tile}"
            required.extend((phase + "_inputs", phase + "_ready", *call_phases(phase)))
        required.append(case + "/comparison")
    required.append("numerical_complete")
    same_json(stages, required, "complete numerical phase order")
    for case in protocol.CASES:
        path = root / f"{case}.npz"
        declared = record["cases"][case]
        if sha256(path.read_bytes()).hexdigest() != declared["npz_sha256"]:
            raise ValueError("window original case bytes differ")
        replay = protocol.replay_case(path, case=case, slots_by_device=slots)
        same_json(declared["replay"], replay, "original array comparison")
        if not (
            replay["passed"] is True
            and declared["passed"] is True
            and declared["complete"] is True
        ):
            raise ValueError("window original case comparison refused")
    validate_wk(root, record, slots)


def validate_wk(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int]
) -> None:
    """Bind completed individual WK captures to the actual BF16/F32 boundary."""
    if set(record["wk_originals"]) != {"wk_decode", "wk_promote"}:
        raise ValueError("window WK original inventory differs")
    arrays = {}
    for name, digest in {
        **record["wk_originals"],
        "wk_boundary": record["wk_boundary_sha256"],
    }.items():
        path = root / f"{name}.npz"
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("window WK original bytes differ")
        with np.load(path, allow_pickle=False) as saved:
            arrays[name] = dict(saved)
    expected = {str(d) for d in slots}
    if any(set(arrays[n]) != expected for n in ("wk_decode", "wk_promote")) or set(
        arrays["wk_boundary"]
    ) != {f"{kind}_{d}" for d in slots for kind in ("bf16", "fp32")}:
        raise ValueError("window WK original owner inventory differs")
    for d in slots:
        raw, promoted = arrays["wk_decode"][str(d)], arrays["wk_promote"][str(d)]
        if (
            raw.dtype != np.uint16
            or raw.shape != (128, 6144)
            or promoted.dtype != np.float32
            or promoted.shape != raw.shape
            or not np.isfinite(promoted).all()
            or not np.array_equal(raw.view(protocol.BF16).astype(np.float32), promoted)
        ):
            raise ValueError("window completed WK numerical boundary differs")
        for name, value in ((f"bf16_{d}", raw), (f"fp32_{d}", promoted)):
            other = arrays["wk_boundary"][name]
            if (
                other.dtype != value.dtype
                or other.shape != value.shape
                or other.tobytes() != value.tobytes()
            ):
                raise ValueError(
                    "window completed WK boundary not its original capture"
                )
