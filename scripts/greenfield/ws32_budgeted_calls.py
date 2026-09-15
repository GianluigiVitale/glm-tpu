"""Fleet-voted budgeted program calls shared by the native runtime and its replay.

Verbatim (b667f00f) from the retired layer-6 window campaign modules:
``save_arrays``, ``validate_memory_owners`` and ``BudgetedCalls`` from
prefill_window_worker.py, ``start_device_trace``/``voted_trace`` from
prefill_phase_baseline.py, and ``same_json``/``validate_call_sequence`` from
prefill_window_evidence.py. The only edit is the BudgetedCalls journal
annotation, which now names the base journal class the native worker passes.
No phase, vote, memory-budget or evidence-replay behaviour changes here.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    MEMORY_FIELDS,
    capture_identified_device_memory,
    capture_resident_buffers,
)
from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically replace only this run's evolving capture; preserve on failure."""
    temporary = path.with_suffix(".npz.pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path.read_bytes()).hexdigest()


def validate_memory_owners(
    rows: list[dict[str, Any]], *, local_slots: Mapping[int, int], process_index: int
) -> None:
    """Join keyed counters to this already-authenticated process/physical owners."""
    if (
        len(local_slots) != 4
        or len(set(local_slots.values())) != 4
        or any(type(s) is not int or not 0 <= s < 32 for s in local_slots.values())
        or len(rows) != 4
        or {r["device_id"] for r in rows} != set(local_slots)
        or any(
            r["platform"] != "tpu" or r["process_index"] != process_index for r in rows
        )
    ):
        raise ValueError("window memory does not cover its four authenticated owners")


class BudgetedCalls:
    """Local phases vote before dispatch; completed calls vote before successors.

    This cannot recover a distributed executable whose peers hang internally;
    the existing owned worker timeout and controller recovery remain mandatory.
    """

    def __init__(
        self,
        *,
        root: Path,
        record: dict[str, Any],
        consensus: Callable[[bool], bool],
        journal: Ws32NumericalJournal,
        local_slots: Mapping[int, int],
        budgeter: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.root, self.record = root, record
        self.consensus, self.journal = consensus, journal
        self.local_slots = local_slots
        self.budgeter = admission.memory_budget if budgeter is None else budgeter
        self.programs: dict[str, Any] = {}
        self.record["call_evidence"] = []

    def phase(self, name: str, action: Callable[[], Any]) -> Any:
        value, error = None, None
        try:
            value = action()
        except Exception as exc:
            error = exc
        try:
            self.record["current_phase"] = name
            if error is not None:
                self.record["phase_error"] = f"{name}: {type(error).__name__}: {error}"
            _atomic_json(self.root / "runner.json", self.record)
            self.journal.phase(name, passed=error is None)
        except Exception as exc:
            error = error or exc
        agreed = self.consensus(error is None)
        if error is not None:
            raise error
        if not agreed:
            raise RuntimeError(f"window numerical peer refused at {name}")
        return value

    def call(
        self,
        phase: str,
        name: str,
        values: tuple[Any, ...],
        *,
        preserve: Callable[[Any], None],
    ) -> Any:
        import jax

        entry: dict[str, Any] = dict(phase=phase, graph=name, completed=False)
        self.record["call_evidence"].append(entry)

        def preflight():
            # Actual analyses, not just the acquired receipt. Capture all live
            # arrays, including candidate results and old control cache aliases.
            analyses = {}
            for n, program in self.programs.items():
                analysis = program.memory_analysis()
                analyses[n] = {key: getattr(analysis, key) for key in MEMORY_FIELDS}
            census = capture_resident_buffers(
                {"active_inputs": values}, devices=tuple(jax.local_devices())
            )
            validate_memory_owners(
                census["devices"],
                local_slots=self.local_slots,
                process_index=int(self.record["jax_process_index"]),
            )
            budget = self.budgeter(census, analyses, active_graph=name)
            entry.update(census=census, compiled_memory=analyses, budget=budget)
            if not budget["estimate_fits"]:
                raise ValueError("window predispatch memory reserve failed")

        self.phase(phase + "/memory", preflight)

        def execute():
            # No host comparison, report, census or fleet vote inside interval.
            started = time.monotonic()
            result = self.programs[name](*values)
            jax.block_until_ready(result)
            entry.update(
                completed=True, completed_call_seconds=time.monotonic() - started
            )
            # Save completed originals before fallible phase publication or
            # post-memory validation can prevent the caller from receiving them.
            # Persistence/comparison is OUTSIDE the diagnostic call interval.
            preserve(result)
            return result

        result = self.phase(phase + "/execute", execute)

        def postflight():
            stats = capture_identified_device_memory(tuple(jax.local_devices()))
            entry["post_memory"] = stats
            validate_memory_owners(
                stats,
                local_slots=self.local_slots,
                process_index=int(self.record["jax_process_index"]),
            )
            before = {
                r["device_id"]: r["memory_stats"] for r in entry["census"]["devices"]
            }
            if any(
                r["bytes_limit"] != before[r["device_id"]]["bytes_limit"]
                or r["peak_bytes_in_use"] < before[r["device_id"]]["peak_bytes_in_use"]
                or r["bytes_limit"] - r["peak_bytes_in_use"]
                < admission.REQUIRED_RESERVE_BYTES
                for r in stats
            ):
                raise ValueError("window post-call memory reserve/counters failed")

        self.phase(phase + "/memory_after", postflight)
        return result


def start_device_trace(path: str) -> None:
    """Reuse protected layer profiler settings; avoid Python-check event explosion."""
    import jax

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    # Keep host and device/HLO tracing at their original defaults.
    jax.profiler.start_trace(path, profiler_options=options)


@contextmanager
def voted_trace(
    calls: Any, path: Path, *, start: Callable, stop: Callable
) -> Iterator[None]:
    """Every host votes entry and exit, including hosts whose peer failed entry."""
    entered = False

    def enter():
        nonlocal entered
        if path.exists():
            raise ValueError("phase trace directory must be new")
        start(str(path))
        entered = True

    try:
        calls.phase("phase_trace/start", enter)
        yield
    finally:

        def leave():
            if entered:
                stop()

        calls.phase("phase_trace/stop", leave)


def same_json(actual: Any, expected: Any, description: str) -> None:
    """Compare full serialized evidence, including nested bool/int/float types."""
    encoded = json.dumps(actual, sort_keys=True, allow_nan=False)
    if actual != json.loads(encoded) or encoded != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"window {description} differs from original replay")


def validate_call_sequence(
    record: Mapping[str, Any],
    calls: Iterable[Mapping[str, Any]],
    *,
    expected: Iterable[tuple[str, str]],
    local_slots: Mapping[int, int],
    names: tuple[str, ...],
    budgeter: Callable[..., dict],
) -> None:
    """Replay ordinary or streamed witnesses with the same memory/owner rules.

    Strict exhaustion matters: the stream checks its trailing bytes only after
    its final yielded record. Never stop after just the expected call count.
    """
    last: dict[int, dict[str, int]] = {}
    for call, (phase, name) in zip(calls, expected, strict=True):
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
        budget = budgeter(census, call["compiled_memory"], active_graph=name)
        same_json(
            call["compiled_memory"],
            {n: record["programs"][n]["compiled_memory"] for n in names},
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
