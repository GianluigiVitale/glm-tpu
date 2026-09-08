"""Bounded equal-work MoE phase baseline; not full-model performance admission.

These helpers never launch a workflow. The protected controller must establish
checkpoint/code/physical-owner identity, validate original numerical arrays and
publish fleet evidence before using the resulting baseline in a phase budget.
Historical B17 arithmetic admission is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import math
import time
from typing import Any

import numpy as np

from glm_tpu.greenfield.benchmarking import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    compare_bounded_tensor,
)

PROTOCOL = "ws32-prefill-moe-equal128-b16-b128-baseline-v1"
KERNEL = "ws32_prefill_moe_scaling_baseline"
CONTROL_ROWS = 16
WINDOW_ROWS = 128
ROW_TILE = 8
WARMUP = 10
ITERATIONS = 50
PHASE_BUDGET_SECONDS = 120.0
COMPILED_MEMORY_LIMIT_BYTES = 1 << 30
TIMING_SCOPE = "PREPLACED_INPUTS_SUM_OF_COMPLETED_CALL_INTERVALS_EXCLUDING_CONSENSUS"


def split_equal_work(inputs: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """Call BEFORE timing: slice live rows only; share all weight arguments."""
    if len(inputs) < 3 or any(x.shape[0] != WINDOW_ROWS for x in inputs[:3]):
        raise ValueError("equal-work baseline requires128 input/route/weight rows")
    return tuple(
        tuple(x[start : start + CONTROL_ROWS] for x in inputs[:3]) + tuple(inputs[3:])
        for start in range(0, WINDOW_ROWS, CONTROL_ROWS)
    )


def device_active_tiles(route_indices: Any) -> Any:
    """Use the SAME metadata generator/arguments as all three grouped calls.

    Execute and complete this OUTSIDE timed MoE calls. Counts are not inferred
    from labels or ceil(expert_rows/8), which misses straddling group boundaries.
    """
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata
    from glm_tpu.greenfield.kernels.prefill_routes import group_prefill_routes

    if route_indices.shape not in ((CONTROL_ROWS, 8), (WINDOW_ROWS, 8)):
        raise ValueError("unregistered MoE baseline geometry")
    routes = group_prefill_routes(route_indices)
    counts = []
    for owner in range(8):
        _, active = make_group_metadata(
            group_sizes=routes.group_sizes,
            m=route_indices.size,
            tm=ROW_TILE,
            start_group=jnp.int32(owner * 32),
            num_nonzero_groups=32,
            visit_empty_groups=False,
        )
        counts.append(active)
    return jnp.where(routes.valid, jnp.stack(counts), -1)


def occupancy(route_indices: np.ndarray, active_tiles: np.ndarray) -> dict[str, Any]:
    """Independently rederive scheduled tiles from the supplied route bytes."""
    ids, tiles = np.asarray(route_indices), np.asarray(active_tiles)
    if (
        ids.dtype != np.int32
        or ids.shape not in ((CONTROL_ROWS, 8), (WINDOW_ROWS, 8))
        or not np.all((ids >= 0) & (ids < 256))
        or any(np.unique(row).size != 8 for row in ids)
        or tiles.shape != (8,)
        or tiles.dtype.kind not in "iu"
    ):
        raise ValueError("invalid supplied route/metadata geometry")
    counts = np.bincount(ids.ravel(), minlength=256)
    ends = np.cumsum(counts)
    starts = np.r_[0, ends[:-1]]
    group_tiles = np.where(
        counts > 0, (ends + ROW_TILE - 1) // ROW_TILE - starts // ROW_TILE, 0
    )
    expected_tiles = group_tiles.reshape(8, 32).sum(axis=1)
    if not np.array_equal(tiles, expected_tiles):
        raise ValueError("executing grouped metadata differs from route offsets")
    owners = counts.reshape(8, 32).sum(axis=1)
    total_tiles, active = int(tiles.sum()), int(np.count_nonzero(counts))
    return dict(
        route_sha256=sha256(ids.tobytes()).hexdigest(),
        rows=int(ids.shape[0]),
        route_rows=int(ids.size),
        group_sizes=counts.tolist(),
        owner_route_rows=owners.tolist(),
        active_experts=active,
        mean_routes_per_active_expert=float(ids.size / active),
        max_expert_routes=int(counts.max()),
        max_owner_routes=int(owners.max()),
        # Gate/up/down use identical row metadata, not three independent totals.
        active_tiles_per_owner_per_projection=tiles.tolist(),
        scheduled_lanes_per_projection=total_tiles * ROW_TILE,
        useful_lane_fraction=float(ids.size / (total_tiles * ROW_TILE)),
        max_owner_active_tiles=int(tiles.max()),
        route_scope="SUPPLIED_SYNTHETIC_SCENARIO_NOT_OBSERVED_MODEL_OCCUPANCY",
    )


def compare_equal_work(
    wide: np.ndarray, small: np.ndarray, scalar: np.ndarray, legacy_row: np.ndarray
) -> dict[str, Any]:
    """No adjustable tolerance; compare every row and the real captured row0."""
    if (
        wide.shape != (WINDOW_ROWS, 1536)
        or small.shape != wide.shape
        or scalar.shape != wide.shape
        or legacy_row.shape != (1, 1536)
    ):
        raise ValueError("equal-work output geometry differs")
    comparisons = {}
    for name, actual, expected in (
        ("wide_vs_scalar", wide, scalar),
        ("control_vs_scalar", small, scalar),
        ("wide_vs_control", wide, small),
    ):
        comparisons[name] = dict(
            aggregate=compare_bounded_tensor(
                actual, expected, REAL_LAYER_OUTPUT_TOLERANCE
            ),
            per_row=[
                compare_bounded_tensor(
                    actual[i : i + 1], expected[i : i + 1], REAL_LAYER_OUTPUT_TOLERANCE
                )
                for i in range(WINDOW_ROWS)
            ],
        )
    legacy = {
        name: compare_bounded_tensor(value[:1], legacy_row, REAL_LAYER_OUTPUT_TOLERANCE)
        for name, value in (("wide", wide), ("control", small), ("scalar", scalar))
    }
    passed = all(
        c["aggregate"]["passed"] and all(row["passed"] for row in c["per_row"])
        for c in comparisons.values()
    ) and all(c["passed"] for c in legacy.values())
    return dict(passed=passed, comparisons=comparisons, row0_vs_legacy=legacy)


def measure_completed_calls(
    program: Callable[..., Any],
    inputs: Sequence[tuple[Any, ...]],
    *,
    complete: Callable[[Any], Any],
    consensus: Callable[[bool], bool],
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Time exactly128 live rows, resident inputs; block EACH call to completion.

    No input slicing/transfer/reference/capture inside the measured interval.
    Sum dispatch-plus-completion intervals, NOT pipelined/sustained throughput.
    Phase-matched failure/budget votes are OUTSIDE each call interval. A peer's
    refusal stops every host before the next call, including within B16 samples.
    """
    if len(inputs) not in (1, 8) or not math.isfinite(deadline):
        raise ValueError("equal-work schedule/deadline differs")
    rows = WINDOW_ROWS if len(inputs) == 1 else CONTROL_ROWS
    if any(len(v) < 3 or any(x.shape[0] != rows for x in v[:3]) for v in inputs):
        raise ValueError("schedule is not exactly128 live rows")
    samples = []
    for iteration in range(WARMUP + ITERATIONS):
        if not consensus(clock() < deadline):
            raise TimeoutError("MoE baseline measured-phase budget exceeded")
        seconds = 0.0
        for values in inputs:
            error = None
            elapsed = 0.0
            started = clock()
            try:
                complete(program(*values))
                elapsed = clock() - started
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("invalid completed-call duration")
            except Exception as exc:
                error = exc
            if not consensus(error is None):
                raise RuntimeError(
                    "MoE baseline completed-call fleet failure"
                ) from error
            if error is not None:
                raise RuntimeError(
                    "MoE baseline consensus ignored local failure"
                ) from error
            seconds += elapsed
            if not consensus(clock() < deadline):
                raise TimeoutError("MoE baseline measured-phase budget exceeded")
        if iteration >= WARMUP:
            samples.append(seconds * 1000)
    return dict(
        protocol=PROTOCOL,
        live_rows=WINDOW_ROWS,
        calls_per_sample=len(inputs),
        rows_per_call=rows,
        warmup=WARMUP,
        iterations=ITERATIONS,
        samples_ms=samples,
        timing_scope=TIMING_SCOPE,
    )


def fleet_timing(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Worst-host wall per aligned sample, not sum/average of chip throughput."""
    if len(records) != 8:
        raise ValueError("timing requires eight authenticated rank records")
    calls = records[0].get("calls_per_sample")
    for record in records:
        if (
            record.get("protocol") != PROTOCOL
            or record.get("live_rows") != WINDOW_ROWS
            or record.get("calls_per_sample") != calls
            or calls not in (1, 8)
            or record.get("rows_per_call") != WINDOW_ROWS // calls
            or record.get("warmup") != WARMUP
            or record.get("iterations") != ITERATIONS
            or record.get("timing_scope") != TIMING_SCOPE
        ):
            raise ValueError("fleet timing protocol differs")
        values = record.get("samples_ms")
        if (
            not isinstance(values, list)
            or len(values) != ITERATIONS
            or any(
                type(v) is not float or not math.isfinite(v) or v <= 0 for v in values
            )
        ):
            raise ValueError("fleet completed-call samples differ")
    samples = np.max([r["samples_ms"] for r in records], axis=0)
    return dict(
        **{k: records[0][k] for k in records[0] if k != "samples_ms"},
        fleet_straggler_samples_ms=samples.tolist(),
        p50_ms_per_128_rows=float(np.percentile(samples, 50)),
        p99_ms_per_128_rows=float(np.percentile(samples, 99)),
        p50_ms_per_live_row=float(np.percentile(samples, 50) / WINDOW_ROWS),
        performance_scope="MOE_PHASE_BASELINE_NOT_MODEL_PREFILL_OR_TTFT",
    )
