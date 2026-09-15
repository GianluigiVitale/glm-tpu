"""Deferred decode preparation through the existing fleet-voted call path.

No model implementation or launch authority. The caller has released prefill
code and retains the completed request cache; the all-live census also counts
any forgotten borrowers. Original graph admission still precedes each call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax

from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    budget_resident_execution, capture_resident_buffers, capture_identified_device_memory,
)
from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
    Ws32StrategyNdDenseOverlay, strategy_nd_dense_tensor_names,
)
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield.ws32_budgeted_calls import BudgetedCalls, validate_memory_owners
from scripts.greenfield.microbench_fp8_matmul import _atomic_json

ROLES = ("exact_materialize", "exact_promote")


def overlay_preflight(overlay: Ws32StrategyNdDenseOverlay, calls: BudgetedCalls,
                      roots: Mapping[str, Any]) -> None:
    """Bound placement before the unchanged loader's first device_put.

    Called with its checksum-verified metadata, not file sizes guessed from
    filenames. Allow two copies of the small overlay during shard/global-array
    assembly; this is a conservative capacity allowance, not a copy operation.
    No model executable is resident at this boundary. Full reserve is retained.
    """
    if not isinstance(overlay, Ws32StrategyNdDenseOverlay) or calls.programs:
        raise ValueError("overlay preflight needs verified metadata and no resident programs")
    census = capture_resident_buffers(roots, devices=tuple(jax.local_devices()))
    validate_memory_owners(census["devices"], local_slots=calls.local_slots,
                           process_index=calls.record["jax_process_index"])
    if census.get("includes_all_live_arrays") is not True:
        raise ValueError("overlay requires all-live census")
    rows = []
    calls.record["overlay_memory"] = dict(census=census, devices=rows,
        reserve_bytes=runtime.RESERVE, temporary_payload_copies=2,
        manifest_sha256=overlay.manifest["manifest_sha256"],
        manifest_file_sha256=overlay.manifest_file_sha256,
        success_file_sha256=overlay.success_file_sha256)
    for row in census["devices"]:
        slot = calls.local_slots[row["device_id"]]
        expert, feature = divmod(slot, 4)
        payload = 0
        for layer in range(3):
            tensors = overlay.records[layer, expert, feature]["tensors"]
            if set(tensors) != set(strategy_nd_dense_tensor_names(layer)):
                raise ValueError("overlay preflight tensor inventory differs")
            for tensor in tensors.values():
                size = tensor["byte_count"]
                if type(size) is not int or size <= 0:
                    raise ValueError("overlay preflight invalid tensor byte count")
                payload += size
        stats = row["memory_stats"]
        resident = sum(item["bytes"] for item in row["buffers"])
        if (resident != row["accounted_resident_bytes"]
                or stats["bytes_limit"] != runtime.DEVICE_LIMIT
                or not 0 <= stats["bytes_in_use"] <= stats["peak_bytes_in_use"] <= runtime.DEVICE_LIMIT):
            raise ValueError("overlay preflight memory counters differ")
        peak_bound = max(max(resident, stats["bytes_in_use"]) + 2 * payload,
                         stats["peak_bytes_in_use"])
        rows.append(dict(device_id=row["device_id"], slot=slot, payload_bytes=payload,
                         estimated_peak_bytes=peak_bound,
                         headroom_bytes=runtime.DEVICE_LIMIT - peak_bound))
    if any(row["headroom_bytes"] < runtime.RESERVE for row in rows):
        raise ValueError("overlay placement would violate memory reserve")


def overlay_completed(calls: BudgetedCalls) -> None:
    """Retain actual per-owner peak immediately after blocking overlay load."""
    rows = capture_identified_device_memory(tuple(jax.local_devices()))
    calls.record["overlay_device_memory_after"] = rows
    validate_memory_owners(rows, local_slots=calls.local_slots,
                           process_index=calls.record["jax_process_index"])
    before = {row["device_id"]: row["memory_stats"]
              for row in calls.record["overlay_memory"]["census"]["devices"]}
    if any(row["bytes_limit"] != runtime.DEVICE_LIMIT
           or row["peak_bytes_in_use"] < before[row["device_id"]]["peak_bytes_in_use"]
           or runtime.DEVICE_LIMIT - row["peak_bytes_in_use"] < runtime.RESERVE for row in rows):
        raise ValueError("overlay completed peak violates counters/reserve")


def memory_budget(census: Mapping, analyses: Mapping, *, active_graph: str) -> dict:
    """Budget actual resident code, all live arrays, output, scratch and reserve.

    Promotion is compiled after the completed BF16 decode boundary, exactly as
    in the original materializer. Thus the first call has one resident program,
    the second two. This is not the long prefill's shared-code/donation budget.
    """
    if active_graph not in ROLES:
        raise ValueError("unregistered deferred decode call")
    resident = ROLES[:ROLES.index(active_graph) + 1]
    if set(analyses) != set(resident):
        raise ValueError("deferred decode resident program inventory differs")
    if any(row["memory_stats"]["bytes_limit"] != runtime.DEVICE_LIMIT
           for row in census["devices"]):
        raise ValueError("deferred decode device limit differs")
    return budget_resident_execution(census, analyses, active_graph=active_graph,
        resident_graphs=resident, required_reserve_bytes=runtime.RESERVE)


def open_phase(*, root: Path, journal: Any, consensus: Any,
               local_slots: Mapping[int, int], identity: Mapping[str, Any]) -> BudgetedCalls:
    """Create only this run's fresh record, voting before any later collective."""
    error = None
    try:
        root.mkdir(exist_ok=False)
    except Exception as exc:
        error = exc
    agreed = consensus(error is None)
    if error is not None:
        raise error
    if not agreed:
        raise RuntimeError("deferred decode peer refused fresh phase directory")
    record = dict(identity, artifact_kind="ws32_delivery_decode_preparation_v1",
                  phase_contract=runtime.PROFILE, complete=False, programs={},
                  numerical_value_replay=False, performance_claim=False)
    return BudgetedCalls(root=root, record=record, consensus=consensus,
        journal=journal, local_slots=local_slots, budgeter=memory_budget)


def preserve_schema(result: Any, entry: dict) -> None:
    """Describe completed outputs before postflight; no full weight readback.

    This is metadata, NOT a numerical/finite-value comparison. The unchanged
    materializer, checkpoint checks and subsequent runtime health remain needed.
    """
    leaves = jax.tree.leaves(result)
    entry["output_schema"] = []
    if not leaves:
        raise ValueError("deferred decode returned no arrays")
    for value in leaves:
        if not isinstance(value, jax.Array) or value.is_deleted():
            raise ValueError("deferred decode output is not a completed live array")
        entry["output_schema"].append(dict(shape=list(value.shape), dtype=str(value.dtype)))


def finish(calls: BudgetedCalls) -> dict:
    """Record completion only after both calls and release of compiler roots."""
    def complete():
        entries = calls.record["call_evidence"]
        if (calls.programs or len(entries) != 2 or any(
                row.get("graph") != name or row.get("completed") is not True
                or not row.get("output_schema") or len(row.get("post_memory", [])) != 4
                for row, name in zip(entries, ROLES, strict=True))):
            raise ValueError("deferred decode completion/release inventory differs")
        calls.record["complete"] = True
    try:
        calls.phase("delivery_decode/complete", complete)
    except Exception:
        calls.record["complete"] = False
        # Do not hide a peer/refusal cause with a second publication exception.
        try:
            _atomic_json(calls.root / "runner.json", calls.record)
        except Exception:
            pass
        raise
    return calls.record
