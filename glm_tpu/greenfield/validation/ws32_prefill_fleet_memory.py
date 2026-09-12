"""Join batched memory evidence to authenticated owners; no serial schema change."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .ws32_prefill_memory import MEMORY_FIELDS, validate_prefill_memory_record


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"batched memory {name} must be nonnegative int")
    return value


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _owners(
    values: Any,
    expected_ids: set[int],
    process: int,
    *,
    nested: bool,
    limit: int,
    reserve: int,
) -> dict[int, Mapping[str, int]]:
    if type(values) is not list or len(values) != 4:
        raise ValueError("batched memory requires four records per boundary")
    result = {}
    for value in values:
        device = _integer(value["device_id"], "device id")
        owner = _integer(value["process_index"], "process index")
        if device not in expected_ids or device in result or owner != process:
            raise ValueError("batched memory duplicate/foreign owner")
        if value["platform"] != "tpu":
            raise ValueError("batched memory requires TPU counters")
        stats = value["memory_stats"] if nested else value
        used, peak, actual_limit = (
            _integer(stats[key], key)
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        )
        if not used <= peak <= actual_limit or actual_limit != limit:
            raise ValueError("batched memory counters/limit drifted")
        if actual_limit - peak < reserve:
            raise ValueError("batched memory measured peak violates reserve")
        result[device] = stats
    if set(result) != expected_ids:
        raise ValueError("batched memory owner coverage differs")
    return result


def validate_batched_fleet_memory(
    records: Sequence[Mapping[str, Any]],
    ordered_captures: Sequence[Mapping[str, Any]],
    flattened_device_ids: Sequence[int],
    *,
    required_reserve_bytes: int,
    expected_device_limit_bytes: int,
    expected_prefill_analyses: Mapping[str, Mapping[str, int]],
    state_ownership_contract: str | None = None,
) -> dict[str, Any]:
    """Recompute budgets and bind three memory boundaries to all32 actual chips.

    Caller must authenticate captures/mesh and runner provenance first. Inputs
    are small JSON evidence, never tensor payloads. Peaks are lifetime allocator
    peaks (including cold/trace work), NOT isolated prefill allocation or TTFT.
    The caller supplies its pinned profile reserve, limit and actual analyses.
    Consumed-state evidence requires an explicit caller contract, never automatic
    schema detection. Both modes retain the SAME physical-owner and measured-peak
    checks. This helper does not authorize a new worker/profile or graph.
    """
    owned = None
    if state_ownership_contract is not None:
        from scripts.greenfield import ws32_owned_prefill_memory as owned
        if state_ownership_contract != owned.CONTRACT:
            raise ValueError("batched fleet memory ownership contract differs")
    reserve = _integer(required_reserve_bytes, "reserve")
    limit = _integer(expected_device_limit_bytes, "limit")
    if not 0 < reserve < limit:
        raise ValueError("batched memory reserve/limit invalid")
    physical = tuple(_integer(i, "physical id") for i in flattened_device_ids)
    if len(physical) != 32 or len(set(physical)) != 32:
        raise ValueError("batched memory physical mesh must have32 unique chips")
    if len(records) != 8 or len(ordered_captures) != 8:
        raise ValueError("batched memory requires eight authenticated hosts")
    pair = {"prefill_chunk", "prefill_tail"}
    if set(expected_prefill_analyses) != pair:
        raise ValueError("batched memory requires acquired graph pair")
    for analysis in expected_prefill_analyses.values():
        if set(analysis) != set(MEMORY_FIELDS):
            raise ValueError("batched memory acquired analysis schema drifted")
        for key, value in analysis.items():
            _integer(value, key)
    seen_devices, seen_processes, seen_slots = set(), set(), set()
    summaries = []
    for rank, (record, capture) in enumerate(zip(records, ordered_captures)):
        process = _integer(capture["jax_process_index"], "captured process")
        ids = [_integer(i, "captured device") for i in capture["local_device_ids"]]
        expected_ids = set(ids)
        if (
            len(ids) != 4
            or len(expected_ids) != 4
            or seen_devices & expected_ids
            or process in seen_processes
            or process >= 8
            or not expected_ids <= set(physical)
            or _integer(record["launch_process_id"], "launch rank") != rank
            or _integer(record["jax_process_index"], "runner process") != process
            or record["hostname"] != capture["hostname"]
        ):
            raise ValueError("batched memory authenticated fleet identity drifted")
        seen_processes.add(process)
        seen_devices.update(expected_ids)
        slots = record["local_device_slots"]
        if type(slots) is not list or len(slots) != 4:
            raise ValueError("batched memory needs four checkpoint owner slots")
        local_ids = set()
        for item in slots:
            slot = _integer(item["device_slot"], "device slot")
            device = _integer(item["device_id"], "slot device id")
            if (
                slot >= 32
                or slot in seen_slots
                or physical[slot] != device
                or device not in expected_ids
                or device in local_ids
                or _integer(item["expert_coordinate"], "expert coordinate") != slot // 4
                or _integer(item["feature_coordinate"], "feature coordinate")
                != slot % 4
            ):
                raise ValueError("batched memory physical device/slot mapping drifted")
            seen_slots.add(slot)
            local_ids.add(device)
        admission = record["prefill_execution"]["memory_admission"]
        if owned is None:
            validate_prefill_memory_record(admission)
            role_analyses = admission["compiled_memory"]
        else:
            owned.validate_record(admission)
            # The producer records actual compiled-object sharing. Expand roles
            # only for binding to the runner's two actual graph analyses; the
            # budget retains one code allocation iff those roles share an object.
            role_analyses = {
                role: admission["compiled_memory"][name]
                for role, name in admission["executable_roles"].items()
            }
        if (
            type(admission["required_reserve_bytes"]) is not int
            or admission["required_reserve_bytes"] != reserve
            or not _same(role_analyses, expected_prefill_analyses)
            or not _same(
                {g: record["compiled_memory_analysis"][g] for g in pair},
                expected_prefill_analyses,
            )
            or admission["census"].get("execution_peak_measured") is not False
        ):
            raise ValueError(
                "batched memory reserve/analysis/acquisition binding drifted"
            )
        # The selected validator recomputes all budgets, including their
        # exact device coverage and resident executable set. Exact analyses above
        # forbid undeclared extra model programs in this fixed worker lifecycle.
        before = _owners(
            admission["census"]["devices"],
            expected_ids,
            process,
            nested=True,
            limit=limit,
            reserve=reserve,
        )
        prefill = _owners(
            record["batched_prefill_memory"],
            expected_ids,
            process,
            nested=False,
            limit=limit,
            reserve=reserve,
        )
        final = _owners(
            record["batched_device_memory_after_execute"],
            expected_ids,
            process,
            nested=False,
            limit=limit,
            reserve=reserve,
        )
        for device in sorted(expected_ids):
            old, middle, last = (
                boundary[device]["peak_bytes_in_use"]
                for boundary in (before, prefill, final)
            )
            if not old <= middle <= last:
                raise ValueError("batched memory lifetime peak decreased")
            summaries.append(
                dict(
                    process_index=process,
                    device_id=device,
                    device_slot=physical.index(device),
                    bytes_limit=limit,
                    post_prefill_lifetime_peak_bytes=middle,
                    final_lifetime_peak_bytes=last,
                    final_headroom_bytes=limit - last,
                )
            )
    if seen_devices != set(physical) or seen_slots != set(range(32)):
        raise ValueError("batched memory fleet lacks all32 physical owners")
    return dict(
        schema_version=("ws32_batched_fleet_memory_v1" if owned is None
                        else "ws32_owned_batched_fleet_memory_v1"),
        **({"state_ownership_contract": state_ownership_contract}
           if owned is not None else {}),
        owner_count=32,
        required_reserve_bytes=reserve,
        peak_scope="allocator_lifetime_including_load_compile_and_trace",
        post_prefill_lifetime_peak_bytes=max(
            d["post_prefill_lifetime_peak_bytes"] for d in summaries
        ),
        final_lifetime_peak_bytes=max(
            d["final_lifetime_peak_bytes"] for d in summaries
        ),
        minimum_final_headroom_bytes=min(d["final_headroom_bytes"] for d in summaries),
        devices=sorted(summaries, key=lambda d: d["device_slot"]),
    )
