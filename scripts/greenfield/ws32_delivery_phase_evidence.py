"""Replay long preparation originals and their all-live physical memory checks.

The parent first authenticates the source ledger, topology, model graph reports
and prefill/final memory. This adds the two preparation phases; it does not
reconstruct weight values or claim independent numerical equality.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import _LOCAL_CONTRACT
from glm_tpu.greenfield.validation.ws32_prefill_fleet_memory import _owners
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield import ws32_delivery_wk as wk
from scripts.greenfield import ws32_delivery_decode as decode
from scripts.greenfield.prefill_window_evidence import same_json, validate_call_sequence
from scripts.greenfield.ws32_history_call_evidence import load_calls
from scripts.greenfield.ws32_delivery_phase_transport import file_limits
from scripts.greenfield.ws32_history_preflight import _plain_path


def _read(root: Path, rank: int, name: str) -> dict:
    path = root / name
    _plain_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= file_limits(rank)[name]:
        raise ValueError("delivery phase original type/size differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("delivery phase record is not an object")
    return value


def _boundary(rows: Any, slots: Mapping[int, int], process: int, *, nested: bool) -> dict:
    return _owners(rows, set(slots), process, nested=nested,
                   limit=runtime.DEVICE_LIMIT, reserve=runtime.RESERVE)


def _advance(before: Mapping, after: Mapping) -> None:
    if set(before) != set(after) or any(
        after[d]["peak_bytes_in_use"] < before[d]["peak_bytes_in_use"] for d in before
    ):
        raise ValueError("delivery between-phase lifetime peak regressed")


def overlay_replay(record: Mapping, parent: Mapping, slots: Mapping[int, int]) -> tuple[dict, dict]:
    """Recompute placement bound from the same fixed verified tensor contract."""
    value = record["overlay_memory"]
    overlay = parent["strategy_nd_dense_overlay"]
    if (value.get("reserve_bytes") != runtime.RESERVE
            or type(value.get("reserve_bytes")) is not int
            or type(value.get("temporary_payload_copies")) is not int
            or value["temporary_payload_copies"] != 2
            or value["census"].get("includes_all_live_arrays") is not True):
        raise ValueError("delivery overlay reserve/census differs")
    for key in ("manifest_sha256", "manifest_file_sha256", "success_file_sha256"):
        if value[key] != overlay[key]:
            raise ValueError("delivery overlay identity differs")
    process = parent["jax_process_index"]
    before = _boundary(value["census"]["devices"], slots, process, nested=True)
    payload = 3 * sum(math.prod(shape) * dtype.itemsize for shape, dtype, _, _ in _LOCAL_CONTRACT)
    expected = []
    for row in value["census"]["devices"]:
        resident = row["accounted_resident_bytes"]
        if (type(resident) is not int or resident < 0
                or any(type(b["bytes"]) is not int or b["bytes"] < 0 for b in row["buffers"])
                or resident != sum(b["bytes"] for b in row["buffers"])):
            raise ValueError("delivery overlay resident bytes differ")
        stats = before[row["device_id"]]
        peak = max(max(resident, stats["bytes_in_use"]) + 2 * payload, stats["peak_bytes_in_use"])
        expected.append(dict(device_id=row["device_id"], slot=slots[row["device_id"]],
                             payload_bytes=payload, estimated_peak_bytes=peak,
                             headroom_bytes=runtime.DEVICE_LIMIT - peak))
    same_json(value["devices"], expected, "delivery overlay placement budget")
    if any(row["headroom_bytes"] < runtime.RESERVE for row in expected):
        raise ValueError("delivery overlay placement reserve failed")
    after = _boundary(record["overlay_device_memory_after"], slots, process, nested=False)
    _advance(before, after)
    return before, after


def validate_rank(*, root: Path, rank: int, parent: Mapping, full_index_layers: Sequence[int]) -> dict:
    """Replay42 WK and two decode calls, linking their original phase records."""
    slots = {r["device_id"]: r["device_slot"] for r in parent["local_device_slots"]}
    process = parent["jax_process_index"]
    if parent.get("batched_prefill_profile") != runtime.PROFILE or len(slots) != 4:
        raise ValueError("delivery phase parent profile/owners differ")
    phases = []
    for folder, field, kind, roles in (
        (f"delivery_wk.rank{rank}", "delivery_wk_phase", "ws32_delivery_wk_phase_v1", wk.ROLES),
        (f"delivery_decode.rank{rank}", "delivery_decode_preparation", "ws32_delivery_decode_preparation_v1", decode.ROLES),
    ):
        record = _read(root, rank, folder + "/runner.json")
        same_json(record, parent[field], "delivery nested original")
        if (record.get("artifact_kind") != kind or record.get("complete") is not True
                or record.get("phase_contract") != runtime.PROFILE
                or record.get("performance_claim") is not False
                or set(record["programs"]) != set(roles)):
            raise ValueError("delivery preparation incomplete or wrong phase")
        for key in ("code_hash", "launch_process_id", "jax_process_index",
                    "checkpoint_manifest_sha256", "checkpoint_success_sha256"):
            same_json(record[key], parent[key], "delivery parent identity")
        for name in roles:
            same_json(record["programs"][name]["admission"], parent["graphs"][name], "delivery graph")
            same_json(record["programs"][name]["compiled_memory"],
                      parent["compiled_memory_analysis"][name], "delivery allocation")
            same_json(record["programs"][name]["compile_seconds"],
                      parent["compile_seconds"][name], "delivery compile wall")
        phases.append(record)
    wk_record, decode_record = phases
    if (wk_record.get("model_math_changed") is not False
            or decode_record.get("numerical_value_replay") is not False):
        raise ValueError("delivery preparation numerical scope differs")
    timings = parent["delivery_phase_timings"]
    if (set(timings) != {"base_checkpoint_load_seconds", "wk_prepare_wall_seconds",
                         "decode_overlay_load_seconds", "decode_materialization_wall_seconds"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in timings.values())
            or timings["wk_prepare_wall_seconds"] != wk_record["phase_wall_seconds"]
            or parent["load_seconds"] != (timings["base_checkpoint_load_seconds"]
                                          + timings["decode_overlay_load_seconds"])):
        raise ValueError("delivery separate phase timing differs")
    expected = [(f"layer{layer}/{name}", name) for layer in full_index_layers for name in wk.ROLES]
    if len(expected) != 42:
        raise ValueError("delivery needs the full21-layer WK schedule")
    calls = load_calls(root / f"delivery_wk.rank{rank}", wk_record, expected_calls=42)
    validate_call_sequence(wk_record, calls, expected=expected, local_slots=slots,
                           names=wk.ROLES, budgeter=wk.memory_budget)
    outputs = []
    for call in calls:
        rows = call["output"]
        dtype, size = ("bfloat16", 128 * 6144 * 2) if call["graph"] == "wk_decode" else ("float32", 128 * 6144 * 4)
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("delivery WK output owner count differs")
        ids = set()
        for row in rows:
            device = row["device_id"]
            if (type(device) is not int or device not in slots or device in ids
                    or type(row["slot"]) is not int or row["slot"] != slots[device]
                    or row["shape"] != [128, 6144] or row["dtype"] != dtype
                    or type(row["bytes"]) is not int or row["bytes"] != size
                    or row["finite"] is not True or not isinstance(row["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
                raise ValueError("delivery WK output schema/owner/health differs")
            ids.add(device)
        if len({r["sha256"] for r in rows}) != 1:
            raise ValueError("delivery WK local replicas disagree")
        outputs.append(rows[0]["sha256"])
    _advance(_boundary(calls[-1]["post_memory"], slots, process, nested=False),
             _boundary(parent["prefill_execution"]["memory_admission"]["census"]["devices"], slots, process, nested=True))
    before_overlay, after_overlay = overlay_replay(decode_record, parent, slots)
    _advance(_boundary(parent["batched_prefill_memory"], slots, process, nested=False), before_overlay)
    decode_calls = decode_record["call_evidence"]
    if not isinstance(decode_calls, list) or len(decode_calls) != 2:
        raise ValueError("delivery exact preparation call count differs")
    previous = after_overlay
    for index, call in enumerate(decode_calls):
        name = decode.ROLES[index]
        validate_call_sequence(decode_record, [call], expected=[(f"delivery_decode/{name}", name)],
            local_slots=slots, names=decode.ROLES[:index + 1], budgeter=decode.memory_budget)
        _advance(previous, _boundary(call["census"]["devices"], slots, process, nested=True))
        previous = _boundary(call["post_memory"], slots, process, nested=False)
        # Shape evidence only; do not pretend this is a retained-value replay.
        if not isinstance(call.get("output_schema"), list) or not call["output_schema"]:
            raise ValueError("delivery missing exact output metadata")
    _advance(previous, _boundary(parent["batched_device_memory_after_execute"], slots, process, nested=False))
    return dict(rank=rank, call_count=44, wk_output_hashes=outputs, numerical_value_replay=False)


def validate_fleet(*, root: Path, records: Sequence[Mapping], full_index_layers: Sequence[int]) -> dict:
    """Parent has already joined all32 owners to authenticated topology."""
    if len(records) != 8:
        raise ValueError("delivery preparation needs eight ranks")
    result = [validate_rank(root=root, rank=rank, parent=record, full_index_layers=full_index_layers)
              for rank, record in enumerate(records)]
    if any(row["wk_output_hashes"] != result[0]["wk_output_hashes"] for row in result[1:]):
        raise ValueError("delivery WK cross-host replicas disagree")
    return dict(schema="ws32_delivery_phase_fleet_v1", owner_count=32, calls_per_rank=44,
                original_phase_records_replayed=True, numerical_value_replay=False)
