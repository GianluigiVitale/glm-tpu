"""Bounded18-call continuation using existing voted calls and original capture.

No CLI, compilation, loader or launch authority. The parent supplies four
admitted programs, completed original WKs, and authenticated DB604/605 witnesses.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield import ws32_dense_frontier_worker as base
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_evidence import encode_arrays
from scripts.greenfield.prefill_window_boundary import capture_owner_arrays
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_dense_norm_originals import require_reproduction


def memory_budget(census: Any, analyses: Any, *, active_graph: str) -> dict:
    if set(analyses) != set(protocol.PROGRAMS) or active_graph not in protocol.PROGRAMS:
        raise ValueError("norm diagnostic requires four resident programs")
    return budget_resident_execution(
        census,
        analyses,
        active_graph=active_graph,
        resident_graphs=protocol.PROGRAMS,
        required_reserve_bytes=protocol.RESERVE,
    )


def capture_packet(
    packet: Mapping[str, Any], *, slots: Mapping[int, int], count: int
) -> tuple[dict, dict]:
    """Addressable-only capture; save bad finite/live-mask evidence before refusal."""
    if type(count) is not int or count not in (32, 128) or len(slots) != 4:
        raise ValueError("norm packet count/owner inventory differs")
    fields = {
        **{
            f"post_norm/{name}": ((128, 1536), "bfloat16")
            for name in ("update", "residual", "normalized", "carried")
        },
        "post_norm/summed": ((128, 1536), "float32"),
        **{
            f"post_norm/{name}": ((128, 1), "float32")
            for name in ("local_square_sum", "square_sum", "inverse")
        },
        "boundary/normalized_mlp": ((128, 1536), "bfloat16"),
        "boundary/live": ((128,), "bool"),
        "post_norm/weight": ((1536,), "bfloat16"),
    }
    if set(packet) != set(fields) or any(
        tuple(packet[k].shape) != (8, 4, *shape) or str(packet[k].dtype) != dtype
        for k, (shape, dtype) in fields.items()
    ):
        raise ValueError("norm packet output schema differs")
    by_device = capture_owner_arrays(packet, slots_by_device=slots)
    arrays, errors = {}, []
    for device, values in by_device.items():
        kept = {}
        for name, value in values.items():
            if name == "boundary/live":
                if not np.array_equal(value, np.arange(128) < count):
                    errors.append(f"slot{slots[device]}/live")
                kept[name.replace("/", "_")] = value
            else:
                live_value = value if name == "post_norm/weight" else value[:count]
                if not np.isfinite(live_value).all():
                    errors.append(f"slot{slots[device]}/{name}/nonfinite")
                if name == "boundary/normalized_mlp" and np.count_nonzero(
                    value[count:]
                ):
                    errors.append(f"slot{slots[device]}/normalized_padding")
                kept[name.replace("/", "_")] = live_value
        arrays.update(encode_arrays(f"slot{slots[device]}", kept))
    return arrays, dict(
        valid=not errors,
        errors=errors,
        count=count,
        fields_per_owner=11,
        owner_slots=dict(slots),
    )


def capture_suffix(
    result: tuple, *, slots: Mapping[int, int], count: int
) -> tuple[dict, dict]:
    """Keep local output live rows and full health; check padding before trimming."""
    if (
        type(count) is not int
        or count not in (32, 128)
        or len(result) != 2
        or len(slots) != 4
    ):
        raise ValueError("norm suffix output scope differs")
    output, health = result
    if (
        output.shape != (8, 4, 128, 1536)
        or str(output.dtype) != "bfloat16"
        or health.shape != (8, 4, 128)
        or str(health.dtype) != "bool"
    ):
        raise ValueError("norm suffix output schema differs")
    owners = capture_owner_arrays(
        dict(output=output, health=health), slots_by_device=slots
    )
    arrays, errors = {}, []
    for device, values in owners.items():
        if (
            not values["health"].all()
            or not np.isfinite(values["output"][:count]).all()
            or np.count_nonzero(values["output"][count:])
        ):
            errors.append(f"slot{slots[device]}/suffix_health_or_padding")
        arrays.update(
            encode_arrays(
                f"slot{slots[device]}",
                dict(output=values["output"][:count], health=values["health"]),
            )
        )
    return arrays, dict(
        valid=not errors, errors=errors, count=count, owner_slots=dict(slots)
    )


def suffix_inputs(mesh: Any, packet: Any, start: int, count: int, dense: Any) -> tuple:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    scalars = tuple(
        jax.make_array_from_callback(
            (), NamedSharding(mesh, P()), lambda index, v=v: np.asarray(v, np.int32)
        )
        for v in (start, count)
    )
    return packet["boundary/normalized_mlp"], *scalars, dense


def execute_after_wk(
    calls: BudgetedCalls,
    *,
    mesh: Any,
    config: Any,
    prompt_tokens: np.ndarray,
    embedding: Any,
    layers: Any,
    wk: Any,
    rope: Any,
    witness: Mapping,
    originals: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Capture5 → retained-byte fleet gate → own suffix5 → fleet gate → cross4.

    Original failures stop escalation with saved evidence; cross-placement
    differences are recorded, not treated as permission to change arithmetic.
    """

    def preflight():
        base.require_config(config)
        if (
            calls.budgeter is not memory_budget
            or set(calls.programs) != set(protocol.PROGRAMS)
            or calls.record.get("protocol") != protocol.PROTOCOL
            or [(v["phase"], v["graph"]) for v in calls.record["call_evidence"]]
            != list(protocol.CALLS[:4])
            or not all(v["completed"] for v in calls.record["call_evidence"])
            or prompt_tokens.shape != (8155,)
            or prompt_tokens.dtype != np.int32
            or sha256(prompt_tokens.tobytes()).hexdigest() != base.PROMPT_SHA
            or set(originals) != {name for name, _, _ in protocol.CAPTURES}
        ):
            raise ValueError("norm preflight original/call/prompt inventory differs")

    calls.phase("norm/preflight", preflight)
    records, saved, packets = {}, {}, {}
    used = 0
    calls.record["dense_norm"] = dict(
        complete=False,
        originals=records,
        numerical_promotion=False,
        performance_claim=False,
        cause_claim=False,
    )

    def save(label: str, arrays: dict, report: dict) -> None:
        nonlocal used
        size = sum(v.nbytes for v in arrays.values())
        if (
            used + size + (len(records) + 1) * (1 << 20)
            > protocol.MODEL_ORIGINALS_LIMIT
        ):
            raise ValueError("norm original payload/capsule budget exceeded")
        path = calls.root / f"{label}.npz"
        for original in (
            path,
            path.with_suffix(".npz.pending"),
            path.with_suffix(".json"),
        ):
            if original.exists() or original.is_symlink():
                raise FileExistsError(original)
        report.update(
            npz_sha256=save_arrays(path, arrays),
            npz_bytes=path.stat().st_size,
            raw_array_bytes=size,
        )
        records[label] = report
        used += size
        _atomic_json(calls.root / f"{label}.json", report)
        saved[label] = arrays

    def preserve_capture(label: str, count: int, result: tuple) -> None:
        out, packet = result
        endpoint = label in ("wide_final", "narrow_128")
        arrays, report = base.capture(
            out,
            local_slots=calls.local_slots,
            process_index=calls.record["jax_process_index"],
            count=count,
            keep_caches=endpoint,
        )
        save(label, arrays, report)
        values, packet_report = capture_packet(
            packet, slots=calls.local_slots, count=count
        )
        save(label + "_norm", values, packet_report)
        if not report["valid"] or not packet_report["valid"]:
            raise ValueError("norm capture unhealthy; original bytes preserved")

    wide = calls.phase("norm/wide_initial", lambda: base.fresh_caches(mesh))
    narrow = calls.phase("norm/narrow_initial", lambda: base.fresh_caches(mesh))
    calls.phase("norm/independent", lambda: base.independent_caches(wide, narrow))
    for label, count, offset in protocol.CAPTURES:
        caches = wide if label == "wide_final" else narrow
        values = calls.phase(
            f"norm/inputs/{label}",
            lambda: base.inputs(
                mesh,
                prompt_tokens[offset : offset + count],
                offset,
                caches,
                embedding,
                layers,
                wk,
                rope,
            ),
        )
        result = calls.call(
            f"norm/capture/{label}",
            "dense01_norm",
            values,
            preserve=lambda r: preserve_capture(label, count, r),
        )
        out, packets[label] = result
        if label != "wide_final":
            narrow = tuple(tuple(layer[2:5]) for layer in out)
        del out, result, values

    def reproduce():
        reports = {
            label: require_reproduction(saved[label], originals[label])
            for label, _, _ in protocol.CAPTURES
        }
        cache_reports = [
            base.compare_owner(
                witness,
                branch=branch,
                slot=slot,
                caches=base.cache_bits(saved[branch], slot),
            )
            for branch in ("wide_final", "narrow_128")
            for slot in calls.local_slots.values()
        ]
        value = dict(
            retained_fields=reports,
            caches=cache_reports,
            reproduced=all(r["reproduced"] for r in cache_reports),
        )
        _atomic_json(calls.root / "reproduction.json", value)
        calls.record["dense_norm"]["reproduction"] = value
        if not value["reproduced"]:
            raise ValueError("norm capture does not reproduce DB604 endpoint caches")

    calls.phase("norm/reproduction", reproduce)

    # All live packets remain reachable and enter BudgetedCalls' all-live census.
    def preserve_suffix(label: str, count: int, result: tuple) -> None:
        arrays, report = capture_suffix(result, slots=calls.local_slots, count=count)
        save(label, arrays, report)
        if not report["valid"]:
            raise ValueError("norm suffix unhealthy; original bytes preserved")

    for label, count, _ in protocol.CAPTURES:
        values = calls.phase(
            f"norm/own_inputs/{label}",
            lambda: suffix_inputs(mesh, packets[label], 0, count, layers[0].dense),
        )
        result = calls.call(
            f"norm/own/{label}",
            "dense_suffix",
            values,
            preserve=lambda r: preserve_suffix("own_" + label, count, r),
        )
        del result, values

    def own_reproduction():
        reports = {}
        for label, _, _ in protocol.CAPTURES:
            expected = {
                f"slot{slot}__{field}": saved[label][f"slot{slot}_layer0__{field}"]
                for slot in calls.local_slots.values()
                for field in ("output", "health")
            }
            reports[label] = {
                **require_reproduction(saved["own_" + label], expected),
                "scope": "OWN_COMPLETED_SUFFIX_OUTPUT_AND_FULL_HEALTH",
            }
        calls.record["dense_norm"]["own_suffix_reproduction"] = reports
        _atomic_json(calls.root / "own_reproduction.json", reports)

    calls.phase("norm/own_reproduction", own_reproduction)

    for offset in (0, 32, 64, 96):
        values = calls.phase(
            f"norm/cross_inputs/{offset}",
            lambda: suffix_inputs(
                mesh, packets["wide_final"], offset, 32, layers[0].dense
            ),
        )
        result = calls.call(
            f"norm/cross/{offset}",
            "dense_suffix",
            values,
            preserve=lambda r: preserve_suffix(f"cross_{offset}", 32, r),
        )
        del result, values

    def finish():
        differences = []
        for start in (0, 32, 64, 96):
            for slot in calls.local_slots.values():
                a = saved[f"cross_{start}"][f"slot{slot}__output"]
                b = saved["own_wide_final"][f"slot{slot}__output"][start : start + 32]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError("norm cross-placement result geometry differs")
                differences.append(
                    dict(
                        slot=slot,
                        start=start,
                        differing_words=int(np.count_nonzero(a != b)),
                        bytes_equal=a.tobytes() == b.tobytes(),
                    )
                )
        if [(r["phase"], r["graph"]) for r in calls.record["call_evidence"]] != list(
            protocol.CALLS
        ) or not all(r["completed"] for r in calls.record["call_evidence"]):
            raise ValueError("norm final18call sequence differs")
        report = dict(
            comparisons=differences,
            numerical_promotion=False,
            cause_claim=False,
            performance_claim=False,
            scope="COMPLETED_SUFFIX_IDENTICAL_ROW_INPUT_PLACEMENT_ONLY",
        )
        _atomic_json(calls.root / "cross_comparison.json", report)
        calls.record["dense_norm"].update(
            complete=True, cross_comparison=report, raw_array_bytes=used
        )

    calls.phase("norm/comparison", finish)
