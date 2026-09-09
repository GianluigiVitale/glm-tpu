"""Independent original-array replay of fixed norm capture and suffix diagnosis.

Reuse the existing five-model-capsule schema/cache reader. No worker pass flag
substitutes for DB605 byte equality, checkpoint-weight or own-suffix reproduction.
This is diagnostic evidence, never an8K fix, launch authority or performance pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_dense_frontier_evidence as base
from scripts.greenfield import ws32_dense_norm_originals as originals_module
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield.prefill_window_evidence import same_json

NORM_WEIGHT = "model.layers.0.post_attention_layernorm.weight"
CAPTURE_NAMES = tuple(n for n, _, _ in protocol.CAPTURES)
ALL_NAMES = (
    *CAPTURE_NAMES,
    *(n + "_norm" for n in CAPTURE_NAMES),
    *("own_" + n for n in CAPTURE_NAMES),
    *(f"cross_{s}" for s in (0, 32, 64, 96)),
)


def canonical_reproduction(value: Mapping) -> dict:
    """Join complete owner records only; never reorder any numerical array."""
    owners = base.comparison_by_owner(dict(owners=value["caches"]))["owners"]
    return {**value, "caches": owners}


def canonical_cross(value: Mapping) -> dict:
    rows = value.get("comparisons")
    if not isinstance(rows, list) or any(
        not isinstance(r, dict)
        or type(r.get("slot")) is not int
        or not 0 <= r["slot"] < 32
        or type(r.get("start")) is not int
        or r["start"] not in (0, 32, 64, 96)
        for r in rows
    ):
        raise ValueError("norm cross comparison owner keys differ")
    keys = [(r["start"], r["slot"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("norm duplicate cross comparison owner")
    return {**value, "comparisons": sorted(rows, key=lambda r: (r["start"], r["slot"]))}


def expected_stages() -> list[str]:
    result = ["identity"] + [
        stage
        for _ in protocol.PROGRAMS
        for stage in ("lower_compile_started", "compiled", "raw_written", "inspected")
    ]
    result.append("dense/admission")
    for phase, _ in protocol.CALLS[:4]:
        result.extend(phase + "/" + s for s in ("memory", "execute", "memory_after"))
    result.extend(
        (
            "norm/preflight",
            "norm/wide_initial",
            "norm/narrow_initial",
            "norm/independent",
        )
    )
    for phase, inputs, labels in (
        ("capture", "inputs", CAPTURE_NAMES),
        ("own", "own_inputs", CAPTURE_NAMES),
        ("cross", "cross_inputs", ("0", "32", "64", "96")),
    ):
        for label in labels:
            result.append(f"norm/{inputs}/{label}")
            result.extend(
                f"norm/{phase}/{label}/{s}"
                for s in ("memory", "execute", "memory_after")
            )
        result.append(
            "norm/"
            + {
                "capture": "reproduction",
                "own": "own_reproduction",
                "cross": "comparison",
            }[phase]
        )
    return result


def _packet_schema(count: int) -> dict:
    return {
        **{
            f"post_norm_{n}": ((count, 1536), np.uint16)
            for n in ("update", "residual", "normalized", "carried")
        },
        "post_norm_summed": ((count, 1536), np.float32),
        **{
            f"post_norm_{n}": ((count, 1), np.float32)
            for n in ("local_square_sum", "square_sum", "inverse")
        },
        "boundary_normalized_mlp": ((count, 1536), np.uint16),
        "boundary_live": ((128,), np.bool_),
        "post_norm_weight": ((1536,), np.uint16),
    }


def read_packet(
    root: Path,
    label: str,
    report: Mapping,
    slots: Mapping[int, int],
    count: int,
    *,
    packet: bool,
    bindings: Mapping,
) -> dict:
    fields = (
        _packet_schema(count)
        if packet
        else {"output": ((count, 1536), np.uint16), "health": ((128,), np.bool_)}
    )
    metadata = dict(
        valid=True,
        errors=[],
        count=count,
        owner_slots={str(d): s for d, s in slots.items()},
    )
    if packet:
        metadata["fields_per_owner"] = 11
    same_json({k: report.get(k) for k in metadata}, metadata, "norm capsule schema")
    same_json(
        json.loads((root / f"{label}.json").read_bytes()),
        report,
        "norm capsule sidecar",
    )
    arrays = base.read_npz(
        root / f"{label}.npz",
        report,
        limit=protocol.MODEL_ORIGINALS_LIMIT,
        size_key="npz_bytes",
    )
    expected = {
        f"slot{s}__{f}": (shape, dtype)
        for s in slots.values()
        for f, (shape, dtype) in fields.items()
    }
    if set(arrays) != set(expected):
        raise ValueError("norm array owner/field inventory differs")
    for key, (shape, dtype) in expected.items():
        a = arrays[key]
        if a.shape != shape or a.dtype != dtype:
            raise ValueError("norm array shape/dtype differs")
        decoded = a.view(ml_dtypes.bfloat16) if dtype == np.uint16 else a
        if not np.isfinite(decoded).all():
            raise ValueError("norm original contains nonfinite values")
    for slot in slots.values():
        if packet:
            if not np.array_equal(
                arrays[f"slot{slot}__boundary_live"], np.arange(128) < count
            ):
                raise ValueError("norm original live mask differs")
            if (
                base.digest(arrays[f"slot{slot}__post_norm_weight"])
                != bindings[slot]["selected"][NORM_WEIGHT]
            ):
                raise ValueError("norm observed weight differs from checkpoint")
        elif not arrays[f"slot{slot}__health"].all():
            raise ValueError("norm suffix full health differs")
    return arrays


def replay_outputs(
    root: Path,
    record: Mapping[str, Any],
    slots: Mapping[int, int],
    witness: Mapping,
    originals: Mapping,
    bindings: Mapping,
) -> tuple[dict, dict]:
    if (
        len(slots) != 4
        or len(set(slots.values())) != 4
        or any(
            type(d) is not int or type(s) is not int or not 0 <= s < 32
            for d, s in slots.items()
        )
    ):
        raise ValueError("norm replay requires four unique physical owners")
    original = record["dense_norm"]
    fixed = dict(
        complete=True,
        numerical_promotion=False,
        performance_claim=False,
        cause_claim=False,
    )
    same_json(
        {k: original.get(k) for k in fixed}, fixed, "norm completed diagnostic scope"
    )
    reports = original["originals"]
    if set(reports) != set(ALL_NAMES) or set(originals) != set(CAPTURE_NAMES):
        raise ValueError("norm19-capsule/reference inventory differs")
    raw_bytes = sum(r["raw_array_bytes"] for r in reports.values())
    if (
        any(
            type(r["raw_array_bytes"]) is not int or r["raw_array_bytes"] <= 0
            for r in reports.values()
        )
        or raw_bytes + len(reports) * (1 << 20) > protocol.MODEL_ORIGINALS_LIMIT
        or type(original.get("raw_array_bytes")) is not int
        or raw_bytes != original["raw_array_bytes"]
    ):
        raise ValueError("norm cumulative originals budget differs")
    owners, fingerprints, saved = base.read_model_capsules(
        root, {n: reports[n] for n in CAPTURE_NAMES}, slots, witness
    )
    reproduction = dict(
        retained_fields={
            n: originals_module.require_reproduction(saved[n], originals[n])
            for n in CAPTURE_NAMES
        },
        caches=owners,
        reproduced=all(o["reproduced"] for o in owners),
    )
    same_json(
        canonical_reproduction(original["reproduction"]),
        canonical_reproduction(reproduction),
        "norm original DB604/605 reproduction",
    )
    same_json(
        canonical_reproduction(json.loads((root / "reproduction.json").read_bytes())),
        canonical_reproduction(reproduction),
        "norm reproduction file",
    )
    if not reproduction["reproduced"]:
        raise ValueError("norm originals do not reproduce DB604")

    packets, suffixes, own = {}, {}, {}
    for label, count, _ in protocol.CAPTURES:
        packets[label] = read_packet(
            root,
            label + "_norm",
            reports[label + "_norm"],
            slots,
            count,
            packet=True,
            bindings=bindings,
        )
        suffixes[label] = read_packet(
            root,
            "own_" + label,
            reports["own_" + label],
            slots,
            count,
            packet=False,
            bindings=bindings,
        )
        expected = {
            f"slot{s}__{f}": saved[label][f"slot{s}_layer0__{f}"]
            for s in slots.values()
            for f in ("output", "health")
        }
        own[label] = {
            **originals_module.require_reproduction(suffixes[label], expected),
            "scope": "OWN_COMPLETED_SUFFIX_OUTPUT_AND_FULL_HEALTH",
        }
    same_json(original["own_suffix_reproduction"], own, "norm own suffix reproduction")
    same_json(
        json.loads((root / "own_reproduction.json").read_bytes()),
        own,
        "norm own reproduction file",
    )
    differences, boundaries = [], []
    for start in (0, 32, 64, 96):
        label = f"cross_{start}"
        cross = read_packet(
            root, label, reports[label], slots, 32, packet=False, bindings=bindings
        )
        narrow = packets[f"narrow_{start+32}"]
        for slot in slots.values():
            a = cross[f"slot{slot}__output"]
            b = suffixes["wide_final"][f"slot{slot}__output"][start : start + 32]
            differences.append(
                dict(
                    slot=slot,
                    start=start,
                    differing_words=int(np.count_nonzero(a != b)),
                    bytes_equal=a.tobytes() == b.tobytes(),
                )
            )
            fields = {}
            for field in _packet_schema(32):
                if field in ("post_norm_weight", "boundary_live"):
                    continue
                key = f"slot{slot}__{field}"
                wide = packets["wide_final"][key][start : start + 32]
                small = narrow[key]
                word = np.uint32 if wide.dtype == np.float32 else np.uint16
                fields[field] = dict(
                    differing_words=int(
                        np.count_nonzero(wide.view(word) != small.view(word))
                    ),
                    wide_sha256=base.digest(wide),
                    narrow_sha256=base.digest(small),
                )
            boundaries.append(dict(slot=slot, start=start, fields=fields))
    cross_report = dict(
        comparisons=differences,
        numerical_promotion=False,
        cause_claim=False,
        performance_claim=False,
        scope="COMPLETED_SUFFIX_IDENTICAL_ROW_INPUT_PLACEMENT_ONLY",
    )
    same_json(
        canonical_cross(original["cross_comparison"]),
        canonical_cross(cross_report),
        "norm cross-placement original replay",
    )
    same_json(
        canonical_cross(json.loads((root / "cross_comparison.json").read_bytes())),
        canonical_cross(cross_report),
        "norm cross comparison file",
    )
    return (
        dict(
            owners=owners,
            reproduced=True,
            retained_fields=reproduction["retained_fields"],
            own_suffix_reproduction=own,
            cross_comparison=cross_report,
            boundaries=boundaries,
            raw_array_bytes=raw_bytes,
            numerical_promotion=False,
            performance_claim=False,
            cause_claim=False,
            scope="NORM_ORIGINALS_AND_REPRODUCED_SUFFIX_NOT_8K_CORRECTNESS",
            padding_scope="FULL_LIVE_MASK_AND_HEALTH_REPLAYED_TRIMMED_NUMERICAL_PADDING_WORKER_CHECK_ONLY",
        ),
        fingerprints,
    )
