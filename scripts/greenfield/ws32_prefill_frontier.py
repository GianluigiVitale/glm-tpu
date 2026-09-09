"""Compact original cache bytes for the fixed first128-row discriminator.

Pure host evidence only: no JAX, model execution, launch or promotion. The outer
worker must bind geometry/page table/slot to the actual frozen runtime and retain
both branches before fleet finalization. Equality here is not model correctness.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import ml_dtypes
import numpy as np

FRONTIER = 128
BF16 = np.dtype(ml_dtypes.bfloat16)
_FIELDS = {
    "schema_version", "slot", "frontier", "shape", "dtype", "layer_ids",
    "block_table", "positions", "physical_pages", "local_rows",
    "rows_sha256", "whole_cache_sha256",
}


def _digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _layout(
    shape: tuple[int, ...], slot: int, table: np.ndarray, layer_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        len(shape) != 4
        or any(type(n) is not int for n in shape)
        or not 1 <= shape[0] <= 78
        or not 1 <= shape[1] <= 16
        or shape[2] != 64
        or shape[3] not in (128, 640)
        or type(slot) is not int or not 0 <= slot < 32
        or table.dtype != np.int32 or table.shape != (1, shape[1])
        or sorted(table[0].tolist()) != list(range(shape[1]))
        or len(layer_ids) != shape[0]
        or any(type(n) is not int or not 0 <= n < 78 for n in layer_ids)
        or tuple(sorted(set(layer_ids))) != layer_ids
    ):
        raise ValueError("first-window owner cache geometry/page table differs")
    # Adapt the independent host page512/stripe64 arithmetic; never infer an
    # expert from device/rank IDs. The caller supplies authenticated mesh slot.
    positions = np.arange(FRONTIER, dtype=np.int32)
    positions = positions[(positions % 512) // 64 == slot // 4]
    pages = table[0, positions // 512]
    rows = positions % 64
    return positions, pages, rows


def capture_cache(
    value: np.ndarray, *, slot: int, block_table: np.ndarray,
    layer_ids: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain written BF16 bits and prove every other cache byte is zero.

    Input is an already addressable owner-local cache, NOT a distributed array.
    Returns uint16 storage for portable NPZ, preserving signed zero exactly.
    Temporary reconstruction is bounded by one local cache, never full weights.
    """
    stored, record = _original_cache(value, slot, block_table, layer_ids)
    replay_cache(stored, record)
    return stored, record


def _original_cache(value, slot, block_table, layer_ids):
    if not isinstance(value, np.ndarray) or value.dtype != BF16:
        raise ValueError("first-window capture requires host BF16 cache")
    shape = tuple(value.shape)
    positions, pages, rows = _layout(shape, slot, block_table, layer_ids)
    stored = np.ascontiguousarray(value[:, pages, rows, :]).view(np.uint16).copy()
    record = dict(
        schema_version=1, slot=slot, frontier=FRONTIER,
        shape=list(shape), dtype="bfloat16", layer_ids=list(layer_ids),
        block_table=block_table.tolist(), positions=positions.tolist(),
        physical_pages=pages.tolist(), local_rows=rows.tolist(),
        rows_sha256=_digest(stored), whole_cache_sha256=_digest(value),
    )
    return stored, record


def capture_cache_evidence(
    value: np.ndarray, *, slot: int, block_table: np.ndarray,
    layer_ids: tuple[int, ...], initial: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve bounded original evidence even when normal replay would refuse.

    Fixed first128 or zero initial state only. Keep all first128 row bits, whole
    cache SHA and counts/first32 offending coordinates/bits. Scan one layer at a
    time, never allocate a full-sized nonfinite/coordinate tensor. This diagnostic
    is NOT a successful replay capsule when violations exist; callers must save
    it before refusing. Malformed dtype/layout still raises before interpretation.
    """
    if type(initial) is not bool:
        raise ValueError("initial cache flag must be boolean")
    rows, record = _original_cache(value, slot, block_table, layer_ids)
    allowed = np.zeros(value.shape[1:3], dtype=np.bool_)
    if not initial:
        allowed[record["physical_pages"], record["local_rows"]] = True
    totals = dict(nonfinite=0, outside_nonzero=0)
    samples = []
    offending_elements = 0
    for index, layer in enumerate(layer_ids):
        bits = np.ascontiguousarray(value[index]).view(np.uint16)
        bad_finite = (bits & 0x7F80) == 0x7F80
        bad_outside = (bits != 0) & ~allowed[:, :, None]
        totals["nonfinite"] += int(np.count_nonzero(bad_finite))
        totals["outside_nonzero"] += int(np.count_nonzero(bad_outside))
        offending_elements += int(np.count_nonzero(bad_finite | bad_outside))
        if len(samples) < 32:
            # flatnonzero on one local layer is bounded (at most5MiB indices).
            indices = np.flatnonzero(bad_finite | bad_outside)[:32-len(samples)]
            for flat in indices:
                page, row, component = (int(n) for n in np.unravel_index(flat, bits.shape))
                samples.append(dict(
                    layer_id=layer, physical_page=page, local_row=row,
                    component=component, bits=int(bits[page, row, component]),
                    nonfinite=bool(bad_finite[page, row, component]),
                    outside_nonzero=bool(bad_outside[page, row, component]),
                ))
    valid = not any(totals.values())
    if valid:
        replay_cache(rows, record)
    return rows, dict(
        cache=record, initial=initial, valid=valid, violations=totals,
        samples=samples, samples_truncated=offending_elements > len(samples),
        numerical_promotion=False,
    )


def replay_cache(rows: np.ndarray, record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the whole-cache digest from original written rows plus zeros.

    The digest is evidence of the observed bytes, not proof the worker is honest;
    outer original-file/owner/source bindings remain mandatory. Shape limits do
    not authorize a nonproduction shape in a protected workload.
    """
    if (
        set(record) != _FIELDS
        or type(record["schema_version"]) is not int or record["schema_version"] != 1
        or type(record["frontier"]) is not int or record["frontier"] != FRONTIER
        or record["dtype"] != "bfloat16"
        or not isinstance(record["shape"], list)
        or not isinstance(record["layer_ids"], list)
        or not isinstance(record["block_table"], list)
    ):
        raise ValueError("first-window cache capsule schema differs")
    raw_table = record["block_table"]
    if (len(raw_table) != 1 or not isinstance(raw_table[0], list)
            or any(type(n) is not int for n in raw_table[0])):
        raise ValueError("first-window page table must contain strict integers")
    # Reject overflowing integers before numpy could wrap/truncate them.
    if any(not 0 <= n < 16 for n in raw_table[0]):
        raise ValueError("first-window page table exceeds bounded capacity")
    shape = tuple(record["shape"])
    positions, pages, local_rows = _layout(
        shape, record["slot"], np.asarray(raw_table, np.int32),
        tuple(record["layer_ids"]),
    )
    for field, wanted in (
        ("positions", positions), ("physical_pages", pages), ("local_rows", local_rows)
    ):
        actual = record[field]
        if (not isinstance(actual, list) or any(type(n) is not int for n in actual)
                or actual != wanted.tolist()):
            raise ValueError("first-window recorded addresses differ")
    if (not isinstance(rows, np.ndarray) or rows.dtype != np.uint16
            or rows.shape != (shape[0], len(positions), shape[3])
            or _digest(rows) != record["rows_sha256"]):
        raise ValueError("first-window original row bits differ")
    if not np.isfinite(np.ascontiguousarray(rows).view(BF16)).all():
        raise ValueError("first-window cache contains nonfinite values")
    rebuilt = np.zeros(shape, np.uint16)
    rebuilt[:, pages, local_rows, :] = rows
    digest = _digest(rebuilt)
    if digest != record["whole_cache_sha256"]:
        raise ValueError("first-window cache has uncaptured/changed bytes outside rows")
    return dict(
        whole_cache_sha256=digest, untouched_bytes_zero=True,
        finite=True, stored_bytes=int(rows.nbytes), owner_positions=positions.tolist(),
    )


def compare_cache(
    wide_rows: np.ndarray, wide: Mapping[str, Any],
    narrow_rows: np.ndarray, narrow: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact same-owner byte comparison; differences are diagnostic, not failure."""
    replay_cache(wide_rows, wide)
    replay_cache(narrow_rows, narrow)
    identity = _FIELDS - {"rows_sha256", "whole_cache_sha256"}
    if any(wide[k] != narrow[k] for k in identity):
        raise ValueError("first-window comparison mixes owners/layouts/frontiers")
    changed = wide_rows != narrow_rows
    writers = []
    for index, layer in enumerate(wide["layer_ids"]):
        where = np.argwhere(changed[index])
        if not len(where):
            continue
        row, component = (int(n) for n in where[0])
        delta = (
            np.ascontiguousarray(wide_rows[index]).view(BF16).astype(np.float64)
            - np.ascontiguousarray(narrow_rows[index]).view(BF16).astype(np.float64)
        )
        writers.append(dict(
            layer_id=layer, differing_elements=int(len(where)),
            first_position=wide["positions"][row], first_component=component,
            first_wide_bits=int(wide_rows[index, row, component]),
            first_narrow_bits=int(narrow_rows[index, row, component]),
            max_abs=float(np.max(np.abs(delta))),
        ))
    return dict(
        bytes_equal=not writers, differing_writers=writers,
        earliest_differing_writer=None if not writers else writers[0]["layer_id"],
        scope="FIRST128_CACHE_WRITERS_NOT_MODEL_CORRECTNESS_OR_ROOT_CAUSE",
        performance_claim=False, numerical_promotion=False,
    )
