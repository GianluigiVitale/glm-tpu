"""Bounded cache proposals for a whole prefill window, not a second cache.

Adapt the existing ``write_prefill_cache_block`` striped addresses. The original
per-layer writer still computes/validates the causal cache used by attention.
Capture only its window rows here; apply them to the owned stack ONLY inside the
caller's all-owner healthy commit. Never infer runtime HBM savings from this API.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .reference.attention import StageLocalKvLayout


class PrefillPendingAddresses(NamedTuple):
    targets: Any
    valid: Any


def prefill_pending_addresses(
    block_table: Any,
    position_offset: Any,
    valid_rows: Any,
    owner_index: Any,
    *,
    physical_pages: int,
    window_rows: int,
    layout: StageLocalKvLayout,
) -> PrefillPendingAddresses:
    """One physical row index per window row; unowned/padded indices DROP.

    Page512/stripe8 means 64 CONSECUTIVE positions per owner, not round-robin
    tokens. Retain a nonnegative out-of-bounds sentinel, never clipped duplicate
    writes to real rows. Malformed spans/pages/owners yield only sentinels.
    """
    if (
        type(physical_pages) is not int
        or physical_pages <= 0
        or type(window_rows) is not int
        or not 1 <= window_rows <= 128
    ):
        raise ValueError("pending rows require positive pages and1..128 rows")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] <= 0
        or block_table.dtype != jnp.int32
    ):
        raise ValueError("pending rows require one nonempty int32 page table")
    for value in (position_offset, valid_rows, owner_index):
        if value.shape != () or value.dtype != jnp.int32:
            raise ValueError("pending offset/count/owner must be int32 scalars")
    capacity = block_table.shape[1] * layout.logical_page_size
    flat_count = physical_pages * layout.local_rows_per_page
    if max(capacity, flat_count) >= 2147483647:
        raise ValueError("pending cache addresses must fit positive int32")
    count = jnp.clip(valid_rows, 0, window_rows)
    start = jnp.clip(position_offset, 0, capacity)
    safe_count = jnp.minimum(count, capacity - start)
    end = start + safe_count
    required = end // layout.logical_page_size + (
        end % layout.logical_page_size != 0
    ).astype(jnp.int32)
    page_live = jnp.arange(block_table.shape[1], dtype=jnp.int32) < required
    page_ids = block_table[0]
    ordered = jnp.sort(jnp.where(page_live, page_ids, jnp.iinfo(jnp.int32).max))
    valid = (
        (valid_rows >= 0)
        & (valid_rows <= window_rows)
        & (position_offset >= 0)
        & (position_offset <= capacity - count)
        & (owner_index >= 0)
        & (owner_index < layout.local_parallel_size)
        & jnp.all(~page_live | ((page_ids >= 0) & (page_ids < physical_pages)))
        & jnp.all(
            (jnp.arange(1, block_table.shape[1]) >= required)
            | (ordered[1:] != ordered[:-1])
        )
    )
    row = jnp.arange(window_rows, dtype=jnp.int32)
    delta = jnp.minimum(row, jnp.maximum(safe_count - 1, 0))
    positions = jnp.minimum(start, capacity - 1) + delta
    pages = page_ids[positions // layout.logical_page_size]
    flat_index = (
        jnp.clip(pages, 0, physical_pages - 1) * layout.local_rows_per_page
        + positions % layout.local_rows_per_page
    )
    owned = (row < count) & (layout.owner(positions) == owner_index) & valid
    return PrefillPendingAddresses(jnp.where(owned, flat_index, flat_count), valid)


def capture_prefill_pending_rows(cache_local: Any, targets: Any) -> Any:
    """Extract only current-window rows from one layer's completed proposal."""
    if (
        cache_local.ndim != 3
        or min(cache_local.shape) <= 0
        or cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("pending capture requires a BF16 per-layer cache")
    if (
        targets.ndim != 1
        or not 1 <= targets.shape[0] <= 128
        or targets.dtype != jnp.int32
    ):
        raise ValueError("pending capture requires1..128 int32 targets")
    flat = cache_local.reshape(-1, cache_local.shape[-1])
    live = (targets >= 0) & (targets < flat.shape[0])
    return jnp.where(live[:, None], flat[jnp.clip(targets, 0, flat.shape[0] - 1)], 0)


def apply_prefill_pending_rows(cache_stack: Any, targets: Any, rows: Any) -> Any:
    """Apply all layer/producer rows; caller MUST gate whole-window commit.

    Valid addresses are unique within each layer (live pages are unique); only
    DROP sentinels may repeat. No full proposed stack exists before this call.
    """
    if (
        cache_stack.ndim != 4
        or min(cache_stack.shape) <= 0
        or cache_stack.dtype != jnp.bfloat16
    ):
        raise ValueError("pending commit requires a BF16 stacked cache")
    if (
        targets.ndim != 1
        or not 1 <= targets.shape[0] <= 128
        or targets.dtype != jnp.int32
        or rows.dtype != cache_stack.dtype
        or rows.shape != (cache_stack.shape[0], targets.shape[0], cache_stack.shape[-1])
    ):
        raise ValueError("pending commit rows/targets disagree with cache stack")
    flat = cache_stack.reshape(cache_stack.shape[0], -1, cache_stack.shape[-1])
    # Defensive negative-to-DROP conversion: JAX otherwise wraps negative indices.
    safe = jnp.where((targets >= 0) & (targets < flat.shape[1]), targets, flat.shape[1])
    return flat.at[:, safe, :].set(rows, mode="drop").reshape(cache_stack.shape)
