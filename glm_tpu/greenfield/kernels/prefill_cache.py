"""Unwired block cache updates for one causal WS32 prompt sequence.

Same striped ownership as the decoder; address validation runs once per block,
not once per row. This does not establish populated-prefix integrity or decide
when repaired index keys become visible. Those remain caller state contracts.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax

from .reference.attention import StageLocalKvLayout


class PrefillCacheWrite(NamedTuple):
    cache: Any
    valid: Any
    # Per-query exclusive bounds, never the block-end bound for every query.
    causal_lengths: Any
    row_valid: Any


def write_prefill_cache_block(
    cache_local: Any,
    rows: Any,
    block_table: Any,
    position_offset: Any,
    valid_rows: Any,
    owner_index: Any,
    *,
    layout: StageLocalKvLayout,
) -> PrefillCacheWrite:
    """Write owned live rows, or leave the entire cache unchanged on bad input.

    Inputs use a single shared page table. Empty blocks are healthy no-ops if
    metadata/prefix mapping is valid. Nonfinite padded rows are ignored. Caller
    must bind offset to its committed append frontier and gate all-chip health.
    Apply separately to KV/unrepaired index destinations. Never alias/promote a
    repaired destination into the active prompt index cache before prefill ends.
    """
    if (
        cache_local.ndim != 3
        or min(cache_local.shape) <= 0
        or cache_local.shape[1:]
        != (layout.local_rows_per_page, layout.packed_cache_width)
        or cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill cache must match BF16 striped owner layout")
    if (
        rows.ndim != 2
        or not 1 <= rows.shape[0] <= 32
        or rows.shape[1] != layout.packed_cache_width
        or rows.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill cache block requires1..32 BF16 rows")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] <= 0
        or block_table.dtype != jnp.int32
    ):
        raise ValueError("prefill cache needs one shared int32 page table")
    for value in (position_offset, valid_rows, owner_index):
        if value.shape != () or value.dtype != jnp.int32:
            raise ValueError("prefill offset/count/owner must be int32 scalars")
    capacity = block_table.shape[1] * layout.logical_page_size
    flat_count = cache_local.shape[0] * layout.local_rows_per_page
    if max(capacity, flat_count) >= 2147483647:
        raise ValueError("prefill cache address range must fit positive int32")
    count = jnp.clip(valid_rows, 0, rows.shape[0])
    # Safe end/addition even for adversarial INT_MIN/INT_MAX metadata.
    safe_start = jnp.clip(position_offset, 0, capacity)
    count_ok = (valid_rows >= 0) & (valid_rows <= rows.shape[0])
    span_ok = (position_offset >= 0) & (position_offset <= capacity - count)
    safe_count = jnp.minimum(count, capacity - safe_start)
    end = safe_start + safe_count
    required_pages = end // layout.logical_page_size + (
        end % layout.logical_page_size != 0
    ).astype(jnp.int32)
    page_live = jnp.arange(block_table.shape[1], dtype=jnp.int32) < required_pages
    page_ids = block_table[0]
    pages_ok = jnp.all(
        ~page_live | ((page_ids >= 0) & (page_ids < cache_local.shape[0]))
    )
    ordered = jnp.sort(jnp.where(page_live, page_ids, jnp.iinfo(jnp.int32).max))
    unique = jnp.all(
        jnp.where(
            jnp.arange(1, block_table.shape[1]) < required_pages,
            ordered[1:] != ordered[:-1],
            True,
        )
    )
    live = jnp.arange(rows.shape[0], dtype=jnp.int32) < count
    finite = jnp.all(jnp.isfinite(rows) | ~live[:, None])
    valid = (
        count_ok
        & span_ok
        & pages_ok
        & unique
        & finite
        & (owner_index >= 0)
        & (owner_index < layout.local_parallel_size)
    )
    # Padded rows have a safe address without offset+row overflow.
    delta = jnp.minimum(
        jnp.arange(rows.shape[0], dtype=jnp.int32), jnp.maximum(safe_count - 1, 0)
    )
    positions = jnp.minimum(safe_start, capacity - 1) + delta
    pages = page_ids[positions // layout.logical_page_size]
    local_row = positions % layout.local_rows_per_page
    owned = live & (layout.owner(positions) == owner_index)
    flat_index = (
        jnp.clip(pages, 0, cache_local.shape[0] - 1) * layout.local_rows_per_page
        + local_row
    )
    # Positive sentinel; mode=drop is explicit and masked indices may repeat.
    targets = jnp.where(owned, flat_index, flat_count)
    clean_rows = jnp.where(live[:, None], rows, jnp.zeros((), rows.dtype))

    def write(cache):
        flat = cache.reshape(flat_count, layout.packed_cache_width)
        return flat.at[targets].set(clean_rows, mode="drop").reshape(cache.shape)

    updated = lax.cond(valid, write, lambda cache: cache, cache_local)
    row_valid = live & valid
    causal_lengths = jnp.where(row_valid, positions + 1, 0)
    return PrefillCacheWrite(updated, valid, causal_lengths, row_valid)
