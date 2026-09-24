"""Unwired block cache updates for one causal WS32 prompt sequence.

Same striped ownership as the decoder; address validation runs once per block,
not once per row. This does not establish populated-prefix integrity or decide
when repaired index keys become visible. Those remain caller state contracts.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax
import jax

from glm_tpu.layers.contracts import StageLocalKvLayout, SelectedPositions, _require_int32, _require_shape


_INT32_MAX = jnp.iinfo(jnp.int32).max


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
        or cache_local.shape[1:] != (layout.local_rows_per_page, layout.packed_cache_width)
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
    required_pages = end // layout.logical_page_size + (end % layout.logical_page_size != 0).astype(jnp.int32)
    page_live = jnp.arange(block_table.shape[1], dtype=jnp.int32) < required_pages
    page_ids = block_table[0]
    pages_ok = jnp.all(~page_live | ((page_ids >= 0) & (page_ids < cache_local.shape[0])))
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
    delta = jnp.minimum(jnp.arange(rows.shape[0], dtype=jnp.int32), jnp.maximum(safe_count - 1, 0))
    positions = jnp.minimum(safe_start, capacity - 1) + delta
    pages = page_ids[positions // layout.logical_page_size]
    local_row = positions % layout.local_rows_per_page
    owned = live & (layout.owner(positions) == owner_index)
    flat_index = jnp.clip(pages, 0, cache_local.shape[0] - 1) * layout.local_rows_per_page + local_row
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


def _require_decode_metadata(
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    local_slot: Any,
    *,
    layout: StageLocalKvLayout,
    physical_page_count: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return safe current-row indices plus a device-resident health bit."""

    if position.shape != (1,) or not jnp.issubdtype(position.dtype, jnp.integer):
        raise ValueError("decode position must be one integer row")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1:
        raise ValueError("block tables must contain one decode row")
    if block_tables.shape[1] == 0 or block_tables.dtype != jnp.int32:
        raise ValueError("block tables must expose int32 logical pages")
    if context_lengths.shape != (1,) or context_lengths.dtype != jnp.int32:
        raise ValueError("context lengths must contain one int32 row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    if physical_page_count <= 0:
        raise ValueError("physical page count must be positive")

    capacity = block_tables.shape[1] * layout.logical_page_size
    current = position[0].astype(jnp.int32)
    length = context_lengths[0]
    safe_current = jnp.clip(current, jnp.int32(0), jnp.int32(capacity - 1))
    logical_page = safe_current // jnp.int32(layout.logical_page_size)
    safe_logical_page = jnp.clip(logical_page, jnp.int32(0), jnp.int32(block_tables.shape[1] - 1))
    physical_page = block_tables[0, safe_logical_page]
    physical_ok = (physical_page >= 0) & (physical_page < physical_page_count)
    safe_physical_page = jnp.clip(physical_page, jnp.int32(0), jnp.int32(physical_page_count - 1))
    within_page = safe_current % jnp.int32(layout.logical_page_size)
    target_owner = within_page // jnp.int32(layout.local_rows_per_page)
    local_row = within_page % jnp.int32(layout.local_rows_per_page)

    page_ids = block_tables[0]
    page_slots = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    required_pages = (jnp.maximum(length, jnp.int32(0)) + jnp.int32(layout.logical_page_size - 1)) // jnp.int32(
        layout.logical_page_size
    )
    live_pages = page_slots < required_pages
    page_table_ok = jnp.all(
        jnp.where(
            live_pages,
            (page_ids >= 0) & (page_ids < physical_page_count),
            True,
        )
    )
    ordered_live_pages = jnp.sort(jnp.where(live_pages, page_ids, jnp.iinfo(jnp.int32).max))
    adjacent_slots = jnp.arange(1, block_tables.shape[1], dtype=jnp.int32)
    page_table_unique = jnp.all(
        jnp.where(
            adjacent_slots < required_pages,
            ordered_live_pages[1:] != ordered_live_pages[:-1],
            True,
        )
    )
    metadata_valid = (
        (length > 0)
        & (length <= capacity)
        & (current == length - 1)
        & (current >= 0)
        & (local_slot >= 0)
        & (local_slot < layout.local_parallel_size)
        & physical_ok
        & page_table_ok
        & page_table_unique
    )
    return (
        safe_physical_page,
        local_row,
        target_owner,
        safe_current,
        metadata_valid[None],
    )


class CanonicalSelectedPositions(NamedTuple):
    """Position-ordered attention copy plus a per-row invariant predicate."""

    selection: SelectedPositions
    contract_valid: jax.Array


class SelectedKvSegment(NamedTuple):
    """Gathered fixed-width selected segment and its protected metadata."""

    values: jax.Array
    positions: jax.Array
    valid_counts: jax.Array
    contract_valid: jax.Array


def canonicalize_selected_positions(
    selected: SelectedPositions,
) -> CanonicalSelectedPositions:
    """Make an ascending-position attention copy without changing DSA state.

    ``contract_valid`` is false for an out-of-range count, a non-``-1`` tail,
    a negative live position, or a duplicate live position.  It remains a
    device value: optimized decode can accumulate it into health state without
    introducing a host callback on the critical path.
    """

    positions = selected.positions
    valid_counts = selected.valid_counts
    if positions.ndim != 2 or positions.shape[1] == 0:
        raise ValueError("selected positions must have shape [rows, positive_width]")
    rows, width = positions.shape
    _require_shape("valid_counts", valid_counts, (rows,))
    _require_int32("selected positions", positions)
    _require_int32("valid_counts", valid_counts)

    counts_in_range = (valid_counts >= 0) & (valid_counts <= width)
    safe_counts = jnp.clip(valid_counts, jnp.int32(0), jnp.int32(width))
    slots = lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    live = slots < safe_counts[:, None]
    sentinel_valid = jnp.all(jnp.where(live, positions >= 0, positions == -1), axis=1)
    keys = jnp.where(live, positions, jnp.int32(_INT32_MAX))
    ordered_keys = jnp.sort(keys, axis=1)
    ordered = jnp.where(slots < safe_counts[:, None], ordered_keys, jnp.int32(-1))
    adjacent_distinct = jnp.all(
        jnp.where(
            slots[:, 1:] < safe_counts[:, None],
            ordered[:, 1:] != ordered[:, :-1],
            True,
        ),
        axis=1,
    )
    valid = counts_in_range & sentinel_valid & adjacent_distinct
    return CanonicalSelectedPositions(SelectedPositions(ordered.astype(jnp.int32), safe_counts), valid)


def selected_positions_for_owner(
    selected: SelectedPositions,
    *,
    layout: StageLocalKvLayout,
    owner_index: int | jax.Array,
) -> CanonicalSelectedPositions:
    """Return one owner's ascending subset with a fixed-width ``-1`` tail."""

    if isinstance(owner_index, int) and not isinstance(owner_index, bool):
        if not 0 <= owner_index < layout.local_parallel_size:
            raise ValueError("owner_index is outside the local stage group")
    elif (
        not hasattr(owner_index, "shape")
        or owner_index.shape != ()
        or not jnp.issubdtype(owner_index.dtype, jnp.integer)
    ):
        raise ValueError("owner_index must be an integer scalar")
    canonical = canonicalize_selected_positions(selected)
    positions = canonical.selection.positions
    counts = canonical.selection.valid_counts
    rows, width = positions.shape
    slots = lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    live = slots < counts[:, None]
    owned = live & (layout.owner(positions) == owner_index)
    keys = jnp.where(owned, positions, jnp.int32(_INT32_MAX))
    ordered = jnp.sort(keys, axis=1)
    owned_counts = jnp.sum(owned, axis=1, dtype=jnp.int32)
    subset = jnp.where(slots < owned_counts[:, None], ordered, jnp.int32(-1))
    return CanonicalSelectedPositions(
        SelectedPositions(subset.astype(jnp.int32), owned_counts),
        canonical.contract_valid,
    )


def gather_stage_local_selected_kv(
    cache_local: jax.Array,
    block_tables: jax.Array,
    selected: SelectedPositions,
    context_lengths: jax.Array,
    *,
    layout: StageLocalKvLayout,
    owner_index: int | jax.Array,
) -> SelectedKvSegment:
    """Gather one chip's owned subset from its context-striped local cache."""

    if cache_local.ndim != 3 or not jnp.issubdtype(cache_local.dtype, jnp.inexact):
        raise ValueError("local cache must be inexact [pages,local_rows,width]")
    if any(dimension <= 0 for dimension in cache_local.shape):
        raise ValueError("local cache dimensions must all be positive")
    num_pages, local_rows, cache_width = cache_local.shape
    if local_rows != layout.local_rows_per_page:
        raise ValueError("local cache rows disagree with the declared KV layout")
    if cache_width != layout.packed_cache_width:
        raise ValueError("local cache width disagrees with the declared KV layout")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [rows,max_blocks]")
    _require_int32("block_tables", block_tables)
    if block_tables.shape[1] == 0:
        raise ValueError("block_tables must expose at least one logical block")

    owned = selected_positions_for_owner(selected, layout=layout, owner_index=owner_index)
    positions = owned.selection.positions
    counts = owned.selection.valid_counts
    rows, width = positions.shape
    _require_shape("block_tables", block_tables, (rows, block_tables.shape[1]))
    _require_shape("context_lengths", context_lengths, (rows,))
    _require_int32("context_lengths", context_lengths)

    slots = lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    live = slots < counts[:, None]
    position_ok = (positions >= 0) & (positions < context_lengths[:, None])
    safe_positions = jnp.where(live & position_ok, positions, jnp.int32(0))
    logical_blocks = safe_positions // jnp.int32(layout.logical_page_size)
    block_ok = logical_blocks < block_tables.shape[1]
    safe_blocks = jnp.clip(logical_blocks, 0, block_tables.shape[1] - 1)
    page_ids = jnp.take_along_axis(block_tables, safe_blocks, axis=1)
    page_ok = (page_ids >= 0) & (page_ids < num_pages)
    safe_pages = jnp.clip(page_ids, 0, max(num_pages - 1, 0))
    within_page = safe_positions % jnp.int32(layout.logical_page_size)
    actual_owner = within_page // jnp.int32(layout.local_rows_per_page)
    owner_ok = actual_owner == owner_index
    local_row = within_page % jnp.int32(layout.local_rows_per_page)
    flat_rows = safe_pages * local_rows + local_row
    flat_cache = cache_local.reshape(num_pages * local_rows, cache_width)
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(rows, width, cache_width)
    slot_ok = live & position_ok & block_ok & page_ok & owner_ok
    gathered = jnp.where(slot_ok[..., None], gathered, jnp.zeros((), cache_local.dtype))
    length_ok = (context_lengths >= 0) & (context_lengths <= block_tables.shape[1] * layout.logical_page_size)
    row_valid = owned.contract_valid & length_ok & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    return SelectedKvSegment(gathered, positions, counts, row_valid)


def gather_stage_local_selected_kv_aligned(
    cache_local: jax.Array,
    block_tables: jax.Array,
    selected: SelectedPositions,
    context_lengths: jax.Array,
    *,
    layout: StageLocalKvLayout,
    owner_index: int | jax.Array,
) -> SelectedKvSegment:
    """Place one owner's rows in their canonical global selected slots.

    Unlike :func:`gather_stage_local_selected_kv`, this helper does not compact
    an owner's subset.  Every lane returns the same ascending positions/counts;
    values owned by another lane are zero.  A topology-local sum can therefore
    reconstruct the selected segment without changing its row order or
    exchanging the complete paged cache.
    """

    if cache_local.ndim != 3 or not jnp.issubdtype(cache_local.dtype, jnp.inexact):
        raise ValueError("local cache must be inexact [pages,local_rows,width]")
    if any(dimension <= 0 for dimension in cache_local.shape):
        raise ValueError("local cache dimensions must all be positive")
    num_pages, local_rows, cache_width = cache_local.shape
    if local_rows != layout.local_rows_per_page:
        raise ValueError("local cache rows disagree with the declared KV layout")
    if cache_width != layout.packed_cache_width:
        raise ValueError("local cache width disagrees with the declared KV layout")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [rows,max_blocks]")
    _require_int32("block_tables", block_tables)
    if block_tables.shape[1] == 0:
        raise ValueError("block_tables must expose at least one logical block")
    if isinstance(owner_index, int) and not isinstance(owner_index, bool):
        if not 0 <= owner_index < layout.local_parallel_size:
            raise ValueError("owner_index is outside the local stage group")
    elif (
        not hasattr(owner_index, "shape")
        or owner_index.shape != ()
        or not jnp.issubdtype(owner_index.dtype, jnp.integer)
    ):
        raise ValueError("owner_index must be an integer scalar")

    canonical = canonicalize_selected_positions(selected)
    positions = canonical.selection.positions
    counts = canonical.selection.valid_counts
    rows, width = positions.shape
    _require_shape("block_tables", block_tables, (rows, block_tables.shape[1]))
    _require_shape("context_lengths", context_lengths, (rows,))
    _require_int32("context_lengths", context_lengths)

    slots = lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    live = slots < counts[:, None]
    position_ok = (positions >= 0) & (positions < context_lengths[:, None])
    safe_positions = jnp.where(live & position_ok, positions, jnp.int32(0))
    logical_blocks = safe_positions // jnp.int32(layout.logical_page_size)
    block_ok = logical_blocks < block_tables.shape[1]
    safe_blocks = jnp.clip(logical_blocks, 0, block_tables.shape[1] - 1)
    page_ids = jnp.take_along_axis(block_tables, safe_blocks, axis=1)
    page_ok = (page_ids >= 0) & (page_ids < num_pages)
    safe_pages = jnp.clip(page_ids, 0, max(num_pages - 1, 0))
    within_page = safe_positions % jnp.int32(layout.logical_page_size)
    actual_owner = within_page // jnp.int32(layout.local_rows_per_page)
    local_row = within_page % jnp.int32(layout.local_rows_per_page)
    flat_rows = safe_pages * local_rows + local_row
    flat_cache = cache_local.reshape(num_pages * local_rows, cache_width)
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(rows, width, cache_width)
    slot_ok = live & position_ok & block_ok & page_ok
    owned = slot_ok & (actual_owner == owner_index)
    gathered = jnp.where(owned[..., None], gathered, jnp.zeros((), cache_local.dtype))
    length_ok = (context_lengths >= 0) & (context_lengths <= block_tables.shape[1] * layout.logical_page_size)
    row_valid = canonical.contract_valid & length_ok & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    return SelectedKvSegment(gathered, positions, counts, row_valid)
