"""Exact stage-local paged KV lookup and sparse MLA attention references.

The optimized path must preserve these semantics while replacing the readable
JAX operations with local shard maps or Pallas kernels.  Context rows are
striped inside every logical page over only the plan's local stage group.
Selected positions remain score ordered in DSA/IndexShare state; a private
ascending-position copy is used for attention so floating-point accumulation
is a deterministic function of the selected set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .dsa import SelectedPositions


_INT32_MAX = jnp.iinfo(jnp.int32).max


@dataclass(frozen=True, slots=True)
class MlaNumericalContract:
    """Compile-relevant GLM-5.2 absorbed-MLA arithmetic contract."""

    num_heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    packed_cache_width: int = 640
    top_k: int = 2048
    score_dtype: str = "float32"
    probability_rounding: str = "unnormalized_cache_dtype_before_pv"
    position_order: str = "ascending_global_position"

    def __post_init__(self) -> None:
        for field in (
            "num_heads",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "packed_cache_width",
            "top_k",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.qk_nope_head_dim + self.qk_rope_head_dim != self.qk_head_dim:
            raise ValueError("qk_head_dim must equal nope plus RoPE dimensions")
        if self.kv_lora_rank + self.qk_rope_head_dim > self.packed_cache_width:
            raise ValueError("packed cache width truncates latent or RoPE state")
        if self.score_dtype != "float32":
            raise ValueError("sparse MLA scores and softmax must be FP32")
        if self.probability_rounding != "unnormalized_cache_dtype_before_pv":
            raise ValueError("sparse MLA probability-rounding contract drifted")
        if self.position_order != "ascending_global_position":
            raise ValueError("sparse MLA accumulation order must be position canonical")

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5


@dataclass(frozen=True, slots=True)
class StageLocalKvLayout:
    """Context striping for one attention layer's topology-local cache."""

    logical_page_size: int = 512
    local_parallel_size: int = 4
    packed_cache_width: int = 640

    def __post_init__(self) -> None:
        for field in (
            "logical_page_size",
            "local_parallel_size",
            "packed_cache_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.logical_page_size % self.local_parallel_size:
            raise ValueError("logical page size must divide over the local stage group")

    @property
    def local_rows_per_page(self) -> int:
        return self.logical_page_size // self.local_parallel_size

    def owner(self, positions: jax.Array) -> jax.Array:
        """Return the local chip owning each non-negative absolute position."""

        return (
            positions.astype(jnp.int32) % jnp.int32(self.logical_page_size)
        ) // jnp.int32(self.local_rows_per_page)


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


class SparseAttentionResult(NamedTuple):
    """Attended latent, additive LSE, and the propagated health predicate."""

    output: jax.Array
    logsumexp: jax.Array
    contract_valid: jax.Array


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _require_int32(name: str, value: jax.Array) -> None:
    if value.dtype != jnp.int32:
        raise ValueError(f"{name} must have dtype int32, got {value.dtype}")


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
    sentinel_valid = jnp.all(
        jnp.where(live, positions >= 0, positions == -1), axis=1
    )
    keys = jnp.where(live, positions, jnp.int32(_INT32_MAX))
    ordered_keys = jnp.sort(keys, axis=1)
    ordered = jnp.where(
        slots < safe_counts[:, None], ordered_keys, jnp.int32(-1)
    )
    adjacent_distinct = jnp.all(
        jnp.where(
            slots[:, 1:] < safe_counts[:, None],
            ordered[:, 1:] != ordered[:, :-1],
            True,
        ),
        axis=1,
    )
    valid = counts_in_range & sentinel_valid & adjacent_distinct
    return CanonicalSelectedPositions(
        SelectedPositions(ordered.astype(jnp.int32), safe_counts), valid
    )


def gather_paged_selected_kv(
    cache: jax.Array,
    block_tables: jax.Array,
    selected: SelectedPositions,
    context_lengths: jax.Array,
) -> SelectedKvSegment:
    """Gather selected rows from an unsharded token-major paged cache.

    This is the readable oracle for stage-local owner gathers.  Invalid
    metadata never reads outside the cache: the returned health predicate is
    false and unsafe/tail values are zeroed.  Serving must refuse a false
    predicate rather than treating the zeroed diagnostic result as valid.
    """

    if cache.ndim != 3 or not jnp.issubdtype(cache.dtype, jnp.inexact):
        raise ValueError("cache must be an inexact [pages,page_size,width] array")
    if any(dimension <= 0 for dimension in cache.shape):
        raise ValueError("cache dimensions must all be positive")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [rows,max_blocks]")
    _require_int32("block_tables", block_tables)
    canonical = canonicalize_selected_positions(selected)
    positions = canonical.selection.positions
    counts = canonical.selection.valid_counts
    rows, width = positions.shape
    _require_shape("block_tables", block_tables, (rows, block_tables.shape[1]))
    _require_shape("context_lengths", context_lengths, (rows,))
    _require_int32("context_lengths", context_lengths)
    if block_tables.shape[1] == 0:
        raise ValueError("block_tables must expose at least one logical block")

    num_pages, page_size, cache_width = cache.shape
    slots = lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    live = slots < counts[:, None]
    position_ok = (positions >= 0) & (positions < context_lengths[:, None])
    safe_positions = jnp.where(live & position_ok, positions, jnp.int32(0))
    logical_blocks = safe_positions // jnp.int32(page_size)
    block_ok = logical_blocks < block_tables.shape[1]
    safe_blocks = jnp.clip(logical_blocks, 0, block_tables.shape[1] - 1)
    page_ids = jnp.take_along_axis(block_tables, safe_blocks, axis=1)
    page_ok = (page_ids >= 0) & (page_ids < num_pages)
    safe_pages = jnp.clip(page_ids, 0, max(num_pages - 1, 0))
    flat_rows = safe_pages * page_size + safe_positions % page_size
    flat_cache = cache.reshape(num_pages * page_size, cache_width)
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(
        rows, width, cache_width
    )
    slot_ok = live & position_ok & block_ok & page_ok
    gathered = jnp.where(slot_ok[..., None], gathered, jnp.zeros((), cache.dtype))
    length_ok = (context_lengths >= 0) & (
        context_lengths <= block_tables.shape[1] * page_size
    )
    row_valid = (
        canonical.contract_valid
        & length_ok
        & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    )
    return SelectedKvSegment(gathered, positions, counts, row_valid)


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
    subset = jnp.where(
        slots < owned_counts[:, None], ordered, jnp.int32(-1)
    )
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

    owned = selected_positions_for_owner(
        selected, layout=layout, owner_index=owner_index
    )
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
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(
        rows, width, cache_width
    )
    slot_ok = live & position_ok & block_ok & page_ok & owner_ok
    gathered = jnp.where(slot_ok[..., None], gathered, jnp.zeros((), cache_local.dtype))
    length_ok = (context_lengths >= 0) & (
        context_lengths <= block_tables.shape[1] * layout.logical_page_size
    )
    row_valid = (
        owned.contract_valid
        & length_ok
        & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    )
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
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(
        rows, width, cache_width
    )
    slot_ok = live & position_ok & block_ok & page_ok
    owned = slot_ok & (actual_owner == owner_index)
    gathered = jnp.where(
        owned[..., None], gathered, jnp.zeros((), cache_local.dtype)
    )
    length_ok = (context_lengths >= 0) & (
        context_lengths <= block_tables.shape[1] * layout.logical_page_size
    )
    row_valid = (
        canonical.contract_valid
        & length_ok
        & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    )
    return SelectedKvSegment(gathered, positions, counts, row_valid)


def sparse_mla_attention(
    query_nope_absorbed: jax.Array,
    query_rope: jax.Array,
    segment: SelectedKvSegment,
    *,
    contract: MlaNumericalContract = MlaNumericalContract(),
) -> SparseAttentionResult:
    """FP32 softmax attention over only the gathered selected latent rows."""

    if query_nope_absorbed.ndim != 3 or query_rope.ndim != 3:
        raise ValueError("sparse MLA queries must have rank three")
    rows = query_nope_absorbed.shape[0]
    _require_shape(
        "query_nope_absorbed",
        query_nope_absorbed,
        (rows, contract.num_heads, contract.kv_lora_rank),
    )
    _require_shape(
        "query_rope",
        query_rope,
        (rows, contract.num_heads, contract.qk_rope_head_dim),
    )
    if query_nope_absorbed.dtype != query_rope.dtype:
        raise ValueError("sparse MLA query components must share one dtype")
    if query_nope_absorbed.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("sparse MLA queries must be BF16 or FP32")
    if segment.positions.ndim != 2:
        raise ValueError("selected KV positions must have shape [rows,top_k]")
    width = segment.positions.shape[1]
    if width != contract.top_k:
        raise ValueError(
            f"selected KV width must equal contract top_k={contract.top_k}, got {width}"
        )
    _require_shape(
        "selected KV segment",
        segment.values,
        (rows, width, contract.packed_cache_width),
    )
    _require_shape("segment valid_counts", segment.valid_counts, (rows,))
    _require_shape("segment contract_valid", segment.contract_valid, (rows,))
    _require_int32("segment positions", segment.positions)
    _require_int32("segment valid_counts", segment.valid_counts)
    if segment.contract_valid.dtype != jnp.bool_:
        raise ValueError("segment contract_valid must have dtype bool")
    if segment.values.dtype != query_nope_absorbed.dtype:
        raise ValueError("queries and selected KV segment must share the cache dtype")

    safe_counts = jnp.clip(
        segment.valid_counts, jnp.int32(0), jnp.int32(width)
    )
    position_slots = jnp.arange(width, dtype=jnp.int32)[None, :]
    position_live = position_slots < safe_counts[:, None]
    positions_well_formed = jnp.all(
        jnp.where(
            position_live,
            segment.positions >= 0,
            segment.positions == -1,
        ),
        axis=1,
    )
    positions_ascending = jnp.all(
        jnp.where(
            position_slots[:, 1:] < safe_counts[:, None],
            segment.positions[:, 1:] > segment.positions[:, :-1],
            True,
        ),
        axis=1,
    )
    metadata_valid = (
        (segment.valid_counts >= 0)
        & (segment.valid_counts <= width)
        & positions_well_formed
        & positions_ascending
    )

    precision = (
        lax.Precision.HIGHEST
        if query_nope_absorbed.dtype == jnp.float32
        else lax.Precision.DEFAULT
    )
    q_nope = query_nope_absorbed
    q_rope = query_rope
    kv_latent = segment.values[..., : contract.kv_lora_rank]
    kv_rope = segment.values[
        ...,
        contract.kv_lora_rank : contract.kv_lora_rank
        + contract.qk_rope_head_dim,
    ]
    scores = (
        jnp.einsum(
            "rhd,rkd->rhk",
            q_nope,
            kv_latent,
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        + jnp.einsum(
            "rhd,rkd->rhk",
            q_rope,
            kv_rope,
            preferred_element_type=jnp.float32,
            precision=precision,
        )
    ) * jnp.float32(contract.softmax_scale)
    slots = jnp.arange(width, dtype=jnp.int32)[None, None, :]
    live = slots < safe_counts[:, None, None]
    masked_scores = jnp.where(live, scores, -jnp.inf)
    has_live = safe_counts > 0
    maximum = jnp.max(masked_scores, axis=-1, keepdims=True)
    safe_maximum = jnp.where(has_live[:, None, None], maximum, 0.0)
    unnormalized = jnp.where(
        live, jnp.exp(masked_scores - safe_maximum), 0.0
    )
    denominator = jnp.sum(unnormalized, axis=-1, keepdims=True)
    safe_denominator = jnp.where(has_live[:, None, None], denominator, 1.0)
    weighted_values = jnp.einsum(
        "rhk,rkd->rhd",
        unnormalized.astype(segment.values.dtype),
        segment.values[..., : contract.kv_lora_rank],
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    output = weighted_values / safe_denominator
    output = jnp.where(has_live[:, None, None], output, 0.0)
    logsumexp = jnp.where(
        has_live[:, None],
        safe_maximum[..., 0] + jnp.log(safe_denominator[..., 0]),
        -jnp.inf,
    )
    return SparseAttentionResult(
        output.astype(query_nope_absorbed.dtype),
        logsumexp.astype(jnp.float32),
        segment.contract_valid & metadata_valid,
    )


def combine_stage_local_attention(
    partial_outputs: jax.Array,
    partial_logsumexp: jax.Array,
    partial_contract_valid: jax.Array,
) -> SparseAttentionResult:
    """Exact FP32 LSE merge over disjoint local-owner selected subsets."""

    if partial_outputs.ndim != 4 or partial_logsumexp.ndim != 3:
        raise ValueError("stage-local partial output/LSE ranks must be four/three")
    owners, rows, heads, _ = partial_outputs.shape
    _require_shape(
        "partial_logsumexp", partial_logsumexp, (owners, rows, heads)
    )
    _require_shape(
        "partial_contract_valid", partial_contract_valid, (owners, rows)
    )
    if owners == 0:
        raise ValueError("stage-local attention requires at least one owner")
    live = jnp.isfinite(partial_logsumexp)
    maximum = jnp.max(jnp.where(live, partial_logsumexp, -jnp.inf), axis=0)
    any_live = jnp.any(live, axis=0)
    safe_maximum = jnp.where(any_live, maximum, 0.0)
    weights = jnp.where(
        live, jnp.exp(partial_logsumexp - safe_maximum[None, ...]), 0.0
    )
    denominator = jnp.sum(weights, axis=0)
    safe_denominator = jnp.where(any_live, denominator, 1.0)
    combined = jnp.sum(
        partial_outputs.astype(jnp.float32) * weights[..., None], axis=0
    ) / safe_denominator[..., None]
    combined = jnp.where(any_live[..., None], combined, 0.0)
    combined_lse = jnp.where(
        any_live, safe_maximum + jnp.log(safe_denominator), -jnp.inf
    )
    valid = jnp.all(partial_contract_valid, axis=0)
    return SparseAttentionResult(
        combined.astype(partial_outputs.dtype), combined_lse, valid
    )


def stage_local_sparse_mla_reference(
    query_nope_absorbed: jax.Array,
    query_rope: jax.Array,
    local_caches: jax.Array,
    block_tables: jax.Array,
    selected: SelectedPositions,
    context_lengths: jax.Array,
    *,
    layout: StageLocalKvLayout,
    contract: MlaNumericalContract = MlaNumericalContract(),
) -> SparseAttentionResult:
    """Readable local-owner gather/attend/LSE-combine semantic reference."""

    if local_caches.ndim != 4:
        raise ValueError("local_caches must be [owners,pages,local_rows,width]")
    if local_caches.shape[0] != layout.local_parallel_size:
        raise ValueError("local cache owner count disagrees with the KV layout")
    if layout.packed_cache_width != contract.packed_cache_width:
        raise ValueError("KV layout and numerical contract cache widths disagree")
    if query_nope_absorbed.shape[0] != 1:
        raise ValueError(
            "decode_batch1 stage-local sparse MLA requires exactly one row"
        )

    outputs = []
    logsumexp = []
    validity = []
    for owner in range(layout.local_parallel_size):
        segment = gather_stage_local_selected_kv(
            local_caches[owner],
            block_tables,
            selected,
            context_lengths,
            layout=layout,
            owner_index=owner,
        )
        partial = sparse_mla_attention(
            query_nope_absorbed,
            query_rope,
            segment,
            contract=contract,
        )
        outputs.append(partial.output)
        logsumexp.append(partial.logsumexp)
        validity.append(partial.contract_valid)
    return combine_stage_local_attention(
        jnp.stack(outputs), jnp.stack(logsumexp), jnp.stack(validity)
    )
