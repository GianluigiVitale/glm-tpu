"""Multi-head latent attention (MLA) of the reference, absorbed form.

Per row at absolute position ``p`` (``normalized`` is the BF16 input norm):

* ``q_residual = rms_norm(normalized @ q_a.T)``; ``kv = normalized @ kv_a.T``
  splits into the latent ``c = rms_norm(kv[:kv_lora])`` and the raw key RoPE
  part ``kv[kv_lora:]`` (all BF16 boundaries);
* ``q = q_residual @ q_b.T`` per head is ``[q_nope (192) | q_rope (64)]``;
* main RoPE uses the host BF16 ``cos|sin`` table row of ``p`` (NumPy FP32
  trigonometry stored in BF16), FP32 products and one final BF16 round, with
  interleaved pairs, for ``q_rope`` and the key RoPE part alike;
* the cache row of ``p`` is ``[c | rope(k_rope)]`` (BF16, written before
  attending, so a row can attend to itself);
* absorbed query ``q_c = q_nope @ W_k`` (``W_k`` = the 192 key rows of the head's
  ``kv_b`` slice), BF16;
* scores over the selected positions ``(q_c . c_t + q_rope . k_rope_t) *
  qk_head_dim**-0.5`` in FP32, softmax in FP32 with the unnormalized
  probabilities rounded to BF16 before the value product, the positions in
  ascending order; ``o = (sum_t P_t c_t) / sum_t P_t`` rounded to BF16;
* per head value ``v = o @ W_v.T`` (the 256 value rows), BF16; output
  ``concat_heads(v) @ o_proj.T``, BF16.

The softmax is ``sparse_mla_attention`` (the reference of the sparse-MLA
kernel, ``glm_tpu/kernels/sparse_mla/kernel.py``) over the rows
``gather_paged_selected_kv`` below gathers (an unsharded cache, one page); the
RoPE table and rotation are the engine's ``build_rotary_table_host`` /
``apply_rotary_fp32_final_round``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from glm_tpu.layers.contracts import (
    MlaNumericalContract,
    SelectedPositions,
    StageLocalKvLayout,
    _require_int32,
    _require_shape,
)
from glm_tpu.kernels.sparse_mla.kernel import sparse_mla_attention, SparseAttentionResult
from glm_tpu.layers.rope import apply_rotary_fp32_final_round, build_rotary_table_host

from .linear import einsum, project
from .norm import rms_norm
from glm_tpu.layers.attention.kv_cache import (
    SelectedKvSegment,
    canonicalize_selected_positions,
    gather_stage_local_selected_kv,
)


class QkvAWeights(NamedTuple):
    q_a: jax.Array  # [q_lora_rank, hidden]
    q_a_norm: jax.Array  # [q_lora_rank]
    kv_a: jax.Array  # [kv_lora_rank + rope_dim, hidden]
    kv_a_norm: jax.Array  # [kv_lora_rank]


class AttentionWeights(NamedTuple):
    q_b: jax.Array  # [heads * (nope + rope), q_lora_rank]
    # [heads * (nope + v), kv_lora_rank]: per head the nope key rows, then the value rows
    kv_b: jax.Array
    o: jax.Array  # [hidden, heads * v]


class PreparedAttention(NamedTuple):
    q_residual: jax.Array  # [rows, q_lora_rank] BF16
    latent: jax.Array  # [rows, kv_lora_rank] BF16 (normalized)
    key_rope_input: jax.Array  # [rows, rope_dim] BF16 (not yet rotated)


def main_rope_table(capacity: int, *, rotary_dim: int, theta: float) -> jax.Array:
    """Host BF16 ``cos|sin`` table, one row per position."""
    table = build_rotary_table_host(capacity, rotary_dim=rotary_dim, theta=theta)
    return jnp.asarray(np.asarray(table))


def rotate(value: jax.Array, table_rows: jax.Array) -> jax.Array:
    """Interleaved main RoPE of ``value[rows, ..., rope_dim]`` with the rows' table rows."""
    half = table_rows.shape[-1] // 2
    shape = (table_rows.shape[0],) + (1,) * (value.ndim - 2) + (half,)
    cos = table_rows[:, :half].reshape(shape)
    sin = table_rows[:, half:].reshape(shape)
    return apply_rotary_fp32_final_round(value, cos, sin, interleaved=True)


def prepare(normalized: jax.Array, weights: QkvAWeights, *, epsilon: float) -> PreparedAttention:
    """The shared q-a / kv-a boundary consumed by attention and the DSA indexer."""
    kv_lora_rank = weights.kv_a_norm.shape[0]
    q_residual = rms_norm(project(normalized, weights.q_a), weights.q_a_norm, epsilon=epsilon)
    kv = project(normalized, weights.kv_a)
    latent = rms_norm(kv[:, :kv_lora_rank], weights.kv_a_norm, epsilon=epsilon)
    return PreparedAttention(q_residual, latent, kv[:, kv_lora_rank:])


def mla_attention(
    prepared: PreparedAttention,
    positions: jax.Array,
    kv_cache: jax.Array,
    selection: SelectedPositions,
    weights: AttentionWeights,
    *,
    contract: MlaNumericalContract,
    rope_table: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Write the rows' cache entries, attend over the selection; return ``(output, cache)``.

    ``kv_cache`` is ``[capacity, kv_lora_rank + rope_dim]``; ``selection`` holds
    each row's selected positions (any order) and live counts. Refuses a selection
    that breaks the attention contract (the oracle's ``contract_valid``: a count
    outside ``[0, top_k]``, a non-``-1`` tail, a duplicate, a negative position or
    one past the row's own).
    """
    rows = positions.shape[0]
    heads = contract.num_heads
    nope, rope = contract.qk_nope_head_dim, contract.qk_rope_head_dim
    table_rows = jnp.take(rope_table, positions, axis=0)

    q = project(prepared.q_residual, weights.q_b).reshape(rows, heads, nope + rope)
    q_rope = rotate(q[..., nope:], table_rows)
    key_rope = rotate(prepared.key_rope_input[:, None, :], table_rows)[:, 0]
    cache_rows = jnp.concatenate((prepared.latent, key_rope), axis=-1)
    kv_cache = kv_cache.at[positions].set(cache_rows)

    kv_b = weights.kv_b.reshape(heads, nope + contract.v_head_dim, contract.kv_lora_rank)
    q_absorbed = einsum("rhd,hdc->rhc", q[..., :nope], kv_b[:, :nope])
    segment = gather_paged_selected_kv(
        kv_cache[None],
        jnp.zeros((rows, 1), jnp.int32),
        selection,
        (positions + 1).astype(jnp.int32),
    )
    attended = sparse_mla_attention(q_absorbed, q_rope, segment, contract=contract)
    if not bool(jnp.all(attended.contract_valid)):
        # A malformed selection (bad count, duplicate, negative or future position) is
        # zeroed by the oracle; the reference refuses it instead of computing with it.
        raise ValueError("DSA selection violates the attention contract")
    values = einsum("rhc,hvc->rhv", attended.output, kv_b[:, nope:])
    output = project(values.reshape(rows, heads * contract.v_head_dim), weights.o)
    return output, kv_cache


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
    gathered = jnp.take(flat_cache, flat_rows.reshape(-1), axis=0).reshape(rows, width, cache_width)
    slot_ok = live & position_ok & block_ok & page_ok
    gathered = jnp.where(slot_ok[..., None], gathered, jnp.zeros((), cache.dtype))
    length_ok = (context_lengths >= 0) & (context_lengths <= block_tables.shape[1] * page_size)
    row_valid = canonical.contract_valid & length_ok & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    return SelectedKvSegment(gathered, positions, counts, row_valid)


def combine_stage_local_attention(
    partial_outputs: jax.Array,
    partial_logsumexp: jax.Array,
    partial_contract_valid: jax.Array,
) -> SparseAttentionResult:
    """Exact FP32 LSE merge over disjoint local-owner selected subsets."""

    if partial_outputs.ndim != 4 or partial_logsumexp.ndim != 3:
        raise ValueError("stage-local partial output/LSE ranks must be four/three")
    owners, rows, heads, _ = partial_outputs.shape
    _require_shape("partial_logsumexp", partial_logsumexp, (owners, rows, heads))
    _require_shape("partial_contract_valid", partial_contract_valid, (owners, rows))
    if owners == 0:
        raise ValueError("stage-local attention requires at least one owner")
    live = jnp.isfinite(partial_logsumexp)
    maximum = jnp.max(jnp.where(live, partial_logsumexp, -jnp.inf), axis=0)
    any_live = jnp.any(live, axis=0)
    safe_maximum = jnp.where(any_live, maximum, 0.0)
    weights = jnp.where(live, jnp.exp(partial_logsumexp - safe_maximum[None, ...]), 0.0)
    denominator = jnp.sum(weights, axis=0)
    safe_denominator = jnp.where(any_live, denominator, 1.0)
    combined = jnp.sum(partial_outputs.astype(jnp.float32) * weights[..., None], axis=0) / safe_denominator[..., None]
    combined = jnp.where(any_live[..., None], combined, 0.0)
    combined_lse = jnp.where(any_live, safe_maximum + jnp.log(safe_denominator), -jnp.inf)
    valid = jnp.all(partial_contract_valid, axis=0)
    return SparseAttentionResult(combined.astype(partial_outputs.dtype), combined_lse, valid)


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
        raise ValueError("decode_batch1 stage-local sparse MLA requires exactly one row")

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
    return combine_stage_local_attention(jnp.stack(outputs), jnp.stack(logsumexp), jnp.stack(validity))
