"""DSA lightning indexer of the reference: keys, queries, scores, exact top-k.

For every query row at absolute position ``p`` the indexer scores all cached
positions ``0..p`` (the current row's key is written first) and keeps the
``top_k`` best:

* key ``k_t = rope(LayerNorm(normalized_t @ wk.T))``: FP32 projection, FP32
  affine LayerNorm (``eps = 1e-6``, ``(x - mean) * rsqrt(var + eps)``), RoPE on
  the first ``rope_dim`` features with interleaved pairs and on-device FP32
  ``cos``/``sin``; the cache stores the key rounded to BF16;
* query ``q_h = rope(q_residual @ wq_b.T)`` per indexer head, FP32;
* head weights ``w_h = (normalized @ weights_proj.T) * heads**-0.5``, FP32;
* score ``s_t = sum_h w_h * relu(q_h . k_t * head_dim**-0.5)``, FP32;
* selection: descending score, ties to the lowest position; positions past
  ``p`` are masked; when ``p + 1 < top_k`` the tail is ``-1``.

Selections stay in score order (the IndexShare state); attention reads a
position-sorted copy. Shared layers reuse the preceding full layer's selection.

Everything is delegated to the exact oracles (``dsa_index_keys``,
``dsa_query_and_head_weights`` and ``exact_topk`` below, the engine's
``dsa_scores(precision="highest")``). The RoPE pairing follows ``indexer_rope_interleave: true`` of
the pinned config, as the engine does.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.layers.contracts import DsaNumericalContract, SelectedPositions, _require_shape
from glm_tpu.layers.attention.dsa_indexer import (
    dsa_scores,
    _NEGATIVE_INFINITY,
    _merge_topk_candidates_scored,
    dsa_index_keys_from_projection,
    local_topk_candidates,
)
from glm_tpu.layers.rope import apply_rotary, rotary_cos_sin
from tests.reference.linear import linear


INDEX_KEY_NORM_EPSILON = 1e-6


class IndexerWeights(NamedTuple):
    wq_b: jax.Array  # [heads * head_dim, q_lora_rank] BF16 (dequantized)
    wk: jax.Array  # [head_dim, hidden] BF16 (dequantized)
    k_norm_weight: jax.Array  # [head_dim] BF16
    k_norm_bias: jax.Array  # [head_dim] BF16
    weights_proj: jax.Array  # [heads, hidden] BF16


class Selection(NamedTuple):
    """One producer's per-row selection (score order, ``-1`` tail)."""

    positions: jax.Array  # [rows, top_k] int32
    valid_counts: jax.Array  # [rows] int32
    scores: jax.Array  # [rows, top_k] float32, -inf tail
    # [rows] float32: the top_k-th minus the (top_k+1)-th score; +inf if all are selected
    margin: jax.Array


def index_keys(
    normalized: jax.Array,
    weights: IndexerWeights,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract,
) -> jax.Array:
    """Cache-resident index keys ``[rows, head_dim]`` in BF16."""
    keys = dsa_index_keys(
        normalized,
        weights.wk,
        weights.k_norm_weight,
        weights.k_norm_bias,
        positions,
        contract=contract,
    )
    return keys.astype(jnp.bfloat16)


def decision_margin(scores: jax.Array, lengths: jax.Array, top_k: int) -> jax.Array:
    """Per row, the ``top_k``-th minus the ``(top_k + 1)``-th causal score.

    ``+inf`` where no ``(top_k + 1)``-th candidate exists (``length <= top_k``,
    including a context no wider than ``top_k``): every position is selected.
    """
    width = scores.shape[1]
    if width <= top_k:
        return jnp.full(lengths.shape, jnp.inf, jnp.float32)
    causal = jnp.arange(width)[None, :] < lengths[:, None]
    ranked = lax.top_k(jnp.where(causal, scores, -jnp.inf), top_k + 1)[0]
    return jnp.where(lengths > top_k, ranked[:, -2] - ranked[:, -1], jnp.inf)


def select(
    normalized: jax.Array,
    q_residual: jax.Array,
    index_cache: jax.Array,
    positions: jax.Array,
    weights: IndexerWeights,
    *,
    contract: DsaNumericalContract,
) -> Selection:
    """Exact causal top-k over ``index_cache`` (``[capacity, head_dim]``)."""
    query, head_weights = dsa_query_and_head_weights(
        normalized,
        q_residual,
        weights.wq_b,
        weights.weights_proj,
        positions,
        contract=contract,
    )
    scores = dsa_scores(query, index_cache, head_weights, precision="highest")
    lengths = positions + 1
    selected = exact_topk(scores, lengths, top_k=contract.top_k)
    live = selected.positions >= 0
    picked = jnp.take_along_axis(scores, jnp.where(live, selected.positions, 0), axis=1)
    margin = decision_margin(scores, lengths, contract.top_k)
    return Selection(
        selected.positions,
        selected.valid_counts,
        jnp.where(live, picked, -jnp.inf).astype(jnp.float32),
        margin.astype(jnp.float32),
    )


def dsa_query_and_head_weights(
    hidden_states: jax.Array,
    q_residual: jax.Array,
    query_weight_out_in: jax.Array,
    head_weight_out_in: jax.Array,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> tuple[jax.Array, jax.Array]:
    """Project current rows to rotated indexer queries and signed head weights."""

    rows = hidden_states.shape[0] if hidden_states.ndim == 2 else -1
    _require_shape("hidden_states", hidden_states, (rows, contract.hidden_size))
    _require_shape("q_residual", q_residual, (rows, contract.q_lora_rank))
    _require_shape(
        "query_weight_out_in",
        query_weight_out_in,
        (contract.num_heads * contract.head_dim, contract.q_lora_rank),
    )
    _require_shape(
        "head_weight_out_in",
        head_weight_out_in,
        (contract.num_heads, contract.hidden_size),
    )
    _require_shape("positions", positions, (rows,))
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")

    with jax.default_matmul_precision("highest"):
        query = linear(
            q_residual,
            query_weight_out_in,
            output_dtype=jnp.float32,
        ).reshape(rows, contract.num_heads, contract.head_dim)
        head_weights = linear(
            hidden_states,
            head_weight_out_in,
            output_dtype=jnp.float32,
        ) * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        query[..., : contract.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=contract.interleaved_rotary,
    )
    query = jnp.concatenate((rotated, query[..., contract.rotary_dim :]), axis=-1)
    return query.astype(jnp.float32), head_weights.astype(jnp.float32)


def dsa_index_keys(
    hidden_states: jax.Array,
    key_weight_out_in: jax.Array,
    key_norm_weight: jax.Array,
    key_norm_bias: jax.Array,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> jax.Array:
    """Project, normalize, and rotate the cache-resident DSA index keys."""

    tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else -1
    _require_shape("hidden_states", hidden_states, (tokens, contract.hidden_size))
    _require_shape(
        "key_weight_out_in",
        key_weight_out_in,
        (contract.head_dim, contract.hidden_size),
    )
    _require_shape("key_norm_weight", key_norm_weight, (contract.head_dim,))
    _require_shape("key_norm_bias", key_norm_bias, (contract.head_dim,))
    _require_shape("positions", positions, (tokens,))
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")

    with jax.default_matmul_precision("highest"):
        projected = linear(
            hidden_states,
            key_weight_out_in,
            output_dtype=jnp.float32,
        )
    return dsa_index_keys_from_projection(
        projected,
        key_norm_weight,
        key_norm_bias,
        positions,
        contract=contract,
    )


def exact_topk(
    scores: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
) -> SelectedPositions:
    """Select exact descending scores with lowest-position tie order.

    Context positions outside each row's valid length are masked. If context
    is shorter than ``top_k``, the fixed-width tail is exactly ``-1`` and
    ``valid_counts`` reports the live prefix. ``lax.approx_max_k`` is banned.
    """

    if scores.ndim != 2:
        raise ValueError("DSA scores must have shape [rows, context]")
    rows, context = scores.shape
    _require_shape("valid_lengths", valid_lengths, (rows,))
    if not jnp.issubdtype(valid_lengths.dtype, jnp.integer):
        raise ValueError("valid lengths must have an integer dtype")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    positions = jnp.arange(context, dtype=jnp.int32)
    masked = jnp.where(
        positions[None, :] < valid_lengths.astype(jnp.int32)[:, None],
        scores.astype(jnp.float32),
        _NEGATIVE_INFINITY,
    )
    padded_width = max(context, top_k)
    if padded_width != context:
        masked = jnp.pad(
            masked,
            ((0, 0), (0, padded_width - context)),
            constant_values=_NEGATIVE_INFINITY,
        )
    _, selected = lax.top_k(masked, top_k)
    selected = selected.astype(jnp.int32)
    valid_counts = jnp.clip(
        valid_lengths.astype(jnp.int32),
        jnp.int32(0),
        jnp.int32(min(context, top_k)),
    )
    slots = lax.broadcasted_iota(jnp.int32, (rows, top_k), 1)
    selected = jnp.where(slots < valid_counts[:, None], selected, jnp.int32(-1))
    return SelectedPositions(selected, valid_counts)


def merge_topk_candidates(
    candidate_scores: jax.Array,
    candidate_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
) -> SelectedPositions:
    """Merge stage-local candidates independent of concatenation order."""

    selected = _merge_topk_candidates_scored(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
    )
    return SelectedPositions(selected.positions, selected.valid_counts)


def distributed_exact_topk_reference(
    shard_scores: jax.Array,
    shard_global_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
) -> SelectedPositions:
    """Reference local-candidate/all-gather/merge semantics for one stage group."""

    if shard_scores.ndim != 3:
        raise ValueError("shard_scores must have shape [local_group, rows, local_context]")
    groups, _, local_context = shard_scores.shape
    _require_shape("shard_global_positions", shard_global_positions, (groups, local_context))
    candidate_scores, candidate_positions = jax.vmap(
        lambda scores, positions: local_topk_candidates(scores, positions, valid_lengths, top_k=top_k)
    )(shard_scores, shard_global_positions)
    return merge_topk_candidates(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
    )
