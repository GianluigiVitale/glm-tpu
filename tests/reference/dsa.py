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

Everything is delegated to the exact oracles of
``glm_tpu/layers/attention/_s3_dsa.py`` (``dsa_index_keys``,
``dsa_query_and_head_weights``, ``dsa_scores(precision="highest")``,
``exact_topk``). The RoPE pairing follows ``indexer_rope_interleave: true`` of
the pinned config, as the engine does.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.layers.attention._s3_dsa import (
    DsaNumericalContract,
    dsa_index_keys,
    dsa_query_and_head_weights,
    dsa_scores,
    exact_topk,
)

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
