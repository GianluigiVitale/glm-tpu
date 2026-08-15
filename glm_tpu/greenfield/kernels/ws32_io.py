"""WS32 embedding, final-logit, and greedy-token boundaries.

Vocabulary tables retain ``P(expert, feature)`` ownership.  The embedding
path reduces one owned hidden shard over expert-8.  The logit path reduces
feature partials over feature-4 and leaves vocabulary rows expert-sharded.
Sampling exchanges only one score/index candidate per expert, never a full
vocabulary vector.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .ws32 import ws32_rms_norm_mapped


class Ws32EmbeddingResult(NamedTuple):
    residual_local: Any
    contract_valid: Any


class Ws32GreedySampleResult(NamedTuple):
    token_id: Any
    contract_valid: Any


def _require_vocabulary_geometry(
    table_local: Any,
    *,
    vocab_size: int,
) -> tuple[int, int]:
    if table_local.ndim != 2 or table_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 vocabulary table must be rank-two BF16")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or (
        vocab_size <= 0
    ):
        raise ValueError("WS32 vocabulary size must be a positive integer")
    local_vocab, local_hidden = table_local.shape
    if local_vocab <= 0 or local_hidden <= 0 or local_vocab * 8 != vocab_size:
        raise ValueError("WS32 vocabulary table must shard exactly over expert-8")
    return local_vocab, local_hidden


def ws32_embedding_mapped(
    token_ids: Any,
    embedding_local: Any,
    *,
    vocab_size: int,
    expert_axis: str = "expert",
) -> Ws32EmbeddingResult:
    """Look up one token without reconstructing the 2D embedding table."""

    local_vocab, _ = _require_vocabulary_geometry(
        embedding_local, vocab_size=vocab_size
    )
    if token_ids.shape != (1,) or token_ids.dtype != jnp.int32:
        raise ValueError("WS32 embedding requires one int32 token")
    token = token_ids[0]
    owner = lax.axis_index(expert_axis)
    start = owner.astype(jnp.int32) * jnp.int32(local_vocab)
    token_valid = (token >= jnp.int32(0)) & (
        token < jnp.int32(vocab_size)
    )
    owns = token_valid & (token >= start) & (
        token < start + jnp.int32(local_vocab)
    )
    local_id = jnp.clip(
        token - start, jnp.int32(0), jnp.int32(local_vocab - 1)
    )
    row = embedding_local[local_id][None, :]
    row = jnp.where(owns, row, jnp.zeros_like(row))
    with jax.named_scope("greenfield_ws32_embedding/expert_owner_reduce"):
        residual = lax.psum(row, axis_name=expert_axis)
    return Ws32EmbeddingResult(residual, token_valid[None])


def ws32_logits_mapped(
    hidden_local: Any,
    lm_head_local: Any,
    *,
    vocab_size: int,
    feature_axis: str = "feature",
) -> Any:
    """Return one physically expert-sharded vocabulary-logit row."""

    local_vocab, local_hidden = _require_vocabulary_geometry(
        lm_head_local, vocab_size=vocab_size
    )
    if hidden_local.shape != (1, local_hidden) or (
        hidden_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 logits require one BF16 hidden feature shard")
    partial = lax.dot_general(
        hidden_local.astype(jnp.float32),
        lm_head_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if partial.shape != (1, local_vocab):
        raise ValueError("WS32 local vocabulary projection geometry drifted")
    with jax.named_scope("greenfield_ws32_logits/feature_reduce"):
        return lax.psum(partial, axis_name=feature_axis).astype(jnp.bfloat16)


def ws32_greedy_sample_mapped(
    local_logits: Any,
    *,
    vocab_size: int,
    expert_axis: str = "expert",
) -> Ws32GreedySampleResult:
    """Select the exact lowest token id among globally tied maxima."""

    if local_logits.ndim != 2 or local_logits.shape[0] != 1 or (
        local_logits.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 sampler requires one BF16 local logit row")
    local_vocab = local_logits.shape[1]
    if local_vocab <= 0 or local_vocab * 8 != vocab_size:
        raise ValueError("WS32 sampler vocabulary ownership drifted")
    finite = jnp.all(jnp.isfinite(local_logits))
    safe_logits = jnp.where(
        jnp.isfinite(local_logits),
        local_logits,
        jnp.asarray(-jnp.inf, dtype=local_logits.dtype),
    )
    local_index = jnp.argmax(safe_logits[0]).astype(jnp.int32)
    owner = lax.axis_index(expert_axis).astype(jnp.int32)
    global_index = owner * jnp.int32(local_vocab) + local_index
    candidate_score = jnp.where(
        finite,
        safe_logits[0, local_index],
        jnp.asarray(-jnp.inf, dtype=local_logits.dtype),
    )[None]
    candidate_index = jnp.where(
        finite, global_index, jnp.int32(vocab_size)
    )[None]
    with jax.named_scope("greenfield_ws32_sampling/expert_candidate_exchange"):
        scores = lax.all_gather(
            candidate_score, axis_name=expert_axis, axis=0, tiled=False
        )
        indices = lax.all_gather(
            candidate_index, axis_name=expert_axis, axis=0, tiled=False
        )
    winning_score = jnp.max(scores, axis=0)
    chosen = jnp.min(
        jnp.where(
            scores == winning_score[None, ...],
            indices,
            jnp.int32(vocab_size),
        ),
        axis=0,
    ).astype(jnp.int32)
    valid = jnp.all(indices < jnp.int32(vocab_size), axis=0)
    return Ws32GreedySampleResult(chosen, valid)


def ws32_final_sample_mapped(
    hidden_local: Any,
    final_norm_weight_local: Any,
    lm_head_local: Any,
    *,
    hidden_size: int,
    vocab_size: int,
    feature_axis: str = "feature",
    expert_axis: str = "expert",
    rms_norm_epsilon: float = 1e-5,
) -> Ws32GreedySampleResult:
    """Normalize, project, and sample without returning full vocabulary."""

    normalized = ws32_rms_norm_mapped(
        hidden_local,
        final_norm_weight_local,
        global_hidden_size=hidden_size,
        feature_axis=feature_axis,
        epsilon=rms_norm_epsilon,
    )
    logits = ws32_logits_mapped(
        normalized,
        lm_head_local,
        vocab_size=vocab_size,
        feature_axis=feature_axis,
    )
    return ws32_greedy_sample_mapped(
        logits, vocab_size=vocab_size, expert_axis=expert_axis
    )
