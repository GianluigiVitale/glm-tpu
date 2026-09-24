"""Owner-masked token embedding on the expert-8 x feature-4 mesh (decode rows and prompt rows)."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp


class EmbeddingResult(NamedTuple):
    residual_local: Any
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


def embed_tokens(
    token_ids: Any,
    embedding_local: Any,
    *,
    vocab_size: int,
    expert_axis: str = "expert",
) -> EmbeddingResult:
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
    return EmbeddingResult(residual, token_valid[None])


def prefill_embed_tokens(
    token_ids: Any, embedding_local: Any, valid_rows: Any, *, vocab_size: int
) -> EmbeddingResult:
    """One expert8 reduction for all live rows, with no vocabulary replication."""
    local_vocab, _ = _require_vocabulary_geometry(
        embedding_local, vocab_size=vocab_size
    )
    if (
        token_ids.ndim != 1
        or not 1 <= token_ids.shape[0] <= 128
        or token_ids.dtype != jnp.int32
    ):
        raise ValueError("batched embedding requires1..128 int32 token IDs")
    if valid_rows.shape != () or valid_rows.dtype != jnp.int32:
        raise ValueError("batched embedding live count must be int32 scalar")
    live = jnp.arange(token_ids.shape[0]) < jnp.clip(valid_rows, 0, token_ids.shape[0])
    token_valid = (token_ids >= 0) & (token_ids < vocab_size)
    start = lax.axis_index("expert").astype(jnp.int32) * jnp.int32(local_vocab)
    owns = live & token_valid & (token_ids >= start) & (token_ids < start + local_vocab)
    safe_tokens = jnp.clip(token_ids, 0, vocab_size - 1)
    local_ids = jnp.clip(safe_tokens - start, 0, local_vocab - 1)
    selected = jnp.where(owns[:, None], embedding_local[local_ids], 0)
    with jax.named_scope("greenfield_ws32_prefill_embedding/expert_owner_reduce"):
        hidden = lax.psum(selected, "expert")
    health = ~live | (token_valid & jnp.all(jnp.isfinite(hidden), axis=1))
    return EmbeddingResult(hidden, health)
