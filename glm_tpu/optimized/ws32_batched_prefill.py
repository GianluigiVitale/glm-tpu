"""Default-off layer-major prompt execution; the promoted decoder is unchanged.

One invocation embeds a live block once and visits each layer once. There is no
token scan of the decoder. A distinct state type owns unrepaired/repaired index
lifetimes and a monotone append frontier. Only the final block runs the head and
promotes repaired keys. This is an unpromoted numerical path, not §21 evidence.

Moved verbatim at S2f from ``glm_tpu/greenfield/runtime/ws32_batched_prefill.py`` (its production definitions; the
research remainder is archived at ``archive/research-20260922``).
"""
from __future__ import annotations

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from .errors import PlanValidationError
from .ws32_decoder import Ws32DecoderConfig, Ws32DecoderState, ws32_decoder_state_specs
from .ws32_io import Ws32EmbeddingResult, _require_vocabulary_geometry


class Ws32BatchedPrefillState(NamedTuple):
    decoder: Ws32DecoderState
    repaired_index_local: Any
    prompt_length: Any
    finished: Any


class Ws32BatchedPrefillResult(NamedTuple):
    state: Ws32BatchedPrefillState
    # -1 until a healthy final block; never a padded row's token.
    next_token: Any


def ws32_batched_prefill_state_specs() -> Ws32BatchedPrefillState:
    return Ws32BatchedPrefillState(
        ws32_decoder_state_specs(), P(None, None, "expert", None), P(), P()
    )


def _require_config(config: Ws32DecoderConfig) -> None:
    # Reuse raw final-layout weight views, never silently consume the promoted
    # convolution aliases or StrategyND overlay as if they were the raw kernels.
    if config.exact_dsa or config.strategy_nd_dense:
        raise PlanValidationError(
            "batched prefill requires raw weight config, no exact aliases/StrategyND"
        )
    if not config.host_main_rope_table or config.logical_page_size != 512:
        raise PlanValidationError("batched prefill requires host main RoPE and page512")


def ws32_prefill_embedding_mapped(
    token_ids: Any, embedding_local: Any, valid_rows: Any, *, vocab_size: int
) -> Ws32EmbeddingResult:
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
    return Ws32EmbeddingResult(hidden, health)


def _all_owners_healthy(local: Any) -> Any:
    """One scalar consensus per block, explicit feature4 then expert8 groups."""
    with jax.named_scope("greenfield_ws32_prefill_commit/health_consensus"):
        return lax.pmin(lax.pmin(local.astype(jnp.int32), "feature"), "expert") != 0


def finish_ws32_batched_prefill(
    result: Ws32BatchedPrefillResult,
) -> tuple[Ws32DecoderState, Any]:
    """One host boundary before serving: refuse incomplete/unhealthy prefill.

    Device-side all-owner consensus has already gated the atomic final commit.
    The returned decoder state owns repaired indices and retains one-row shapes;
    the next decode input is the returned greedy token, not the last prompt ID.
    """
    import numpy as np

    state = result.state
    if (
        not bool(np.asarray(state.finished))
        or not np.asarray(state.decoder.contract_valid).all()
        or not np.asarray(state.decoder.position == state.prompt_length).all()
        or not np.asarray(
            state.decoder.context_lengths == state.prompt_length + 1
        ).all()
        or not np.asarray(result.next_token >= 0).all()
    ):
        raise ValueError("batched prefill is not complete and healthy; decode refused")
    return state.decoder, result.next_token
