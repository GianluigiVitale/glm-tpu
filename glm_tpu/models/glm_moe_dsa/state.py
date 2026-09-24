"""Default-off layer-major prompt execution; the promoted decoder is unchanged.

One invocation embeds a live block once and visits each layer once. There is no
token scan of the decoder. A distinct state type owns unrepaired/repaired index
lifetimes and a monotone append frontier. Only the final block runs the head and
promotes repaired keys. This is an unpromoted numerical path, not §21 evidence.

Moved verbatim at S2f out of the research package (its production definitions; the research
remainder, and the module these definitions came from, are at ``archive/research-20260922``).
"""
from __future__ import annotations

from typing import Any, NamedTuple
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from glm_tpu.exceptions import PlanValidationError
from glm_tpu.config.cache import CacheConfig


class BatchedPrefillState(NamedTuple):
    decoder: DecoderState
    repaired_index_local: Any
    prompt_length: Any
    finished: Any


class BatchedPrefillResult(NamedTuple):
    state: BatchedPrefillState
    # -1 until a healthy final block; never a padded row's token.
    next_token: Any


def batched_prefill_state_specs() -> BatchedPrefillState:
    return BatchedPrefillState(
        decoder_state_specs(), P(None, None, "expert", None), P(), P()
    )


def _require_config(config: CacheConfig) -> None:
    # Reuse raw final-layout weight views, never silently consume the promoted
    # convolution aliases or StrategyND overlay as if they were the raw kernels.
    if config.exact_dsa or config.strategy_nd_dense:
        raise PlanValidationError(
            "batched prefill requires raw weight config, no exact aliases/StrategyND"
        )
    if not config.host_main_rope_table or config.logical_page_size != 512:
        raise PlanValidationError("batched prefill requires host main RoPE and page512")


def finish_batched_prefill(
    result: BatchedPrefillResult,
) -> tuple[DecoderState, Any]:
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


class DecoderState(NamedTuple):
    kv_cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    position: Any
    block_tables: Any
    context_lengths: Any
    contract_valid: Any


class DecodeStepResult(NamedTuple):
    state: DecoderState
    next_token: Any
    final_residual_local: Any


def decoder_state_specs() -> DecoderState:
    return DecoderState(
        P(None, None, "expert", None),
        P(None, None, "expert", None),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
    )


def decode_result_specs() -> DecodeStepResult:
    return DecodeStepResult(
        decoder_state_specs(), P(), P(None, "feature")
    )


def _validate_local_state(
    state: DecoderState,
    config: CacheConfig,
) -> None:
    geometry = config.geometry
    expected_kv = (
        geometry.num_layers,
        config.page_count,
        config.local_rows_per_page,
        config.packed_cache_width,
    )
    expected_index = (
        len(config.full_index_slots),
        config.page_count,
        config.local_rows_per_page,
        geometry.dsa_indexer_head_dim,
    )
    if state.kv_cache_local.shape != expected_kv or (
        state.kv_cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 local KV state geometry drifted")
    if state.index_cache_local.shape != expected_index or (
        state.index_cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 local index state geometry drifted")
    if state.selected_positions.shape != (1, geometry.dsa_top_k) or (
        state.selected_positions.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected-position state geometry drifted")
    if state.selected_valid_counts.shape != (1,) or (
        state.selected_valid_counts.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected-count state geometry drifted")
    if state.selected_scores.shape != (1, geometry.dsa_top_k) or (
        state.selected_scores.dtype != jnp.float32
    ):
        raise ValueError("WS32 selected-score state geometry drifted")
    if state.position.shape != (1,) or state.position.dtype != jnp.int32:
        raise ValueError("WS32 position state geometry drifted")
    if state.block_tables.shape != (1, config.page_count) or (
        state.block_tables.dtype != jnp.int32
    ):
        raise ValueError("WS32 block-table state geometry drifted")
    if state.context_lengths.shape != (1,) or (
        state.context_lengths.dtype != jnp.int32
    ):
        raise ValueError("WS32 context-length state geometry drifted")
    if state.contract_valid.shape != (1,) or (
        state.contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("WS32 decoder health must be one boolean row")
