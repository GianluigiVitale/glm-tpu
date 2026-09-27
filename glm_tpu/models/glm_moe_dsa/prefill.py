"""The production prefill block and its program builder (B128 / B114 layer-major blocks).

One call embeds a block of prompt rows once and visits every layer once (no token scan of the
decoder): each layer runs the four rolled 32-row attention/DSA prefixes and one MLP suffix over the
whole block (``decoder_layer.prefill_layer_window``). The block commits its proposed caches and
frontier only when every owner is healthy; the final block also runs the greedy head and promotes
the repaired index keys. The program takes the resident BF16 weight tree
(``weights.bf16_weight_specs``).

The admitted profile is the only one (S2d fold): MLP window with rolled prefixes, routed-expert
panels, the canonical dense placement and the one-pass DSA selector. The original
(``greenfield/runtime/ws32_batched_prefill.py``) is archived at ``archive/research-20260922``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from glm_tpu.exceptions import PlanValidationError
from glm_tpu.kernels.sparse_mla.kernel import SparseMlaConfig
from glm_tpu.layers.sampler import split_final_sample
from glm_tpu.models.glm_moe_dsa.state import (
    BatchedPrefillResult,
    BatchedPrefillState,
    require_config,
    batched_prefill_state_specs,
    DecoderState,
    validate_local_state,
)
from glm_tpu.layers.embed import prefill_embed_tokens
from glm_tpu.config.cache import CacheConfig
from glm_tpu.models.glm_moe_dsa.weights import Bf16DecoderWeights, bf16_weight_specs
from glm_tpu.models.glm_moe_dsa.decoder_layer import prefill_layer_window


@dataclass(frozen=True, slots=True)
class PrefillProgram:
    config: CacheConfig
    block_rows: int
    execute: Any


def batched_prefill(
    token_ids: Any,
    valid_rows: Any,
    state: BatchedPrefillState,
    weights: Bf16DecoderWeights,
    materialized_wk: tuple[Any, ...],
    main_rope_table: Any,
    *,
    config: CacheConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> BatchedPrefillResult:
    """Propose one complete layer-major block; commit only all-owner success.

    The position in the carried state is the sole append offset. Intermediate
    states cannot be passed back after repair promotion: ``finished`` refuses
    further prefill. Failed proposals leave both cache trees/frontiers intact
    and latch false health on all owners. Prefix authenticity belongs to the
    fresh allocator or authenticated restore, not a guessed nonzero position.
    """
    require_config(config)
    validate_local_state(state.decoder, config)
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("batched prefill requires expert8/feature4")
    if token_ids.ndim != 1 or not 1 <= token_ids.shape[0] <= 128 or token_ids.dtype != jnp.int32:
        raise ValueError("batched prefill row count/dtype exceeds selected mode")
    if token_ids.shape[0] not in (114, 128):
        raise ValueError("canonical dense requires physical B114/B128")
    for value, dtype in (
        (valid_rows, jnp.int32),
        (state.prompt_length, jnp.int32),
        (state.finished, jnp.bool_),
    ):
        if value.shape != () or value.dtype != dtype:
            raise ValueError("batched prefill scalar metadata geometry drifted")
    if (
        state.repaired_index_local.shape != state.decoder.index_cache_local.shape
        or state.repaired_index_local.dtype != jnp.bfloat16
    ):
        raise ValueError("batched repaired buffer geometry drifted")
    if main_rope_table.shape != config.main_rope_table_shape or main_rope_table.dtype != jnp.bfloat16:
        raise ValueError("batched main rotary table geometry drifted")
    if len(weights.layers) != config.geometry.num_layers or len(materialized_wk) != len(config.full_index_slots):
        raise ValueError("batched layer/repair owner cardinality drifted")
    for layer_id, layer in enumerate(weights.layers):
        full = config.full_index_slot_by_layer[layer_id] is not None
        dense = config.geometry.mlp_layer_types[layer_id] == "dense"
        if (layer.dsa is not None) != full or (layer.dense is not None) != dense or (layer.moe is not None) == dense:
            raise ValueError("batched weights disagree with model layer schedule")
    for wk in materialized_wk:
        if wk.shape != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size) or wk.dtype != jnp.float32:
            raise ValueError("batched repair requires completed full FP32 wk per producer")

    rows = token_ids.shape[0]
    decoder = state.decoder
    offset = decoder.position[0]
    count = jnp.clip(valid_rows, 0, rows)
    # Avoid signed overflow even for adversarial caller metadata. Clipping is
    # only for safe execution; the original values still determine refusal.
    start = jnp.clip(offset, 0, config.context_capacity - 1)
    end = start + jnp.minimum(count, config.context_capacity - start)
    span_valid = (
        (valid_rows > 0)
        & (valid_rows <= rows)
        & (offset >= 0)
        & (offset == start)
        & (state.prompt_length > 0)
        & (state.prompt_length < config.context_capacity)
        & (start <= state.prompt_length - count)
        & (decoder.context_lengths[0] == start + 1)
        & ~state.finished
        & jnp.all(decoder.contract_valid)
    )
    live = jnp.arange(rows, dtype=jnp.int32) < count
    positions = jnp.minimum(start + jnp.arange(rows, dtype=jnp.int32), config.context_capacity - 1)
    rope = jnp.take(main_rope_table, positions, axis=0, mode="clip")
    embedded = prefill_embed_tokens(
        token_ids,
        weights.embedding_local,
        valid_rows,
        vocab_size=config.geometry.vocab_size,
    )
    update = embedded.residual_local
    residual = jnp.zeros_like(update)
    health = embedded.contract_valid & span_valid
    kv = decoder.kv_cache_local
    unrepaired = decoder.index_cache_local
    repaired = state.repaired_index_local
    selected = jnp.full((rows, config.geometry.dsa_top_k), -1, jnp.int32)
    counts = jnp.zeros((rows,), jnp.int32)
    scores = jnp.full(selected.shape, -jnp.inf, jnp.float32)
    sparse = SparseMlaConfig(segment_block=config.sparse_segment_block)
    for layer_id, layer in enumerate(weights.layers):
        slot = config.full_index_slot_by_layer[layer_id]
        # Shared layers never use/write this placeholder index buffer. Their
        # selections come from the actual preceding producer; KV is always OWN.
        source_slot = 0 if slot is None else slot
        with jax.named_scope(f"batched_prefill/layer_{layer_id}"):
            result = prefill_layer_window(
                update,
                residual,
                kv[layer_id],
                unrepaired[source_slot],
                repaired[source_slot],
                selected,
                counts,
                scores,
                offset,
                valid_rows,
                decoder.block_tables,
                layer.qkv_a,
                layer.attention,
                layer.dsa,
                None if slot is None else materialized_wk[slot],
                layer.post_attention_norm_weight_local,
                layer.dense,
                layer.moe,
                health,
                main_rope_table_rows=rope,
                dsa_contract=config.dsa_contract,
                attention_contract=config.attention_contract,
                moe_contract=config.moe_contract,
                rms_norm_epsilon=config.rms_norm_epsilon,
                sparse_attention_config=sparse,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
            )
        kv = kv.at[layer_id].set(result.cache_local)
        if slot is not None:
            unrepaired = unrepaired.at[slot].set(result.unrepaired_index_cache)
            repaired = repaired.at[slot].set(result.repaired_index_cache)
        update, residual = result.output_local, result.carried_residual_local
        selected, counts, scores = (
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
        )
        health = health & result.contract_valid

    last = jnp.maximum(count - 1, 0)
    # Branch only on replicated scheduling metadata, NOT potentially differing
    # per-owner health: the head contains collectives and must execute uniformly.
    final = (end == state.prompt_length) & ~state.finished

    def sample(_: Any) -> tuple[Any, Any]:
        head = split_final_sample(
            lax.dynamic_slice_in_dim(update, last, 1),
            lax.dynamic_slice_in_dim(residual, last, 1),
            weights.final_norm_weight_local,
            weights.lm_head_local,
            hidden_size=config.geometry.hidden_size,
            vocab_size=config.geometry.vocab_size,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        return head.token_id, jnp.all(head.contract_valid)

    # final is replicated metadata; no head compute for intermediate blocks.
    token, head_health = lax.cond(
        final,
        sample,
        lambda _: (jnp.full((1,), -1, jnp.int32), jnp.bool_(True)),
        operand=None,
    )
    healthy = _all_owners_healthy(span_valid & jnp.all(~live | health) & head_health)

    def commit(_: Any) -> BatchedPrefillState:
        active_index = lax.cond(final, lambda _: repaired, lambda _: unrepaired, operand=None)
        next_decoder = DecoderState(
            kv,
            active_index,
            lax.dynamic_slice_in_dim(selected, last, 1),
            lax.dynamic_slice_in_dim(counts, last, 1),
            lax.dynamic_slice_in_dim(scores, last, 1),
            end[None],
            decoder.block_tables,
            (end + 1)[None],
            healthy[None],
        )
        return BatchedPrefillState(next_decoder, repaired, state.prompt_length, final)

    def refuse(_: Any) -> BatchedPrefillState:
        return state._replace(decoder=decoder._replace(contract_valid=jnp.zeros((1,), jnp.bool_)))

    return BatchedPrefillResult(
        lax.cond(healthy, commit, refuse, operand=None),
        jnp.where(healthy & final, token, -1),
    )


def build_prefill_program(
    mesh: Any,
    config: CacheConfig,
    *,
    block_rows: int,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> PrefillProgram:
    """Build the prefill program for one physical block size (B128, or the B114 tail)."""
    import numpy as np

    require_config(config)
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or not 1 <= block_rows <= 128:
        raise PlanValidationError("batched prefill block rows exceed selected mode")
    if block_rows not in (114, 128):
        raise PlanValidationError("canonical dense requires physical B114/B128")
    if tuple(mesh.axis_names) != ("expert", "feature") or np.asarray(mesh.devices).shape != (8, 4):
        raise PlanValidationError("batched prefill requires exact expert8/feature4 mesh")

    def body(
        tokens: Any,
        count: Any,
        state: BatchedPrefillState,
        weights: Bf16DecoderWeights,
        wk: tuple[Any, ...],
        rope: Any,
    ) -> BatchedPrefillResult:
        if tokens.shape != (block_rows,):
            raise ValueError("batched prefill static row count drifted")
        return batched_prefill(
            tokens,
            count,
            state,
            weights,
            wk,
            rope,
            config=config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
        )

    specs = batched_prefill_state_specs()
    execute = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                specs,
                bf16_weight_specs(config),
                tuple(P() for _ in config.full_index_slots),
                P(),
            ),
            out_specs=BatchedPrefillResult(specs, P()),
            check_vma=False,
        )
    )
    return PrefillProgram(config, block_rows, execute)


def _all_owners_healthy(local: Any) -> Any:
    """One scalar consensus per block, explicit feature4 then expert8 groups."""
    with jax.named_scope("prefill_commit/health_consensus"):
        return lax.pmin(lax.pmin(local.astype(jnp.int32), "feature"), "expert") != 0
