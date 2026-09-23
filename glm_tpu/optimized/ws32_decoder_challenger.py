"""The production greedy decode step (one token, all layers, one compiled shard_map program).

Every layer runs ``bf16_resident.transformer_layer_bf16``: resident BF16 non-routed tables, the
two-stage DSA selection, the frozen selected-KV sparse attention and the route-grouped FP8
routed experts (``ROUTED_PROJECTION`` tiles); the head is the greedy split final sample. The
release profile is the only one (S2d: the former ``Ws32PerfOptions`` accepted nothing else).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..greenfield.errors import PlanValidationError
from ..greenfield.kernels.ws32_io import ws32_embedding_mapped, ws32_split_final_sample_mapped
from ..greenfield.runtime import ws32_decoder as decoder
from .bf16_resident import bf16_weight_specs, transformer_layer_bf16
from .fp8_routed_experts import RoutedProjectionConfig

# Routed-expert projection tiles of the release decoder: 256 x 256 (RoutedProjectionConfig's own
# default is 512). The tile shape is part of the compiled program.
ROUTED_PROJECTION = RoutedProjectionConfig(output_tile=256, contraction_tile=256)


@dataclass(frozen=True, slots=True)
class Ws32ChallengerDecoderProgram:
    config: decoder.Ws32DecoderConfig
    execute: Any


def ws32_decode_challenger_mapped(
    token_ids: Any,
    state: decoder.Ws32DecoderState,
    weights: decoder.Ws32DecoderWeights,
    *,
    config: decoder.Ws32DecoderConfig,
    main_rope_table: Any | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    active: Any | None = None,
) -> decoder.Ws32DecodeStepResult:
    """One greedy decode step over every layer (mirror of the frozen ``_ws32_decode_impl``)."""

    decoder._validate_local_state(state, config)
    if active is not None and (active.shape != () or active.dtype != jnp.bool_):
        raise ValueError('decoder active mask must be a boolean scalar')
    if config.exact_dsa or config.strategy_nd_dense:
        raise PlanValidationError("WS32 challenger decoder supports the raw default path only")
    if config.host_main_rope_table != (main_rope_table is not None):
        raise ValueError("WS32 host main-rotary table flag/input presence drifted")
    main_rope_table_row = None
    if main_rope_table is not None:
        if main_rope_table.shape != config.main_rope_table_shape or main_rope_table.dtype != jnp.bfloat16:
            raise ValueError("WS32 main rotary table geometry drifted")
        with jax.named_scope("greenfield_ws32_main_rope_table_lookup"):
            main_rope_table_row = jnp.take(main_rope_table, state.position, axis=0, mode="clip")[0]
    if token_ids.shape != (1,) or token_ids.dtype != jnp.int32:
        raise ValueError("WS32 decoder input must be one int32 token")
    if len(weights.layers) != config.geometry.num_layers:
        raise ValueError("WS32 decoder weight layer count drifted")
    embedded = ws32_embedding_mapped(token_ids, weights.embedding_local, vocab_size=config.geometry.vocab_size)
    hidden_update = embedded.residual_local
    carried_residual = jnp.zeros_like(hidden_update)
    kv_cache = state.kv_cache_local
    index_cache = state.index_cache_local
    selected_positions = state.selected_positions
    selected_valid_counts = state.selected_valid_counts
    selected_scores = state.selected_scores
    health = state.contract_valid & embedded.contract_valid
    for layer_id, layer_weights in enumerate(weights.layers):
        index_slot = config.full_index_slot_by_layer[layer_id]
        result = transformer_layer_bf16(
            hidden_update, carried_residual, kv_cache[layer_id],
            index_cache[0 if index_slot is None else index_slot],
            selected_positions, selected_valid_counts, selected_scores,
            state.position, state.block_tables, state.context_lengths,
            layer_weights, health,
            indexer_kind=config.geometry.indexer_types[layer_id],
            mlp_kind=config.geometry.mlp_layer_types[layer_id], config=config,
            routed_projection=ROUTED_PROJECTION,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret, main_rope_table_row=main_rope_table_row,
        )
        hidden_update = result.output_local
        carried_residual = result.carried_residual_local
        # Apply stopping at the layer being updated. Keeping the original
        # whole cache alive until a final select defeats buffer donation and
        # requires another multi-GiB bank in a concurrent decode graph.
        layer_cache = result.cache_local
        if active is not None:
            layer_cache = jnp.where(active, layer_cache, kv_cache[layer_id])
        kv_cache = kv_cache.at[layer_id].set(layer_cache)
        if index_slot is not None:
            layer_index = result.index_cache_local
            if active is not None:
                layer_index = jnp.where(active, layer_index, index_cache[index_slot])
            index_cache = index_cache.at[index_slot].set(layer_index)
        selected_positions = result.selected_positions
        selected_valid_counts = result.selected_valid_counts
        selected_scores = result.selected_scores
        health = result.contract_valid
    sampled = ws32_split_final_sample_mapped(
        hidden_update, carried_residual, weights.final_norm_weight_local,
        weights.lm_head_local, hidden_size=config.geometry.hidden_size,
        vocab_size=config.geometry.vocab_size, rms_norm_epsilon=config.rms_norm_epsilon,
    )
    next_state = decoder.Ws32DecoderState(
        kv_cache, index_cache, selected_positions, selected_valid_counts, selected_scores,
        state.position + jnp.ones_like(state.position), state.block_tables,
        state.context_lengths + jnp.ones_like(state.context_lengths),
        health & sampled.contract_valid,
    )
    next_token = sampled.token_id
    if active is not None:
        # Only small frontier/selection arrays need final masking. Cache leaves
        # were already committed or retained one layer at a time above.
        next_state = next_state._replace(**{
            name:jnp.where(active,getattr(next_state,name),getattr(state,name))
            for name in state._fields if name not in ('kv_cache_local','index_cache_local')})
        next_token = jnp.where(active,next_token,token_ids)
    return decoder.Ws32DecodeStepResult(next_state, next_token, sampled.final_residual_local)


def build_ws32_challenger_decoder_program(
    mesh: Any,
    config: decoder.Ws32DecoderConfig,
    *,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    mask_finished: bool = False,
) -> Ws32ChallengerDecoderProgram:
    """Jitted shard_map with the frozen decoder's argument order.

    Arguments: ``(token, state, weights[, main_rope_table])`` -- the rotary table exactly when
    ``config.host_main_rope_table``. Same in/out specs as the frozen programs (weights: the
    resident BF16 tree). ``mask_finished=True`` appends one scalar boolean active argument and
    retains inactive cache rows at each layer's commit boundary.
    """

    import numpy as np

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 challenger decoder requires one exact expert8 x feature4 mesh")
    if type(mask_finished) is not bool:
        raise ValueError('mask_finished must be a static boolean')
    weight_specs = bf16_weight_specs(config)
    specs = (P(), decoder.ws32_decoder_state_specs(), weight_specs)
    if config.host_main_rope_table:
        specs += (P(),)
    if mask_finished:
        specs += (P(),)

    def body(tokens: Any, state: Any, weights: Any, *extra: Any) -> Any:
        expected = int(config.host_main_rope_table) + int(mask_finished)
        if len(extra) != expected:
            raise ValueError("challenger decoder input/config presence drifted")
        rope = extra[0] if config.host_main_rope_table else None
        with jax.named_scope("glm_perf_ws32_complete_decoder"):
            return ws32_decode_challenger_mapped(
                tokens, state, weights, config=config, main_rope_table=rope,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                active=extra[-1] if mask_finished else None,
            )

    execute = jax.jit(jax.shard_map(
        body, mesh=mesh, in_specs=specs,
        out_specs=decoder.ws32_decode_result_specs(), check_vma=False,
    ))
    return Ws32ChallengerDecoderProgram(config, execute)
