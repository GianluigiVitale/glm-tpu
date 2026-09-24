"""The production greedy decode step (one token, all layers, one compiled shard_map program).

Every layer runs ``bf16_resident.transformer_layer_bf16``: resident BF16 non-routed tables, the
two-stage DSA selection, the frozen selected-KV sparse attention and the route-grouped FP8
routed experts (``ROUTED_PROJECTION`` tiles); the head is the greedy split final sample. The
release profile is the only one (S2d: the former ``Ws32PerfOptions`` accepted nothing else).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import lax

from glm_tpu.exceptions import PlanValidationError
from glm_tpu.layers.embed import embed_tokens
from glm_tpu.layers.sampler import split_final_sample
from glm_tpu.models.glm_moe_dsa.state import (
    DecoderState,
    DecodeStepResult,
    decode_result_specs,
    _validate_local_state,
    decoder_state_specs,
)
from glm_tpu.config import cache
from glm_tpu.models.glm_moe_dsa.weights import Fp8DecoderWeights, bf16_weight_specs
from glm_tpu.models.glm_moe_dsa.decoder_layer import transformer_layer_bf16
from glm_tpu.kernels.fp8_grouped_matmul.kernel import RoutedProjectionConfig


# Routed-expert projection tiles of the release decoder: 256 x 256 (RoutedProjectionConfig's own
# default is 512). The tile shape is part of the compiled program.
ROUTED_PROJECTION = RoutedProjectionConfig(output_tile=256, contraction_tile=256)


@dataclass(frozen=True, slots=True)
class DecoderProgram:
    config: cache.CacheConfig
    execute: Any


def decode_step(
    token_ids: Any,
    state: DecoderState,
    weights: Fp8DecoderWeights,
    *,
    config: cache.CacheConfig,
    main_rope_table: Any | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    active: Any | None = None,
) -> DecodeStepResult:
    """One greedy decode step over every layer (mirror of the frozen ``_ws32_decode_impl``)."""

    _validate_local_state(state, config)
    if active is not None and (active.shape != () or active.dtype != jnp.bool_):
        raise ValueError("decoder active mask must be a boolean scalar")
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
    embedded = embed_tokens(token_ids, weights.embedding_local, vocab_size=config.geometry.vocab_size)
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
            hidden_update,
            carried_residual,
            kv_cache[layer_id],
            index_cache[0 if index_slot is None else index_slot],
            selected_positions,
            selected_valid_counts,
            selected_scores,
            state.position,
            state.block_tables,
            state.context_lengths,
            layer_weights,
            health,
            indexer_kind=config.geometry.indexer_types[layer_id],
            mlp_kind=config.geometry.mlp_layer_types[layer_id],
            config=config,
            routed_projection=ROUTED_PROJECTION,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
            main_rope_table_row=main_rope_table_row,
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
    sampled = split_final_sample(
        hidden_update,
        carried_residual,
        weights.final_norm_weight_local,
        weights.lm_head_local,
        hidden_size=config.geometry.hidden_size,
        vocab_size=config.geometry.vocab_size,
        rms_norm_epsilon=config.rms_norm_epsilon,
    )
    next_state = DecoderState(
        kv_cache,
        index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        state.position + jnp.ones_like(state.position),
        state.block_tables,
        state.context_lengths + jnp.ones_like(state.context_lengths),
        health & sampled.contract_valid,
    )
    next_token = sampled.token_id
    if active is not None:
        # Only small frontier/selection arrays need final masking. Cache leaves
        # were already committed or retained one layer at a time above.
        next_state = next_state._replace(
            **{
                name: jnp.where(active, getattr(next_state, name), getattr(state, name))
                for name in state._fields
                if name not in ("kv_cache_local", "index_cache_local")
            }
        )
        next_token = jnp.where(active, next_token, token_ids)
    return DecodeStepResult(next_state, next_token, sampled.final_residual_local)


def build_decoder_program(
    mesh: Any,
    config: cache.CacheConfig,
    *,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    mask_finished: bool = False,
) -> DecoderProgram:
    """Jitted shard_map with the frozen decoder's argument order.

    Arguments: ``(token, state, weights[, main_rope_table])`` -- the rotary table exactly when
    ``config.host_main_rope_table``. Same in/out specs as the frozen programs (weights: the
    resident BF16 tree). ``mask_finished=True`` appends one scalar boolean active argument and
    retains inactive cache rows at each layer's commit boundary.
    """

    import numpy as np

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(np.asarray(mesh.devices, dtype=object).shape) != (8, 4):
        raise PlanValidationError("WS32 challenger decoder requires one exact expert8 x feature4 mesh")
    if type(mask_finished) is not bool:
        raise ValueError("mask_finished must be a static boolean")
    weight_specs = bf16_weight_specs(config)
    specs = (P(), decoder_state_specs(), weight_specs)
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
            return decode_step(
                tokens,
                state,
                weights,
                config=config,
                main_rope_table=rope,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                active=extra[-1] if mask_finished else None,
            )

    execute = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=specs,
            out_specs=decode_result_specs(),
            check_vma=False,
        )
    )
    return DecoderProgram(config, execute)


class BatchedDecodeResult(NamedTuple):
    state: Any
    next_token: Any
    metadata: Any  # [conversation, token/health/position/context-length]


def build_batched_decoder_program(
    mesh, config, *, batch_size=8, donate_state=True, sparse_attention_interpret=False, linear_interpret=False
):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("batched decode requires one to eight conversations")
    if config.host_main_rope_table is not True:
        raise ValueError("batched decode requires the shared host rotary table")
    single = build_decoder_program(
        mesh,
        config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        mask_finished=True,
    )
    mapped = jax.vmap(single.execute, in_axes=(0, 0, None, None, 0))

    def pack(tokens, health, position, lengths):
        healthy = jax.lax.pmin(health.astype(jnp.int32), ("expert", "feature"))
        return jnp.stack((tokens[:, 0], healthy[:, 0], position[:, 0], lengths[:, 0]), axis=1)

    pack = jax.shard_map(pack, mesh=mesh, in_specs=(P(),) * 4, out_specs=P(), check_vma=False)

    def execute(tokens, state, weights, rope, active):
        if tokens.shape != (batch_size, 1) or tokens.dtype != jnp.int32:
            raise ValueError("batched decoder tokens must be int32[batch,1]")
        if active.shape != (batch_size,) or active.dtype != jnp.bool_:
            raise ValueError("batched decoder active mask must be bool[batch]")
        out = mapped(tokens, state, weights, rope, active)
        next_state, next_token = out.state, out.next_token
        metadata = pack(next_token, next_state.contract_valid, next_state.position, next_state.context_lengths)
        return BatchedDecodeResult(next_state, next_token, metadata)

    return jax.jit(execute, donate_argnums=(1,) if donate_state else ())


class PackedDecodeResult(NamedTuple):
    decoded: Any
    # int32 [token, all-owner health, next position, next context length]
    metadata: Any


@dataclass(frozen=True, slots=True)
class PackedDecoderProgram:
    execute: Any


def pack_decode_metadata(token, health, position, lengths, draw_valid):
    if (
        token.shape != (1,)
        or token.dtype != jnp.int32
        or health.shape != (1,)
        or health.dtype != jnp.bool_
        or position.shape != (1,)
        or position.dtype != jnp.int32
        or lengths.shape != (1,)
        or lengths.dtype != jnp.int32
        or draw_valid.shape != ()
        or draw_valid.dtype != jnp.bool_
    ):
        raise ValueError("packed decode metadata geometry/dtype drifted")
    valid = lax.pmin((health[0] & draw_valid).astype(jnp.int32), ("expert", "feature"))
    return jnp.stack((token[0], valid, position[0], lengths[0]))


def build_packed_decoder_program(mesh, config, *, sparse_attention_interpret=False, linear_interpret=False):
    """The decode step's arguments ``(token, state, weights[, main_rope_table])``; no state donation
    or speculative extra model step is introduced (the runtime applies donation above 8,192 slots)."""
    base = build_decoder_program(
        mesh, config, sparse_attention_interpret=sparse_attention_interpret, linear_interpret=linear_interpret
    )
    pack = jax.shard_map(pack_decode_metadata, mesh=mesh, in_specs=(P(),) * 5, out_specs=P(), check_vma=False)

    def execute(token, state, weights, *extra):
        expected = int(config.host_main_rope_table)
        if len(extra) != expected:
            raise ValueError("packed decoder arguments disagree with rotary/sampling flags")
        rope = extra[:1] if config.host_main_rope_table else ()
        # The greedy step has no draw to admit; the metadata still ANDs this constant into the
        # all-owner health (part of the compiled program).
        valid = jnp.bool_(True)
        result = base.execute(token, state, weights, *rope)
        metadata = pack(
            result.next_token, result.state.contract_valid, result.state.position, result.state.context_lengths, valid
        )
        return PackedDecodeResult(result, metadata)

    return PackedDecoderProgram(jax.jit(execute))
