"""Retained ordinary greedy decoder, extracted from trained source 5ff7b01e.

D1 grouped experts, resident BF16 tables, D10 selection and frozen selected-KV
attention. Experimental decode variants are not release options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..greenfield.errors import PlanValidationError
from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..greenfield.kernels.reference.dsa import DsaNumericalContract
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from ..greenfield.kernels.ws32 import (
    ws32_fused_add_rms_norm_mapped,
    ws32_router_from_shards_mapped,
)
from ..greenfield.kernels.ws32_io import ws32_embedding_mapped, ws32_split_final_sample_mapped
from ..greenfield.kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MlpResult,
    Ws32MoeWeights,
    Ws32QkvAWeights,
    Ws32StrategyNdDenseWeights,
    Ws32TransformerLayerResult,
    ws32_attention_layer_mapped,
    ws32_mlp_mapped,
)
from ..greenfield.kernels.ws32_sampling import NucleusConfig, ws32_split_nucleus_sample_mapped
from ..greenfield.runtime import ws32_decoder as decoder
from .fp8_routed_experts import RoutedProjectionConfig, ws32_moe_grouped_routes_mapped

Sampler = Literal["greedy", "nucleus", "nucleus_candidates"]


@dataclass(frozen=True, slots=True)
class Ws32PerfOptions:
    """Fixed retained ordinary profile; experiments are not release options."""
    grouped_routes: bool = True
    routed_projection: RoutedProjectionConfig = field(default_factory=lambda:
        RoutedProjectionConfig(output_tile=256, contraction_tile=256))
    sampler: str = 'greedy'
    bf16_resident: bool = True
    lse_attention: bool = False
    dsa_two_stage: bool = True
    fused_feature_reductions: bool = False

    def __post_init__(self):
        if (self.grouped_routes is not True or self.bf16_resident is not True
                or self.sampler != 'greedy' or self.lse_attention is not False
                or self.dsa_two_stage is not True or self.fused_feature_reductions is not False
                or self.routed_projection != RoutedProjectionConfig(output_tile=256, contraction_tile=256)):
            raise ValueError('optimized release requires the retained ordinary greedy profile')


@dataclass(frozen=True, slots=True)
class Ws32ChallengerDecoderProgram:
    config: decoder.Ws32DecoderConfig
    options: Ws32PerfOptions
    sampling: NucleusConfig | None
    execute: Any
    takes_uniform: bool






def ws32_decode_challenger_mapped(
    token_ids: Any,
    state: decoder.Ws32DecoderState,
    weights: decoder.Ws32DecoderWeights,
    *,
    config: decoder.Ws32DecoderConfig,
    options: Ws32PerfOptions,
    sampling: NucleusConfig | None = None,
    uniform: Any | None = None,
    main_rope_table: Any | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> decoder.Ws32DecodeStepResult:
    """Mirror of ``_ws32_decode_impl`` (no observers) with challenger bodies."""

    decoder._validate_local_state(state, config)
    if config.exact_dsa or config.strategy_nd_dense:
        raise PlanValidationError("WS32 challenger decoder supports the raw default path only")
    if config.host_main_rope_table != (main_rope_table is not None):
        raise ValueError("WS32 host main-rotary table flag/input presence drifted")
    if (options.sampler == "greedy") != (sampling is None) or (sampling is None) != (uniform is None):
        raise ValueError("sampled challenger heads need NucleusConfig and one uniform")
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
    sparse_config = SparseMlaConfig(segment_block=config.sparse_segment_block)
    for layer_id, layer_weights in enumerate(weights.layers):
        index_slot = config.full_index_slot_by_layer[layer_id]
        from .bf16_resident import transformer_layer_bf16

        result = transformer_layer_bf16(
            hidden_update, carried_residual, kv_cache[layer_id],
            index_cache[0 if index_slot is None else index_slot],
            selected_positions, selected_valid_counts, selected_scores,
            state.position, state.block_tables, state.context_lengths,
            layer_weights, health,
            indexer_kind=config.geometry.indexer_types[layer_id],
            mlp_kind=config.geometry.mlp_layer_types[layer_id], config=config,
            routed_projection=options.routed_projection,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret, main_rope_table_row=main_rope_table_row,
            lse_attention=options.lse_attention, dsa_two_stage=options.dsa_two_stage,
            fused_feature_reductions=options.fused_feature_reductions,
        )
        hidden_update = result.output_local
        carried_residual = result.carried_residual_local
        kv_cache = kv_cache.at[layer_id].set(result.cache_local)
        if index_slot is not None:
            index_cache = index_cache.at[index_slot].set(result.index_cache_local)
        selected_positions = result.selected_positions
        selected_valid_counts = result.selected_valid_counts
        selected_scores = result.selected_scores
        health = result.contract_valid
    sample_head = ws32_split_final_sample_mapped
    sampled = sample_head(
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
    return decoder.Ws32DecodeStepResult(next_state, sampled.token_id, sampled.final_residual_local)


def build_ws32_challenger_decoder_program(
    mesh: Any,
    config: decoder.Ws32DecoderConfig,
    *,
    options: Ws32PerfOptions = Ws32PerfOptions(),
    sampling: NucleusConfig | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32ChallengerDecoderProgram:
    """Jitted shard_map with the frozen sampled decoder's argument order.

    Arguments: ``(token, state, weights[, main_rope_table][, uniform])`` --
    the rotary table exactly when ``config.host_main_rope_table`` and the FP32
    uniform exactly when the sampler is not greedy.  Same in/out specs as the
    frozen programs, so the frozen state/weight pytrees are consumed as is.
    """

    import numpy as np

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 challenger decoder requires one exact expert8 x feature4 mesh")
    if (options.sampler == "greedy") != (sampling is None):
        raise ValueError("NucleusConfig is required exactly for sampled heads")
    if sampling is not None and not isinstance(sampling, NucleusConfig):
        raise ValueError("explicit NucleusConfig required for sampled requests")
    takes_uniform = options.sampler != "greedy"
    from .bf16_resident import bf16_weight_specs

    weight_specs = bf16_weight_specs(config)
    specs = (P(), decoder.ws32_decoder_state_specs(), weight_specs)
    if config.host_main_rope_table:
        specs += (P(),)
    if takes_uniform:
        specs += (P(),)

    def body(tokens: Any, state: Any, weights: Any, *extra: Any) -> Any:
        expected = int(config.host_main_rope_table) + int(takes_uniform)
        if len(extra) != expected:
            raise ValueError("challenger decoder input/config presence drifted")
        rope = extra[0] if config.host_main_rope_table else None
        uniform = extra[-1] if takes_uniform else None
        with jax.named_scope("glm_perf_ws32_complete_decoder"):
            return ws32_decode_challenger_mapped(
                tokens, state, weights, config=config, options=options,
                sampling=sampling, uniform=uniform, main_rope_table=rope,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
            )

    execute = jax.jit(jax.shard_map(
        body, mesh=mesh, in_specs=specs,
        out_specs=decoder.ws32_decode_result_specs(), check_vma=False,
    ))
    return Ws32ChallengerDecoderProgram(config, options, sampling, execute, takes_uniform)
