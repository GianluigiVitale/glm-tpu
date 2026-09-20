"""Challenger WS32 decode step: frozen bodies with the route-grouped MoE and the
candidate-set sampler swapped in (opt-in, CPU-proven, not TPU-admitted).

The frozen ``glm_tpu.greenfield.runtime.ws32_decoder._ws32_decode_impl`` and
``glm_tpu.greenfield.kernels.ws32_layer.ws32_transformer_layer_mapped`` have
no hook for an alternative MLP body, and editing them changes the frozen
``MODEL_SOURCE`` identity every sealed run declares.  This module therefore
mirrors those two compositions line for line (same fused norms, attention,
DSA, cache and health plumbing, same weight/state pytrees and shard specs) and
replaces only:

* the sparse-layer MLP: ``ws32_moe_pallas_from_routes_mapped`` ->
  ``ws32_moe_grouped_routes_mapped`` (``Ws32PerfOptions.grouped_routes``);
* the sampled head: ``ws32_split_nucleus_sample_mapped`` ->
  ``ws32_split_nucleus_sample_candidates_mapped``
  (``Ws32PerfOptions.sampler == "nucleus_candidates"``).

``tests/perf/test_decoder_challenger.py`` proves the challenger step equal to
the frozen sampled/greedy step on the forced 32-device CPU mesh (state, cache,
token and final residual) and records the collective/kernel census of both.
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
from .ws32_sampling_candidates import ws32_split_nucleus_sample_candidates_mapped

Sampler = Literal["greedy", "nucleus", "nucleus_candidates"]


@dataclass(frozen=True, slots=True)
class Ws32PerfOptions:
    """Which challenger bodies the decode program uses."""

    grouped_routes: bool = True
    routed_projection: RoutedProjectionConfig | None = None
    sampler: Sampler = "nucleus_candidates"
    candidates_per_shard: int = 256
    # D8: weights are a glm_tpu.perf.bf16_resident.Bf16DecoderWeights pytree
    # (non-routed tables pre-decoded to BF16); implies grouped routed experts.
    bf16_resident: bool = False
    lse_attention: bool = False
    # Kaggle global-max/FP32 numerator research path; trained admission pending.
    global_max_attention: bool = False
    dsa_two_stage: bool = False
    # Experimental/rejected on TPU: CPU parity does not establish finite TPU
    # behavior. See docs/perf/D4_D8_PROGRESS_20260919.md; leave disabled.
    fused_feature_reductions: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.grouped_routes, bool):
            raise ValueError("grouped_routes must be a boolean")
        if self.sampler not in ("greedy", "nucleus", "nucleus_candidates"):
            raise ValueError("sampler must be greedy, nucleus or nucleus_candidates")
        if type(self.candidates_per_shard) is not int or self.candidates_per_shard <= 0:
            raise ValueError("candidates_per_shard must be a positive integer")
        if not isinstance(self.bf16_resident, bool):
            raise ValueError("bf16_resident must be a boolean")
        if not isinstance(self.fused_feature_reductions, bool):
            raise ValueError("fused_feature_reductions must be a boolean")
        if self.fused_feature_reductions and not self.bf16_resident:
            raise ValueError("fused_feature_reductions currently requires bf16_resident")
        if not isinstance(self.dsa_two_stage, bool):
            raise ValueError("dsa_two_stage must be a boolean")
        if self.dsa_two_stage and not self.bf16_resident:
            raise ValueError("dsa_two_stage currently requires bf16_resident")
        if not isinstance(self.lse_attention, bool):
            raise ValueError("lse_attention must be a boolean")
        if self.lse_attention and not self.bf16_resident:
            raise ValueError("lse_attention currently requires bf16_resident")
        if type(self.global_max_attention) is not bool:
            raise ValueError('global_max_attention must be a boolean')
        if self.global_max_attention and (not self.bf16_resident or self.lse_attention):
            raise ValueError('global_max_attention requires bf16_resident and excludes lse_attention')
        if self.bf16_resident and not self.grouped_routes:
            raise ValueError("bf16_resident implies grouped routed experts")


@dataclass(frozen=True, slots=True)
class Ws32ChallengerDecoderProgram:
    config: decoder.Ws32DecoderConfig
    options: Ws32PerfOptions
    sampling: NucleusConfig | None
    execute: Any
    takes_uniform: bool


def ws32_mlp_challenger_mapped(
    post_attention_residual_local: Any,
    post_attention_norm_weight_local: Any,
    dense_weights: Ws32DenseWeights | Ws32StrategyNdDenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    *,
    mlp_kind: str,
    options: Ws32PerfOptions,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
    precomputed_normalized_local: Any | None = None,
    add_residual: bool = True,
) -> Ws32MlpResult:
    """``ws32_mlp_mapped`` with the sparse branch routed through the grouped kernel."""

    if mlp_kind == "dense" or not options.grouped_routes:
        return ws32_mlp_mapped(
            post_attention_residual_local, post_attention_norm_weight_local,
            dense_weights, moe_weights, mlp_kind=mlp_kind, contract=contract,
            block_shape=block_shape, rms_norm_epsilon=rms_norm_epsilon,
            linear_interpret=linear_interpret,
            precomputed_normalized_local=precomputed_normalized_local,
            add_residual=add_residual,
        )
    if mlp_kind != "sparse" or moe_weights is None or dense_weights is not None:
        raise ValueError("WS32 challenger MLP: sparse layers carry MoE weights only")
    if contract.stage_size != 8 or contract.hidden_size % 4:
        raise ValueError("WS32 challenger MLP requires the expert8/feature4 contract")
    local_hidden = contract.hidden_size // 4
    if post_attention_residual_local.shape != (1, local_hidden) or (
        post_attention_residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 challenger MLP residual must be one BF16 feature shard")
    if precomputed_normalized_local is None:
        raise ValueError("WS32 challenger MLP requires the fused-norm output")
    normalized = precomputed_normalized_local
    if normalized.shape != post_attention_residual_local.shape or normalized.dtype != jnp.bfloat16:
        raise ValueError("WS32 challenger MLP normalization geometry drifted")
    route_indices, route_weights = ws32_router_from_shards_mapped(
        normalized, moe_weights.router_weight_local, moe_weights.correction_bias_local,
        top_k=contract.top_k,
    )
    update = ws32_moe_grouped_routes_mapped(
        normalized, route_indices, route_weights, *moe_weights[2:],
        contract=contract, config=options.routed_projection, interpret=linear_interpret,
    )
    output = (
        (post_attention_residual_local + update).astype(jnp.bfloat16)
        if add_residual else update
    )
    return Ws32MlpResult(output, route_indices, route_weights)


def ws32_transformer_layer_challenger_mapped(
    hidden_update_local: Any,
    carried_residual_local: Any,
    cache_local: Any,
    index_cache_local: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    selected_scores: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    qkv_a_weights: Ws32QkvAWeights,
    attention_weights: Ws32AttentionWeights,
    dsa_weights: Ws32DsaWeights | None,
    post_attention_norm_weight_local: Any,
    dense_weights: Ws32DenseWeights | Ws32StrategyNdDenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    incoming_contract_valid: Any,
    *,
    options: Ws32PerfOptions,
    indexer_kind: str,
    mlp_kind: str,
    dsa_contract: DsaNumericalContract,
    attention_contract: MlaNumericalContract,
    moe_contract: GlmMoeNumericalContract,
    cache_layout: StageLocalKvLayout,
    block_shape: tuple[int, int],
    rms_norm_epsilon: float,
    sparse_attention_config: SparseMlaConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table_row: Any | None = None,
) -> Ws32TransformerLayerResult:
    """Mirror of ``ws32_transformer_layer_mapped`` with the challenger MLP."""

    if indexer_kind not in ("full", "shared") or mlp_kind not in ("dense", "sparse"):
        raise ValueError("WS32 challenger layer kinds drifted")
    if (dsa_weights is None) != (indexer_kind == "shared"):
        raise ValueError("WS32 full indexer alone must carry DSA weights")
    if (dense_weights is None) != (mlp_kind == "sparse") or (moe_weights is None) != (mlp_kind == "dense"):
        raise ValueError("WS32 challenger layer weight presence drifted")
    normalized_input, combined_residual = ws32_fused_add_rms_norm_mapped(
        hidden_update_local, carried_residual_local, qkv_a_weights.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size, epsilon=rms_norm_epsilon,
    )
    attention = ws32_attention_layer_mapped(
        combined_residual, cache_local, index_cache_local, selected_positions,
        selected_valid_counts, selected_scores, position, block_tables, context_lengths,
        qkv_a_weights, attention_weights, dsa_weights,
        exact_dsa_weights=None, dsa_contract=dsa_contract,
        attention_contract=attention_contract, cache_layout=cache_layout,
        block_shape=block_shape, sparse_attention_config=sparse_attention_config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret, main_rope_table_row=main_rope_table_row,
        precomputed_normalized_local=normalized_input, add_residual=False,
    )
    normalized_mlp, post_attention_residual = ws32_fused_add_rms_norm_mapped(
        attention.output_local, combined_residual, post_attention_norm_weight_local,
        global_hidden_size=moe_contract.hidden_size, epsilon=rms_norm_epsilon,
    )
    mlp = ws32_mlp_challenger_mapped(
        post_attention_residual, post_attention_norm_weight_local, dense_weights,
        moe_weights, mlp_kind=mlp_kind, options=options, contract=moe_contract,
        block_shape=block_shape, rms_norm_epsilon=rms_norm_epsilon,
        linear_interpret=linear_interpret, precomputed_normalized_local=normalized_mlp,
        add_residual=False,
    )
    return Ws32TransformerLayerResult(
        mlp.output_local, post_attention_residual, normalized_input,
        attention.cache_local, attention.index_cache_local,
        attention.selected_positions, attention.selected_valid_counts,
        attention.selected_scores, mlp.route_indices, mlp.route_weights,
        incoming_contract_valid & attention.contract_valid,
    )


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
    observe_layer: Any | None = None,
    observe_head: Any | None = None,
) -> decoder.Ws32DecodeStepResult:
    """Mirror of ``_ws32_decode_impl`` with optional diagnostic observations."""

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
        if options.bf16_resident:
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
                global_max_attention=options.global_max_attention,
                fused_feature_reductions=options.fused_feature_reductions,
            )
        else:
          result = ws32_transformer_layer_challenger_mapped(
            hidden_update, carried_residual, kv_cache[layer_id],
            index_cache[0 if index_slot is None else index_slot],
            selected_positions, selected_valid_counts, selected_scores,
            state.position, state.block_tables, state.context_lengths,
            layer_weights.qkv_a, layer_weights.attention, layer_weights.dsa,
            layer_weights.post_attention_norm_weight_local, layer_weights.dense,
            layer_weights.moe, health,
            options=options,
            indexer_kind=config.geometry.indexer_types[layer_id],
            mlp_kind=config.geometry.mlp_layer_types[layer_id],
            dsa_contract=config.dsa_contract, attention_contract=config.attention_contract,
            moe_contract=config.moe_contract, cache_layout=config.cache_layout,
            block_shape=config.geometry.fp8_block_shape,
            rms_norm_epsilon=config.rms_norm_epsilon, sparse_attention_config=sparse_config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret, main_rope_table_row=main_rope_table_row,
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
        if observe_layer is not None:
            observe_layer(layer_id, result.normalized_input_local, hidden_update,
                          carried_residual, selected_positions, selected_valid_counts, selected_scores)
    if observe_head is not None:
        observe_head(hidden_update, carried_residual)
    if options.sampler == "greedy":
        sample_head = ws32_split_final_sample_mapped
    elif options.sampler == "nucleus":
        sample_head = partial(ws32_split_nucleus_sample_mapped, uniform=uniform, config=sampling)
    else:
        sample_head = partial(
            ws32_split_nucleus_sample_candidates_mapped, uniform=uniform, config=sampling,
            candidates_per_shard=options.candidates_per_shard,
        )
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
    if options.bf16_resident:
        from .bf16_resident import bf16_weight_specs

        weight_specs = bf16_weight_specs(config)
    else:
        weight_specs = decoder.ws32_decoder_weight_specs(config)
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
