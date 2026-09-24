"""Prefill layer of the production engine: the attention/DSA prefix and the MLP suffix.

``ws32_prefill_transformer_layer_mapped`` is the per-tile prefix of one layer (fused add +
RMSNorm, resident BF16 attention preparation, the resident/one-pass DSA indexer, the owner-local
LSE attention ``prefill_attention.prefill_index_share_lse_mapped`` and the post-attention norm);
``ws32_prefill_mlp_mapped`` is the suffix the layer window runs once per block (resident dense MLP,
or the frozen router and the routed-expert panels). Bodies of
``greenfield/kernels/ws32_prefill_layer.py`` with the admitted profile hard-wired (S2d fold); the
frozen module stays untouched as the numerical oracle of the tests.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.kernels.sparse_mla.kernel import SparseMlaConfig
from glm_tpu.layers.contracts import MlaNumericalContract, DsaNumericalContract, GlmMoeNumericalContract, StageLocalKvLayout
from glm_tpu.layers.moe.router import router_from_shards, prefill_router
from glm_tpu.layers.norm import sharded_fused_add_rms_norm
from glm_tpu.models.glm_moe_dsa.weights import (
    Bf16AttentionWeights, Bf16DenseWeights, Bf16DsaWeights, Bf16MoeWeights, Bf16QkvAWeights, Bf16LayerWeights)
from glm_tpu.layers.attention.mla import prefill_index_share_lse, prefill_prepare_attention, index_share_attention_bf16, prepare_attention_bf16
from glm_tpu.layers.attention.dsa_indexer import prefill_dsa, dsa_bf16
from glm_tpu.layers.mlp import prefill_dense, dense_bf16
from glm_tpu.layers.moe.routed_experts import prefill_moe_from_routes, moe_grouped_routes
from glm_tpu.kernels.fp8_grouped_matmul.kernel import RoutedProjectionConfig
from glm_tpu.config import cache


class PrefillLayerResult(NamedTuple):
    output_local: Any
    carried_residual_local: Any
    cache_local: Any
    unrepaired_index_cache: Any
    repaired_index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any
    normalized_input_local: Any


class PrefillPrefixResult(NamedTuple):
    """Proposed attention state and actual split-residual MLP boundary."""

    normalized_mlp_local: Any
    carried_residual_local: Any
    cache_local: Any
    unrepaired_index_cache: Any
    repaired_index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any
    normalized_input_local: Any


def prefill_mlp(
    normalized_mlp: Any,
    live: Any,
    dense_weights: Bf16DenseWeights | None,
    moe_weights: Bf16MoeWeights | None,
    *,
    moe_contract: GlmMoeNumericalContract,
    linear_interpret: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """The MLP suffix of a <=128-row layer window: dense, or router + routed-expert panels.

    No normalization or residual addition here: the prefix has already applied
    both split-residual norms. The caller must retain prefix health and state.
    """
    if (
        normalized_mlp.ndim != 2
        or not 1 <= normalized_mlp.shape[0] <= 128
        or normalized_mlp.dtype != jnp.bfloat16
        or normalized_mlp.shape[1] * 4 != moe_contract.hidden_size
        or live.shape != (normalized_mlp.shape[0],)
        or live.dtype != jnp.bool_
        or (dense_weights is None) == (moe_weights is None)
    ):
        raise ValueError("prefill MLP boundary geometry/branch drifted")
    rows = normalized_mlp.shape[0]
    mlp_valid = jnp.ones((rows,), jnp.bool_)
    if dense_weights is not None:
        output = prefill_dense(
            normalized_mlp,
            dense_weights.gate_local,
            dense_weights.up_local,
            dense_weights.down_local,
            interpret=linear_interpret,
        )
        route_indices = jnp.full((rows, moe_contract.top_k), -1, jnp.int32)
        route_weights = jnp.zeros((rows, moe_contract.top_k), jnp.float32)
    else:
        if moe_weights.router_weight_local.shape[0] * 8 != moe_contract.num_experts:
            raise ValueError("prefill router and grouped expert counts disagree")
        route_indices, route_weights, router_valid = prefill_router(
            normalized_mlp,
            moe_weights.router_weight_local,
            moe_weights.correction_bias_local,
            live,
            top_k=moe_contract.top_k,
        )
        output, grouped_valid = prefill_moe_from_routes(
            normalized_mlp,
            route_indices,
            route_weights,
            moe_weights.expert_gate_bits_local,
            moe_weights.expert_gate_scale_local,
            moe_weights.expert_up_bits_local,
            moe_weights.expert_up_scale_local,
            moe_weights.expert_down_bits_local,
            moe_weights.expert_down_scale_local,
            moe_weights.shared_gate_local,
            moe_weights.shared_up_local,
            moe_weights.shared_down_local,
            contract=moe_contract,
            interpret=linear_interpret,
        )
        mlp_valid = router_valid & grouped_valid
    return output, route_indices, route_weights, mlp_valid


def prefill_transformer_layer(
    hidden_update_local: Any,
    carried_residual_local: Any,
    cache_local: Any,
    unrepaired_index_cache: Any,
    repaired_index_cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    selected_scores: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    qkv_a_weights: Bf16QkvAWeights,
    attention_weights: Bf16AttentionWeights,
    dsa_weights: Bf16DsaWeights | None,
    materialized_wk: Any | None,
    post_attention_norm_weight_local: Any,
    dense_weights: Bf16DenseWeights | None,
    moe_weights: Bf16MoeWeights | None,
    incoming_contract_valid: Any,
    *,
    main_rope_table_rows: Any,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    rms_norm_epsilon: float = 1e-5,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> PrefillPrefixResult:
    """The attention/DSA prefix of one full/shared-indexer layer on a <=32-row prompt tile.

    Static weight presence selects branches; inactive branches do not trace model
    compute. Carry split residuals through BOTH norms without first rounding their
    FP32 sum. Metadata and local health stay on device; no Python per-row dispatch.
    Failed health forbids committing any returned cache/residual on any chip.
    """
    if (
        hidden_update_local.ndim != 2
        or not 1 <= hidden_update_local.shape[0] <= 32
        or (
            hidden_update_local.dtype != jnp.bfloat16
            or carried_residual_local.shape != hidden_update_local.shape
            or carried_residual_local.dtype != jnp.bfloat16
            or hidden_update_local.shape[1] * 4 != dsa_contract.hidden_size
        )
    ):
        raise ValueError("prefill split residual geometry drifted")
    rows = hidden_update_local.shape[0]
    if (
        incoming_contract_valid.shape != (rows,)
        or incoming_contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("prefill layer requires per-row boolean incoming health")
    if valid_rows.shape != () or valid_rows.dtype != jnp.int32:
        raise ValueError("prefill live count must be int32 scalar")
    if (dsa_weights is None) != (materialized_wk is None):
        raise ValueError(
            "full indexer requires both raw DSA owners and completed repair wk"
        )
    if (dense_weights is None) == (moe_weights is None):
        raise ValueError("prefill layer requires exactly one dense or MoE branch")
    if (
        moe_contract.hidden_size != dsa_contract.hidden_size
        or moe_contract.stage_size != 8
        or (dsa_contract.top_k != attention_contract.top_k)
    ):
        raise ValueError("prefill layer numerical contracts disagree")
    if (
        selected_positions.shape != (rows, dsa_contract.top_k)
        or selected_positions.dtype != jnp.int32
        or (
            selected_valid_counts.shape != (rows,)
            or selected_valid_counts.dtype != jnp.int32
            or selected_scores.shape != selected_positions.shape
            or selected_scores.dtype != jnp.float32
        )
    ):
        raise ValueError("prefill IndexShare metadata geometry drifted")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    update = jnp.where(live[:, None], hidden_update_local, 0)
    residual = jnp.where(live[:, None], carried_residual_local, 0)
    normalized, combined = sharded_fused_add_rms_norm(
        update,
        residual,
        qkv_a_weights.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size,
        epsilon=rms_norm_epsilon,
    )
    prepared = prefill_prepare_attention(
        combined,
        qkv_a_weights,
        precomputed_normalized_local=normalized,
        rms_norm_epsilon=rms_norm_epsilon,
        linear_interpret=linear_interpret,
    )
    dsa_valid = jnp.bool_(True)
    if dsa_weights is not None:
        dsa = prefill_dsa(
            prepared,
            unrepaired_index_cache,
            repaired_index_cache,
            position_offset,
            valid_rows,
            block_table,
            dsa_weights,
            materialized_wk,
            contract=dsa_contract,
            linear_interpret=linear_interpret,
        )
        unrepaired_index_cache, repaired_index_cache = (
            dsa.unrepaired_index_cache,
            dsa.repaired_index_cache,
        )
        selected_positions, selected_valid_counts, selected_scores = (
            dsa.selected_positions,
            dsa.selected_valid_counts,
            dsa.selected_scores,
        )
        dsa_valid = dsa.contract_valid
    attention = prefill_index_share_lse(
        combined,
        prepared,
        cache_local,
        selected_positions,
        selected_valid_counts,
        position_offset,
        valid_rows,
        block_table,
        attention_weights,
        main_rope_table_rows=main_rope_table_rows,
        contract=attention_contract,
        sparse_attention_config=sparse_attention_config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        add_residual=False,
    )
    normalized_mlp, post_residual = sharded_fused_add_rms_norm(
        attention.output_local,
        combined,
        post_attention_norm_weight_local,
        global_hidden_size=moe_contract.hidden_size,
        epsilon=rms_norm_epsilon,
    )
    normalized_mlp = jnp.where(live[:, None], normalized_mlp, 0)
    post_residual = jnp.where(live[:, None], post_residual, 0)
    selected_live = (
        jnp.arange(dsa_contract.top_k)[None] < selected_valid_counts[:, None]
    )
    prefix_valid = (
        incoming_contract_valid
        & dsa_valid
        & attention.contract_valid
        & (
            ~live
            | (
                jnp.all(jnp.isfinite(normalized), axis=1)
                & jnp.all(jnp.isfinite(normalized_mlp), axis=1)
                & jnp.all(jnp.isfinite(post_residual), axis=1)
                & jnp.all(~selected_live | jnp.isfinite(selected_scores), axis=1)
            )
        )
    )
    return PrefillPrefixResult(
        normalized_mlp,
        post_residual,
        attention.cache_local,
        unrepaired_index_cache,
        repaired_index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        prefix_valid,
        normalized,
    )


def prefill_layer_window(
    hidden_update_local: Any,
    carried_residual_local: Any,
    cache_local: Any,
    unrepaired_index_cache: Any,
    repaired_index_cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    selected_scores: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    qkv_a_weights: Bf16QkvAWeights,
    attention_weights: Bf16AttentionWeights,
    dsa_weights: Bf16DsaWeights | None,
    materialized_wk: Any | None,
    post_attention_norm_weight_local: Any,
    dense_weights: Bf16DenseWeights | None,
    moe_weights: Bf16MoeWeights | None,
    incoming_contract_valid: Any,
    *,
    main_rope_table_rows: Any,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    rms_norm_epsilon: float = 1e-5,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> PrefillLayerResult:
    """Four rolled <=32-row prefixes at B128, one dense (canonical) or MoE MLP suffix.

    IndexShare metadata is sliced by original row, never reused across rows.
    Later tiles consume this layer's proposed KV and unrepaired index writes;
    repaired keys stay separate. Zero-live trailing tiles use an in-capacity
    offset and cannot advance any frontier or write cache. M64 repair stays in
    the existing narrow prefix. No attention/DSA/cache row guard is raised.
    """
    if hidden_update_local.ndim != 2 or not 1 <= hidden_update_local.shape[0] <= 128:
        raise ValueError("prefill layer window requires1..128 rows")
    rows = hidden_update_local.shape[0]
    # The canonical placement applies to every dense MLP (layers 0..2 in GLM); MoE layers run the
    # router and routed-expert panels. Neither changes the causal prefix schedule.
    canonical_dense = dense_weights is not None
    if canonical_dense and (rows not in (114, 128) or moe_weights is not None):
        raise ValueError(
            "canonical dense requires original rolled B114/B128 dense window"
        )
    if (
        carried_residual_local.shape != hidden_update_local.shape
        or incoming_contract_valid.shape != (rows,)
        or selected_positions.shape != (rows, dsa_contract.top_k)
        or selected_scores.shape != selected_positions.shape
        or selected_valid_counts.shape != (rows,)
        or main_rope_table_rows.shape != (rows, attention_contract.qk_rope_head_dim)
        or position_offset.shape != ()
        or position_offset.dtype != jnp.int32
        or valid_rows.shape != ()
        or valid_rows.dtype != jnp.int32
    ):
        raise ValueError("prefill layer window metadata geometry drifted")
    # Both buffers have logical page512, stripe8; local cache has64 rows/page.
    capacity = cache_local.shape[0] * cache_local.shape[1] * 8
    start = jnp.clip(position_offset, 0, capacity - 1)
    count = jnp.clip(valid_rows, 0, rows)
    span_valid = (
        (position_offset == start)
        & (valid_rows >= 0)
        & (valid_rows <= rows)
        & (count <= capacity - start)
    )
    # A rolled device loop carries only the three proposed caches. Row outputs
    # are stacked, never historical copies of the caches.
    tile_rows = min(rows, 32)
    tiles = (rows + tile_rows - 1) // tile_rows
    padded_rows = tiles * tile_rows

    def tile_input(value: Any, padding: Any = 0) -> Any:
        value = jnp.pad(
            value,
            ((0, padded_rows - rows), *((0, 0) for _ in value.shape[1:])),
            constant_values=padding,
        )
        return value.reshape((tiles, tile_rows, *value.shape[1:]))

    inputs = (
        jnp.arange(tiles, dtype=jnp.int32),
        tile_input(hidden_update_local),
        tile_input(carried_residual_local),
        tile_input(selected_positions, -1),
        tile_input(selected_valid_counts),
        tile_input(selected_scores, -jnp.inf),
        tile_input(incoming_contract_valid, False),
        tile_input(main_rope_table_rows),
    )

    def prefix_body(caches: tuple[Any, ...], values: tuple[Any, ...]) -> tuple:
        tile, update, residual, selected, counts, scores, health, rope = values
        first = tile * tile_rows
        offset = start + jnp.minimum(first, capacity - 1 - start)
        result = prefill_transformer_layer(
            update,
            residual,
            *caches,
            selected,
            counts,
            scores,
            offset,
            jnp.clip(count - first, 0, tile_rows),
            block_table,
            qkv_a_weights,
            attention_weights,
            dsa_weights,
            materialized_wk,
            post_attention_norm_weight_local,
            dense_weights,
            moe_weights,
            health & span_valid,
            main_rope_table_rows=rope,
            dsa_contract=dsa_contract,
            attention_contract=attention_contract,
            moe_contract=moe_contract,
            rms_norm_epsilon=rms_norm_epsilon,
            sparse_attention_config=sparse_attention_config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
        )
        return (
            result.cache_local,
            result.unrepaired_index_cache,
            result.repaired_index_cache,
        ), (
            result.normalized_mlp_local,
            result.carried_residual_local,
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
            result.contract_valid,
            result.normalized_input_local,
        )

    with jax.named_scope("greenfield_ws32_prefill_rolled_prefix"):
        caches, stacked = lax.scan(
            prefix_body,
            (cache_local, unrepaired_index_cache, repaired_index_cache),
            inputs,
            unroll=1,
        )
    cache_local, unrepaired_index_cache, repaired_index_cache = caches
    (
        normalized,
        carried_residual,
        positions,
        valid_counts,
        scores,
        prefix_health,
        normalized_input,
    ) = tuple(v.reshape((padded_rows, *v.shape[2:]))[:rows] for v in stacked)

    live = jnp.arange(rows, dtype=jnp.int32) < count
    if canonical_dense:
        output, ids, weights, mlp_health = prefill_dense_canonical(
            normalized,
            live,
            dense_weights,
            moe_contract=moe_contract,
            linear_interpret=linear_interpret,
        )
    else:
        output, ids, weights, mlp_health = prefill_mlp(
            normalized,
            live,
            dense_weights,
            moe_weights,
            moe_contract=moe_contract,
            linear_interpret=linear_interpret,
        )
    output = jnp.where(live[:, None], output, 0)
    health = (
        prefix_health
        & mlp_health
        & (~live | jnp.all(jnp.isfinite(output), axis=1))
    )
    return PrefillLayerResult(
        output,
        carried_residual,
        cache_local,
        unrepaired_index_cache,
        repaired_index_cache,
        positions,
        valid_counts,
        scores,
        ids,
        weights,
        health,
        normalized_input,
    )


class AttentionLayerResult(NamedTuple):
    output_local: Any
    cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


class TransformerLayerResult(NamedTuple):
    output_local: Any
    carried_residual_local: Any
    normalized_input_local: Any
    cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any


class MlpResult(NamedTuple):
    output_local: Any
    route_indices: Any
    route_weights: Any


def attention_layer_bf16(residual_local: Any, cache_local: Any, index_cache_local: Any, selected_positions: Any,
                         selected_valid_counts: Any, selected_scores: Any, position: Any, block_tables: Any,
                         context_lengths: Any, qkv_a: Bf16QkvAWeights, attention: Bf16AttentionWeights,
                         dsa: Bf16DsaWeights | None, *, normalized: Any, dsa_contract: DsaNumericalContract,
                         attention_contract: MlaNumericalContract, cache_layout: StageLocalKvLayout,
                         sparse_attention_config: SparseMlaConfig, sparse_attention_interpret: bool,
                         main_rope_table_row: Any, expert_axis: str = "expert",
                         feature_axis: str = "feature") -> AttentionLayerResult:
    prepared = prepare_attention_bf16(residual_local, qkv_a, normalized=normalized,
                                     feature_axis=feature_axis)
    dsa_valid = jnp.ones((1,), dtype=jnp.bool_)
    if dsa is not None:
        result = dsa_bf16(prepared, index_cache_local, position, block_tables, context_lengths, dsa,
                          expert_axis=expert_axis, feature_axis=feature_axis, contract=dsa_contract,
                          cache_layout=cache_layout)
        index_cache_local = result.index_cache_local
        selected_positions, selected_valid_counts, selected_scores = (
            result.selected_positions, result.selected_valid_counts, result.selected_scores)
        dsa_valid = result.contract_valid
    attended = index_share_attention_bf16(
        residual_local, prepared, cache_local, selected_positions, selected_valid_counts, position,
        block_tables, context_lengths, attention, expert_axis=expert_axis, contract=attention_contract,
        cache_layout=cache_layout, main_rope_table_row=main_rope_table_row,
        sparse_attention_config=sparse_attention_config, sparse_attention_interpret=sparse_attention_interpret,
    )
    return AttentionLayerResult(
        attended.output_local, attended.cache_local, index_cache_local, selected_positions,
        selected_valid_counts, selected_scores, dsa_valid & attended.contract_valid,
    )


def mlp_bf16(post_attention_residual_local: Any, normalized: Any, dense: Bf16DenseWeights | None,
             moe: Bf16MoeWeights | None, *, mlp_kind: str, contract: GlmMoeNumericalContract,
             routed_projection: RoutedProjectionConfig | None, interpret: bool,
             expert_axis: str = "expert", feature_axis: str = "feature") -> MlpResult:
    if mlp_kind == "dense":
        if dense is None:
            raise ValueError("bf16 dense layer needs dense weights")
        update = dense_bf16(normalized, dense, expert_axis=expert_axis, feature_axis=feature_axis)
        return MlpResult(update, jnp.full((1, contract.top_k), jnp.int32(-1), dtype=jnp.int32),
                             jnp.zeros((1, contract.top_k), dtype=jnp.float32))
    if moe is None:
        raise ValueError("bf16 sparse layer needs MoE weights")
    route_indices, route_weights = router_from_shards(
        normalized, moe.router_weight_local, moe.correction_bias_local, top_k=contract.top_k,
    )
    update = moe_grouped_routes(
        normalized, route_indices, route_weights, *moe[2:8],
        None, None, None, None, None, None,
        contract=contract, config=routed_projection, interpret=interpret,
        shared_bf16=(moe.shared_gate_local, moe.shared_up_local, moe.shared_down_local),
    )
    return MlpResult(update, route_indices, route_weights)


# ----------------------------------------------------------------------------- layer / step
def transformer_layer_bf16(hidden_update_local: Any, carried_residual_local: Any, cache_local: Any,
                           index_cache_local: Any, selected_positions: Any, selected_valid_counts: Any,
                           selected_scores: Any, position: Any, block_tables: Any, context_lengths: Any,
                           layer: Bf16LayerWeights, incoming_contract_valid: Any, *, indexer_kind: str,
                           mlp_kind: str, config: cache.CacheConfig,
                           routed_projection: RoutedProjectionConfig | None, sparse_attention_interpret: bool,
                           linear_interpret: bool, main_rope_table_row: Any) -> TransformerLayerResult:
    if (layer.dsa is None) != (indexer_kind == "shared"):
        raise ValueError("bf16 layer: full indexer alone must carry DSA weights")
    if (layer.dense is None) != (mlp_kind == "sparse") or (layer.moe is None) != (mlp_kind == "dense"):
        raise ValueError("bf16 layer weight presence drifted")
    dsa_contract = config.dsa_contract
    normalized_input, combined_residual = sharded_fused_add_rms_norm(
        hidden_update_local, carried_residual_local, layer.qkv_a.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    attention = attention_layer_bf16(
        combined_residual, cache_local, index_cache_local, selected_positions, selected_valid_counts,
        selected_scores, position, block_tables, context_lengths, layer.qkv_a, layer.attention, layer.dsa,
        normalized=normalized_input, dsa_contract=dsa_contract, attention_contract=config.attention_contract,
        cache_layout=config.cache_layout,
        sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
        sparse_attention_interpret=sparse_attention_interpret, main_rope_table_row=main_rope_table_row,
    )
    normalized_mlp, post_attention_residual = sharded_fused_add_rms_norm(
        attention.output_local, combined_residual, layer.post_attention_norm_weight_local,
        global_hidden_size=config.moe_contract.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    mlp = mlp_bf16(post_attention_residual, normalized_mlp, layer.dense, layer.moe, mlp_kind=mlp_kind,
                   contract=config.moe_contract, routed_projection=routed_projection, interpret=linear_interpret)
    return TransformerLayerResult(
        mlp.output_local, post_attention_residual, normalized_input, attention.cache_local,
        attention.index_cache_local, attention.selected_positions, attention.selected_valid_counts,
        attention.selected_scores, mlp.route_indices, mlp.route_weights,
        incoming_contract_valid & attention.contract_valid,
    )


def prefill_dense_canonical(
    normalized: Any,
    live: Any,
    dense: Bf16DenseWeights,
    *,
    moe_contract: GlmMoeNumericalContract,
    linear_interpret: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Four uniform device calls, each32 live slots in physical rows0:32 of128.

    Only the dense suffix changes. No cache, normalization, host stride, MoE,
    precision or checkpoint change. B114 is padded to the same four placements
    and cropped back after assembly; malformed input geometry fails at trace.
    Padded input rows are zeroed before entering the original dense arithmetic.
    All owners execute all four iterations, including empty tiles.
    """
    if (
        type(linear_interpret) is not bool
        or normalized.ndim != 2
        or normalized.shape[0] not in (114, 128)
        or normalized.dtype != jnp.bfloat16
        or normalized.shape[1] * 4 != moe_contract.hidden_size
        or live.shape != (normalized.shape[0],)
        or live.dtype != jnp.bool_
        or not isinstance(dense, Bf16DenseWeights)
    ):
        raise ValueError(
            "canonical dense requires B114/B128 and original dense weights"
        )
    rows, width = normalized.shape
    safe = jnp.where(live[:, None], normalized, jnp.bfloat16(0))
    tiles = jnp.pad(safe, ((0, 128 - rows), (0, 0))).reshape(4, 32, width)
    masks = jnp.pad(live, ((0, 128 - rows),), constant_values=False).reshape(4, 32)

    def body(unused: None, inputs: tuple[Any, Any]) -> tuple[None, tuple]:
        values, mask = inputs
        padded = jnp.pad(values, ((0, 96), (0, 0)))
        padded_live = jnp.pad(mask, ((0, 96),), constant_values=False)
        output, ids, weights, valid = prefill_mlp(
            padded,
            padded_live,
            dense,
            None,
            moe_contract=moe_contract,
            linear_interpret=linear_interpret,
        )
        # Preserve the existing suffix's live output/health contract; never hide
        # a nonfinite live result by masking it after deriving validity.
        valid = valid & (~padded_live | jnp.all(jnp.isfinite(output), axis=1))
        output = jnp.where(padded_live[:, None], output, jnp.bfloat16(0))
        return None, tuple(v[:32] for v in (output, ids, weights, valid))

    with jax.named_scope("greenfield_ws32_prefill_dense_canonical"):
        _, stacked = lax.scan(body, None, (tiles, masks), unroll=1)
    return tuple(v.reshape((128, *v.shape[2:]))[:rows] for v in stacked)
