"""Prefill layer window of the production engine: four rolled 32-row prefixes, one MLP suffix.

One layer of a B128 (or B114) prompt block: a rolled device loop runs the attention/DSA prefix
(``prefill_layer.ws32_prefill_transformer_layer_mapped``) on four 32-row tiles, each tile
consuming the KV and unrepaired index rows its predecessors proposed; the MLP suffix then runs
once over the whole block -- the canonical four-placement dense MLP for dense layers
(``prefill_dense_canonical``), the router and routed-expert panels for MoE layers. The executed
copy of the frozen window ``greenfield/kernels/ws32_prefill_window.py`` with the admitted profile
hard-wired (S2d fold); the frozen module stays untouched as the numerical oracle of the tests.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from ..greenfield.kernels.reference.attention import MlaNumericalContract
from ..greenfield.kernels.reference.dsa import DsaNumericalContract
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from ..greenfield.kernels.ws32_prefill_layer import Ws32PrefillLayerResult
from .bf16_resident import (
    Bf16AttentionWeights,
    Bf16DenseWeights,
    Bf16DsaWeights,
    Bf16MoeWeights,
    Bf16QkvAWeights,
)
from .prefill_dense_canonical import ws32_prefill_dense_canonical_mapped
from .prefill_layer import ws32_prefill_mlp_mapped, ws32_prefill_transformer_layer_mapped


def ws32_prefill_layer_window_mapped(
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
) -> Ws32PrefillLayerResult:
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
        result = ws32_prefill_transformer_layer_mapped(
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
        output, ids, weights, mlp_health = ws32_prefill_dense_canonical_mapped(
            normalized,
            live,
            dense_weights,
            moe_contract=moe_contract,
            linear_interpret=linear_interpret,
        )
    else:
        output, ids, weights, mlp_health = ws32_prefill_mlp_mapped(
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
    return Ws32PrefillLayerResult(
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
