"""Opt-in <=128-row layer window with bounded causal attention/DSA tiles.

The static tile loop is inside the compiled device program. It is not a token
scan of the model and has no host stage dispatch. Returned caches are proposals;
the decoder's existing all-owner atomic commit remains the only state frontier.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .pallas import SparseMlaConfig
from .reference.attention import MlaNumericalContract
from .reference.dsa import DsaNumericalContract
from .reference.moe import GlmMoeNumericalContract
from .ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
)
from .ws32_prefill_layer import (
    Ws32PrefillLayerResult,
    ws32_prefill_mlp_mapped,
    ws32_prefill_transformer_layer_mapped,
)


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
    qkv_a_weights: Ws32QkvAWeights,
    attention_weights: Ws32AttentionWeights,
    dsa_weights: Ws32DsaWeights | None,
    materialized_wk: Any | None,
    post_attention_norm_weight_local: Any,
    dense_weights: Ws32DenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    incoming_contract_valid: Any,
    *,
    main_rope_table_rows: Any,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    rms_norm_epsilon: float = 1e-5,
    key_tile: int = 4096,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32PrefillLayerResult:
    """Four <=32-row prefixes at B128, one dense/router/grouped-MLP suffix.

    IndexShare metadata is sliced by original row, never reused across rows.
    Later tiles consume this layer's proposed KV and unrepaired index writes;
    repaired keys stay separate. Zero-live trailing tiles use an in-capacity
    offset and cannot advance any frontier or write cache. M64 repair stays in
    the existing narrow prefix. No attention/DSA/cache row guard is raised.
    """
    if hidden_update_local.ndim != 2 or not 1 <= hidden_update_local.shape[0] <= 128:
        raise ValueError("prefill layer window requires1..128 rows")
    rows = hidden_update_local.shape[0]
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
    prefixes = []
    for tile_start in range(0, rows, 32):
        tile_end = min(tile_start + 32, rows)
        tile_count = jnp.clip(count - tile_start, 0, tile_end - tile_start)
        # Bound the addition first so malformed INTMAX inputs cannot overflow.
        offset = start + jnp.minimum(jnp.int32(tile_start), capacity - 1 - start)
        result = ws32_prefill_transformer_layer_mapped(
            hidden_update_local[tile_start:tile_end],
            carried_residual_local[tile_start:tile_end],
            cache_local,
            unrepaired_index_cache,
            repaired_index_cache,
            selected_positions[tile_start:tile_end],
            selected_valid_counts[tile_start:tile_end],
            selected_scores[tile_start:tile_end],
            offset,
            tile_count,
            block_table,
            qkv_a_weights,
            attention_weights,
            dsa_weights,
            materialized_wk,
            post_attention_norm_weight_local,
            dense_weights,
            moe_weights,
            incoming_contract_valid[tile_start:tile_end] & span_valid,
            main_rope_table_rows=main_rope_table_rows[tile_start:tile_end],
            dsa_contract=dsa_contract,
            attention_contract=attention_contract,
            moe_contract=moe_contract,
            rms_norm_epsilon=rms_norm_epsilon,
            key_tile=key_tile,
            sparse_attention_config=sparse_attention_config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
            prefix_only=True,
        )
        cache_local = result.cache_local
        unrepaired_index_cache = result.unrepaired_index_cache
        repaired_index_cache = result.repaired_index_cache
        prefixes.append(result)

    def join(field: str) -> Any:
        return jnp.concatenate([getattr(p, field) for p in prefixes], axis=0)

    normalized = join("normalized_mlp_local")
    live = jnp.arange(rows, dtype=jnp.int32) < count
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
        join("contract_valid")
        & mlp_health
        & (~live | jnp.all(jnp.isfinite(output), axis=1))
    )
    return Ws32PrefillLayerResult(
        output,
        join("carried_residual_local"),
        cache_local,
        unrepaired_index_cache,
        repaired_index_cache,
        join("selected_positions"),
        join("selected_valid_counts"),
        join("selected_scores"),
        ids,
        weights,
        health,
        join("normalized_input_local"),
    )
