"""Private prefill window mirror for D8 dependency injection.

The frozen body is unchanged except that the canonical dense dependency is
imported at module scope, allowing a private binding to the BF16 MLP. Tests
compare its AST with the frozen body after removing that one local import.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.reference.attention import MlaNumericalContract
from ..greenfield.kernels.reference.dsa import DsaNumericalContract
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from ..greenfield.kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
)
from ..greenfield.kernels.ws32_prefill_layer import (
    Ws32PrefillLayerResult,
    Ws32PrefillPrefixResult,
    ws32_prefill_mlp_mapped,
    ws32_prefill_transformer_layer_mapped,
)


from ..greenfield.kernels.ws32_prefill_dense_canonical import ws32_prefill_dense_canonical_mapped


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
    paired_position_sort: bool = False,
    sorted_local_merge: bool = False,
    expert_panels: bool = False,
    rolled_prefix: bool = False,
    canonical_dense: bool = False,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
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
    if any(
        type(v) is not bool
        for v in (rolled_prefix, expert_panels, sorted_local_merge, canonical_dense)
    ):
        raise ValueError("prefill window choices must be static booleans")
    if rolled_prefix and _observe is not None:
        raise ValueError("rolled prefix does not support trace-time observation hooks")
    rows = hidden_update_local.shape[0]
    if canonical_dense and (
        not rolled_prefix
        or not expert_panels
        or rows not in (114, 128)
        or dense_weights is None
        or moe_weights is not None
        or _observe is not None
    ):
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
    prefixes = []
    # A rolled device loop carries only the three proposed caches. Row outputs
    # are stacked, never historical copies of the caches. This is a new compiler
    # realization: BF16 scan outputs do NOT claim separate-executable identity.
    if rolled_prefix:
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
                key_tile=key_tile,
                paired_position_sort=paired_position_sort,
                sorted_local_merge=sorted_local_merge,
                sparse_attention_config=sparse_attention_config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                prefix_only=True,
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
        flat = tuple(v.reshape((padded_rows, *v.shape[2:]))[:rows] for v in stacked)
        prefixes.append(
            Ws32PrefillPrefixResult(
                flat[0], flat[1], *caches, flat[2], flat[3], flat[4], flat[5], flat[6]
            )
        )

    for tile_start in () if rolled_prefix else range(0, rows, 32):
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
            paired_position_sort=paired_position_sort,
            sorted_local_merge=sorted_local_merge,
            sparse_attention_config=sparse_attention_config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
            prefix_only=True,
            _observe=(
                None
                if _observe is None
                else lambda name, arrays, start=tile_start: _observe(
                    f"tile{start}/{name}", arrays
                )
            ),
        )
        cache_local = result.cache_local
        unrepaired_index_cache = result.unrepaired_index_cache
        repaired_index_cache = result.repaired_index_cache
        prefixes.append(result)

    def join(field: str) -> Any:
        return jnp.concatenate([getattr(p, field) for p in prefixes], axis=0)

    normalized = join("normalized_mlp_local")
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
            expert_panels=expert_panels,
            _observe=_observe,
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
