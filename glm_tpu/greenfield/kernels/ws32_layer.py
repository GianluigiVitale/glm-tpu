"""Complete batch-one WS32 attention, DSA, and IndexShare layer bodies.

These functions execute inside one ``shard_map`` over the exact
``expert=8 x feature=4`` mesh.  The residual stays ``[1, hidden/4]`` on the
feature axis.  Context pages and complete attention/DSA heads stay on the
expert axis and are replicated only across feature columns.  Repeated
collectives therefore have physical size four or eight; no operation
materializes a full hidden row on all 32 chips.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .pallas.fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_matmul_f32,
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
)
from .pallas.sparse_attention import SparseMlaConfig, pregathered_sparse_mla_pallas
from .reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    gather_stage_local_selected_kv_aligned,
)
from .reference.dsa import (
    DsaNumericalContract,
    SelectedPositions,
    dsa_index_keys_from_projection,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates_with_scores,
)
from .reference.linear import linear, residual_add
from .reference.moe import GlmMoeNumericalContract
from .reference.qkv_a import (
    FusedQkvAContract,
    one_row_fused_qkv_a_convolution,
)
from .reference.rmsnorm import rms_norm
from .reference.rotary import (
    apply_rotary,
    apply_rotary_fp32_final_round,
    rotary_cos_sin,
)
from .prefill_cache import _require_decode_metadata
from .ws32 import (
    ws32_dense_pallas_mapped,
    ws32_fused_add_rms_norm_mapped,
    ws32_fp8_expert_linear_pallas_mapped,
    ws32_fp8_feature_linear_pallas_mapped,
    ws32_moe_pallas_from_routes_mapped,
    ws32_rms_norm_mapped,
    ws32_router_from_shards_mapped,
    ws32_strategy_nd_dense_final_layout_mapped,
)


class Ws32QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_bits_local: Any
    q_a_scale_local: Any
    q_a_norm_weight: Any
    kv_a_bits_local: Any
    kv_a_scale_local: Any
    kv_a_norm_weight: Any


class Ws32DsaWeights(NamedTuple):
    wq_b_bits_local: Any
    wq_b_scale_local: Any
    wk_bits_local: Any
    wk_scale_local: Any
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Ws32ExactDsaWeights(NamedTuple):
    """Externally materialized DB526/527 owners for one full indexer.

    The tuple is deliberately separate from the base final-layout checkpoint
    tree.  A one-shot device executable derives these persistent owners from
    that tree; recurrent decode then transfers only the live hidden row and
    compact query/key metadata.
    """

    qkv_a_bits: Any
    qkv_a_scale: Any
    wq_b_weight_aliases: tuple[Any, Any, Any, Any]
    wk_weight: Any
    head_weight_local: Any


class Ws32AttentionWeights(NamedTuple):
    q_b_bits_local: Any
    q_b_scale_local: Any
    kv_b_bits_local: Any
    kv_b_scale_local: Any
    o_bits_local: Any
    o_scale_local: Any


class Ws32DenseWeights(NamedTuple):
    gate_bits_local: Any
    gate_scale_local: Any
    up_bits_local: Any
    up_scale_local: Any
    down_bits_local: Any
    down_scale_local: Any


class Ws32StrategyNdDenseWeights(NamedTuple):
    """Four ordered legacy-rank shards in their final WS32 ownership."""

    merged_bits_in_out_local: Any
    merged_scale_in_out_local: Any
    down_bits_in_out_local: Any
    down_scale_in_out_local: Any


class Ws32MoeWeights(NamedTuple):
    router_weight_local: Any
    correction_bias_local: Any
    expert_gate_bits_local: Any
    expert_gate_scale_local: Any
    expert_up_bits_local: Any
    expert_up_scale_local: Any
    expert_down_bits_local: Any
    expert_down_scale_local: Any
    shared_gate_bits_local: Any
    shared_gate_scale_local: Any
    shared_up_bits_local: Any
    shared_up_scale_local: Any
    shared_down_bits_local: Any
    shared_down_scale_local: Any


class Ws32PreparedAttention(NamedTuple):
    normalized_local: Any
    normalized_for_exact_dsa: Any
    q_residual: Any
    current_kv: Any


class Ws32DsaResult(NamedTuple):
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


class Ws32AttentionResult(NamedTuple):
    output_local: Any
    cache_local: Any
    contract_valid: Any


class Ws32AttentionLayerResult(NamedTuple):
    output_local: Any
    cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


class Ws32TransformerLayerResult(NamedTuple):
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


class Ws32MlpResult(NamedTuple):
    output_local: Any
    route_indices: Any
    route_weights: Any


def _pallas_config(block_shape: tuple[int, int]) -> Fp8BlockMatmulConfig:
    return Fp8BlockMatmulConfig(
        block_shape=block_shape,
        output_tile=block_shape[0],
        contraction_tile=block_shape[1],
    )


def _require_ws32_layout(
    cache_layout: StageLocalKvLayout,
    *,
    hidden_size: int,
) -> int:
    if cache_layout.local_parallel_size != 8:
        raise ValueError("WS32 context ownership requires expert axis size 8")
    if hidden_size <= 0 or hidden_size % 4:
        raise ValueError("WS32 hidden width must divide over feature axis 4")
    return hidden_size // 4


def ws32_prepare_attention_mapped(
    residual_local: Any,
    weights: Ws32QkvAWeights,
    *,
    exact_dsa_weights: Ws32ExactDsaWeights | None = None,
    precomputed_normalized_local: Any | None = None,
    hidden_size: int = 6144,
    feature_axis: str = "feature",
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
) -> Ws32PreparedAttention:
    """Compute the shared normalized residual and compact q/kv-a states."""

    local_hidden = _require_ws32_layout(
        StageLocalKvLayout(local_parallel_size=8), hidden_size=hidden_size
    )
    if residual_local.shape != (1, local_hidden) or (
        residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 attention residual must be one BF16 feature shard")
    if weights.input_norm_weight_local.shape != (local_hidden,):
        raise ValueError("WS32 input norm weight must match the hidden shard")
    q_lora_rank = weights.q_a_norm_weight.shape[0]
    if q_lora_rank <= 0 or weights.q_a_bits_local.shape != (
        q_lora_rank,
        local_hidden,
    ) or (
        weights.q_a_scale_local.shape
        != (
            (q_lora_rank + block_shape[0] - 1) // block_shape[0],
            (local_hidden + block_shape[1] - 1) // block_shape[1],
        )
    ):
        raise ValueError("WS32 q-a final-owner geometry drifted")
    kv_lora_rank = weights.kv_a_norm_weight.shape[0]
    kv_width = weights.kv_a_bits_local.shape[0]
    if kv_lora_rank <= 0 or kv_width <= kv_lora_rank or (
        weights.kv_a_bits_local.shape[1:] != (local_hidden,)
    ) or (
        weights.kv_a_scale_local.shape
        != (
            (kv_width + block_shape[0] - 1) // block_shape[0],
            (local_hidden + block_shape[1] - 1) // block_shape[1],
        )
    ):
        raise ValueError("WS32 kv-a final-owner geometry drifted")

    if precomputed_normalized_local is None:
        normalized = ws32_rms_norm_mapped(
            residual_local,
            weights.input_norm_weight_local,
            global_hidden_size=hidden_size,
            feature_axis=feature_axis,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized_local
        if normalized.shape != residual_local.shape or (
            normalized.dtype != jnp.bfloat16
        ):
            raise ValueError(
                "WS32 precomputed attention normalization geometry drifted"
            )
    if exact_dsa_weights is None:
        normalized_for_exact_dsa = normalized
        q_a = ws32_fp8_feature_linear_pallas_mapped(
            normalized,
            weights.q_a_bits_local,
            weights.q_a_scale_local,
            feature_axis=feature_axis,
            block_shape=block_shape,
            interpret=linear_interpret,
        )
        q_residual = rms_norm(
            q_a, weights.q_a_norm_weight, epsilon=lora_norm_epsilon
        )
        projected_kv = ws32_fp8_feature_linear_pallas_mapped(
            normalized,
            weights.kv_a_bits_local,
            weights.kv_a_scale_local,
            feature_axis=feature_axis,
            block_shape=block_shape,
            interpret=linear_interpret,
        )
    else:
        if block_shape != (128, 128):
            raise ValueError("WS32 exact qkv-a requires 128x128 FP8 blocks")
        with jax.named_scope("greenfield_ws32_exact_dsa/normalized_feature_gather"):
            normalized_full = lax.all_gather(
                normalized,
                axis_name=feature_axis,
                axis=1,
                tiled=True,
            )
        normalized_for_exact_dsa = normalized_full
        projected = one_row_fused_qkv_a_convolution(
            normalized_full,
            exact_dsa_weights.qkv_a_bits,
            exact_dsa_weights.qkv_a_scale,
            weights.q_a_norm_weight,
            contract=FusedQkvAContract(
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                kv_a_width=kv_width,
                epsilon=lora_norm_epsilon,
            ),
        )
        q_residual = projected.q_residual
        projected_kv = projected.kv_a_projection
    current_kv = jnp.concatenate(
        (
            rms_norm(
                projected_kv[..., :kv_lora_rank],
                weights.kv_a_norm_weight,
                epsilon=lora_norm_epsilon,
            ),
            projected_kv[..., kv_lora_rank:],
        ),
        axis=-1,
    ).astype(jnp.bfloat16)
    return Ws32PreparedAttention(
        normalized,
        normalized_for_exact_dsa,
        q_residual,
        current_kv,
    )


def ws32_dsa_mapped(
    prepared: Ws32PreparedAttention,
    index_cache_local: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    weights: Ws32DsaWeights,
    *,
    exact_weights: Ws32ExactDsaWeights | None = None,
    expert_axis: str = "expert",
    feature_axis: str = "feature",
    contract: DsaNumericalContract = DsaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(
        local_parallel_size=8
    ),
    block_shape: tuple[int, int] = (128, 128),
    linear_interpret: bool = False,
) -> Ws32DsaResult:
    """Write one index key and merge exact top-k over eight context owners."""

    local_hidden = _require_ws32_layout(
        cache_layout, hidden_size=contract.hidden_size
    )
    normalized = prepared.normalized_local
    if normalized.shape != (1, local_hidden) or normalized.dtype != jnp.bfloat16:
        raise ValueError("WS32 DSA normalized input geometry drifted")
    if prepared.q_residual.shape != (1, contract.q_lora_rank):
        raise ValueError("WS32 DSA q residual geometry drifted")
    if index_cache_local.ndim != 3 or index_cache_local.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.head_dim,
    ):
        raise ValueError("WS32 DSA index cache geometry drifted")
    if index_cache_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 DSA index cache must remain BF16")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if weights.wq_b_bits_local.shape != (
        local_heads * contract.head_dim,
        contract.q_lora_rank,
    ):
        raise ValueError("WS32 DSA wq-b owner geometry drifted")
    if weights.wk_bits_local.shape != (contract.head_dim, local_hidden):
        raise ValueError("WS32 DSA wk owner geometry drifted")
    if weights.head_weight_local.shape != (local_heads, local_hidden):
        raise ValueError("WS32 DSA head-weight owner geometry drifted")
    if weights.key_norm_weight.shape != (contract.head_dim,) or (
        weights.key_norm_bias.shape != (contract.head_dim,)
    ):
        raise ValueError("WS32 DSA key norm geometry drifted")

    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = (
        _require_decode_metadata(
            position,
            block_tables,
            context_lengths,
            owner,
            layout=cache_layout,
            physical_page_count=index_cache_local.shape[0],
        )
    )
    if exact_weights is None:
        projected_query = fp8_block_matmul_f32(
            prepared.q_residual,
            weights.wq_b_bits_local,
            weights.wq_b_scale_local,
            config=_pallas_config(block_shape),
            interpret=linear_interpret,
        )
        query = projected_query.reshape(1, local_heads, contract.head_dim)
        head_weight_partial = lax.dot_general(
            normalized.astype(jnp.float32),
            weights.head_weight_local.astype(jnp.float32),
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        with jax.named_scope("greenfield_ws32_dsa/head_weight_feature_reduce"):
            local_head_weights = lax.psum(
                head_weight_partial, axis_name=feature_axis
            ) * jnp.float32(contract.num_heads**-0.5)
        cos, sin = rotary_cos_sin(
            position,
            rotary_dim=contract.rotary_dim,
            theta=contract.theta,
            dtype=jnp.float32,
        )
        rotated = apply_rotary(
            query[..., : contract.rotary_dim],
            cos[:, None, :],
            sin[:, None, :],
            interleaved=contract.interleaved_rotary,
        )
        local_query = jnp.concatenate(
            (rotated, query[..., contract.rotary_dim :]), axis=-1
        ).astype(jnp.float32)
        packed_query = jnp.concatenate(
            (local_query.reshape(1, -1), local_head_weights), axis=-1
        )
        query_axis = expert_axis
        local_query_width = local_heads * contract.head_dim
    else:
        exact_local_heads = contract.num_heads // 4
        aliases = exact_weights.wq_b_weight_aliases
        if len(aliases) != 4 or any(
            weight.shape
            != (
                exact_local_heads * contract.head_dim,
                contract.q_lora_rank,
            )
            or weight.dtype != jnp.float32
            for weight in aliases
        ):
            raise ValueError("WS32 exact DSA tuple4 query owner drifted")
        if exact_weights.head_weight_local.shape != (
            exact_local_heads,
            contract.hidden_size,
        ) or exact_weights.head_weight_local.dtype != jnp.bfloat16:
            raise ValueError("WS32 exact DSA head owner drifted")
        if exact_weights.wk_weight.shape != (
            contract.head_dim,
            contract.hidden_size,
        ) or exact_weights.wk_weight.dtype != jnp.float32:
            raise ValueError("WS32 exact DSA wk owner drifted")
        normalized_full = prepared.normalized_for_exact_dsa
        if normalized_full.shape != (1, contract.hidden_size) or (
            normalized_full.dtype != jnp.bfloat16
        ):
            raise ValueError("WS32 exact DSA normalized owner drifted")
        with jax.named_scope("greenfield_ws32_exact_dsa/tuple4_query"):
            local_query, local_head_weights = ws32_grouped_dsa_query_and_head(
                prepared.q_residual,
                normalized_full,
                aliases,
                exact_weights.head_weight_local,
                position,
                contract=contract,
            )
        packed_query = jnp.concatenate(
            (local_query.reshape(1, -1), local_head_weights), axis=-1
        )
        query_axis = feature_axis
        local_query_width = exact_local_heads * contract.head_dim
    with jax.named_scope(f"greenfield_ws32_dsa/query_{query_axis}_gather"):
        gathered_query = lax.all_gather(
            packed_query,
            axis_name=query_axis,
            axis=0,
            tiled=False,
        )
    query = jnp.transpose(
        gathered_query[..., :local_query_width], (1, 0, 2)
    ).reshape(1, contract.num_heads, contract.head_dim)
    head_weights = jnp.transpose(
        gathered_query[..., local_query_width:], (1, 0, 2)
    ).reshape(1, contract.num_heads)

    if exact_weights is None:
        key_partial = fp8_block_matmul_f32(
            normalized,
            weights.wk_bits_local,
            weights.wk_scale_local,
            config=_pallas_config(block_shape),
            interpret=linear_interpret,
        )
        with jax.named_scope("greenfield_ws32_dsa/key_feature_reduce"):
            projected_key = lax.psum(key_partial, axis_name=feature_axis)
        current_key_f32 = dsa_index_keys_from_projection(
            projected_key,
            weights.key_norm_weight,
            weights.key_norm_bias,
            position,
            contract=contract,
        ).astype(jnp.float32)
        score_precision = "highest"
    else:
        with jax.named_scope("greenfield_ws32_exact_dsa/exact_current_key"):
            current_key_f32 = ws32_exact_dsa_current_key(
                normalized_full,
                exact_weights.wk_weight,
                weights.key_norm_weight,
                weights.key_norm_bias,
                position,
                contract=contract,
            )
        score_precision = "default"
    current_key = current_key_f32.astype(index_cache_local.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_key[0])

    index_cache_local = lax.cond(
        metadata_valid[0] & (owner == target_owner),
        write_current,
        lambda value: value,
        index_cache_local,
    )
    page_ids = block_tables[0]
    page_ok = (page_ids >= 0) & (page_ids < index_cache_local.shape[0])
    safe_pages = jnp.clip(page_ids, 0, index_cache_local.shape[0] - 1)
    logical_cache = jnp.take(index_cache_local, safe_pages, axis=0)
    logical_pages = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    local_rows = jnp.arange(
        cache_layout.local_rows_per_page, dtype=jnp.int32
    )
    global_positions = (
        logical_pages[:, None] * jnp.int32(cache_layout.logical_page_size)
        + owner.astype(jnp.int32)
        * jnp.int32(cache_layout.local_rows_per_page)
        + local_rows[None, :]
    )
    global_positions = jnp.where(
        page_ok[:, None], global_positions, jnp.int32(-1)
    ).reshape(-1)
    local_keys = logical_cache.reshape(-1, contract.head_dim)
    score_scope = (
        "greenfield_ws32_exact_dsa/default_score"
        if exact_weights is not None
        else "greenfield_ws32_dsa/highest_score"
    )
    with jax.named_scope(score_scope):
        local_scores = dsa_scores(
            query, local_keys, head_weights, precision=score_precision
        )
    candidate_scores, candidate_positions = local_topk_candidates(
        local_scores,
        global_positions,
        context_lengths,
        top_k=contract.top_k,
    )
    with jax.named_scope("greenfield_ws32_dsa/candidate_score_expert_gather"):
        gathered_scores = lax.all_gather(
            candidate_scores,
            axis_name=expert_axis,
            axis=0,
            tiled=False,
        )
    with jax.named_scope("greenfield_ws32_dsa/candidate_position_expert_gather"):
        gathered_positions = lax.all_gather(
            candidate_positions,
            axis_name=expert_axis,
            axis=0,
            tiled=False,
        )
    selected = merge_topk_candidates_with_scores(
        gathered_scores,
        gathered_positions,
        context_lengths,
        top_k=contract.top_k,
        global_context_size=(
            block_tables.shape[1] * cache_layout.logical_page_size
        ),
    )
    selection_valid = canonicalize_selected_positions(
        SelectedPositions(selected.positions, selected.valid_counts)
    ).contract_valid
    return Ws32DsaResult(
        index_cache_local,
        selected.positions,
        selected.valid_counts,
        selected.scores,
        metadata_valid & selection_valid,
    )


def ws32_grouped_dsa_query_and_head(
    q_residual: Any,
    normalized: Any,
    query_weight_aliases: tuple[Any, ...],
    head_weight: Any,
    position: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> tuple[Any, Any]:
    """Candidate WS32 form of the DB525 grouped query association.

    DB525 proves that TPU-v4 needs one 16-KiB grouped reduction for the
    recurrent DSA query.  PP8 obtains that geometry from four aliases of an
    eight-head owner.  WS32's existing checkpoint instead owns four heads on
    each expert coordinate, so the bounded discriminator must compare that
    existing layout with eight aliases against a feature-owned eight-head
    layout with four aliases.  This helper admits exactly those two physical
    geometries and keeps the BF16 q-a completion boundary explicit.

    The default path does not select this helper. The default-off exact path
    selects only the tuple4 layout proven by the position-8155 discriminator.
    """

    alias_count = len(query_weight_aliases)
    if alias_count not in (4, 8):
        raise ValueError("WS32 exact DSA query requires four or eight aliases")
    if q_residual.shape != (1, contract.q_lora_rank) or (
        q_residual.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 exact DSA q residual geometry drifted")
    if normalized.shape != (1, contract.hidden_size) or (
        normalized.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 exact DSA normalized geometry drifted")
    first = query_weight_aliases[0]
    if first.ndim != 2 or first.shape[1] != contract.q_lora_rank or (
        first.dtype != jnp.float32
    ):
        raise ValueError("WS32 exact DSA query owner geometry drifted")
    if any(
        weight.shape != first.shape or weight.dtype != jnp.float32
        for weight in query_weight_aliases
    ):
        raise ValueError("WS32 exact DSA query aliases must be identical owners")
    if first.shape[0] % contract.head_dim:
        raise ValueError("WS32 exact DSA query owner is not head aligned")
    local_heads = first.shape[0] // contract.head_dim
    if local_heads * alias_count != contract.num_heads:
        raise ValueError("WS32 exact DSA grouped query must retain 32 heads")
    if head_weight.shape != (local_heads, contract.hidden_size) or (
        head_weight.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 exact DSA head-weight owner geometry drifted")
    # Four x 8-head and eight x 4-head both request the DB525 16-KiB grouped
    # TPU reduction.  Reject any shape that only happens to have 32 heads but
    # changes the physical result bytes.
    grouped_bytes = alias_count * first.shape[0] * 4
    if grouped_bytes != 16_384:
        raise ValueError("WS32 exact DSA grouped reduction must be 16 KiB")

    query_input = lax.optimization_barrier(q_residual)
    grouped = jnp.stack(
        tuple(
            linear(query_input, weight, output_dtype=jnp.float32)
            for weight in query_weight_aliases
        )
    )
    projected_query = lax.optimization_barrier(grouped)[0]

    with jax.default_matmul_precision("highest"):
        projected_head_weight = linear(
            lax.optimization_barrier(normalized),
            head_weight,
            output_dtype=jnp.float32,
        ) * jnp.float32(contract.num_heads**-0.5)
    query = projected_query.reshape(1, local_heads, contract.head_dim)
    cos, sin = rotary_cos_sin(
        position,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        query[..., : contract.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=contract.interleaved_rotary,
    )
    return (
        jnp.concatenate(
            (rotated, query[..., contract.rotary_dim :]), axis=-1
        ).astype(jnp.float32),
        projected_head_weight.astype(jnp.float32),
    )


def ws32_exact_dsa_current_key(
    normalized: Any,
    materialized_wk: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    position: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> Any:
    """Apply the DB527 normalized/wk/divide-sqrt recurrent-key boundary."""

    if normalized.shape != (1, contract.hidden_size) or (
        normalized.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 exact DSA normalized geometry drifted")
    if materialized_wk.shape != (contract.head_dim, contract.hidden_size) or (
        materialized_wk.dtype != jnp.float32
    ):
        raise ValueError("WS32 exact DSA wk must be one complete FP32 owner")
    if key_norm_weight.shape != (contract.head_dim,) or (
        key_norm_bias.shape != key_norm_weight.shape
    ):
        raise ValueError("WS32 exact DSA key norm geometry drifted")
    normalized_input = lax.optimization_barrier(normalized)
    projected_key = linear(
        normalized_input,
        materialized_wk,
        output_dtype=jnp.float32,
    )
    return dsa_index_keys_from_projection(
        projected_key,
        key_norm_weight,
        key_norm_bias,
        position,
        contract=contract,
        key_norm_mode="divide_sqrt",
    ).astype(jnp.float32)


def ws32_index_share_attention_mapped(
    residual_local: Any,
    prepared: Ws32PreparedAttention,
    cache_local: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    weights: Ws32AttentionWeights,
    *,
    expert_axis: str = "expert",
    contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(
        local_parallel_size=8
    ),
    block_shape: tuple[int, int] = (128, 128),
    rope_theta: float = 8_000_000.0,
    main_rope_table_row: Any | None = None,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(
        segment_block=512
    ),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    add_residual: bool = True,
) -> Ws32AttentionResult:
    """Consume exact DSA state and execute one WS32 sparse-MLA update.

    ``main_rope_table_row`` is the accepted GLM runtime's host BF16 ``cos|sin``
    row for this position (spec §23.8).  When it is None the main-attention
    rotary is evaluated on device, which is what the 2K/8K Gate D records used;
    on-device ``cos``/``sin`` lose accuracy with the rotary angle (measured
    2026-09-05: 5.3x the legacy's FP64 deviation at 8K, 26.6x at 262,656), so
    long-context runs pass the row and rotate with FP32 products and one final
    BF16 round, reproducing the legacy main path (DB531).
    """

    local_hidden = _require_ws32_layout(
        cache_layout, hidden_size=residual_local.shape[-1] * 4
    )
    if residual_local.shape != (1, local_hidden) or (
        residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 IndexShare residual geometry drifted")
    if not isinstance(add_residual, bool):
        raise ValueError("WS32 attention residual-add flag must be boolean")
    if prepared.normalized_local.shape != residual_local.shape or (
        prepared.q_residual.ndim != 2
        or prepared.q_residual.shape[0] != 1
        or prepared.q_residual.shape[1] <= 0
    ) or prepared.current_kv.shape != (
        1,
        contract.kv_lora_rank + contract.qk_rope_head_dim,
    ):
        raise ValueError("WS32 prepared attention state geometry drifted")
    if cache_local.ndim != 3 or cache_local.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.packed_cache_width,
    ):
        raise ValueError("WS32 attention cache geometry drifted")
    if cache_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 attention cache must remain BF16")
    if selected_positions.shape != (1, contract.top_k) or (
        selected_positions.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected positions geometry drifted")
    if selected_valid_counts.shape != (1,) or (
        selected_valid_counts.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected valid-count geometry drifted")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if weights.q_b_bits_local.shape != (
        local_heads * contract.qk_head_dim,
        prepared.q_residual.shape[-1],
    ):
        raise ValueError("WS32 q-b final-owner geometry drifted")
    combined_width = contract.qk_nope_head_dim + contract.v_head_dim
    if weights.kv_b_bits_local.shape != (
        local_heads * combined_width,
        contract.kv_lora_rank,
    ):
        raise ValueError("WS32 kv-b final-owner geometry drifted")
    if weights.o_bits_local.shape != (
        local_hidden,
        local_heads * contract.v_head_dim,
    ):
        raise ValueError("WS32 output-projection owner geometry drifted")

    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = (
        _require_decode_metadata(
            position,
            block_tables,
            context_lengths,
            owner,
            layout=cache_layout,
            physical_page_count=cache_local.shape[0],
        )
    )
    q_states = fp8_block_matmul(
        prepared.q_residual,
        weights.q_b_bits_local,
        weights.q_b_scale_local,
        config=_pallas_config(block_shape),
        interpret=linear_interpret,
    ).reshape(1, local_heads, contract.qk_head_dim)
    q_nope = q_states[..., : contract.qk_nope_head_dim]
    q_rope_unrotated = q_states[..., contract.qk_nope_head_dim :]
    if main_rope_table_row is None:
        # Statement order here is deliberately identical to the pre-§23.8 code:
        # reordering independent traced operations changes the StableHLO text and
        # therefore the acquired pins, so the default-off path stays byte-stable.
        cos, sin = rotary_cos_sin(
            position,
            rotary_dim=contract.qk_rope_head_dim,
            theta=rope_theta,
            dtype=q_rope_unrotated.dtype,
        )
        q_rope = apply_rotary(
            q_rope_unrotated,
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )

        current_latent = prepared.current_kv[..., : contract.kv_lora_rank]
        current_rope_input = prepared.current_kv[
            ...,
            contract.kv_lora_rank : contract.kv_lora_rank
            + contract.qk_rope_head_dim,
        ][:, None, :]
        current_rope = apply_rotary(
            current_rope_input,
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )[:, 0, :]
    else:
        if main_rope_table_row.shape != (contract.qk_rope_head_dim,) or (
            main_rope_table_row.dtype != jnp.bfloat16
        ):
            raise ValueError("WS32 main rotary table row geometry drifted")
        half = contract.qk_rope_head_dim // 2
        # These two slices are KV-latent work, not rotary work; they stay
        # outside the scope so `greenfield_ws32_main_rope_table` names only the
        # operations the §23.8 linter rule is about.
        current_latent = prepared.current_kv[..., : contract.kv_lora_rank]
        current_rope_input = prepared.current_kv[
            ...,
            contract.kv_lora_rank : contract.kv_lora_rank
            + contract.qk_rope_head_dim,
        ][:, None, :]
        with jax.named_scope("greenfield_ws32_main_rope_table"):
            cos = main_rope_table_row[:half][None, :]
            sin = main_rope_table_row[half:][None, :]
            q_rope = apply_rotary_fp32_final_round(
                q_rope_unrotated,
                cos[:, None, :],
                sin[:, None, :],
                interleaved=True,
            )
            current_rope = apply_rotary_fp32_final_round(
                current_rope_input,
                cos[:, None, :],
                sin[:, None, :],
                interleaved=True,
            )[:, 0, :]
    padding = contract.packed_cache_width - (
        contract.kv_lora_rank + contract.qk_rope_head_dim
    )
    current_cache_row = jnp.concatenate(
        (
            current_latent,
            current_rope,
            jnp.zeros((1, padding), dtype=jnp.bfloat16),
        ),
        axis=-1,
    ).astype(jnp.bfloat16)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_cache_row[0])

    cache_local = lax.cond(
        metadata_valid[0] & (owner == target_owner),
        write_current,
        lambda value: value,
        cache_local,
    )
    selected = SelectedPositions(
        selected_positions, selected_valid_counts
    )
    aligned = gather_stage_local_selected_kv_aligned(
        cache_local,
        block_tables,
        selected,
        context_lengths,
        layout=cache_layout,
        owner_index=owner,
    )
    with jax.named_scope("greenfield_ws32_attention/selected_cache_expert_exchange"):
        selected_cache = lax.psum(aligned.values, axis_name=expert_axis)
    q_absorbed = fp8_structured_kv_b_q_absorb(
        q_nope,
        weights.kv_b_bits_local,
        weights.kv_b_scale_local,
        config=_pallas_config(block_shape),
        interpret=linear_interpret,
    )
    with jax.named_scope("greenfield_ws32_attention/pregathered_sparse_mla"):
        attended = pregathered_sparse_mla_pallas(
            q_absorbed,
            q_rope,
            selected_cache,
            aligned.valid_counts,
            contract=replace(contract, num_heads=local_heads),
            config=sparse_attention_config,
            interpret=sparse_attention_interpret,
        )
    value_states = fp8_structured_kv_b_value(
        attended,
        weights.kv_b_bits_local,
        weights.kv_b_scale_local,
        qk_nope_head_dim=contract.qk_nope_head_dim,
        config=_pallas_config(block_shape),
        interpret=linear_interpret,
    )
    output_input = value_states.reshape(
        1, local_heads * contract.v_head_dim
    )
    update = ws32_fp8_expert_linear_pallas_mapped(
        output_input,
        weights.o_bits_local,
        weights.o_scale_local,
        expert_axis=expert_axis,
        block_shape=block_shape,
        interpret=linear_interpret,
    )
    return Ws32AttentionResult(
        residual_add(residual_local, update) if add_residual else update,
        cache_local,
        metadata_valid & aligned.contract_valid,
    )


def ws32_attention_layer_mapped(
    residual_local: Any,
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
    *,
    exact_dsa_weights: Ws32ExactDsaWeights | None = None,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(
        local_parallel_size=8
    ),
    block_shape: tuple[int, int] = (128, 128),
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(
        segment_block=512
    ),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table_row: Any | None = None,
    precomputed_normalized_local: Any | None = None,
    add_residual: bool = True,
) -> Ws32AttentionLayerResult:
    """Execute either a full-indexer or IndexShare attention layer."""

    prepared = ws32_prepare_attention_mapped(
        residual_local,
        qkv_a_weights,
        exact_dsa_weights=exact_dsa_weights,
        precomputed_normalized_local=precomputed_normalized_local,
        hidden_size=dsa_contract.hidden_size,
        block_shape=block_shape,
        linear_interpret=linear_interpret,
    )
    dsa_valid = jnp.ones((1,), dtype=jnp.bool_)
    if dsa_weights is not None:
        dsa = ws32_dsa_mapped(
            prepared,
            index_cache_local,
            position,
            block_tables,
            context_lengths,
            dsa_weights,
            exact_weights=exact_dsa_weights,
            contract=dsa_contract,
            cache_layout=cache_layout,
            block_shape=block_shape,
            linear_interpret=linear_interpret,
        )
        index_cache_local = dsa.index_cache_local
        selected_positions = dsa.selected_positions
        selected_valid_counts = dsa.selected_valid_counts
        selected_scores = dsa.selected_scores
        dsa_valid = dsa.contract_valid
    attention = ws32_index_share_attention_mapped(
        residual_local,
        prepared,
        cache_local,
        selected_positions,
        selected_valid_counts,
        position,
        block_tables,
        context_lengths,
        attention_weights,
        contract=attention_contract,
        cache_layout=cache_layout,
        block_shape=block_shape,
        main_rope_table_row=main_rope_table_row,
        sparse_attention_config=sparse_attention_config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        add_residual=add_residual,
    )
    return Ws32AttentionLayerResult(
        attention.output_local,
        attention.cache_local,
        index_cache_local,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        dsa_valid & attention.contract_valid,
    )


def ws32_transformer_layer_mapped(
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
    exact_dsa_weights: Ws32ExactDsaWeights | None = None,
    indexer_kind: str,
    mlp_kind: str,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(
        stage_size=8
    ),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(
        local_parallel_size=8
    ),
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(
        segment_block=512
    ),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table_row: Any | None = None,
) -> Ws32TransformerLayerResult:
    """Execute one complete WS32 dense or sparse transformer layer.

    The function is deliberately the only composition boundary consumed by
    the eventual 78-layer decoder.  It leaves the residual feature-sharded,
    keeps KV pages context-sharded on ``expert``, and confines every repeated
    reduction to exactly one physical mesh axis.
    """

    if indexer_kind not in ("full", "shared"):
        raise ValueError("WS32 indexer kind must be full or shared")
    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("WS32 MLP kind must be dense or sparse")
    if (dsa_weights is None) != (indexer_kind == "shared"):
        raise ValueError("WS32 full indexer alone must carry DSA weights")
    if exact_dsa_weights is not None and indexer_kind != "full":
        raise ValueError("WS32 exact DSA owners must exist only on full indexers")
    if (dense_weights is None) != (mlp_kind == "sparse"):
        raise ValueError("WS32 dense weights must exist only on dense layers")
    if (moe_weights is None) != (mlp_kind == "dense"):
        raise ValueError("WS32 MoE weights must exist only on sparse layers")
    if moe_contract.stage_size != 8:
        raise ValueError("WS32 transformer requires an expert-axis-8 contract")
    local_hidden = _require_ws32_layout(
        cache_layout, hidden_size=dsa_contract.hidden_size
    )
    if hidden_update_local.shape != (1, local_hidden) or (
        hidden_update_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 layer update must be one BF16 feature shard")
    if carried_residual_local.shape != hidden_update_local.shape or (
        carried_residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 carried residual must match the BF16 update shard")
    if moe_contract.hidden_size != dsa_contract.hidden_size:
        raise ValueError("WS32 attention and MoE hidden widths disagree")
    if post_attention_norm_weight_local.shape != (local_hidden,) or (
        post_attention_norm_weight_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 post-attention norm owner geometry drifted")
    if incoming_contract_valid.shape != (1,) or (
        incoming_contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("WS32 incoming layer health must be one boolean row")

    normalized_input, combined_residual = ws32_fused_add_rms_norm_mapped(
        hidden_update_local,
        carried_residual_local,
        qkv_a_weights.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size,
        epsilon=rms_norm_epsilon,
    )
    attention = ws32_attention_layer_mapped(
        combined_residual,
        cache_local,
        index_cache_local,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        position,
        block_tables,
        context_lengths,
        qkv_a_weights,
        attention_weights,
        dsa_weights,
        exact_dsa_weights=exact_dsa_weights,
        dsa_contract=dsa_contract,
        attention_contract=attention_contract,
        cache_layout=cache_layout,
        block_shape=block_shape,
        sparse_attention_config=sparse_attention_config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        main_rope_table_row=main_rope_table_row,
        precomputed_normalized_local=normalized_input,
        add_residual=False,
    )
    normalized_mlp, post_attention_residual = (
        ws32_fused_add_rms_norm_mapped(
            attention.output_local,
            combined_residual,
            post_attention_norm_weight_local,
            global_hidden_size=moe_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
    )
    mlp = ws32_mlp_mapped(
        post_attention_residual,
        post_attention_norm_weight_local,
        dense_weights,
        moe_weights,
        mlp_kind=mlp_kind,
        contract=moe_contract,
        block_shape=block_shape,
        rms_norm_epsilon=rms_norm_epsilon,
        linear_interpret=linear_interpret,
        precomputed_normalized_local=normalized_mlp,
        add_residual=False,
    )

    return Ws32TransformerLayerResult(
        mlp.output_local,
        post_attention_residual,
        normalized_input,
        attention.cache_local,
        attention.index_cache_local,
        attention.selected_positions,
        attention.selected_valid_counts,
        attention.selected_scores,
        mlp.route_indices,
        mlp.route_weights,
        incoming_contract_valid & attention.contract_valid,
    )


def ws32_mlp_mapped(
    post_attention_residual_local: Any,
    post_attention_norm_weight_local: Any,
    dense_weights: Ws32DenseWeights | Ws32StrategyNdDenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    *,
    mlp_kind: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(
        stage_size=8
    ),
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
    precomputed_normalized_local: Any | None = None,
    add_residual: bool = True,
) -> Ws32MlpResult:
    """Apply post-attention normalization and the exact WS32 MLP branch."""

    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("WS32 MLP kind must be dense or sparse")
    if (dense_weights is None) != (mlp_kind == "sparse"):
        raise ValueError("WS32 dense weights must exist only on dense layers")
    if (moe_weights is None) != (mlp_kind == "dense"):
        raise ValueError("WS32 MoE weights must exist only on sparse layers")
    if contract.stage_size != 8:
        raise ValueError("WS32 MLP requires an expert-axis-8 contract")
    if contract.hidden_size % 4:
        raise ValueError("WS32 MLP hidden width must divide over feature axis 4")
    local_hidden = contract.hidden_size // 4
    if post_attention_residual_local.shape != (1, local_hidden) or (
        post_attention_residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 MLP residual must be one BF16 feature shard")
    if post_attention_norm_weight_local.shape != (local_hidden,) or (
        post_attention_norm_weight_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 MLP norm owner geometry drifted")
    if not isinstance(add_residual, bool):
        raise ValueError("WS32 MLP residual-add flag must be boolean")

    if precomputed_normalized_local is None:
        normalized = ws32_rms_norm_mapped(
            post_attention_residual_local,
            post_attention_norm_weight_local,
            global_hidden_size=contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized_local
        if normalized.shape != post_attention_residual_local.shape or (
            normalized.dtype != jnp.bfloat16
        ):
            raise ValueError("WS32 precomputed MLP normalization geometry drifted")
    if mlp_kind == "dense":
        assert dense_weights is not None
        if isinstance(dense_weights, Ws32StrategyNdDenseWeights):
            if linear_interpret:
                raise ValueError(
                    "WS32 StrategyND dense path has no interpreted fallback"
                )
            update = ws32_strategy_nd_dense_final_layout_mapped(
                normalized,
                dense_weights.merged_bits_in_out_local,
                dense_weights.merged_scale_in_out_local,
                dense_weights.down_bits_in_out_local,
                dense_weights.down_scale_in_out_local,
                block_shape=block_shape,
            )
        else:
            update = ws32_dense_pallas_mapped(
                normalized,
                dense_weights.gate_bits_local,
                dense_weights.gate_scale_local,
                dense_weights.up_bits_local,
                dense_weights.up_scale_local,
                dense_weights.down_bits_local,
                dense_weights.down_scale_local,
                block_shape=block_shape,
                interpret=linear_interpret,
            )
        route_indices = jnp.full(
            (1, contract.top_k), jnp.int32(-1), dtype=jnp.int32
        )
        route_weights = jnp.zeros((1, contract.top_k), dtype=jnp.float32)
    else:
        assert moe_weights is not None
        route_indices, route_weights = ws32_router_from_shards_mapped(
            normalized,
            moe_weights.router_weight_local,
            moe_weights.correction_bias_local,
            top_k=contract.top_k,
        )
        update = ws32_moe_pallas_from_routes_mapped(
            normalized,
            route_indices,
            route_weights,
            moe_weights.expert_gate_bits_local,
            moe_weights.expert_gate_scale_local,
            moe_weights.expert_up_bits_local,
            moe_weights.expert_up_scale_local,
            moe_weights.expert_down_bits_local,
            moe_weights.expert_down_scale_local,
            moe_weights.shared_gate_bits_local,
            moe_weights.shared_gate_scale_local,
            moe_weights.shared_up_bits_local,
            moe_weights.shared_up_scale_local,
            moe_weights.shared_down_bits_local,
            moe_weights.shared_down_scale_local,
            contract=contract,
            interpret=linear_interpret,
        )
    return Ws32MlpResult(
        (
            residual_add(post_attention_residual_local, update)
            if add_residual
            else update
        ),
        route_indices,
        route_weights,
    )
