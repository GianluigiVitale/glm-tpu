"""Raw-FP8 batch-one layer bodies for a topology-local stage group.

These bodies run *inside* a surrounding global ``shard_map``.  They therefore
accept already-local final-owner shards and an explicit local slot rather than
constructing a nested mesh.  Every repeated collective is restricted to the
provided stage-local axis groups.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import jax
from jax import lax
import jax.numpy as jnp

from .pallas import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_up_gate,
    fp8_selected_swiglu_down,
    fp8_selected_up_gate,
)
from .reference.fp8 import dequantize_fp8_bits_block_weight
from .reference.attention import (
    MlaNumericalContract,
    SparseAttentionResult,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    combine_stage_local_attention,
    gather_stage_local_selected_kv,
    sparse_mla_attention,
)
from .reference.dsa import (
    DsaNumericalContract,
    SelectedPositions,
    dsa_index_keys,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates,
)
from .reference.linear import linear, residual_add, silu
from .reference.moe import GlmMoeNumericalContract, route_glm_noaux_tc
from .reference.rmsnorm import rms_norm
from .reference.rotary import apply_rotary, rotary_cos_sin


class StageLocalDsaFp8Result(NamedTuple):
    """Updated local key cache and compact exact DSA selection."""

    index_cache: Any
    selected_positions: Any
    valid_counts: Any
    contract_valid: Any


class StageLocalIndexShareFp8Result(NamedTuple):
    """Updated local KV cache and post-attention residual."""

    output: Any
    cache: Any
    contract_valid: Any


def _axis_groups(
    groups: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if groups is None:
        return None
    value = tuple(tuple(int(rank) for rank in group) for group in groups)
    if not value or any(not group for group in value):
        raise ValueError("stage-local axis groups must be non-empty")
    return value


def _require_decode_metadata(
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    local_slot: Any,
    *,
    layout: StageLocalKvLayout,
    physical_page_count: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return safe current-row indices plus a device-resident health bit."""

    if position.shape != (1,) or not jnp.issubdtype(position.dtype, jnp.integer):
        raise ValueError("decode position must be one integer row")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1:
        raise ValueError("block tables must contain one decode row")
    if block_tables.shape[1] == 0 or block_tables.dtype != jnp.int32:
        raise ValueError("block tables must expose int32 logical pages")
    if context_lengths.shape != (1,) or context_lengths.dtype != jnp.int32:
        raise ValueError("context lengths must contain one int32 row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    if physical_page_count <= 0:
        raise ValueError("physical page count must be positive")

    capacity = block_tables.shape[1] * layout.logical_page_size
    current = position[0].astype(jnp.int32)
    length = context_lengths[0]
    safe_current = jnp.clip(current, jnp.int32(0), jnp.int32(capacity - 1))
    logical_page = safe_current // jnp.int32(layout.logical_page_size)
    safe_logical_page = jnp.clip(
        logical_page, jnp.int32(0), jnp.int32(block_tables.shape[1] - 1)
    )
    physical_page = block_tables[0, safe_logical_page]
    physical_ok = (physical_page >= 0) & (physical_page < physical_page_count)
    safe_physical_page = jnp.clip(
        physical_page, jnp.int32(0), jnp.int32(physical_page_count - 1)
    )
    within_page = safe_current % jnp.int32(layout.logical_page_size)
    target_owner = within_page // jnp.int32(layout.local_rows_per_page)
    local_row = within_page % jnp.int32(layout.local_rows_per_page)

    page_ids = block_tables[0]
    page_slots = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    required_pages = (
        jnp.maximum(length, jnp.int32(0))
        + jnp.int32(layout.logical_page_size - 1)
    ) // jnp.int32(layout.logical_page_size)
    live_pages = page_slots < required_pages
    page_table_ok = jnp.all(
        jnp.where(
            live_pages,
            (page_ids >= 0) & (page_ids < physical_page_count),
            True,
        )
    )
    ordered_live_pages = jnp.sort(
        jnp.where(live_pages, page_ids, jnp.iinfo(jnp.int32).max)
    )
    adjacent_slots = jnp.arange(
        1, block_tables.shape[1], dtype=jnp.int32
    )
    page_table_unique = jnp.all(
        jnp.where(
            adjacent_slots < required_pages,
            ordered_live_pages[1:] != ordered_live_pages[:-1],
            True,
        )
    )
    metadata_valid = (
        (length > 0)
        & (length <= capacity)
        & (current == length - 1)
        & (current >= 0)
        & (local_slot >= 0)
        & (local_slot < layout.local_parallel_size)
        & physical_ok
        & page_table_ok
        & page_table_unique
    )
    return (
        safe_physical_page,
        local_row,
        target_owner,
        safe_current,
        metadata_valid[None],
    )


def _local_dsa_query(
    q_residual: Any,
    normalized: Any,
    query_weight: Any,
    head_weight: Any,
    position: Any,
    *,
    contract: DsaNumericalContract,
) -> tuple[Any, Any]:
    local_heads = query_weight.shape[0] // contract.head_dim
    if query_weight.shape != (
        local_heads * contract.head_dim,
        contract.q_lora_rank,
    ):
        raise ValueError("local DSA query shard has an invalid shape")
    if head_weight.shape != (local_heads, contract.hidden_size):
        raise ValueError("local DSA head-weight shard has an invalid shape")
    with jax.default_matmul_precision("highest"):
        query = linear(
            q_residual, query_weight, output_dtype=jnp.float32
        ).reshape(1, local_heads, contract.head_dim)
        weights = linear(
            normalized, head_weight, output_dtype=jnp.float32
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
    return (
        jnp.concatenate(
            (rotated, query[..., contract.rotary_dim :]), axis=-1
        ).astype(jnp.float32),
        weights.astype(jnp.float32),
    )


def _reshape_gathered_heads(
    value: Any,
    *,
    num_heads: int,
    head_width: int,
) -> Any:
    return jnp.transpose(value, (1, 0, 2, 3)).reshape(
        1, num_heads, head_width
    )


def stage_local_dsa_fp8_mapped(
    residual: Any,
    index_cache: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    q_a_bits: Any,
    q_a_scale: Any,
    q_a_norm_weight: Any,
    wq_b_bits: Any,
    wq_b_scale: Any,
    wk_bits: Any,
    wk_scale: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    head_weight: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: DsaNumericalContract = DsaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    precomputed_normalized: Any | None = None,
    precomputed_q_residual: Any | None = None,
) -> StageLocalDsaFp8Result:
    """Write one BF16 index key, score local pages, and merge exact top-k.

    The function is intended for a surrounding global ``shard_map``.  It
    dequantizes only the already-local final-owner query shard, communicates
    inside explicit stage groups, and never materializes a global key cache.
    Invalid page metadata is clipped before every read/write and propagated as
    a device-resident false health predicate.
    """

    groups = _axis_groups(axis_index_groups)
    if residual.shape != (1, contract.hidden_size):
        raise ValueError("DSA residual must contain one exact hidden row")
    if index_cache.ndim != 3 or index_cache.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.head_dim,
    ):
        raise ValueError("local DSA key cache has an invalid shape")
    if residual.dtype != jnp.bfloat16 or index_cache.dtype != jnp.bfloat16:
        raise ValueError("DSA residual and key cache must remain BF16")
    if cache_layout.local_parallel_size <= 0 or (
        contract.num_heads % cache_layout.local_parallel_size
    ):
        raise ValueError("DSA heads must divide over the local stage")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if input_norm_weight.shape != (contract.hidden_size,):
        raise ValueError("DSA input norm shape is invalid")
    if q_a_bits.shape != (contract.q_lora_rank, contract.hidden_size):
        raise ValueError("DSA q_a FP8 shape is invalid")
    if q_a_norm_weight.shape != (contract.q_lora_rank,):
        raise ValueError("DSA q_a norm shape is invalid")
    if wq_b_bits.shape != (
        local_heads * contract.head_dim,
        contract.q_lora_rank,
    ):
        raise ValueError("DSA local wq_b FP8 shape is invalid")
    if wk_bits.shape != (contract.head_dim, contract.hidden_size):
        raise ValueError("DSA wk FP8 shape is invalid")
    if key_norm_weight.shape != (contract.head_dim,) or key_norm_bias.shape != (
        contract.head_dim,
    ):
        raise ValueError("DSA key norm shapes are invalid")
    if head_weight.shape != (local_heads, contract.hidden_size):
        raise ValueError("DSA local head-weight shape is invalid")

    (
        physical_page,
        local_row,
        target_owner,
        _,
        metadata_valid,
    ) = _require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        local_slot,
        layout=cache_layout,
        physical_page_count=index_cache.shape[0],
    )
    local_wq_b_weight = dequantize_fp8_bits_block_weight(
        wq_b_bits, wq_b_scale, block_shape=block_shape
    )
    wk_weight = dequantize_fp8_bits_block_weight(
        wk_bits, wk_scale, block_shape=block_shape
    )
    if (precomputed_normalized is None) != (precomputed_q_residual is None):
        raise ValueError("DSA shared q_a intermediates must be supplied together")
    if precomputed_normalized is None:
        q_a_weight = dequantize_fp8_bits_block_weight(
            q_a_bits, q_a_scale, block_shape=block_shape
        )
        normalized = rms_norm(
            residual, input_norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            linear(normalized, q_a_weight),
            q_a_norm_weight,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized
        q_residual = precomputed_q_residual
        if normalized.shape != residual.shape or normalized.dtype != residual.dtype:
            raise ValueError("DSA precomputed normalized residual is invalid")
        if q_residual.shape != (1, contract.q_lora_rank) or (
            q_residual.dtype != residual.dtype
        ):
            raise ValueError("DSA precomputed q residual is invalid")
    local_query, local_head_weights = _local_dsa_query(
        q_residual,
        normalized,
        local_wq_b_weight,
        head_weight,
        position,
        contract=contract,
    )
    query_width = local_heads * contract.head_dim
    packed_query = jnp.concatenate(
        (local_query.reshape(1, query_width), local_head_weights), axis=-1
    )
    gathered_query = lax.all_gather(
        packed_query,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    query = jnp.transpose(
        gathered_query[..., :query_width], (1, 0, 2)
    ).reshape(1, contract.num_heads, contract.head_dim)
    gathered_head_weights = jnp.transpose(
        gathered_query[..., query_width:], (1, 0, 2)
    ).reshape(1, contract.num_heads)

    current_key = dsa_index_keys(
        normalized,
        wk_weight,
        key_norm_weight,
        key_norm_bias,
        position,
        contract=contract,
    ).astype(index_cache.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_key[0])

    index_cache = lax.cond(
        metadata_valid[0] & (local_slot == target_owner),
        write_current,
        lambda value: value,
        index_cache,
    )

    page_ids = block_tables[0]
    page_ok = (page_ids >= 0) & (page_ids < index_cache.shape[0])
    safe_pages = jnp.clip(
        page_ids, jnp.int32(0), jnp.int32(index_cache.shape[0] - 1)
    )
    logical_cache = jnp.take(index_cache, safe_pages, axis=0)
    logical_pages = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    local_rows = jnp.arange(
        cache_layout.local_rows_per_page, dtype=jnp.int32
    )
    global_positions = (
        logical_pages[:, None] * jnp.int32(cache_layout.logical_page_size)
        + local_slot.astype(jnp.int32)
        * jnp.int32(cache_layout.local_rows_per_page)
        + local_rows[None, :]
    )
    global_positions = jnp.where(
        page_ok[:, None], global_positions, jnp.int32(-1)
    ).reshape(-1)
    local_keys = logical_cache.reshape(-1, contract.head_dim)
    local_scores = dsa_scores(query, local_keys, gathered_head_weights)
    candidate_scores, candidate_positions = local_topk_candidates(
        local_scores,
        global_positions,
        context_lengths,
        top_k=contract.top_k,
    )
    gathered_scores = lax.all_gather(
        candidate_scores,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    gathered_positions = lax.all_gather(
        candidate_positions,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    selected = merge_topk_candidates(
        gathered_scores,
        gathered_positions,
        context_lengths,
        top_k=contract.top_k,
        global_context_size=(
            block_tables.shape[1] * cache_layout.logical_page_size
        ),
    )
    selection_valid = canonicalize_selected_positions(
        selected
    ).contract_valid
    return StageLocalDsaFp8Result(
        index_cache,
        selected.positions,
        selected.valid_counts,
        metadata_valid & selection_valid,
    )


def stage_local_index_share_fp8_mapped(
    residual: Any,
    cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    q_a_bits: Any,
    q_a_scale: Any,
    q_a_norm_weight: Any,
    q_b_bits: Any,
    q_b_scale: Any,
    kv_a_bits: Any,
    kv_a_scale: Any,
    kv_a_norm_weight: Any,
    kv_b_bits: Any,
    kv_b_scale: Any,
    o_bits: Any,
    o_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    precomputed_normalized: Any | None = None,
    precomputed_q_residual: Any | None = None,
) -> StageLocalIndexShareFp8Result:
    """Consume compact DSA state and execute raw-FP8 stage-local sparse MLA."""

    groups = _axis_groups(axis_index_groups)
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("IndexShare residual must contain exactly one row")
    hidden = residual.shape[1]
    if cache.ndim != 3 or cache.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.packed_cache_width,
    ):
        raise ValueError("local IndexShare cache has an invalid shape")
    if residual.dtype != jnp.bfloat16 or cache.dtype != jnp.bfloat16:
        raise ValueError("IndexShare residual and cache must remain BF16")
    if contract.num_heads % cache_layout.local_parallel_size:
        raise ValueError("attention heads must divide over the local stage")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if selected_positions.shape != (1, contract.top_k):
        raise ValueError("IndexShare selected-position shape is invalid")
    if selected_valid_counts.shape != (1,):
        raise ValueError("IndexShare selected-count shape is invalid")
    if input_norm_weight.shape != (hidden,):
        raise ValueError("IndexShare input norm shape is invalid")
    if q_a_bits.shape[1:] != (hidden,) or q_a_bits.ndim != 2:
        raise ValueError("IndexShare q_a FP8 shape is invalid")
    q_lora_rank = q_a_bits.shape[0]
    if q_a_norm_weight.shape != (q_lora_rank,):
        raise ValueError("IndexShare q_a norm shape is invalid")
    if q_b_bits.shape != (
        local_heads * contract.qk_head_dim,
        q_lora_rank,
    ):
        raise ValueError("IndexShare local q_b FP8 shape is invalid")
    if kv_a_bits.shape != (
        contract.kv_lora_rank + contract.qk_rope_head_dim,
        hidden,
    ):
        raise ValueError("IndexShare kv_a FP8 shape is invalid")
    if kv_a_norm_weight.shape != (contract.kv_lora_rank,):
        raise ValueError("IndexShare kv_a norm shape is invalid")
    combined_width = contract.qk_nope_head_dim + contract.v_head_dim
    if kv_b_bits.shape != (
        local_heads * combined_width,
        contract.kv_lora_rank,
    ):
        raise ValueError("IndexShare local kv_b FP8 shape is invalid")
    if o_bits.shape != (hidden, local_heads * contract.v_head_dim):
        raise ValueError("IndexShare local output FP8 shape is invalid")

    (
        physical_page,
        local_row,
        target_owner,
        _,
        metadata_valid,
    ) = _require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        local_slot,
        layout=cache_layout,
        physical_page_count=cache.shape[0],
    )
    local_q_b_weight = dequantize_fp8_bits_block_weight(
        q_b_bits, q_b_scale, block_shape=block_shape
    )
    kv_a_weight = dequantize_fp8_bits_block_weight(
        kv_a_bits, kv_a_scale, block_shape=block_shape
    )
    local_kv_b_weight = dequantize_fp8_bits_block_weight(
        kv_b_bits, kv_b_scale, block_shape=block_shape
    )
    local_o_weight = dequantize_fp8_bits_block_weight(
        o_bits, o_scale, block_shape=block_shape
    )
    if (precomputed_normalized is None) != (precomputed_q_residual is None):
        raise ValueError(
            "IndexShare shared q_a intermediates must be supplied together"
        )
    if precomputed_normalized is None:
        q_a_weight = dequantize_fp8_bits_block_weight(
            q_a_bits, q_a_scale, block_shape=block_shape
        )
        normalized = rms_norm(
            residual, input_norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            linear(normalized, q_a_weight),
            q_a_norm_weight,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized
        q_residual = precomputed_q_residual
        if normalized.shape != residual.shape or normalized.dtype != residual.dtype:
            raise ValueError(
                "IndexShare precomputed normalized residual is invalid"
            )
        if q_residual.shape != (1, q_lora_rank) or (
            q_residual.dtype != residual.dtype
        ):
            raise ValueError("IndexShare precomputed q residual is invalid")
    q_states = linear(q_residual, local_q_b_weight).reshape(
        1, local_heads, contract.qk_head_dim
    )
    q_nope = q_states[..., : contract.qk_nope_head_dim]
    q_rope_unrotated = q_states[..., contract.qk_nope_head_dim :]
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

    current_kv = linear(normalized, kv_a_weight)
    current_latent = rms_norm(
        current_kv[..., : contract.kv_lora_rank],
        kv_a_norm_weight,
        epsilon=rms_norm_epsilon,
    )
    current_rope = apply_rotary(
        current_kv[
            ...,
            contract.kv_lora_rank : contract.kv_lora_rank
            + contract.qk_rope_head_dim,
        ][:, None, :],
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
            jnp.zeros((1, padding), dtype=normalized.dtype),
        ),
        axis=-1,
    ).astype(cache.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_cache_row[0])

    cache = lax.cond(
        metadata_valid[0] & (local_slot == target_owner),
        write_current,
        lambda value: value,
        cache,
    )

    local_kv_b = local_kv_b_weight.reshape(
        local_heads, combined_width, contract.kv_lora_rank
    )
    local_weight_uk = local_kv_b[:, : contract.qk_nope_head_dim, :]
    local_weight_uv = jnp.transpose(
        local_kv_b[:, contract.qk_nope_head_dim :, :], (0, 2, 1)
    )
    q_absorbed_local = jnp.einsum(
        "rhp,hpl->rhl",
        q_nope.astype(jnp.float32),
        local_weight_uk.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(cache.dtype)
    packed_query = jnp.concatenate((q_absorbed_local, q_rope), axis=-1)
    gathered_query = lax.all_gather(
        packed_query,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    full_query = _reshape_gathered_heads(
        gathered_query,
        num_heads=contract.num_heads,
        head_width=contract.kv_lora_rank + contract.qk_rope_head_dim,
    )
    q_absorbed = full_query[..., : contract.kv_lora_rank]
    full_q_rope = full_query[..., contract.kv_lora_rank :]
    selected = SelectedPositions(selected_positions, selected_valid_counts)
    segment = gather_stage_local_selected_kv(
        cache,
        block_tables,
        selected,
        context_lengths,
        layout=cache_layout,
        owner_index=local_slot,
    )
    partial = sparse_mla_attention(
        q_absorbed, full_q_rope, segment, contract=contract
    )
    gathered_outputs = lax.all_gather(
        partial.output,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    gathered_lse = lax.all_gather(
        partial.logsumexp,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    gathered_validity = lax.all_gather(
        partial.contract_valid,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    combined: SparseAttentionResult = combine_stage_local_attention(
        gathered_outputs, gathered_lse, gathered_validity
    )
    attended_local = lax.dynamic_slice_in_dim(
        combined.output,
        local_slot.astype(jnp.int32) * jnp.int32(local_heads),
        local_heads,
        axis=1,
    )
    value_states = jnp.einsum(
        "rhl,hlv->rhv",
        attended_local.astype(jnp.float32),
        local_weight_uv.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(residual.dtype)
    local_update = linear(
        value_states.reshape(1, local_heads * contract.v_head_dim),
        local_o_weight,
    )
    update = lax.psum(
        local_update,
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    output = residual_add(residual, update)
    return StageLocalIndexShareFp8Result(
        output,
        cache,
        metadata_valid & combined.contract_valid,
    )


def stage_local_dense_fp8_mapped(
    residual: Any,
    norm_weight: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    axis_name: str,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    epsilon: float = 1e-5,
) -> Any:
    """Execute one dense SwiGLU from local raw shards and one local combine."""

    groups = _axis_groups(axis_index_groups)
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("dense decode residual must contain exactly one row")
    hidden = residual.shape[1]
    if norm_weight.shape != (hidden,):
        raise ValueError("dense norm shape disagrees with hidden size")
    if gate_bits.shape != up_bits.shape or gate_bits.shape[1] != hidden:
        raise ValueError("dense local gate/up shards are invalid")
    if down_bits.shape != (hidden, gate_bits.shape[0]):
        raise ValueError("dense local down shard is invalid")
    gate_weight = dequantize_fp8_bits_block_weight(
        gate_bits, gate_scale, block_shape=block_shape
    )
    up_weight = dequantize_fp8_bits_block_weight(
        up_bits, up_scale, block_shape=block_shape
    )
    down_weight = dequantize_fp8_bits_block_weight(
        down_bits, down_scale, block_shape=block_shape
    )
    normalized = rms_norm(residual, norm_weight, epsilon=epsilon)
    activated = (
        silu(linear(normalized, gate_weight))
        * linear(normalized, up_weight)
    ).astype(normalized.dtype)
    local_update = linear(activated, down_weight)
    update = lax.psum(
        local_update,
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    return residual_add(residual, update)


def stage_local_moe_fp8_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[Any, Any, Any]:
    """Route and execute selected raw-FP8 experts without a BF16 overlay."""

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size):
        raise ValueError("MoE decode hidden state must contain one exact row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    local_expert_shape = (
        contract.local_experts,
        contract.intermediate_size,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != local_expert_shape or expert_up_bits.shape != local_expert_shape:
        raise ValueError("local expert gate/up FP8 shapes are invalid")
    if expert_down_bits.shape != (
        contract.local_experts,
        contract.hidden_size,
        contract.intermediate_size,
    ):
        raise ValueError("local expert down FP8 shape is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    expert_start = local_slot.astype(jnp.int32) * jnp.int32(
        contract.local_experts
    )
    routed_parts = []
    for position in range(contract.top_k):
        global_expert = route_indices[0, position]
        owns_expert = (
            (global_expert >= expert_start)
            & (global_expert < expert_start + contract.local_experts)
        )
        local_expert = jnp.clip(
            global_expert - expert_start, 0, contract.local_experts - 1
        )

        def compute(_: None) -> Any:
            gate = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_gate_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_gate_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            up = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_up_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_up_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            down = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_down_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_down_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            activated = (
                silu(linear(hidden_states, gate))
                * linear(hidden_states, up)
            ).astype(hidden_states.dtype)
            output = linear(activated, down)
            return output * route_weights[0, position].astype(hidden_states.dtype)

        routed_parts.append(
            lax.cond(
                owns_expert,
                compute,
                lambda _: jnp.zeros_like(hidden_states),
                operand=None,
            )
        )
    local_routed = jnp.sum(
        jnp.stack(routed_parts, axis=0),
        axis=0,
        dtype=hidden_states.dtype,
    )
    shared_gate = dequantize_fp8_bits_block_weight(
        shared_gate_bits,
        shared_gate_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_up = dequantize_fp8_bits_block_weight(
        shared_up_bits,
        shared_up_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_down = dequantize_fp8_bits_block_weight(
        shared_down_bits,
        shared_down_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_activated = (
        silu(linear(hidden_states, shared_gate))
        * linear(hidden_states, shared_up)
    ).astype(hidden_states.dtype)
    local_shared = linear(shared_activated, shared_down)
    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    output = (combined[0] * scale + combined[1]).astype(hidden_states.dtype)
    return output, route_indices, route_weights


def stage_local_moe_pallas_from_routes_mapped(
    hidden_states: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> Any:
    """Execute final-layout raw-FP8 MoE work and one local combine.

    Routed tables are already packed in the Pallas MXU access order:
    gate/up ``[local_experts, hidden, intermediate]`` and down
    ``[local_experts, intermediate, hidden]``. Shared-expert shards retain
    checkpoint ``[out, in]`` orientation because their feature sharding is
    directly compatible with the standalone Pallas matmuls. No complete
    expert table is transposed or decoded by the runtime.

    The routed top-k reduction is performed in BF16 before the stage-local
    collective. Routed and shared partials occupy separate leading rows of
    the same ``psum`` and the routed factor is applied only after reduction.
    """

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size):
        raise ValueError("Pallas MoE hidden state must contain one exact row")
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError("Pallas MoE hidden state must be BF16")
    if route_indices.shape != (1, contract.top_k) or (
        route_indices.dtype != jnp.int32
    ):
        raise ValueError("Pallas MoE routes must be one exact int32 top-k row")
    if route_weights.shape != (1, contract.top_k) or (
        route_weights.dtype != jnp.float32
    ):
        raise ValueError("Pallas MoE route weights must be one FP32 top-k row")
    if local_slot.shape != () or local_slot.dtype != jnp.int32:
        raise ValueError("Pallas MoE local_slot must be an int32 scalar")
    if config.block_shape != contract.fp8_block_shape:
        raise ValueError(
            "Pallas tile block shape must match the MoE numerical contract"
        )

    expected_gate_shape = (
        contract.local_experts,
        contract.hidden_size,
        contract.intermediate_size,
    )
    expected_down_shape = (
        contract.local_experts,
        contract.intermediate_size,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != expected_gate_shape or (
        expert_up_bits.shape != expected_gate_shape
    ):
        raise ValueError(
            "Pallas routed gate/up tables must use final [G,K,N] layout"
        )
    if expert_down_bits.shape != expected_down_shape:
        raise ValueError(
            "Pallas routed down table must use final [G,K,N] layout"
        )
    shared_gate_shape = (
        contract.local_shared_intermediate,
        contract.hidden_size,
    )
    if shared_gate_bits.shape != shared_gate_shape or (
        shared_up_bits.shape != shared_gate_shape
    ):
        raise ValueError(
            "Pallas shared gate/up shards must retain [local_out,in] layout"
        )
    if shared_down_bits.shape != (
        contract.hidden_size,
        contract.local_shared_intermediate,
    ):
        raise ValueError(
            "Pallas shared down shard must retain [out,local_in] layout"
        )

    expert_start = local_slot * jnp.int32(contract.local_experts)
    gate, up = fp8_selected_up_gate(
        hidden_states,
        route_indices[0],
        expert_start,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        config=config,
        interpret=interpret,
    )
    routed_outputs = fp8_selected_swiglu_down(
        gate,
        up,
        route_indices[0],
        expert_start,
        expert_down_bits,
        expert_down_scale,
        config=config,
        interpret=interpret,
    )
    weighted_routed = (
        routed_outputs
        * route_weights[0, :, None].astype(hidden_states.dtype)
    ).astype(hidden_states.dtype)
    local_routed = jnp.sum(
        weighted_routed,
        axis=0,
        dtype=hidden_states.dtype,
    )[None, :]

    # Selected routed kernels explicitly apply every 128-wide scale block
    # inside their wider contraction tile.  The standalone shared kernels
    # consume one scalar scale per contraction tile, so keep that tile at one
    # checkpoint scale block even when routed experts use k=512.
    shared_config = Fp8BlockMatmulConfig(
        block_shape=config.block_shape,
        row_tile=config.row_tile,
        output_tile=config.output_tile,
        contraction_tile=config.block_shape[1],
        output_dtype=config.output_dtype,
        accumulator_dtype=config.accumulator_dtype,
    )
    shared_gate, shared_up = fp8_block_up_gate(
        hidden_states,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        config=shared_config,
        interpret=interpret,
    )
    shared_activated = (silu(shared_gate) * shared_up).astype(
        hidden_states.dtype
    )
    local_shared = fp8_block_matmul(
        shared_activated,
        shared_down_bits,
        shared_down_scale,
        config=shared_config,
        interpret=interpret,
    )

    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    return (combined[0] * scale + combined[1]).astype(hidden_states.dtype)


def stage_local_moe_pallas_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> tuple[Any, Any, Any]:
    """Route once, execute the final-layout Pallas MoE, and return routing."""

    if router_weight.shape != (contract.num_experts, contract.hidden_size):
        raise ValueError("Pallas MoE router weight shape is invalid")
    if correction_bias.shape != (contract.num_experts,):
        raise ValueError("Pallas MoE correction bias shape is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    output = stage_local_moe_pallas_from_routes_mapped(
        hidden_states,
        route_indices,
        route_weights,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        shared_down_bits,
        shared_down_scale,
        local_slot,
        axis_name=axis_name,
        contract=contract,
        axis_index_groups=axis_index_groups,
        config=config,
        interpret=interpret,
    )
    return output, route_indices, route_weights
