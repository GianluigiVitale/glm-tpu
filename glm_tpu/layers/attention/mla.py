"""Absorbed-MLA attention of the production engine: prefill (resident BF16 projections,
owner-local LSE) and the decode layer's attention on the resident BF16 tables.

``prefill_prepare_attention`` calls the resident projection ``glm_tpu.layers.linear.prefill_linear``;
``prefill_index_share_lse`` is the index-share attention with the documented owner-local softmax
boundary and calls the resident BF16 matmuls explicitly (S2d fold of the former function
rebinding). Cache writes, causal bounds, padded-row handling and finite operand admission remain
explicit. No checkpoint format is changed. The research-era FP8 bodies they came from are
archived at ``archive/research-20260922``.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.kernels.sparse_mla.kernel import SparseMlaConfig, SparseAttentionResult, pregathered_sparse_mla_pallas
from glm_tpu.layers.attention.kv_cache import (
    write_prefill_cache_block,
    gather_stage_local_selected_kv,
    require_decode_metadata,
    gather_stage_local_selected_kv_aligned,
)
from glm_tpu.layers.contracts import MlaNumericalContract, StageLocalKvLayout, SelectedPositions
from glm_tpu.layers.linear import (
    residual_add,
    resident_matmul,
    resident_q_absorb,
    resident_value,
    feature_linear,
    dot_f32,
    expert_linear,
    prefill_linear,
)
from glm_tpu.layers.norm import rms_norm, sharded_rms_norm
from glm_tpu.layers.rope import apply_rotary_fp32_final_round
from glm_tpu.models.glm_moe_dsa.weights import Bf16AttentionWeights, Bf16QkvAWeights
from glm_tpu.kernels.sparse_mla.partial_kernel import gathered_partial_attention


def _require_block(value: Any) -> int:
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill attention requires WS32 expert8/feature4 mesh")
    if value.ndim != 2 or not 1 <= value.shape[0] <= 32 or value.shape[1] <= 0 or value.dtype != jnp.bfloat16:
        raise ValueError("prefill attention requires1..32 BF16 feature rows")
    return value.shape[0]


def prefill_prepare_attention(
    residual_local: Any,
    weights: Bf16QkvAWeights,
    *,
    precomputed_normalized_local: Any | None = None,
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
) -> PreparedAttention:
    """Multirow q/kv-a preparation, not the legacy-association convolution.

    Accept fused-add RMSNorm output to preserve split-residual normalization.
    Changed association needs its own real-layer and §21 decoder admission.
    """
    _require_block(residual_local)
    hidden = residual_local.shape[1]
    if weights.input_norm_weight_local.shape != (hidden,):
        raise ValueError("prefill input norm owner geometry drifted")
    for table, norm in (
        (weights.q_a_local, weights.q_a_norm_weight),
        (weights.kv_a_local, weights.kv_a_norm_weight),
    ):
        if table.ndim != 2 or table.shape[1] != hidden or norm.ndim != 1:
            raise ValueError("prefill qkv-a owner geometry drifted")
    qrank = weights.q_a_norm_weight.shape[0]
    kvrank = weights.kv_a_norm_weight.shape[0]
    if qrank <= 0 or weights.q_a_local.shape[0] != qrank or (kvrank <= 0 or weights.kv_a_local.shape[0] <= kvrank):
        raise ValueError("prefill qkv-a norm geometry drifted")
    if precomputed_normalized_local is None:
        normalized = sharded_rms_norm(
            residual_local,
            weights.input_norm_weight_local,
            global_hidden_size=hidden * 4,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized_local
        if normalized.shape != residual_local.shape or normalized.dtype != jnp.bfloat16:
            raise ValueError("prefill precomputed normalization geometry drifted")
    q = prefill_linear(
        normalized,
        weights.q_a_local,
        reduction_axis="feature",
        interpret=linear_interpret,
    )
    kv = prefill_linear(
        normalized,
        weights.kv_a_local,
        reduction_axis="feature",
        interpret=linear_interpret,
    )
    return PreparedAttention(
        normalized,
        normalized,
        rms_norm(q, weights.q_a_norm_weight, epsilon=lora_norm_epsilon),
        jnp.concatenate(
            (
                rms_norm(kv[:, :kvrank], weights.kv_a_norm_weight, epsilon=lora_norm_epsilon),
                kv[:, kvrank:],
            ),
            axis=-1,
        ).astype(jnp.bfloat16),
    )


def prefill_index_share_lse(
    residual_local: Any,
    prepared: PreparedAttention,
    cache_local: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    weights: Bf16AttentionWeights,
    *,
    main_rope_table_rows: Any,
    contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(local_parallel_size=8),
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    add_residual: bool = True,
) -> AttentionResult:
    """Write a block once, then attend with EACH query's exclusive causal bound.

    Selections are supplied by the producer/IndexShare, never generated here. Exact
    own-score selection/ties remain producer obligations. Malformed live selections
    make health false; callers must refuse the whole state, including proposed writes.
    Padded rows have zero output and cannot write/read cache. Invalid write metadata
    leaves the entire cache unchanged. Health is per row and must be combined across
    all owners before committing a complete layer/block.
    """
    rows = _require_block(residual_local)
    hidden = residual_local.shape[1]
    if not isinstance(add_residual, bool):
        raise ValueError("prefill residual-add flag must be boolean")
    if cache_layout.local_parallel_size != 8 or (
        cache_layout.packed_cache_width != contract.packed_cache_width or contract.num_heads % 8
    ):
        raise ValueError("prefill attention requires expert8 cache/head ownership")
    if prepared.normalized_local.shape != residual_local.shape or (
        prepared.q_residual.ndim != 2
        or prepared.q_residual.shape[0] != rows
        or prepared.q_residual.dtype != jnp.bfloat16
        or prepared.current_kv.shape != (rows, contract.kv_lora_rank + contract.qk_rope_head_dim)
        or prepared.current_kv.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill prepared attention geometry drifted")
    if selected_positions.shape != (rows, contract.top_k) or selected_positions.dtype != jnp.int32:
        raise ValueError("prefill selected position geometry drifted")
    if selected_valid_counts.shape != (rows,) or selected_valid_counts.dtype != jnp.int32:
        raise ValueError("prefill selected count geometry drifted")
    if main_rope_table_rows.shape != (rows, contract.qk_rope_head_dim) or (main_rope_table_rows.dtype != jnp.bfloat16):
        raise ValueError("prefill requires host BF16 main rotary rows")
    if valid_rows.shape != () or valid_rows.dtype != jnp.int32:
        raise ValueError("prefill live count must be int32 scalar")
    heads = contract.num_heads // 8
    if weights.q_b_local.shape != (
        heads * contract.qk_head_dim,
        prepared.q_residual.shape[1],
    ) or (
        weights.kv_b_local.shape
        != (
            heads * (contract.qk_nope_head_dim + contract.v_head_dim),
            contract.kv_lora_rank,
        )
        or weights.o_local.shape != (hidden, heads * contract.v_head_dim)
    ):
        raise ValueError("prefill attention weight owner geometry drifted")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    clean_q = jnp.where(live[:, None], prepared.q_residual, 0)
    kv = jnp.where(live[:, None], prepared.current_kv, 0)
    rope = jnp.where(live[:, None], main_rope_table_rows, 0)
    q = resident_matmul(clean_q, weights.q_b_local, interpret=linear_interpret).reshape(
        rows, heads, contract.qk_head_dim
    )
    half = contract.qk_rope_head_dim // 2
    with jax.named_scope("prefill_main_rope_table"):
        cos, sin = rope[:, None, :half], rope[:, None, half:]
        q_rope = apply_rotary_fp32_final_round(
            q[..., contract.qk_nope_head_dim :],
            cos,
            sin,
            interleaved=True,
        )
        key_rope = apply_rotary_fp32_final_round(
            kv[:, None, contract.kv_lora_rank :],
            cos,
            sin,
            interleaved=True,
        )[:, 0]
    cache_rows = jnp.pad(
        jnp.concatenate((kv[:, : contract.kv_lora_rank], key_rope), axis=-1),
        ((0, 0), (0, contract.packed_cache_width - kv.shape[1])),
    )
    owner = lax.axis_index("expert")
    write = write_prefill_cache_block(
        cache_local,
        cache_rows,
        block_table,
        position_offset,
        valid_rows,
        owner,
        layout=cache_layout,
    )
    selected = SelectedPositions(
        jnp.where(write.row_valid[:, None], selected_positions, -1),
        jnp.where(write.row_valid, selected_valid_counts, 0),
    )
    q_absorbed = resident_q_absorb(q[..., : contract.qk_nope_head_dim], weights.kv_b_local, interpret=linear_interpret)
    partial = lse_attention(
        q_absorbed,
        q_rope,
        write.cache,
        jnp.broadcast_to(block_table, (rows, block_table.shape[1])),
        selected,
        write.causal_lengths,
        contract=contract,
        layout=cache_layout,
        config=sparse_attention_config,
        interpret=sparse_attention_interpret,
        validate_finite=True,
    )
    attended = partial.output
    values = resident_value(
        attended,
        weights.kv_b_local,
        qk_nope_head_dim=contract.qk_nope_head_dim,
        interpret=linear_interpret,
    )
    update = prefill_linear(
        values.reshape(rows, heads * contract.v_head_dim),
        weights.o_local,
        reduction_axis="expert",
        interpret=linear_interpret,
    )
    output = residual_add(residual_local, update) if add_residual else update
    output = jnp.where(write.row_valid[:, None], output, 0)
    # Local selected-cache and gathered-query finiteness is admitted inside
    # the LSE exchange on every owner; retain the frozen projection checks.
    finite_operands = (
        jnp.all(jnp.isfinite(clean_q), axis=1)
        & jnp.all(jnp.isfinite(rope), axis=1)
        & jnp.all(jnp.isfinite(q_absorbed), axis=(1, 2))
        & jnp.all(jnp.isfinite(q_rope), axis=(1, 2))
    )
    health = (
        write.valid
        & partial.contract_valid
        & (~live | (selected_valid_counts == jnp.minimum(write.causal_lengths, contract.top_k)))
        & jnp.all(jnp.isfinite(output), axis=-1)
        & (~live | finite_operands)
    )
    return AttentionResult(output, write.cache, health)


class PreparedAttention(NamedTuple):
    normalized_local: Any
    normalized_for_exact_dsa: Any
    q_residual: Any
    current_kv: Any


class AttentionResult(NamedTuple):
    output_local: Any
    cache_local: Any
    contract_valid: Any


def merge_attention_scatter(partial: SparseAttentionResult, *, expert_axis: str = "expert") -> SparseAttentionResult:
    """FP32 LSE weights and denominator, then reduce-scatter numerator by head."""
    live = jnp.isfinite(partial.logsumexp)
    # Gather only the tiny LSE table; output tensors never all-gather.
    heads = partial.logsumexp.shape[-1]
    metadata = jnp.concatenate((partial.logsumexp, partial.contract_valid.astype(jnp.float32)[:, None]), axis=-1)
    gathered = lax.all_gather(metadata, expert_axis, axis=0, tiled=False)
    all_lse = gathered[..., :heads]
    maximum = jnp.max(jnp.where(jnp.isfinite(all_lse), all_lse, -jnp.inf), axis=0)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    weight = jnp.where(live, jnp.exp(partial.logsumexp - maximum), 0.0)
    # Keep the latent payload 512-wide. Appending one denominator lane makes
    # TPU padding inflate the entire scatter; the gathered LSE already supplies it.
    weighted = partial.output.astype(jnp.float32) * weight[..., None]
    summed = lax.psum_scatter(weighted, expert_axis, scatter_dimension=1, tiled=True)
    local_heads = summed.shape[1]
    start = lax.axis_index(expert_axis) * local_heads
    local_lse = lax.dynamic_slice_in_dim(all_lse, start, local_heads, axis=2)
    local_max = lax.dynamic_slice_in_dim(maximum, start, local_heads, axis=1)
    denominator = jnp.sum(jnp.where(jnp.isfinite(local_lse), jnp.exp(local_lse - local_max[None]), 0.0), axis=0)
    output = summed / jnp.where(denominator > 0, denominator, 1.0)[..., None]
    lse = jnp.where(
        denominator > 0, local_max + jnp.log(jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny)), -jnp.inf
    )
    valid = jnp.all(gathered[..., heads] == 1.0, axis=0)
    return SparseAttentionResult(output.astype(partial.output.dtype), lse, valid)


def lse_attention(
    query_nope: Any,
    query_rope: Any,
    cache: Any,
    block_tables: Any,
    selected: SelectedPositions,
    context_lengths: Any,
    *,
    contract: MlaNumericalContract,
    layout: StageLocalKvLayout,
    config: SparseMlaConfig = SparseMlaConfig(),
    interpret: bool = False,
    expert_axis: str = "expert",
    validate_finite: bool = False,
) -> SparseAttentionResult:
    """Gather head-sharded queries, attend owned keys, return local head outputs."""
    if type(validate_finite) is not bool:
        raise ValueError("finite admission flag must be a static boolean")
    owners = lax.axis_size(expert_axis)
    if layout.local_parallel_size != owners or contract.num_heads % owners:
        raise ValueError("LSE attention head/cache ownership disagrees with mesh")
    rows = query_nope.shape[0]
    if (
        query_nope.shape != (rows, contract.num_heads // owners, contract.kv_lora_rank)
        or query_rope.shape != (rows, contract.num_heads // owners, contract.qk_rope_head_dim)
        or query_nope.dtype != jnp.bfloat16
        or query_rope.dtype != jnp.bfloat16
        or cache.dtype != jnp.bfloat16
    ):
        raise ValueError("LSE attention requires BF16 queries/cache at the declared geometry")
    if selected.positions.shape != (rows, contract.top_k):
        raise ValueError("LSE selected positions must match query rows and contract top_k")
    with jax.named_scope("lse_attention"):
        packed = jnp.concatenate((query_nope, query_rope), axis=-1)
        queries = lax.all_gather(packed, expert_axis, axis=1, tiled=True)

        def partial_from_segment(segment, local_contract):
            output, lse = gathered_partial_attention(
                queries,
                segment.values,
                segment.valid_counts,
                contract=local_contract,
                config=config,
                interpret=interpret,
            )
            valid = segment.contract_valid
            if validate_finite:
                valid = valid & jnp.all(jnp.isfinite(segment.values), axis=(1, 2))
                valid = valid & jnp.all(jnp.isfinite(queries), axis=(1, 2))
            return SparseAttentionResult(output, lse, valid)

        segment = gather_stage_local_selected_kv(
            cache,
            block_tables,
            selected,
            context_lengths,
            layout=layout,
            owner_index=lax.axis_index(expert_axis),
        )
        partial = partial_from_segment(segment, contract)
        return merge_attention_scatter(partial, expert_axis=expert_axis)


def prepare_attention_bf16(
    residual_local: Any,
    weights: Bf16QkvAWeights,
    *,
    normalized: Any,
    feature_axis: str,
    lora_norm_epsilon: float = 1e-5,
) -> PreparedAttention:
    """Mirror of ``ws32_prepare_attention_mapped`` (raw path) on BF16 tables."""

    kv_lora_rank = weights.kv_a_norm_weight.shape[0]
    q_a = feature_linear(normalized, weights.q_a_local, feature_axis)
    q_residual = rms_norm(q_a, weights.q_a_norm_weight, epsilon=lora_norm_epsilon)
    projected_kv = feature_linear(normalized, weights.kv_a_local, feature_axis)
    current_kv = jnp.concatenate(
        (
            rms_norm(projected_kv[..., :kv_lora_rank], weights.kv_a_norm_weight, epsilon=lora_norm_epsilon),
            projected_kv[..., kv_lora_rank:],
        ),
        axis=-1,
    ).astype(jnp.bfloat16)
    return PreparedAttention(normalized, normalized, q_residual, current_kv)


def index_share_attention_bf16(
    residual_local: Any,
    prepared: PreparedAttention,
    cache_local: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    weights: Bf16AttentionWeights,
    *,
    expert_axis: str,
    contract: MlaNumericalContract,
    cache_layout: StageLocalKvLayout,
    main_rope_table_row: Any,
    sparse_attention_config: SparseMlaConfig,
    sparse_attention_interpret: bool,
) -> AttentionResult:
    """Mirror of ``ws32_index_share_attention_mapped`` (host rotary table path) on BF16 tables."""

    if main_rope_table_row is None:
        raise ValueError("bf16 IndexShare attention mirrors the host main-rotary path only")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        owner,
        layout=cache_layout,
        physical_page_count=cache_local.shape[0],
    )
    q_states = (
        dot_f32(prepared.q_residual, weights.q_b_local)
        .astype(jnp.bfloat16)
        .reshape(1, local_heads, contract.qk_head_dim)
    )
    q_nope = q_states[..., : contract.qk_nope_head_dim]
    q_rope_unrotated = q_states[..., contract.qk_nope_head_dim :]
    half = contract.qk_rope_head_dim // 2
    current_latent = prepared.current_kv[..., : contract.kv_lora_rank]
    current_rope_input = prepared.current_kv[
        ..., contract.kv_lora_rank : contract.kv_lora_rank + contract.qk_rope_head_dim
    ][:, None, :]
    with jax.named_scope("main_rope_table"):
        cos = main_rope_table_row[:half][None, :]
        sin = main_rope_table_row[half:][None, :]
        q_rope = apply_rotary_fp32_final_round(q_rope_unrotated, cos[:, None, :], sin[:, None, :], interleaved=True)
        current_rope = apply_rotary_fp32_final_round(
            current_rope_input, cos[:, None, :], sin[:, None, :], interleaved=True
        )[:, 0, :]
    padding = contract.packed_cache_width - (contract.kv_lora_rank + contract.qk_rope_head_dim)
    current_cache_row = jnp.concatenate(
        (current_latent, current_rope, jnp.zeros((1, padding), dtype=jnp.bfloat16)), axis=-1
    ).astype(jnp.bfloat16)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_cache_row[0])

    cache_local = lax.cond(metadata_valid[0] & (owner == target_owner), write_current, lambda v: v, cache_local)
    selected = SelectedPositions(selected_positions, selected_valid_counts)
    # Structured kv_b: head h owns rows [h*448, h*448+448) = [192 key rows | 256 value rows] x 512 latents.
    combined = contract.qk_nope_head_dim + contract.v_head_dim
    kv_b = weights.kv_b_local.reshape(local_heads, combined, contract.kv_lora_rank)
    key_rows = kv_b[:, : contract.qk_nope_head_dim, :]  # [h, 192, 512]
    value_rows = kv_b[:, contract.qk_nope_head_dim :, :]  # [h, 256, 512]
    q_absorbed = jnp.einsum("rhq,hqk->rhk", q_nope, key_rows, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    aligned = gather_stage_local_selected_kv_aligned(
        cache_local,
        block_tables,
        selected,
        context_lengths,
        layout=cache_layout,
        owner_index=owner,
    )
    with jax.named_scope("bf16_attention/selected_cache_expert_exchange"):
        selected_cache = lax.psum(aligned.values, axis_name=expert_axis)
    with jax.named_scope("bf16_attention/sparse_mla"):
        attended = pregathered_sparse_mla_pallas(
            q_absorbed,
            q_rope,
            selected_cache,
            aligned.valid_counts,
            contract=replace(contract, num_heads=local_heads),
            config=sparse_attention_config,
            interpret=sparse_attention_interpret,
        )
    attention_valid = aligned.contract_valid
    value_states = jnp.einsum("rhk,hvk->rhv", attended, value_rows, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    output_input = value_states.reshape(1, local_heads * contract.v_head_dim)
    update = expert_linear(output_input, weights.o_local, expert_axis)
    return AttentionResult(update, cache_local, metadata_valid & attention_valid)
