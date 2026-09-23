"""Prefill attention of the production engine: resident BF16 projections, owner-local LSE.

``ws32_prefill_prepare_attention_mapped`` is the body of ``greenfield/kernels/ws32_prefill_attention.py``
calling the production resident projection ``prefill_linear.ws32_prefill_linear_mapped``;
``prefill_index_share_lse_mapped`` mirrors the frozen ``ws32_prefill_index_share_attention_mapped``
with the documented owner-local softmax boundary and calls the resident BF16 matmuls of
``prefill_bf16`` explicitly (S2d fold of the former function rebinding). Cache writes, causal
bounds, padded-row handling and finite operand admission remain explicit. No checkpoint format is
changed; the frozen FP8 modules stay untouched as the numerical oracle of the tests.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from ..greenfield.kernels.prefill_cache import write_prefill_cache_block
from ..greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..greenfield.kernels.reference.dsa import SelectedPositions
from ..greenfield.kernels.reference.linear import residual_add
from ..greenfield.kernels.reference.rmsnorm import rms_norm
from ..greenfield.kernels.reference.rotary import apply_rotary_fp32_final_round
from ..greenfield.kernels.ws32 import ws32_rms_norm_mapped
from ..greenfield.kernels.ws32_layer import Ws32AttentionResult, Ws32PreparedAttention
from ..greenfield.kernels.ws32_prefill_attention import _require_block
from .bf16_resident import Bf16AttentionWeights, Bf16QkvAWeights
from .lse_attention import lse_attention_mapped
from .prefill_bf16 import resident_matmul, resident_q_absorb, resident_value
from .prefill_linear import ws32_prefill_linear_mapped


def ws32_prefill_prepare_attention_mapped(
    residual_local: Any,
    weights: Bf16QkvAWeights,
    *,
    precomputed_normalized_local: Any | None = None,
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
) -> Ws32PreparedAttention:
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
    if (
        qrank <= 0
        or weights.q_a_local.shape[0] != qrank
        or (kvrank <= 0 or weights.kv_a_local.shape[0] <= kvrank)
    ):
        raise ValueError("prefill qkv-a norm geometry drifted")
    if precomputed_normalized_local is None:
        normalized = ws32_rms_norm_mapped(
            residual_local,
            weights.input_norm_weight_local,
            global_hidden_size=hidden * 4,
            epsilon=rms_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized_local
        if normalized.shape != residual_local.shape or normalized.dtype != jnp.bfloat16:
            raise ValueError("prefill precomputed normalization geometry drifted")
    q = ws32_prefill_linear_mapped(
        normalized,
        weights.q_a_local,
        reduction_axis="feature",
        interpret=linear_interpret,
    )
    kv = ws32_prefill_linear_mapped(
        normalized,
        weights.kv_a_local,
        reduction_axis="feature",
        interpret=linear_interpret,
    )
    return Ws32PreparedAttention(
        normalized,
        normalized,
        rms_norm(q, weights.q_a_norm_weight, epsilon=lora_norm_epsilon),
        jnp.concatenate(
            (
                rms_norm(
                    kv[:, :kvrank], weights.kv_a_norm_weight, epsilon=lora_norm_epsilon
                ),
                kv[:, kvrank:],
            ),
            axis=-1,
        ).astype(jnp.bfloat16),
    )


def prefill_index_share_lse_mapped(
    residual_local: Any,
    prepared: Ws32PreparedAttention,
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
) -> Ws32AttentionResult:
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
        cache_layout.packed_cache_width != contract.packed_cache_width
        or contract.num_heads % 8
    ):
        raise ValueError("prefill attention requires expert8 cache/head ownership")
    if prepared.normalized_local.shape != residual_local.shape or (
        prepared.q_residual.ndim != 2
        or prepared.q_residual.shape[0] != rows
        or prepared.q_residual.dtype != jnp.bfloat16
        or prepared.current_kv.shape
        != (rows, contract.kv_lora_rank + contract.qk_rope_head_dim)
        or prepared.current_kv.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill prepared attention geometry drifted")
    if (
        selected_positions.shape != (rows, contract.top_k)
        or selected_positions.dtype != jnp.int32
    ):
        raise ValueError("prefill selected position geometry drifted")
    if (
        selected_valid_counts.shape != (rows,)
        or selected_valid_counts.dtype != jnp.int32
    ):
        raise ValueError("prefill selected count geometry drifted")
    if main_rope_table_rows.shape != (rows, contract.qk_rope_head_dim) or (
        main_rope_table_rows.dtype != jnp.bfloat16
    ):
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
    q = resident_matmul(clean_q, weights.q_b_local, interpret=linear_interpret).reshape(rows, heads, contract.qk_head_dim)
    half = contract.qk_rope_head_dim // 2
    with jax.named_scope("greenfield_ws32_prefill_main_rope_table"):
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
    q_absorbed = resident_q_absorb(
        q[..., : contract.qk_nope_head_dim], weights.kv_b_local, interpret=linear_interpret
    )
    partial = lse_attention_mapped(
        q_absorbed, q_rope, write.cache,
        jnp.broadcast_to(block_table, (rows, block_table.shape[1])),
        selected, write.causal_lengths, contract=contract, layout=cache_layout,
        config=sparse_attention_config, interpret=sparse_attention_interpret,
        validate_finite=True,
    )
    attended = partial.output
    values = resident_value(
        attended,
        weights.kv_b_local,
        qk_nope_head_dim=contract.qk_nope_head_dim,
        interpret=linear_interpret,
    )
    update = ws32_prefill_linear_mapped(
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
        & (
            ~live
            | (
                selected_valid_counts
                == jnp.minimum(write.causal_lengths, contract.top_k)
            )
        )
        & jnp.all(jnp.isfinite(output), axis=-1)
        & (~live | finite_operands)
    )
    return Ws32AttentionResult(output, write.cache, health)
