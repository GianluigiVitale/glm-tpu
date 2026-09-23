"""Opt-in causal multirow attention; not yet wired into the decoder.

Uses final WS32 raw-FP8 owners, feature4/expert8 collectives and host main-RoPE
rows. The caller owns append-frontier/prefix integrity and all-chip health.
This module never promotes repaired index keys or reuses another layer's KV.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .pallas.fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
)
from .pallas.sparse_attention import SparseMlaConfig, pregathered_sparse_mla_pallas
from .prefill_cache import write_prefill_cache_block
from .reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
    gather_stage_local_selected_kv_aligned,
)
from .reference.dsa import SelectedPositions
from .reference.linear import residual_add
from .reference.rmsnorm import rms_norm
from .reference.rotary import apply_rotary_fp32_final_round
from .ws32 import ws32_rms_norm_mapped
from .ws32_layer import (
    Ws32AttentionResult,
    Ws32AttentionWeights,
    Ws32PreparedAttention,
    Ws32QkvAWeights,
)
from .ws32_prefill_linear import ws32_prefill_linear_mapped


def _require_block(value: Any) -> int:
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill attention requires WS32 expert8/feature4 mesh")
    if (
        value.ndim != 2
        or not 1 <= value.shape[0] <= 32
        or value.shape[1] <= 0
        or value.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill attention requires1..32 BF16 feature rows")
    return value.shape[0]


def ws32_prefill_prepare_attention_mapped(
    residual_local: Any,
    weights: Ws32QkvAWeights,
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
    for bits, norm in (
        (weights.q_a_bits_local, weights.q_a_norm_weight),
        (weights.kv_a_bits_local, weights.kv_a_norm_weight),
    ):
        if bits.ndim != 2 or bits.shape[1] != hidden or norm.ndim != 1:
            raise ValueError("prefill qkv-a owner geometry drifted")
    qrank = weights.q_a_norm_weight.shape[0]
    kvrank = weights.kv_a_norm_weight.shape[0]
    if (
        qrank <= 0
        or weights.q_a_bits_local.shape[0] != qrank
        or (kvrank <= 0 or weights.kv_a_bits_local.shape[0] <= kvrank)
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
        weights.q_a_bits_local,
        weights.q_a_scale_local,
        reduction_axis="feature",
        interpret=linear_interpret,
    )
    kv = ws32_prefill_linear_mapped(
        normalized,
        weights.kv_a_bits_local,
        weights.kv_a_scale_local,
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


def ws32_prefill_index_share_attention_mapped(
    residual_local: Any,
    prepared: Ws32PreparedAttention,
    cache_local: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    weights: Ws32AttentionWeights,
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
    if weights.q_b_bits_local.shape != (
        heads * contract.qk_head_dim,
        prepared.q_residual.shape[1],
    ) or (
        weights.kv_b_bits_local.shape
        != (
            heads * (contract.qk_nope_head_dim + contract.v_head_dim),
            contract.kv_lora_rank,
        )
        or weights.o_bits_local.shape != (hidden, heads * contract.v_head_dim)
    ):
        raise ValueError("prefill attention weight owner geometry drifted")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    clean_q = jnp.where(live[:, None], prepared.q_residual, 0)
    kv = jnp.where(live[:, None], prepared.current_kv, 0)
    rope = jnp.where(live[:, None], main_rope_table_rows, 0)
    q = fp8_block_matmul(
        clean_q,
        weights.q_b_bits_local,
        weights.q_b_scale_local,
        config=Fp8BlockMatmulConfig(),
        interpret=linear_interpret,
    ).reshape(rows, heads, contract.qk_head_dim)
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
    aligned = gather_stage_local_selected_kv_aligned(
        write.cache,
        jnp.broadcast_to(block_table, (rows, block_table.shape[1])),
        selected,
        write.causal_lengths,
        layout=cache_layout,
        owner_index=owner,
    )
    with jax.named_scope(
        "greenfield_ws32_prefill_attention/selected_cache_expert_exchange"
    ):
        selected_cache = lax.psum(aligned.values, axis_name="expert")
    q_absorbed = fp8_structured_kv_b_q_absorb(
        q[..., : contract.qk_nope_head_dim],
        weights.kv_b_bits_local,
        weights.kv_b_scale_local,
        prefill=True,
        interpret=linear_interpret,
    )
    attended = pregathered_sparse_mla_pallas(
        q_absorbed,
        q_rope,
        selected_cache,
        aligned.valid_counts,
        contract=replace(contract, num_heads=heads),
        config=sparse_attention_config,
        prefill=True,
        interpret=sparse_attention_interpret,
    )
    values = fp8_structured_kv_b_value(
        attended,
        weights.kv_b_bits_local,
        weights.kv_b_scale_local,
        qk_nope_head_dim=contract.qk_nope_head_dim,
        prefill=True,
        interpret=linear_interpret,
    )
    update = ws32_prefill_linear_mapped(
        values.reshape(rows, heads * contract.v_head_dim),
        weights.o_bits_local,
        weights.o_scale_local,
        reduction_axis="expert",
        interpret=linear_interpret,
    )
    output = residual_add(residual_local, update) if add_residual else update
    output = jnp.where(write.row_valid[:, None], output, 0)
    # Sparse MLA's finite-mask/null-sink fallback can turn NaN scores into zero
    # output. Finite final output alone therefore cannot certify live operands.
    finite_operands = (
        jnp.all(jnp.isfinite(clean_q), axis=1)
        & jnp.all(jnp.isfinite(rope), axis=1)
        & jnp.all(jnp.isfinite(q_absorbed), axis=(1, 2))
        & jnp.all(jnp.isfinite(q_rope), axis=(1, 2))
        & jnp.all(jnp.isfinite(selected_cache), axis=(1, 2))
    )
    health = (
        write.valid
        & aligned.contract_valid
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
