"""P1 prefill attention: frozen FP8/cache composition with local LSE exchange.

Mirrors ws32_prefill_index_share_attention_mapped outside MODEL_SOURCE. The
only arithmetic change is the documented owner-local softmax boundary. Cache
writes, causal bounds, projection kernels, padded-row handling and finite
operand admission remain explicit. No checkpoint format is changed.
"""
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas.fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
)
from ..greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from ..greenfield.kernels.prefill_cache import write_prefill_cache_block
from ..greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..greenfield.kernels.reference.dsa import SelectedPositions
from ..greenfield.kernels.reference.linear import residual_add
from ..greenfield.kernels.reference.rotary import apply_rotary_fp32_final_round
from ..greenfield.kernels.ws32_layer import Ws32AttentionResult, Ws32AttentionWeights, Ws32PreparedAttention
from ..greenfield.kernels.ws32_prefill_attention import _require_block
from ..greenfield.kernels.ws32_prefill_linear import ws32_prefill_linear_mapped
from .lse_attention import lse_attention_mapped


def prefill_index_share_lse_mapped(
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
    q_absorbed = fp8_structured_kv_b_q_absorb(
        q[..., : contract.qk_nope_head_dim],
        weights.kv_b_bits_local,
        weights.kv_b_scale_local,
        prefill=True,
        interpret=linear_interpret,
    )
    partial = lse_attention_mapped(
        q_absorbed, q_rope, write.cache,
        jnp.broadcast_to(block_table, (rows, block_table.shape[1])),
        selected, write.causal_lengths, contract=contract, layout=cache_layout,
        config=sparse_attention_config, interpret=sparse_attention_interpret,
        validate_finite=True,
    )
    attended = partial.output
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
