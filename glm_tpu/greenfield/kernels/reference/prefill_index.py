"""Exact stage-local prompt index-key reconstruction for protected prefill."""

from __future__ import annotations

from typing import Any

from jax import lax
import jax.numpy as jnp

from .dsa import DsaNumericalContract
from .dsa_association import affine_key_layer_norm
from .fp8 import dequantize_fp8_bits_block_weight
from .rotary import apply_rotary, rotary_cos_sin


def materialize_stage_local_prefill_index_wk(
    wk_bits: Any,
    wk_scale: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    fp8_block_shape: tuple[int, int] = (128, 128),
) -> Any:
    """Materialize one final-owner adapted ``wk`` before prefill repair.

    DB518 and DB519 jointly require this to be a completed executable
    boundary: raw FP8 is decoded to BF16, rounded, and only then promoted to
    FP32.  The returned value is passed as an independent repair parameter;
    it must never be fused into the prompt-key projection executable.
    """

    expected_shape = (contract.head_dim, contract.hidden_size)
    if wk_bits.shape != expected_shape or wk_bits.dtype != jnp.uint8:
        raise ValueError("prefill repair wk bits have an invalid shape/dtype")
    expected_scale_shape = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(
            expected_shape, fp8_block_shape, strict=True
        )
    )
    if wk_scale.shape != expected_scale_shape or (
        wk_scale.dtype != jnp.float32
    ):
        raise ValueError("prefill repair wk scales have an invalid shape/dtype")
    return dequantize_fp8_bits_block_weight(
        wk_bits,
        wk_scale,
        block_shape=fp8_block_shape,
        output_dtype=jnp.bfloat16,
    ).astype(jnp.float32)


def physical_m64_prompt_index_key_chunk(
    normalized_chunk: Any,
    positions: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    physical_rows: int = 64,
) -> Any:
    """Project exact normalized inputs with the accepted M64 association.

    The teacher-forced scan records the exact BF16 normalization output already
    consumed by DSA.  The public chunk stays logical and stage-local; only the
    key projection and affine key LayerNorm execute in explicit physical-row
    partitions.  Recurrent ``decode_batch1`` remains one row.
    """

    if (
        not isinstance(physical_rows, int)
        or isinstance(physical_rows, bool)
        or physical_rows <= 0
    ):
        raise ValueError("physical prompt-key rows must be positive")
    if normalized_chunk.ndim != 2 or normalized_chunk.shape[1] != (
        contract.hidden_size
    ):
        raise ValueError("prompt-key normalized chunk has an invalid shape")
    chunk_rows = normalized_chunk.shape[0]
    if chunk_rows <= 0 or chunk_rows % physical_rows:
        raise ValueError("prompt-key chunk must divide into physical rows")
    if positions.shape != (chunk_rows,) or not jnp.issubdtype(
        positions.dtype, jnp.integer
    ):
        raise ValueError("prompt-key positions must be one integer row per token")
    if normalized_chunk.dtype != jnp.bfloat16:
        raise ValueError("prompt-key normalized chunk must remain BF16")
    if wk_weight.shape != (contract.head_dim, contract.hidden_size) or (
        wk_weight.dtype != jnp.float32
    ):
        raise ValueError("prompt-key wk must be the adapted FP32 owner weight")
    if key_norm_weight.shape != (contract.head_dim,) or (
        key_norm_bias.shape != key_norm_weight.shape
    ):
        raise ValueError("prompt-key affine norm shapes are invalid")
    if key_norm_weight.dtype != jnp.bfloat16 or (
        key_norm_bias.dtype != jnp.bfloat16
    ):
        raise ValueError("prompt-key affine norm parameters must remain BF16")

    partitions = normalized_chunk.reshape(
        chunk_rows // physical_rows,
        physical_rows,
        contract.hidden_size,
    )

    def project_and_normalize(partition: Any) -> Any:
        projected = lax.dot_general(
            partition.astype(jnp.float32),
            wk_weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=(lax.Precision.DEFAULT, lax.Precision.HIGHEST),
            preferred_element_type=jnp.float32,
        )
        projected = projected.astype(jnp.float32)
        return affine_key_layer_norm(
            projected,
            key_norm_weight,
            key_norm_bias,
            epsilon=contract.key_layer_norm_epsilon,
            mode="divide_sqrt",
        )

    keys = lax.map(project_and_normalize, partitions).reshape(
        chunk_rows, contract.head_dim
    )
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        keys[:, : contract.rotary_dim],
        cos,
        sin,
        interleaved=contract.interleaved_rotary,
    )
    return jnp.concatenate(
        (rotated, keys[:, contract.rotary_dim :]), axis=-1
    ).astype(jnp.float32)


def repair_stage_local_prompt_index_cache(
    index_cache: Any,
    prompt_normalized_inputs: Any,
    block_tables: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    local_slot: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    logical_page_size: int = 512,
    local_rows_per_page: int = 128,
    prompt_chunk: int = 2048,
    physical_rows: int = 64,
) -> Any:
    """Overwrite one final-owner cache with exact prompt keys.

    Exact normalized inputs never leave their PP8 stage.  Each local lane
    repeats the accepted projection/key-norm association and writes only the
    page rows it owns.
    """

    if prompt_normalized_inputs.ndim != 2 or (
        prompt_normalized_inputs.shape[1] != contract.hidden_size
    ):
        raise ValueError("prompt normalized-input history has an invalid shape")
    if prompt_normalized_inputs.dtype != jnp.bfloat16:
        raise ValueError("prompt normalized-input history must remain BF16")
    if index_cache.ndim != 3 or index_cache.shape[1:] != (
        local_rows_per_page,
        contract.head_dim,
    ):
        raise ValueError("stage-local prompt index cache has an invalid shape")
    if index_cache.dtype != jnp.bfloat16:
        raise ValueError("stage-local prompt index cache must remain BF16")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1 or (
        block_tables.dtype != jnp.int32
    ):
        raise ValueError("prompt repair block table must be one int32 row")
    if logical_page_size != local_rows_per_page * 4:
        raise ValueError("prompt repair is pinned to the PP8 LP4 page layout")
    if prompt_chunk <= 0 or prompt_chunk % physical_rows:
        raise ValueError("prompt repair chunk must divide into physical rows")
    if wk_weight.shape != (contract.head_dim, contract.hidden_size) or (
        wk_weight.dtype != jnp.float32
    ):
        raise ValueError(
            "prompt repair wk must be an externally materialized FP32 owner leaf"
        )
    prompt_tokens = prompt_normalized_inputs.shape[0]
    padded_tokens = (
        (prompt_tokens + prompt_chunk - 1) // prompt_chunk * prompt_chunk
    )
    flat_cache = index_cache.reshape(-1, contract.head_dim)
    for chunk_start in range(0, padded_tokens, prompt_chunk):
        positions = jnp.arange(
            chunk_start,
            chunk_start + prompt_chunk,
            dtype=jnp.int32,
        )
        safe_positions = jnp.minimum(
            positions, jnp.int32(prompt_tokens - 1)
        )
        normalized_chunk = jnp.take(
            prompt_normalized_inputs, safe_positions, axis=0
        )
        keys = physical_m64_prompt_index_key_chunk(
            normalized_chunk,
            positions,
            wk_weight,
            key_norm_weight,
            key_norm_bias,
            contract=contract,
            physical_rows=physical_rows,
        )

        logical_pages = positions // jnp.int32(logical_page_size)
        page_rows = positions % jnp.int32(logical_page_size)
        target_owner = page_rows // jnp.int32(local_rows_per_page)
        local_rows = page_rows % jnp.int32(local_rows_per_page)
        table_valid = logical_pages < jnp.int32(block_tables.shape[1])
        safe_logical_pages = jnp.clip(
            logical_pages,
            jnp.int32(0),
            jnp.int32(block_tables.shape[1] - 1),
        )
        physical_pages = jnp.take(
            block_tables[0], safe_logical_pages, axis=0
        )
        page_valid = (
            (physical_pages >= jnp.int32(0))
            & (physical_pages < jnp.int32(index_cache.shape[0]))
        )
        valid = (
            (positions < jnp.int32(prompt_tokens))
            & table_valid
            & page_valid
            & (target_owner == local_slot)
        )
        slots = (
            physical_pages * jnp.int32(local_rows_per_page) + local_rows
        )
        slots = jnp.where(valid, slots, jnp.int32(flat_cache.shape[0]))
        flat_cache = flat_cache.at[slots].set(
            keys.astype(index_cache.dtype), mode="drop"
        )
    return flat_cache.reshape(index_cache.shape)
