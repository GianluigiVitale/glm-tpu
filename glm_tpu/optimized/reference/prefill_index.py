"""Exact stage-local prompt index-key reconstruction for protected prefill."""

from __future__ import annotations

from typing import Any

from jax import lax
import jax.numpy as jnp

from .dsa import DsaNumericalContract
from .dsa_association import affine_key_layer_norm
from .fp8 import dequantize_fp8_bits_block_weight
from .dsa_host_rope import rotary_cos_sin_from_rows
from .rotary import apply_rotary, rotary_cos_sin


def decode_stage_local_prefill_index_wk_bf16(
    wk_bits: Any,
    wk_scale: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    fp8_block_shape: tuple[int, int] = (128, 128),
) -> Any:
    """Decode and finish the BF16 half of one final-owner ``wk`` adapter."""

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
    )


def promote_stage_local_prefill_index_wk(
    wk_bf16: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> Any:
    """Promote an already-completed stage-local BF16 ``wk`` to FP32."""

    expected_shape = (contract.head_dim, contract.hidden_size)
    if wk_bf16.shape != expected_shape or wk_bf16.dtype != jnp.bfloat16:
        raise ValueError("prefill repair BF16 wk has an invalid shape/dtype")
    return wk_bf16.astype(jnp.float32)


def physical_m64_prompt_index_key_chunk(
    normalized_chunk: Any,
    positions: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    physical_rows: int = 64,
    rope_table_rows: Any | None = None,
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
    if rope_table_rows is None:
        cos, sin = rotary_cos_sin(
            positions,
            rotary_dim=contract.rotary_dim,
            theta=contract.theta,
            dtype=jnp.float32,
        )
    else:
        # Host FP32 cos|sin rows gathered by position: on-device cos/sin at
        # large rotary angles are inaccurate on TPU (protected V2/V3 replays).
        cos, sin = rotary_cos_sin_from_rows(
            rope_table_rows, positions, rotary_dim=contract.rotary_dim
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
