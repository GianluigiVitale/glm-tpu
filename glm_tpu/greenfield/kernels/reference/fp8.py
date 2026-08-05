"""Device-side FP8 E4M3FN bit decoding and block-scale application."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax.numpy as jnp


@lru_cache(maxsize=1)
def fp8_e4m3fn_lookup() -> tuple[float, ...]:
    """Return the exact 256-entry E4M3FN decode table as host literals."""

    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    return tuple(values)


def dequantize_fp8_bits_block_weight(
    weight_bits: Any,
    scale: Any,
    *,
    block_shape: tuple[int, int] = (128, 128),
    output_dtype: Any = jnp.bfloat16,
) -> Any:
    """Decode raw final-checkpoint U8 bits and apply colocated FP32 scales.

    Leading dimensions are preserved, so the same operation handles a single
    matrix or an expert stack.  Only the final checkpoint-oriented ``[out,in]``
    dimensions are block-scaled.  Finiteness of raw encodings is a loader gate;
    this arithmetic path never copies or decodes the payload on the host.
    """

    if weight_bits.ndim < 2 or scale.ndim != weight_bits.ndim:
        raise ValueError("FP8 bits and scales must have matching rank >= 2")
    if tuple(weight_bits.shape[:-2]) != tuple(scale.shape[:-2]):
        raise ValueError("FP8 bits/scales leading dimensions disagree")
    if weight_bits.dtype != jnp.uint8:
        raise ValueError(f"FP8 storage must be uint8 bits, got {weight_bits.dtype}")
    if len(block_shape) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in block_shape
    ):
        raise ValueError("block_shape must contain two positive integers")
    expected = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(
            weight_bits.shape[-2:], block_shape, strict=True
        )
    )
    if tuple(scale.shape[-2:]) != expected:
        raise ValueError(
            f"FP8 scale tail must be {expected}, got {scale.shape[-2:]}"
        )
    lookup = jnp.asarray(fp8_e4m3fn_lookup(), dtype=jnp.float32)
    out_blocks = jnp.arange(weight_bits.shape[-2]) // block_shape[0]
    in_blocks = jnp.arange(weight_bits.shape[-1]) // block_shape[1]
    expanded_scale = scale[..., out_blocks[:, None], in_blocks[None, :]]
    decoded = lookup[weight_bits.astype(jnp.int32)]
    return (decoded * expanded_scale.astype(jnp.float32)).astype(output_dtype)
