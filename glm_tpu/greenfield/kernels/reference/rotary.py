"""GLM rotary-position reference math with explicit pairing semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rotary_cos_sin(
    positions: jax.Array,
    *,
    rotary_dim: int,
    theta: float,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:
    """Build one FP32-derived cosine/sine value per rotary pair.

    Returned arrays have shape ``positions.shape + (rotary_dim // 2,)``.
    GLM-5.2 uses ``rotary_dim=64`` and ``theta=8_000_000`` for both main MLA
    and DSA indexer rotation.
    """

    if not isinstance(rotary_dim, int) or isinstance(rotary_dim, bool):
        raise ValueError("rotary_dim must be an integer")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
        raise ValueError("rotary theta must be positive")
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("rotary positions must have an integer dtype")
    output_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(output_dtype, jnp.inexact):
        raise ValueError("rotary table dtype must be inexact")

    frequencies = jnp.power(
        jnp.float32(theta),
        -jnp.arange(0, rotary_dim, 2, dtype=jnp.float32)
        / jnp.float32(rotary_dim),
    )
    angles = positions.astype(jnp.float32)[..., None] * frequencies
    return jnp.cos(angles).astype(output_dtype), jnp.sin(angles).astype(output_dtype)


def apply_rotary(
    value: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    interleaved: bool,
) -> jax.Array:
    """Rotate the final dimension using explicit GLM pairing.

    ``interleaved=True`` pairs ``(0,1), (2,3), ...`` and re-interleaves the
    result, matching the accepted vLLM layout. ``False`` pairs the first and
    second halves, matching the Hugging Face half-split formula. ``cos`` and
    ``sin`` contain one value per pair and must be broadcastable over all
    leading dimensions.
    """

    if value.ndim < 1 or value.shape[-1] <= 0 or value.shape[-1] % 2:
        raise ValueError("rotary input final dimension must be positive and even")
    half = value.shape[-1] // 2
    if cos.shape[-1:] != (half,) or sin.shape != cos.shape:
        raise ValueError(
            f"rotary cos/sin must share a final pair dimension of {half}"
        )
    if not isinstance(interleaved, bool):
        raise ValueError("interleaved must be boolean")

    cos_value = cos.astype(value.dtype)
    sin_value = sin.astype(value.dtype)
    if interleaved:
        first = value[..., 0::2]
        second = value[..., 1::2]
        out_first = first * cos_value - second * sin_value
        out_second = second * cos_value + first * sin_value
        return jnp.stack((out_first, out_second), axis=-1).reshape(value.shape)
    first = value[..., :half]
    second = value[..., half:]
    return jnp.concatenate(
        (
            first * cos_value - second * sin_value,
            second * cos_value + first * sin_value,
        ),
        axis=-1,
    )
