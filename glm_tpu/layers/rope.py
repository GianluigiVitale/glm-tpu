"""GLM rotary-position reference math with explicit pairing semantics."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from glm_tpu.config.cache import CacheConfig
from glm_tpu.exceptions import PlanValidationError


MAIN_ROPE_THETA = 8_000_000.0


def build_rotary_table_host(
    context_capacity: int,
    *,
    rotary_dim: int,
    theta: float,
) -> np.ndarray:
    """Build the accepted BF16 cos/sin table as a host runtime asset.

    The accepted GLM runtime constructs positive FP32 powers, takes their
    reciprocal, evaluates NumPy FP32 cos/sin once, and stores the table in
    BF16.  Keeping this separate from :func:`rotary_cos_sin` is intentional:
    the latter remains the established dynamic DSA path.
    """

    if not isinstance(context_capacity, int) or isinstance(context_capacity, bool) or context_capacity <= 0:
        raise ValueError("rotary table capacity must be a positive integer")
    if not isinstance(rotary_dim, int) or isinstance(rotary_dim, bool) or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary table dimension must be positive and even")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
        raise ValueError("rotary table theta must be positive")

    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    frequencies = np.float32(1.0) / np.power(
        np.float32(theta),
        dimensions / np.float32(rotary_dim),
        dtype=np.float32,
    )
    positions = np.arange(context_capacity, dtype=np.float32)
    angles = np.multiply(positions[:, None], frequencies[None, :], dtype=np.float32)
    table = np.concatenate(
        (
            np.cos(angles, dtype=np.float32),
            np.sin(angles, dtype=np.float32),
        ),
        axis=-1,
    ).astype(ml_dtypes.bfloat16)
    return np.ascontiguousarray(table)


def rotary_table_sha256(table: np.ndarray) -> str:
    """Hash one contiguous BF16 table in storage order."""

    value = np.ascontiguousarray(table)
    if value.ndim != 2 or value.dtype != ml_dtypes.bfloat16:
        raise ValueError("rotary table hash requires a two-dimensional BF16 array")
    return sha256(value.view(np.uint16).tobytes()).hexdigest()


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
        -jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / jnp.float32(rotary_dim),
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
        raise ValueError(f"rotary cos/sin must share a final pair dimension of {half}")
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


def apply_rotary_fp32_final_round(
    value: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    interleaved: bool,
) -> jax.Array:
    """Rotate in FP32 and round only the completed result.

    The accepted main-MLA path consumes a BF16 rotary-table row, evaluates the
    multiply/add expression in FP32, and stores one final BF16 result.  This is
    intentionally a separate, default-off primitive: the established DSA
    rotary path keeps :func:`apply_rotary` and its existing numerical contract.

    An FP32 optimization barrier prevents an early graph rewrite from moving a
    downstream BF16 conversion into the products.  The barrier itself may be
    eliminated after it has served that purpose, so protected TPU callers must
    still lint the FP32 products and final conversion before treating the result
    as exactness evidence.
    """

    if value.ndim < 1 or value.shape[-1] <= 0 or value.shape[-1] % 2:
        raise ValueError("rotary input final dimension must be positive and even")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        raise ValueError("rotary input dtype must be inexact")
    half = value.shape[-1] // 2
    if cos.shape[-1:] != (half,) or sin.shape != cos.shape:
        raise ValueError(f"rotary cos/sin must share a final pair dimension of {half}")
    if not isinstance(interleaved, bool):
        raise ValueError("interleaved must be boolean")

    value_f32 = value.astype(jnp.float32)
    cos_f32 = cos.astype(jnp.float32)
    sin_f32 = sin.astype(jnp.float32)
    if interleaved:
        first = value_f32[..., 0::2]
        second = value_f32[..., 1::2]
        completed = jnp.stack(
            (
                first * cos_f32 - second * sin_f32,
                second * cos_f32 + first * sin_f32,
            ),
            axis=-1,
        ).reshape(value.shape)
    else:
        first = value_f32[..., :half]
        second = value_f32[..., half:]
        completed = jnp.concatenate(
            (
                first * cos_f32 - second * sin_f32,
                second * cos_f32 + first * sin_f32,
            ),
            axis=-1,
        )
    return jax.lax.optimization_barrier(completed).astype(value.dtype)


def rotary_cos_sin_from_rows(
    rope_table_rows: jax.Array, positions: jax.Array, *, rotary_dim: int
) -> tuple[jax.Array, jax.Array]:
    """Split FP32 ``cos|sin`` rows shaped ``positions.shape + (rotary_dim,)``."""

    if not isinstance(rotary_dim, int) or isinstance(rotary_dim, bool):
        raise TypeError("rotary_dim must be an integer")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    if rope_table_rows.shape != (*positions.shape, rotary_dim):
        raise ValueError("DSA rotary rows must match positions and rotary_dim")
    if rope_table_rows.dtype != jnp.float32:
        raise ValueError("DSA rotary rows must be FP32")
    half = rotary_dim // 2
    return rope_table_rows[..., :half], rope_table_rows[..., half:]


def build_main_rope_table(config: CacheConfig) -> Any:
    """Host BF16 ``cos|sin`` table for the main-attention rotary (spec §23.8).

    Built with the accepted GLM runtime's own construction
    (:func:`build_rotary_table_host`: positive FP32 powers, reciprocal, NumPy
    FP32 trigonometry, stored BF16), which DB531 proved reproduces the legacy
    64-wide rotary suffix bitwise when applied with FP32 products and one final
    BF16 round.
    """
    from glm_tpu.layers.rope import build_rotary_table_host

    table = build_rotary_table_host(
        config.context_capacity,
        rotary_dim=config.geometry.qk_rope_head_dim,
        theta=MAIN_ROPE_THETA,
    )
    if table.shape != config.main_rope_table_shape:
        raise PlanValidationError("WS32 main rotary table geometry drifted")
    return table
