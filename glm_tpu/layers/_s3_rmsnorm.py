"""Exact readable normalization primitives for GLM-5.2.

The transformer RMSNorm contract follows the Hugging Face reference exactly:
square/mean/rsqrt are evaluated in FP32, the normalized value is rounded back
to the activation dtype, and only then is the checkpoint weight applied.  The
functions in this module are native JAX and have no legacy-engine dependency.
"""

from __future__ import annotations


import jax
from jax import lax
import jax.numpy as jnp


ACCEPTED_SCHEDULE_ROWS = 32


def _accepted_schedule_normalized(value: jax.Array, epsilon: float) -> jax.Array:
    """Normalize ``value`` (FP32) with the accepted decode-step variance schedule.

    The accepted program reduces every RMS variance over an M32 operand,
    ``f32[32, W] -> f32[32]`` along the row's final dimension.  A single live
    row reduced as ``f32[1, W] -> f32[]`` lands its FP32 scale 1-4 ulps low
    (layer-1 scale-frontier certificate; bounded TPU replay 2026-09-02).  Rows
    are padded to 32 and kept alive by one FP32 ``optimization_barrier`` so the
    reduce keeps the accepted shape; no rounding is introduced.
    """

    width = value.shape[-1]
    rows_2d = value.reshape((-1, width))
    rows = rows_2d.shape[0]
    padded_rows = max(ACCEPTED_SCHEDULE_ROWS, rows)
    if padded_rows > rows:
        rows_2d = jnp.pad(rows_2d, ((0, padded_rows - rows), (0, 0)))
    carried = lax.optimization_barrier(rows_2d)
    variance = jnp.mean(lax.square(carried), axis=-1, keepdims=True)
    inverse = lax.rsqrt(variance + jnp.float32(epsilon))
    return (carried * inverse)[:rows].reshape(value.shape)


def rms_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    accepted_schedule: bool = False,
) -> jax.Array:
    """Apply GLM RMSNorm over the final dimension.

    ``hidden_states`` may have any non-empty leading shape. ``weight`` is the
    unsharded logical checkpoint vector for the final dimension; sharded
    callers must pass the corresponding local final-dimension shard.
    """

    if hidden_states.ndim < 1:
        raise ValueError("hidden_states must have at least one dimension")
    if weight.shape != (hidden_states.shape[-1],):
        raise ValueError(
            "RMSNorm weight must match the final hidden dimension: "
            f"expected={(hidden_states.shape[-1],)} got={weight.shape}"
        )
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    if not jnp.issubdtype(hidden_states.dtype, jnp.inexact):
        raise ValueError("RMSNorm activations must have an inexact dtype")
    if not jnp.issubdtype(weight.dtype, jnp.inexact):
        raise ValueError("RMSNorm weight must have an inexact dtype")

    if not isinstance(accepted_schedule, bool):
        raise ValueError("RMSNorm accepted-schedule flag must be boolean")
    activation_dtype = hidden_states.dtype
    value = hidden_states.astype(jnp.float32)
    if accepted_schedule:
        normalized = _accepted_schedule_normalized(value, epsilon)
    else:
        variance = jnp.mean(lax.square(value), axis=-1, keepdims=True)
        normalized = value * lax.rsqrt(variance + jnp.float32(epsilon))
    return (
        normalized.astype(activation_dtype)
        * weight.astype(activation_dtype)
    ).astype(activation_dtype)


def fused_add_rms_norm(
    hidden_states: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    accepted_schedule: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Match the accepted decoder's fused residual-add/RMSNorm boundary.

    The normalization consumes the unrounded FP32 sum.  The independently
    returned residual is the same sum rounded to the activation dtype.  These
    two values are intentionally not interchangeable: rounding the sum before
    the variance reduction changes BF16 model arithmetic.
    """

    if hidden_states.shape != residual.shape:
        raise ValueError("fused RMSNorm hidden and residual shapes must match")
    if hidden_states.dtype != residual.dtype:
        raise ValueError("fused RMSNorm hidden and residual dtypes must match")
    if hidden_states.ndim < 1:
        raise ValueError("fused RMSNorm inputs must have at least one dimension")
    if weight.shape != (hidden_states.shape[-1],):
        raise ValueError(
            "fused RMSNorm weight must match the final hidden dimension"
        )
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("fused RMSNorm epsilon must be positive")
    if not jnp.issubdtype(hidden_states.dtype, jnp.inexact):
        raise ValueError("fused RMSNorm activations must have an inexact dtype")
    if not jnp.issubdtype(weight.dtype, jnp.inexact):
        raise ValueError("fused RMSNorm weight must have an inexact dtype")

    if not isinstance(accepted_schedule, bool):
        raise ValueError("fused RMSNorm accepted-schedule flag must be boolean")
    activation_dtype = hidden_states.dtype
    summed = hidden_states.astype(jnp.float32) + residual.astype(jnp.float32)
    carried_residual = summed.astype(activation_dtype)
    if accepted_schedule:
        normalized = _accepted_schedule_normalized(summed, epsilon)
    else:
        variance = jnp.mean(lax.square(summed), axis=-1, keepdims=True)
        normalized = summed * lax.rsqrt(variance + jnp.float32(epsilon))
    output = (
        normalized.astype(weight.dtype) * weight
    ).astype(activation_dtype)
    return output, carried_residual


def final_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    accepted_schedule: bool = False,
) -> jax.Array:
    """Named final-decoder boundary; arithmetic is the same RMSNorm contract."""

    return rms_norm(
        hidden_states, weight, epsilon=epsilon, accepted_schedule=accepted_schedule
    )
