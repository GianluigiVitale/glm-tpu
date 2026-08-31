"""Exact readable normalization primitives for GLM-5.2.

The transformer RMSNorm contract follows the Hugging Face reference exactly:
square/mean/rsqrt are evaluated in FP32, the normalized value is rounded back
to the activation dtype, and only then is the checkpoint weight applied.  The
functions in this module are native JAX and have no legacy-engine dependency.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp


class FusedAddRmsNormAuxiliaryResult(NamedTuple):
    """Device-resident candidate tuple at the fused layer RMS boundary."""

    output: jax.Array
    carried_residual: jax.Array
    rms_input_fp32: jax.Array


class FusedAddRmsNormCompensatedAuxiliaryResult(NamedTuple):
    """Device-resident compensated value pending frontier cross-binding."""

    output: jax.Array
    carried_residual: jax.Array
    restored_rms_input_fp32: jax.Array


def rms_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
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

    activation_dtype = hidden_states.dtype
    value = hidden_states.astype(jnp.float32)
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

    activation_dtype = hidden_states.dtype
    summed = hidden_states.astype(jnp.float32) + residual.astype(jnp.float32)
    carried_residual = summed.astype(activation_dtype)
    variance = jnp.mean(lax.square(summed), axis=-1, keepdims=True)
    normalized = summed * lax.rsqrt(variance + jnp.float32(epsilon))
    output = (
        normalized.astype(weight.dtype) * weight
    ).astype(activation_dtype)
    return output, carried_residual


def fused_add_rms_norm_with_auxiliary(
    hidden_states: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
) -> FusedAddRmsNormAuxiliaryResult:
    """Default-off tuple candidate retaining the exact FP32 RMS input.

    Primary arithmetic is written independently and identically to
    :func:`fused_add_rms_norm`.  The third tuple member is the same FP32 sum
    consumed by the variance reduction; it is not substituted into the BF16
    recurrent residual and has no host consumer here.
    """

    if hidden_states.shape != residual.shape:
        raise ValueError(
            "fused auxiliary RMSNorm hidden and residual shapes must match"
        )
    if hidden_states.dtype != residual.dtype:
        raise ValueError(
            "fused auxiliary RMSNorm hidden and residual dtypes must match"
        )
    if hidden_states.ndim < 1:
        raise ValueError(
            "fused auxiliary RMSNorm inputs must have at least one dimension"
        )
    if weight.shape != (hidden_states.shape[-1],):
        raise ValueError(
            "fused auxiliary RMSNorm weight must match the final hidden dimension"
        )
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or epsilon <= 0
    ):
        raise ValueError("fused auxiliary RMSNorm epsilon must be positive")
    if not jnp.issubdtype(hidden_states.dtype, jnp.inexact):
        raise ValueError(
            "fused auxiliary RMSNorm activations must have an inexact dtype"
        )
    if not jnp.issubdtype(weight.dtype, jnp.inexact):
        raise ValueError("fused auxiliary RMSNorm weight must have an inexact dtype")

    activation_dtype = hidden_states.dtype
    rms_input_fp32 = hidden_states.astype(jnp.float32) + residual.astype(
        jnp.float32
    )
    carried_residual = rms_input_fp32.astype(activation_dtype)
    variance = jnp.mean(lax.square(rms_input_fp32), axis=-1, keepdims=True)
    normalized = rms_input_fp32 * lax.rsqrt(variance + jnp.float32(epsilon))
    output = (
        normalized.astype(weight.dtype) * weight
    ).astype(activation_dtype)
    return FusedAddRmsNormAuxiliaryResult(
        output,
        carried_residual,
        rms_input_fp32,
    )


def fused_add_rms_norm_with_compensated_auxiliary(
    hidden_states: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
) -> FusedAddRmsNormCompensatedAuxiliaryResult:
    """Default-off compensated dependency at the fused RMS boundary.

    The accepted primary arithmetic continues to consume ``rms_input_fp32``.
    Independently, the BF16 recurrent value is widened, its discarded FP32
    correction is recovered, and the two are recombined behind explicit
    optimization barriers.  The restored value is returned only as a rooted
    device auxiliary; it is never substituted into either primary output.

    This source spelling declares a graph identity, not a numerical proof.
    Candidate-coherent replay must independently prove that the restored value
    equals the exact FP32 RMS input before any compile or TPU successor exists.
    """

    if hidden_states.shape != residual.shape:
        raise ValueError(
            "fused compensated RMSNorm hidden and residual shapes must match"
        )
    if hidden_states.dtype != residual.dtype:
        raise ValueError(
            "fused compensated RMSNorm hidden and residual dtypes must match"
        )
    if hidden_states.ndim < 1:
        raise ValueError(
            "fused compensated RMSNorm inputs must have at least one dimension"
        )
    if weight.shape != (hidden_states.shape[-1],):
        raise ValueError(
            "fused compensated RMSNorm weight must match the final hidden dimension"
        )
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or epsilon <= 0
    ):
        raise ValueError("fused compensated RMSNorm epsilon must be positive")
    if not jnp.issubdtype(hidden_states.dtype, jnp.inexact):
        raise ValueError(
            "fused compensated RMSNorm activations must have an inexact dtype"
        )
    if not jnp.issubdtype(weight.dtype, jnp.inexact):
        raise ValueError(
            "fused compensated RMSNorm weight must have an inexact dtype"
        )

    activation_dtype = hidden_states.dtype
    rms_input_fp32 = hidden_states.astype(jnp.float32) + residual.astype(
        jnp.float32
    )
    carried_residual = rms_input_fp32.astype(activation_dtype)
    rounded_fp32 = carried_residual.astype(jnp.float32)
    correction_fp32 = lax.optimization_barrier(rms_input_fp32 - rounded_fp32)
    restored_rms_input_fp32 = lax.optimization_barrier(
        rounded_fp32 + correction_fp32
    )
    variance = jnp.mean(lax.square(rms_input_fp32), axis=-1, keepdims=True)
    normalized = rms_input_fp32 * lax.rsqrt(variance + jnp.float32(epsilon))
    output = (
        normalized.astype(weight.dtype) * weight
    ).astype(activation_dtype)
    return FusedAddRmsNormCompensatedAuxiliaryResult(
        output,
        carried_residual,
        restored_rms_input_fp32,
    )


def final_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
) -> jax.Array:
    """Named final-decoder boundary; arithmetic is the same RMSNorm contract."""

    return rms_norm(hidden_states, weight, epsilon=epsilon)
