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


def final_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
) -> jax.Array:
    """Named final-decoder boundary; arithmetic is the same RMSNorm contract."""

    return rms_norm(hidden_states, weight, epsilon=epsilon)
