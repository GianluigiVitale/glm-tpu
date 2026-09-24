"""Normalizations of the reference.

* ``rms_norm``: GLM RMSNorm (``input``/``post_attention``/final norms use the
  fused form below; the q-a and kv-a LoRA norms use this one). FP32 mean of
  squares and ``rsqrt(mean + eps)``, the normalized value rounded to BF16, then
  multiplied by the BF16 weight in BF16.
* ``add_rms_norm``: the fused residual boundary of every layer. The BF16 update
  and residual are added in FP32; the norm consumes the unrounded sum while the
  returned residual is that sum rounded to BF16.
* The DSA indexer key LayerNorm (FP32, affine, ``eps = 1e-6``) is applied inside
  :func:`tests.reference.dsa.index_keys`.

Both functions are the exact oracles (``glm_tpu.layers.norm.rms_norm`` and
``fused_add_rms_norm`` below);
the engine's sharded forms differ only in the association of the
square sum (four feature partials plus one reduction).
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from glm_tpu.layers import norm
from glm_tpu.layers.norm import _accepted_schedule_normalized


def rms_norm(x: jax.Array, weight: jax.Array, *, epsilon: float) -> jax.Array:
    """RMSNorm over the final dimension; returns the activation dtype."""
    return norm.rms_norm(x, weight, epsilon=epsilon)


def add_rms_norm(
    update: jax.Array, residual: jax.Array, weight: jax.Array, *, epsilon: float
) -> tuple[jax.Array, jax.Array]:
    """Return ``(normalized, residual)`` of ``update + residual``."""
    return fused_add_rms_norm(update, residual, weight, epsilon=epsilon)


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
