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

Both functions are the exact oracles of ``glm_tpu/optimized/reference/
rmsnorm.py``; the engine's sharded forms differ only in the association of the
square sum (four feature partials plus one reduction).
"""

from __future__ import annotations

import jax

from glm_tpu.optimized.reference import rmsnorm


def rms_norm(x: jax.Array, weight: jax.Array, *, epsilon: float) -> jax.Array:
    """RMSNorm over the final dimension; returns the activation dtype."""
    return rmsnorm.rms_norm(x, weight, epsilon=epsilon)


def add_rms_norm(
    update: jax.Array, residual: jax.Array, weight: jax.Array, *, epsilon: float
) -> tuple[jax.Array, jax.Array]:
    """Return ``(normalized, residual)`` of ``update + residual``."""
    return rmsnorm.fused_add_rms_norm(update, residual, weight, epsilon=epsilon)
