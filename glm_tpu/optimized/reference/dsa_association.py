"""Bounded layer-0 DSA arithmetic used to diagnose the sealed 8K drift.

This module does not import or execute the legacy engine.  It independently
reconstructs the exact source-level arithmetic and static shapes used by the
sealed oracle so individual associations can be changed one at a time.  The
32-row functions are diagnostic-only; production ``decode_batch1`` remains a
true one-row executable.

Moved verbatim at S2f from ``glm_tpu/greenfield/kernels/reference/dsa_association.py`` (its production definitions; the
research remainder is archived at ``archive/research-20260922``).
"""
from __future__ import annotations

from typing import Any, Literal
from jax import lax
import jax.numpy as jnp


KeyNormMode = Literal["divide_sqrt", "multiply_rsqrt"]


def affine_key_layer_norm(
    value: Any,
    weight: Any,
    bias: Any,
    *,
    epsilon: float,
    mode: KeyNormMode,
) -> Any:
    """Apply one selectable FP32 index-key LayerNorm association."""

    if value.ndim != 2 or weight.shape != (value.shape[1],) or (
        bias.shape != weight.shape
    ):
        raise ValueError("index-key LayerNorm shapes drifted")
    value_f32 = value.astype(jnp.float32)
    mean = jnp.mean(value_f32, axis=-1, keepdims=True)
    centered = value_f32 - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    denominator = variance + jnp.float32(epsilon)
    if mode == "divide_sqrt":
        normalized = centered / jnp.sqrt(denominator)
    elif mode == "multiply_rsqrt":
        normalized = centered * lax.rsqrt(denominator)
    else:
        raise ValueError(f"unsupported index-key LayerNorm mode {mode!r}")
    return (
        normalized * weight.astype(jnp.float32)
        + bias.astype(jnp.float32)
    ).astype(jnp.float32)
