"""Normalization on the expert-8 x feature-4 mesh: the sharded RMSNorm and fused residual-add
RMSNorm (one feature-4 reduction each), the GLM RMSNorm and final norm, and the FP32 LayerNorms
of the 128-wide DSA indexer key.

Moved verbatim at S2f out of the research package (its production definitions; the research
remainder, and the module these definitions came from, are at ``archive/research-20260922``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal
from jax import lax
import jax
import jax.numpy as jnp

from glm_tpu.layers.contracts import require_shape


KeyNormMode = Literal["divide_sqrt", "multiply_rsqrt"]


ACCEPTED_SCHEDULE_ROWS = 32


# New complete-decoder primitives are intentionally appended below the
# protected one-layer body.  Its TPU HLO records source coordinates in this
# file, so inserting above it would invalidate an already sealed graph.
def sharded_rms_norm(
    hidden_local: Any,
    weight_local: Any,
    *,
    global_hidden_size: int,
    feature_axis: str = "feature",
    epsilon: float = 1e-5,
) -> Any:
    """RMSNorm one persistent hidden shard with one feature-4 reduction."""

    if hidden_local.ndim < 1 or hidden_local.shape[-1] <= 0:
        raise ValueError("RMSNorm requires a nonempty hidden shard")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("RMSNorm activation must be bfloat16")
    if weight_local.shape != (hidden_local.shape[-1],):
        raise ValueError("RMSNorm weight must match the local hidden shard")
    if (
        not isinstance(global_hidden_size, int)
        or isinstance(global_hidden_size, bool)
        or global_hidden_size < hidden_local.shape[-1]
        or global_hidden_size % hidden_local.shape[-1]
    ):
        raise ValueError("RMSNorm global hidden geometry is invalid")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or (epsilon <= 0):
        raise ValueError("RMSNorm epsilon must be positive")

    value = hidden_local.astype(jnp.float32)
    local_square_sum = jnp.sum(lax.square(value), axis=-1, keepdims=True)
    with jax.named_scope("rmsnorm/feature_square_reduce"):
        square_sum = lax.psum(local_square_sum, axis_name=feature_axis)
    inverse = lax.rsqrt(square_sum / jnp.float32(global_hidden_size) + jnp.float32(epsilon))
    normalized = value * inverse
    return (normalized.astype(hidden_local.dtype) * weight_local.astype(hidden_local.dtype)).astype(hidden_local.dtype)


def sharded_fused_add_rms_norm(
    hidden_update_local: Any,
    carried_residual_local: Any,
    weight_local: Any,
    *,
    global_hidden_size: int,
    feature_axis: str = "feature",
    epsilon: float = 1e-5,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, Any]:
    """Apply the accepted split residual/RMSNorm boundary on feature-4.

    The two BF16 inputs are added in FP32.  RMSNorm consumes that unrounded
    sum, while the independently returned residual is the same sum rounded to
    BF16.  Keeping these outputs distinct is part of GLM's numerical contract.
    """

    if hidden_update_local.shape != carried_residual_local.shape:
        raise ValueError("split residual shapes differ")
    if hidden_update_local.dtype != jnp.bfloat16 or (carried_residual_local.dtype != jnp.bfloat16):
        raise ValueError("split residual inputs must be bfloat16")
    if hidden_update_local.ndim < 1 or hidden_update_local.shape[-1] <= 0:
        raise ValueError("split residual requires a nonempty hidden shard")
    if weight_local.shape != (hidden_update_local.shape[-1],) or (weight_local.dtype != jnp.bfloat16):
        raise ValueError("split RMSNorm weight must match the hidden shard")
    if (
        not isinstance(global_hidden_size, int)
        or isinstance(global_hidden_size, bool)
        or global_hidden_size < hidden_update_local.shape[-1]
        or global_hidden_size % hidden_update_local.shape[-1]
    ):
        raise ValueError("split RMSNorm global hidden geometry is invalid")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or (epsilon <= 0):
        raise ValueError("split RMSNorm epsilon must be positive")

    summed = hidden_update_local.astype(jnp.float32) + (carried_residual_local.astype(jnp.float32))
    carried = summed.astype(jnp.bfloat16)
    local_square_sum = jnp.sum(lax.square(summed), axis=-1, keepdims=True)
    with jax.named_scope("fused_rmsnorm/feature_square_reduce"):
        square_sum = lax.psum(local_square_sum, axis_name=feature_axis)
    inverse = lax.rsqrt(square_sum / jnp.float32(global_hidden_size) + jnp.float32(epsilon))
    normalized = summed * inverse
    output = (normalized.astype(jnp.bfloat16) * weight_local).astype(jnp.bfloat16)
    if _observe is not None:
        _observe(
            "post_norm",
            dict(
                update=hidden_update_local,
                residual=carried_residual_local,
                summed=summed,
                local_square_sum=local_square_sum,
                square_sum=square_sum,
                inverse=inverse,
                normalized=output,
                carried=carried,
                weight=weight_local,
            ),
        )
    return output, carried


def affine_key_layer_norm(
    value: Any,
    weight: Any,
    bias: Any,
    *,
    epsilon: float,
    mode: KeyNormMode,
) -> Any:
    """Apply one selectable FP32 index-key LayerNorm association."""

    if value.ndim != 2 or weight.shape != (value.shape[1],) or (bias.shape != weight.shape):
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
    return (normalized * weight.astype(jnp.float32) + bias.astype(jnp.float32)).astype(jnp.float32)


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
    return (normalized.astype(activation_dtype) * weight.astype(activation_dtype)).astype(activation_dtype)


def final_norm(
    hidden_states: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    accepted_schedule: bool = False,
) -> jax.Array:
    """Named final-decoder boundary; arithmetic is the same RMSNorm contract."""

    return rms_norm(hidden_states, weight, epsilon=epsilon, accepted_schedule=accepted_schedule)


def affine_layer_norm(
    value: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    *,
    epsilon: float,
    mode: Literal["divide_sqrt", "multiply_rsqrt"] = "multiply_rsqrt",
) -> jax.Array:
    """FP32 biased LayerNorm used only by the 128-wide indexer key."""

    require_shape("key LayerNorm weight", weight, (value.shape[-1],))
    require_shape("key LayerNorm bias", bias, (value.shape[-1],))
    value_f32 = value.astype(jnp.float32)
    mean = jnp.mean(value_f32, axis=-1, keepdims=True)
    variance = jnp.mean(lax.square(value_f32 - mean), axis=-1, keepdims=True)
    denominator = variance + jnp.float32(epsilon)
    if mode == "divide_sqrt":
        normalized = (value_f32 - mean) / jnp.sqrt(denominator)
    elif mode == "multiply_rsqrt":
        normalized = (value_f32 - mean) * lax.rsqrt(denominator)
    else:
        raise ValueError(f"unknown key LayerNorm association {mode!r}")
    return normalized * weight.astype(jnp.float32) + bias.astype(jnp.float32)
