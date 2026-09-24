"""Exactness-first WS32_2D one-row dense and MoE reference bodies.

Moved verbatim at S2f out of the research package (its production definitions; the research
remainder, and the module these definitions came from, are at ``archive/research-20260922``).
"""
from __future__ import annotations

from typing import Any, Callable
from jax import lax
import jax
import jax.numpy as jnp


# New complete-decoder primitives are intentionally appended below the
# protected one-layer body.  Its TPU HLO records source coordinates in this
# file, so inserting above it would invalidate an already sealed graph.
def ws32_rms_norm_mapped(
    hidden_local: Any,
    weight_local: Any,
    *,
    global_hidden_size: int,
    feature_axis: str = "feature",
    epsilon: float = 1e-5,
) -> Any:
    """RMSNorm one persistent hidden shard with one feature-4 reduction."""

    if hidden_local.ndim < 1 or hidden_local.shape[-1] <= 0:
        raise ValueError("WS32 RMSNorm requires a nonempty hidden shard")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 RMSNorm activation must be bfloat16")
    if weight_local.shape != (hidden_local.shape[-1],):
        raise ValueError("WS32 RMSNorm weight must match the local hidden shard")
    if (
        not isinstance(global_hidden_size, int)
        or isinstance(global_hidden_size, bool)
        or global_hidden_size < hidden_local.shape[-1]
        or global_hidden_size % hidden_local.shape[-1]
    ):
        raise ValueError("WS32 RMSNorm global hidden geometry is invalid")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or (
        epsilon <= 0
    ):
        raise ValueError("WS32 RMSNorm epsilon must be positive")

    value = hidden_local.astype(jnp.float32)
    local_square_sum = jnp.sum(lax.square(value), axis=-1, keepdims=True)
    with jax.named_scope("greenfield_ws32_rmsnorm/feature_square_reduce"):
        square_sum = lax.psum(local_square_sum, axis_name=feature_axis)
    inverse = lax.rsqrt(
        square_sum / jnp.float32(global_hidden_size) + jnp.float32(epsilon)
    )
    normalized = value * inverse
    return (
        normalized.astype(hidden_local.dtype)
        * weight_local.astype(hidden_local.dtype)
    ).astype(hidden_local.dtype)


def ws32_fused_add_rms_norm_mapped(
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
        raise ValueError("WS32 split residual shapes differ")
    if hidden_update_local.dtype != jnp.bfloat16 or (
        carried_residual_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 split residual inputs must be bfloat16")
    if hidden_update_local.ndim < 1 or hidden_update_local.shape[-1] <= 0:
        raise ValueError("WS32 split residual requires a nonempty hidden shard")
    if weight_local.shape != (hidden_update_local.shape[-1],) or (
        weight_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 split RMSNorm weight must match the hidden shard")
    if (
        not isinstance(global_hidden_size, int)
        or isinstance(global_hidden_size, bool)
        or global_hidden_size < hidden_update_local.shape[-1]
        or global_hidden_size % hidden_update_local.shape[-1]
    ):
        raise ValueError("WS32 split RMSNorm global hidden geometry is invalid")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or (
        epsilon <= 0
    ):
        raise ValueError("WS32 split RMSNorm epsilon must be positive")

    summed = hidden_update_local.astype(jnp.float32) + (
        carried_residual_local.astype(jnp.float32)
    )
    carried = summed.astype(jnp.bfloat16)
    local_square_sum = jnp.sum(lax.square(summed), axis=-1, keepdims=True)
    with jax.named_scope("greenfield_ws32_fused_rmsnorm/feature_square_reduce"):
        square_sum = lax.psum(local_square_sum, axis_name=feature_axis)
    inverse = lax.rsqrt(
        square_sum / jnp.float32(global_hidden_size) + jnp.float32(epsilon)
    )
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


def ws32_router_from_shards_mapped(
    hidden_local: Any,
    router_weight_local: Any,
    correction_bias_local: Any,
    *,
    top_k: int,
    expert_axis: str = "expert",
    feature_axis: str = "feature",
) -> tuple[Any, Any]:
    """Compute exact GLM routing from a 2D router shard and compact gathers."""

    from glm_tpu.layers.moe.router import route_glm_noaux_tc_logits

    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 router requires one live hidden row")
    if router_weight_local.ndim != 2 or (
        router_weight_local.shape[1] != hidden_local.shape[1]
    ):
        raise ValueError("WS32 router weight geometry drifted")
    local_experts = router_weight_local.shape[0]
    if correction_bias_local.shape != (local_experts,):
        raise ValueError("WS32 router bias geometry drifted")
    local_logits = lax.dot_general(
        hidden_local.astype(jnp.float32),
        router_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("greenfield_ws32_router/feature_reduce"):
        local_logits = lax.psum(local_logits, axis_name=feature_axis)
    with jax.named_scope("greenfield_ws32_router/expert_gather"):
        logits = lax.all_gather(
            local_logits,
            axis_name=expert_axis,
            axis=1,
            tiled=True,
        )
        correction_bias = lax.all_gather(
            correction_bias_local,
            axis_name=expert_axis,
            axis=0,
            tiled=True,
        )
    return route_glm_noaux_tc_logits(
        logits, correction_bias, top_k=top_k
    )
