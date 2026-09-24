"""The MoE router: exact GLM ``noaux_tc`` expert selection with unbiased normalized sigmoid weights
(``route_glm_noaux_tc_logits``), for the decode row from its sharded router projection
(``router_from_shards``) and for a block of prompt rows (``prefill_router``).

This module restates the model contract directly in JAX and imports no other model or
inference engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from glm_tpu.layers.contracts import _require_shape


def route_glm_noaux_tc_logits(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    top_k: int = 8,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return selected expert ids and unbiased normalized sigmoid weights.

    The correction bias participates only in selection.  The selected weights
    are gathered from the unbiased sigmoid scores and normalized in FP32.
    Routed scaling is intentionally absent; the accepted v4 path applies it to
    the reduced routed output.
    """

    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    _require_shape("correction_bias", correction_bias, (router_logits.shape[1],))
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not (0 < top_k <= router_logits.shape[1]):
        raise ValueError("top_k must be in [1, num_experts]")
    scores = jax.nn.sigmoid(router_logits.astype(jnp.float32))
    biased_scores = scores + correction_bias.astype(jnp.float32)[None, :]
    _, indices = lax.top_k(biased_scores, top_k)
    weights = jnp.take_along_axis(scores, indices, axis=-1)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True, dtype=jnp.float32)
    if _observe is not None:
        _observe(
            "router_selection",
            dict(
                scores=scores,
                biased_scores=biased_scores,
                indices=indices,
                weights=weights,
            ),
        )
    return indices.astype(jnp.int32), weights.astype(jnp.float32)


def router_from_shards(
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
    if router_weight_local.ndim != 2 or (router_weight_local.shape[1] != hidden_local.shape[1]):
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
    return route_glm_noaux_tc_logits(logits, correction_bias, top_k=top_k)


def prefill_router(
    hidden_local: Any,
    router_weight_local: Any,
    correction_bias_local: Any,
    live: Any,
    *,
    top_k: int = 8,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, Any, Any]:
    """Exact noaux_tc selection of this multirow FP32 router's own logits.

    Correction bias affects IDs only, never mixture weights. Padded rows return
    valid placeholder IDs/zero weights for the grouped kernel; they do not claim
    zero execution cost. Final-block executables should use narrow static rows.
    """
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill router requires WS32 expert8/feature4 mesh")
    if hidden_local.ndim != 2 or not 1 <= hidden_local.shape[0] <= 128 or hidden_local.dtype != jnp.bfloat16:
        raise ValueError("prefill router requires1..128 BF16 feature rows")
    rows = hidden_local.shape[0]
    if live.shape != (rows,) or live.dtype != jnp.bool_:
        raise ValueError("prefill router requires boolean live rows")
    if (
        router_weight_local.ndim != 2
        or router_weight_local.shape[1] != hidden_local.shape[1]
        or (
            correction_bias_local.shape != (router_weight_local.shape[0],)
            or router_weight_local.dtype != jnp.bfloat16
            or correction_bias_local.dtype != jnp.float32
        )
    ):
        raise ValueError("prefill router owner geometry/dtype drifted")
    clean = jnp.where(live[:, None], hidden_local, 0)
    partial = lax.dot_general(
        clean.astype(jnp.float32),
        router_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("greenfield_ws32_prefill_router/feature_reduce"):
        local_logits = lax.psum(partial, "feature")
    with jax.named_scope("greenfield_ws32_prefill_router/expert_gather"):
        logits = lax.all_gather(local_logits, "expert", axis=1, tiled=True)
        bias = lax.all_gather(correction_bias_local, "expert", axis=0, tiled=True)
    indices, weights = route_glm_noaux_tc_logits(logits, bias, top_k=top_k, _observe=_observe)
    if _observe is not None:
        _observe(
            "router",
            dict(
                input=hidden_local,
                clean=clean,
                live=live,
                weight=router_weight_local,
                partial=partial,
                local_logits=local_logits,
                logits=logits,
                bias=bias,
            ),
        )
    valid = ~live | (
        jnp.all(jnp.isfinite(clean), axis=1)
        & jnp.all(jnp.isfinite(logits), axis=1)
        & jnp.all(jnp.isfinite(bias))
        & jnp.all(jnp.isfinite(weights), axis=1)
    )
    return indices, jnp.where(live[:, None], weights, 0), valid
