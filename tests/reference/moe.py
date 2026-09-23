"""Sparse MoE of the reference: noaux_tc router, routed SwiGLU experts, shared expert.

* Router logits ``normalized @ router.T`` in FP32 (BF16 weight widened).
* noaux_tc (``n_group = topk_group = 1``, sigmoid scoring): ``s = sigmoid(logits)``;
  the ``top_k`` experts maximize ``s + e_score_correction_bias`` (ties to the
  lowest expert id); the route weights are the *unbiased* ``s`` of those experts
  normalized in FP32 (``norm_topk_prob``). The correction bias never enters the
  weights. This is ``route_glm_noaux_tc_logits`` of
  ``glm_tpu/greenfield/kernels/reference/moe.py``.
* Each routed expert is a BF16 SwiGLU (:func:`tests.reference.linear.swiglu`);
  its output is weighted in BF16 by the BF16-rounded route weight; the ``top_k``
  weighted outputs are summed in FP32 and rounded once to BF16.
* ``output = bf16(bf16(routed * 2.5) + shared)`` (``routed_scaling_factor``
  applied to the routed sum, not to the weights; the same value in exact
  arithmetic as Hugging Face's scaled weights).

The engine's decode sums the weighted routes of one expert-owner chip in BF16
before its FP32 cross-chip reduction; its prefill sums them in FP32 like this
reference (``fp32_route_sum``). The two differ only when two routes of a row
land on the same owner, by BF16 rounding of that partial sum.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.greenfield.kernels.reference.moe import route_glm_noaux_tc_logits

from .linear import einsum, project, swiglu, swiglu_activation


class Routes(NamedTuple):
    """One sparse layer's routing decision per row."""

    expert_ids: jax.Array  # [rows, top_k] int32, descending biased score
    weights: jax.Array  # [rows, top_k] FP32 (normalized, unbiased)
    # [rows] FP32: the k-th minus the (k+1)-th biased score (the decision's slack);
    # +inf when top_k equals the expert count
    margin: jax.Array


class DenseWeights(NamedTuple):
    gate: jax.Array  # [intermediate, hidden]
    up: jax.Array  # [intermediate, hidden]
    down: jax.Array  # [hidden, intermediate]


class MoeWeights(NamedTuple):
    router: jax.Array  # [experts, hidden] BF16
    correction_bias: jax.Array  # [experts] FP32
    expert_gate: jax.Array  # [experts, intermediate, hidden]
    expert_up: jax.Array  # [experts, intermediate, hidden]
    expert_down: jax.Array  # [experts, hidden, intermediate]
    shared: DenseWeights


def route(normalized: jax.Array, weights: MoeWeights, *, top_k: int) -> Routes:
    """noaux_tc routing of every row, with the margin of its top-k decision."""
    logits = project(normalized, weights.router, dtype=jnp.float32)
    expert_ids, route_weights = route_glm_noaux_tc_logits(
        logits, weights.correction_bias, top_k=top_k
    )
    if top_k < logits.shape[1]:
        biased = jax.nn.sigmoid(logits) + weights.correction_bias[None, :]
        ranked = lax.top_k(biased, top_k + 1)[0]
        margin = ranked[:, top_k - 1] - ranked[:, top_k]
    else:  # every expert is routed: the decision has no (k+1)-th candidate
        margin = jnp.full(logits.shape[:1], jnp.inf)
    return Routes(expert_ids, route_weights, margin.astype(jnp.float32))


def dense_mlp(normalized: jax.Array, weights: DenseWeights) -> jax.Array:
    return swiglu(normalized, weights.gate, weights.up, weights.down)


def moe(
    normalized: jax.Array,
    weights: MoeWeights,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[jax.Array, Routes]:
    """One sparse MLP update ``[rows, hidden]`` in BF16 and its routing decision."""
    routes = route(normalized, weights, top_k=top_k)
    expert_ids, route_weights = routes.expert_ids, routes.weights
    routed = jnp.zeros(normalized.shape, jnp.float32)
    for slot in range(top_k):
        ids = expert_ids[:, slot]
        gate = einsum(
            "rh,rih->ri", normalized, jnp.take(weights.expert_gate, ids, axis=0)
        )
        up = einsum("rh,rih->ri", normalized, jnp.take(weights.expert_up, ids, axis=0))
        activated = swiglu_activation(gate, up)
        output = einsum(
            "ri,rhi->rh", activated, jnp.take(weights.expert_down, ids, axis=0)
        )
        weighted = (output * route_weights[:, slot, None].astype(jnp.bfloat16)).astype(
            jnp.bfloat16
        )
        routed = routed + weighted.astype(jnp.float32)
    shared = dense_mlp(normalized, weights.shared)
    scale = jnp.asarray(routed_scaling_factor, jnp.bfloat16)
    return (routed.astype(jnp.bfloat16) * scale + shared).astype(jnp.bfloat16), routes
