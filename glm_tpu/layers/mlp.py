"""Canonical dense MLP placement of the production prefill (four 32-row placements in B128).

TPU results of the dense MLP depend on the physical row placement; this runs the dense suffix as
four 32-live-row placements inside B128 (B114 padded and cropped) so production reproduces the
retained narrow-suffix bytes. The body of ``greenfield/kernels/ws32_prefill_dense_canonical.py``
calling the production MLP suffix explicitly (S2d fold of the former function rebinding); the
frozen module stays untouched as the numerical oracle of the tests.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.models.glm_moe_dsa.weights import Bf16DenseWeights
from glm_tpu.layers.linear import _dot_f32, _require_rows, resident_matmul_f32


def prefill_dense(
    hidden_local: Any,
    gate_local: Any,
    up_local: Any,
    down_local: Any,
    *,
    interpret: bool = False,
) -> Any:
    """Batched reciprocal-sharded dense MLP, not the StrategyND overlay.

    This is a candidate building block with the same numerical boundaries as
    ``ws32_dense_pallas_mapped``. It is not automatically interchangeable with
    the promoted decoder's StrategyND dense association. A future layer path
    must earn its own bounded comparisons and §21 decoder evidence.
    """

    _require_rows(hidden_local)
    if gate_local.ndim != 2 or (gate_local.shape != up_local.shape or gate_local.shape[1] != hidden_local.shape[1]):
        raise ValueError("WS32 prefill gate/up geometry drifted")
    local_intermediate, local_hidden = gate_local.shape
    if down_local.shape != (local_hidden, local_intermediate):
        raise ValueError("WS32 prefill down geometry drifted")
    gate_partial = resident_matmul_f32(hidden_local, gate_local, interpret=interpret)
    up_partial = resident_matmul_f32(hidden_local, up_local, interpret=interpret)
    with jax.named_scope("greenfield_ws32_prefill_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(jnp.stack((gate_partial, up_partial)), axis_name="feature").astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down_partial = resident_matmul_f32(activated, down_local, interpret=interpret)
    with jax.named_scope("greenfield_ws32_prefill_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name="expert").astype(jnp.bfloat16)


# ----------------------------------------------------------------------------- MLP bodies
def dense_bf16(normalized: Any, weights: Bf16DenseWeights, *, expert_axis: str, feature_axis: str) -> Any:
    """Mirror of ``ws32_dense_pallas_mapped`` on BF16 tables."""

    gate_partial = _dot_f32(normalized, weights.gate_local)
    up_partial = _dot_f32(normalized, weights.up_local)
    with jax.named_scope("glm_perf_bf16_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(jnp.stack((gate_partial, up_partial), axis=0), axis_name=feature_axis).astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down_partial = _dot_f32(activated, weights.down_local)
    with jax.named_scope("glm_perf_bf16_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name=expert_axis).astype(jnp.bfloat16)
