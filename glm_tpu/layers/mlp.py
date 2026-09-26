"""The dense MLP bodies of the production engine: the multirow prefill MLP (``prefill_dense``) and
the decode layer's MLP on the resident BF16 tables (``dense_bf16``), each with a feature-4 gate/up
reduction and an expert-8 down reduction.

The canonical row placement of the prefill dense suffix (four 32-live-row placements inside B128)
is ``glm_tpu.models.glm_moe_dsa.decoder_layer.prefill_dense_canonical``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.layers.contracts import Bf16DenseWeights
from glm_tpu.layers.linear import dot_f32, require_rows, resident_matmul_f32


def prefill_dense(
    hidden_local: Any,
    gate_local: Any,
    up_local: Any,
    down_local: Any,
    *,
    interpret: bool = False,
) -> Any:
    """Batched reciprocal-sharded dense MLP, not the StrategyND overlay.

    It has the numerical boundaries of the FP8-table ``ws32_dense_pallas_mapped``
    (archived at ``archive/research-20260922``; :func:`dense_bf16` follows it), but it
    is not interchangeable with the decoder's StrategyND dense association: a layer
    path that switches between them needs its own bounded comparisons and
    decoder-level evidence.
    """

    require_rows(hidden_local)
    if gate_local.ndim != 2 or (gate_local.shape != up_local.shape or gate_local.shape[1] != hidden_local.shape[1]):
        raise ValueError("prefill gate/up geometry drifted")
    local_intermediate, local_hidden = gate_local.shape
    if down_local.shape != (local_hidden, local_intermediate):
        raise ValueError("prefill down geometry drifted")
    gate_partial = resident_matmul_f32(hidden_local, gate_local, interpret=interpret)
    up_partial = resident_matmul_f32(hidden_local, up_local, interpret=interpret)
    with jax.named_scope("prefill_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(jnp.stack((gate_partial, up_partial)), axis_name="feature").astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down_partial = resident_matmul_f32(activated, down_local, interpret=interpret)
    with jax.named_scope("prefill_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name="expert").astype(jnp.bfloat16)


# ----------------------------------------------------------------------------- MLP bodies
def dense_bf16(normalized: Any, weights: Bf16DenseWeights, *, expert_axis: str, feature_axis: str) -> Any:
    """The decode dense MLP on BF16 tables.

    Otherwise as the FP8-table ``ws32_dense_pallas_mapped`` archived at ``archive/research-20260922``.
    """

    gate_partial = dot_f32(normalized, weights.gate_local)
    up_partial = dot_f32(normalized, weights.up_local)
    with jax.named_scope("bf16_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(jnp.stack((gate_partial, up_partial), axis=0), axis_name=feature_axis).astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down_partial = dot_f32(activated, weights.down_local)
    with jax.named_scope("bf16_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name=expert_axis).astype(jnp.bfloat16)
