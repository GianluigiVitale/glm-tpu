"""Resident BF16 projections of the production prefill.

``resident_matmul*``, ``resident_q_absorb`` and ``resident_value`` are the non-routed projections
of the prefill bodies over the resident BF16 tables (``glm_tpu.models.glm_moe_dsa.weights``).
They refuse raw uint8 bits and any scale argument, so a raw FP8 table can never reach them
silently. Projection accumulation order is a numerical boundary even though decoded operands
and explicit rounding points are exact.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import jax
from jax import lax

from glm_tpu.layers.contracts import Bf16DsaWeights


def _require_table(weight, scale):
    if weight.ndim != 2 or weight.dtype != jnp.bfloat16 or scale is not None:
        raise ValueError("D8 prefill requires resident BF16 tables with no scale argument")


def resident_matmul_f32(lhs, weight, scale=None, *, interpret=False):
    _require_table(weight, scale)
    if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16 or lhs.shape[1] != weight.shape[1]:
        raise ValueError("D8 prefill matmul requires matching BF16 operands")
    # CPU's batched DotThunk cannot execute BF16 x BF16 -> FP32. The
    # interpret path widens already-rounded BF16 operands exactly, matching
    # the reference Pallas interpreter; production keeps BF16 MXU operands.
    return dot_f32(lhs.astype(jnp.float32), weight.astype(jnp.float32)) if interpret else dot_f32(lhs, weight)


def resident_matmul(lhs, weight, scale=None, *, interpret=False):
    return resident_matmul_f32(lhs, weight, scale, interpret=interpret).astype(jnp.bfloat16)


def resident_q_absorb(query, weight, scale=None, *, interpret=False):
    _require_table(weight, scale)
    if query.ndim != 3 or query.dtype != jnp.bfloat16 or weight.shape[0] % query.shape[1]:
        raise ValueError("D8 prefill structured query geometry drifted")
    heads, qwidth = query.shape[1:]
    table = weight.reshape(heads, -1, weight.shape[1])
    if table.shape[1] <= qwidth:
        raise ValueError("D8 prefill structured table needs key and value rows")
    key = table[:, :qwidth]
    if interpret:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    return jnp.einsum("rhq,hqk->rhk", query, key, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def resident_value(latent, weight, scale=None, *, qk_nope_head_dim=192, interpret=False):
    _require_table(weight, scale)
    if (
        latent.ndim != 3
        or latent.dtype != jnp.bfloat16
        or weight.shape[0] % latent.shape[1]
        or weight.shape[1] != latent.shape[2]
    ):
        raise ValueError("D8 prefill structured value geometry drifted")
    table = weight.reshape(latent.shape[1], -1, latent.shape[2])
    if not 0 < qk_nope_head_dim < table.shape[1]:
        raise ValueError("D8 prefill structured key/value split drifted")
    value = table[:, qk_nope_head_dim:]
    if interpret:
        latent, value = latent.astype(jnp.float32), value.astype(jnp.float32)
    return jnp.einsum("rhk,hvk->rhv", latent, value, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def residual_add(residual: jax.Array, update: jax.Array) -> jax.Array:
    """Add one transformer residual without dtype promotion or broadcasting."""

    if residual.shape != update.shape:
        raise ValueError("residual and update shapes must match exactly")
    if residual.dtype != update.dtype:
        raise ValueError("residual and update dtypes must match exactly")
    return (residual + update).astype(residual.dtype)


def require_rows(value: Any) -> None:
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError("WS32 prefill requires nonempty [rows,features]")
    if value.dtype != jnp.bfloat16:
        raise ValueError("WS32 prefill activations must be bfloat16")


def prefill_linear(
    lhs_local: Any,
    weight_local: Any,
    *,
    reduction_axis: Literal["feature", "expert"],
    interpret: bool = False,
) -> Any:
    """Project live prompt rows, reduce local FP32 partials, then round BF16.

    ``feature`` contracts hidden shards; ``expert`` contracts reciprocal
    intermediate shards. Neither operation reconstructs full-pod hidden state.
    The weight tile is shared across rows inside the existing Pallas call,
    not reloaded by a scan of individual token projections.
    """

    require_rows(lhs_local)
    if reduction_axis not in ("feature", "expert"):
        raise ValueError("WS32 prefill reduction must be feature or expert")
    partial = resident_matmul_f32(lhs_local, weight_local, interpret=interpret)
    with jax.named_scope(f"prefill_linear/{reduction_axis}_reduce"):
        return lax.psum(partial, axis_name=reduction_axis).astype(jnp.bfloat16)


# ----------------------------------------------------------------------------- projections
def dot_f32(x: Any, weight_out_in: Any) -> Any:
    """``x @ W.T`` with BF16 operands and an FP32 accumulator (the MXU boundary of the FP8 kernels)."""

    return lax.dot_general(x, weight_out_in, (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def feature_linear(x: Any, weight_local: Any, feature_axis: str) -> Any:
    partial = dot_f32(x, weight_local)
    with jax.named_scope("bf16_linear/feature_reduce"):
        return lax.psum(partial, axis_name=feature_axis).astype(jnp.bfloat16)


def expert_linear(x: Any, weight_local: Any, expert_axis: str) -> Any:
    partial = dot_f32(x, weight_local)
    with jax.named_scope("bf16_linear/expert_reduce"):
        return lax.psum(partial, axis_name=expert_axis).astype(jnp.bfloat16)


# ----------------------------------------------------------------------------- attention bodies
def head_weight_partial(normalized: Any, weights: Bf16DsaWeights) -> Any:
    return lax.dot_general(
        normalized.astype(jnp.float32),
        weights.head_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
