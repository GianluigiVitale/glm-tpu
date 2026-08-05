"""Raw-FP8 batch-one layer bodies for a topology-local stage group.

These bodies run *inside* a surrounding global ``shard_map``.  They therefore
accept already-local final-owner shards and an explicit local slot rather than
constructing a nested mesh.  The only communication is one psum over the
provided stage-local axis group.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
from jax import lax
import jax.numpy as jnp

from .reference.fp8 import dequantize_fp8_bits_block_weight
from .reference.linear import linear, residual_add, silu
from .reference.moe import GlmMoeNumericalContract, route_glm_noaux_tc
from .reference.rmsnorm import rms_norm


def _axis_groups(
    groups: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if groups is None:
        return None
    value = tuple(tuple(int(rank) for rank in group) for group in groups)
    if not value or any(not group for group in value):
        raise ValueError("stage-local axis groups must be non-empty")
    return value


def stage_local_dense_fp8_mapped(
    residual: Any,
    norm_weight: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    axis_name: str,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    epsilon: float = 1e-5,
) -> Any:
    """Execute one dense SwiGLU from local raw shards and one local combine."""

    groups = _axis_groups(axis_index_groups)
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("dense decode residual must contain exactly one row")
    hidden = residual.shape[1]
    if norm_weight.shape != (hidden,):
        raise ValueError("dense norm shape disagrees with hidden size")
    if gate_bits.shape != up_bits.shape or gate_bits.shape[1] != hidden:
        raise ValueError("dense local gate/up shards are invalid")
    if down_bits.shape != (hidden, gate_bits.shape[0]):
        raise ValueError("dense local down shard is invalid")
    gate_weight = dequantize_fp8_bits_block_weight(
        gate_bits, gate_scale, block_shape=block_shape
    )
    up_weight = dequantize_fp8_bits_block_weight(
        up_bits, up_scale, block_shape=block_shape
    )
    down_weight = dequantize_fp8_bits_block_weight(
        down_bits, down_scale, block_shape=block_shape
    )
    normalized = rms_norm(residual, norm_weight, epsilon=epsilon)
    activated = (
        silu(linear(normalized, gate_weight))
        * linear(normalized, up_weight)
    ).astype(normalized.dtype)
    local_update = linear(activated, down_weight)
    update = lax.psum(
        local_update,
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    return residual_add(residual, update)


def stage_local_moe_fp8_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[Any, Any, Any]:
    """Route and execute selected raw-FP8 experts without a BF16 overlay."""

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size):
        raise ValueError("MoE decode hidden state must contain one exact row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    local_expert_shape = (
        contract.local_experts,
        contract.intermediate_size,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != local_expert_shape or expert_up_bits.shape != local_expert_shape:
        raise ValueError("local expert gate/up FP8 shapes are invalid")
    if expert_down_bits.shape != (
        contract.local_experts,
        contract.hidden_size,
        contract.intermediate_size,
    ):
        raise ValueError("local expert down FP8 shape is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    expert_start = local_slot.astype(jnp.int32) * jnp.int32(
        contract.local_experts
    )
    routed_parts = []
    for position in range(contract.top_k):
        global_expert = route_indices[0, position]
        owns_expert = (
            (global_expert >= expert_start)
            & (global_expert < expert_start + contract.local_experts)
        )
        local_expert = jnp.clip(
            global_expert - expert_start, 0, contract.local_experts - 1
        )

        def compute(_: None) -> Any:
            gate = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_gate_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_gate_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            up = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_up_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_up_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            down = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_down_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_down_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            activated = (
                silu(linear(hidden_states, gate))
                * linear(hidden_states, up)
            ).astype(hidden_states.dtype)
            output = linear(activated, down)
            return output * route_weights[0, position].astype(hidden_states.dtype)

        routed_parts.append(
            lax.cond(
                owns_expert,
                compute,
                lambda _: jnp.zeros_like(hidden_states),
                operand=None,
            )
        )
    local_routed = jnp.sum(
        jnp.stack(routed_parts, axis=0),
        axis=0,
        dtype=hidden_states.dtype,
    )
    shared_gate = dequantize_fp8_bits_block_weight(
        shared_gate_bits,
        shared_gate_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_up = dequantize_fp8_bits_block_weight(
        shared_up_bits,
        shared_up_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_down = dequantize_fp8_bits_block_weight(
        shared_down_bits,
        shared_down_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_activated = (
        silu(linear(hidden_states, shared_gate))
        * linear(hidden_states, shared_up)
    ).astype(hidden_states.dtype)
    local_shared = linear(shared_activated, shared_down)
    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    output = (combined[0] * scale + combined[1]).astype(hidden_states.dtype)
    return output, route_indices, route_weights
