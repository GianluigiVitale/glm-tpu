"""Exactness-first WS32_2D one-row dense and MoE reference bodies."""

from __future__ import annotations

from typing import Any

from jax import lax
import jax
import jax.numpy as jnp

from .reference.fp8 import dequantize_fp8_bits_block_weight
from .reference.moe import GlmMoeNumericalContract


def _fp32_dot(lhs: Any, rhs_out_in: Any) -> Any:
    return lax.dot_general(
        lhs,
        rhs_out_in,
        dimension_numbers=(((lhs.ndim - 1,), (rhs_out_in.ndim - 1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _bf16_dot(lhs: Any, rhs_out_in: Any) -> Any:
    return lax.dot_general(
        lhs,
        rhs_out_in,
        dimension_numbers=(((lhs.ndim - 1,), (rhs_out_in.ndim - 1,)), ((), ())),
        preferred_element_type=jnp.bfloat16,
    )


def _dequantize(bits: Any, scale: Any, block_shape: tuple[int, int]) -> Any:
    return dequantize_fp8_bits_block_weight(
        bits,
        scale,
        block_shape=block_shape,
        output_dtype=jnp.bfloat16,
    )


def ws32_dense_fp8_mapped(
    hidden_local: Any,
    gate_bits_local: Any,
    gate_scale_local: Any,
    up_bits_local: Any,
    up_scale_local: Any,
    down_bits_local: Any,
    down_scale_local: Any,
    *,
    expert_axis: str = "expert",
    feature_axis: str = "feature",
    block_shape: tuple[int, int] = (128, 128),
) -> Any:
    """Run one reciprocal 2D dense MLP while keeping the residual sharded."""

    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 dense input must contain one live row")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 dense input must be bfloat16")
    if gate_bits_local.shape != up_bits_local.shape or (
        gate_bits_local.shape[-1] != hidden_local.shape[-1]
    ):
        raise ValueError("WS32 dense gate/up local shapes drifted")
    if down_bits_local.shape != (
        hidden_local.shape[-1],
        gate_bits_local.shape[0],
    ):
        raise ValueError("WS32 dense reciprocal down shape drifted")

    with jax.named_scope("greenfield_ws32_dense"):
        gate_partial = _fp32_dot(
            hidden_local,
            _dequantize(gate_bits_local, gate_scale_local, block_shape),
        )
        up_partial = _fp32_dot(
            hidden_local,
            _dequantize(up_bits_local, up_scale_local, block_shape),
        )
        with jax.named_scope("feature_gate_up_reduce"):
            gate_up = lax.psum(
                jnp.stack((gate_partial, up_partial), axis=0),
                axis_name=feature_axis,
            ).astype(jnp.bfloat16)
        activated = (
            gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]
        ).astype(jnp.bfloat16)
        down_partial = _fp32_dot(
            activated,
            _dequantize(down_bits_local, down_scale_local, block_shape),
        )
        with jax.named_scope("expert_down_reduce"):
            return lax.psum(
                down_partial, axis_name=expert_axis
            ).astype(jnp.bfloat16)


def ws32_moe_fp8_from_routes_mapped(
    hidden_local: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits_local: Any,
    expert_gate_scale_local: Any,
    expert_up_bits_local: Any,
    expert_up_scale_local: Any,
    expert_down_bits_local: Any,
    expert_down_scale_local: Any,
    shared_gate_bits_local: Any,
    shared_gate_scale_local: Any,
    shared_up_bits_local: Any,
    shared_up_scale_local: Any,
    shared_down_bits_local: Any,
    shared_down_scale_local: Any,
    *,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    expert_axis: str = "expert",
    feature_axis: str = "feature",
) -> Any:
    """Run exact top-k routes on reciprocal expert/feature weight shards.

    Shared-expert weights are explicitly replicated over ``expert`` and
    feature-sharded, the first of the specification's three allowed shared
    layouts.  Routed identities are sharded over ``expert``.  Only compact
    route metadata is replicated; the hidden row is never reconstructed.
    """

    if contract.stage_size != 8:
        raise ValueError("WS32 MoE contract requires stage_size=8")
    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 MoE input must contain one live row")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 MoE input must be bfloat16")
    if route_indices.shape != (1, contract.top_k) or (
        route_indices.dtype != jnp.int32
    ):
        raise ValueError("WS32 routes must contain one exact int32 top-k row")
    if route_weights.shape != (1, contract.top_k) or (
        route_weights.dtype != jnp.float32
    ):
        raise ValueError("WS32 route weights must contain one FP32 top-k row")
    local_experts = expert_gate_bits_local.shape[0]
    if local_experts <= 0 or contract.num_experts % local_experts:
        raise ValueError("WS32 local expert ownership is invalid")
    if local_experts != contract.local_experts:
        raise ValueError("WS32 local expert count disagrees with the contract")
    expert_axis_size = contract.num_experts // local_experts
    expert_start = lax.axis_index(expert_axis) * local_experts
    routed_parts = []
    for route_position in range(contract.top_k):
        expert_id = route_indices[0, route_position]
        owns = (expert_id >= expert_start) & (
            expert_id < expert_start + local_experts
        )
        local_expert = jnp.clip(
            expert_id - expert_start,
            jnp.int32(0),
            jnp.int32(local_experts - 1),
        )

        def compute(_: None) -> Any:
            gate_bits = lax.dynamic_index_in_dim(
                expert_gate_bits_local, local_expert, 0, False
            )
            gate_scale = lax.dynamic_index_in_dim(
                expert_gate_scale_local, local_expert, 0, False
            )
            up_bits = lax.dynamic_index_in_dim(
                expert_up_bits_local, local_expert, 0, False
            )
            up_scale = lax.dynamic_index_in_dim(
                expert_up_scale_local, local_expert, 0, False
            )
            down_bits = lax.dynamic_index_in_dim(
                expert_down_bits_local, local_expert, 0, False
            )
            down_scale = lax.dynamic_index_in_dim(
                expert_down_scale_local, local_expert, 0, False
            )
            gate_partial = _fp32_dot(
                hidden_local,
                _dequantize(gate_bits, gate_scale, contract.fp8_block_shape),
            )
            up_partial = _fp32_dot(
                hidden_local,
                _dequantize(up_bits, up_scale, contract.fp8_block_shape),
            )
            with jax.named_scope(
                f"greenfield_ws32_moe/route_{route_position}_feature_reduce"
            ):
                gate_up = lax.psum(
                    jnp.stack((gate_partial, up_partial), axis=0),
                    axis_name=feature_axis,
                ).astype(jnp.bfloat16)
            activated = (
                gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]
            ).astype(jnp.bfloat16)
            output = _bf16_dot(
                activated,
                _dequantize(down_bits, down_scale, contract.fp8_block_shape),
            )
            return (
                output
                * route_weights[0, route_position].astype(jnp.bfloat16)
            ).astype(jnp.bfloat16)

        routed_parts.append(
            lax.cond(
                owns,
                compute,
                lambda _: jnp.zeros_like(hidden_local),
                operand=None,
            )
        )
    local_routed = jnp.sum(
        jnp.stack(tuple(routed_parts), axis=0),
        axis=0,
        dtype=jnp.bfloat16,
    )
    with jax.named_scope("greenfield_ws32_moe/routed_expert_reduce"):
        routed = lax.psum(
            local_routed.astype(jnp.float32), axis_name=expert_axis
        ).astype(jnp.bfloat16)

    shared_gate_partial = _fp32_dot(
        hidden_local,
        _dequantize(
            shared_gate_bits_local,
            shared_gate_scale_local,
            contract.fp8_block_shape,
        ),
    )
    shared_up_partial = _fp32_dot(
        hidden_local,
        _dequantize(
            shared_up_bits_local,
            shared_up_scale_local,
            contract.fp8_block_shape,
        ),
    )
    with jax.named_scope("greenfield_ws32_moe/shared_feature_reduce"):
        shared_gate_up = lax.psum(
            jnp.stack((shared_gate_partial, shared_up_partial), axis=0),
            axis_name=feature_axis,
        ).astype(jnp.bfloat16)
    shared_activated = (
        shared_gate_up[0]
        * jax.nn.sigmoid(shared_gate_up[0])
        * shared_gate_up[1]
    ).astype(jnp.bfloat16)
    shared = _bf16_dot(
        shared_activated,
        _dequantize(
            shared_down_bits_local,
            shared_down_scale_local,
            contract.fp8_block_shape,
        ),
    )
    if expert_axis_size != 8:
        raise ValueError("WS32 routed expert ownership requires expert axis 8")
    routed_scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=jnp.bfloat16
    )
    return (routed * routed_scale + shared).astype(jnp.bfloat16)


def ws32_moe_pallas_from_routes_mapped(
    hidden_local: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits_local: Any,
    expert_gate_scale_local: Any,
    expert_up_bits_local: Any,
    expert_up_scale_local: Any,
    expert_down_bits_local: Any,
    expert_down_scale_local: Any,
    shared_gate_bits_local: Any,
    shared_gate_scale_local: Any,
    shared_up_bits_local: Any,
    shared_up_scale_local: Any,
    shared_down_bits_local: Any,
    shared_down_scale_local: Any,
    *,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    expert_axis: str = "expert",
    feature_axis: str = "feature",
    interpret: bool = False,
) -> Any:
    """Run the WS32 one-row MoE using the existing raw-FP8 Pallas kernels.

    The sealed WS32 derivative stores each routed projection in checkpoint
    ``[expert, out, in]`` order.  Selecting one owned expert therefore gives
    the exact rank-two ``[out, in]`` input required by the standalone Pallas
    kernel without a transpose, decoded-weight overlay, or full-table copy.
    Gate/up accumulators stay FP32 through the feature-4 reduction; down and
    route weighting retain the reference BF16 boundaries before the expert-8
    combine.  This is a distinct default-off challenger, not a replacement
    for :func:`ws32_moe_fp8_from_routes_mapped`.
    """

    # Local imports deliberately keep the frozen reference function's source
    # locations—and therefore its protected compiler metadata—unchanged.
    from .pallas.fp8_matmul import (
        Fp8BlockMatmulConfig,
        fp8_block_matmul,
        fp8_block_matmul_f32,
    )

    if contract.stage_size != 8:
        raise ValueError("WS32 Pallas contract requires stage_size=8")
    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 Pallas input must contain one live row")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 Pallas input must be bfloat16")
    if route_indices.shape != (1, contract.top_k) or (
        route_indices.dtype != jnp.int32
    ):
        raise ValueError("WS32 Pallas routes must contain one exact int32 row")
    if route_weights.shape != (1, contract.top_k) or (
        route_weights.dtype != jnp.float32
    ):
        raise ValueError("WS32 Pallas route weights must contain one FP32 row")

    local_hidden = hidden_local.shape[-1]
    local_experts = expert_gate_bits_local.shape[0]
    if local_experts != contract.local_experts:
        raise ValueError("WS32 Pallas local expert ownership drifted")
    expected_gate = (local_experts, contract.intermediate_size, local_hidden)
    expected_down = (local_experts, local_hidden, contract.intermediate_size)
    if expert_gate_bits_local.shape != expected_gate or (
        expert_up_bits_local.shape != expected_gate
    ):
        raise ValueError("WS32 Pallas routed gate/up shapes drifted")
    if expert_down_bits_local.shape != expected_down:
        raise ValueError("WS32 Pallas routed down shape drifted")
    expected_gate_scale = (
        local_experts,
        contract.intermediate_size // contract.fp8_block_shape[0],
        local_hidden // contract.fp8_block_shape[1],
    )
    expected_down_scale = (
        local_experts,
        local_hidden // contract.fp8_block_shape[0],
        contract.intermediate_size // contract.fp8_block_shape[1],
    )
    if expert_gate_scale_local.shape != expected_gate_scale or (
        expert_up_scale_local.shape != expected_gate_scale
    ):
        raise ValueError("WS32 Pallas routed gate/up scale shapes drifted")
    if expert_down_scale_local.shape != expected_down_scale:
        raise ValueError("WS32 Pallas routed down scale shape drifted")
    if shared_gate_bits_local.shape != (
        contract.intermediate_size,
        local_hidden,
    ) or shared_up_bits_local.shape != (
        contract.intermediate_size,
        local_hidden,
    ):
        raise ValueError("WS32 Pallas shared gate/up shapes drifted")
    if shared_down_bits_local.shape != (
        local_hidden,
        contract.intermediate_size,
    ):
        raise ValueError("WS32 Pallas shared down shape drifted")

    config = Fp8BlockMatmulConfig(
        block_shape=contract.fp8_block_shape,
        output_tile=contract.fp8_block_shape[0],
        contraction_tile=contract.fp8_block_shape[1],
    )
    expert_start = lax.axis_index(expert_axis) * local_experts
    routed_parts = []
    for route_position in range(contract.top_k):
        expert_id = route_indices[0, route_position]
        owns = (expert_id >= expert_start) & (
            expert_id < expert_start + local_experts
        )
        local_expert = jnp.clip(
            expert_id - expert_start,
            jnp.int32(0),
            jnp.int32(local_experts - 1),
        )

        def compute(_: None) -> Any:
            gate_bits = lax.dynamic_index_in_dim(
                expert_gate_bits_local, local_expert, 0, False
            )
            gate_scale = lax.dynamic_index_in_dim(
                expert_gate_scale_local, local_expert, 0, False
            )
            up_bits = lax.dynamic_index_in_dim(
                expert_up_bits_local, local_expert, 0, False
            )
            up_scale = lax.dynamic_index_in_dim(
                expert_up_scale_local, local_expert, 0, False
            )
            down_bits = lax.dynamic_index_in_dim(
                expert_down_bits_local, local_expert, 0, False
            )
            down_scale = lax.dynamic_index_in_dim(
                expert_down_scale_local, local_expert, 0, False
            )
            gate_partial = fp8_block_matmul_f32(
                hidden_local,
                gate_bits,
                gate_scale,
                config=config,
                interpret=interpret,
            )
            up_partial = fp8_block_matmul_f32(
                hidden_local,
                up_bits,
                up_scale,
                config=config,
                interpret=interpret,
            )
            with jax.named_scope(
                f"greenfield_ws32_moe_pallas/route_{route_position}_feature_reduce"
            ):
                gate_up = lax.psum(
                    jnp.stack((gate_partial, up_partial), axis=0),
                    axis_name=feature_axis,
                ).astype(jnp.bfloat16)
            activated = (
                gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]
            ).astype(jnp.bfloat16)
            output = fp8_block_matmul(
                activated,
                down_bits,
                down_scale,
                config=config,
                interpret=interpret,
            )
            return (
                output
                * route_weights[0, route_position].astype(jnp.bfloat16)
            ).astype(jnp.bfloat16)

        routed_parts.append(
            lax.cond(
                owns,
                compute,
                lambda _: jnp.zeros_like(hidden_local),
                operand=None,
            )
        )

    local_routed = jnp.sum(
        jnp.stack(tuple(routed_parts), axis=0),
        axis=0,
        dtype=jnp.bfloat16,
    )
    with jax.named_scope("greenfield_ws32_moe_pallas/routed_expert_reduce"):
        routed = lax.psum(
            local_routed.astype(jnp.float32), axis_name=expert_axis
        ).astype(jnp.bfloat16)

    shared_gate_partial = fp8_block_matmul_f32(
        hidden_local,
        shared_gate_bits_local,
        shared_gate_scale_local,
        config=config,
        interpret=interpret,
    )
    shared_up_partial = fp8_block_matmul_f32(
        hidden_local,
        shared_up_bits_local,
        shared_up_scale_local,
        config=config,
        interpret=interpret,
    )
    with jax.named_scope("greenfield_ws32_moe_pallas/shared_feature_reduce"):
        shared_gate_up = lax.psum(
            jnp.stack((shared_gate_partial, shared_up_partial), axis=0),
            axis_name=feature_axis,
        ).astype(jnp.bfloat16)
    shared_activated = (
        shared_gate_up[0]
        * jax.nn.sigmoid(shared_gate_up[0])
        * shared_gate_up[1]
    ).astype(jnp.bfloat16)
    shared = fp8_block_matmul(
        shared_activated,
        shared_down_bits_local,
        shared_down_scale_local,
        config=config,
        interpret=interpret,
    )
    routed_scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=jnp.bfloat16
    )
    return (routed * routed_scale + shared).astype(jnp.bfloat16)
