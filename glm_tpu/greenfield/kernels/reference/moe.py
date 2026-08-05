"""Exactness-first GLM-5.2 MoE reference implementation.

This module restates the model contract directly in JAX.  It imports no
legacy model or TPU-inference code.  The accepted engine is only an oracle for
captured tensors and numerical comparisons.

The PP8 form owns all 256 routed experts inside one four-chip pipeline stage:
64 complete experts per chip.  The shared expert is tensor-sharded over its
intermediate dimension.  Routed and shared partials are stacked before one
stage-local ``psum``; the two value domains remain separate and routed scale
2.5 is applied only after reduction.  Consequently the only collective in the
kernel is the combine over the four-chip ``expert`` axis.

The implementation deliberately targets a true batch-one decode row.  It is
a readable fallback and correctness oracle; optimized GMM/Pallas kernels may
replace its selected-expert loop only after matching it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P


@dataclass(frozen=True, slots=True)
class GlmMoeNumericalContract:
    """Compile-relevant arithmetic contract for one sparse GLM layer."""

    hidden_size: int = 6144
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 8
    stage_size: int = 4
    routed_scaling_factor: float = 2.5
    fp8_block_shape: tuple[int, int] = (128, 128)
    activation_dtype: str = "bfloat16"
    router_dtype: str = "float32"
    fp8_scale_dtype: str = "float32"
    scoring_function: str = "sigmoid"
    routing_method: str = "noaux_tc"
    routing_tie_policy: str = "jax_lax_top_k_lowest_expert_id"
    reduction_association: str = (
        "top_k_axis_then_stage_psum; routed_and_shared_stacked_not_mixed"
    )

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "stage_size",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.num_experts % self.stage_size:
            raise ValueError("num_experts must divide evenly over the stage")
        if self.intermediate_size % self.stage_size:
            raise ValueError("shared intermediate size must divide over the stage")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if self.scoring_function != "sigmoid":
            raise ValueError("GLM-5.2 requires sigmoid MoE scoring")
        if self.routing_method != "noaux_tc":
            raise ValueError("GLM-5.2 requires noaux_tc routing")

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.stage_size

    @property
    def local_shared_intermediate(self) -> int:
        return self.intermediate_size // self.stage_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "fp8_block_shape": list(self.fp8_block_shape),
            "fp8_scale_dtype": self.fp8_scale_dtype,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "reduction_association": self.reduction_association,
            "routed_scaling_factor": self.routed_scaling_factor,
            "router_dtype": self.router_dtype,
            "routing_method": self.routing_method,
            "routing_tie_policy": self.routing_tie_policy,
            "scoring_function": self.scoring_function,
            "stage_size": self.stage_size,
            "top_k": self.top_k,
        }


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _bf16_dot(lhs: jax.Array, rhs_out_in: jax.Array) -> jax.Array:
    """Contract the final axes and request an activation-dtype result.

    Weights retain checkpoint orientation ``[out, in]``.  TPU bfloat16 MXU
    execution uses floating-point accumulators and returns bfloat16 here, which
    matches the accepted v4 dequantized-GMM boundary.
    """

    return lax.dot_general(
        lhs,
        rhs_out_in,
        dimension_numbers=(((lhs.ndim - 1,), (rhs_out_in.ndim - 1,)), ((), ())),
        preferred_element_type=lhs.dtype,
    )


def _silu(value: jax.Array) -> jax.Array:
    return value * jax.nn.sigmoid(value)


def dequantize_fp8_block_weight(
    weight: jax.Array,
    scale: jax.Array,
    *,
    block_shape: tuple[int, int] = (128, 128),
    output_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Fold FP8 inverse block scales into one checkpoint-oriented weight.

    ``weight`` is ``[out, in]`` and ``scale`` is
    ``[ceil(out/block_out), ceil(in/block_in)]``.  GLM dimensions are exact
    multiples of 128; tails are supported so corruption tests can exercise
    non-production shapes without silently changing ownership.
    """

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("weight and scale must both be rank two")
    if len(block_shape) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in block_shape
    ):
        raise ValueError("block_shape must contain two positive integers")
    expected_scale = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(weight.shape, block_shape, strict=True)
    )
    if scale.shape != expected_scale:
        raise ValueError(
            f"scale must have shape {expected_scale} for weight {weight.shape}, "
            f"got {scale.shape}"
        )
    expanded = jnp.repeat(scale.astype(jnp.float32), block_shape[0], axis=0)
    expanded = jnp.repeat(expanded, block_shape[1], axis=1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.astype(jnp.float32) * expanded).astype(output_dtype)


def route_glm_noaux_tc_logits(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    top_k: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Return selected expert ids and unbiased normalized sigmoid weights.

    The correction bias participates only in selection.  The selected weights
    are gathered from the unbiased sigmoid scores and normalized in FP32.
    Routed scaling is intentionally absent; the accepted v4 path applies it to
    the reduced routed output.
    """

    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    _require_shape(
        "correction_bias", correction_bias, (router_logits.shape[1],)
    )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not (
        0 < top_k <= router_logits.shape[1]
    ):
        raise ValueError("top_k must be in [1, num_experts]")
    scores = jax.nn.sigmoid(router_logits.astype(jnp.float32))
    _, indices = lax.top_k(
        scores + correction_bias.astype(jnp.float32)[None, :], top_k
    )
    weights = jnp.take_along_axis(scores, indices, axis=-1)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True, dtype=jnp.float32)
    return indices.astype(jnp.int32), weights.astype(jnp.float32)


def route_glm_noaux_tc(
    hidden_states: jax.Array,
    router_weight: jax.Array,
    correction_bias: jax.Array,
    *,
    top_k: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Compute the FP32 router projection and exact noaux_tc selection."""

    if hidden_states.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("hidden_states and router_weight must both be rank two")
    if hidden_states.shape[1] != router_weight.shape[1]:
        raise ValueError("router weight input width must match hidden size")
    logits = lax.dot_general(
        hidden_states.astype(jnp.float32),
        router_weight.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return route_glm_noaux_tc_logits(logits, correction_bias, top_k=top_k)


def _validate_moe_shapes(
    hidden_states: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    contract: GlmMoeNumericalContract,
) -> None:
    _require_shape("hidden_states", hidden_states, (1, contract.hidden_size))
    _require_shape(
        "expert_gate",
        expert_gate,
        (contract.num_experts, contract.intermediate_size, contract.hidden_size),
    )
    _require_shape("expert_up", expert_up, expert_gate.shape)
    _require_shape(
        "expert_down",
        expert_down,
        (contract.num_experts, contract.hidden_size, contract.intermediate_size),
    )
    _require_shape(
        "shared_gate",
        shared_gate,
        (contract.intermediate_size, contract.hidden_size),
    )
    _require_shape("shared_up", shared_up, shared_gate.shape)
    _require_shape(
        "shared_down",
        shared_down,
        (contract.hidden_size, contract.intermediate_size),
    )


def reference_moe_from_routes(
    hidden_states: jax.Array,
    route_indices: jax.Array,
    route_weights: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
) -> jax.Array:
    """Unsharded exactness oracle using the same batch-one association."""

    _validate_moe_shapes(
        hidden_states,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        contract,
    )
    _require_shape("route_indices", route_indices, (1, contract.top_k))
    _require_shape("route_weights", route_weights, (1, contract.top_k))

    routed_parts = []
    for position in range(contract.top_k):
        expert_id = route_indices[0, position]
        gate = lax.dynamic_index_in_dim(expert_gate, expert_id, axis=0, keepdims=False)
        up = lax.dynamic_index_in_dim(expert_up, expert_id, axis=0, keepdims=False)
        down = lax.dynamic_index_in_dim(expert_down, expert_id, axis=0, keepdims=False)
        activated = _silu(_bf16_dot(hidden_states, gate)) * _bf16_dot(
            hidden_states, up
        )
        expert_out = _bf16_dot(activated, down)
        routed_parts.append(
            expert_out * route_weights[0, position].astype(hidden_states.dtype)
        )
    routed = jnp.sum(
        jnp.stack(routed_parts, axis=0), axis=0, dtype=hidden_states.dtype
    )
    shared_activated = _silu(_bf16_dot(hidden_states, shared_gate)) * _bf16_dot(
        hidden_states, shared_up
    )
    shared = _bf16_dot(shared_activated, shared_down)
    scale = jnp.asarray(contract.routed_scaling_factor, dtype=hidden_states.dtype)
    return (routed * scale + shared).astype(hidden_states.dtype)


def _stage_local_moe_mapped(
    hidden_states: jax.Array,
    route_indices: jax.Array,
    route_weights: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    contract: GlmMoeNumericalContract,
    axis_name: str,
) -> jax.Array:
    """Per-chip body; leading expert/intermediate axes are already local."""

    stage_rank = lax.axis_index(axis_name)
    expert_start = stage_rank * contract.local_experts
    routed_parts = []
    for position in range(contract.top_k):
        global_expert = route_indices[0, position]
        owns_expert = jnp.logical_and(
            global_expert >= expert_start,
            global_expert < expert_start + contract.local_experts,
        )
        local_expert = jnp.clip(
            global_expert - expert_start, 0, contract.local_experts - 1
        )

        def compute(_: None) -> jax.Array:
            gate = lax.dynamic_index_in_dim(
                expert_gate, local_expert, axis=0, keepdims=False
            )
            up = lax.dynamic_index_in_dim(
                expert_up, local_expert, axis=0, keepdims=False
            )
            down = lax.dynamic_index_in_dim(
                expert_down, local_expert, axis=0, keepdims=False
            )
            activated = _silu(_bf16_dot(hidden_states, gate)) * _bf16_dot(
                hidden_states, up
            )
            output = _bf16_dot(activated, down)
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
        jnp.stack(routed_parts, axis=0), axis=0, dtype=hidden_states.dtype
    )
    shared_activated = _silu(_bf16_dot(hidden_states, shared_gate)) * _bf16_dot(
        hidden_states, shared_up
    )
    local_shared = _bf16_dot(shared_activated, shared_down)

    # One physical collective with two independent value domains.  Stacking
    # preserves the routed/shared arithmetic boundary while sharing a launch.
    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0), axis_name=axis_name
    )
    scale = jnp.asarray(contract.routed_scaling_factor, dtype=hidden_states.dtype)
    return (combined[0] * scale + combined[1]).astype(hidden_states.dtype)


def stage_local_moe_from_routes(
    hidden_states: jax.Array,
    route_indices: jax.Array,
    route_weights: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: Mesh,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_name: str = "expert",
) -> jax.Array:
    """Execute the reference MoE over exactly one topology-local stage."""

    if mesh.axis_names != (axis_name,):
        raise ValueError(
            f"one-layer mesh axes must be exactly {(axis_name,)}, got {mesh.axis_names}"
        )
    if mesh.devices.size != contract.stage_size:
        raise ValueError(
            f"stage mesh must contain {contract.stage_size} devices, "
            f"got {mesh.devices.size}"
        )
    _validate_moe_shapes(
        hidden_states,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        contract,
    )
    _require_shape("route_indices", route_indices, (1, contract.top_k))
    _require_shape("route_weights", route_weights, (1, contract.top_k))
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError("stage-local GLM MoE activation must be bfloat16")

    mapped = jax.shard_map(
        lambda *args: _stage_local_moe_mapped(
            *args, contract=contract, axis_name=axis_name
        ),
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            P(),
            P(axis_name),
            P(axis_name),
            P(axis_name),
            P(axis_name),
            P(axis_name),
            P(None, axis_name),
        ),
        out_specs=P(),
        check_vma=False,
    )
    return mapped(
        hidden_states,
        route_indices,
        route_weights,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
    )


def stage_local_moe(
    hidden_states: jax.Array,
    router_weight: jax.Array,
    correction_bias: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    *,
    mesh: Mesh,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_name: str = "expert",
) -> jax.Array:
    """Route and execute one complete batch-one GLM MoE layer."""

    _require_shape(
        "router_weight",
        router_weight,
        (contract.num_experts, contract.hidden_size),
    )
    _require_shape("correction_bias", correction_bias, (contract.num_experts,))
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    return stage_local_moe_from_routes(
        hidden_states,
        route_indices,
        route_weights,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        mesh=mesh,
        contract=contract,
        axis_name=axis_name,
    )
