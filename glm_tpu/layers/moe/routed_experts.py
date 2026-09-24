"""The routed-expert MoE of the production engine: prefill (routed FP8 expert panels, resident
BF16 shared expert) and decode (``moe_grouped_routes``, the route-grouped FP8 projections of
``glm_tpu.kernels.fp8_grouped_matmul.kernel``).

In prefill the routes are grouped by expert (stable), the activations packed into M32 panels
(``glm_tpu.kernels.fp8_grouped_matmul.panels``) and the three routed projections run through the
FP8 panel kernel (``glm_tpu.kernels.fp8_grouped_matmul.panel_kernel``); the weighted routes are
summed in FP32 before the expert reduction (unlike decode's BF16 route sum). The shared expert
uses the resident BF16 tables. The admitted profile is hard-wired (S2d fold).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.layers.contracts import GlmMoeNumericalContract
from glm_tpu.layers.linear import resident_matmul, resident_matmul_f32
from glm_tpu.kernels.fp8_grouped_matmul.kernel import RoutedProjectionConfig, fp8_routed_projection
from glm_tpu.kernels.fp8_grouped_matmul.panel_kernel import prefill_panel_fp8_matmul
from glm_tpu.kernels.fp8_grouped_matmul.panels import build_expert_panels


def prefill_moe_from_routes(
    hidden_local: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits_local: Any,
    expert_gate_scale_local: Any,
    expert_up_bits_local: Any,
    expert_up_scale_local: Any,
    expert_down_bits_local: Any,
    expert_down_scale_local: Any,
    shared_gate_local: Any,
    shared_up_local: Any,
    shared_down_local: Any,
    *,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Return (local hidden shard, local health); caller must gate on all chips.

    Requires the validated expert8/feature4 mesh and replicated routing inputs.
    A false health bit forbids serving; it must not be discarded by a caller.
    The panels pack activations, not weights, and reuse one expert-relative plan
    for all three routed projections.
    """
    if contract.stage_size != 8:
        raise ValueError("WS32 prefill MoE requires stage_size=8")
    if (
        hidden_local.ndim != 2
        or hidden_local.shape[0] <= 0
        or hidden_local.dtype != jnp.bfloat16
        or hidden_local.shape[1] * 4 != contract.hidden_size
    ):
        raise ValueError("WS32 prefill MoE requires BF16 live rows and feature4 hidden shards")
    rows, local_hidden = hidden_local.shape
    expected_routes = (rows, contract.top_k)
    if route_indices.shape != expected_routes or route_weights.shape != expected_routes:
        raise ValueError("prefill routing geometry differs from contract")
    if route_weights.dtype != jnp.float32:
        raise ValueError("prefill route weights must be FP32")
    gate_shape = (contract.local_experts, contract.intermediate_size, local_hidden)
    if expert_gate_bits_local.shape != gate_shape or expert_up_bits_local.shape != gate_shape:
        raise ValueError("prefill expert gate/up geometry differs from contract")
    if expert_down_bits_local.shape != (
        contract.local_experts,
        local_hidden,
        contract.intermediate_size,
    ):
        raise ValueError("prefill expert down geometry differs from contract")
    if (
        shared_gate_local.shape != gate_shape[1:]
        or shared_up_local.shape != gate_shape[1:]
        or shared_down_local.shape != (local_hidden, contract.intermediate_size)
    ):
        raise ValueError("prefill shared expert geometry differs from contract")

    routes = group_prefill_routes(route_indices, num_experts=contract.num_experts)
    valid = routes.valid & jnp.all(jnp.isfinite(route_weights) & (route_weights >= 0))
    sorted_hidden = gather_prefill_route_rows(hidden_local, routes, top_k=contract.top_k)
    offset = lax.axis_index("expert").astype(jnp.int32) * contract.local_experts
    if contract.fp8_block_shape != (128, 128):
        raise ValueError("expert panels require checkpoint scale blocks128x128")
    panels = build_expert_panels(
        routes.group_sizes,
        offset,
        rows=sorted_hidden.shape[0],
        local_groups=contract.local_experts,
    )

    def project(x, w, s, dtype):
        return prefill_panel_fp8_matmul(x, w, s, panels, result_dtype=dtype, interpret=interpret)

    gate, gate_ok = project(sorted_hidden, expert_gate_bits_local, expert_gate_scale_local, jnp.float32)
    up, up_ok = project(sorted_hidden, expert_up_bits_local, expert_up_scale_local, jnp.float32)
    with jax.named_scope("prefill_moe/feature_reduce"):
        gate_up = lax.psum(jnp.stack((gate, up)), axis_name="feature").astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down, down_ok = project(activated, expert_down_bits_local, expert_down_scale_local, jnp.bfloat16)
    sorted_weights = route_weights.reshape(-1)[routes.sorted_flat_ids].astype(jnp.bfloat16)
    weighted = (down * sorted_weights[:, None]).astype(jnp.bfloat16)
    route_outputs = restore_prefill_route_rows(weighted, routes, top_k=contract.top_k)
    # Preserve the original [route,live rows,hidden] expression and reduction
    # axis. Expert execution order must not become route accumulation order.
    with jax.named_scope("prefill_moe/fp32_route_sum"):
        local_routed = jnp.sum(
            jnp.swapaxes(route_outputs, 0, 1).astype(jnp.float32),
            axis=0,
            dtype=jnp.float32,
        )
    with jax.named_scope("prefill_moe/expert_reduce"):
        local_sum_operand = local_routed.astype(jnp.float32)
        routed = lax.psum(local_sum_operand, axis_name="expert").astype(jnp.bfloat16)

    shared_gate = resident_matmul_f32(hidden_local, shared_gate_local, interpret=interpret)
    shared_up = resident_matmul_f32(hidden_local, shared_up_local, interpret=interpret)
    with jax.named_scope("prefill_moe/shared_feature_reduce"):
        shared_gate_up = lax.psum(jnp.stack((shared_gate, shared_up)), axis_name="feature").astype(jnp.bfloat16)
    shared_activation = (shared_gate_up[0] * jax.nn.sigmoid(shared_gate_up[0]) * shared_gate_up[1]).astype(jnp.bfloat16)
    shared = resident_matmul(shared_activation, shared_down_local, interpret=interpret)
    output = (routed * jnp.asarray(contract.routed_scaling_factor, jnp.bfloat16) + shared).astype(jnp.bfloat16)
    valid = valid & gate_ok & up_ok & down_ok & jnp.all(jnp.isfinite(output))
    return output, valid


class PrefillRoutes(NamedTuple):
    """Permutation metadata; ``valid`` MUST join the caller's serving health."""

    sorted_flat_ids: Any
    inverse_permutation: Any
    group_sizes: Any
    valid: Any


def group_prefill_routes(route_indices: Any, *, num_experts: int = 256) -> PrefillRoutes:
    """Group by expert, preserving original token/slot order inside each group.

    Invalid dynamic IDs or duplicate experts produce ``valid=False``; they are
    not silently admitted. No host callback or per-token dispatch is used.
    """
    if route_indices.ndim != 2 or min(route_indices.shape) <= 0:
        raise ValueError("routes must be nonempty [tokens,top_k]")
    if route_indices.dtype != jnp.int32:
        raise ValueError("routes must be int32")
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if route_indices.shape[1] > num_experts:
        raise ValueError("top_k exceeds expert count")
    flat = route_indices.reshape(-1)
    valid_ids = (flat >= 0) & (flat < num_experts)
    ordered_rows = jnp.sort(route_indices, axis=1)
    unique = jnp.all(ordered_rows[:, 1:] != ordered_rows[:, :-1])
    valid = jnp.all(valid_ids) & unique
    # The placeholder only keeps metadata construction bounded on invalid
    # input. The explicit health bit remains false and forbids serving.
    safe_flat = jnp.where(valid_ids, flat, 0)
    order = jnp.argsort(safe_flat, stable=True)
    inverse = jnp.zeros_like(order).at[order].set(jnp.arange(flat.size, dtype=jnp.int32))
    counts = jnp.bincount(safe_flat, length=num_experts).astype(jnp.int32)
    return PrefillRoutes(order, inverse, counts, valid)


def gather_prefill_route_rows(hidden: Any, routes: PrefillRoutes, *, top_k: int) -> Any:
    """Gather only local features; no hidden-state all-gather or dead capacity."""
    if hidden.ndim != 2 or top_k <= 0 or routes.sorted_flat_ids.shape != (hidden.shape[0] * top_k,):
        raise ValueError("hidden and route geometry disagree")
    return hidden[routes.sorted_flat_ids // top_k]


def restore_prefill_route_rows(sorted_values: Any, routes: PrefillRoutes, *, top_k: int) -> Any:
    """Restore [tokens,slot,features] before the original BF16 slot reduction."""
    if (
        sorted_values.ndim != 2
        or top_k <= 0
        or sorted_values.shape[0] != routes.inverse_permutation.size
        or sorted_values.shape[0] % top_k
    ):
        raise ValueError("sorted output and route geometry disagree")
    return sorted_values[routes.inverse_permutation].reshape(-1, top_k, sorted_values.shape[-1])


def moe_grouped_routes(
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
    config: RoutedProjectionConfig | None = None,
    expert_axis: str = "expert",
    feature_axis: str = "feature",
    interpret: bool = False,
    shared_bf16: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Route-grouped challenger for ``ws32_moe_pallas_from_routes_mapped``.

    ``shared_bf16 = (gate, up, down)`` supplies pre-decoded BF16 shared-expert
    tables (``glm_tpu.perf.bf16_resident``); the shared FP8 operands are then
    ignored (pass None) and the shared projections are plain dots at the same
    FP32-accumulate boundaries.

    Same inputs, ownership rule, arithmetic boundaries and output as the frozen
    body.  Structural differences only: all routed gate/up partials and the
    shared expert's partials cross the feature axis in ONE stacked FP32
    ``psum``; routed down projections run in one grouped kernel; no per-route
    ``lax.cond``.  ``config`` defaults to 512x512 tiles when the checkpoint
    block is 128x128 and to the frozen one-block tiles otherwise.
    """

    if contract.stage_size != 8:
        raise ValueError("WS32 grouped MoE requires stage_size=8")
    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 grouped MoE input must contain one live row")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 grouped MoE input must be bfloat16")
    if route_indices.shape != (1, contract.top_k) or route_indices.dtype != jnp.int32:
        raise ValueError("WS32 grouped MoE routes must contain one exact int32 row")
    if route_weights.shape != (1, contract.top_k) or route_weights.dtype != jnp.float32:
        raise ValueError("WS32 grouped MoE route weights must contain one FP32 row")
    local_hidden = hidden_local.shape[-1]
    local_experts = expert_gate_bits_local.shape[0]
    if local_experts != contract.local_experts:
        raise ValueError("WS32 grouped MoE local expert ownership drifted")
    expected_gate = (local_experts, contract.intermediate_size, local_hidden)
    expected_down = (local_experts, local_hidden, contract.intermediate_size)
    if expert_gate_bits_local.shape != expected_gate or (expert_up_bits_local.shape != expected_gate):
        raise ValueError("WS32 grouped MoE routed gate/up shapes drifted")
    if expert_down_bits_local.shape != expected_down:
        raise ValueError("WS32 grouped MoE routed down shape drifted")
    if shared_bf16 is None:
        if shared_gate_bits_local.shape != expected_gate[1:] or (
            shared_up_bits_local.shape != expected_gate[1:] or shared_down_bits_local.shape != expected_down[1:]
        ):
            raise ValueError("WS32 grouped MoE shared expert shapes drifted")
    else:
        if (
            len(shared_bf16) != 3
            or any(t.dtype != jnp.bfloat16 for t in shared_bf16)
            or shared_bf16[0].shape != expected_gate[1:]
            or (shared_bf16[1].shape != expected_gate[1:] or shared_bf16[2].shape != expected_down[1:])
        ):
            raise ValueError("WS32 grouped MoE BF16 shared tables drifted")
    if config is None:
        block = tuple(contract.fp8_block_shape)
        config = (
            RoutedProjectionConfig(block_shape=block)
            if block == (128, 128)
            else RoutedProjectionConfig.frozen_tiles(block)
        )
    elif tuple(config.block_shape) != tuple(contract.fp8_block_shape):
        raise ValueError("WS32 grouped MoE tile config block differs from the contract")

    top_k = contract.top_k
    expert_start = lax.axis_index(expert_axis).astype(jnp.int32) * local_experts
    routes = route_indices[0]
    owned = (routes >= expert_start) & (routes < expert_start + local_experts)
    local_ids = jnp.clip(routes - expert_start, jnp.int32(0), jnp.int32(local_experts - 1))
    always = jnp.ones((1,), jnp.bool_)
    zero_id = jnp.zeros((1,), jnp.int32)

    with jax.named_scope("moe_grouped/routed_gate_up"):
        routed_gate_up = fp8_routed_projection(
            jnp.broadcast_to(hidden_local, (top_k, local_hidden)),
            (
                (expert_gate_bits_local, expert_gate_scale_local),
                (expert_up_bits_local, expert_up_scale_local),
            ),
            local_ids,
            owned,
            config=config,
            result_dtype=jnp.float32,
            interpret=interpret,
        )
    with jax.named_scope("moe_grouped/shared_gate_up"):
        if shared_bf16 is None:
            shared_gate_up = fp8_routed_projection(
                hidden_local,
                (
                    (shared_gate_bits_local[None], shared_gate_scale_local[None]),
                    (shared_up_bits_local[None], shared_up_scale_local[None]),
                ),
                zero_id,
                always,
                config=config,
                result_dtype=jnp.float32,
                interpret=interpret,
            )
        else:
            dims = (((1,), (1,)), ((), ()))
            shared_gate_up = jnp.stack(
                (
                    lax.dot_general(hidden_local, shared_bf16[0], dims, preferred_element_type=jnp.float32),
                    lax.dot_general(hidden_local, shared_bf16[1], dims, preferred_element_type=jnp.float32),
                ),
                axis=1,
            )
    stacked = jnp.concatenate((routed_gate_up, shared_gate_up), axis=0)
    with jax.named_scope("moe_grouped/gate_up_feature_reduce"):
        gate_up = lax.psum(stacked, axis_name=feature_axis).astype(jnp.bfloat16)
    activated = (gate_up[:, 0] * jax.nn.sigmoid(gate_up[:, 0]) * gate_up[:, 1]).astype(jnp.bfloat16)
    with jax.named_scope("moe_grouped/routed_down"):
        down = fp8_routed_projection(
            activated[:top_k],
            ((expert_down_bits_local, expert_down_scale_local),),
            local_ids,
            owned,
            config=config,
            result_dtype=jnp.bfloat16,
            interpret=interpret,
        )[:, 0]
    weighted = (down * route_weights[0][:, None].astype(jnp.bfloat16)).astype(jnp.bfloat16)
    # Same [route, 1, hidden] BF16 reduction shape as the frozen route stack.
    local_routed = jnp.sum(weighted[:, None, :], axis=0, dtype=jnp.bfloat16)
    with jax.named_scope("moe_grouped/routed_expert_reduce"):
        routed = lax.psum(local_routed.astype(jnp.float32), axis_name=expert_axis).astype(jnp.bfloat16)
    with jax.named_scope("moe_grouped/shared_down"):
        if shared_bf16 is None:
            shared = fp8_routed_projection(
                activated[top_k:],
                ((shared_down_bits_local[None], shared_down_scale_local[None]),),
                zero_id,
                always,
                config=config,
                result_dtype=jnp.bfloat16,
                interpret=interpret,
            )[:, 0]
        else:
            shared = lax.dot_general(
                activated[top_k:],
                shared_bf16[2],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
    routed_scale = jnp.asarray(contract.routed_scaling_factor, dtype=jnp.bfloat16)
    return (routed * routed_scale + shared).astype(jnp.bfloat16)
