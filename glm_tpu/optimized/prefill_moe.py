"""Prefill MoE of the production engine: routed FP8 expert panels, resident BF16 shared expert.

The body of ``greenfield/kernels/ws32_prefill_moe.py`` with the shared-expert projections calling
``prefill_bf16.resident_matmul_f32``/``resident_matmul`` explicitly (S2d fold of the former
function rebinding); the routed experts keep the frozen FP8 panel kernel. The frozen module stays
untouched as the numerical oracle of the tests.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas.fp8_matmul import Fp8BlockMatmulConfig
from ..greenfield.kernels.pallas.prefill_grouped_fp8 import prefill_grouped_fp8_matmul
from ..greenfield.kernels.prefill_routes import (
    gather_prefill_route_rows,
    group_prefill_routes,
    restore_prefill_route_rows,
)
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from .prefill_bf16 import resident_matmul, resident_matmul_f32


def ws32_prefill_moe_from_routes_mapped(
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
    row_tile: int = 8,
    interpret: bool = False,
    capture_boundaries: bool = False,
    fp32_route_sum: bool = False,
    expert_panels: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, dict[str, Any]]:
    """Return (local hidden shard, local health); caller must gate on all chips.

    Requires the validated expert8/feature4 mesh and replicated routing inputs.
    A false health bit forbids serving; it must not be discarded by a caller.
    No router or full decoder is replaced by adding this building block.
    capture_boundaries is diagnostic-only and may change compiled rounding;
    captured results do not certify the uninstrumented execution.
    fp32_route_sum is a separate numerical candidate: sum BF16 weighted route
    values in FP32 without an intermediate BF16 round before expert reduction.
    expert_panels is a separate default-off M32/N256 kernel candidate. It packs
    activations, not weights, and reuses one expert-relative plan for all three
    routed projections. No TPU performance or arithmetic admission is implied.
    """
    if contract.stage_size != 8:
        raise ValueError("WS32 prefill MoE requires stage_size=8")
    if type(expert_panels) is not bool:
        raise ValueError("expert panels must be an explicit static boolean")
    if (
        hidden_local.ndim != 2
        or hidden_local.shape[0] <= 0
        or hidden_local.dtype != jnp.bfloat16
        or hidden_local.shape[1] * 4 != contract.hidden_size
    ):
        raise ValueError(
            "WS32 prefill MoE requires BF16 live rows and feature4 hidden shards"
        )
    rows, local_hidden = hidden_local.shape
    expected_routes = (rows, contract.top_k)
    if route_indices.shape != expected_routes or route_weights.shape != expected_routes:
        raise ValueError("prefill routing geometry differs from contract")
    if route_weights.dtype != jnp.float32:
        raise ValueError("prefill route weights must be FP32")
    gate_shape = (contract.local_experts, contract.intermediate_size, local_hidden)
    if (
        expert_gate_bits_local.shape != gate_shape
        or expert_up_bits_local.shape != gate_shape
    ):
        raise ValueError("prefill expert gate/up geometry differs from contract")
    if expert_down_bits_local.shape != (
        contract.local_experts,
        local_hidden,
        contract.intermediate_size,
    ):
        raise ValueError("prefill expert down geometry differs from contract")
    if (
        shared_gate_bits_local.shape != gate_shape[1:]
        or shared_up_bits_local.shape != gate_shape[1:]
        or shared_down_bits_local.shape != (local_hidden, contract.intermediate_size)
    ):
        raise ValueError("prefill shared expert geometry differs from contract")

    routes = group_prefill_routes(route_indices, num_experts=contract.num_experts)
    valid = routes.valid & jnp.all(jnp.isfinite(route_weights) & (route_weights >= 0))
    sorted_hidden = gather_prefill_route_rows(
        hidden_local, routes, top_k=contract.top_k
    )
    offset = lax.axis_index("expert").astype(jnp.int32) * contract.local_experts
    panels = None
    if expert_panels:
        from ..greenfield.kernels.prefill_expert_panels import build_expert_panels

        if contract.fp8_block_shape != (128, 128):
            raise ValueError("expert panels require checkpoint scale blocks128x128")
        panels = build_expert_panels(
            routes.group_sizes,
            offset,
            rows=sorted_hidden.shape[0],
            local_groups=contract.local_experts,
        )

    def project(x, w, s, dtype):
        if panels is not None:
            from ..greenfield.kernels.pallas.prefill_panel_fp8 import prefill_panel_fp8_matmul

            return prefill_panel_fp8_matmul(
                x, w, s, panels, result_dtype=dtype, interpret=interpret
            )
        return prefill_grouped_fp8_matmul(
            x,
            w,
            s,
            routes.group_sizes,
            offset,
            block_shape=contract.fp8_block_shape,
            row_tile=row_tile,
            result_dtype=dtype,
            interpret=interpret,
        )

    gate, gate_ok = project(
        sorted_hidden, expert_gate_bits_local, expert_gate_scale_local, jnp.float32
    )
    up, up_ok = project(
        sorted_hidden, expert_up_bits_local, expert_up_scale_local, jnp.float32
    )
    with jax.named_scope("greenfield_ws32_prefill_moe/feature_reduce"):
        gate_up = lax.psum(jnp.stack((gate, up)), axis_name="feature").astype(
            jnp.bfloat16
        )
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(
        jnp.bfloat16
    )
    down, down_ok = project(
        activated, expert_down_bits_local, expert_down_scale_local, jnp.bfloat16
    )
    sorted_weights = route_weights.reshape(-1)[routes.sorted_flat_ids].astype(
        jnp.bfloat16
    )
    weighted = (down * sorted_weights[:, None]).astype(jnp.bfloat16)
    route_outputs = restore_prefill_route_rows(weighted, routes, top_k=contract.top_k)
    # Preserve the original [route,live rows,hidden] expression and reduction
    # axis. Expert execution order must not become route accumulation order.
    if fp32_route_sum:
        with jax.named_scope("greenfield_ws32_prefill_moe/fp32_route_sum"):
            local_routed = jnp.sum(
                jnp.swapaxes(route_outputs, 0, 1).astype(jnp.float32),
                axis=0,
                dtype=jnp.float32,
            )
    else:
        local_routed = jnp.sum(
            jnp.swapaxes(route_outputs, 0, 1), axis=0, dtype=jnp.bfloat16
        )
    with jax.named_scope("greenfield_ws32_prefill_moe/expert_reduce"):
        local_sum_operand = local_routed.astype(jnp.float32)
        routed = lax.psum(local_sum_operand, axis_name="expert").astype(jnp.bfloat16)

    config = Fp8BlockMatmulConfig(
        block_shape=contract.fp8_block_shape,
        output_tile=contract.fp8_block_shape[0],
        contraction_tile=contract.fp8_block_shape[1],
    )
    shared_gate = resident_matmul_f32(
        hidden_local,
        shared_gate_bits_local,
        shared_gate_scale_local,
        config=config,
        interpret=interpret,
    )
    shared_up = resident_matmul_f32(
        hidden_local,
        shared_up_bits_local,
        shared_up_scale_local,
        config=config,
        interpret=interpret,
    )
    with jax.named_scope("greenfield_ws32_prefill_moe/shared_feature_reduce"):
        shared_gate_up = lax.psum(
            jnp.stack((shared_gate, shared_up)), axis_name="feature"
        ).astype(jnp.bfloat16)
    shared_activation = (
        shared_gate_up[0] * jax.nn.sigmoid(shared_gate_up[0]) * shared_gate_up[1]
    ).astype(jnp.bfloat16)
    shared = resident_matmul(
        shared_activation,
        shared_down_bits_local,
        shared_down_scale_local,
        config=config,
        interpret=interpret,
    )
    output = (
        routed * jnp.asarray(contract.routed_scaling_factor, jnp.bfloat16) + shared
    ).astype(jnp.bfloat16)
    valid = valid & gate_ok & up_ok & down_ok & jnp.all(jnp.isfinite(output))
    if capture_boundaries:

        def restore(value):
            return restore_prefill_route_rows(value, routes, top_k=contract.top_k)

        return (
            output,
            valid,
            {
                "routed_partial": jnp.stack((restore(gate), restore(up)), axis=2),
                "routed_reduced": jnp.stack(
                    (restore(gate_up[0]), restore(gate_up[1])), axis=2
                ),
                "weighted_routes": route_outputs,
                "local_sum_operand": local_sum_operand,
                "routed": routed,
                "shared_partial": jnp.stack((shared_gate, shared_up), axis=1),
                "shared_reduced": jnp.swapaxes(shared_gate_up, 0, 1),
                "shared": shared,
            },
        )
    return output, valid
