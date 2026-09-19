"""Pool routed weight panels while retaining the decoder's per-row reductions.

The ordinary D1 path reduces routed and shared gate/up partials together. A
prefill suffix changes both this grouping and the collective shapes. Retaining
the original shapes is a numerical candidate, not an admission claim.
"""
import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.prefill_routes import (
    group_prefill_routes, gather_prefill_route_rows, restore_prefill_route_rows,
)
from ..greenfield.kernels.prefill_expert_panels import build_expert_panels
from ..greenfield.kernels.pallas.prefill_panel_fp8 import prefill_panel_fp8_matmul
from .bf16_resident import _dot_f32, dense_bf16
from .speculative_experts import small_expert_plan, small_expert_projection


def dense_rows_bf16(normalized, weights, *, expert_axis='expert', feature_axis='feature'):
    """Preserve one-row dense dots/collectives for the three initial layers."""
    return jnp.concatenate([dense_bf16(normalized[i:i+1], weights,
        expert_axis=expert_axis, feature_axis=feature_axis)
        for i in range(normalized.shape[0])], axis=0)


def moe_rows_bf16(hidden, route_ids, route_weights, weights, *, contract, interpret=False,
                  small_expert_tiles=False):
    """One expert-relative panel plan, ordinary per-token collective geometry."""
    rows, width = hidden.shape
    top_k = contract.top_k
    if (not 1 <= rows <= 8 or hidden.dtype != jnp.bfloat16
            or width*4 != contract.hidden_size or contract.stage_size != 8
            or route_ids.shape != (rows,top_k) or route_ids.dtype != jnp.int32
            or route_weights.shape != (rows,top_k) or route_weights.dtype != jnp.float32
            or contract.fp8_block_shape != (128,128) or type(small_expert_tiles) is not bool):
        raise ValueError('verifier MoE requires 1..8 feature4 BF16 rows and valid route geometry')
    offset = lax.axis_index('expert').astype(jnp.int32)*contract.local_experts
    if small_expert_tiles:
        plan = small_expert_plan(route_ids, offset, num_experts=contract.num_experts,
                                 local_experts=contract.local_experts)
        both, gate_ok = small_expert_projection(jnp.repeat(hidden, top_k, axis=0),
            ((weights.expert_gate_bits_local, weights.expert_gate_scale_local),
             (weights.expert_up_bits_local, weights.expert_up_scale_local)),
            plan, result_dtype=jnp.float32, interpret=interpret)
        gates, ups = both[:,0].reshape(rows,top_k,-1), both[:,1].reshape(rows,top_k,-1)
        up_ok = gate_ok
    else:
        plan = group_prefill_routes(route_ids, num_experts=contract.num_experts)
        sorted_hidden = gather_prefill_route_rows(hidden,plan,top_k=top_k)
        panels = build_expert_panels(plan.group_sizes,offset,
            rows=rows*top_k,local_groups=contract.local_experts)
        gate, gate_ok = prefill_panel_fp8_matmul(sorted_hidden,
            weights.expert_gate_bits_local,weights.expert_gate_scale_local,
            panels,result_dtype=jnp.float32,interpret=interpret)
        up, up_ok = prefill_panel_fp8_matmul(sorted_hidden,
            weights.expert_up_bits_local,weights.expert_up_scale_local,
            panels,result_dtype=jnp.float32,interpret=interpret)
        gates = restore_prefill_route_rows(gate,plan,top_k=top_k)
        ups = restore_prefill_route_rows(up,plan,top_k=top_k)
    valid = plan.valid & jnp.all(jnp.isfinite(route_weights) & (route_weights >= 0))
    shared_gates = jnp.concatenate([_dot_f32(hidden[i:i+1],weights.shared_gate_local)
        for i in range(rows)])
    shared_ups = jnp.concatenate([_dot_f32(hidden[i:i+1],weights.shared_up_local)
        for i in range(rows)])

    def activate(values):
        g,u,sg,su = values
        partials = jnp.concatenate((jnp.stack((g,u),axis=1),jnp.stack((sg,su))[None]),axis=0)
        with jax.named_scope('glm_perf_verify_moe/canonical_feature_reduce'):
            reduced = lax.psum(partials,'feature').astype(jnp.bfloat16)
        return (reduced[:,0]*jax.nn.sigmoid(reduced[:,0])*reduced[:,1]).astype(jnp.bfloat16)

    activated = lax.map(activate,(gates,ups,shared_gates,shared_ups))
    if small_expert_tiles:
        down, down_ok = small_expert_projection(activated[:,:top_k].reshape(rows*top_k,-1),
            ((weights.expert_down_bits_local, weights.expert_down_scale_local),),
            plan, result_dtype=jnp.bfloat16, interpret=interpret)
        downs = down[:,0].reshape(rows,top_k,-1)
    else:
        sorted_activated = activated[:,:top_k].reshape(rows*top_k,-1)[plan.sorted_flat_ids]
        down, down_ok = prefill_panel_fp8_matmul(sorted_activated,
            weights.expert_down_bits_local,weights.expert_down_scale_local,
            panels,result_dtype=jnp.bfloat16,interpret=interpret)
        downs = restore_prefill_route_rows(down,plan,top_k=top_k)

    def combine(values):
        d,rw,sa = values
        weighted = (d*rw[:,None].astype(jnp.bfloat16)).astype(jnp.bfloat16)
        local = jnp.sum(weighted[:,None,:],axis=0,dtype=jnp.bfloat16)
        with jax.named_scope('glm_perf_verify_moe/canonical_expert_reduce'):
            routed = lax.psum(local.astype(jnp.float32),'expert').astype(jnp.bfloat16)
        shared = _dot_f32(sa[None],weights.shared_down_local).astype(jnp.bfloat16)
        return (routed*jnp.asarray(contract.routed_scaling_factor,jnp.bfloat16)+shared)[0].astype(jnp.bfloat16)

    output = lax.map(combine,(downs,route_weights,activated[:,top_k]))
    return output,valid & gate_ok & up_ok & down_ok & jnp.all(jnp.isfinite(output))
