"""Small verification blocks sharing the ordinary decoder's M8 expert tiles.

An expert can appear at most once per token. For at most eight proposed tokens,
one M8 tile therefore holds every occurrence of an owned expert. Gate/up share
the same kernel and K128 accumulation order as the ordinary D1 projection.
This is an experimental primitive, not trained-token or TPU admission.
"""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .fp8_routed_experts import RoutedProjectionConfig, _routed_scale_table, _scale_entry


class SmallExpertPlan(NamedTuple):
    expert_ids: Any
    route_slots: Any
    active_groups: Any
    valid: Any


def small_expert_plan(routes, expert_offset, *, num_experts, local_experts):
    """Pack at most eight rows of distinct per-token routes, on this owner."""
    if (routes.ndim != 2 or not 1 <= routes.shape[0] <= 8
            or routes.dtype != jnp.int32 or not 1 <= routes.shape[1] <= num_experts
            or type(num_experts) is not int or type(local_experts) is not int
            or local_experts <= 0 or num_experts % local_experts
            or expert_offset.shape != () or expert_offset.dtype != jnp.int32):
        raise ValueError('small expert plan requires <=8 int32 rows and contiguous ownership')
    flat = routes.reshape(-1)
    slots = flat.size
    capacity = min(slots, local_experts)
    ordered = jnp.sort(routes, axis=1)
    valid = jnp.all((flat >= 0) & (flat < num_experts))
    valid &= jnp.all(ordered[:, 1:] != ordered[:, :-1])
    valid &= (expert_offset >= 0) & (expert_offset + local_experts <= num_experts)
    valid &= expert_offset % local_experts == 0
    matches = flat[None, :] == (expert_offset + jnp.arange(local_experts))[:, None]
    counts = jnp.sum(matches, axis=1, dtype=jnp.int32)
    ids = jnp.nonzero(counts > 0, size=capacity, fill_value=0)[0].astype(jnp.int32)
    active = jnp.sum(counts > 0, dtype=jnp.int32)
    # Sentinel `slots` is a zero input row and a discarded output row. Only
    # real owned occurrences ever address a live result slot.
    indices = jnp.sort(jnp.where(matches[ids], jnp.arange(slots), slots), axis=1)
    if slots < 8:
        indices = jnp.pad(indices, ((0, 0), (0, 8-slots)), constant_values=slots)
    indices = indices[:, :8]
    indices = jnp.where(jnp.arange(capacity)[:, None] < active, indices, slots)
    valid &= jnp.all(counts <= 8)
    return SmallExpertPlan(ids, indices.astype(jnp.int32), active, valid)


def small_expert_projection(lhs, weights, plan, *,
                            config=RoutedProjectionConfig(output_tile=256, contraction_tile=256),
                            result_dtype=jnp.float32, interpret=False):
    """Return [route slots, tables, N], exactly zero on all unowned slots.

`lhs` contains flattened original token/route rows, not a prefill permutation.
The same plan can be reused for fused gate/up and the down projection. Invalid
plan health must join the caller's fleet health; no output is served from it.
"""
    if (lhs.ndim != 2 or lhs.dtype != jnp.bfloat16
            or not isinstance(weights, tuple) or not 1 <= len(weights) <= 2):
        raise ValueError('small expert projection requires BF16 rows and one/two table pairs')
    slots, k = lhs.shape
    if min(slots, k) <= 0 or weights[0][0].ndim != 3:
        raise ValueError('invalid small expert projection geometry')
    experts, n, wk = weights[0][0].shape
    bn, bk = config.block_shape
    tn, tk = config.output_tile, config.contraction_tile
    tables, capacity = len(weights), min(slots, experts)
    dtype = jnp.dtype(result_dtype)
    if (wk != k or min(experts, n) <= 0 or n % tn or k % tk
            or dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
            or not config.write_empty_slot
            or plan.expert_ids.shape != (capacity,) or plan.expert_ids.dtype != jnp.int32
            or plan.route_slots.shape != (capacity, 8) or plan.route_slots.dtype != jnp.int32
            or plan.active_groups.shape != () or plan.active_groups.dtype != jnp.int32
            or plan.valid.shape != () or plan.valid.dtype != jnp.bool_):
        raise ValueError('small expert plan/tile/result geometry differs')
    for bits, scale in weights:
        if (bits.shape != (experts, n, k) or bits.dtype != jnp.uint8
                or scale.shape != (experts, n//bn, k//bk) or scale.dtype != jnp.float32):
            raise ValueError('small expert weight/scale geometry differs')
    bpn, bpk = config.blocks_per_output_tile, config.blocks_per_contraction_tile
    k_tiles = k // tk
    valid = plan.valid & (plan.active_groups >= 0) & (plan.active_groups <= capacity)
    valid &= jnp.all((plan.expert_ids >= 0) & (plan.expert_ids < experts))
    valid &= jnp.all((plan.route_slots >= 0) & (plan.route_slots <= slots))
    active = jnp.where(valid, plan.active_groups, 0)
    metadata = jnp.concatenate((jnp.clip(plan.expert_ids, 0, experts-1), active[None]))
    packed = jnp.concatenate((lhs, jnp.zeros((1, k), lhs.dtype)), axis=0)[
        jnp.clip(plan.route_slots, 0, slots)]
    scales = tuple(_routed_scale_table(s, contraction_blocks=k//bk) for _, s in weights)

    def kernel(meta, x_ref, *refs):
        weight_refs, scale_refs = refs[:tables], refs[tables:2*tables]
        out_ref = refs[2*tables+1]
        accumulators = refs[2*tables+2:]
        group, ni, ki = pl.program_id(0), pl.program_id(1), pl.program_id(2)
        live = group < meta[capacity]

        @pl.when(ki == 0)
        def initialize():
            for acc in accumulators:
                acc[...] = jnp.zeros((8, tn), jnp.float32)

        @pl.when(live)
        def compute():
            x = x_ref[...]
            for j in range(bpk):
                block_row = ki * jnp.int32(bpk) + jnp.int32(j)
                for table in range(tables):
                    slab = scale_refs[table][...]
                    for i in range(bpn):
                        block_column = ni*jnp.int32(bpn) + jnp.int32(i)
                        bits = weight_refs[table][pl.ds(i*bn, bn), pl.ds(j*bk, bk)]
                        decoded = (lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.float32)
                                   * _scale_entry(slab, block_row, block_column)).astype(jnp.bfloat16)
                        update = lax.dot_general(x[:, j*bk:(j+1)*bk], decoded,
                            dimension_numbers=(((1,), (1,)), ((), ())),
                            preferred_element_type=jnp.float32)
                        accumulators[table][:, i*bn:(i+1)*bn] += update

        # Even an empty owner executes and writes one defined zero window.
        @pl.when(ki == k_tiles-1)
        def store():
            for table, acc in enumerate(accumulators):
                out_ref[table:table+1, :, :] = acc[...][None, ...].astype(dtype)

    def lhs_index(group, ni, ki, meta):
        return group, 0, ki

    def weight_index(group, ni, ki, meta):
        return meta[group], ni, ki

    def scale_index(group, ni, ki, meta):
        return meta[group], (ki*jnp.int32(bpk))//jnp.int32(8), 0

    def output_index(group, ni, ki, meta):
        return group, 0, 0, ni

    output_spec = pl.BlockSpec((None, tables, 8, tn), output_index)
    call = pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((capacity, tables, 8, n), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,
            in_specs=tuple([pl.BlockSpec((None, 8, tk), lhs_index)]
                + [pl.BlockSpec((None, tn, tk), weight_index)]*tables
                + [pl.BlockSpec((None, 8, 128), scale_index)]*tables + [output_spec]),
            out_specs=output_spec, grid=(jnp.maximum(active, 1), n//tn, k_tiles),
            scratch_shapes=tuple(pltpu.VMEM((8, tn), jnp.float32) for _ in range(tables))),
        input_output_aliases={2+2*tables: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=('arbitrary','parallel','arbitrary')),
        interpret=interpret,
        name=f'glm_perf_verify_expert_m8_s{slots}_t{tables}_tn{tn}_tk{tk}')
    packed_output = call(metadata, packed, *(b for b, _ in weights), *scales,
                         jnp.zeros((capacity, tables, 8, n), dtype))
    values = jnp.swapaxes(packed_output, 1, 2).reshape(capacity*8, tables, n)
    indices = jnp.where(valid, plan.route_slots, slots).reshape(-1)
    restored = jnp.zeros((slots+1, tables, n), dtype).at[indices].set(values)[:slots]
    return restored, valid
