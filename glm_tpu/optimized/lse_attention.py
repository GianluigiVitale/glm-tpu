"""Owner-local partial attention and LSE merge used by retained prefill only.

This is a numerical boundary, not a bitwise replacement: owner-local
softmax maxima change BF16 probability rounding and partial outputs round once
before the FP32 LSE merge. Empty owners contribute zero, including an entirely
empty selection. Decode retains the frozen selected-KV exchange.
"""

from dataclasses import replace
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from ..greenfield.kernels.pallas.sparse_attention import (
    SparseMlaConfig, stage_local_sparse_mla_pallas,
)
from ..greenfield.kernels.reference.attention import (
    MlaNumericalContract, SparseAttentionResult, StageLocalKvLayout,
    gather_stage_local_selected_kv, selected_positions_for_owner, CanonicalSelectedPositions,
)
from ..greenfield.kernels.reference.dsa import SelectedPositions


def merge_attention_scatter(partial: SparseAttentionResult, *, expert_axis: str = "expert") -> SparseAttentionResult:
    """FP32 LSE weights and denominator, then reduce-scatter numerator by head."""
    live = jnp.isfinite(partial.logsumexp)
    # Gather only the tiny LSE table; output tensors never all-gather.
    heads = partial.logsumexp.shape[-1]
    metadata = jnp.concatenate((partial.logsumexp, partial.contract_valid.astype(jnp.float32)[:, None]), axis=-1)
    gathered = lax.all_gather(metadata, expert_axis, axis=0, tiled=False)
    all_lse = gathered[..., :heads]
    maximum = jnp.max(jnp.where(jnp.isfinite(all_lse), all_lse, -jnp.inf), axis=0)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    weight = jnp.where(live, jnp.exp(partial.logsumexp - maximum), 0.0)
    # Keep the latent payload 512-wide. Appending one denominator lane makes
    # TPU padding inflate the entire scatter; the gathered LSE already supplies it.
    weighted = partial.output.astype(jnp.float32) * weight[..., None]
    summed = lax.psum_scatter(weighted, expert_axis, scatter_dimension=1, tiled=True)
    local_heads = summed.shape[1]
    start = lax.axis_index(expert_axis) * local_heads
    local_lse = lax.dynamic_slice_in_dim(all_lse, start, local_heads, axis=2)
    local_max = lax.dynamic_slice_in_dim(maximum, start, local_heads, axis=1)
    denominator = jnp.sum(jnp.where(jnp.isfinite(local_lse), jnp.exp(local_lse-local_max[None]), 0.0), axis=0)
    output = summed / jnp.where(denominator > 0, denominator, 1.0)[..., None]
    lse = jnp.where(denominator > 0, local_max + jnp.log(jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny)), -jnp.inf)
    valid = jnp.all(gathered[..., heads] == 1.0, axis=0)
    return SparseAttentionResult(output.astype(partial.output.dtype), lse, valid)




def lse_attention_mapped(
    query_nope: Any, query_rope: Any, cache: Any, block_tables: Any,
    selected: SelectedPositions, context_lengths: Any, *,
    contract: MlaNumericalContract, layout: StageLocalKvLayout,
    config: SparseMlaConfig = SparseMlaConfig(), interpret: bool = False,
    expert_axis: str = "expert", gathered: bool = True, validate_finite: bool = False,
    owned_key_capacity: int | None = None,
) -> SparseAttentionResult:
    """Gather head-sharded queries, attend owned keys, return local head outputs."""
    if type(validate_finite) is not bool or (validate_finite and not gathered):
        raise ValueError("finite admission requires the gathered LSE path")
    if owned_key_capacity is not None:
        block = min(config.segment_block, contract.top_k)
        if (not gathered or type(owned_key_capacity) is not int
                or owned_key_capacity < block or owned_key_capacity % block):
            raise ValueError("owned key capacity requires gathered attention and whole segment blocks")
        if owned_key_capacity >= contract.top_k:
            owned_key_capacity = None
    owners = lax.axis_size(expert_axis)
    if layout.local_parallel_size != owners or contract.num_heads % owners:
        raise ValueError("LSE attention head/cache ownership disagrees with mesh")
    rows = query_nope.shape[0]
    if (query_nope.shape != (rows, contract.num_heads//owners, contract.kv_lora_rank)
            or query_rope.shape != (rows, contract.num_heads//owners, contract.qk_rope_head_dim)
            or query_nope.dtype != jnp.bfloat16 or query_rope.dtype != jnp.bfloat16
            or cache.dtype != jnp.bfloat16):
        raise ValueError("LSE attention requires BF16 queries/cache at the declared geometry")
    if selected.positions.shape != (rows, contract.top_k):
        raise ValueError("LSE selected positions must match query rows and contract top_k")
    with jax.named_scope("glm_perf_lse_attention"):
        packed = jnp.concatenate((query_nope, query_rope), axis=-1)
        queries = lax.all_gather(packed, expert_axis, axis=1, tiled=True)
        if gathered:
            def partial_from_segment(segment, local_contract):
                output, lse = gathered_partial_attention(
                    queries, segment.values, segment.valid_counts, contract=local_contract,
                    config=config, interpret=interpret,
                )
                valid = segment.contract_valid
                if validate_finite:
                    valid = valid & jnp.all(jnp.isfinite(segment.values), axis=(1, 2))
                    valid = valid & jnp.all(jnp.isfinite(queries), axis=(1, 2))
                return SparseAttentionResult(output, lse, valid)

            if owned_key_capacity is None:
                segment = gather_stage_local_selected_kv(
                    cache, block_tables, selected, context_lengths, layout=layout,
                    owner_index=lax.axis_index(expert_axis),
                )
                partial = partial_from_segment(segment, contract)
            else:
                from .function_bindings import bind_dependencies

                owner = lax.axis_index(expert_axis)
                owned = selected_positions_for_owner(selected, layout=layout, owner_index=owner)

                def attend_owned(width):
                    # Reuse every frozen gather check, but supply the already
                    # canonicalized owner's subset before gathering cache bytes.
                    subset = CanonicalSelectedPositions(
                        SelectedPositions(owned.selection.positions[:, :width], owned.selection.valid_counts),
                        owned.contract_valid,
                    )
                    gather = bind_dependencies(gather_stage_local_selected_kv,
                        selected_positions_for_owner=lambda *args, **kwargs: subset)
                    segment = gather(cache, block_tables, selected, context_lengths,
                                     layout=layout, owner_index=owner)
                    return partial_from_segment(segment, replace(contract, top_k=width))

                # Branches contain no collectives. Each owner can fall back
                # independently; every owner reaches the same merge afterwards.
                partial = lax.cond(jnp.all(owned.selection.valid_counts <= owned_key_capacity),
                    lambda _: attend_owned(owned_key_capacity), lambda _: attend_owned(contract.top_k), None)
        else:
            partial = stage_local_sparse_mla_pallas(
                queries[..., :contract.kv_lora_rank], queries[..., contract.kv_lora_rank:],
                cache, block_tables, selected, context_lengths, layout=layout,
                owner_index=lax.axis_index(expert_axis), contract=contract,
                config=config, interpret=interpret,
            )
        return merge_attention_scatter(partial, expert_axis=expert_axis)


def gathered_partial_attention(queries, cache, counts, *, contract, config, interpret=False, maximum_rows=32):
    """Contiguous local tiles and online softmax, returning a partial and LSE.

    Gather sanitizes padding before the PV dot. Empty owners return exact zero
    and -inf. Each row skips tiles beyond its owned count. The row grid also
    supports prefill tiles without a selected-KV collective.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, heads, _ = queries.shape
    latent, width = contract.kv_lora_rank, contract.packed_cache_width
    if (type(maximum_rows) is not int or maximum_rows not in (32,128)
            or not 1 <= rows <= maximum_rows or heads != contract.num_heads
            or cache.shape != (rows, contract.top_k, width)
            or counts.shape != (rows,) or counts.dtype != jnp.int32):
        raise ValueError("partial attention row/cache/count geometry drifted")
    block = min(config.segment_block, contract.top_k)
    if contract.top_k % block:
        raise ValueError("partial attention block must divide top_k")
    if queries.shape[-1] != latent + contract.qk_rope_head_dim:
        raise ValueError("partial attention packed query width drifted")
    queries = jnp.pad(queries, ((0, 0), (0, 0), (0, width-queries.shape[-1])))
    blocks = contract.top_k // block
    cache = cache.reshape(rows, blocks, block, width)

    def kernel(count, q, kv, out, lse, maximum, denominator, accumulator):
        row, tile = pl.program_id(0), pl.program_id(1)
        @pl.when(tile == 0)
        def init():
            maximum[...] = jnp.full(maximum.shape, -jnp.inf, jnp.float32)
            denominator[...] = jnp.zeros(denominator.shape, jnp.float32)
            accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
            out[0] = jnp.zeros(out.shape[1:], jnp.bfloat16)
            lse[0] = jnp.full(lse.shape[1:], -jnp.inf, jnp.float32)

        @pl.when(tile * block < count[row])
        def attend():
            scores = lax.dot_general(q[0], kv[0, 0], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32) * jnp.float32(contract.softmax_scale)
            live = jnp.arange(block)[None, :] + tile * block < count[row]
            scores = jnp.where(live, scores, -jnp.inf)
            m = jnp.maximum(maximum[...], jnp.max(scores, axis=1, keepdims=True))
            correction = jnp.exp(maximum[...] - m)
            probs = jnp.exp(scores-m)
            den = denominator[...] * correction + jnp.sum(probs, axis=1, keepdims=True)
            acc = accumulator[...] * correction + lax.dot_general(probs.astype(jnp.bfloat16), kv[0, 0, :, :latent], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            maximum[...], denominator[...], accumulator[...] = m, den, acc
            @pl.when(tile == (count[row]-1)//block)
            def finish():
                out[0] = (acc/den).astype(jnp.bfloat16)
                lse[0] = jnp.broadcast_to(m+jnp.log(den), (heads, 128))

    call = pl.pallas_call(kernel, grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(rows, blocks),
        in_specs=(pl.BlockSpec((1, heads, width), lambda r, b, c: (r, 0, 0)),
                  pl.BlockSpec((1, 1, block, width), lambda r, b, c: (r, b, 0, 0))),
        out_specs=(pl.BlockSpec((1, heads, latent), lambda r, b, c: (r, 0, 0)),
                   pl.BlockSpec((1, heads, 128), lambda r, b, c: (r, 0, 0))),
        scratch_shapes=(pltpu.VMEM((heads, 1), jnp.float32), pltpu.VMEM((heads, 1), jnp.float32), pltpu.VMEM((heads, latent), jnp.float32))),
        out_shape=(jax.ShapeDtypeStruct((rows, heads, latent), jnp.bfloat16), jax.ShapeDtypeStruct((rows, heads, 128), jnp.float32)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"), vmem_limit_bytes=config.vmem_limit_bytes),
        interpret=interpret, name="glm_perf_gathered_partial_attention")
    output, lse = call(counts, queries, cache)
    return output, lse[..., 0]
