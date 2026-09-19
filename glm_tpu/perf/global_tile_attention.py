"""Experimental D5 with global tile maxima and one final output scatter.

Preserves the frozen selected-slot tiles and BF16 probability rounding relative
to each tile's running global maximum. Owner-local FP32 sums still reassociate
the denominator and PV contraction, so this is a numerical boundary, not a
bitwise replacement. It avoids the selected-KV payload exchange at the cost of
one tiny maximum collective per tile and more local zero-padded dot work.
"""
import jax
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.pallas.sparse_attention import SparseMlaConfig, _FINITE_MASK_VALUE
from ..greenfield.kernels.reference.attention import (
    SparseAttentionResult, gather_stage_local_selected_kv_aligned,
)


def global_tile_attention_mapped(query_nope, query_rope, cache, block_tables,
        selected, context_lengths, *, contract, layout,
        config=SparseMlaConfig(segment_block=512), interpret=False,
        expert_axis='expert', validate_finite=False):
    owners = lax.axis_size(expert_axis)
    owner = lax.axis_index(expert_axis)
    rows = query_nope.shape[0]
    heads, latent, width = contract.num_heads, contract.kv_lora_rank, contract.packed_cache_width
    if (not 1 <= rows <= 32 or owners != layout.local_parallel_size or heads % owners
            or query_nope.shape != (rows, heads//owners, latent)
            or query_rope.shape != (rows, heads//owners, contract.qk_rope_head_dim)
            or query_nope.dtype != jnp.bfloat16 or query_rope.dtype != jnp.bfloat16
            or cache.dtype != jnp.bfloat16 or selected.positions.shape != (rows, contract.top_k)
            or type(interpret) is not bool or type(validate_finite) is not bool):
        raise ValueError('global tile attention geometry/options drifted')
    block = min(config.segment_block, contract.top_k)
    if contract.top_k % block or width < latent+contract.qk_rope_head_dim:
        raise ValueError('global attention requires whole selected tiles and padded query width')
    segment = gather_stage_local_selected_kv_aligned(cache, block_tables, selected,
        context_lengths, layout=layout, owner_index=owner)
    with jax.named_scope('glm_perf_global_tile_attention'):
        query = lax.all_gather(jnp.concatenate((query_nope,query_rope),axis=-1),
                               expert_axis,axis=1,tiled=True)
        query = jnp.pad(query,((0,0),(0,0),(0,width-query.shape[-1])))
        slots = jnp.arange(contract.top_k)[None,:]
        owned = ((slots < segment.valid_counts[:,None]) &
                 ((segment.positions % layout.logical_page_size)//layout.local_rows_per_page == owner))
        maximum = jnp.full((rows,heads,1),_FINITE_MASK_VALUE,jnp.float32)
        denominator = jnp.zeros_like(maximum)
        accumulator = jnp.zeros((rows,heads,latent),jnp.float32)
        for first in range(0,contract.top_k,block):
            kv = segment.values[:,first:first+block]
            live = owned[:,None,first:first+block]
            q_operand = query.astype(jnp.float32) if interpret else query
            kv_operand = kv.astype(jnp.float32) if interpret else kv
            scores = lax.dot_general(q_operand,kv_operand,
                (((2,),(2,)),((0,),(0,))),precision=lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32) * jnp.float32(contract.softmax_scale)
            scores = jnp.where(live,scores,jnp.float32(_FINITE_MASK_VALUE))
            block_max = lax.pmax(jnp.max(scores,axis=-1,keepdims=True),expert_axis)
            next_max = jnp.maximum(maximum,block_max)
            correction = jnp.exp(maximum-next_max)
            probabilities = jnp.where(live,jnp.exp(scores-next_max),0)
            denominator = denominator*correction + jnp.sum(probabilities,axis=-1,keepdims=True)
            rounded = probabilities.astype(jnp.bfloat16)
            p_operand = rounded.astype(jnp.float32) if interpret else rounded
            partial = lax.dot_general(p_operand,kv_operand[:,:,:latent],
                (((2,),(1,)),((0,),(0,))),precision=lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32)
            accumulator = accumulator*correction + partial
            maximum = next_max
        valid = segment.contract_valid
        if validate_finite:
            valid &= jnp.all(jnp.isfinite(segment.values),axis=(1,2)) & jnp.all(jnp.isfinite(query),axis=(1,2))
        metadata = jnp.concatenate((denominator[:,:,0],valid[:,None].astype(jnp.float32)),axis=1)
        metadata = lax.all_gather(metadata,expert_axis,axis=0,tiled=False)
        local_heads = heads//owners
        start = owner*local_heads
        total = jnp.sum(metadata[:,:,:heads],axis=0)
        total = lax.dynamic_slice_in_dim(total,start,local_heads,axis=1)
        local_max = lax.dynamic_slice_in_dim(maximum[:,:,0],start,local_heads,axis=1)
        total += jnp.exp(jnp.float32(_FINITE_MASK_VALUE)-local_max)
        output = lax.psum_scatter(accumulator,expert_axis,scatter_dimension=1,tiled=True)
        output = jnp.where(local_max[:,:,None] > _FINITE_MASK_VALUE,
                           output/total[:,:,None],0).astype(jnp.bfloat16)
        lse = jnp.where(segment.valid_counts[:,None]>0,local_max+jnp.log(total),-jnp.inf)
        return SparseAttentionResult(output,lse,jnp.all(metadata[:,:,-1]==1,axis=0))
