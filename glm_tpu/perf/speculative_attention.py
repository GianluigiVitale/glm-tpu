"""Causal multi-row attention using one shared cache proposal.

Prepare and write each proposed row once before batched reads. Each query keeps
its own causal length, so a later proposal is never a visible key to an earlier
query. Changes to matrix and collective shapes require numerical qualification.
The preupdated-cache options are internal: callers must use this write path.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_batching import sequential_vmap

from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.reference.dsa import dsa_index_keys_from_projection
from ..greenfield.kernels.reference.rotary import apply_rotary_fp32_final_round
from ..greenfield.kernels.stage_local import _require_decode_metadata
from ..greenfield.kernels.ws32_layer import Ws32AttentionLayerResult
from .bf16_resident import (
    _dot_f32, prepare_attention_bf16, dsa_bf16, index_share_attention_bf16,
)


def attention_rows_bf16(residual, normalized, kv, index, selected, counts, scores,
                        position, block_tables, context_lengths, layer, rope, *,
                        config, interpret=False, rowwise_dsa=False):
    rows = residual.shape[0]
    if not 1 <= rows <= 8 or normalized.shape != residual.shape or type(rowwise_dsa) is not bool:
        raise ValueError('verification attention requires 1..8 matching hidden rows')
    positions = position[0] + jnp.arange(rows, dtype=jnp.int32)
    lengths = context_lengths[0] + jnp.arange(rows, dtype=jnp.int32)
    # Preserve one-row preparation/reduction shapes for cache contents. The
    # expensive selected-KV exchange and query work below are still batched.
    prepared = lax.map(lambda xs: prepare_attention_bf16(xs[0][None], layer.qkv_a,
        normalized=xs[1][None], feature_axis='feature'), (residual, normalized))
    rotary = jnp.take(rope, positions, axis=0, mode='clip')
    owner = lax.axis_index('expert')
    ac, dc, layout = config.attention_contract, config.dsa_contract, config.cache_layout

    def cache_row(p, row):
        half = ac.qk_rope_head_dim // 2
        cos, sin = row[:half][None,None,:], row[half:][None,None,:]
        latent = p.current_kv[..., :ac.kv_lora_rank]
        unrotated = p.current_kv[..., ac.kv_lora_rank:ac.kv_lora_rank+ac.qk_rope_head_dim][:,None,:]
        rotated = apply_rotary_fp32_final_round(unrotated,cos,sin,interleaved=True)[:,0,:]
        padding = ac.packed_cache_width - ac.kv_lora_rank - ac.qk_rope_head_dim
        return jnp.concatenate((latent,rotated,jnp.zeros((1,padding),jnp.bfloat16)),axis=-1)[0].astype(jnp.bfloat16)

    kv_rows = jax.vmap(cache_row)(prepared, rotary)
    if layer.dsa is not None:
        def key_row(p, pos):
            projected = lax.psum(_dot_f32(p.normalized_local,layer.dsa.wk_local),'feature')
            return dsa_index_keys_from_projection(projected,layer.dsa.key_norm_weight,
                layer.dsa.key_norm_bias,pos[None],contract=dc)[0].astype(index.dtype)
        index_rows = lax.map(lambda xs: key_row(*xs),(prepared,positions))
    else:
        index_rows = jnp.zeros((rows,index.shape[-1]),index.dtype)

    def write(caches, values):
        pos,length,k,key = values
        page,row,target,_,valid = _require_decode_metadata(pos[None],block_tables,length[None],
            owner,layout=layout,physical_page_count=caches[0].shape[0])
        def update(c):
            new_kv = c[0].at[page,row].set(k)
            new_index = c[1].at[page,row].set(key) if layer.dsa is not None else c[1]
            return new_kv,new_index
        return lax.cond(valid[0] & (owner == target),update,lambda c:c,caches),valid[0]

    (kv,index),write_ok = lax.scan(write,(kv,index),(positions,lengths,kv_rows,index_rows))

    def select(p, pos, length):
        return dsa_bf16(p,index,pos[None],block_tables,length[None],layer.dsa,
            expert_axis='expert',feature_axis='feature',contract=dc,cache_layout=layout,
            two_stage=True,cache_preupdated=True)
    if rowwise_dsa:
        # vmap of the scalar fallback condition executes both branches. Keep
        # each row's exact cut decision and FP32 score arithmetic independent;
        # the selected-KV attention exchange below remains batched.
        select = sequential_vmap(select)

    def attend(x,p,pos,length,ids,count,score,rot):
        valid = jnp.ones((1,),jnp.bool_)
        if layer.dsa is not None:
            result = select(p,pos,length)
            ids,count,score,valid = (result.selected_positions[0],result.selected_valid_counts[0],
                result.selected_scores[0],result.contract_valid)
        result = index_share_attention_bf16(x[None],p,kv,ids[None],count[None],pos[None],
            block_tables,length[None],layer.attention,expert_axis='expert',contract=ac,
            cache_layout=layout,main_rope_table_row=rot,
            sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
            sparse_attention_interpret=interpret,lse_attention=False,cache_preupdated=True,
            rowwise_head_projections=True)
        return result.output_local[0],ids,count,score,(valid & result.contract_valid)[0]

    out,ids,count,score,valid = jax.vmap(attend)(residual,prepared,positions,lengths,
        selected,counts,scores,rotary)
    return Ws32AttentionLayerResult(out,kv,index,ids,count,score,valid & write_ok)
