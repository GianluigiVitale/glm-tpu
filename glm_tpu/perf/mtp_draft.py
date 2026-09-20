"""Single native MTP transformer proposal, with separate draft cache ownership.

Full-index rows bootstrap/refresh from target hidden states; a recurrent single
row reuses the previous shortlist and skips the entire DSA key/indexer path.
Returns an uncommitted proposal. The existing prefix committer can advance a
private draft state; callers must restore its committed root before refreshing
with accepted target hidden rows. No host acceptance loop is implemented here.
"""
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped,ws32_router_from_shards_mapped
from ..greenfield.kernels.ws32_io import ws32_embedding_mapped,ws32_logits_mapped,ws32_greedy_sample_mapped
from ..greenfield.runtime import ws32_decoder as decoder
from .bf16_resident import Bf16DecoderWeights,bf16_weight_specs
from .mtp_projection import MtpProjectionWeights,project_mapped
from .speculative_attention import attention_rows_bf16
from .speculative_moe import moe_rows_bf16
from .speculative_verify import VerificationProposal,proposal_specs


class MtpWeights(NamedTuple):
    # Exactly one full-index sparse transformer, native shared_head norm;
    # embedding and vocabulary table reference the target's resident arrays.
    decoder: Bf16DecoderWeights
    projection: MtpProjectionWeights


def mtp_config(target):
    """Retain target attention/MoE/cache geometry with one full-index MTP layer."""
    return replace(target,geometry=replace(target.geometry,num_layers=1,
        first_dense_layers=0,mlp_layer_types=('sparse',),indexer_types=('full',)))


def mtp_weight_specs(config):
    return MtpWeights(bf16_weight_specs(config),
        MtpProjectionWeights(P('feature'),P('feature'),P('feature',None)))


def draft_mapped(shifted_tokens,previous_hidden,state,weights,rope,*,config,
                 index_share=False,sparse_attention_interpret=False,linear_interpret=False):
    """Compute 1..8 full-index rows or one recurrent IndexShare row.

    At position t input is embedding(x[t+1]) plus normalized hidden h[t]; the
    prediction is x[t+2]. The caller supplies target hidden rows for refresh,
    and prior MTP hidden for recurrence. Input IDs and hidden rows must belong
    to the same authenticated request/prefix; array shapes cannot prove this.
    """
    decoder._validate_local_state(state,config)
    base=weights.decoder
    if (config.geometry.num_layers!=1 or config.geometry.mlp_layer_types!=('sparse',)
            or config.geometry.indexer_types!=('full',) or config.exact_dsa
            or config.strategy_nd_dense or not config.host_main_rope_table
            or len(base.layers)!=1 or base.layers[0].dsa is None
            or base.layers[0].dense is not None or base.layers[0].moe is None
            or shifted_tokens.ndim!=1 or not 1<=shifted_tokens.size<=8
            or shifted_tokens.dtype!=jnp.int32 or type(index_share) is not bool
            or (index_share and shifted_tokens.size!=1)
            or rope.shape!=config.main_rope_table_shape or rope.dtype!=jnp.bfloat16):
        raise ValueError('native MTP requires one full-index sparse layer and bounded int32 rows')
    rows=shifted_tokens.size
    embedded=jax.vmap(lambda t:ws32_embedding_mapped(t[None],base.embedding_local,
        vocab_size=config.geometry.vocab_size))(shifted_tokens)
    positions=state.position[0]+jnp.arange(rows,dtype=jnp.int32)
    projected=project_mapped(embedded.residual_local[:,0],previous_hidden,positions,
        weights.projection,hidden_size=config.geometry.hidden_size,epsilon=config.rms_norm_epsilon)
    span_ok=((state.position[0]>=0)&(state.position[0]<=config.context_capacity-rows)
             &(state.context_lengths[0]==state.position[0]+1))
    if index_share:
        span_ok &= (state.position[0]>0)&(state.selected_valid_counts[0]>0)
    health=embedded.contract_valid[:,0]&projected.contract_valid&state.contract_valid[0]&span_ok
    norm_base=partial(ws32_fused_add_rms_norm_mapped,
        global_hidden_size=config.geometry.hidden_size,epsilon=config.rms_norm_epsilon)
    def norm(h,c,w):
        values=[norm_base(h[i:i+1],c[i:i+1],w) for i in range(rows)]
        return tuple(jnp.concatenate([v[k] for v in values],axis=0) for k in range(2))
    layer=base.layers[0]
    normalized,residual=norm(projected.hidden_local,jnp.zeros_like(projected.hidden_local),
                             layer.qkv_a.input_norm_weight_local)
    selected=jnp.broadcast_to(state.selected_positions,(rows,config.geometry.dsa_top_k))
    counts=jnp.broadcast_to(state.selected_valid_counts,(rows,))
    scores=jnp.broadcast_to(state.selected_scores,selected.shape)
    # Removing the DSA component skips key production, index writes and top-k,
    # while attention still writes this recurrent row to its own MLA KV cache.
    attention_layer=layer._replace(dsa=None) if index_share else layer
    attended=attention_rows_bf16(residual,normalized,state.kv_cache_local[0],
        state.index_cache_local[0],selected,counts,scores,state.position,
        state.block_tables,state.context_lengths,attention_layer,rope,config=config,
        interpret=sparse_attention_interpret,rowwise_dsa=True)
    normalized,carried=norm(attended.output_local,residual,layer.post_attention_norm_weight_local)
    moe=layer.moe
    routes,route_weights=jax.vmap(lambda x:ws32_router_from_shards_mapped(
        x[None],moe.router_weight_local,moe.correction_bias_local,
        top_k=config.moe_contract.top_k))(normalized)
    hidden,mlp_ok=moe_rows_bf16(normalized,routes[:,0],route_weights[:,0],moe,
        contract=config.moe_contract,interpret=linear_interpret,small_expert_tiles=True)
    normalized,final_residual=norm(hidden,carried,base.final_norm_weight_local)
    logits=jax.vmap(lambda x:ws32_logits_mapped(x[None],base.lm_head_local,
        vocab_size=config.geometry.vocab_size))(normalized)
    sampled=jax.vmap(lambda x:ws32_greedy_sample_mapped(x,
        vocab_size=config.geometry.vocab_size))(logits)
    health &= (attended.contract_valid&mlp_ok&sampled.contract_valid[:,0]
               &jnp.all(jnp.isfinite(normalized),axis=1)&jnp.all(jnp.isfinite(final_residual),axis=1))
    health=lax.pmin(health.astype(jnp.int32),('expert','feature'))!=0
    return VerificationProposal(attended.cache_local[None],attended.index_cache_local[None],
        attended.selected_positions,attended.selected_valid_counts,attended.selected_scores,
        sampled.token_id[:,0],final_residual,normalized,health)


def build_mtp_draft(mesh,config,**options):
    if tuple(mesh.axis_names)!=('expert','feature') or tuple(mesh.devices.shape)!=(8,4):
        raise ValueError('native MTP requires expert8 by feature4 mesh')
    return jax.jit(jax.shard_map(partial(draft_mapped,config=config,**options),mesh=mesh,
        in_specs=(P(),P(None,'feature'),decoder.ws32_decoder_state_specs(),mtp_weight_specs(config),P()),
        out_specs=proposal_specs(),check_vma=False))
