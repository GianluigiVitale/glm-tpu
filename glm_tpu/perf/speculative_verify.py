"""Layer-major greedy verification candidate; no serving admission.

Default attention scans causal rows inside each layer. Optional batched attention
prewrites a shared cache proposal and applies each query's causal mask. The MLP
pools proposed rows, reusing expert weight panels; an option preserves ordinary
per-row MoE reductions. This is not a scan of complete decoder steps. Different
matrix/collective shapes require CPU and trained TPU numerical comparison.
"""
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped, ws32_router_from_shards_mapped
from ..greenfield.kernels.ws32_io import ws32_embedding_mapped, ws32_logits_mapped, ws32_greedy_sample_mapped
from ..greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
from ..greenfield.runtime import ws32_decoder as decoder
from .bf16_resident import attention_layer_bf16, bf16_weight_specs, dense_bf16
from .function_bindings import bind_dependencies
from .prefill_bf16 import resident_matmul, resident_matmul_f32
from .speculative_moe import dense_rows_bf16, moe_rows_bf16
from .speculative_attention import attention_rows_bf16


class VerificationProposal(NamedTuple):
    kv_cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    predictions: Any
    final_residual_local: Any
    normalized_hidden_local: Any
    contract_valid: Any


def proposal_specs():
    state = decoder.ws32_decoder_state_specs()
    return VerificationProposal(state.kv_cache_local, state.index_cache_local,
        P(), P(), P(), P(), P(None, 'feature'), P(None, 'feature'), P())


_pooled_moe = bind_dependencies(ws32_prefill_moe_from_routes_mapped,
    fp8_block_matmul_f32=resident_matmul_f32, fp8_block_matmul=resident_matmul)


def verify_mapped(tokens, state, weights, rope, *, config,
                  sparse_attention_interpret=False, linear_interpret=False,
                  expert_panels=True, canonical_mlp=False, batched_attention=False):
    """Propose every input row, returning no committed decoder state.

    Input is [pending target token, draft token 0, ...]. Prediction i is the
    target's next-token choice after input i. The caller compares prediction i
    to draft i and commits only the prefix ending at correction/bonus prediction.
    """
    decoder._validate_local_state(state, config)
    if (tokens.ndim != 1 or not 1 <= tokens.size <= 8 or tokens.dtype != jnp.int32
            or config.exact_dsa or config.strategy_nd_dense or not config.host_main_rope_table
            or rope.shape != config.main_rope_table_shape or rope.dtype != jnp.bfloat16
            or len(weights.layers) != config.geometry.num_layers
            or type(expert_panels) is not bool or type(canonical_mlp) is not bool
            or type(batched_attention) is not bool
            or (canonical_mlp and not expert_panels)):
        raise ValueError('verification requires 1..8 int32 rows and resident raw host-RoPE weights')
    rows = tokens.size
    embedded = jax.vmap(lambda t: ws32_embedding_mapped(t[None], weights.embedding_local,
        vocab_size=config.geometry.vocab_size))(tokens)
    hidden, carried = embedded.residual_local[:, 0], jnp.zeros_like(embedded.residual_local[:, 0])
    span_ok = (state.position[0] >= 0) & (state.position[0] <= config.context_capacity - rows)
    health = embedded.contract_valid[:, 0] & state.contract_valid[0] & span_ok
    selected = jnp.broadcast_to(state.selected_positions, (rows, config.geometry.dsa_top_k))
    counts = jnp.broadcast_to(state.selected_valid_counts, (rows,))
    scores = jnp.broadcast_to(state.selected_scores, selected.shape)
    kv, index = state.kv_cache_local, state.index_cache_local
    norm = partial(ws32_fused_add_rms_norm_mapped,
        global_hidden_size=config.geometry.hidden_size, epsilon=config.rms_norm_epsilon)
    for layer_id, layer in enumerate(weights.layers):
        normalized, residual = norm(hidden, carried, layer.qkv_a.input_norm_weight_local)
        slot = config.full_index_slot_by_layer[layer_id]

        def attend(caches, row):
            i, x, n, ids, count, score = row
            position = state.position + i
            result = attention_layer_bf16(x[None], *caches, ids[None], count[None], score[None],
                position, state.block_tables, state.context_lengths + i,
                layer.qkv_a, layer.attention, layer.dsa, normalized=n[None],
                dsa_contract=config.dsa_contract, attention_contract=config.attention_contract,
                cache_layout=config.cache_layout,
                sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
                sparse_attention_interpret=sparse_attention_interpret,
                main_rope_table_row=jnp.take(rope, position, axis=0, mode='clip')[0],
                lse_attention=False, dsa_two_stage=True)
            return (result.cache_local, result.index_cache_local), (
                result.output_local[0], result.selected_positions[0],
                result.selected_valid_counts[0], result.selected_scores[0], result.contract_valid[0])

        if batched_attention:
            result = attention_rows_bf16(residual, normalized, kv[layer_id],
                index[0 if slot is None else slot], selected, counts, scores, state.position,
                state.block_tables, state.context_lengths, layer, rope, config=config,
                interpret=sparse_attention_interpret)
            caches = result.cache_local, result.index_cache_local
            output, selected, counts, scores, valid = (result.output_local,
                result.selected_positions, result.selected_valid_counts,
                result.selected_scores, result.contract_valid)
        else:
            caches, attended = lax.scan(attend, (kv[layer_id], index[0 if slot is None else slot]),
                (jnp.arange(rows, dtype=jnp.int32), residual, normalized, selected, counts, scores))
            output, selected, counts, scores, valid = attended
        kv = kv.at[layer_id].set(caches[0])
        if slot is not None:
            index = index.at[slot].set(caches[1])
        normalized, carried = norm(output, residual, layer.post_attention_norm_weight_local)
        if layer.dense is not None:
            dense = dense_rows_bf16 if canonical_mlp else dense_bf16
            hidden = dense(normalized, layer.dense, expert_axis='expert', feature_axis='feature')
            mlp_ok = jnp.all(jnp.isfinite(hidden))
        else:
            moe = layer.moe
            routes, route_weights = jax.vmap(lambda x: ws32_router_from_shards_mapped(
                x[None], moe.router_weight_local, moe.correction_bias_local,
                top_k=config.moe_contract.top_k))(normalized)
            if canonical_mlp:
                hidden, mlp_ok = moe_rows_bf16(normalized, routes[:, 0], route_weights[:, 0],
                    moe, contract=config.moe_contract, interpret=linear_interpret)
            else:
                hidden, mlp_ok = _pooled_moe(normalized, routes[:, 0], route_weights[:, 0], *moe[2:8],
                    moe.shared_gate_local, None, moe.shared_up_local, None, moe.shared_down_local, None,
                    contract=config.moe_contract, interpret=linear_interpret,
                    expert_panels=expert_panels, fp32_route_sum=False)
        health = health & valid & mlp_ok
    normalized, residual = norm(hidden, carried, weights.final_norm_weight_local)
    logits = jax.vmap(lambda x: ws32_logits_mapped(x[None], weights.lm_head_local,
        vocab_size=config.geometry.vocab_size))(normalized)
    sampled = jax.vmap(lambda x: ws32_greedy_sample_mapped(x,
        vocab_size=config.geometry.vocab_size))(logits)
    health &= sampled.contract_valid[:, 0] & jnp.all(jnp.isfinite(normalized), axis=1)
    health &= jnp.all(jnp.isfinite(residual), axis=1)
    # P() metadata must have the same meaning on every owner, including owners
    # with no routed work. Do not publish just one replica's local health.
    health = lax.pmin(health.astype(jnp.int32), ('expert', 'feature')) != 0
    return VerificationProposal(kv, index, selected, counts, scores, sampled.token_id[:, 0],
        residual, normalized, health)


def commit_prefix_mapped(original, proposal, count, *, config):
    """Restore every rejected physical cache row before admitting the prefix.

    count is the number of consumed target input rows, including the row which
    produced a correction/bonus token. Zero leaves the original state unchanged.
    Invalid count/health refuses atomically and latches failed contract health.
    Original and proposal must come from the same invocation; this internal
    helper is not an authentication boundary for arbitrary caller-supplied data.
    """
    if count.shape != () or count.dtype != jnp.int32:
        raise ValueError('accepted prefix count must be an int32 scalar')
    rows = proposal.predictions.size
    valid = (count >= 0) & (count <= rows) & original.contract_valid[0]
    valid &= jnp.all(proposal.contract_valid)
    valid = lax.pmin(valid.astype(jnp.int32), ('expert', 'feature')) != 0
    safe_count = jnp.clip(count, 0, rows)
    kv, index = proposal.kv_cache_local, proposal.index_cache_local
    page_size, local_rows = config.logical_page_size, config.cache_layout.local_rows_per_page
    owner = lax.axis_index('expert')

    def restore(i, caches):
        position = original.position[0] + i
        logical = jnp.clip(position // page_size, 0, original.block_tables.shape[1] - 1)
        page = original.block_tables[0, logical]
        offset = position % page_size
        row = offset % local_rows
        owns = (offset // local_rows == owner) & (page >= 0) & (page < caches[0].shape[1])
        physical = jnp.clip(page, 0, caches[0].shape[1] - 1)
        def replace(values):
            return (values[0].at[:, physical, row].set(original.kv_cache_local[:, physical, row]),
                    values[1].at[:, physical, row].set(original.index_cache_local[:, physical, row]))
        return lax.cond((i >= safe_count) & owns, replace, lambda values: values, caches)

    kv, index = lax.fori_loop(0, rows, restore, (kv, index))
    last = jnp.maximum(safe_count - 1, 0)
    candidate = original._replace(kv_cache_local=kv, index_cache_local=index,
        selected_positions=proposal.selected_positions[last][None],
        selected_valid_counts=proposal.selected_valid_counts[last][None],
        selected_scores=proposal.selected_scores[last][None],
        position=original.position + safe_count, context_lengths=original.context_lengths + safe_count)
    candidate = jax.tree.map(lambda new, old: jnp.where(safe_count > 0, new, old), candidate, original)
    return jax.tree.map(lambda new, old: jnp.where(valid, new, old), candidate, original)._replace(
        contract_valid=original.contract_valid & valid)


def build_verifier(mesh, config, **options):
    if tuple(mesh.axis_names) != ('expert', 'feature') or tuple(mesh.devices.shape) != (8, 4):
        raise ValueError('verifier requires expert8 by feature4 mesh')
    return jax.jit(jax.shard_map(partial(verify_mapped, config=config, **options), mesh=mesh,
        in_specs=(P(), decoder.ws32_decoder_state_specs(), bf16_weight_specs(config), P()),
        out_specs=proposal_specs(), check_vma=False))


def build_prefix_committer(mesh, config):
    return jax.jit(jax.shard_map(partial(commit_prefix_mapped, config=config), mesh=mesh,
        in_specs=(decoder.ws32_decoder_state_specs(), proposal_specs(), P()),
        out_specs=decoder.ws32_decoder_state_specs(), check_vma=False))
