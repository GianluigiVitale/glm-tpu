"""Opt-in P2 prefill composition with unchanged cache/commit/repair control flow.

The frozen composition functions have no dependency-injection hooks. Bind fresh
function objects to private copies of their globals instead of patching modules
or duplicating their 1,000+ lines of state management. The DSA selector and,
optionally, the attention exchange are replaced. The original functions, bytecode, defaults and module globals remain
unchanged, and both programs can be traced/executed in the same process.

This explicitly follows the pinned frozen call graph (selector -> DSA -> layer
-> window -> runtime -> builder). A changed call graph needs review and tests.
"""
from .function_bindings import bind_dependencies as _bind_dependencies

from ..greenfield.kernels import ws32_prefill_dsa as dsa
from ..greenfield.kernels import ws32_prefill_layer as layer
from ..greenfield.kernels import ws32_prefill_window as window
from ..greenfield.runtime import ws32_batched_prefill as runtime
from .dsa_candidates import prefill_dsa_one_pass_mapped


def _one_pass_selector(*args, key_tile=4096, paired_position_sort=False,
                       sorted_local_merge=False, **kwargs):
    # These control only the frozen tile chain that P2 replaces. Precision,
    # causal bounds, geometry and top-k still pass through unchanged.
    if type(key_tile) is not int or not 128 <= key_tile <= 4096 or key_tile % 128:
        raise ValueError("prefill key tile must be a multiple of 128 up to 4096")
    if type(paired_position_sort) is not bool or type(sorted_local_merge) is not bool:
        raise ValueError("prefill sort options must be static booleans")
    if sorted_local_merge and kwargs.get("top_k", 2048) & (kwargs.get("top_k", 2048)-1):
        raise ValueError("sorted local merge requires power-of-two top_k")
    del key_tile, paired_position_sort, sorted_local_merge
    return prefill_dsa_one_pass_mapped(*args, **kwargs)


def build_ws32_prefill_challenger_program(mesh, config, *, lse_attention=False, bf16_resident=False,
                                         owned_key_capacity=None, pooled_moe=False, wide_indexshare=False,
                                         feature_row_attention=False, **options):
    """Frozen greedy prefill API with P2 and optional P1 local attention.

    With bf16_resident=True, consumes the same Bf16DecoderWeights as decode.
    M64 repaired-key production, canonical dense placement, routed FP8 kernels
    and all-owner atomic state admission are preserved.
    owned_key_capacity opts into smaller owner buffers with full-width fallback;
    it requires LSE attention and retains the primitive's segment arithmetic.
    """
    if type(lse_attention) is not bool:
        raise ValueError("lse_attention must be a static boolean")
    if type(bf16_resident) is not bool:
        raise ValueError("bf16_resident must be a static boolean")
    if type(pooled_moe) is not bool:
        raise ValueError("pooled_moe must be a static boolean")
    if type(feature_row_attention) is not bool or (feature_row_attention and (
            not lse_attention or not bf16_resident or not options.get('mlp_window')
            or not options.get('rolled_prefix'))):
        raise ValueError('feature-row attention requires resident LSE rolled prefill windows')
    if type(wide_indexshare) is not bool or (wide_indexshare and (
            not bf16_resident or not lse_attention or owned_key_capacity is None
            or not all(options.get(k) is True for k in ('mlp_window','rolled_prefix','expert_panels','canonical_dense'))
            or any(options.get(k,False) for k in ('pending_cache_rows','flat_pending_rows','capture_barrier')))):
        raise ValueError('wide IndexShare requires resident bounded LSE, canonical rolled panels and full cache proposals')
    runtime_source = runtime
    if pooled_moe:
        if (not bf16_resident or not all(options.get(k) is True for k in
                ('mlp_window', 'rolled_prefix', 'expert_panels', 'canonical_dense'))
                or any(options.get(k, False) for k in
                       ('pending_cache_rows', 'flat_pending_rows', 'capture_barrier'))):
            raise ValueError('pooled MoE requires resident canonical rolled panels and full cache proposals')
        from . import pooled_prefill_runtime as runtime_source
    if owned_key_capacity is not None:
        block = min(config.sparse_segment_block, config.geometry.dsa_top_k)
        if (not lse_attention or type(owned_key_capacity) is not int
                or owned_key_capacity < block or owned_key_capacity % block):
            raise ValueError("owned key capacity requires LSE attention and whole segment blocks")
    attention_dependencies = {}
    if lse_attention:
        from .prefill_attention import prefill_index_share_lse_mapped

        if owned_key_capacity is not None or wide_indexshare or feature_row_attention:
            from functools import partial
            from .lse_attention import lse_attention_mapped, gathered_partial_attention

            if wide_indexshare:
                lse_attention_mapped = _bind_dependencies(lse_attention_mapped,
                    gathered_partial_attention=partial(gathered_partial_attention,maximum_rows=128))

            if feature_row_attention:
                from .feature_row_attention import feature_row_lse_attention
                lse_attention_mapped = partial(feature_row_lse_attention,attention_body=lse_attention_mapped)

            prefill_index_share_lse_mapped = _bind_dependencies(
                prefill_index_share_lse_mapped,
                lse_attention_mapped=partial(lse_attention_mapped, owned_key_capacity=owned_key_capacity))
        attention_dependencies["ws32_prefill_index_share_attention_mapped"] = prefill_index_share_lse_mapped
    if bf16_resident:
        from .prefill_bf16 import bind_bf16_prefill
        from ..greenfield.kernels.ws32_prefill_attention import ws32_prefill_index_share_attention_mapped

        layer_body, window_body = bind_bf16_prefill(
            _bind_dependencies, _one_pass_selector,
            attention_dependencies.get("ws32_prefill_index_share_attention_mapped",
                                       ws32_prefill_index_share_attention_mapped), pooled_moe=pooled_moe,
            wide_indexshare=wide_indexshare)
    else:
        dsa_body = _bind_dependencies(dsa.ws32_prefill_dsa_mapped,
                                     ws32_prefill_dsa_from_query_mapped=_one_pass_selector)
        layer_body = _bind_dependencies(layer.ws32_prefill_transformer_layer_mapped,
                                       ws32_prefill_dsa_mapped=dsa_body, **attention_dependencies)
        window_body = _bind_dependencies(window.ws32_prefill_layer_window_mapped,
                                        ws32_prefill_transformer_layer_mapped=layer_body)
    runtime_dependencies = {}
    if pooled_moe:
        from .pooled_prefill import chunked_embedding
        runtime_dependencies['ws32_prefill_embedding_mapped'] = chunked_embedding
    runtime_body = _bind_dependencies(runtime_source.ws32_batched_prefill_mapped,
                                     ws32_prefill_transformer_layer_mapped=layer_body,
                                     ws32_prefill_layer_window_mapped=window_body, **runtime_dependencies)
    builder_dependencies = {}
    if bf16_resident:
        from .bf16_resident import bf16_weight_specs
        from .prefill_bf16 import _adapt_weights

        original_runtime = runtime_body

        def runtime_body(tokens, count, state, weights, wk, rope, **kwargs):
            return original_runtime(tokens, count, state, _adapt_weights(weights), wk, rope, **kwargs)

        builder_dependencies["ws32_decoder_weight_specs"] = bf16_weight_specs
    builder = _bind_dependencies(runtime_source.build_ws32_batched_prefill_program,
                                 ws32_batched_prefill_mapped=runtime_body, **builder_dependencies)
    return builder(mesh, config, **options)
