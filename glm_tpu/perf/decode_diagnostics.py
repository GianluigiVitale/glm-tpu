"""One-step layer-boundary diagnostics, never a serving or timing path.

Each layer completes in its own executable. This deliberately changes compiler
boundaries; agreement or disagreement diagnoses the composed graph, and does not
certify its arithmetic. Only aggregate activation statistics enter the report.
"""
import numpy as np


def collect_greedy_trail(step, token, state, expected, *, cache_check=None):
    """Diagnostic continuation from one immutable prefill state, no timing claim.

    ``step(index, token, state)`` must complete/admit the execution (including
    the caller's fleet vote). Returned token arrays stay private to the caller.
    """
    from hashlib import sha256
    import jax
    if expected.shape != (29,) or expected.dtype != np.int32:
        raise ValueError('diagnostic trail requires the authenticated 29-ID reference')
    observed=[int(np.asarray(token)[0])]
    stats=[]
    healthy=[]
    for index in range(1,29):
        result=step(index,token,state)
        jax.block_until_ready(result)
        state,token=result.state,result.next_token
        observed.append(int(np.asarray(token)[0]))
        healthy.append(bool(np.asarray(state.contract_valid).all()))
        stats.append(activation_stats(result.final_residual_local))
    values=np.asarray(observed,np.int32)
    mismatches=np.flatnonzero(values!=expected)
    return values,dict(healthy=all(healthy),all_steps_finite=all(s['finite'] for s in stats),
        steps=28,first_residual=stats[0],final_residual=stats[-1],
        final_cache_finite=None if cache_check is None else bool(cache_check(state)),
        peak_residual_abs=max(s['max_abs'] for s in stats),
        token_comparison=dict(compared=29,matches=int(np.sum(values==expected)),
            all_equal=not bool(mismatches.size),first_mismatch_index=None if not mismatches.size else int(mismatches[0]),
            observed_sha256=sha256(values.tobytes()).hexdigest()),timing_claim=False)


def activation_stats(value):
    arrays = [np.asarray(s.data).astype(np.float64) for s in value.addressable_shards]
    return dict(finite=all(np.isfinite(x).all() for x in arrays),
                nonzero=sum(int(np.count_nonzero(x)) for x in arrays),
                elements=sum(x.size for x in arrays),
                max_abs=max(float(np.max(np.abs(x))) for x in arrays),
                rms=float(np.sqrt(sum(float(np.square(x).sum()) for x in arrays)
                                  / sum(x.size for x in arrays))))


def diagnose_layerwise(mesh, config, options, token, state, weights, rope,
                       *, compile_program, observe=lambda report: None,
                       sparse_attention_interpret=False, linear_interpret=False):
    """Compile callback admits each graph before execution; caller owns leases.

    Returns the split greedy-head tuple and an aggregate diagnostic report.
    Input caches are immutable; only one decode step is explored. Extra outputs
    and layer boundaries can change rounding relative to the complete decoder.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from ..greenfield.kernels.ws32_io import ws32_embedding_mapped, ws32_split_final_sample_mapped
    from ..greenfield.kernels.ws32_layer import Ws32TransformerLayerResult
    from .bf16_resident import bf16_weight_specs, transformer_layer_bf16

    if (not options.bf16_resident or options.sampler != 'greedy'
            or config.exact_dsa or config.strategy_nd_dense or not config.host_main_rope_table):
        raise ValueError('layerwise diagnostic requires resident raw-default greedy with host rope')
    def mapped(body, ins, outs):
        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False))
    def embed(t, w, r, p):
        out = ws32_embedding_mapped(t, w, vocab_size=config.geometry.vocab_size)
        return out.residual_local, out.contract_valid, jnp.take(r, p, axis=0, mode='clip')[0]
    specs = bf16_weight_specs(config)
    fn = mapped(embed, (P(), specs.embedding_local, P(), P()), (P(None,'feature'), P(), P()))
    args = (token, weights.embedding_local, rope, state.position)
    hidden, healthy, row = compile_program('diagnostic_embedding', fn, args)(*args)
    carried = jnp.zeros_like(hidden)
    healthy = healthy & state.contract_valid
    selected, counts, scores = state.selected_positions, state.selected_valid_counts, state.selected_scores
    report = dict(schema='glm_perf_layerwise_diagnostic_v1', compiler_boundary_changed=True,
                  embedding=activation_stats(hidden), layers=[])
    observe(report)
    cache_spec = P(None,'expert',None)
    out_specs = Ws32TransformerLayerResult(P(None,'feature'), P(None,'feature'), P(None,'feature'),
                    cache_spec, cache_spec, P(), P(), P(), P(), P(), P())
    compiled = {}
    for i, layer in enumerate(weights.layers):
        indexer, mlp = config.geometry.indexer_types[i], config.geometry.mlp_layer_types[i]
        slot = config.full_index_slot_by_layer[i]
        args = (hidden, carried, state.kv_cache_local[i], state.index_cache_local[0 if slot is None else slot],
                selected, counts, scores, state.position, state.block_tables, state.context_lengths, layer, healthy, row)
        key = (indexer, mlp)
        if key not in compiled:
            def body(h, c, kv, idx, pos, cnt, score, p, table, length, w, ok, rope_row,
                     indexer=indexer, mlp=mlp):
                return transformer_layer_bf16(h,c,kv,idx,pos,cnt,score,p,table,length,w,ok,
                    indexer_kind=indexer, mlp_kind=mlp, config=config,
                    routed_projection=options.routed_projection,
                    sparse_attention_interpret=sparse_attention_interpret, linear_interpret=linear_interpret,
                    main_rope_table_row=rope_row, lse_attention=options.lse_attention,
                    dsa_two_stage=options.dsa_two_stage, fused_feature_reductions=options.fused_feature_reductions)
            ins = (P(None,'feature'),P(None,'feature'),cache_spec,cache_spec,P(),P(),P(),P(),P(),P(),specs.layers[i],P(),P())
            compiled[key] = compile_program('diagnostic_layer_'+indexer+'_'+mlp, mapped(body,ins,out_specs), args)
        out = compiled[key](*args)
        jax.block_until_ready(out)
        hidden, carried = out.output_local, out.carried_residual_local
        selected, counts, scores, healthy = out.selected_positions, out.selected_valid_counts, out.selected_scores, out.contract_valid
        report['layers'].append(dict(layer=i, indexer=indexer, mlp=mlp,
            update=activation_stats(hidden), carried=activation_stats(carried),
            normalized_input=activation_stats(out.normalized_input_local),
            healthy=bool(np.asarray(healthy).all()),
            selected_count=int(np.asarray(counts)[0])))
        observe(report)
    def head(h, c, n, w):
        return tuple(ws32_split_final_sample_mapped(h,c,n,w,hidden_size=config.geometry.hidden_size,
                    vocab_size=config.geometry.vocab_size,rms_norm_epsilon=config.rms_norm_epsilon))
    args = (hidden, carried, weights.final_norm_weight_local, weights.lm_head_local)
    fn = mapped(head,(P(None,'feature'),P(None,'feature'),specs.final_norm_weight_local,specs.lm_head_local),
                (P(),P(),P(None,'feature')))
    result = compile_program('diagnostic_head',fn,args)(*args)
    jax.block_until_ready(result)
    report['final_residual'] = activation_stats(result[2])
    report['head_healthy'] = bool(np.asarray(result[1]).all())
    observe(report)
    return result, report
