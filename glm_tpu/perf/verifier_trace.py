"""Instrumented layer/head observations of existing ordinary/verifier bodies.

Extra live outputs can change compiler rounding. Always compare the instrumented
result with the uninstrumented executable before attributing a trained mismatch.
These builders neither serve output nor replace the ordinary or verifier path.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from ..greenfield.kernels.ws32_io import ws32_logits_mapped
from ..greenfield.runtime import ws32_decoder as decoder
from .bf16_resident import bf16_weight_specs
from .speculative_verify import verify_mapped, proposal_specs
from .ws32_decoder_challenger import ws32_decode_challenger_mapped


class TargetTrace(NamedTuple):
    normalized_inputs: object
    hidden_updates: object
    carried_residuals: object
    selected_positions: object
    selected_counts: object
    selected_scores: object
    final_normalized: object
    top_ids: object
    top_scores: object


def trace_specs():
    return TargetTrace(*(P(None, None, 'feature'),)*3,
                       P(), P(), P(), P(None, 'feature'), P(), P())


def _head_observation(hidden, carried, weights, config):
    """Use the live pre-norm pair, never renormalize the rounded final residual."""
    normalized, ids, scores = [], [], []
    for row in range(hidden.shape[0]):
        h, _ = ws32_fused_add_rms_norm_mapped(hidden[row:row+1], carried[row:row+1],
            weights.final_norm_weight_local, global_hidden_size=config.geometry.hidden_size,
            epsilon=config.rms_norm_epsilon)
        logits = ws32_logits_mapped(h, weights.lm_head_local,
                                   vocab_size=config.geometry.vocab_size)[0]
        local_scores, local_ids = lax.top_k(logits.astype(jnp.float32), 2)
        local_ids += lax.axis_index('expert')*logits.size
        candidates = lax.all_gather(local_scores, 'expert', axis=0, tiled=True)
        candidate_ids = lax.all_gather(local_ids, 'expert', axis=0, tiled=True)
        # Exact highest-score, then lowest-ID ordering across all shard winners.
        negative, ordered_ids = lax.sort((-candidates, candidate_ids), num_keys=2)
        normalized.append(h)
        ids.append(ordered_ids[:2])
        scores.append(-negative[:2])
    return jnp.concatenate(normalized), jnp.stack(ids), jnp.stack(scores)


def build_target_trace(mesh, config, *, ordinary_options=None,
                       unrolled_attention=False,
                       sparse_attention_interpret=False, linear_interpret=False):
    """None selects the current M8 verifier; options select ordinary decode."""
    if tuple(mesh.axis_names) != ('expert', 'feature') or tuple(mesh.devices.shape) != (8, 4):
        raise ValueError('target trace requires expert8 by feature4 mesh')
    if type(unrolled_attention) is not bool or (unrolled_attention and ordinary_options is not None):
        raise ValueError('unrolled attention is a verifier-only trace option')
    if ordinary_options is not None and (
            not ordinary_options.bf16_resident or ordinary_options.sampler != 'greedy'):
        raise ValueError('ordinary trace requires the resident greedy decoder')

    def body(tokens, state, weights, rope):
        layers, head = [], []

        def observe_layer(index, normalized, hidden, carried, selected, counts, scores):
            if index != len(layers):
                raise ValueError('nonsequential diagnostic layer observation')
            layers.append((normalized, hidden, carried, selected, counts, scores))

        def observe_head(hidden, carried):
            head.append(_head_observation(hidden, carried, weights, config))

        observers = dict(observe_layer=observe_layer, observe_head=observe_head,
            sparse_attention_interpret=sparse_attention_interpret, linear_interpret=linear_interpret)
        if ordinary_options is None:
            result = verify_mapped(tokens, state, weights, rope, config=config,
                canonical_mlp=True, batched_attention=not unrolled_attention, small_expert_tiles=True,
                rowwise_dsa=not unrolled_attention, unrolled_attention=unrolled_attention, **observers)
        else:
            result = ws32_decode_challenger_mapped(tokens, state, weights,
                main_rope_table=rope, config=config, options=ordinary_options, **observers)
        if len(layers) != config.geometry.num_layers or len(head) != 1:
            raise ValueError('incomplete target trace')
        return result, TargetTrace(*(jnp.stack([layer[i] for layer in layers]) for i in range(6)),
                                   *head[0])

    if ordinary_options is None:
        result_specs = proposal_specs()
    else:
        result_specs = decoder.Ws32DecodeStepResult(decoder.ws32_decoder_state_specs(),
                                                   P(), P(None, 'feature'))
    return jax.jit(jax.shard_map(body, mesh=mesh,
        in_specs=(P(), decoder.ws32_decoder_state_specs(), bf16_weight_specs(config), P()),
        out_specs=(result_specs, trace_specs()), check_vma=False))


def compare_trace_window(tokens, reference_states, reference_results, proposal, *,
                         traced_ordinary, traced_verifier, ready, put, compare, healthy):
    """Keep original roots for each row; report instrumentation changes explicitly."""
    import numpy as np
    observed, candidate = ready(traced_verifier(put(tokens), reference_states[0]))
    healthy(observed.contract_valid, 'traced_verify')
    traces = []
    baseline_instrumentation = []
    for row, reference in enumerate(reference_results):
        result, trace = ready(traced_ordinary(put(tokens[row:row+1]), reference_states[row]))
        healthy(result.state.contract_valid, f'traced_ordinary_{row}')
        baseline_instrumentation.append(dict(
            prediction_equal=bool(np.array_equal(np.asarray(result.next_token), np.asarray(reference.next_token))),
            head_matches_prediction=bool(np.array_equal(np.asarray(trace.top_ids[:,0]), np.asarray(result.next_token))),
            residual=compare(result.final_residual_local, reference.final_residual_local),
            state={name:compare(getattr(result.state,name),getattr(reference.state,name)) for name in reference.state._fields}))
        traces.append(trace)
    sequential = TargetTrace(*(jnp.concatenate([t[i] for t in traces], axis=1 if i<6 else 0)
                               for i in range(len(TargetTrace._fields))))
    layers = []
    for i in range(candidate.normalized_inputs.shape[0]):
        comparisons = {field:compare(getattr(candidate,field)[i],getattr(sequential,field)[i])
                      for field in TargetTrace._fields[:6]}
        layers.append(dict(layer=i, comparisons=comparisons))
    actual_ids = np.asarray(candidate.top_ids)
    ordinary_ids = np.asarray(sequential.top_ids)
    return dict(compiler_outputs_changed=True,
        verifier_instrumentation=dict(
            predictions_equal=bool(np.array_equal(np.asarray(observed.predictions),np.asarray(proposal.predictions))),
            fields={name:compare(getattr(observed,name),getattr(proposal,name)) for name in proposal._fields}),
        ordinary_instrumentation=baseline_instrumentation,
        layers=layers,
        final_normalized=compare(candidate.final_normalized,sequential.final_normalized),
        verifier_head_matches_prediction=bool(np.array_equal(actual_ids[:,0],np.asarray(observed.predictions))),
        ordinary_head_matches_prediction=all(x['head_matches_prediction'] for x in baseline_instrumentation),
        top_two_ids_equal_by_row=np.all(actual_ids==ordinary_ids,axis=1).tolist(),
        verifier_top_score=np.asarray(candidate.top_scores[:,0]).astype(float).tolist(),
        ordinary_top_score=np.asarray(sequential.top_scores[:,0]).astype(float).tolist(),
        verifier_logit_margin=np.asarray(candidate.top_scores[:,0]-candidate.top_scores[:,1]).astype(float).tolist(),
        ordinary_logit_margin=np.asarray(sequential.top_scores[:,0]-sequential.top_scores[:,1]).astype(float).tolist())
