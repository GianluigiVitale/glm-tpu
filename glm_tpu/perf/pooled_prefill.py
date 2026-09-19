"""Pool MoE suffix rows while preserving B128 router/dense/prefix calls.

Larger expert kernels are a documented TPU numerical boundary. No serving
default uses this composition. Every <=32-row attention/repair prefix remains
inside the existing B128 (or final B114) window.
"""
import inspect

import jax.numpy as jnp

from ..greenfield.kernels.ws32_io import Ws32EmbeddingResult
from ..greenfield.kernels.ws32_prefill_layer import ws32_prefill_router_mapped
from ..greenfield.runtime.ws32_batched_prefill import ws32_prefill_embedding_mapped
from .function_bindings import bind_dependencies


def chunked_embedding(tokens, weights, count, *, vocab_size):
    parts = [ws32_prefill_embedding_mapped(tokens[i:i+128], weights,
        jnp.clip(count-i, 0, min(128, len(tokens)-i)), vocab_size=vocab_size)
        for i in range(0, len(tokens), 128)]
    return Ws32EmbeddingResult(*(jnp.concatenate([p[n] for p in parts]) for n in range(2)))


def _normalized_suffix(normalized, live, dense, moe, *, moe_contract, **options):
    if dense is not None or moe is None:
        raise ValueError('pooled normalization capture requires a sparse layer')
    rows = len(normalized)
    return (normalized, jnp.full((rows, moe_contract.top_k), -1, jnp.int32),
            jnp.zeros((rows, moe_contract.top_k), jnp.float32), jnp.ones((rows,), jnp.bool_))


def bind_pooled_window(window_body, moe_body):
    """Return a private layer callable; frozen functions/globals stay intact."""
    prefix = bind_dependencies(window_body, ws32_prefill_mlp_mapped=_normalized_suffix)
    signature = inspect.signature(window_body)

    def pooled(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        rows = len(values['hidden_update_local'])
        if rows <= 128:
            return window_body(*args, **kwargs)
        if (rows > 1024 or rows % 128 not in (0, 114)
                or not values['rolled_prefix'] or not values['expert_panels']
                or values['_observe'] is not None):
            raise ValueError('pooled window requires <=1024 rows in canonical chunks, rolled panels and no observer')
        sparse = values['moe_weights'] is not None
        if sparse == (values['dense_weights'] is not None):
            raise ValueError('pooled window requires exactly one MLP branch')
        capacity = values['cache_local'].shape[0] * values['cache_local'].shape[1] * 8
        start = jnp.clip(values['position_offset'], 0, capacity-1)
        count = jnp.clip(values['valid_rows'], 0, rows)
        valid_span = ((values['position_offset'] == start) & (values['valid_rows'] >= 0)
                      & (values['valid_rows'] <= rows) & (count <= capacity-start))
        row_fields = ('hidden_update_local', 'carried_residual_local', 'selected_positions',
            'selected_valid_counts', 'selected_scores', 'incoming_contract_valid', 'main_rope_table_rows')
        cache_fields = ('cache_local', 'unrepaired_index_cache', 'repaired_index_cache')
        parts = []
        current = {name: values[name] for name in cache_fields}
        for first in range(0, rows, 128):
            last = min(first+128, rows)
            call = {**values, **current, **{name: values[name][first:last] for name in row_fields}}
            call['position_offset'] = start + jnp.minimum(first, capacity-1-start)
            call['valid_rows'] = jnp.clip(count-first, 0, last-first)
            call['incoming_contract_valid'] = call['incoming_contract_valid'] & valid_span
            result = (prefix if sparse else window_body)(**call)
            parts.append(result)
            current = {name: getattr(result, name) for name in cache_fields}
        joined = {name: jnp.concatenate([getattr(p, name) for p in parts])
                  for name in parts[0]._fields if name not in cache_fields}
        if sparse:
            normalized = joined['output_local']
            live = jnp.arange(rows) < count
            weights, contract = values['moe_weights'], values['moe_contract']
            if weights.router_weight_local.shape[0] * 8 != contract.num_experts:
                raise ValueError('prefill router and grouped expert counts disagree')
            routes = [ws32_prefill_router_mapped(normalized[i:i+128], weights.router_weight_local,
                weights.correction_bias_local, live[i:i+128], top_k=contract.top_k)
                for i in range(0, rows, 128)]
            ids, mixture, router_health = (jnp.concatenate([r[n] for r in routes]) for n in range(3))
            output, healthy = moe_body(normalized, ids, mixture, *weights[2:], contract=contract,
                interpret=values['linear_interpret'], fp32_route_sum=True, expert_panels=True)
            joined.update(output_local=jnp.where(live[:, None], output, 0),
                route_indices=ids, route_weights=mixture,
                contract_valid=joined['contract_valid'] & router_health & healthy
                    & (~live | jnp.all(jnp.isfinite(output), axis=1)))
        return parts[-1]._replace(**joined, **current)

    return pooled
