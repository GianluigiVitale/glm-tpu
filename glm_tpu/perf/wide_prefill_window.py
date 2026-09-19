"""One wide sparse IndexShare prefix; original full-indexer/dense windows."""
import inspect

import jax.numpy as jnp

from ..greenfield.kernels.ws32_prefill_layer import Ws32PrefillLayerResult


def bind_wide_indexshare_window(window_body, prefix_body, mlp_body):
    """Preserve B128 suffixes and bypass the four-prefix scan on shared layers."""
    signature = inspect.signature(window_body)

    def wide(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        rows = len(values['hidden_update_local'])
        if rows <= 32 or values['dsa_weights'] is not None or values['dense_weights'] is not None:
            return window_body(*args, **kwargs)
        if (rows > 128 or not values['rolled_prefix'] or not values['expert_panels']
                or values['moe_weights'] is None or values['_observe'] is not None):
            raise ValueError('wide IndexShare requires <=128 sparse rows, rolled panels and no observer')
        capacity = values['cache_local'].shape[0] * values['cache_local'].shape[1] * 8
        start = jnp.clip(values['position_offset'], 0, capacity-1)
        count = jnp.clip(values['valid_rows'], 0, rows)
        span_valid = ((values['position_offset'] == start) & (values['valid_rows'] >= 0)
                      & (values['valid_rows'] <= rows) & (count <= capacity-start))
        call = {k:v for k,v in values.items() if k not in ('rolled_prefix','expert_panels','canonical_dense')}
        call.update(position_offset=start,valid_rows=count,
                    incoming_contract_valid=values['incoming_contract_valid'] & span_valid,prefix_only=True)
        # Keep the original scan's physical row padding (B114 still has four
        # M32 prefixes). Dropping those masked rows changes compiler dot shapes.
        padded = ((rows+31)//32)*32
        row_padding = dict(hidden_update_local=0,carried_residual_local=0,
            selected_positions=-1,selected_valid_counts=0,selected_scores=-jnp.inf,
            incoming_contract_valid=False,main_rope_table_rows=0)
        for name,fill in row_padding.items():
            value=call[name]
            call[name]=jnp.pad(value,((0,padded-rows),*((0,0) for _ in value.shape[1:])),constant_values=fill)
        prefix = prefix_body(**call)
        caches = {'cache_local','unrepaired_index_cache','repaired_index_cache'}
        prefix = prefix._replace(**{name:getattr(prefix,name)[:rows] for name in prefix._fields if name not in caches})
        live = jnp.arange(rows) < count
        output, ids, weights, health = mlp_body(prefix.normalized_mlp_local,live,
            values['dense_weights'],values['moe_weights'],moe_contract=values['moe_contract'],
            linear_interpret=values['linear_interpret'],expert_panels=True)
        output = jnp.where(live[:,None],output,0)
        health = prefix.contract_valid & health & (~live | jnp.all(jnp.isfinite(output),axis=1))
        return Ws32PrefillLayerResult(output,prefix.carried_residual_local,prefix.cache_local,
            prefix.unrepaired_index_cache,prefix.repaired_index_cache,prefix.selected_positions,
            prefix.selected_valid_counts,prefix.selected_scores,ids,weights,health,prefix.normalized_input_local)

    wide.__signature__ = signature
    return wide
