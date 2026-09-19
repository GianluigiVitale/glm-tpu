"""Divide replicated prefill query rows across feature chips, then gather rows."""
import jax
from jax import lax

from ..greenfield.kernels.reference.attention import SparseAttentionResult
from ..greenfield.kernels.reference.dsa import SelectedPositions
from .lse_attention import lse_attention_mapped


def feature_row_lse_attention(query_nope, query_rope, cache, block_tables,
        selected, context_lengths, *, attention_body=lse_attention_mapped, **kwargs):
    """Same per-row attention arithmetic, on feature-replicated cache/query data.

    Only the WS32 prefill composition establishes feature replication of these
    inputs. This is not a wrapper for arbitrary feature-sharded queries/caches.
    Expert-axis head ownership and owner-local fallback remain unchanged. All
    result rows (including health and empty-row sentinels) are restored before
    subsequent projections or the model's all-owner commit checks.
    """
    rows = query_nope.shape[0]
    features = lax.axis_size('feature')
    if features!=4 or rows<4 or rows%4:
        raise ValueError('feature-row attention requires feature4 and whole groups of four rows')
    width = rows//features
    start = lax.axis_index('feature')*width
    def take(value):
        if value.shape[0]!=rows:
            raise ValueError('feature-row attention row geometry differs')
        return lax.dynamic_slice_in_dim(value,start,width,axis=0)
    with jax.named_scope('glm_perf_feature_row_attention'):
        part = attention_body(take(query_nope),take(query_rope),cache,take(block_tables),
            SelectedPositions(take(selected.positions),take(selected.valid_counts)),
            take(context_lengths),**kwargs)
        return SparseAttentionResult(*(lax.all_gather(value,'feature',axis=0,tiled=True) for value in part))
