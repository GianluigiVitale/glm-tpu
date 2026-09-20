"""Research GLM MLA adaptation of Kaggle's global-max attention reduction.

Design source: ARahim3/kaggle-tpu-lab, revision
1aa1f083ae253470d9f355fe9c3eb12003300e91, glm53-flash/engine/glm53/model.py::_attend.
Unlike D5's normalized BF16 partials, retain FP32 unnormalized numerators until
after the global reduction. Exponent weights still round to BF16 before PV.
Global maximum and summation order differ from frozen online softmax; this is
not an exact or trained-admitted replacement. Research decoder/verifier builders
can opt in; default execution and native serving do not select it.

MIT License
Copyright (c) 2026 Abdur Rahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import jax.numpy as jnp
from jax import lax

from ..greenfield.kernels.reference.attention import (
    SparseAttentionResult, gather_stage_local_selected_kv,
)


def global_max_attention_mapped(query_nope, query_rope, cache, block_tables,
                                selected, context_lengths, *, contract, layout,
                                expert_axis='expert'):
    """Gather queries, compute owner-local scores/PV, scatter FP32 sums by head.

    Queries are already absorbed and rotary-transformed. Cache stays owner-local;
    selection validates each query's causal context and canonicalizes positions.
    Empty rows return zero output and -inf LSE. Nonfinite live inputs invalidate
    every replica, while unselected cache entries are not live operands.
    """
    owners = lax.axis_size(expert_axis)
    rows = query_nope.shape[0]
    if (rows < 1 or layout.local_parallel_size != owners or contract.num_heads % owners
            or layout.packed_cache_width != contract.packed_cache_width
            or query_nope.shape != (rows, contract.num_heads//owners, contract.kv_lora_rank)
            or query_rope.shape != (rows, contract.num_heads//owners, contract.qk_rope_head_dim)
            or any(x.dtype != jnp.bfloat16 for x in (query_nope, query_rope, cache))
            or selected.positions.shape != (rows, contract.top_k)):
        raise ValueError('global-max attention requires matching BF16 WS32 query/cache geometry')
    padding = contract.packed_cache_width-contract.kv_lora_rank-contract.qk_rope_head_dim
    packed = jnp.concatenate((query_nope, query_rope,
        jnp.zeros((*query_nope.shape[:2], padding), jnp.bfloat16)), axis=-1)
    queries = lax.all_gather(packed, expert_axis, axis=1, tiled=True)
    segment = gather_stage_local_selected_kv(cache, block_tables, selected,
        context_lengths, layout=layout, owner_index=lax.axis_index(expert_axis))
    valid = segment.contract_valid & jnp.all(jnp.isfinite(queries), axis=(1, 2))
    valid &= jnp.all(jnp.isfinite(segment.values), axis=(1, 2))
    # Health is kept separately; sanitize failed operands to avoid NaN collectives.
    queries = jnp.where(jnp.isfinite(queries), queries, jnp.zeros((), queries.dtype))
    keys = jnp.where(jnp.isfinite(segment.values), segment.values, jnp.zeros((), cache.dtype))
    scores = jnp.einsum('rhd,rkd->rhk', queries, keys,
        precision=lax.Precision.DEFAULT, preferred_element_type=jnp.float32)
    scores *= jnp.float32(contract.softmax_scale)
    live = jnp.arange(contract.top_k)[None, None, :] < segment.valid_counts[:, None, None]
    valid &= jnp.all(jnp.where(live, jnp.isfinite(scores), True), axis=(1, 2))
    scores = jnp.where(live & jnp.isfinite(scores), scores, -jnp.inf)
    maximum = lax.pmax(jnp.max(scores, axis=-1), expert_axis)
    safe_maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.)
    exponent = jnp.where(live, jnp.exp(scores-safe_maximum[..., None]), 0.)
    denominator = jnp.sum(exponent, axis=-1, dtype=jnp.float32)
    numerator = jnp.einsum('rhk,rkd->rhd', exponent.astype(cache.dtype),
        keys[..., :contract.kv_lora_rank], precision=lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32)
    numerator = lax.psum_scatter(numerator, expert_axis, scatter_dimension=1, tiled=True)
    denominator = lax.psum_scatter(denominator, expert_axis, scatter_dimension=1, tiled=True)
    local_heads = contract.num_heads//owners
    local_maximum = lax.dynamic_slice_in_dim(safe_maximum,
        lax.axis_index(expert_axis)*local_heads, local_heads, axis=1)
    output = numerator/jnp.where(denominator > 0, denominator, 1.)[..., None]
    lse = jnp.where(denominator > 0,
        local_maximum+jnp.log(jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny)), -jnp.inf)
    valid = lax.pmin(valid.astype(jnp.int32), expert_axis) != 0
    return SparseAttentionResult(output.astype(cache.dtype), lse, valid)
