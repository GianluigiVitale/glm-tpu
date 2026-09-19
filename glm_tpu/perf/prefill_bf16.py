"""D8 prefill bindings: BF16 non-routed tables, frozen routed FP8 and repair.

Only private function namespaces consume the adapted weight containers below.
The external API uses Bf16DecoderWeights, shared with decode; no raw-FP8 API
silently accepts these containers. Projection accumulation order is a numerical
boundary even though decoded operands and explicit rounding points are exact.
"""
import jax.numpy as jnp

from ..greenfield.kernels import ws32_prefill_attention as attention
from ..greenfield.kernels import ws32_prefill_dsa as dsa
from ..greenfield.kernels import ws32_prefill_layer as layer
from ..greenfield.kernels import ws32_prefill_linear as linear
from ..greenfield.kernels import ws32_prefill_moe as moe
from ..greenfield.kernels import ws32_prefill_dense_canonical as canonical
from ..greenfield.kernels.ws32_layer import (
    Ws32QkvAWeights, Ws32AttentionWeights, Ws32DsaWeights, Ws32DenseWeights, Ws32MoeWeights,
)
from ..greenfield.runtime.ws32_decoder import Ws32DecoderWeights, Ws32LayerWeights
from .bf16_resident import Bf16DecoderWeights, _dot_f32
from . import prefill_window as window


def _require_table(weight, scale):
    if weight.ndim != 2 or weight.dtype != jnp.bfloat16 or scale is not None:
        raise ValueError("D8 prefill requires resident BF16 tables with no scale argument")


def resident_matmul_f32(lhs, weight, scale, *, config=None, interpret=False):
    _require_table(weight, scale)
    if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16 or lhs.shape[1] != weight.shape[1]:
        raise ValueError("D8 prefill matmul requires matching BF16 operands")
    # CPU's batched DotThunk cannot execute BF16 x BF16 -> FP32. The
    # interpret path widens already-rounded BF16 operands exactly, matching
    # the reference Pallas interpreter; production keeps BF16 MXU operands.
    return _dot_f32(lhs.astype(jnp.float32), weight.astype(jnp.float32)) if interpret else _dot_f32(lhs, weight)


def resident_matmul(lhs, weight, scale, *, config=None, interpret=False):
    return resident_matmul_f32(lhs, weight, scale, config=config, interpret=interpret).astype(jnp.bfloat16)


def resident_q_absorb(query, weight, scale, *, prefill=False, interpret=False):
    _require_table(weight, scale)
    if query.ndim != 3 or query.dtype != jnp.bfloat16 or weight.shape[0] % query.shape[1]:
        raise ValueError("D8 prefill structured query geometry drifted")
    heads, qwidth = query.shape[1:]
    table = weight.reshape(heads, -1, weight.shape[1])
    if table.shape[1] <= qwidth:
        raise ValueError("D8 prefill structured table needs key and value rows")
    key = table[:, :qwidth]
    if interpret:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    return jnp.einsum('rhq,hqk->rhk', query, key,
                      preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def resident_value(latent, weight, scale, *, qk_nope_head_dim=192, prefill=False, interpret=False):
    _require_table(weight, scale)
    if (latent.ndim != 3 or latent.dtype != jnp.bfloat16 or
            weight.shape[0] % latent.shape[1] or weight.shape[1] != latent.shape[2]):
        raise ValueError("D8 prefill structured value geometry drifted")
    table = weight.reshape(latent.shape[1], -1, latent.shape[2])
    if not 0 < qk_nope_head_dim < table.shape[1]:
        raise ValueError("D8 prefill structured key/value split drifted")
    value = table[:, qk_nope_head_dim:]
    if interpret:
        latent, value = latent.astype(jnp.float32), value.astype(jnp.float32)
    return jnp.einsum('rhk,hvk->rhv', latent, value,
                      preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _adapt_weights(weights: Bf16DecoderWeights) -> Ws32DecoderWeights:
    """Aliases only, inside the mapped body; *_bits slots carry BF16, scales None."""
    if not isinstance(weights, Bf16DecoderWeights):
        raise ValueError("BF16 prefill requires Bf16DecoderWeights")
    layers = []
    for item in weights.layers:
        q, a, d, m, dn = item.qkv_a, item.attention, item.dsa, item.moe, item.dense
        qkv = Ws32QkvAWeights(q.input_norm_weight_local, q.q_a_local, None,
                              q.q_a_norm_weight, q.kv_a_local, None, q.kv_a_norm_weight)
        att = Ws32AttentionWeights(a.q_b_local, None, a.kv_b_local, None, a.o_local, None)
        ds = None if d is None else Ws32DsaWeights(d.wq_b_local, None, d.wk_local, None,
                                                  d.key_norm_weight, d.key_norm_bias, d.head_weight_local)
        dense = None if dn is None else Ws32DenseWeights(dn.gate_local, None, dn.up_local, None, dn.down_local, None)
        sparse = None if m is None else Ws32MoeWeights(*m[:8], m.shared_gate_local, None,
                                                       m.shared_up_local, None, m.shared_down_local, None)
        layers.append(Ws32LayerWeights(qkv, att, ds, item.post_attention_norm_weight_local, dense, sparse))
    return Ws32DecoderWeights(weights.embedding_local, tuple(layers),
                              weights.final_norm_weight_local, weights.lm_head_local)


def bind_bf16_prefill(bind, selector, attention_body, *, pooled_moe=False, wide_indexshare=False):
    """Return private layer/window bodies with the frozen health/repair schedule."""
    lin = bind(linear.ws32_prefill_linear_mapped, fp8_block_matmul_f32=resident_matmul_f32)
    dense = bind(linear.ws32_prefill_dense_mapped, fp8_block_matmul_f32=resident_matmul_f32)
    sparse = bind(moe.ws32_prefill_moe_from_routes_mapped,
                  fp8_block_matmul_f32=resident_matmul_f32, fp8_block_matmul=resident_matmul)
    mlp = bind(layer.ws32_prefill_mlp_mapped,
               ws32_prefill_dense_mapped=dense, ws32_prefill_moe_from_routes_mapped=sparse)
    prep = bind(attention.ws32_prefill_prepare_attention_mapped, ws32_prefill_linear_mapped=lin)
    att = bind(attention_body, fp8_block_matmul=resident_matmul,
               fp8_structured_kv_b_q_absorb=resident_q_absorb,
               fp8_structured_kv_b_value=resident_value, ws32_prefill_linear_mapped=lin)
    inputs = bind(dsa.ws32_prefill_dsa_inputs_mapped, fp8_block_matmul_f32=resident_matmul_f32)
    ds = bind(dsa.ws32_prefill_dsa_mapped, ws32_prefill_dsa_inputs_mapped=inputs,
              ws32_prefill_dsa_from_query_mapped=selector)
    layer_body = bind(layer.ws32_prefill_transformer_layer_mapped,
                      ws32_prefill_prepare_attention_mapped=prep,
                      ws32_prefill_index_share_attention_mapped=att,
                      ws32_prefill_dsa_mapped=ds, ws32_prefill_mlp_mapped=mlp)
    canonical_body = bind(canonical.ws32_prefill_dense_canonical_mapped, ws32_prefill_mlp_mapped=mlp)
    window_body = bind(window.ws32_prefill_layer_window_mapped,
                       ws32_prefill_transformer_layer_mapped=layer_body,
                       ws32_prefill_mlp_mapped=mlp,
                       ws32_prefill_dense_canonical_mapped=canonical_body)
    prefix_window = None
    if wide_indexshare:
        from . import wide_prefill_primitives as wide
        from .wide_prefill_window import bind_wide_indexshare_window
        from .pooled_prefill import _normalized_suffix
        wide_prep = bind(prep, _require_block=wide._require_block)
        wide_att = bind(att, _require_block=wide._require_block,
                        write_prefill_cache_block=wide.write_prefill_cache_block)
        wide_layer = bind(wide.ws32_prefill_transformer_layer_mapped,
            ws32_prefill_prepare_attention_mapped=wide_prep,
            ws32_prefill_index_share_attention_mapped=wide_att)
        if pooled_moe:
            narrow_prefix = bind(window_body, ws32_prefill_mlp_mapped=_normalized_suffix)
            prefix_window = bind_wide_indexshare_window(narrow_prefix,wide_layer,_normalized_suffix)
        window_body = bind_wide_indexshare_window(window_body,wide_layer,mlp)
    if pooled_moe:
        from .pooled_prefill import bind_pooled_window
        window_body = bind_pooled_window(window_body, sparse,prefix_window_body=prefix_window)
    return layer_body, window_body
