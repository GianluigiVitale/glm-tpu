"""Resident BF16 projections of the production prefill, and the FP8-shaped weight view.

``resident_matmul*``, ``resident_q_absorb`` and ``resident_value`` replace the raw-FP8 Pallas
matmuls of the frozen prefill bodies for every non-routed table (the production prefill modules
call them explicitly since the S2d fold). They refuse raw uint8 bits and any scale argument, so a
raw FP8 table can never reach them silently. ``_adapt_weights`` is the view the prefill bodies
still take: BF16 tables in the ``*_bits_local`` fields with ``None`` scales. Projection
accumulation order is a numerical boundary even though decoded operands and explicit rounding
points are exact.
"""
import jax.numpy as jnp

from ..greenfield.kernels.ws32_layer import (
    Ws32QkvAWeights, Ws32AttentionWeights, Ws32DsaWeights, Ws32DenseWeights, Ws32MoeWeights,
)
from ..greenfield.runtime.ws32_decoder import Ws32DecoderWeights, Ws32LayerWeights
from .bf16_resident import Bf16DecoderWeights, _dot_f32


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
