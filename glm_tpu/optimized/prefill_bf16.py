"""Resident BF16 projections of the production prefill.

``resident_matmul*``, ``resident_q_absorb`` and ``resident_value`` are the non-routed projections
of the prefill bodies over the resident BF16 tables of ``bf16_resident``. They refuse raw uint8
bits and any scale argument, so a raw FP8 table can never reach them silently. Projection
accumulation order is a numerical boundary even though decoded operands and explicit rounding
points are exact.
"""
import jax.numpy as jnp

from .bf16_resident import _dot_f32


def _require_table(weight, scale):
    if weight.ndim != 2 or weight.dtype != jnp.bfloat16 or scale is not None:
        raise ValueError("D8 prefill requires resident BF16 tables with no scale argument")


def resident_matmul_f32(lhs, weight, scale=None, *, interpret=False):
    _require_table(weight, scale)
    if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16 or lhs.shape[1] != weight.shape[1]:
        raise ValueError("D8 prefill matmul requires matching BF16 operands")
    # CPU's batched DotThunk cannot execute BF16 x BF16 -> FP32. The
    # interpret path widens already-rounded BF16 operands exactly, matching
    # the reference Pallas interpreter; production keeps BF16 MXU operands.
    return _dot_f32(lhs.astype(jnp.float32), weight.astype(jnp.float32)) if interpret else _dot_f32(lhs, weight)


def resident_matmul(lhs, weight, scale=None, *, interpret=False):
    return resident_matmul_f32(lhs, weight, scale, interpret=interpret).astype(jnp.bfloat16)


def resident_q_absorb(query, weight, scale=None, *, interpret=False):
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


def resident_value(latent, weight, scale=None, *, qk_nope_head_dim=192, interpret=False):
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
