"""Multi-head latent attention (MLA) of the reference, absorbed form.

Per row at absolute position ``p`` (``normalized`` is the BF16 input norm):

* ``q_residual = rms_norm(normalized @ q_a.T)``; ``kv = normalized @ kv_a.T``
  splits into the latent ``c = rms_norm(kv[:kv_lora])`` and the raw key RoPE
  part ``kv[kv_lora:]`` (all BF16 boundaries);
* ``q = q_residual @ q_b.T`` per head is ``[q_nope (192) | q_rope (64)]``;
* main RoPE uses the host BF16 ``cos|sin`` table row of ``p`` (NumPy FP32
  trigonometry stored in BF16), FP32 products and one final BF16 round, with
  interleaved pairs, for ``q_rope`` and the key RoPE part alike;
* the cache row of ``p`` is ``[c | rope(k_rope)]`` (BF16, written before
  attending, so a row can attend to itself);
* absorbed query ``q_c = q_nope @ W_k`` (``W_k`` = the 192 key rows of the head's
  ``kv_b`` slice), BF16;
* scores over the selected positions ``(q_c . c_t + q_rope . k_rope_t) *
  qk_head_dim**-0.5`` in FP32, softmax in FP32 with the unnormalized
  probabilities rounded to BF16 before the value product, the positions in
  ascending order; ``o = (sum_t P_t c_t) / sum_t P_t`` rounded to BF16;
* per head value ``v = o @ W_v.T`` (the 256 value rows), BF16; output
  ``concat_heads(v) @ o_proj.T``, BF16.

The softmax is ``sparse_mla_attention`` of ``glm_tpu/greenfield/kernels/
reference/attention.py`` over the rows ``gather_paged_selected_kv`` gathers (an
unsharded cache, one page); the RoPE table and rotation are the engine's
``build_rotary_table_host`` / ``apply_rotary_fp32_final_round``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract,
    gather_paged_selected_kv,
    sparse_mla_attention,
)
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.reference.rotary import (
    apply_rotary_fp32_final_round,
    build_rotary_table_host,
)

from .linear import einsum, project
from .norm import rms_norm


class QkvAWeights(NamedTuple):
    q_a: jax.Array  # [q_lora_rank, hidden]
    q_a_norm: jax.Array  # [q_lora_rank]
    kv_a: jax.Array  # [kv_lora_rank + rope_dim, hidden]
    kv_a_norm: jax.Array  # [kv_lora_rank]


class AttentionWeights(NamedTuple):
    q_b: jax.Array  # [heads * (nope + rope), q_lora_rank]
    # [heads * (nope + v), kv_lora_rank]: per head the nope key rows, then the value rows
    kv_b: jax.Array
    o: jax.Array  # [hidden, heads * v]


class PreparedAttention(NamedTuple):
    q_residual: jax.Array  # [rows, q_lora_rank] BF16
    latent: jax.Array  # [rows, kv_lora_rank] BF16 (normalized)
    key_rope_input: jax.Array  # [rows, rope_dim] BF16 (not yet rotated)


def main_rope_table(capacity: int, *, rotary_dim: int, theta: float) -> jax.Array:
    """Host BF16 ``cos|sin`` table, one row per position."""
    table = build_rotary_table_host(capacity, rotary_dim=rotary_dim, theta=theta)
    return jnp.asarray(np.asarray(table))


def rotate(value: jax.Array, table_rows: jax.Array) -> jax.Array:
    """Interleaved main RoPE of ``value[rows, ..., rope_dim]`` with the rows' table rows."""
    half = table_rows.shape[-1] // 2
    shape = (table_rows.shape[0],) + (1,) * (value.ndim - 2) + (half,)
    cos = table_rows[:, :half].reshape(shape)
    sin = table_rows[:, half:].reshape(shape)
    return apply_rotary_fp32_final_round(value, cos, sin, interleaved=True)


def prepare(
    normalized: jax.Array, weights: QkvAWeights, *, epsilon: float
) -> PreparedAttention:
    """The shared q-a / kv-a boundary consumed by attention and the DSA indexer."""
    kv_lora_rank = weights.kv_a_norm.shape[0]
    q_residual = rms_norm(
        project(normalized, weights.q_a), weights.q_a_norm, epsilon=epsilon
    )
    kv = project(normalized, weights.kv_a)
    latent = rms_norm(kv[:, :kv_lora_rank], weights.kv_a_norm, epsilon=epsilon)
    return PreparedAttention(q_residual, latent, kv[:, kv_lora_rank:])


def mla_attention(
    prepared: PreparedAttention,
    positions: jax.Array,
    kv_cache: jax.Array,
    selection: SelectedPositions,
    weights: AttentionWeights,
    *,
    contract: MlaNumericalContract,
    rope_table: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Write the rows' cache entries, attend over the selection; return ``(output, cache)``.

    ``kv_cache`` is ``[capacity, kv_lora_rank + rope_dim]``; ``selection`` holds
    each row's selected positions (any order) and live counts.
    """
    rows = positions.shape[0]
    heads = contract.num_heads
    nope, rope = contract.qk_nope_head_dim, contract.qk_rope_head_dim
    table_rows = jnp.take(rope_table, positions, axis=0)

    q = project(prepared.q_residual, weights.q_b).reshape(rows, heads, nope + rope)
    q_rope = rotate(q[..., nope:], table_rows)
    key_rope = rotate(prepared.key_rope_input[:, None, :], table_rows)[:, 0]
    cache_rows = jnp.concatenate((prepared.latent, key_rope), axis=-1)
    kv_cache = kv_cache.at[positions].set(cache_rows)

    kv_b = weights.kv_b.reshape(
        heads, nope + contract.v_head_dim, contract.kv_lora_rank
    )
    q_absorbed = einsum("rhd,hdc->rhc", q[..., :nope], kv_b[:, :nope])
    segment = gather_paged_selected_kv(
        kv_cache[None],
        jnp.zeros((rows, 1), jnp.int32),
        selection,
        (positions + 1).astype(jnp.int32),
    )
    attended = sparse_mla_attention(
        q_absorbed, q_rope, segment, contract=contract
    ).output
    values = einsum("rhc,hvc->rhv", attended, kv_b[:, nope:])
    output = project(values.reshape(rows, heads * contract.v_head_dim), weights.o)
    return output, kv_cache
