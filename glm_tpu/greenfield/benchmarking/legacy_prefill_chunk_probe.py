"""Chunk pipeline for the legacy-geometry layer-1 prompt-key probe.

Composes :mod:`legacy_prefill_geometry` into the complete layer-0 block for a
chunk of prompt rows (embedding -> input norm -> fused q-a/kv-a -> q-a norm ->
kv-a latent/rope -> per-owner q_b, absorbed attention, value, o_proj partials
-> DB533 row-0 BF16 association -> post-attention norm -> per-owner gate/up,
SwiGLU, down partials -> association -> layer-1 input norm) and the accepted
64-row prompt-key path for layers 0 and 1.  Rows >= 1 use a plain FP32 causal
softmax (not the legacy Pallas kernel); row 0 is exact by construction.

Pure JAX on one device; importable for CPU shape tests.  No claim.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.prefill_index import (
    materialize_stage_local_prefill_index_wk,
    physical_m64_prompt_index_key_chunk,
)
from ..kernels.reference.rmsnorm import rms_norm
from ..kernels.reference.rotary import apply_rotary_fp32_final_round
from ..kernels.stage_local import _strategy_nd_row0_bf16_reduce
from . import legacy_prefill_geometry as geo

WEIGHT_KEYS = (
    "input_norm0", "q_a_norm", "kv_a_norm", "post_norm", "input_norm1",
    "k_norm0_weight", "k_norm0_bias", "k_norm1_weight", "k_norm1_bias",
    "qkv_bits", "qkv_scale", "qb_bits", "qb_scale", "w_uk_t", "w_uv",
    "o_bits", "o_scale", "gu_bits", "gu_scale", "down_bits", "down_scale",
    "wk0_bits", "wk0_scale", "wk1_bits", "wk1_scale", "rope_table",
)


def legacy_geometry_chunk_pipeline(
    embedding: Any,
    positions: Any,
    weights: dict[str, Any],
    *,
    softmax_scale: float,
    contract: DsaNumericalContract = DsaNumericalContract(),
    row_association: Any = _strategy_nd_row0_bf16_reduce,
) -> dict[str, Any]:
    """Run the legacy-geometry layer-0 block and layer-0/1 prompt keys for a chunk."""

    missing = [key for key in WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError(f"chunk pipeline weights missing: {missing}")
    if embedding.ndim != 2 or embedding.shape[1] != geo.HIDDEN or embedding.dtype != jnp.bfloat16:
        raise ValueError("chunk embedding must be bf16[M, 6144]")
    rows = embedding.shape[0]
    if positions.shape != (rows,) or not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("chunk positions must be one integer per row")
    w = weights
    scale = jnp.float32(softmax_scale)

    n0 = geo.legacy_input_norm(embedding, w["input_norm0"])
    proj = geo.legacy_fused_qkv_a(n0, w["qkv_bits"], w["qkv_scale"])  # [32, M, 82]
    q_a = geo.legacy_q_a_norm(proj, w["q_a_norm"])  # [M, 2048]
    kv = geo.legacy_kv_a_lanes(proj)  # [M, 576]
    latent = rms_norm(kv[:, :512], w["kv_a_norm"], epsilon=1e-5)  # bf16 [M, 512]
    rope_rows = w["rope_table"][positions]  # bf16 [M, 64]
    cos = rope_rows[:, :32][:, None, :]
    sin = rope_rows[:, 32:][:, None, :]
    k_pe = apply_rotary_fp32_final_round(kv[:, 512:576][:, None, :], cos, sin, interleaved=True)[:, 0, :]
    causal = jnp.arange(rows)[None, :] > jnp.arange(rows)[:, None]  # key after query -> masked

    def owner_attention(owner: tuple[Any, ...]) -> Any:
        qb_b, qb_s, uk_t, uv, o_b, o_s = owner
        q = geo.legacy_convolution_rows(q_a, geo.legacy_dequantize_fp8(qb_b, qb_s)).reshape(rows, 2, 256)
        q_nope = q[:, :, :192]
        q_rope = apply_rotary_fp32_final_round(q[:, :, 192:], cos, sin, interleaved=True)
        ql = jnp.einsum(
            "mhp,hpl->hml", q_nope.astype(jnp.float32), uk_t.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        scores = jnp.einsum("hml,sl->hms", ql, latent.astype(jnp.float32), preferred_element_type=jnp.float32)
        scores = scores + jnp.einsum(
            "mhr,sr->hms", q_rope.astype(jnp.float32), k_pe.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        scores = jnp.where(causal[None, :, :], jnp.float32(-jnp.inf), scores * scale)
        probs = jax.nn.softmax(scores, axis=-1)
        o_latent = jnp.einsum(
            "hms,sl->hml", probs, latent.astype(jnp.float32), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        value = jnp.einsum("hml,hlv->mhv", o_latent, uv, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return geo.legacy_convolution_rows(value.reshape(rows, 512), geo.legacy_dequantize_fp8(o_b, o_s))

    partials = lax.map(
        owner_attention, (w["qb_bits"], w["qb_scale"], w["w_uk_t"], w["w_uv"], w["o_bits"], w["o_scale"])
    )  # [32, M, 6144]
    attention = geo.strategy_nd_row_association_reduce(partials, row_association)
    attention_pairwise = geo.bf16_pairwise_tree_reduce(partials)
    carried0, n_mlp = geo.legacy_post_attention_norm(attention, embedding, w["post_norm"])

    def owner_dense(owner: tuple[Any, ...]) -> Any:
        gu_b, gu_s, d_b, d_s = owner
        gate_up = geo.legacy_convolution_rows(n_mlp, geo.legacy_dequantize_fp8(gu_b, gu_s))
        hidden = geo.legacy_swiglu(gate_up, half_width=384)
        return geo.legacy_convolution_rows(hidden, geo.legacy_dequantize_fp8(d_b, d_s))

    dense_partials = lax.map(owner_dense, (w["gu_bits"], w["gu_scale"], w["down_bits"], w["down_scale"]))
    dense = geo.strategy_nd_row_association_reduce(dense_partials, row_association)
    carried1, n1 = geo.legacy_layer1_input_norm(dense, attention, embedding, w["input_norm1"])
    wk0 = materialize_stage_local_prefill_index_wk(w["wk0_bits"], w["wk0_scale"], contract=contract)
    wk1 = materialize_stage_local_prefill_index_wk(w["wk1_bits"], w["wk1_scale"], contract=contract)
    physical_rows = 64 if rows % 64 == 0 else rows
    keys0 = physical_m64_prompt_index_key_chunk(
        n0, positions, wk0, w["k_norm0_weight"], w["k_norm0_bias"], contract=contract, physical_rows=physical_rows
    )
    keys1 = physical_m64_prompt_index_key_chunk(
        n1, positions, wk1, w["k_norm1_weight"], w["k_norm1_bias"], contract=contract, physical_rows=physical_rows
    )
    return {
        "keys0": keys0.astype(jnp.bfloat16),
        "keys1": keys1.astype(jnp.bfloat16),
        "normalized0_row0": n0[0],
        "q_a_row0": q_a[0],
        "latent_row0": latent[0],
        "attention_row0": attention[0],
        "attention_pairwise_row0": attention_pairwise[0],
        "carried0_row0": carried0[0],
        "dense_row0": dense[0],
        "carried1_row0": carried1[0],
        "normalized1_row0": n1[0],
    }
