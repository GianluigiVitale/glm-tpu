"""Pure-JAX (jnp) reference of the GLM-5.2 DSA lightning-indexer forward math.

Stage-2a XLA-reference per docs/01-dsa-kernel-design.md §1.1 (math) and §1.2 (the
RoPE-interleave conflict). This module is the layout-parameterized transcription
that the parity harness (parity/test_indexer_reference.py) proves faithful to the
HF torch math, and that the rope experiment (parity/glm_indexer_rope_experiment.py)
runs against real GLM-5.2-FP8 weights.

Verified against BOTH references:

* HF: ~/glm-tpu/reference/modeling_glm_moe_dsa.py (GlmMoeDsaIndexer, lines 166-262)
    - wq_b/wk/k_norm/weights_proj construction + softmax_scale: :191-195
      (k_norm is nn.LayerNorm(head_dim, eps=1e-6) WITH bias, :193)
    - q = wq_b(q_resid).view(B, S, H, D): :231-232
    - rope-FIRST split of the 128-dim head: q_rot = q[..., :qk_rope_head_dim]: :233
      (k likewise, :236)
    - k = k_norm(wk(hidden_states)): :235
    - rope applied to the rope slice: :239 (HF hard-codes the NON-interleaved
      half-split `apply_rotary_pos_emb`, :133-163, with the comment "The indexer
      uses NON-interleaved (half-split) RoPE"; the interleaved alternative is HF's
      own `apply_rotary_pos_emb_interleave`, :302-338)
    - scores = relu(q.float() @ k.float()^T * head_dim**-0.5): :246-247 (fp32)
    - weights = weights_proj(h).float() * n_heads**-0.5: :250
    - index_scores = weights @ scores (weighted head-sum): :251
    - causal mask (key_pos > query_pos -> -inf): :254-259
    - topk = min(index_topk, S); int32 indices: :261-262
    - cos/sin: GlmMoeDsaRotaryEmbedding :117-130 with
      inv_freq = base**(-arange(0, dim, 2)/dim), :106-115, where the config class
      points dim at qk_rope_head_dim (configuration_glm_moe_dsa.py:148-150).

* vLLM GPU: /home/gianl/vllm-build/vllm/model_executor/models/deepseek_v2.py
  (Indexer.forward, lines 678-742)
    - q = wq_b(qr).view(-1, n_head, head_dim): :681-682
    - rope-FIRST split (q_pe = q[..., :rope_dim]): :702-705
    - k = k_norm(wk(h)) (fused wk+weights_proj GEMM, split): :706-713
    - rope layout: the indexer rotary_emb is built with
      is_neox_style = not getattr(config, "indexer_rope_interleave", False)
      (deepseek_v2.py:1003-1008) -> INTERLEAVED for GLM-5.2
      (config indexer_rope_interleave: true); the non-neox apply is
      rotary_embedding/common.py:145-185 (pairs (0,1),(2,3),... via x[..., ::2] /
      x[..., 1::2], re-interleaved output), inv_freq identical to HF
      (rotary_embedding/base.py:81-104).
    - scale folding: weights = weights_proj_out * q_scale * softmax_scale *
      n_head**-0.5 applied POST-relu (:737-740) — legal because
      relu(a*x) = a*relu(x) for a > 0 (design doc §1.1). This reference applies
      softmax_scale PRE-relu like HF (:246-247); bit-equivalent in fp32.
    - Hadamard rotation + fp8 scoring (HF modeling.py:211-215 documents the
      equivalence: the Hadamard transform is orthogonal so Hq·Hk = q·k, and fp8
      is a precision optimization) are SKIPPED here — this reference computes
      the bf16/fp32-equivalent scores directly, exactly like the HF module.

RoPE layout note (docs/01 §1.2): `interleaved=True` reproduces the vLLM behavior
(pairs (2i, 2i+1) each rotated by angle pos*theta^(-2i/64)); `interleaved=False`
reproduces HF's half-split behavior (pairs (i, i+32)). HF's own
`apply_rotary_pos_emb_interleave` (modeling.py:302-338) emits the rotated pairs
in DE-interleaved order (cat([even', odd'])) while vLLM re-interleaves
(stack+flatten, common.py:178-181); the two differ only by a fixed permutation
applied identically to q and k, so all q·k scores — the only thing the indexer
consumes — are identical. The parity test asserts this at the score level.

All math is fp32 (HF computes scores in fp32, modeling.py:246; fp32 score
accumulation is also the mla.v2 house style — design doc §1.1).

Weight layout convention: all projection weights are passed in the torch
nn.Linear layout `[out_features, in_features]` (exactly what the checkpoint
stores); y = x @ W^T.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax

_NEG_INF = float("-inf")


def rope_cos_sin(positions, rope_dim: int, rope_theta: float):
    """cos/sin tables for the indexer rope slice, one angle per rotation pair.

    inv_freq = rope_theta**(-arange(0, rope_dim, 2)/rope_dim)  — identical in both
    references (HF modeling.py:106-115 with dim = qk_rope_head_dim per
    configuration_glm_moe_dsa.py:148-150; vLLM rotary_embedding/base.py:81-92).
    HF then materializes emb = cat(freqs, freqs) (modeling.py:126) so its cos/sin
    are [T, rope_dim]; the first half holds the per-pair angle (modeling.py:330).
    We return only the per-pair half, shape [T, rope_dim//2] fp32.
    """
    positions = jnp.asarray(positions, dtype=jnp.float32)
    half = rope_dim // 2
    inv_freq = rope_theta ** (
        -jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim
    )  # [half]
    freqs = positions[:, None] * inv_freq[None, :]  # [T, half]
    assert freqs.shape[-1] == half
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin, *, interleaved: bool):
    """Rotate the rope slice `x [..., rope_dim]` under the given pairing layout.

    cos/sin: [..., rope_dim//2], broadcastable against x's leading dims.

    interleaved=False — HF half-split (`apply_rotary_pos_emb` via rotate_half,
    modeling.py:133-163): pairs (i, i + rope_dim//2), each rotated by angle_i;
    output cat([x1', x2']) in place.

    interleaved=True — vLLM non-neox (rotary_embedding/common.py:169-181):
    pairs (2i, 2i+1), each rotated by angle_i; output re-interleaved via
    stack(..., -1).flatten(-2). (HF's `apply_rotary_pos_emb_interleave`,
    modeling.py:333-337, emits the same rotated pairs cat'd de-interleaved —
    a fixed permutation shared by q and k, so scores are unchanged; see module
    docstring.)
    """
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return jnp.stack([o1, o2], axis=-1).reshape(x.shape)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # rotate_half formulation (modeling.py:133-137, :161-162):
    #   out = x*cat(cos,cos) + cat(-x2, x1)*cat(sin,sin)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate([o1, o2], axis=-1)


def indexer_scores(
    h_TD,
    q_resid_TQ,
    wq_b,
    wk,
    k_norm_w,
    k_norm_b,
    weights_proj,
    positions,
    rope_theta,
    *,
    interleaved: bool,
    rope_dim: int = 64,
):
    """Raw (unmasked, signed) indexer scores `[T, T]`, fp32.

    Args:
      h_TD:        [T, hidden] pre-attention hidden states (post input_layernorm)
                   — what both references feed as `hidden_states`
                   (HF modeling.py:652+446-448; vLLM mla_attention.py:343-345).
      q_resid_TQ:  [T, q_lora_rank] = q_a_layernorm(q_a_proj(h))
                   (HF modeling.py:423, passed at :446-448; vLLM `qr`, :678).
      wq_b:        [n_heads*head_dim, q_lora_rank]   (modeling.py:191)
      wk:          [head_dim, hidden]                (modeling.py:192)
      k_norm_w/b:  [head_dim] LayerNorm weight/bias, eps=1e-6 (modeling.py:193)
      weights_proj:[n_heads, hidden]                 (modeling.py:194)
      positions:   [T] int absolute positions (rope only; causality is applied
                   in `topk_indices`, mirroring HF where the mask is added after
                   scoring, modeling.py:254-259).
      rope_theta:  scalar (GLM-5.2: 8e6).
      interleaved: rope pairing layout — see module docstring / docs/01 §1.2.
      rope_dim:    qk_rope_head_dim (GLM-5.2: 64).

    Returns [T, T] fp32 scores; scores[t, s] = Σ_h w[t,h]·relu(q[t,h]·k[s]·D^-0.5)
    (modeling.py:246-251). `w` is a plain linear head and can be NEGATIVE —
    index_scores are signed (design doc §1.1).
    """
    h = jnp.asarray(h_TD, dtype=jnp.float32)
    q_resid = jnp.asarray(q_resid_TQ, dtype=jnp.float32)
    wq_b = jnp.asarray(wq_b, dtype=jnp.float32)
    wk = jnp.asarray(wk, dtype=jnp.float32)
    weights_proj = jnp.asarray(weights_proj, dtype=jnp.float32)

    head_dim = wk.shape[0]
    n_heads = wq_b.shape[0] // head_dim
    T = h.shape[0]

    # q = wq_b(q_resid) -> [T, H, D]                       (modeling.py:231-232)
    q = (q_resid @ wq_b.T).reshape(T, n_heads, head_dim)

    # k = LayerNorm(wk(h), eps=1e-6) with bias             (modeling.py:235, :193)
    k_pre = h @ wk.T  # [T, D]
    mean = jnp.mean(k_pre, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(k_pre - mean), axis=-1, keepdims=True)  # biased
    k = (k_pre - mean) / jnp.sqrt(var + 1e-6)
    k = k * jnp.asarray(k_norm_w, jnp.float32) + jnp.asarray(k_norm_b, jnp.float32)

    # rope on the FIRST rope_dim dims of the head          (modeling.py:233,236,239;
    #                                                       deepseek_v2.py:702-712)
    cos, sin = rope_cos_sin(positions, rope_dim, float(rope_theta))  # [T, rope_dim//2]
    q_rot = apply_rope(q[:, :, :rope_dim], cos[:, None, :], sin[:, None, :],
                       interleaved=interleaved)
    k_rot = apply_rope(k[:, :rope_dim], cos, sin, interleaved=interleaved)
    q = jnp.concatenate([q_rot, q[:, :, rope_dim:]], axis=-1)
    k = jnp.concatenate([k_rot, k[:, rope_dim:]], axis=-1)

    # scores[t,h,s] = relu(q[t,h]·k[s] * D**-0.5)          (modeling.py:246-247)
    softmax_scale = head_dim ** -0.5  # modeling.py:195
    scores_ths = jnp.maximum(
        jnp.einsum("thd,sd->ths", q, k,
                   preferred_element_type=jnp.float32) * softmax_scale,
        0.0,
    )

    # w[t,h] = weights_proj(h)[t,h] * H**-0.5              (modeling.py:250)
    w = (h @ weights_proj.T) * (n_heads ** -0.5)  # [T, H]

    # index_scores[t,s] = Σ_h w[t,h]·scores[t,h,s]         (modeling.py:251)
    return jnp.einsum("th,ths->ts", w, scores_ths,
                      preferred_element_type=jnp.float32)


def causal_mask_scores(scores, query_positions=None, key_positions=None):
    """Apply the HF causal mask (key_pos > query_pos -> -inf), modeling.py:254-259."""
    T, S = scores.shape
    qpos = jnp.arange(T) if query_positions is None else jnp.asarray(query_positions)
    kpos = jnp.arange(S) if key_positions is None else jnp.asarray(key_positions)
    return jnp.where(kpos[None, :] > qpos[:, None], _NEG_INF, scores)


def topk_indices(scores, k: int, causal: bool = True):
    """Top-k key indices per query, int32 `[T, min(k, S)]`.

    Mirrors HF modeling.py:254-262: causal mask (key_pos > query_pos -> -inf,
    assuming query t of the square score map sits at position t), then
    topk = min(k, S) over the masked row, indices as int32. Exact
    `lax.top_k` — `jax.lax.approx_max_k` is banned (design doc §1.4).

    Tie semantics: `lax.top_k` and `torch.topk` order equal values differently;
    parity gates must compare selected SETS with the tie group at the k-th
    boundary treated as interchangeable (design doc §1.4 / gate S). Rows whose
    valid (causal) prefix is shorter than k fill the remainder with arbitrary
    members of the -inf tie group, exactly like torch.topk on the HF side.
    """
    if causal:
        scores = causal_mask_scores(scores)
    kk = min(int(k), scores.shape[-1])  # modeling.py:261
    _, idx = lax.top_k(scores, kk)
    return idx.astype(jnp.int32)  # modeling.py:262
