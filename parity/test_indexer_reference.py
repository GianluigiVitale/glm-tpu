"""CPU parity test: jnp GLM-5.2 DSA indexer reference vs the HF torch math.

Machine-gated (exit code), per docs/01-dsa-kernel-design.md §7 (2a) / §4 gate S:

  (a) score parity vs a torch reimplementation lifted from the HF modeling file
      (transformers.models.glm_moe_dsa.modeling_glm_moe_dsa — asserted
      byte-identical to ~/glm-tpu/reference/modeling_glm_moe_dsa.py), random
      weights, T=64, BOTH rope layouts with each side using the SAME layout
      (proves the jnp transcription is faithful; it does NOT settle which layout
      is right — that is parity/glm_indexer_rope_experiment.py). fp32, atol 1e-5.
      For the non-interleaved layout the actual `GlmMoeDsaIndexer` nn.Module
      forward (the layout it hard-codes) is run as a third implementation and
      its selected sets compared too.
  (b) selected-set equality vs torch.topk (sets modulo boundary tie groups —
      design doc §1.4 tie semantics).
  (c) determinism + engineered-tie test on `topk_indices`.

CPU-ONLY: JAX_PLATFORMS=cpu is forced before jax import — this test must never
initialize a TPU.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # BEFORE jax import; never touch TPU

import inspect

import numpy as np
import torch

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_indexer_reference import (  # noqa: E402
    causal_mask_scores,
    indexer_scores,
    topk_indices,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- real GLM-5.2 indexer dims (configs/glm-5.2-fp8-config.json) ----
HIDDEN = 6144
Q_LORA = 2048
N_HEADS = 32       # index_n_heads
HEAD_DIM = 128     # index_head_dim
ROPE_DIM = 64      # qk_rope_head_dim
ROPE_THETA = 8.0e6
T = 64


def _load_hf_module():
    """Import the HF modeling module; prefer reference/, else installed twin.

    reference/modeling_glm_moe_dsa.py uses package-relative imports (`from ...`)
    so it is not importable as a standalone file; the installed
    transformers.models.glm_moe_dsa module is used instead and its
    GlmMoeDsaIndexer + both rope apply functions are asserted byte-identical to
    the reference/ copy, so the torch side IS the reference file's math.
    """
    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf

    ref_src = open(os.path.join(REPO, "reference", "modeling_glm_moe_dsa.py")).read()
    for obj in (hf.GlmMoeDsaIndexer, hf.apply_rotary_pos_emb,
                hf.apply_rotary_pos_emb_interleave, hf.rotate_half):
        src = inspect.getsource(obj)
        assert src in ref_src, (
            f"installed transformers {obj.__name__} diverges from "
            "reference/modeling_glm_moe_dsa.py — refresh the reference copy")
    return hf


HF = _load_hf_module()


def hf_cos_sin(positions_t: torch.Tensor):
    """cos/sin exactly as GlmMoeDsaRotaryEmbedding.forward (modeling.py:106-130).

    dim = qk_rope_head_dim per configuration_glm_moe_dsa.py:148-150;
    emb = cat(freqs, freqs) -> [1, T, ROPE_DIM]; attention_scaling = 1
    (rope_type "default").
    """
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    freqs = positions_t.float()[:, None] * inv_freq[None, :]  # [T, 32]
    emb = torch.cat((freqs, freqs), dim=-1)                   # [T, 64]
    return emb.cos()[None], emb.sin()[None]                   # [1, T, 64]


def torch_indexer_scores(h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj,
                         positions, *, interleaved: bool):
    """Faithful transcription of GlmMoeDsaIndexer.forward (modeling.py:229-251),
    fp32, batch=1, rope layout selected via HF's OWN apply functions
    (apply_rotary_pos_emb modeling.py:133-163 — what the indexer hard-codes at
    :239; apply_rotary_pos_emb_interleave :302-338 — the vLLM-behavior layout,
    deepseek_v2.py:1003-1008). Returns raw signed scores [T, T] (mask + topk are
    applied separately, mirroring modeling.py:254-262)."""
    B, S = 1, h.shape[0]
    hB = h[None]                                            # [1, T, HIDDEN]
    cos, sin = hf_cos_sin(positions)

    q = torch.nn.functional.linear(q_resid[None], wq_b)     # :231  [1,T,H*D]
    q = q.view(B, S, N_HEADS, HEAD_DIM)                     # :232
    q_rot, q_pass = torch.split(q, [ROPE_DIM, HEAD_DIM - ROPE_DIM], dim=-1)  # :233

    k = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(hB, wk), (HEAD_DIM,),
        k_norm_w, k_norm_b, eps=1e-6).unsqueeze(2)          # :235 (:193 eps/bias)
    k_rot, k_pass = torch.split(k, [ROPE_DIM, HEAD_DIM - ROPE_DIM], dim=-1)  # :236

    apply = (HF.apply_rotary_pos_emb_interleave if interleaved
             else HF.apply_rotary_pos_emb)                  # :239 vs :302
    q_rot, k_rot = apply(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)                  # :240
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)       # :241

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
    scores = scores * (HEAD_DIM ** -0.5)                    # :246 (:195)
    scores = torch.nn.functional.relu(scores)               # :247

    weights = torch.nn.functional.linear(hB, weights_proj).float() \
        * (N_HEADS ** -0.5)                                 # :250
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)  # :251
    return index_scores[0]                                  # [T, T]


def torch_masked_scores(raw_scores, positions):
    """Causal mask exactly as modeling.py:257-259."""
    key_positions = torch.arange(raw_scores.shape[-1])
    causal = key_positions[None, :] > positions[:, None]
    return raw_scores.masked_fill(causal, float("-inf"))


def assert_sets_equal_modulo_ties(masked_row: np.ndarray, idx_a, idx_b, k, row):
    """Gate-S set comparison (design doc §1.4/§4): indices with score strictly
    above the k-th value are mandatory in both sets; members of the boundary tie
    group are interchangeable."""
    k = min(k, masked_row.shape[0])
    sa, sb = set(map(int, idx_a)), set(map(int, idx_b))
    assert len(sa) == len(idx_a) and len(sb) == len(idx_b), f"dup indices row {row}"
    kth = np.sort(masked_row)[::-1][k - 1]
    if np.isneginf(kth):
        mandatory = set(np.nonzero(masked_row > kth)[0].tolist())
        optional = set(np.nonzero(np.isneginf(masked_row))[0].tolist())
    else:
        mandatory = set(np.nonzero(masked_row > kth)[0].tolist())
        optional = set(np.nonzero(masked_row == kth)[0].tolist())
    for name, s in (("jnp", sa), ("torch", sb)):
        assert mandatory <= s, f"row {row}: {name} missing mandatory {mandatory - s}"
        extra = s - mandatory
        assert extra <= optional, f"row {row}: {name} selected non-tie {extra - optional}"


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    scale = 0.02
    h = rng.standard_normal((T, HIDDEN)).astype(np.float32)
    q_resid = rng.standard_normal((T, Q_LORA)).astype(np.float32)
    wq_b = (rng.standard_normal((N_HEADS * HEAD_DIM, Q_LORA)) * scale).astype(np.float32)
    wk = (rng.standard_normal((HEAD_DIM, HIDDEN)) * scale).astype(np.float32)
    k_norm_w = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
    k_norm_b = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
    weights_proj = (rng.standard_normal((N_HEADS, HIDDEN)) * scale).astype(np.float32)
    positions = np.arange(T, dtype=np.int64)
    return h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj, positions


def run_module_forward(h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj,
                       positions):
    """The actual GlmMoeDsaIndexer nn.Module forward (non-interleaved rope only —
    the layout it hard-codes at modeling.py:239). Returns topk indices [T, topk]."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    cfg = GlmMoeDsaConfig(
        hidden_size=HIDDEN, q_lora_rank=Q_LORA, index_n_heads=N_HEADS,
        index_head_dim=HEAD_DIM, qk_rope_head_dim=ROPE_DIM, index_topk=16,
        num_hidden_layers=1, first_k_dense_replace=1,
        rope_parameters={"rope_theta": ROPE_THETA, "rope_type": "default"},
    )
    with torch.no_grad():
        mod = HF.GlmMoeDsaIndexer(cfg, layer_idx=0).float()
        mod.wq_b.weight.copy_(torch.from_numpy(wq_b))
        mod.wk.weight.copy_(torch.from_numpy(wk))
        mod.k_norm.weight.copy_(torch.from_numpy(k_norm_w))
        mod.k_norm.bias.copy_(torch.from_numpy(k_norm_b))
        mod.weights_proj.weight.copy_(torch.from_numpy(weights_proj))
        pos = torch.from_numpy(positions)[None]
        cos, sin = hf_cos_sin(pos[0])
        out = mod(torch.from_numpy(h)[None], torch.from_numpy(q_resid)[None],
                  (cos, sin), attention_mask=None, position_ids=pos)
    return out[0].numpy()


def test_a_score_parity():
    h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj, positions = make_inputs()
    results = {}
    for interleaved in (False, True):
        s_jnp = np.asarray(indexer_scores(
            h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj, positions,
            ROPE_THETA, interleaved=interleaved))
        with torch.no_grad():
            s_tor = torch_indexer_scores(
                *(torch.from_numpy(a) for a in
                  (h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj)),
                torch.from_numpy(positions), interleaved=interleaved).numpy()
        diff = np.max(np.abs(s_jnp - s_tor))
        mag = np.max(np.abs(s_tor))
        assert s_jnp.dtype == np.float32
        assert diff <= 1e-5, (
            f"score parity FAIL interleaved={interleaved}: max|Δ|={diff:.3e}")
        results[interleaved] = (s_jnp, s_tor)
        print(f"  (a) interleaved={interleaved!s:5} max|Δ|={diff:.3e} "
              f"(max|score|={mag:.3f})  PASS")
    # The two layouts must genuinely differ (else the experiment cannot discriminate)
    x_diff = np.max(np.abs(results[False][0] - results[True][0]))
    assert x_diff > 1e-2, f"layouts indistinguishable at random weights ({x_diff:.3e})"
    print(f"  (a) cross-layout max|Δ|={x_diff:.3f} (layouts distinguishable)  PASS")
    return results


def test_b_topk_sets(score_pairs):
    h, q_resid, *_rest, positions = make_inputs()
    pos_t = torch.from_numpy(positions)
    for interleaved, (s_jnp, s_tor) in score_pairs.items():
        masked_t = torch_masked_scores(torch.from_numpy(s_tor), pos_t)
        masked_np = masked_t.numpy()
        for k in (8, 16, T):
            idx_j = np.asarray(topk_indices(jnp.asarray(s_jnp), k, causal=True))
            idx_t = masked_t.topk(min(k, T), dim=-1).indices.to(torch.int32).numpy()
            # modeling.py:261-262
            for row in range(T):
                assert_sets_equal_modulo_ties(
                    masked_np[row], idx_j[row], idx_t[row], k, row)
        print(f"  (b) interleaved={interleaved!s:5} selected sets == torch.topk "
              f"(k=8,16,{T}, modulo boundary ties)  PASS")
    # third implementation: the real HF nn.Module (non-interleaved layout)
    inputs = make_inputs()
    idx_mod = run_module_forward(*inputs)  # index_topk=16 in the config
    s_jnp = score_pairs[False][0]
    masked_np = np.asarray(causal_mask_scores(jnp.asarray(s_jnp)))
    idx_j = np.asarray(topk_indices(jnp.asarray(s_jnp), 16, causal=True))
    for row in range(T):
        assert_sets_equal_modulo_ties(masked_np[row], idx_j[row], idx_mod[row], 16, row)
    print("  (b) vs actual GlmMoeDsaIndexer.forward (non-interleaved, k=16)  PASS")


def test_c_determinism_and_ties():
    h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj, positions = make_inputs(1)
    s = np.asarray(indexer_scores(h, q_resid, wq_b, wk, k_norm_w, k_norm_b,
                                  weights_proj, positions, ROPE_THETA,
                                  interleaved=True))
    # engineer ties: coarse-quantize so many equal values straddle the boundary
    s_tied = np.round(s * 2.0) / 2.0
    k = 8
    idx1 = np.asarray(topk_indices(jnp.asarray(s_tied), k))
    idx2 = np.asarray(topk_indices(jnp.asarray(s_tied), k))
    idx3 = np.asarray(jax.jit(lambda x: topk_indices(x, k))(jnp.asarray(s_tied)))
    assert np.array_equal(idx1, idx2), "topk_indices non-deterministic across calls"
    assert np.array_equal(idx1, idx3), "topk_indices jit vs eager mismatch"
    masked = np.array(causal_mask_scores(jnp.asarray(s_tied)))  # writable copy
    n_boundary_ties = 0
    idx_t = torch.from_numpy(masked).topk(k, dim=-1).indices.numpy()
    for row in range(T):
        assert_sets_equal_modulo_ties(masked[row], idx1[row], idx_t[row], k, row)
        # selected-VALUE multisets must agree exactly even where index sets differ
        va = np.sort(masked[row][idx1[row]])[::-1]
        vb = np.sort(masked[row][idx_t[row]])[::-1]
        assert np.array_equal(va, vb), f"row {row}: selected value multisets differ"
        if len(set(map(int, idx1[row]))) and \
           set(map(int, idx1[row])) != set(map(int, idx_t[row])):
            n_boundary_ties += 1
    kth = np.sort(masked, axis=-1)[:, ::-1][:, k - 1]
    tie_rows = int(np.sum(np.sum(masked == kth[:, None], axis=-1) > 1))
    assert tie_rows > 0, "tie engineering failed — test exercises no tie rows"
    print(f"  (c) deterministic (2x eager + jit identical); {tie_rows} tie rows "
          f"exercised, {n_boundary_ties} rows differ from torch only inside the "
          f"boundary tie group  PASS")


def test_d_precision_pinned():
    """Precision audit (the A2 lesson, kernelprobe-2b on-metal round): on TPU
    the DEFAULT matmul precision computes fp32 dots via bf16 MXU passes
    (~4.5e-3 relative self-error) and `preferred_element_type=f32` does NOT
    prevent it. The oracle must pin HIGHEST *internally*, never rely on the
    caller's context. Lower indexer_scores under a HOSTILE caller context
    ('bfloat16') and assert every dot_general in the HLO carries HIGHEST."""
    h, q_resid, wq_b, wk, k_norm_w, k_norm_b, weights_proj, positions = \
        make_inputs()

    def fn(h_, q_):
        return indexer_scores(h_, q_, wq_b, wk, k_norm_w, k_norm_b,
                              weights_proj, positions, ROPE_THETA,
                              interleaved=True)

    with jax.default_matmul_precision("bfloat16"):  # hostile caller context
        hlo = jax.jit(fn).lower(jnp.asarray(h),
                                jnp.asarray(q_resid)).as_text()
    dots = [ln.strip() for ln in hlo.splitlines() if "dot_general" in ln]
    # 5 matmuls: wq_b proj, wk proj, q·k einsum, weights_proj, head-sum.
    assert len(dots) >= 5, f"expected >=5 dot_generals, found {len(dots)}"
    bad = [ln for ln in dots if "HIGHEST" not in ln]
    assert not bad, (
        "indexer_scores lowered dot(s) WITHOUT pinned HIGHEST precision (the "
        f"A2 bf16-MXU hazard would skew the oracle on TPU): {bad}")
    print(f"  (d) all {len(dots)} dot_generals pinned HIGHEST under a "
          "hostile 'bfloat16' caller context  PASS")


def main():
    assert jax.default_backend() == "cpu", (
        f"backend={jax.default_backend()} — this test must run on CPU only")
    torch.manual_seed(0)
    print(f"GLM-5.2 DSA indexer jnp-reference parity (T={T}, fp32, CPU)")
    pairs = test_a_score_parity()
    test_b_topk_sets(pairs)
    test_c_determinism_and_ties()
    test_d_precision_pinned()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
