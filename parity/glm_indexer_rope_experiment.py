#!/usr/bin/env python
"""RoPE-layout experiment for the GLM-5.2 DSA indexer, on REAL weights.

Settles docs/01-dsa-kernel-design.md §1.2 (interleaved vs non-interleaved indexer
rope) as far as it can be settled offline, by running the experiments the design
doc specifies:

  E1 — score-vs-attention agreement (the offline discriminator; doc §1.2):
     load layer 0's real weights (a `full` indexer layer) from the staged
     GLM-5.2-FP8 checkpoint, run ~1K tokens of natural text, compute
     `index_scores` under BOTH rope layouts, and compare each layout's
     scores/selected sets against the SAME layer's dense MLA attention row-mass
     per key (sum over the 64 heads' softmax rows; kv from kv_b_proj) — the DSA
     indexer is trained against the attention distribution, so the correct
     layout must show markedly higher agreement. Metrics (doc §1.2 E1): recall
     of indexer-top-n vs attention-top-n (n ∈ {64, 256}), per-row Spearman, plus
     attention-mass capture of the indexer-top-n set.

  E2 — weight forensics (corroborating only; doc §1.2): within-pair similarity
     statistics of the 64 rope rows of wk / wq_b under the two pairing
     hypotheses (interleaved pairs (2i, 2i+1) vs half-split pairs (i, i+32)).
     Tiebreaker signal, never sole evidence.

  E3 — behavioral at ctx > 2048 (passkey + logprob-divergence-from-dense under
     both layouts) NEEDS the 2a sparse path on the pod and is NOT run here; it
     is the residual uncertainty this script documents in its verdict.

Why score-level works at short ctx: at S <= 2048 topk selects every token, so no
OUTPUT-level test can discriminate (doc §1.2 "critical constraint") — but the
scores and top-n selected sets themselves discriminate at any ctx, which is
exactly what E1 inspects.

Weights: layer 0 of gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/ (everything
needed lives in model-00001-of-00141.safetensors, ~5 GB — auto-downloaded to
--weights-dir if absent). FP8 tensors are dequantized with their 128x128-block
`weight_scale_inv` (config quantization_config.weight_block_size = [128, 128],
same GroupShape(128,128) dequant as vLLM deepseek_v2.py:746-779).

CPU-ONLY: JAX_PLATFORMS=cpu is forced before jax import; never initializes TPU.

Usage:
  JAX_PLATFORMS=cpu python parity/glm_indexer_rope_experiment.py \
      [--seq-len 1024] [--weights-dir DIR]
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # BEFORE jax import; never touch TPU

import argparse
import json
import subprocess
import sys

import numpy as np
import torch

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_indexer_reference import (  # noqa: E402
    apply_rope,
    causal_mask_scores,
    indexer_scores,
    rope_cos_sin,
)

GCS_PREFIX = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
SHARD = "model-00001-of-00141.safetensors"
DEFAULT_WEIGHTS_DIR = (
    "/tmp/claude-2001/-home-gianl/8712a40e-7bbe-436d-b0ad-d676cbd502d5/"
    "scratchpad/glm52"
)

# ---- GLM-5.2 dims (configs/glm-5.2-fp8-config.json) ----
HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
N_HEADS_ATTN = 64
QK_NOPE = 192
QK_ROPE = 64
QK_HEAD = 256          # qk_nope + qk_rope
V_HEAD = 256
IDX_HEADS = 32
IDX_HEAD_DIM = 128
ROPE_THETA = 8.0e6
RMS_EPS_LAYER = 1e-5   # config rms_norm_eps (input_layernorm, modeling.py:637)
RMS_EPS_LORA = 1e-6    # q_a/kv_a_layernorm use the RMSNorm default (modeling.py:377,385)
FP8_BLOCK = 128

L0 = "model.layers.0."
ATT = L0 + "self_attn."
NEEDED = [
    "model.embed_tokens.weight",
    L0 + "input_layernorm.weight",
    ATT + "q_a_proj.weight", ATT + "q_a_proj.weight_scale_inv",
    ATT + "q_a_layernorm.weight",
    ATT + "q_b_proj.weight", ATT + "q_b_proj.weight_scale_inv",
    ATT + "kv_a_proj_with_mqa.weight", ATT + "kv_a_proj_with_mqa.weight_scale_inv",
    ATT + "kv_a_layernorm.weight",
    ATT + "kv_b_proj.weight", ATT + "kv_b_proj.weight_scale_inv",
    ATT + "indexer.wq_b.weight", ATT + "indexer.wq_b.weight_scale_inv",
    ATT + "indexer.wk.weight", ATT + "indexer.wk.weight_scale_inv",
    ATT + "indexer.k_norm.weight", ATT + "indexer.k_norm.bias",
    ATT + "indexer.weights_proj.weight",
]

# ~1.1K words of original natural-language text with recurring entities and
# genuine long-range references (the E1 input; repetition-free so the attention
# pattern is retrieval-like rather than induction-degenerate).
TEXT = """
The lighthouse keeper of Almadora was named Teodor Vasquez, and for thirty-one
years he had climbed the same two hundred and twelve steps every evening to
light the great lamp. Almadora itself was a town of fishermen and salt farmers,
pressed between the cliffs and a sea that changed color with the seasons: green
in spring, iron-grey in winter, and in the last weeks of summer a deep wine
red that the old people blamed on the algae and the young people blamed on
nothing at all, because they had stopped asking. Teodor kept a logbook. In it
he recorded the ships that passed, the storms that came, and the small
irregularities of the light itself, which he had learned to read the way a
doctor reads a pulse. On the fourth of October he wrote a single line that
would later be quoted in every account of what happened: the lamp burns blue
when the wind comes from the south.

His granddaughter, Ines, was the first to take the observation seriously. She
was twenty-three, had studied optical engineering in the capital, and had come
back to Almadora not out of nostalgia but because her scholarship had ended and
the rent in the city had not. Ines borrowed a spectrometer from her old
laboratory, carried it up the two hundred and twelve steps, and pointed it at
the flame. What she found contradicted everything she had been taught about
combustion. The blue component was not a property of the fuel. It was a
property of the air, and it appeared only when the barometer fell faster than
three millibars in an hour. She wrote to her former professor, a patient man
named Doctor Halvorsen, who replied with a list of eleven possible
explanations, ten of which she eliminated in a single weekend of measurements.

The eleventh explanation was the one nobody wanted. There was a mineral in the
cliffs, an unstable silicate the mining company had catalogued and abandoned
in the nineteen sixties, and when the pressure dropped, the cliff face
exhaled. The gas was harmless in the open air and invisible by daylight, but
it fluoresced faintly in the heat of the lamp, and it meant the rock was
fracturing. Halvorsen took the night train down from the capital, verified the
spectra himself, and stood for a long time at the base of the tower with his
hand flat against the stone, as if the cliff could be consoled. He estimated
the town had between four and nine years before the western terrace failed.

What makes the story worth telling is not the geology but the arithmetic of
belief. The town council met three times. At the first meeting they voted to
commission a second opinion, which agreed with the first. At the second
meeting they voted to commission a third opinion, which agreed with the
second. At the third meeting, the mayor, a former schoolteacher named Rosa
Ibanez, did something unusual: she read Teodor's logbook aloud, beginning with
the entry from the fourth of October, and then read the spectrometer tables
Ines had compiled, and then asked the council to explain, line by line, which
number they believed was lying. Nobody could. The evacuation of the western
terrace began the following spring, unhurried and stubborn and complete, and
when the rock finally let go, six years later, it took forty-one empty houses
and not one life.

Engineers still study Almadora, though rarely for the reasons its residents
expect. The interesting failure was never the cliff; cliffs fail on schedule
once you know what they are made of. The interesting failure was the decade of
signals that had been recorded and ignored: the lamp had burned blue in south
winds since before Teodor was born, and three generations of keepers had
written it down as a curiosity, the way one writes down an unusually large
moon. Data, it turns out, is not knowledge until someone is frightened enough
to cross-reference it. Ines said as much at the inquiry. When the chairman
asked her why the anomaly had waited seventy years for an explanation, she
answered that instruments do not ask questions, and that the town had owned a
spectrometer for exactly eleven days.

There is a coda, mostly forgotten. The lamp itself, the original oil-burning
mechanism with its counterweighted clockwork, was moved to the maritime museum
in the capital, where Doctor Halvorsen would bring his first-year students to
see it. He liked to end the visit with a question. If the keeper's logbook
recorded the blue flame for seventy years, and the physics had been understood
for fifty, and the spectrometer had existed for eleven days before the truth
came out, then which instrument actually made the discovery? The students
usually said the spectrometer. Halvorsen would shake his head and tap the
glass case that held the logbook. Patience, he told them, is also an
instrument. It has the widest aperture and the slowest shutter, and it is the
only one that works when you do not yet know what you are looking for. Ines,
for her part, stayed in Almadora. She rebuilt the light with a modern lens,
kept her grandfather's habit of one line per evening, and on quiet nights,
when the wind swung to the south and the barometer began to fall, she would
watch the new lamp burn its steady, incorruptible white, and miss, just a
little, the color of the warning.
""".strip()


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------

def ensure_files(d: str):
    os.makedirs(d, exist_ok=True)
    for f in (SHARD, "tokenizer.json", "model.safetensors.index.json"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"[setup] downloading {f} from {GCS_PREFIX} ...")
            subprocess.run(
                ["gcloud", "storage", "cp", f"{GCS_PREFIX}/{f}", p], check=True)
    # verify every needed param lives in the shard we have (index json maps
    # params -> shards)
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    missing = [k for k in NEEDED if wm.get(k) != SHARD]
    assert not missing, f"params not in {SHARD}: {missing} — shard map changed?"


def dequant_fp8(w8: torch.Tensor, scale_inv: torch.Tensor) -> np.ndarray:
    """FP8-e4m3 block dequant: w[i,j] * scale_inv[i//128, j//128] (128x128 blocks
    per config quantization_config.weight_block_size; vLLM equivalent:
    scaled_dequantize(..., GroupShape(block,block)), deepseek_v2.py:769-779)."""
    out, inn = w8.shape
    bo, bi = scale_inv.shape
    assert bo == (out + FP8_BLOCK - 1) // FP8_BLOCK
    assert bi == (inn + FP8_BLOCK - 1) // FP8_BLOCK
    s = scale_inv.repeat_interleave(FP8_BLOCK, 0)[:out]
    s = s.repeat_interleave(FP8_BLOCK, 1)[:, :inn]
    return (w8.to(torch.float32) * s).numpy()


def load_weights(d: str, token_ids: np.ndarray):
    from safetensors import safe_open

    w = {}
    with safe_open(os.path.join(d, SHARD), framework="pt") as f:
        def get(name):
            return f.get_tensor(name)

        def linear(name):
            t = get(name + ".weight")
            if t.dtype == torch.float8_e4m3fn:
                return dequant_fp8(t, get(name + ".weight_scale_inv").float())
            return t.to(torch.float32).numpy()

        # embed rows only for the tokens we use (the table is 1.9 GB bf16)
        emb = get("model.embed_tokens.weight")  # [154880, 6144] bf16
        w["embed_rows"] = emb[torch.from_numpy(token_ids)].to(torch.float32).numpy()
        del emb
        w["input_layernorm"] = get(L0 + "input_layernorm.weight").float().numpy()
        for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
            w[n] = linear(ATT + n)
        w["q_a_layernorm"] = get(ATT + "q_a_layernorm.weight").float().numpy()
        w["kv_a_layernorm"] = get(ATT + "kv_a_layernorm.weight").float().numpy()
        w["idx_wq_b"] = linear(ATT + "indexer.wq_b")
        w["idx_wk"] = linear(ATT + "indexer.wk")
        w["idx_k_norm_w"] = get(ATT + "indexer.k_norm.weight").float().numpy()
        w["idx_k_norm_b"] = get(ATT + "indexer.k_norm.bias").float().numpy()
        w["idx_weights_proj"] = get(ATT + "indexer.weights_proj.weight").float().numpy()
    return w


# --------------------------------------------------------------------------
# forward pieces (fp32, jnp)
# --------------------------------------------------------------------------

def rmsnorm(x, weight, eps):
    """GlmMoeDsaRMSNorm (modeling.py:48-62), fp32."""
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * weight


def dense_attention_mass(h, q_resid, w, positions):
    """Layer-0 dense MLA attention row-mass per key, [T, T] fp32.

    Transcribes GlmMoeDsaAttention.forward (modeling.py:409-438): q from
    q_b_proj(q_resid) with nope-FIRST/rope-LAST split (:424-425 — note the MAIN
    attention splits [nope, rope], opposite order to the indexer's rope-first);
    kv from kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj (:427-430);
    INTERLEAVED rope on the rope slice (:434, apply_rotary_pos_emb_interleave —
    uncontested: config rope_interleave=true, vLLM is_neox_style=False,
    deepseek_v2.py:984-987); scaling = qk_head_dim**-0.5 (:398; rope_type
    "default" so no yarn mscale). Ground truth per doc §1.2 E1 = sum over the
    64 heads of each head's softmax row (row-mass per key).
    """
    T = h.shape[0]
    q = (q_resid @ jnp.asarray(w["q_b_proj"], jnp.float32).T)
    q = q.reshape(T, N_HEADS_ATTN, QK_HEAD)
    q_nope, q_rot = q[..., :QK_NOPE], q[..., QK_NOPE:]          # :425

    ckv = h @ jnp.asarray(w["kv_a_proj_with_mqa"], jnp.float32).T  # [T, 576]
    k_pass, k_rot = ckv[:, :KV_LORA], ckv[:, KV_LORA:]          # :428
    k_pass = rmsnorm(k_pass, jnp.asarray(w["kv_a_layernorm"], jnp.float32),
                     RMS_EPS_LORA)
    k_pass = (k_pass @ jnp.asarray(w["kv_b_proj"], jnp.float32).T)
    k_pass = k_pass.reshape(T, N_HEADS_ATTN, QK_NOPE + V_HEAD)  # :429
    k_nope = k_pass[..., :QK_NOPE]                              # :430 (v unused)

    cos, sin = rope_cos_sin(positions, QK_ROPE, ROPE_THETA)
    q_rot = apply_rope(q_rot, cos[:, None, :], sin[:, None, :], interleaved=True)
    k_rot = apply_rope(k_rot, cos, sin, interleaved=True)       # :434
    k_rot = jnp.broadcast_to(k_rot[:, None, :], (T, N_HEADS_ATTN, QK_ROPE))  # :435

    qf = jnp.concatenate([q_nope, q_rot], axis=-1)              # :437
    kf = jnp.concatenate([k_nope, k_rot], axis=-1)              # :438

    logits = jnp.einsum("thd,shd->hts", qf, kf,
                        preferred_element_type=jnp.float32) * (QK_HEAD ** -0.5)
    mask = jnp.arange(T)[None, :] > jnp.arange(T)[:, None]
    logits = jnp.where(mask[None], -jnp.inf, logits)
    probs = jax.nn.softmax(logits, axis=-1)                     # fp32
    return jnp.sum(probs, axis=0)  # [T, T] row-mass per key (Σ over heads)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank) — for Spearman."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else 0.0


def e1_metrics(idx_scores: np.ndarray, attn_mass: np.ndarray, ns=(64, 256)):
    """Per doc §1.2 E1: recall of indexer-top-n vs attention-top-n, per-row
    Spearman over the causal prefix, and attention-mass capture of the
    indexer-top-n set. Rows qualify for top-n metrics when they have > n valid
    keys (t >= n); Spearman uses rows with t >= 64."""
    T = idx_scores.shape[0]
    out = {}
    for n in ns:
        recalls, masses = [], []
        for t in range(n, T):
            s, a = idx_scores[t, :t + 1], attn_mass[t, :t + 1]
            top_s = np.argpartition(s, -n)[-n:]
            top_a = np.argpartition(a, -n)[-n:]
            recalls.append(len(np.intersect1d(top_s, top_a)) / n)
            masses.append(a[top_s].sum() / a.sum())
        out[f"recall@{n}"] = float(np.mean(recalls))
        out[f"mass@{n}"] = float(np.mean(masses))
    rhos = [spearman(idx_scores[t, :t + 1], attn_mass[t, :t + 1])
            for t in range(64, T)]
    out["spearman"] = float(np.mean(rhos))
    return out


def e2_forensics(w):
    """Doc §1.2 E2 — corroborating only. Within-pair statistics of the 64 rope
    rows of wk (rows 0:64) and of each wq_b head (rows h*128 + 0:64) under the
    two pairing hypotheses: interleaved (2i, 2i+1) vs half-split (i, i+32)."""
    def pair_stats(rows):  # rows [64, IN]
        res = {}
        for name, pairs in (
            ("interleaved", [(2 * i, 2 * i + 1) for i in range(QK_ROPE // 2)]),
            ("half-split", [(i, i + QK_ROPE // 2) for i in range(QK_ROPE // 2)]),
        ):
            coss, nrat = [], []
            for a, b in pairs:
                ra, rb = rows[a], rows[b]
                na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
                coss.append(abs(float(ra @ rb / (na * nb))))
                nrat.append(abs(float(np.log2(na / nb))))
            res[name] = (float(np.mean(coss)), float(np.mean(nrat)))
        return res

    wk_rows = w["idx_wk"][:QK_ROPE]
    wq = w["idx_wq_b"].reshape(IDX_HEADS, IDX_HEAD_DIM, Q_LORA)
    wq_rows = wq[:, :QK_ROPE, :].reshape(IDX_HEADS * QK_ROPE, Q_LORA)
    # aggregate wq_b per head then average
    stats = {"wk": pair_stats(wk_rows)}
    agg = {"interleaved": [], "half-split": []}
    for hh in range(IDX_HEADS):
        st = pair_stats(wq[hh, :QK_ROPE, :])
        for kname, v in st.items():
            agg[kname].append(v)
    stats["wq_b"] = {kname: tuple(np.mean(np.array(v), axis=0))
                     for kname, v in agg.items()}
    del wq_rows
    return stats


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="tokens of natural text (doc E1: ~512-2048)")
    args = ap.parse_args()

    assert jax.default_backend() == "cpu", (
        f"backend={jax.default_backend()} — this experiment must run on CPU only")

    ensure_files(args.weights_dir)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.weights_dir, "tokenizer.json"))
    ids = [tok.token_to_id("[gMASK]"), tok.token_to_id("<sop>")]  # GLM prefix
    ids += tok.encode(TEXT).ids
    if len(ids) < args.seq_len:
        print(f"[warn] text has only {len(ids)} tokens < requested {args.seq_len}")
    ids = np.array(ids[:args.seq_len], dtype=np.int64)
    T = len(ids)
    print(f"[setup] T={T} tokens of natural text (layer 0, fp32, CPU)")

    w = load_weights(args.weights_dir, ids)
    print("[setup] layer-0 weights loaded + fp8 dequantized "
          f"(wq_b {w['idx_wq_b'].shape}, wk {w['idx_wk'].shape})")

    # h = input_layernorm(embed(ids))  — the pre-attention hidden state layer 0
    # receives (modeling.py:652,740); q_resid = q_a_layernorm(q_a_proj(h)) (:423)
    h = rmsnorm(jnp.asarray(w["embed_rows"], jnp.float32),
                jnp.asarray(w["input_layernorm"], jnp.float32), RMS_EPS_LAYER)
    q_resid = rmsnorm(h @ jnp.asarray(w["q_a_proj"], jnp.float32).T,
                      jnp.asarray(w["q_a_layernorm"], jnp.float32), RMS_EPS_LORA)
    positions = np.arange(T)

    # indexer scores under both layouts (raw, then causal-masked)
    scores = {}
    for name, inter in (("interleaved", True), ("non-interleaved", False)):
        s = indexer_scores(h, q_resid, w["idx_wq_b"], w["idx_wk"],
                           w["idx_k_norm_w"], w["idx_k_norm_b"],
                           w["idx_weights_proj"], positions, ROPE_THETA,
                           interleaved=inter)
        scores[name] = np.asarray(causal_mask_scores(s))
    print("[e1] indexer scores computed under both layouts")

    attn = np.asarray(dense_attention_mass(h, q_resid, w, positions))
    print("[e1] dense MLA attention row-mass computed (ground truth)")

    results = {name: e1_metrics(s, attn) for name, s in scores.items()}

    # how different are the two layouts' selections at all? (discriminative power)
    ov = []
    for t in range(64, T):
        a = np.argpartition(scores["interleaved"][t, :t + 1], -64)[-64:]
        b = np.argpartition(scores["non-interleaved"][t, :t + 1], -64)[-64:]
        ov.append(len(np.intersect1d(a, b)) / 64)
    inter_overlap = float(np.mean(ov))

    print("\n===== E1 — score-vs-attention agreement (doc §1.2, layer 0, "
          f"T={T}) =====")
    print(f"{'metric':<12}{'interleaved':>14}{'non-interleaved':>17}")
    keys = list(results["interleaved"].keys())
    wins = {"interleaved": 0, "non-interleaved": 0}
    for kname in keys:
        vi, vn = results["interleaved"][kname], results["non-interleaved"][kname]
        wins["interleaved" if vi > vn else "non-interleaved"] += 1
        print(f"{kname:<12}{vi:>14.4f}{vn:>17.4f}")
    print(f"(inter-layout top-64 selected-set overlap: {inter_overlap:.3f} — "
          "the layouts genuinely select different sets)")

    print("\n===== E2 — weight forensics (corroborating only) =====")
    e2 = e2_forensics(w)
    for proj, st in e2.items():
        for hyp, (mcos, mnr) in st.items():
            print(f"{proj:<6} pairs {hyp:<12} mean|cos|={mcos:.4f}  "
                  f"mean|log2 norm-ratio|={mnr:.4f}")

    # ---- verdict ----
    n_metrics = len(keys)
    winner = max(wins, key=wins.get)
    margin = results[winner]["spearman"] - \
        results["non-interleaved" if winner == "interleaved" else "interleaved"]["spearman"]
    print("\n===== VERDICT =====")
    if wins[winner] == n_metrics and margin > 0.02:
        print(f"VERDICT: {winner.upper()} rope layout wins E1 on real weights — "
              f"all {n_metrics} agreement metrics higher "
              f"(Spearman margin {margin:+.4f}). "
              + ("Matches the doc §1.2 prior (vLLM honors "
                 "indexer_rope_interleave=true)."
                 if winner == "interleaved" else
                 "CONTRADICTS the doc §1.2 prior (vLLM/interleaved) — HF's "
                 "half-split call would be the trained layout; re-review before "
                 "trusting 2a."))
    else:
        print(f"VERDICT: INCONCLUSIVE — metric wins {wins}, Spearman margin "
              f"{margin:+.4f}. Do not hard-code a layout from this alone.")
    print("Residual uncertainty: this is the OFFLINE variant (E1+E2). The "
          "decisive E3 (passkey + prompt-logprob divergence-from-dense at ctx "
          "4K-16K under both layouts, doc §1.2) needs the 2a sparse path on "
          "the pod and remains to be run; at ctx<=2048 sparse==dense so only "
          "score/set-level evidence is available offline. E1 here uses layer 0 "
          "only (the doc suggests averaging layers 0-2; layers 1-2 live in "
          "other shards).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
