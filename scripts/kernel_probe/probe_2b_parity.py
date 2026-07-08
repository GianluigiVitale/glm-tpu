#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§2b single-chip real-MXU parity probe for the frozen DSA kernels.

Runbook: ~/glm-tpu/docs/11-pod-runbook.md §2b. Runs on the SAME idle v4 chip as
§2a, immediately after §2a returns ACCEPT. Three gates, each printing its deltas
VERBATIM (the actual numbers, not just pass/fail) and a PASS/FAIL, with a
results file written under docs/artifacts/:

  GATE A — indexer.  indexer_scores_pallas vs indexer_scores_xla (real-MXU
      kernel-vs-XLA core parity) AND hierarchical_topk selection vs the HF-math
      oracle (~/glm-tpu/parity/glm_indexer_reference.py):
        * S1 (fp32): selected-set EXACT modulo ties (design doc §4 gate S1).
        * S2 (bf16): boundary-band criterion — every selection mismatch lies in
          the k-th-score band |s - s_kth| <= EPS * score_scale, EPS = 2^-8
          (round-5 finding; design doc §4 gate S2).
  GATE B — sparse-MLA decode.  dsa_sparse_decode vs dsa_sparse_decode_xla:
        * fp32 max-abs <= 2e-6   (round-5: measured 9.54e-7, 2.1x headroom)
        * bf16 max-abs <= 2.5e-3 (round-5: measured 1.953e-3 = 2 ulps at |o|<0.5)
  GATE C — pack_new_kv OOB geometry at kv_len 511 / 512 / 513 (the v4 pod
      E0200 core-halt class): real kv_utils.pack_new_kv, byte-identical vs a
      numpy oracle AND no out-of-bounds VMEM read (the interpreter/bounds-checked
      TPU raises IndexError on a clamp regression).

NUMERIC BARS ARE INTERPRETER-DERIVED (round-5 caveat): CPU computes bf16 dots in
fp32 and fp32 HIGHEST exactly, so the interpret-mode deltas are near-zero; the
TPU MXU single-pass bf16 and 3/6-pass fp32 decompositions DIFFER. §2b re-measures
them on real silicon — that is the whole point of this gate. The printed numbers
are the deliverable; the PASS/FAIL grades them against the documented bars.

FROZEN KERNELS (c1456937 behavior): TEST-HARNESS code — MUST NOT modify any
kernel logic.

-----------------------------------------------------------------------------
PREREQUISITES for the METAL run (owner / main-thread, POST-GPQA):
  * ~/tpu-inference on branch  glm-5.2-v4-next  (run_kernel_gates.sh enforces).
  * EXCLUSIVE single chip: TPU_VISIBLE_DEVICES=0, no engine/GPQA/Ray running.

Metal invocation (what the orchestrator runs, only after §2a ACCEPT):
    TPU_VISIBLE_DEVICES=0 ~/vllm-env/bin/python \
        scripts/kernel_probe/probe_2b_parity.py --tag <date-or-run>

CPU wiring-validation NOW (never touches the TPU):
    ~/vllm-env/bin/python scripts/kernel_probe/probe_2b_parity.py --interpret

Exit codes: 0 = all gates PASS   1 = a gate FAILED   2 = misconfiguration.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # ~/glm-tpu
_PARITY_DIR = _ROOT / "parity"
_ARTIFACTS = _ROOT / "docs" / "artifacts"

# -- GLM-5.2 geometry ---------------------------------------------------------
ROPE_THETA = 8.0e6
# indexer real head dims (H=32, D=128, rope 64, pages of 128):
IDX_SEQ_LENS = [512, 300, 129, 128]
IDX_HIDDEN, IDX_QLORA = 256, 128
IDX_HEADS, IDX_HEAD_DIM, IDX_ROPE_DIM = 32, 128, 64
IDX_PAGE, IDX_TOPK, IDX_MERGE_BLOCK = 128, 64, 256
# sparse-MLA decode geometry (design doc §3.1):
S_H, S_LKV, S_RDIM, S_KVW, S_SEG = 64, 512, 64, 640, 2048
S_SM_SCALE = (S_LKV + S_RDIM) ** -0.5
# pack_new_kv OOB (page_size 512, bkv_p 1 — the v4 pod decode config):
PK_BKV_SZ, PK_LKV, PK_RD = 512, 128, 128

# -- bars ---------------------------------------------------------------------
IDX_SCORE_FP32_BAR = 5e-4      # kernel-vs-XLA fp32 score parity (3-pass MXU vs XLA)
IDX_S2_EPS = 2.0 ** -8         # bf16 boundary-band relative width (round-5)
SPARSE_FP32_BAR = 2e-6
SPARSE_BF16_BAR = 2.5e-3
_NEG_INF = float("-inf")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interpret", action="store_true",
                   help="Run all kernels in INTERPRET mode on CPU (forces "
                   "JAX_PLATFORMS=cpu). Proves harness wiring, not silicon parity.")
    p.add_argument("--tag", default=None,
                   help="Tag for the results file docs/artifacts/"
                   "kernelprobe-2b-<tag>.results.txt (default: UTC date, "
                   "'-interpret' suffix in interpret mode).")
    return p.parse_args()


# ===========================================================================
# Small tee logger: everything printed also lands in the results file verbatim.
# ===========================================================================
class _Tee:
    def __init__(self, path: Path):
        self.lines: list[str] = []
        self.path = path

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.lines.append(line)

    def flush_to_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")


# ===========================================================================
# GATE A helpers — HF-math oracle case (projection + rope + scoring), mirrors
# tests/kernels/test_dsa_indexer_kernel.py (harness code, not kernel logic).
# ===========================================================================
def _build_paged_cache(np, jnp, keys_per_req, page_size, seed, dtype,
                       extra_blocks=1, num_extra_pages=3):
    rng = np.random.RandomState(seed)
    R = len(keys_per_req)
    D = np.asarray(keys_per_req[0]).shape[-1]
    lens = [np.asarray(k).shape[0] for k in keys_per_req]
    blocks_needed = [max((L + page_size - 1) // page_size, 1) for L in lens]
    max_blocks = max(blocks_needed) + extra_blocks
    num_pages = sum(blocks_needed) + num_extra_pages
    perm = rng.permutation(num_pages)
    cache = rng.standard_normal((num_pages, page_size, D)).astype(np.float32)
    block_tables = np.zeros((R, max_blocks), dtype=np.int32)  # pad -> page 0
    nxt = 0
    for r, k_r in enumerate(keys_per_req):
        k_r = np.asarray(k_r, dtype=np.float32)
        for b in range(blocks_needed[r]):
            page = int(perm[nxt]); nxt += 1
            block_tables[r, b] = page
            tok = k_r[b * page_size:(b + 1) * page_size]
            cache[page, :tok.shape[0]] = tok
    return (jnp.asarray(cache, dtype=dtype), jnp.asarray(block_tables),
            jnp.asarray(lens, dtype=jnp.int32))


def _make_reference_case(np, jnp, jax, ref, seed=0):
    """Decode step (last token = query) at real head dims + the reference's own
    full-map fp32 scores as oracle. Returns kernel inputs + per-req oracle rows.

    The WHOLE builder runs under matmul precision 'highest': on TPU the
    default precision computes fp32 einsums via bf16 MXU passes, which put
    ~4.5e-3 errors in the ORACLE's own scores (the first on-metal A2 run
    failed on exactly one boundary pair this way — kernel and XLA twin agreed
    to 4.8e-7 while the oracle was off by 4.5e-3 on the disputed position).
    The HF reference semantics are fp32-exact; 'highest' restores that on
    device and is a no-op on CPU."""
    with jax.default_matmul_precision("highest"):
        return _make_reference_case_inner(np, jnp, jax, ref, seed)


def _make_reference_case_inner(np, jnp, jax, ref, seed=0):
    key = jax.random.PRNGKey(seed)
    wkey, *skeys = jax.random.split(key, len(IDX_SEQ_LENS) + 1)
    ks = jax.random.split(wkey, 5)
    hidden, q_lora, nH, dH = IDX_HIDDEN, IDX_QLORA, IDX_HEADS, IDX_HEAD_DIM
    wts = dict(
        wq_b=jax.random.normal(ks[0], (nH * dH, q_lora), jnp.float32) * (q_lora ** -0.5),
        wk=jax.random.normal(ks[1], (dH, hidden), jnp.float32) * (hidden ** -0.5),
        k_norm_w=1.0 + 0.1 * jax.random.normal(ks[2], (dH,), jnp.float32),
        k_norm_b=0.1 * jax.random.normal(ks[3], (dH,), jnp.float32),
        weights_proj=jax.random.normal(ks[4], (nH, hidden), jnp.float32) * (hidden ** -0.5),
    )
    q_rows, w_rows, keys_per_req, oracle_rows = [], [], [], []
    for r, L in enumerate(IDX_SEQ_LENS):
        hk, qk = jax.random.split(skeys[r])
        h = jax.random.normal(hk, (L, hidden), jnp.float32)
        q_resid = jax.random.normal(qk, (L, q_lora), jnp.float32)
        positions = jnp.arange(L)
        scores_map = ref.indexer_scores(
            h, q_resid, wts["wq_b"], wts["wk"], wts["k_norm_w"], wts["k_norm_b"],
            wts["weights_proj"], positions, ROPE_THETA,
            interleaved=True, rope_dim=IDX_ROPE_DIM)
        masked = ref.causal_mask_scores(scores_map)
        oracle_rows.append(np.asarray(masked[L - 1]))
        # rebuild q/k/w post-projection (reference's own helpers) as kernel inputs.
        q = (q_resid @ wts["wq_b"].T).reshape(L, nH, dH)
        k_pre = h @ wts["wk"].T
        mean = jnp.mean(k_pre, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(k_pre - mean), axis=-1, keepdims=True)
        k = (k_pre - mean) / jnp.sqrt(var + 1e-6)
        k = k * wts["k_norm_w"] + wts["k_norm_b"]
        cos, sin = ref.rope_cos_sin(positions, IDX_ROPE_DIM, ROPE_THETA)
        q_rot = ref.apply_rope(q[:, :, :IDX_ROPE_DIM], cos[:, None, :],
                               sin[:, None, :], interleaved=True)
        k_rot = ref.apply_rope(k[:, :IDX_ROPE_DIM], cos, sin, interleaved=True)
        q = jnp.concatenate([q_rot, q[:, :, IDX_ROPE_DIM:]], axis=-1)
        k = jnp.concatenate([k_rot, k[:, IDX_ROPE_DIM:]], axis=-1)
        w = (h @ wts["weights_proj"].T) * (nH ** -0.5)
        q_rows.append(np.asarray(q[L - 1]))
        w_rows.append(np.asarray(w[L - 1]))
        keys_per_req.append(np.asarray(k))
    kv_cache, block_tables, kv_lens = _build_paged_cache(
        np, jnp, keys_per_req, IDX_PAGE, seed + 1, jnp.float32)
    return (jnp.asarray(np.stack(q_rows)), jnp.asarray(np.stack(w_rows)),
            kv_cache, block_tables, kv_lens, oracle_rows)


def _tie_aware_mismatches(np, score_row, got_idx, want_idx, atol):
    """Return (n_total_mismatch, n_out_of_band): symmetric set diff, then how
    many differing indices are NOT within `atol` of the k-th (want) score."""
    got, want = set(int(i) for i in got_idx), set(int(i) for i in want_idx)
    diff = got ^ want
    if not diff:
        return 0, 0
    row = np.asarray(score_row)
    kth = row[list(want)].min()
    out = [i for i in diff if not np.isclose(row[i], kth, atol=atol, rtol=1e-5)]
    return len(diff), len(out)


def gate_A_indexer(log, np, jnp, jax, interpret):
    from tpu_inference.kernels.dsa.indexer_kernel import (
        indexer_scores_pallas, indexer_scores_xla, hierarchical_topk)
    sys.path.insert(0, str(_PARITY_DIR))
    import glm_indexer_reference as ref

    log("")
    log("=" * 78)
    log("GATE A — indexer: pallas-vs-XLA core parity + top-k vs HF oracle")
    log("=" * 78)
    q, w, cache, bt, klens, oracle_rows = _make_reference_case(np, jnp, jax, ref)
    klens_np = np.asarray(klens)

    # ---- fp32 ----------------------------------------------------------------
    s_pal = np.asarray(indexer_scores_pallas(q, w, cache, bt, klens,
                                             interpret=interpret))
    s_xla = np.asarray(indexer_scores_xla(q, w, cache, bt, klens))
    idx_pal, nv = hierarchical_topk(jnp.asarray(s_pal), klens, IDX_TOPK,
                                    merge_block=IDX_MERGE_BLOCK)
    idx_pal, nv = np.asarray(idx_pal), np.asarray(nv)

    log("")
    log("[A1] fp32 kernel-vs-XLA score parity (max|s_pal - s_xla| on valid prefix):")
    a1_max = 0.0
    for r, L in enumerate(IDX_SEQ_LENS):
        d = float(np.abs(s_pal[r, :L] - s_xla[r, :L]).max())
        a1_max = max(a1_max, d)
        neg_ok = bool(np.all(np.isneginf(s_pal[r, L:]) == np.isneginf(s_xla[r, L:])))
        log(f"    req{r} L={L:4d}: max_abs={d:.3e}   -inf-fill agree={neg_ok}")
    a1_pass = a1_max <= IDX_SCORE_FP32_BAR
    log(f"    -> A1 max_abs={a1_max:.3e}  bar<={IDX_SCORE_FP32_BAR:.0e}  "
        f"{'PASS' if a1_pass else 'FAIL'}")

    log("")
    log("[A2] S1 fp32 selection vs HF oracle (selected-set EXACT modulo ties):")
    a2_bad = 0
    for r, L in enumerate(IDX_SEQ_LENS):
        k_eff = min(IDX_TOPK, L)
        ref_idx = np.asarray(ref.topk_indices(
            jnp.asarray(oracle_rows[r][None, :]), IDX_TOPK, causal=False))[0][:k_eff]
        got = idx_pal[r, :k_eff]
        n_mis, n_bad = _tie_aware_mismatches(np, oracle_rows[r], got, ref_idx,
                                             atol=1e-4)
        a2_bad += n_bad
        log(f"    req{r} L={L:4d} k={k_eff}: n_valid={int(nv[r])} "
            f"set-mismatch={n_mis} non-tie-mismatch={n_bad}")
    a2_pass = a2_bad == 0
    log(f"    -> A2 non-tie mismatches={a2_bad}  bar=0  "
        f"{'PASS' if a2_pass else 'FAIL'}")

    # ---- bf16 (S2 boundary band) --------------------------------------------
    qb, wb, cb = (q.astype(jnp.bfloat16), w.astype(jnp.bfloat16),
                  cache.astype(jnp.bfloat16))
    s_pal_bf = np.asarray(indexer_scores_pallas(qb, wb, cb, bt, klens,
                                                interpret=interpret))
    idx_bf, _ = hierarchical_topk(jnp.asarray(s_pal_bf), klens, IDX_TOPK,
                                  merge_block=IDX_MERGE_BLOCK)
    idx_bf = np.asarray(idx_bf)

    log("")
    log(f"[A3] S2 bf16 selection vs fp32 oracle (band |s - s_kth| <= 2^-8*scale, "
        f"eps={IDX_S2_EPS:.3e}):")
    a3_bad = 0
    for r, L in enumerate(IDX_SEQ_LENS):
        k_eff = min(IDX_TOPK, L)
        scale = float(np.abs(oracle_rows[r][:L]).max())
        band = IDX_S2_EPS * max(scale, 1.0)
        ref_idx = np.asarray(ref.topk_indices(
            jnp.asarray(oracle_rows[r][None, :]), IDX_TOPK, causal=False))[0][:k_eff]
        got = idx_bf[r, :k_eff]
        n_mis, n_bad = _tie_aware_mismatches(np, oracle_rows[r], got, ref_idx,
                                             atol=band)
        a3_bad += n_bad
        # verbatim bf16 score perturbation on the valid prefix (round-5 stat).
        pert = np.abs(s_pal_bf[r, :L] - s_pal[r, :L])
        log(f"    req{r} L={L:4d} k={k_eff}: scale={scale:.3f} band={band:.3e} "
            f"set-mismatch={n_mis} out-of-band={n_bad} | "
            f"bf16 score |Δ| med={np.median(pert):.3e} p99={np.percentile(pert,99):.3e} "
            f"max={pert.max():.3e}")
    a3_pass = a3_bad == 0
    log(f"    -> A3 out-of-band mismatches={a3_bad}  bar=0  "
        f"{'PASS' if a3_pass else 'FAIL'}")

    passed = a1_pass and a2_pass and a3_pass
    log("")
    log(f"GATE A: {'PASS' if passed else 'FAIL'}  "
        f"(A1 score-parity, A2 S1 fp32 exact-mod-ties, A3 S2 bf16 band)")
    return passed


# ===========================================================================
# GATE B — sparse-MLA decode: kernel vs XLA oracle, fp32 + bf16.
# ===========================================================================
def gate_B_sparse(log, np, jnp, interpret):
    from tpu_inference.kernels.dsa.sparse_mla_kernel import (
        dsa_sparse_decode, dsa_sparse_decode_xla)
    log("")
    log("=" * 78)
    log("GATE B — sparse-MLA decode: dsa_sparse_decode vs dsa_sparse_decode_xla")
    log("=" * 78)
    rng = np.random.default_rng(1234)
    R = 4
    seg_valid = jnp.asarray([17, 128, 1000, 2048], jnp.int32)

    def make(dtype):
        qn = jnp.asarray(rng.normal(size=(R, S_H, S_LKV)), dtype)
        qp = jnp.asarray(rng.normal(size=(R, S_H, S_RDIM)), dtype)
        seg = np.zeros((R, S_SEG, S_KVW), np.float32)
        seg[..., :S_LKV + S_RDIM] = rng.normal(size=(R, S_SEG, S_LKV + S_RDIM))
        return qn, qp, jnp.asarray(seg, dtype)

    results = {}
    for name, dtype, bar in (("fp32", jnp.float32, SPARSE_FP32_BAR),
                             ("bf16", jnp.bfloat16, SPARSE_BF16_BAR)):
        qn, qp, seg = make(dtype)
        out = np.asarray(dsa_sparse_decode(qn, qp, seg, seg_valid,
                                           sm_scale=S_SM_SCALE,
                                           interpret=interpret))
        ref = np.asarray(dsa_sparse_decode_xla(qn, qp, seg, seg_valid,
                                               sm_scale=S_SM_SCALE))
        d = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())
        ok = d <= bar
        results[name] = ok
        log(f"    {name}: R={R} seg_valid={[int(v) for v in np.asarray(seg_valid)]} "
            f"max_abs={d:.3e}  bar<={bar:.1e}  {'PASS' if ok else 'FAIL'}")
    passed = all(results.values())
    log("")
    log(f"GATE B: {'PASS' if passed else 'FAIL'}")
    return passed


# ===========================================================================
# GATE C — pack_new_kv OOB geometry at kv_len 511/512/513.
# Mirrors tests/kernels/mla_v2_pack_new_kv_oob_test.py (harness code).
# ===========================================================================
def gate_C_pack_oob(log, np, jnp, jax, interpret):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from tpu_inference.kernels.mla.v2 import kv_utils
    log("")
    log("=" * 78)
    log("GATE C — pack_new_kv OOB geometry at kv_len 511 / 512 / 513 (E0200 class)")
    log("=" * 78)

    def cdiv(a, b):
        return (a + b - 1) // b

    def run_pack(kvc, kpe, offset, update_sz, q_end, kv_len):
        def kern(sref, kvi, kpi, oc, op):
            oc[...] = kvi[...]
            op[...] = kpi[...]
            kv_utils.pack_new_kv(oc, op, sref[0], sref[1], sref[2], sref[3],
                                 PK_BKV_SZ)
        scalars = jnp.array([offset, update_sz, q_end, kv_len], jnp.int32)
        fn = pl.pallas_call(
            kern,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                grid=(1,)),
            out_shape=[jax.ShapeDtypeStruct(kvc.shape, kvc.dtype),
                       jax.ShapeDtypeStruct(kpe.shape, kpe.dtype)],
            interpret=pltpu.InterpretParams() if interpret else False)
        oc, op = fn(scalars, kvc, kpe)
        return np.asarray(oc), np.asarray(op)

    def build_case(offset, update_sz, q_end, kv_len, kv_dtype, seed=0):
        rng = np.random.default_rng(seed)
        P = kv_utils.get_dtype_packing(kv_dtype)
        rows = PK_BKV_SZ // P + 2

        def rnd(shape):
            return jnp.array(rng.random(size=shape, dtype=np.float32) * 0.5
                             ).astype(kv_dtype)
        kvc = np.array(rnd((rows, P, PK_LKV)))
        kpe = np.array(rnd((rows, P, PK_RD)))
        tob = offset % PK_BKV_SZ
        n_cache_rows = cdiv(tob, P)
        new_start = q_end - kv_len + offset
        src_slot0 = new_start % P
        new_kvc = np.array(rnd((update_sz, PK_LKV)))
        new_kpe = np.array(rnd((update_sz, PK_RD)))
        for j in range(update_sz):
            r = n_cache_rows + (src_slot0 + j) // P
            s = (src_slot0 + j) % P
            kvc[r, s] = new_kvc[j]
            kpe[r, s] = new_kpe[j]
        exp_kvc, exp_kpe = kvc.copy(), kpe.copy()
        for j in range(update_sz):
            p = tob + j
            exp_kvc[p // P, p % P] = new_kvc[j]
            exp_kpe[p // P, p % P] = new_kpe[j]
        return jnp.asarray(kvc), jnp.asarray(kpe), exp_kvc, exp_kpe, P

    passed = True
    # kv_len 511/512/513: new token at offset kv_len-1; sweep even/odd q_end
    # (od==0 / od==-1 sub-word alignment). bf16 (kv_packing=2) is the pod dtype.
    for kv_len in (511, 512, 513):
        offset = kv_len - 1
        for q_end in (10, 11):
            try:
                kvc, kpe, exp_kvc, exp_kpe, P = build_case(
                    offset, 1, q_end, kv_len, jnp.bfloat16)
                oc, op = run_pack(kvc, kpe, offset, 1, q_end, kv_len)
                valid = PK_BKV_SZ // P
                bit_c = np.array_equal(oc[:valid].view(np.uint8),
                                       exp_kvc[:valid].view(np.uint8))
                bit_p = np.array_equal(op[:valid].view(np.uint8),
                                       exp_kpe[:valid].view(np.uint8))
                ok = bit_c and bit_p
                passed = passed and ok
                log(f"    kv_len={kv_len} offset={offset} q_end={q_end} bf16: "
                    f"no-OOB=True byte-identical(kvc={bit_c},kpe={bit_p})  "
                    f"{'PASS' if ok else 'FAIL'}")
            except Exception as e:  # OOB regression -> IndexError on this path.
                passed = False
                log(f"    kv_len={kv_len} offset={offset} q_end={q_end} bf16: "
                    f"OOB/ERROR — {type(e).__name__}: {e}  FAIL")
    log("")
    log(f"GATE C: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    args = _parse_args()
    if args.interpret:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import numpy as np
    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    if not args.interpret and backend != "tpu":
        print("GATE 2b: MISCONFIGURED — metal run requested (no --interpret) but "
              f"jax backend is '{backend}', not 'tpu'. Use run_kernel_gates.sh "
              "with TPU_VISIBLE_DEVICES=0, or pass --interpret for the CPU check.")
        return 2

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    tag = args.tag or now_utc.strftime("%Y%m%d")
    if args.interpret and not (args.tag and "interpret" in args.tag):
        tag = f"{tag}-interpret"
    results_path = _ARTIFACTS / f"kernelprobe-2b-{tag}.results.txt"
    log = _Tee(results_path)

    # provenance
    try:
        fork_dir = Path(os.path.expanduser("~/tpu-inference"))
        branch = subprocess.check_output(
            ["git", "-C", str(fork_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(fork_dir), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        branch, commit = "unknown", "unknown"

    mode = "INTERPRET (CPU wiring-check)" if args.interpret else "METAL (real MXU)"
    log("=" * 78)
    log("§2b — single-chip real-MXU parity probe for the frozen DSA kernels")
    log("=" * 78)
    log(f"utc               : {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    log(f"mode              : {mode}")
    log(f"jax backend       : {backend}")
    log(f"fork branch/commit: {branch} / {commit}   "
        f"(expected: glm-5.2-v4-next)")
    log(f"results file      : {results_path}")
    if not args.interpret and branch != "glm-5.2-v4-next":
        log(f"WARNING           : fork is on '{branch}', not glm-5.2-v4-next "
            "— run via run_kernel_gates.sh which enforces the branch.")

    a = gate_A_indexer(log, np, jnp, jax, args.interpret)
    b = gate_B_sparse(log, np, jnp, args.interpret)
    c = gate_C_pack_oob(log, np, jnp, jax, args.interpret)

    log("")
    log("=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"  GATE A (indexer parity + top-k)  : {'PASS' if a else 'FAIL'}")
    log(f"  GATE B (sparse-MLA decode parity): {'PASS' if b else 'FAIL'}")
    log(f"  GATE C (pack_new_kv OOB 511/512/513): {'PASS' if c else 'FAIL'}")
    all_pass = a and b and c
    log("-" * 78)
    log(f"  §2b OVERALL: {'PASS — parity green' if all_pass else 'FAIL'}"
        + ("" if args.interpret else " on silicon"))
    if args.interpret:
        log("  (interpret mode: proves harness wiring end-to-end; NOT silicon "
            "parity. bf16 bars are interpreter artifacts — re-measured on metal.)")
    log("=" * 78)
    log.flush_to_file()
    print(f"\n[results written] {results_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
