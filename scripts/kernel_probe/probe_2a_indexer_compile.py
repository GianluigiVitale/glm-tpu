#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§2a single-chip indexer-kernel Mosaic COMPILE probe (highest-risk-first gate).

Runbook: ~/glm-tpu/docs/11-pod-runbook.md §2a. This is the cheapest, highest-risk
Stage-2 gate: it runs FIRST after §0/§1, on ONE idle v4 chip, BEFORE any
engine-level DSA work or the staging switch.

THE ONE QUESTION (round-5 flagged, no validated precedent):
    Does Mosaic accept the ``(1, H)`` w-tile as a matmul LHS on real v4 silicon?

The frozen kernel ``_indexer_scores_kernel`` (indexer_kernel.py) does the signed
weighted head-sum as a ``[1, H] @ [H, P] -> [1, P]`` ``lax.dot_general`` with
``w_ref[...]`` (shape ``(1, H)``) as the LHS. A 1-row MXU LHS has no precedent in
the DSV4 kernels; interpret-mode (CPU) cannot see whether Mosaic's real-hardware
layout engine accepts it. This probe compiles + runs the kernel with
``interpret=False`` on TINY inputs (R=2, ctx=512, page 128, H=32, D=128) and
prints an unambiguous verdict.

    GATE 2a: ACCEPT   -> proceed to §2b (probe_2b_parity.py).
    GATE 2a: REJECT   -> apply the DOCUMENTED FALLBACK, re-run, commit the fix:
        replace the (1,H)@[H,P] matmul with a broadcast-multiply
        ``w[0, :, None] * s`` ([H,1]*[H,P] -> [H,P]) followed by a sublane
        reduction ``jnp.sum(..., axis=0)`` over the H axis (-> [1,P]). See
        docs/01-dsa-kernel-design.md §1.3 and indexer_kernel.py
        _indexer_scores_kernel.

FROZEN KERNEL: the DSA kernels are frozen at c1456937 behavior per the adopted
audit; this is TEST-HARNESS code and MUST NOT modify any kernel logic.

-----------------------------------------------------------------------------
PREREQUISITES for the METAL run (owner / main-thread, POST-GPQA):
  * ~/tpu-inference MUST be checked out on branch  glm-5.2-v4-next  (the staging
    branch: mainline + r5fix + 2int + dcp + r6fix, with the DSA kernels frozen
    at c1456937 behavior). run_kernel_gates.sh enforces this before launch.
  * EXCLUSIVE single-chip access: TPU_VISIBLE_DEVICES=0, NO engine / GPQA / Ray
    worker running (this consumes the chip). run_kernel_gates.sh guards this.

Metal invocation (what the orchestrator runs):
    TPU_VISIBLE_DEVICES=0 ~/vllm-env/bin/python \
        scripts/kernel_probe/probe_2a_indexer_compile.py

CPU wiring-validation (safe to run NOW, never touches the TPU):
    JAX_PLATFORMS is forced to cpu by --interpret; this proves the harness is
    correctly wired (inputs valid, kernel callable, verdict + exit-code logic
    work) before the script ever hits the pod. interpret=True ACCEPT means
    "harness wired", NOT the real Mosaic decision.
    ~/vllm-env/bin/python scripts/kernel_probe/probe_2a_indexer_compile.py --interpret

Exit codes:  0 = ACCEPT   1 = REJECT (Mosaic error)   2 = misconfiguration.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# TINY probe shapes (runbook §2a): R=2, ctx=512, page 128, H=32, D=128, topk=64.
R = 2
CTX = 512
PAGE_SIZE = 128
H = 32
D = 128
TOPK = 64  # not consumed by the scoring kernel; documents the decode geometry.

_FALLBACK = (
    "FALLBACK (documented, apply on REJECT): replace the (1,H)@[H,P] weighted "
    "head-sum matmul in _indexer_scores_kernel with a broadcast-multiply "
    "w[0, :, None] * s  ([H,1]*[H,P] -> [H,P])  followed by a sublane reduction "
    "jnp.sum(..., axis=0) over the H axis (-> [1,P]).  Refs: "
    "docs/01-dsa-kernel-design.md §1.3, indexer_kernel.py _indexer_scores_kernel.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--interpret", action="store_true",
        help="Run the Pallas kernel in INTERPRET mode on CPU (forces "
        "JAX_PLATFORMS=cpu). Proves harness wiring; NOT the real Mosaic gate.")
    p.add_argument(
        "--vmem-limit-bytes", type=int, default=None,
        help="Optional vmem_limit_bytes passed to the kernel (metal tuning).")
    return p.parse_args()


def _build_tiny_inputs(np, jnp):
    """Two decode requests over a shuffled paged indexer k-cache (R=2, ctx=512,
    page 128, H=32, D=128). Layout mirrors the kernel's input contract and the
    test-suite's _build_paged_cache; kv_lens deliberately differ (one full page
    boundary, one partial) so both the valid-page matmul path and the
    fully-invalid -inf-fill path are exercised."""
    rng = np.random.RandomState(0)
    max_blocks = (CTX + PAGE_SIZE - 1) // PAGE_SIZE           # 4
    kv_lens = np.array([CTX, CTX - PAGE_SIZE], dtype=np.int32)  # [512, 384]
    num_pages = R * max_blocks + 2
    perm = rng.permutation(num_pages)
    cache = rng.standard_normal((num_pages, PAGE_SIZE, D)).astype(np.float32)
    block_tables = np.zeros((R, max_blocks), dtype=np.int32)  # pad -> page 0
    nxt = 0
    for r in range(R):
        for b in range(max_blocks):
            block_tables[r, b] = int(perm[nxt])
            nxt += 1
    q = rng.standard_normal((R, H, D)).astype(np.float32)
    w = rng.standard_normal((R, H)).astype(np.float32)        # SIGNED head gates
    return (jnp.asarray(q), jnp.asarray(w), jnp.asarray(cache),
            jnp.asarray(block_tables), jnp.asarray(kv_lens))


def main() -> int:
    args = _parse_args()
    if args.interpret:
        # Force CPU BEFORE importing jax so the TPU backend is never grabbed.
        os.environ["JAX_PLATFORMS"] = "cpu"

    import numpy as np
    import jax
    import jax.numpy as jnp
    from tpu_inference.kernels.dsa.indexer_kernel import indexer_scores_pallas

    backend = jax.default_backend()
    mode = "INTERPRET (CPU wiring-check)" if args.interpret else "METAL (real Mosaic gate)"
    print("=" * 78)
    print("GATE 2a — single-chip indexer-kernel Mosaic COMPILE probe")
    print("=" * 78)
    print(f"mode              : {mode}")
    print(f"jax backend       : {backend}")
    print(f"shapes            : R={R} ctx={CTX} page={PAGE_SIZE} H={H} D={D} topk={TOPK}")
    print(f"kernel            : indexer_scores_pallas  (interpret={args.interpret})")
    print(f"the one question  : does Mosaic accept the (1,H) w-tile as a matmul LHS?")
    print("-" * 78)

    if not args.interpret and backend != "tpu":
        print("GATE 2a: MISCONFIGURED — metal run requested (no --interpret) but the "
              f"jax backend is '{backend}', not 'tpu'.")
        print("  Run under run_kernel_gates.sh with TPU_VISIBLE_DEVICES=0, or pass "
              "--interpret for the CPU wiring-check.")
        return 2

    q, w, kv_cache, block_tables, kv_lens = _build_tiny_inputs(np, jnp)

    try:
        scores = indexer_scores_pallas(q, w, kv_cache, block_tables, kv_lens,
                                       interpret=args.interpret,
                                       vmem_limit_bytes=args.vmem_limit_bytes)
        # Force materialization -> triggers the Mosaic compile on real TPU.
        scores = jax.block_until_ready(scores)
        arr = np.asarray(scores)
        assert arr.shape == (R, (CTX // PAGE_SIZE) * PAGE_SIZE), arr.shape
        finite = np.isfinite(arr)
        # Sanity: valid prefix finite, invalid tail exactly -inf.
        assert finite.any(), "no finite scores produced"
        assert np.all(np.isneginf(arr[~finite])), "non -inf invalid entries"
        print(f"kernel ran        : output {arr.shape} fp32, "
              f"finite fraction {finite.mean():.4f}, "
              f"valid-score range [{arr[finite].min():.4f}, {arr[finite].max():.4f}]")
        print("-" * 78)
        print("GATE 2a: ACCEPT" +
              ("  (harness wired — interpret mode; not the real Mosaic decision)"
               if args.interpret else
               "  — Mosaic accepted the (1,H) w-tile matmul LHS on real silicon"))
        return 0
    except Exception as e:  # noqa: BLE001 — we WANT to catch a Mosaic reject.
        print("kernel raised     : the Mosaic compile / run FAILED")
        print("-" * 78)
        print("---- full traceback (verbatim) ----")
        traceback.print_exc()
        print("---- end traceback ----")
        print("-" * 78)
        err = f"{type(e).__name__}: {e}"
        # Keep the verdict line to a single, greppable line (truncate long errors).
        one_line = err.replace("\n", " ⏎ ")
        if len(one_line) > 400:
            one_line = one_line[:400] + " …[truncated]"
        print(f"GATE 2a: REJECT — {one_line}")
        print(_FALLBACK)
        return 1


if __name__ == "__main__":
    sys.exit(main())
