# 19 — The 256K sparse-decode throughput verdict: a measured null, and why it is structural

**Status:** CLOSED 2026-07-31 (owner accepted the measured null). Goal condition 3b resolves as:
256K **correctness** proven (mechanism smoke 4/4 — the first 256K retrievals on this stack, dcp=8
sanity 2/2 first metal tokens), **decode throughput** an honest null with a fully measured
structural explanation. This document is the deliverable.

**Authoritative artifacts:** `docs/artifacts/e0-sparse-decode-breakdown.md`,
`docs/artifacts/DENSE_DECODE_BREAKDOWN.md`, `docs/artifacts/SPARSE_VS_DENSE_DIFFERENTIAL.md`
(all 64 cores × 15 decode steps per arm, in-worker `GLM_JAX_TRACE` captures, parser
`scripts/analysis/parse_xplane.py`, <1% agreement vs xprof's C++ `hlo_stats`);
`docs/18-sparse-decode-256k-ladder.md` (the pre-measurement judged experiment ladder);
RESEARCH_LOG 07-27 04:20 (the A/B numbers) and 07-31 05:10/06:00 (the measured breakdowns).

## The measured facts (262,144-token context, dcp=8, v4-64, GLM-5.2-FP8 753B)

| | sparse DSA | dense MLA |
|---|---|---|
| decode wall | 1.90–2.19 tok/s (455–526 ms/step) | **3.83 tok/s (260.8 ms/step)** |
| device step (profiled) | 389.4 ms busy | 269.3 ms busy |
| collectives | 183.7 ms (45.6%) | 171.0 ms (63.5%) |
| — all-reduce invocations | **232/step** | **232/step (identical)** |
| selection (gathers + top-k/sort + glue) | ~94 ms | ~1 ms |
| attend kernel | 12.7 ms (top-2048) | **8.8 ms (full 262K stripe)** |
| prefill | ~1,340 s | 983 s |

## Why the null is structural, not an engineering gap

1. **The TP-32 single-token all-reduce floor (~127 ms/step, 232 × ~0.55 ms) is common-mode** —
   invocation-count-identical in both arms (as are all-to-all 312 and psum 81). It is 64% of the
   dense step. No sparse-side work touches it; shrinking it (fused/reassociated per-layer
   reductions) helps both arms ~equally.
2. **Context parallelism already made the attend nearly free.** Dense attends its contiguous ~32K
   keys/rank in 8.8 ms — streaming HBM reads at full bandwidth. Sparse reads 16× fewer bytes but
   scattered, plus pays scoring, distributed exact top-2048, position restore, and selected-KV
   gathers: ~94 ms of selection to save an 8.8 ms attend.
3. **Therefore the ceiling of any exact-semantics sparse optimization is parity, not a win**: the
   recoverable sparse-specific cost is ~110 of the 120 ms/step gap (78% selection machinery, ~15 ms
   pure DMA-contention side effects); driving it to zero lands at dense's step, minus nothing —
   because the attend saving available is bounded by dense's 8.8 ms.
4. The crossover geometry (fixed dcp, growing L) sits far beyond 1M on v4: dense attend grows
   ~linearly from 8.8 ms while exact selection stays ~O(100 ms) — the measured slopes do not cross
   at any deployable context on this hardware.

## What the DSA sparse kernel's value actually is (all measured, all banked)

- **Selection quality:** the ≥95%@128K passkey gate CLOSED 77/77 (runs 293–314) — selected-set-exact
  selection preserving retrieval at depth, where gate2/gate3-era corruption once masqueraded as
  kernel failure.
- **256K correctness:** smoke 4/4 at mechanism depths; dcp=8 sparse serving works.
- **Prefill:** the 12.6× efficiency campaign made sparse prefill beat dense at 128K (chunked
  amortization is sparsity's friend; single-token decode is its enemy).
- **16× attend-FLOP reduction** — material on FLOP-bound (not latency-floor-bound) targets: the
  finding, inverted, is a hardware statement: *on pods where per-layer collective latency dominates
  single-token decode and context parallelism shards the KV, sparse attention's decode win
  evaporates; its wins live in prefill, quality, and FLOP-constrained regimes* (v6e/v7 with faster
  ICI + native FP8 shift the balance — the v4 gates in the code lift cleanly).

## Future-work levers recorded (not pursued for the gate)

1. Fused/reassociated per-layer all-reduce schedule (~127 ms common-mode target; helps dense too).
2. Fold the selected-KV gather into the `dsa_sparse_decode` Pallas kernel (index-driven DMA;
   removes the +37 ms materialized gather) — exact-semantics.
3. Fused partial top-2048 replacing XLA `top_k` + full position-sort (+35 ms) — exact-semantics.
4. The docs/18 ladder retains the judged designs; the measured re-rank demotes every top-k/scorer
   rung (ceilings ≤35 ms) and moots decode-side wins per §"Why the null is structural".

## Instrument legacy (what it took to measure this — docs/17-family lessons)

The two-day capture campaign produced six root-caused fixes (phase-profiler decode blindness via
the compiled-DAG path; the GC'd profiler-server handle — upstreamable one-liner; a forked fd
accepting-then-resetting; the wrong hook method; a None-attach; aged-out /tmp golden manifests
masquerading as TPU wedges) and three standing instruments: `GLM_JAX_TRACE` (in-worker decode-step
tracing, CPU-tested), self-healing mount/manifest keepers, and the reusable xplane parser. The
flight recorder's dispatch rule (`request_distribution[0] == num_reqs`) is the canonical decode
test for any future per-step hook.
