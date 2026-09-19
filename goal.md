# Goal — Make WS32 prefill and decode fast (GLM-5.2-FP8 on 32 TPU v4)

Owner objective (2026-09-19): raise prefill and decode speed as far as the
hardware allows, learning from ARahim3/kaggle-tpu-lab (glm53-flash ~1,600
tok/s prefill, 64 tok/s decode; qwen38-27b 10,300 / 130 tok/s; 8 v5e chips).
Baselines: 62.8 tok/s
prefill at 2K (45.5 at 128K, 32.2 at 256K); 7.7 tok/s decode (6.6 sampled). Targets: thousands of prompt tok/s, 30-60+ generated tok/s.
Read AGENTS.md, HANDOFF.md, docs/perf/REFERENCE_LOWHANGING_FRUIT_20260919.md
first (the ranked plan and evidence).

## Authority

TPU runs ARE allowed: all 32 v4 chips of pod db-v4-64-od (8 hosts x 4 chips,
zone us-central2-b, SSH via `gcloud compute tpus tpu-vm ssh db-v4-64-od
--worker=all`). Use them for microbenchmarks, acquisitions, layer timings
and full runs. One workload at a time under ~/.glm-tpu-workload.lock; check
the fleet is idle first; leave all 8 hosts clean afterwards. Never
create/delete/resize TPU/VM/queued resources. Work autonomously: take the
decision you would recommend and continue; commit and push to origin (private)
on the perf branch without asking; no force-push, no Co-Authored-By lines.
pytest stays JAX_PLATFORMS=cpu.

## State

Branch `perf/reference-lowhanging-fruit-20260919`, from main `493b67de`.
Research remains outside frozen MODEL_SOURCE `edecdd94`; release checks pass
(524 passed, 1 skipped). Originals, failed experiments and historical receipts
are preserved. Twenty-eight completed acquisitions have authenticated all-eight-host
cleanup in `docs/perf/tpu-workload-cleanup-20260919.json`.

**Trained-weight D1/D8/D10 passes all 29 DB610 tokens on all eight hosts:**
138.95 prompt tok/s at 2K, 14.55–14.69 model decode tok/s, decode p50
66.98–67.70 ms; peak HBM 28.229 GB/chip. Decode timing excludes host checks
and delivery. Receipt: `docs/perf/tpu-real-no-d5-20260919T184958Z.json`.
Prefill includes D8 resident weights and P1/P2. This is short-trail parity,
not a general quality claim. D4 now passes the same trail in both request loops with bitwise final state/residual:
**13.32 -> 14.04 wall decode tok/s (+5.44%)**, 23 timed steps after five warm,
including host checks and in-memory delivery, excluding transport. Receipt:
`docs/perf/tpu-real-request-loop-20260919T192804Z.json`.

The grouped-MoE empty-owner store bug is fixed and TPU-proved (zero empty
outputs, bitwise-equal live outputs). Earlier grouped-MoE timings, including
72.1 ms and 64.3 ms synthetic steps, are not correctness-qualified speedups.
Full-trail ablations isolate decode D5 as the remaining divergence: D1/D8 and
D1/D8/D10 pass 29/29; D1/D8/D5 passes 17/29. D5 stays disabled. A closer-scale
global-tile attention alternative is slower on TPU (134 vs frozen 118 us) and
remains outside model builders. See `docs/perf/REAL_WEIGHT_VALIDATION_20260919.md`
and `docs/perf/D5_D10_PROGRESS_20260919.md` for receipts and numerical boundaries.

Complete synthetic prefill measures 123.32 prompt tok/s at 2K with canonical
B128, 174.13 with pooled B512, and 85.44 at 128K with B128. These are synthetic
measurements; B512's first token differs from B128. Receipts under `docs/perf`:
`tpu-microbench-prefill-model-20260919T150628Z.json`,
`tpu-microbench-prefill-pooled512-20260919T183523Z.json`, and
`tpu-microbench-prefill-128k-20260919T152806Z.json`.
Adding bounded owner capacity 512 reduces later B512 blocks from 2.925–2.961 s
to 1.971–2.007 s. Its whole-prompt timing includes staggered profiler teardown,
so no clean full-prompt rate is claimed. Its device trace timelines are empty;
no bottleneck breakdown is inferred. Timing now has a fleet barrier after
profiling. Receipts: `tpu-microbench-prefill-pooled512-owned512-20260919T192803Z.json`
and `tpu-trace-prefill-bounded-audit-20260919T192803Z.json` under `docs/perf`.

P3 wide sparse IndexShare prefixes have complete CPU32 bitwise proofs:
pooled B242 vs B128+B114 from both empty and 512-token prefixes, including all
caches, selections, scores, tokens, frontiers and finished-state refusal.
Canonical tail padding is required. Full DSA/M64 repair and dense placement
remain narrow. TPU admission, timing and trained-weight parity are pending;
the option stays off by default. The short synthetic eight-layer trace completed with usable device events; the
paired full-model narrow/wide acquisition is running. P5 feature-row attention
now passes the complete populated-prefix CPU32 bitwise proof; its primitive TPU
comparison is queued. It divides replicated query rows over feature4 and restores
all result/health rows before projection and commit. See
`docs/perf/D4_P4_PROGRESS_20260919.md` for implementation and scope.

D9 is closed: v4 FP8 software decode consumes 43 of 46 us per 3 MB;
packed decode is 2.5x slower. D4 previously improved synthetic sampled wall
throughput 14.11 -> 14.71 tok/s with identical tokens/final state; trained-weight
comparison is now recorded above. P4 N512 panels are slower and rejected. Fused feature
reductions fail the composed TPU step and remain disabled.

## Next work, in order

1. Obtain a usable sparse-layer profile and clean paired narrow/wide B512
   timings with bounded owner attention. Validate the best prefill candidate
   with trained weights, both B512 and B498 graphs, against DB610.
2. Profile and improve the measured 128K path, preserving canonical dense
   placement and all cache/repair/health guards. Decode D5 remains experimental;
   no frozen body is promoted without proof and measurement.
3. D7 MTP last, after the trained-weight challenger. Routed FP8 remains
   decode-bound; multi-row execution is the remaining exact route to amortize it.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
