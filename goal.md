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

Branch perf/reference-lowhanging-fruit-20260919 from main 493b67de. New code
sits outside the frozen MODEL_SOURCE pin (edecdd94): glm_tpu/perf, tests/perf,
tools/perf_op_census.py, tools/perf_tpu_microbench.py; `python
tools/check_release.py` still passes. Measured on the pod, synthetic weights
at real geometry (docs/perf/tpu-microbench-*.json, tpu-trace-*.json): frozen
78-layer greedy step 120.3 ms (8.3 tok/s); + grouped MoE (D1) 105.7 ms; +
BF16-resident non-routed weights (D8, exact decode, +1.8 GB/chip) **72.1 ms
(13.9 tok/s)**. Trace of the 72 ms: collectives 35 ms (routed-expert psum
17.9 ms = straggler wait for the busiest owner; selected-KV 2.6 MB psum
6.6 ms), routed FP8 kernels 12 ms, gathers 6.4 ms, DSA sort/top-k 5.4 ms.
TPU v4 decodes FP8 on the VPU at ~40 us per 3 MB; BF16 tables cost 10 us.
D1/D2/D8 have CPU proofs (D1/D2 bitwise; D8 same tokens, KV within 1 ulp).

D5/D10 implemented and confirmed with a corrected synthetic generator:
**64.30–64.43 ms/token (15.52–15.55 tok/s)** for the 78-layer step, versus
D1+D8 72.55–72.90 ms and frozen 121.20–121.54 ms. Receipt:
`docs/perf/tpu-microbench-replica-correct-20260919T143305Z.json`. Thirty timed
steps; p99 is 84.9–85.6 ms, worse than D1+D8's 79.7–80.2 ms. D5 remains a
numerical boundary; real weights have not been validated. The original fixture
seeded nominal replicas differently; it is fixed and CPU32-tested. Older
synthetic receipts are preserved as exploratory measurements.

At M32/full-128K, corrected P1 attention is **3.92 -> 1.64 ms**; P2 DSA is
**23.24 -> 2.04 ms (~11.4x)** against the admitted paired/sorted tiled settings,
with bitwise-equal results. These are primitive timings, not prompt tok/s.
P1/P2 are integrated in `build_ws32_prefill_challenger_program` (P1 explicit
`lse_attention=True`); CPU state/cache/token/refusal and selected-NaN tests pass.
D8 residency was not integrated at that measurement; follow-up below.
Release checks: 524 passed,
1 skipped; frozen source pin unchanged. Those acquisitions ended with all eight hosts verified idle;
`docs/perf/tpu-workload-cleanup-20260919.json` records the acquisitions.
See `docs/perf/D5_D10_PROGRESS_20260919.md` for evidence and numerical boundaries.

D8 prefill is now CPU-validated and integrated (`bf16_resident=True`): shared
resident weights with decode, original M64 repair and canonical B114/B128 dense
placement, plus routed expert panels. Code checkpoint `8fd462cf`. Complete 2K
synthetic prefill improved **64.14 -> 123.32 prompt tok/s (1.92x)**, all 78
layers, capacity 2,560, all-rank health true. Peak allocator 28.44 GB/chip;
receipt `docs/perf/tpu-microbench-prefill-model-20260919T150628Z.json`.
Synthetic final tokens differ; trained-weight validation remains open.
Fused Q/KV/WK/head reductions
passed CPU bitwise proofs but **failed the full TPU trial** (all-zero token
trail, 122.4 ms); they remain disabled. Isolated exact-geometry TPU projections
are bitwise equal, so the composed-step failure still needs localization.
The unfused challenger repeated
**64.12–64.24 ms, 15.57–15.60 tok/s** over 100 timed steps. See
`docs/perf/D4_D8_PROGRESS_20260919.md` for the rejected candidate and boundaries.
Those nine acquisitions ended with all eight hosts idle.

Complete synthetic 128K prefill is now measured: **85.44 prompt tok/s**, 1,534.11 s
for all 131,072 tokens/78 layers, capacity 131,584, all-rank health/admission true.
Peak allocator 30.210 GB/chip, 2.804 GB headroom. Receipt:
`docs/perf/tpu-microbench-prefill-128k-20260919T152806Z.json`. This is not a paired
real-weight speedup. D4 compact host loop is TPU-proved: **14.11 -> 14.71 sampled wall tok/s**
(+4.29%), same tokens and bitwise final state/residual. Receipt:
`docs/perf/tpu-microbench-request-loop-20260919T153427Z.json`. P4 N512 panels
are TPU-bitwise but rejected: gates are neutral and down projections 10–25%
slower. The frozen N256 panel remains in use. Bounded owner attention is TPU-bitwise with fallback: balanced median
**1.74 -> 0.97 ms**, but worse p99 (3.38 vs 1.99 ms); mixed/concentrated cases
remain near baseline. The prefill builder now has a CPU-bitwise B114/B128 integration option,
remaining off by default and pending full-model TPU trials. See `docs/perf/D4_P4_PROGRESS_20260919.md`. All 14 recorded acquisitions ended
with authenticated idle on all eight hosts at that checkpoint.

Real-weight DB610 acquisition completed: **138.61 prompt tok/s** and
**15.18–15.28 model decode tok/s**, peak 28.229 GB/chip, all graph/memory/health
checks passed. **Numerical validation failed:** prefill token matched, then all
28 decode outputs were zero; all eight hosts agreed. Candidate remains rejected,
not a serving speedup. Receipt `docs/perf/tpu-real-db610-20260919T163936Z.json`.
The layerwise diagnostic localizes the failure to the first sparse MoE
(layer 3): update 5.45e35 from normalized input <=2.3125; the following RMSNorm
collapses to zero. Split execution has the same failure. Receipt
`docs/perf/tpu-real-diagnostic-20260919T171314Z.json`. An explicit store for the
forced empty-owner grid row passed 21 CPU tests and the paired TPU probe:
original empty outputs are garbage on all eight hosts; fixed outputs are exact
zero on all 32 chips, with bitwise-equal live projections. Receipt
`docs/perf/tpu-microbench-empty-routes-20260919T173208Z.json`. A corrected real
comparison is next. Earlier grouped-MoE timings remain affected by this bug and
are not correctness-qualified speedups. All 18 completed
acquisitions ended with authenticated idle on all eight hosts. See
`docs/perf/REAL_WEIGHT_VALIDATION_20260919.md`.

## Next work, in order

(D9 closed: FP8 decode is at its v4 software floor, 43 of 46 us per 3 MB;
packed decode 2.5x slower. Routed experts stay decode-bound; only multi-row
steps (D7 MTP) or INT8 experts (non-exact) can cut them further.)
1. Diagnose and correct the all-zero real-weight decode trail, then repeat
   DB610 token validation. Add first-step activation diagnostics and distinguish
   a fused-executable failure from arithmetic/handoff differences. Include the
   proven D4 host loop only after model admission. No frozen body is promoted.
2. Prefill: profile and improve the measured 85.44 tok/s synthetic 128K path,
   retaining canonical-dense placement and cache/repair/health guards. P3 and
   bounded owner attention remain candidates; P4 N512 is rejected by TPU timings.
3. Localize the TPU-only failure of fused feature reductions if its potential
   gain warrants another diagnostic. D7 MTP stays last, after real weights.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
