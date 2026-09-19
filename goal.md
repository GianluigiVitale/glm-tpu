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
1 skipped; frozen source pin unchanged. All eight hosts are verified idle;
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
All eight hosts are idle; the cleanup receipt now covers nine acquisitions.

## Next work, in order

(D9 closed: FP8 decode is at its v4 software floor, 43 of 46 us per 3 MB;
packed decode 2.5x slower. Routed experts stay decode-bound; only multi-row
steps (D7 MTP) or INT8 experts (non-exact) can cut them further.)
1. D4 host loop; diagnose the TPU-only failure of fused feature reductions.
   D5/D10 and P1/P2 primitive measurements are complete; no frozen body has
   been promoted.
2. Prefill: measure the integrated D8/P1/P2 builder at 128K, retaining
   canonical-dense row placement and all cache/repair/health guards. D8 CPU
   integration and synthetic 2K measurement are complete; synthetic 128K and
   real-weight admission remain open. P3/P4 are the next prefill levers after
   measuring the integrated long-context path.
3. Real-weight validation of the challenger: acquisition, HLO/memory
   admission, 2K run against DB610 tokens, receipts, README table. D7 MTP last.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
