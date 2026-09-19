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

## Next work, in order

(D9 closed: FP8 decode is at its v4 software floor, 43 of 46 us per 3 MB;
packed decode 2.5x slower. Routed experts stay decode-bound; only multi-row
steps (D7 MTP) or INT8 experts (non-exact) can cut them further.)
1. D5 LSE-merge attention (queries gathered, local partial softmax,
   psum_scatter outputs) replacing the 2.6 MB selected-KV psum (-5 ms decode;
   84 MB per 32-row tile in prefill). Primitives exist.
2. D10 DSA: score against cache pages without the full-cache gather; two-stage
   top-k with cut check (-5..-7 ms). D4 host loop; fuse q_a/kv_a/wk/head psums.
3. Prefill: apply D8 (BF16 tables already resident) and P1/P2 (LSE merge; one
   top_k per tile + two-stage merge; repo estimate 1,772 of 2,802 s at 128K).
4. Real-weight validation of the challenger: acquisition, HLO/memory
   admission, 2K run against DB610 tokens, receipts, README table. D7 MTP last.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
