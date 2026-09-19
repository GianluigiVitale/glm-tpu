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

D5/D10 now implemented and measured (synthetic 78-layer step): **64.8–65.1 ms,
15.4 tok/s**, versus D1+D8 72.7–72.9 ms and frozen 121.1–121.3 ms in the same
run. D10 alone is 66.2–66.5 ms and preserves the recorded D8 token trail;
D5 changes the synthetic trail and remains a documented numerical boundary.
Receipt: docs/perf/tpu-microbench-d5-d10-step-20260919T135511Z.json.
CPU: 30 perf tests passed; release 524 passed, 1 skipped; frozen pin unchanged.
P1/P2 multirow primitives have CPU coverage. The opt-in P2 prefill builder
preserves all state/cache/token leaves bitwise in the CPU fixture, including
atomic refusal. P1 is also integrated behind `lse_attention=True` with CPU state/refusal and
selected-NaN tests. Full prefill still needs D8 residency and TPU admission.
Final D5 repeat is 65.3–65.9 ms with unchanged token trail (no extra speedup
from health packing). P1 attention is 3.92 -> 1.64 ms at M32/128K; P2 one-pass
is 2.06 ms, bitwise equal. Its 115.8 ms default-tiled comparison is NOT the
admitted paired/sorted configuration (23.2 ms measured, about 11.5x vs P2).
A generator audit then found unequal nominal replicas in the synthetic fixture.
The generator now folds only partitioned axes; CPU32 replica tests pass. Earlier
receipts remain exploratory; a corrected-replication confirmation is next.
See docs/perf/D5_D10_PROGRESS_20260919.md for scope and numerical boundaries.

## Next work, in order

(D9 closed: FP8 decode is at its v4 software floor, 43 of 46 us per 3 MB;
packed decode 2.5x slower. Routed experts stay decode-bound; only multi-row
steps (D7 MTP) or INT8 experts (non-exact) can cut them further.)
1. Finish D5 health-exchange and P1/P2 primitive TPU measurements. D5 is
   implemented; measured decode gain is ~1 ms, smaller than the old estimate.
2. D10 is implemented (-6.4 ms measured). Next: D4 host loop and fused
   q_a/kv_a/wk/head psums.
3. Prefill: apply D8 (BF16 tables already resident), then measure the integrated
   P1/P2 builder at 2K/128K. P1/P2 primitives are measured; their combined
   prefill program has CPU proofs but no complete-model TPU admission.
4. Real-weight validation of the challenger: acquisition, HLO/memory
   admission, 2K run against DB610 tokens, receipts, README table. D7 MTP last.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
