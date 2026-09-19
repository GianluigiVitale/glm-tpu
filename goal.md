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
tools/check_release.py` still passes. Measured on the pod (synthetic weights,
docs/perf/tpu-microbench-*.json): frozen 78-layer step 120.7 ms; frozen MoE
layer body 0.444 ms (1.90 concentrated) vs grouped kernel 0.356 (1.37), exact;
sampler head frozen 0.98 ms vs candidates 0.69; a one-row FP8 projection costs
43-60 us in every decoding variant but 10 us from a resident BF16 table: TPU v4
decodes FP8 on the VPU, ~40 ms of the step. D1/D2 implemented and proven
bitwise-equal on CPU and pod.

## Next work, in order

1. D8 pre-decoded BF16 residency for all non-routed weights (+1.8 GB/chip,
   exact: bf16(f32(bits)*scale) is what the kernel computes): perf mirror of
   the attention/shared/dense bodies with plain dots, one-time device decode
   at load; expect ~-22 ms/step. Then a packed FP8->BF16 decode kernel for
   the routed experts (~9 ms).
2. Finish the challenger step measurement (phase 5) and trace it per host
   (--trace); parse with scripts/analysis/parse_xplane.py for the category
   breakdown; put the receipt in docs/perf.
3. D5/P1 LSE-merge attention replacing the zero-padded selected-KV psum
   (2.6 MB/layer decode, 84 MB per 32-row tile prefill); primitives exist.
4. P2 prefill DSA selection: one top_k per tile + two-stage merge with cut
   check (repo estimate 1,772 of 2,802 s at 128K).
5. D4 host loop (fused reads, device-resident uniforms, one vote), D6 decode
   DSA two-stage top-k, P3/P4/P5 wider prefill tiles / GEMM rows / pieces.
6. Real-weight validation: acquisition, HLO/memory admission, 2K run against
   DB610 tokens, receipts, README table. D7 MTP speculative decoding last.

## Rules

Exactness first: a CPU bitwise proof or documented numerical boundary, then a
TPU measurement, before any swap replaces a frozen body. Keep weights,
credentials, private prompts and raw DBs out of Git; only
gs://driftbench-dsv4-uc (US-CENTRAL2). Preserve originals, research branches,
history and DB616-621 evidence. Estimates are estimates; measured numbers cite
their receipt. Update the State section as items land.
