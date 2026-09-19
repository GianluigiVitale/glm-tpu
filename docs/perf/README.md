# Paused WS32 performance research — 2026-09-19

The trained GLM-5.2-FP8 challenger reached **138.85 prompt tokens/s** at 2,034
prompt tokens and **14.04 decode wall tokens/s** on 32 TPU v4 chips. All 29
DB610 reference tokens matched on all eight hosts. This is research evidence;
the supported inference command still executes the frozen admitted engine.

| Measurement | Result | Boundary |
|---|---:|---|
| D8/P1/P2 prefill | 138.85 prompt tok/s | B128 with B114 tail; includes block health votes; excludes cold load/compile |
| D1/D8/D10 model decode | 14.71–14.82 tok/s | Excludes host checks and delivery |
| Legacy request loop on that model | 13.32 wall tok/s | Host checks and in-memory delivery |
| D4 packed request loop on that model | 14.04 wall tok/s | Same boundary; 5.44% paired gain |

[Original paired receipt](tpu-real-request-loop-20260919T192804Z.json) records
five warm decode steps and 23 timed steps. Both loops pass 29/29 reference tokens,
health/finite checks and bitwise final-state/residual comparison. Peak measured
HBM is 28,228,678,144 bytes/chip at capacity 8,192. This does not establish
long-context admission, sampled task quality or network-delivered speed.

The source manifest is authoritative: every model/real-worker file matches
`5ff7b01e7a6520c652e7ce6dcc4a7012363e0ff8`; an unused synthetic microbenchmark
tool changed during snapshot preparation. [Source audit](tpu-real-request-loop-source-audit-20260919T192804Z.json).
The research graph checks do not inherit the protected release's HLO admission.

## Main versus preserved research

Main retains the supported engine and these compact results, the
[GLM-5.3 quantization assessment](GLM53_QUANT_ASSESSMENT_20260919.md), its public
metadata receipt, a trace supporting the assessment, and the cancellation record.
It does not silently replace the numerical source or claim the faster engine
is deployed. The full experiments, proofs, rejected variants and replay tools
remain at **`f493cbd56b5c7c72ce3d28455aec39b90140afbc`** on private branch
`perf/reference-lowhanging-fruit-20260919`.

| Preserved paths at that exact commit | Purpose |
|---|---|
| `glm_tpu/perf/`, `tests/perf/` | Challenger bodies, CPU proofs and unfinished prototypes |
| `tools/perf_real_validation.py` | Explicit real-weight validation worker |
| `tools/perf_tpu_microbench.py`, `tools/perf_op_census.py` | Research measurements and compiler census |
| `docs/perf/` | Full acquisition, failure, trace and cleanup history |
| `goal.md` | Paused plan and historical authority; not an active goal |

Decode D5 failed the real token trail and remains excluded from the passing
candidate. Earlier synthetic grouped-expert timings included an empty-owner
store bug; they are not correctness-qualified speedups. Wide prefill and the
newest primitive candidates lack completed TPU comparisons. None is promoted
by this documentation checkpoint. Recover code without modifying main using
`git show f493cbd56b5c7c72ce3d28455aec39b90140afbc:<path>` or a separate worktree.

## Pause and review

The owner cleared the goal and requested assessment of GLM-5.3 on 2026-09-19.
The active paired experiment was stopped; two queued comparisons were cancelled
before acquiring a workload lease. Compact feature reduction was never launched
on TPU. All eight hosts were subsequently authenticated idle.
[Cancellation receipt](tpu-owner-pause-20260919.json). Originals and existing
weights remain intact. No GLM-5.3 tensor payload was acquired or executed.

This curation starts from main `493b67de` (616 files) and adds seven compact
documents/receipts, for 623 tracked files. No existing file is removed and no
runtime, test, dependency or configuration changes. The review covers the
changed prose, generated receipt provenance/structure, local links, unchanged
numerical source and the private recovery pin; it is self-review, not independent
review or a repeat of every previous implementation audit. New ledger entries
state their specific consumers. CPU release checks and final regional backup
verification are required before main publication; final-pin receipts live
outside Git under `gs://driftbench-dsv4-uc/results/perf_checkpoint_20260919/`.
