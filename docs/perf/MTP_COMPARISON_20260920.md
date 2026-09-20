# Native MTP speculation on 32 TPU v4 — 2026-09-20

The first long GLM-5.2-FP8 comparison favors ordinary decoding. Native MTP
served as the drafter inside target-verified greedy speculation; it was not
a separate unverified output mode. The experimental implementation remains
on private branch `perf/reference-lowhanging-fruit-20260919`.

## Completed long comparison

All modes used the same 338-token prompt, capacity 8,192 and 6,144-token output
budget. Each received fresh prompt caches and the same greedy policy. All eight
hosts completed source/checkpoint/native-pack verification, graph and memory
admission, output agreement within each mode, and authenticated cleanup.

| Mode | Wall output tok/s | Relative to ordinary | Output versus ordinary |
|---|---:|---:|---|
| Ordinary | 14.2902 | 1.0000x | Baseline |
| One native draft, two target rows | 13.0927 | 0.9162x | First mismatch at index 6 |
| Two native drafts, three target rows | 12.7682 | 0.8935x | First mismatch at index 6 |

[Eight-host receipt](tpu-real-native-mtp-20260920T015817Z.json), executed from
immutable source `bcec7ddd`. The rate counts 6,143 timed decode tokens after
first-token delivery. It includes native drafting/refresh, target verification,
rejected work, cache commits, host votes/agreement and rank0 local JSONL
write/flush. Prefill/bootstrap, cold loading/compilation and network transport
are excluded. Reasoning tokens are included in generated output counts.

One-draft acceptance was 2,788/3,354 (83.12%), averaging 1.8310 tokens/round.
Two-draft acceptance was 2,071/2,662 at the first position and 1,410/2,662 at
the second, averaging 2.3077 tokens/round. Target verification alone cost
approximately 406/426 seconds; total speculative decode wall was 469/481
seconds versus ordinary 430. High draft acceptance did not overcome verifier
and request overhead. Native prompt bootstrap added 0.681–0.731 seconds.
Peak allocator use remained 28,228,678,144 bytes/chip in this acquisition.

The short DB610 check matched all 29 reference tokens in every mode. Its
ordinary/one-draft/two-draft rates were 13.0775/12.6385/14.3078 wall tok/s,
so a short 9.4% two-draft gain did not carry over to the long prompt.

## Correctness and recommendation

All three long responses reached the output cap during reasoning, without a
finished answer. Exact private answer oracles therefore establish no completed
answer correctness. Token equality and task-answer correctness are separate:
matching DB610 does not prove long-request equivalence or broad model quality.

The multi-row verifier has a documented floating-point boundary; the observed
first divergence has not been causally isolated. Neither speculative path is
qualified as a token-exact replacement. Keep the ordinary research path as
the qualified performance candidate. The supported main inference command
continues to use its existing admitted engine.

## Representative repeats — pending

A separate immutable run on prose, code and structured output has started,
with two paired repetitions per case and matched 7,168-token caps. Its results,
completed-answer assessments and authenticated cleanup remain pending.
This section must be reconciled before final publication; the first long
request alone is not the full representative comparison.

## Recovery and publication scope

The completed implementation, CPU proofs, upstream/source audit, rejected
variants and raw-evidence recovery paths are preserved at research checkpoint
`dc047933`. Use a separate worktree or `git show dc047933:<path>` for
`glm_tpu/perf/`, `tests/perf/`, `tools/perf_real_validation.py` and `docs/perf/`.
The native execution pin `bcec7ddd` is distinct from the later summarizer pin.
The receipt records archived source and summary-file hashes.

This candidate checkpoint contains documentation and compact evidence only.
It does not deploy the experimental engine, alter frozen `MODEL_SOURCE`
`edecdd94`, or replace DB616–621 evidence. Final review, release checks,
regional backup verification and main publication are still pending.
