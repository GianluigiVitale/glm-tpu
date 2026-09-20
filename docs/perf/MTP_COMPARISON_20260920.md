# Native MTP speculation on 32 TPU v4 — 2026-09-20

The completed GLM-5.2-FP8 comparison does not reach the 25% wall-throughput
target. Two-draft speculation loses on prose and gains 4.3–10.9% on the code
and structured prompts; it does not preserve ordinary token output. Native MTP
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

Measured average rounds cost 139.85 ms (one draft) and 180.74 ms (two drafts).
Holding those costs fixed, perfect acceptance would estimate only 14.30 and
16.60 tok/s respectively, versus ordinary 14.29. The two-draft path would need
2.583 tokens/round merely to break even; it emitted 2.308. A 25% gain would
require 3.228 tokens/round at that cost, exceeding its three-row output limit.
These are fixed-cost estimates from this workload, not measured perfect
acceptance or a hardware-wide speed limit: different token trajectories and
expert routes can change the cost. Better draft accuracy alone is insufficient
for the working 25% target at the measured round cost.

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

## Completed representative repeats

The [eight-host representative receipt](tpu-real-native-suite-20260920T031727Z.json)
records two fresh requests per prompt, from immutable execution `dc047933`.
All modes use the same prompt, greedy policy, capacity 8,192 and 7,168-token
output cap, stopping normally at EOS. All 17 graphs passed source, collective
and memory checks; DB610 matched 29/29 in every mode. All eight hosts completed
and were authenticated idle. No further workload was queued.

| Request | Ordinary wall tok/s | One draft | Two drafts | Two-draft paired change |
|---|---:|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 | −6.6% to −5.9% |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 | +4.3% to +4.6% |
| Structured output | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 | +10.7% to +10.9% |

Ranges span the two requests and synchronized host reports. The eight hosts are
not independent trials; these are descriptive ranges, not confidence intervals.
Every mode reproduced its own token trail exactly on repetition. Both speculative
modes differ from ordinary: first mismatch index 6 on prose, 5 on code, and
816/823 for one-/two-draft structured output. The multi-row floating-point
boundary is documented in research; it has not been isolated as the sole cause.

| Request | R2 accepted first drafts | R3 accepted first / second drafts | R2 / R3 tokens per round |
|---|---:|---:|---:|
| Prose | 2,801/3,332 | 1,246/1,545 · 954/1,545 | 1.841 / 2.423 |
| Code/reasoning | 3,385/3,782 | 2,425/2,634 · 2,108/2,634 | 1.895 / 2.721 |
| Structured output | 1,189/1,201 | 687/699 · 659/699 | 1.989 / 2.924 |

These counts repeat exactly. Second-position acceptance uses all proposals at
that position as its denominator, including rounds rejected earlier. Target
verification dominates: in first-repeat R3 it took about 246/278 decode seconds
on prose, 425/480 on code, and 114/129 on structured output. The receipt retains
draft, verification, refresh, commit, host-vote/agreement and total wall costs.
Maximum recorded native peak HBM remained 28,228,678,144 bytes/chip.

The new prompts contain 204/347/696 tokens. Ordinary first-repeat prefill took
2.16–2.45 / 3.19–3.50 / 6.09–6.25 seconds across hosts. Ordinary warmed TTFT was
2.41–2.72 / 3.45–3.77 / 6.35–6.54 seconds; two-draft TTFT was
2.62–2.73 / 3.90–4.05 / 7.50–7.70 seconds. Native bootstrap adds roughly
0.4/0.7/1.4 seconds on these prompts. These measurements do not establish a
prefill speedup. Cold loading/compilation and network transport remain excluded.
Ordinary per-token p50/p99 and speculative per-round p50 are retained separately
in the receipt; a round latency is not a per-token latency.

## Completed-answer checks

- **Prose:** all modes ended at EOS, but scoped assistant self-review identified
  technical corrections in each distinct answer. Reviews cover cache mechanics,
  acceptance/correction, rollback, arithmetic and latency; they are hash-bound
  to the private responses. They are not independent reviews or model-wide scores.
- **Code/reasoning:** all six responses exhausted 7,168 tokens during reasoning.
  None delivered a final schedule or function. The private exact scheduling
  oracle therefore establishes no finished-answer correctness, and no generated
  function was executed. Intermediate reasoning is not counted as a solution.
- **Structured output:** all six ended at EOS with exact correct JSON values,
  types and ordering. Markdown code fences violate the requested standalone JSON
  format. This is a format failure, not an arithmetic failure; removing only the
  fences yields the exact oracle result.

Generated counts differ by mode: prose 2,655/6,134/3,745; code 7,168 in every
mode; structured 2,629/2,390/2,045. Reasoning tokens are included. Throughput
compares generated-token rates, not equal-answer completion time. The same
formatting/quality boundaries recur on both repetitions. These three prompts
and the unfinished long question do not establish general model quality.

Keep ordinary decoding as the DB610-qualified research baseline. Native MTP
speculation is a preserved experiment, with a small prompt-dependent benefit
and unresolved output equivalence. It does not meet the working 25% criterion
or the 30+ tok/s aspiration. Wider verification was already rejected by its CPU
numerical boundary; no rejected variant is promoted to manufacture a speedup.

## Recovery and publication scope

The completed implementation, CPU proofs, upstream/source audit, rejected
variants and raw-evidence recovery paths are preserved at research checkpoint
`f097649a`. Use a separate worktree or `git show f097649a:<path>` for
`glm_tpu/perf/`, `tests/perf/`, `tools/perf_real_validation.py` and `docs/perf/`.
The native execution pins `bcec7ddd` and `dc047933` are distinct from the later
summary/research checkpoint.
The receipt records archived source and summary-file hashes.

This checkpoint contains documentation and compact evidence only.
It does not deploy the experimental engine, alter frozen `MODEL_SOURCE`
`edecdd94`, or replace DB616–621 evidence. Originals and rejected research remain
preserved. The final publication is bound to release/self-review, leased regional
checksum/generation verification and fresh authenticated idle checks. Its exact
main pin and backup receipts are recorded outside Git under
`gs://driftbench-dsv4-uc/results/mtp_evidence_20260920/`; an absent publication
receipt must not be interpreted as a successful merge.
