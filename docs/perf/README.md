# WS32 performance checkpoint — 2026-09-19

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

This checkpoint retains the supported engine, the paired measurement/source
receipts above and the historical cancellation record. The faster engine is
preserved on private branch `perf/reference-lowhanging-fruit-20260919`;
it is not deployed by this documentation update. The complete ordinary-path
experiments, proofs, rejected variants and replay tools are recoverable at
**`f493cbd56b5c7c72ce3d28455aec39b90140afbc`**. The later MTP research checkpoint
is **`f097649a`**, containing the completed native comparisons, CPU proofs,
source audit, rejected variants and numerical limits.

| Preserved paths at `f493cbd5` | Purpose |
|---|---|
| `glm_tpu/perf/`, `tests/perf/` | Challenger bodies, CPU proofs and unfinished prototypes |
| `tools/perf_real_validation.py` | Explicit real-weight validation worker |
| `tools/perf_tpu_microbench.py`, `tools/perf_op_census.py` | Research measurements and compiler census |
| `docs/perf/` | Full acquisition, failure, trace and cleanup history |
| `goal.md` | Historical broad plan at this recovery pin; superseded by the current branch's MTP goal |

Decode D5 failed the real token trail and remains excluded from the passing
candidate. Earlier synthetic grouped-expert timings included an empty-owner
store bug; they are not correctness-qualified speedups. Wide prefill and the
newest primitive candidates lack completed TPU comparisons. None is promoted
by this documentation checkpoint. Recover code without modifying main using
`git show f493cbd56b5c7c72ce3d28455aec39b90140afbc:<path>` or a separate worktree.

## Completed MTP experiment and historical pause

The owner cleared the broad performance goal on 2026-09-19.
The active paired experiment was stopped; two queued comparisons were cancelled
before acquiring a workload lease. Compact feature reduction was never launched
on TPU. All eight hosts were subsequently authenticated idle.
[Cancellation receipt](tpu-owner-pause-20260919.json). Originals and existing
weights remain intact. This receipt proves cleanup at cancellation, not the
fleet's present status.

The owner subsequently activated native MTP/speculative decoding on the same
GLM-5.2-FP8 model and 32 TPU v4 chips. The two proposed GLM-5.3 repositories are
out of scope. Follow `goal.md` and `docs/perf/MTP_PROGRESS_20260919.md` in the
research worktree for live acquisitions and next actions. The trained verifier
has matched the short DB610 successor trail, with numerical cache differences;
native drafting has now executed in a completed long comparison. Ordinary
14.2902 wall tok/s exceeded one-draft 13.0927 and two-draft 12.7682. Both
speculative trails diverge from ordinary at token index 6, and all three ended
unfinished at the output cap. The [MTP comparison](MTP_COMPARISON_20260920.md)
records that receipt and the completed representative repeats. Two drafts lose
5.9–6.6% on prose, gain 4.3–4.6% on code and 10.7–10.9% on structured output.
The 25% working target is unmet; speculative trails differ from ordinary.
Prose needs scoped corrections, code remains unfinished, and structured values
are correct but Markdown fences fail the requested standalone format. Both runs
completed authenticated eight-host cleanup. The implementation and rejected
experiments remain on the research branch; ordinary stays the DB610-qualified
research baseline.

## Checkpoint review

This curation starts from main `493b67de` (616 files) and adds four compact
documents/receipts, for 620 tracked files. No existing file is removed and no
runtime, test, dependency or configuration changes. The review covers the
changed prose, generated receipt provenance/structure, local links, unchanged
numerical source and the private recovery pin; it is self-review, not independent
review or a repeat of every previous implementation audit. New ledger entries
state their specific consumers. The 2026-09-20 CPU release check passes:
524 tests passed, one skipped, frozen-source/content/package checks passed.
Changed-document links resolve; the paired receipt matches all eight preserved
original rank hashes and its source manifest. Main publication requires verified
regional backup; the authoritative final-pin receipts live
outside Git under `gs://driftbench-dsv4-uc/results/perf_checkpoint_20260919/`.
The original staged checkpoint is preserved at `132dc75b` on
`preserve/perf-checkpoint-draft-20260919`; its superseded scope/status text is
not current guidance. Out-of-scope assessment files and a supporting synthetic
trace remain recoverable in research history instead of being added to main.

The subsequent MTP evidence checkpoint starts from private main `5e9ce605` and
adds the comparison document plus two compact completed receipts, for 623
tracked files. It changes no runtime, test, dependency or configuration source.
Originals, research and DB616–621 remain preserved; no files are deleted.
Final release/self-review, leased regional checksum/generation verification and
fresh idle checks bind publication to an exact pin. Those final records live
under `gs://driftbench-dsv4-uc/results/mtp_evidence_20260920/`. An absent final
record does not establish successful publication.
