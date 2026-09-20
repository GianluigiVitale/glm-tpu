# Goal — Faster decode with speculative decoding / MTP on GLM-5.2-FP8, 32 TPU v4

Owner objective (2026-09-19): test whether speculative decoding, preferably the
current model's native multi-token prediction (MTP) layer, significantly improves
accepted output tokens per second on our existing GLM-5.2-FP8 / WS32 setup.
Keep the current model and hardware. Ignore
`vikasclawd/GLM-5.3-Int4-Int8Mix` and `Tech2wild/GLM-5.3-Int4-Int8Mix`:
do not investigate, acquire, port or switch to either repository for this goal.

Activated through the owner's goal objective on 2026-09-19. The previous broad
performance goal was cleared, its active run stopped, and its queued runs
cancelled. Follow only this MTP/speculation scope; do not resume unrelated queues.

Read AGENTS.md, HANDOFF.md, docs/perf/REAL_WEIGHT_VALIDATION_20260919.md and
docs/perf/D4_P4_PROGRESS_20260919.md first. Use the earlier ranked plan as
background; this goal moves D7 MTP/speculation to the front. Do not resume the
unrelated wide-prefill or primitive queues automatically.

## State and baseline

Research branch `perf/reference-lowhanging-fruit-20260919`, worktree
`/home/gianl/glm-tpu-perf-ref`, from main `493b67de`. Sustained-baseline milestone
`c594aaf0` is committed/pushed. Frozen `MODEL_SOURCE` remains `edecdd94`.
Preserve all history, including paused snapshot `f493cbd5` and rejected trials.

**Qualified ordinary baseline:** D1 grouped experts with empty-owner fix, D8
BF16-resident non-routed weights, D10 DSA, D4 packed host loop. Prefill uses
D8/P1/P2 with canonical B128/B114. At 2,034 prompt tokens:

- **138.85 prompt tok/s; 14.04 wall decode tok/s**, including host checks and
  in-memory delivery, excluding network transport and cold load/compile.
- All 29 DB610 tokens match on eight hosts; legacy/packed final state and
  residual are bitwise equal. Five warm steps and 23 timed steps.
- Model-only decode: 14.71–14.82 tok/s. Peak HBM: 28,228,678,144 bytes/chip
  at capacity 8,192. New draft weights/caches/temporaries need new admission.
- Receipt: `docs/perf/tpu-real-request-loop-20260919T192804Z.json`;
  source audit: `docs/perf/tpu-real-request-loop-source-audit-20260919T192804Z.json`.
  Measured model/worker match `5ff7b01e`; the changed microbenchmark was unused.

**Verifier measured on trained weights:** latest immutable `a7b1ca1b` run
`perf_real_mtp_verifier_m8_20260919T235646Z` completed with all eight hosts idle.
Two-/three-row candidates match all 28 DB610 successors, but caches and selected
position arrays differ. Perfect-acceptance estimates including padded tail are
16.39–16.45 / 17.44–17.50 tok/s (1.084–1.089x / 1.151–1.155x paired model calls).
Three-row full blocks estimate 18.62–18.69 tok/s (1.229–1.233x). These exclude
native drafting, refresh, host votes and delivery: **no accepted speculative
speedup is established**. Whole-cache error includes unchanged prompt rows;
future workers also measure the written span. Receipt:
`docs/perf/tpu-real-mtp-verifier-m8-20260919T235646Z.json`.
Five-row verification remains outside its CPU numerical envelope and rejected.

**Native MTP components implemented and CPU-tested:** input projection;
canonical target prompt hidden export; separate one-layer transformer;
full-index refresh and recurrent IndexShare; native source placement/binding.
The native head norm is used once and target embedding/head arrays are shared.
Synthetic checks cover composition, causality, skipped DSA, prefix rollback,
invalid-input refusal and ordinary prefill state/token agreement. Receipts:
`docs/perf/mtp-projection-cpu-20260920.json`,
`docs/perf/mtp-prefill-export-cpu-20260920.json`,
`docs/perf/mtp-native-components-cpu-20260920.json`.
These are not independent trained native-model or end-to-end acceptance proofs.

The authenticated source inventory/headers contain all 1,569 layer-78 tensors
(10,032,632,960 source bytes); the retained runtime pack omits MTP. All source
intervals reconcile into 39 raw tables per chip, 363,837,792 raw bytes/chip,
with exact coverage and no overlaps. The native-only packer, generation-bound
reader and final-owner loader now have 19 CPU checks
(`docs/perf/mtp-pack-cpu-20260920.json`). Native-only acquisition completed: all 32 owner payloads and cross-host source
tensor hashes agree, with all eight hosts authenticated idle. Receipt:
`docs/perf/mtp-native-acquisition-20260920T013227Z.json`.
See `docs/perf/mtp-source-audit-20260919.json` and
`docs/perf/mtp-native-placement-20260920.json`. The native state orchestration and guarded greedy host session are implemented
and CPU-tested, including a complete synthetic device/session trajectory.
Trained loading and HLO/memory admission have now passed in the active run;
the first short accepted-throughput results are recorded below.

**Completed fresh-question baseline:**
`perf_real_long_question_20260920T005757Z`, immutable `248ef059`, finished with
all eight hosts authenticated idle. DB610 matched 29/29; all hosts agreed on the
fresh output. **14.4141 wall decode tok/s** over 6,143 timed decode steps and
6,144 generated tokens total, including host votes and rank0 token-file
write/flush. The 338-token prompt took 2.9555 s; warmed TTFT 3.1954 s. Cold
loading/compilation and network transport are excluded. It reached the 6,144-token
cap during reasoning, without a completed answer: correctness is not established.
Use this same prompt/budget for the MTP-assisted comparison. Receipt:
`docs/perf/tpu-real-long-question-20260920T005757Z.json`.

**Completed native comparison:**
`perf_real_native_mtp_20260920T015817Z`, immutable source `bcec7ddd`, exited
successfully; strict eight-host aggregation and authenticated cleanup passed.
All 16 graphs passed source/HLO/memory admission. The paired long question
(338 prompt tokens, 6,144 generated) measured **ordinary 14.2902, one-draft
R2 13.0927, two-draft R3 12.7682 wall tok/s**: speculation was 8.4%/10.7%
slower. Acceptance was 2,788/3,354 for R2; R3 accepted 2,071/2,662 first and
1,410/2,662 second drafts, averaging 2.3077 tokens/round. Both speculative
outputs diverge from ordinary at token index 6; neither is a token-exact
replacement. All three responses ended during reasoning at the output cap,
so finished-answer correctness remains unestablished. Target verification
consumed approximately 406/426 seconds and dominates cost. Native peak HBM
remained 28,228,678,144 bytes/chip in this acquisition.

Short DB610 matched 29/29 in all modes: ordinary 13.0772–13.0779,
R2 12.6385–12.6397, R3 14.3076–14.3085 wall tok/s. Its 9.4% R3 gain did
not carry over to the long prompt. Drafting, verification, rejected work,
commits, votes and rank0 JSONL write/flush are included; prefill/bootstrap,
cold loading/compilation and network transport are excluded from decode rate.
Receipt: `docs/perf/tpu-real-native-mtp-20260920T015817Z.json`.
Native acquisition/index remain recorded in
`docs/perf/mtp-native-acquisition-20260920T013227Z.json`.

**Representative comparison completed:**
`perf_real_native_suite_20260920T031727Z`, immutable source `dc047933`, exited
successfully. All eight hosts completed all 17 graph/memory admissions, DB610
29/29 in every mode, and both repeats of all three representative cases.
Authenticated cleanup and the strict completed-fleet summary passed.

| Request | Ordinary wall tok/s | One native draft | Two native drafts | Two-draft paired change |
|---|---:|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 | −6.6% to −5.9% |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 | +4.3% to +4.6% |
| Structured | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 | +10.7% to +10.9% |

Ranges cover two fresh requests and synchronized host reports, not independent
host trials or confidence intervals. Every mode reproduced its own token trail
on repetition. Speculative trails differ from ordinary at index 6 for prose,
5 for code, and 816/823 for structured R2/R3. No mode meets the 25% working
criterion or qualifies as a token-exact replacement. Keep ordinary as the
DB610-qualified research baseline; do not deploy this speculative experiment.

Prose completed but needs technical corrections under scoped assistant self-review;
all six code responses exhausted 7,168 tokens during reasoning without a final
answer; all six structured answers have exact correct values but Markdown fences
violate the requested standalone JSON format. These are bounded answer checks,
not independent review or a model-wide quality score. Target verification
remains the dominant speculative cost. Maximum recorded native peak HBM is
28,228,678,144 bytes/chip. No prefill speedup is claimed from decode measurements.

Completed receipt: `docs/perf/tpu-real-native-suite-20260920T031727Z.json`.
The controller and all workers have exited; all eight hosts are authenticated
idle. No additional TPU workload is queued. The objective is complete as a
measured mixed/negative result: the 25% gain target was not reached. Final release
checks, self-review, regional backup, publication and requirement audit passed.

The suite compares ordinary/R2/R3 on prose/code/structured cases, two repeats
per case. All use a matched 7,168-token cap within capacity 8,192, since the
first long comparison exhausted 6,144 tokens during reasoning. Prompt IDs and
oracles are unchanged; the earlier unlaunched input set is preserved. Receipt:
`docs/perf/mtp-representative-inputs-extended-20260920.json`. Suite SHA256:
`8604eec56772ff2739e8b1c80b90afc69a04d233035d902ad624a45519569c49`.
The finished first native run remains immutable at `bcec7ddd` and its completed
receipt/cleanup are preserved.

Private main now contains the reviewed evidence-only checkpoint `c142d284`
from `release/mtp-evidence-20260920`. Both completed receipts and the comparison
are published; implementation is preserved at research checkpoint `f097649a`.
CPU release checks passed (524 passed, one skipped). Pre-main and final regional
mirrors passed checksum/generation verification; all eight hosts were freshly
authenticated idle. No experimental runtime is deployed.
Publication receipt: `docs/perf/mtp-evidence-promotion-20260920.json`.

Decode D5 failed trained token parity and remains disabled. Earlier synthetic
72.1/64.3 ms timings contain the empty-owner bug and are not qualified baselines.
All historical trials and numerical boundaries remain in
`docs/perf/MTP_PROGRESS_20260919.md`; no unrelated queues should resume.

Requirement/evidence audit: `docs/perf/MTP_COMPLETION_AUDIT_20260920.md`.
It records the completed experiment, numerical/answer boundaries and verified
publication. Keep ordinary as the DB610-qualified research baseline.

## Work, in order

1. **Establish MTP feasibility.** Inspect retained checkpoint inventories for
   layer 78, its norms, embedding/hidden projection, transformer, head and scales.
   Read the authoritative GLM-5.2 MTP implementation and IndexShare semantics.
   Config alone is not proof of weights. Prefer existing verified assets; if
   runtime packing omitted MTP, recover only necessary GLM-5.2 tensors from the
   verified retained source under existing storage and identity rules.
2. **Build and prove multi-row target verification.** One target pass must
   amortize work across proposed tokens. Compare each row's predictions and
   accepted-prefix state against sequential greedy decode on CPU. Cover populated
   cache, causality, DSA/IndexShare boundaries, partial acceptance, first rejection,
   EOS, token limits, health failure and rollback of rejected cache/frontier
   writes. Keep draft state separate from committed target state. Repeated
   single-token target calls are not parallel verification.
3. **Measure verifier economics on v4.** Compare ordinary decode with small
   verification blocks, e.g. 2/3/5 target rows for 1/2/4 draft tokens. Acquire
   HLO, memory, device and wall timings. Label perfect-acceptance speed as an
   upper-bound estimate, never measured speculative throughput. If even perfect
   acceptance cannot win, fix the verifier bottleneck or record the negative
   result before extending the drafter.
4. **Implement native MTP drafting and greedy acceptance.** Verify every draft
   token with the target, commit only the accepted prefix and appropriate target
   correction/bonus token, and preserve all-host agreement and delivery/recovery
   guards. A deterministic n-gram drafter may test the verifier, but repetitive
   outputs alone cannot establish general speedups. Keep sampled speculation
   separate until acceptance/RNG semantics are proved; preserve the target
   distribution rather than silently approximating it.
5. **Run paired real-weight comparisons.** First reproduce DB610 parity, then
   use longer ordinary prose and code/reasoning continuations plus a structured
   case. Match prompts, capacity, output budgets and greedy policy. Check output
   agreement and explicitly report any multi-row numerical boundary. Include
   drafting, verification, rejected work, host votes and delivery in wall time;
   separate startup and prefill. Repeat paired measurements sufficiently to
   distinguish gains from noise and report prompt-dependent behavior.
6. **Keep a useful measured result.** Report accepted tokens per round, acceptance
   by draft position, draft/verify/wall cost, accepted output tok/s, latency,
   HBM and correctness. Working success criterion: at least 25% higher wall
   throughput on representative continuations; 30+ tok/s is an aspiration,
   not a prediction. Preserve a losing experiment and keep the faster ordinary
   path if necessary. Speculation targets decode; measure prefill/TTFT effects
   separately and do not claim a prefill speedup from decode measurements.

## Authority and rules

When activated, use all 32 v4 chips of existing pod `db-v4-64-od` (8 hosts x 4),
zone `us-central2-b`, SSH via `gcloud compute tpus tpu-vm ssh db-v4-64-od
--worker=all`. One workload at a time under `~/.glm-tpu-workload.lock` and the
existing pod lease. Authenticate idle first; leave all eight hosts clean.
Respect sync/cron locks. Never create/delete/resize TPU/VM/queued resources.
Disable automatic workload retries: the cancelled SSH supervisor retried its
terminated job and had to be stopped explicitly.

Work autonomously within this scope. Keep experiments outside frozen source
until CPU proofs, TPU measurements and source/HLO/memory admission support
promotion. Every pytest run uses `JAX_PLATFORMS=cpu`. No environment upgrades
or full-model safety copies just to try a draft path. Keep weights, credentials,
private prompts and raw DBs out of Git. Only `gs://driftbench-dsv4-uc`,
US-CENTRAL2; preserve storage bounds, originals, research and DB616–621 evidence.

Commit and push useful milestones to the private perf branch without asking;
no force-push, history rewriting or Co-Authored-By lines. At a useful reviewed
milestone, clean up and merge eligible work into private main after release
checks and verified regional backup, as in the previous curation. Preserve
unfinished/rejected experiments on research branches. State exactly whether
main gains implementation or only evidence; documentation does not deploy the
faster engine. The separate documentation/evidence checkpoint was reconciled
and fast-forwarded to private main at `5e9ce605` on 2026-09-20. Pre-/post-merge
regional mirrors passed content/checksum/generation verification; all eight
hosts were authenticated idle. It does not deploy the faster engine or complete
the active MTP goal. See `docs/perf/perf-checkpoint-promotion-20260920.json`.

Update this State and a dedicated MTP progress document as results land.
Finish with measured baseline versus speculative wall tok/s, correctness and
workload boundaries, authenticated cleanup, and the actual merge state.
