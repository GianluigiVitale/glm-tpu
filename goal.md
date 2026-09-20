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
`/home/gianl/glm-tpu-perf-ref`, from main `493b67de`. Native component milestone
`fb2a8d13` is committed/pushed. Frozen `MODEL_SOURCE` remains `edecdd94`.
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

**Native MTP components now implemented, CPU-tested only:** input projection;
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
(`docs/perf/mtp-pack-cpu-20260920.json`). Native-only acquisition is running;
a completed verified fleet pack is not yet available.
See `docs/perf/mtp-source-audit-20260919.json` and
`docs/perf/mtp-native-placement-20260920.json`. Next native work: actual pack/load,
prompt bootstrap and target-history refresh orchestration, then guarded host
acceptance/delivery and paired trained measurements.

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

**Active native acquisition:** `perf_native_mtp_acquire_20260920T013227Z`,
immutable source `50a8bbc5`, launched after idle/lease checks and was confirmed
live at 01:35:10 UTC. It is CPU-only and recovers only native layer-78 payloads
from audited source generations. Workload/pod/cron leases are held; no automatic
workload retries. Do not overlap another acquisition or TPU run. Next manual
poll at or after 01:45:10 UTC unless diagnosing a known failure or answering an
explicit status request. A prior preflight refused a busy backup lease before
remote work; it is preserved. Remaining native integration: bootstrap/history
refresh orchestration and guarded host acceptance/delivery, then paired real
measurements and representative correctness checks.

Decode D5 failed trained token parity and remains disabled. Earlier synthetic
72.1/64.3 ms timings contain the empty-owner bug and are not qualified baselines.
All historical trials and numerical boundaries remain in
`docs/perf/MTP_PROGRESS_20260919.md`; no unrelated queues should resume.

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
