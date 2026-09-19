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

Branch `perf/reference-lowhanging-fruit-20260919`, worktree
`/home/gianl/glm-tpu-perf-ref`, from main `493b67de`. Paused research snapshot:
`f493cbd56b5c7c72ce3d28455aec39b90140afbc`. Inspect live Git state first.
Frozen `MODEL_SOURCE` remains `edecdd94`; preserve all history and originals.

The passing trained-weight path is D1 grouped experts (empty-owner fix), D8
BF16-resident non-routed weights, D10 DSA and D4 packed host loop. Prefill uses
D8/P1/P2, canonical B128 with B114 tail. At 2,034 prompt tokens:

- **138.85 prompt tok/s; 14.04 wall decode tok/s**, including host checks and
  in-memory delivery, excluding network transport and cold load/compile.
- All 29 DB610 tokens match on all eight hosts. Legacy and packed request loops
  have bitwise-equal final state/residual; five warm steps and 23 timed steps.
- Model-only decode is 14.71–14.82 tok/s. Do not compare that directly with
  speculative wall throughput. Peak HBM: 28,228,678,144 bytes/chip at capacity
  8,192; draft weights/caches and verification temporaries require new admission.
- Receipt: `docs/perf/tpu-real-request-loop-20260919T192804Z.json`.
  Source audit: `docs/perf/tpu-real-request-loop-source-audit-20260919T192804Z.json`.
  Measured model/real-worker files match `5ff7b01e`; the manifest identifies a
  changed, unused microbenchmark tool. New runs must archive an immutable commit.

Config declares `num_nextn_predict_layers: 1` and
`index_share_for_mtp_iteration: true`. The authenticated source inventory and
three generation-bound shard headers contain all 1,569 layer-78 tensors
(10,032,632,960 payload bytes); the runtime pack omits them. The MTP body matches
full-index layer 74's schema, plus four BF16 MTP-specific tables; embedding and
output head are shared with the target. No MTP payload has been newly loaded or
executed. See `docs/perf/mtp-source-audit-20260919.json` and
`docs/perf/MTP_PROGRESS_20260919.md` for current proof/implementation status.
A CPU verifier prototype matches the three tested target predictions, with a
documented numerical boundary (residual relative L2 0.007373; no bitwise model
proof). Acceptance, physical rollback, causal independence and refusal checks
pass within their recorded scopes. See
`docs/perf/mtp-cpu-verifier-boundary-20260919.json`. No complete MTP path or
speculative throughput improvement has been validated. The next acquisition is
synthetic 2/3/5-row verifier economics; require trained-weight target agreement
before promotion and never label perfect-acceptance estimates as throughput.
Decode D5 failed real token parity and stays disabled.
Earlier synthetic 72.1/64.3 ms timings were affected by the empty-owner bug;
do not use them as correctness-qualified baselines. All eight hosts were
authenticated idle after cancellation; recheck before launching.
Receipt: `docs/perf/tpu-owner-pause-20260919.json`.

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
faster engine. A documentation-only checkpoint was prepared separately in
`/home/gianl/glm-tpu-perf-checkpoint`, but has not been merged to main; inspect
and reconcile it instead of treating it as completed.

Update this State and a dedicated MTP progress document as results land.
Finish with measured baseline versus speculative wall tok/s, correctness and
workload boundaries, authenticated cleanup, and the actual merge state.
