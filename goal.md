# Goal — GLM-5.2-FP8 TPU v4: long context, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read this, `docs/glm-tpu-revolution.md` in full
(incl. §21–§23), tails of `HANDOFF.md` and `docs/greenfield/GATE_D_LESSONS.md`; inspect state.

## Never idle on a running job

Never wait on process-name patterns. Re-read actual files and process identities. A free lease
does NOT prove an orphaned fleet stopped. `scripts/greenfield/watch_ws32_run.py` observes exact
PID/start-time/boot identities and libtpu holders, holding both user leases; unknown SSH retries.
Its resumable `watch.jsonl` proves only observation, never numerical success or clean census.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
review. **Hard ceiling: live <2,500,000,000,000 bytes.** No full-size backup, no moving
cost elsewhere. Soft delete off; report live and soft-deleted separately. Rsync writes only `repos/`.
Before any >100-GB artifact state need/size/replacement. Live 1,991,594,668,825 B at 09-07 06:51Z.

## Achieved

**CLOSED:** Gate D §21.6 (DB567, 20/20 exact tokens, adjudicated event 1); Gate G §22 (WS32_2D);
Step B (DB568/569 chunk identity), §21.2 per-rank enforcement; B′ (DB570 legacy-faithful main rotary);
Step C (DB571/572 capacity-independent numerics; decode 129.9/143.0/160.3 ms, peak
26.4/27.8/29.7 GB/chip at 8K/128K/256K capacity). L7 depth 1.0 DB573: passkey 891482 correct,
prefill 16,354 s, decode 142.68 ms. Legacy ids matched diagnostically only. Two review rounds resolved.

## Invariants

Native-JAX greenfield; legacy `tpu-inference` = oracle/utilities only. Never create/manage infra.
Serialize TPU work under both leases; runs end with an authenticated 8/8 zero-work census. Evidence
append-only, fail-closed, SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not proof.
Never weaken history. Optimizations default off. pytest ALWAYS `JAX_PLATFORMS=cpu` (controller = pod
worker 0). A seal needs a PUBLISHED pin and a clean enforcement surface, scoped to the surface
pathspecs: editing a surface file mid-run VOIDS its seal hours later; tests and `goal.md`/`HANDOFF`
are safe. §23.5 asserts nothing raw-token or cross-oracle exact. Optimized-HLO pins are
code-pin-specific (StableHLO is not): re-acquire whenever Python changes.

## Decisions

Decide autonomously toward a correct, provable finished project; record reasoning/alternatives in
`HANDOFF.md`. Reviewer gates persistence, install, execution and destructive apply. Never block owner.

## Efficiency/review

Smallest decisive test first; stop on first invariant failure; prefer offline adjudication to runs.
Adversarial reviewer = separate **gpt-6-astra** agent; resolve all P0–P2, cap hardening at 3 rounds
(2 used) — past that state the residual risk and move to the runs. Cron `sync-glm.sh` every 5 min;
verify origin + mirror before protected work. Never EU.

## Owner stop — 2026-09-07 08:10Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`.
L7 depth 0.0 tag `greenfield_ws32_short_decoder_128k_d0_0_numerical_cap131072_hrope_20260907T064941550123130Z`
at pin `679e2392` is LIVE, 13/62 chunks observed at 08:07Z. Its orchestrator/remote shells exited, leaving
8 original workers running. Do not restart. Monitor `889cb8f5` Astra-reviewed;
PID 2248787 holds both leases. Owner paused agent work; no new runs until owner resumes.
Leave workers/monitor intact; session stop is NOT fleet cleanup. After READY_FOR_CENSUS: prove cleanup,
collect original JSON/NPZ/HLO/log/trace with exact existing-object checks, then RECOVER=1 seal.
Never recover while workers live. Surface unchanged; `.ended` may never appear after shell loss.
Recipe: `docs/greenfield/WS32_ORPHAN_RECOVERY.md`; argv/env capsule in `configs/`.
Then sealer fixes (incl. E0 256-step guard; wrapper fixed), L7 0.05/0.95, L8/E0, §18.
