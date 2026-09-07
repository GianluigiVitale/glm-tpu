# Goal — GLM-5.2-FP8 TPU v4: long context, §18

FULL ACCESS. Continue autonomously until full project completion is proved under §18.
Old pause notes are historical only. Diagnose, fix, resume; pause only on a new owner stop
or a genuine authority blocker.
Keep <4000 chars. At start/compaction read this, `docs/glm-tpu-revolution.md` in full
(incl. §21–§23), tails of `HANDOFF.md` and `docs/greenfield/GATE_D_LESSONS.md`; inspect state.

## Supervise running jobs

Never wait on process-name patterns. Re-read actual files and process identities. A free lease
does NOT prove an orphaned fleet stopped. `scripts/greenfield/watch_ws32_run.py` observes exact
PID/start-time/boot identities and libtpu holders, holding both user leases; unknown SSH retries.
Its resumable `watch.jsonl` proves only observation, never numerical success or clean census.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
review. **Hard ceiling: live <2,500,000,000,000 bytes.** No full-size backup, no moving
cost elsewhere. Soft delete off; report live and soft-deleted separately. Rsync writes only `repos/`.
Before any >100-GB artifact state need/size/replacement. Latest storage inventory: `HANDOFF.md`.

## Achieved

**CLOSED:** Gate D §21.6 (DB567, 20/20 exact tokens, adjudicated event 1); Gate G §22 (WS32_2D);
Step B (DB568/569 chunk identity), §21.2 per-rank enforcement; B′ (DB570 legacy-faithful main rotary);
Step C (DB571/572 capacity-independent numerics; decode 129.9/143.0/160.3 ms, peak
26.4/27.8/29.7 GB/chip at 8K/128K/256K capacity). L7 depth 1.0 DB573: passkey 891482 correct,
prefill 16,354 s, decode 142.68 ms. L7 depth 0.0 DB574: passkey 705269 correct, 143.50 ms.
Two of four depths sealed; legacy ids matched diagnostically only.

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
`HANDOFF.md`. Reviewer gates persistence, install, execution and destructive apply.

## Efficiency/review

Smallest decisive test first; stop on first invariant failure; prefer offline adjudication to runs.
Adversarial reviewer = separate **gpt-6-astra** agent; resolve all P0–P2; avoid redundant review.
Cron `sync-glm.sh` every 5 min; verify origin + mirror before protected work. Never EU.

## Snapshot — 2026-09-07 12:27Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`.
L7 depth 0.0 source `679e2392`, recovery `ae994e5a`, DB574, SUCCESS `f123beba…0667`;
original workers and replacement monitor ended. Original evidence recovered without rerun;
authenticated 8/8 process/device cleanup, remote generation replay and DB verified.
Sealer fixes `dd0f1c07` now integrating: dedicated pinned sealing checkout, complete E0 window,
extractor dependency coverage. Focused tests/review/publish/mirror before next protected launch.
Next: acquire current HLO pins, L7 depths 0.05/0.95, L8 acquisition + 256-step E0, §18 audit.
Recovery history: `docs/greenfield/WS32_ORPHAN_RECOVERY.md`; exact run details in HANDOFF.
