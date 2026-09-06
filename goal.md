# Goal — GLM-5.2-FP8 TPU v4: long context, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read this, `docs/glm-tpu-revolution.md` in full
(incl. §21–§23), tails of `HANDOFF.md` and `docs/greenfield/GATE_D_LESSONS.md`; inspect state.

## Never idle on a running job

**Never wait with `pgrep -f`/`pkill -f`: the pattern matches the WAITER'S OWN command line, so it
waits on itself forever.** 3 stalls so far, the last 10 idle hours on a free pod after a successful
acquisition. Wait on the lease, a file lock with no process name:
`/home/gianl/bin/wait-for-pod-run.sh [run_dir]`, backgrounded. Any watcher matches a FILE, never a
process pattern. Never report a claim about a file without re-reading the file.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
dependency review. **Hard ceiling: live <2,500,000,000,000 bytes.** No full-size backup, no moving
cost elsewhere. Soft delete off; report live and soft-deleted separately. Rsync writes only `repos/`.
Before any >100-GB artifact state need/size/replacement. Live: 1,976,189,248,736 B.

## Achieved

**Gate D CLOSED (§21.6):** DB 567, pin `4286509`, SUCCESS `e40760f8…f662`: tokens 20/20 exact, event
1 == record `4da05468…`, alarm acked, p50 130.369 ms. Gate E met; F at 2K only. M2048 DEFERRED.
**Gate G CLOSED (§22):** PP8 2K 245.6 ms vs WS32 2K 122.6 ms; WS32_2D promoted. **§23 Step B:**
chunked prefill reproduces every DB567 witness at C=2048 (DB 568) and C=512 (DB 569). **§21.2
enforcement (`48372a34`):** the sealer RE-DERIVES items 3-4 from each run's own arrays, every rank;
pre-registration proven from the run's pin; the reference-row registry is enforced at the loader.
§21.2 states the ceiling: these prove a change is in PUBLISHED history, not that it was reviewed.

## Invariants

Native-JAX greenfield; legacy `tpu-inference` = oracle/utilities only. Never create/manage infra.
Serialize TPU work under both leases; runs end with an authenticated 8/8 zero-work census. Evidence
append-only, fail-closed, SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not proof.
Never weaken history. Optimizations default off. pytest ALWAYS `JAX_PLATFORMS=cpu` (controller = pod
worker 0). Never edit the run worktree during a run or seal. An adjudicated seal needs a PUBLISHED
pin and a clean enforcement surface.

## Decisions

Never block the owner. When a question arises, take the decision you would recommend — the one best
serving a correct, provable, finished project — record it with its reasoning and the alternative in
`HANDOFF.md`, and proceed: scope, sequencing, cost/benefit, what to build or drop, pod time. The
reviewer, not the owner, gates persistence, install, execution, destructive apply. Never ask.

## Efficiency/review

Smallest decisive test first; stop on first invariant failure; prefer offline adjudication to runs.
Adversarial reviewer = separate **Opus 5** agent; resolve all P0-P2, but cap enforcement-hardening
at 3 rounds — past that state the residual risk in the spec and move to the runs. Cron `sync-glm.sh`
every 5 min; verify origin + mirror before protected work. Never EU.

## Resume — 2026-09-06 15:20Z

Run worktree at `48372a34`, pushed+mirrored. A′ tables-ON 8K acquisition DONE
(`…acquire_hrope_…042442Z`, 7 graphs, table 8192x64, census clean). B′ run 1
(`…numerical_hrope_…151844Z`, no record bound) RUNNING. Next: (1) offline §21.2 adjudication of run
1's arrays, commit+push the record; (2) B′ run 2 with `GLM_GREENFIELD_WS32_DSA_ADJUDICATION=1`,
sealed; (3) Step C acquisitions+capacity runs at 131,072 and 262,656; (4) L7 four 128K runs (1.0,
0.0, 0.05, 0.95); (5) L8 256K E0 at 262,656; (6) §18 proof incl. base vs effective throughput
(Gate H). Prefill dominates: ~122 ms/prompt token at 8K, so 128K is ~4.5 h and 256K ~9 h per run.
Step C measures the real per-capacity cost.
