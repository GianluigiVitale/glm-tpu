# Goal — GLM-5.2-FP8 TPU v4: long context, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read this, `docs/glm-tpu-revolution.md` in full
(incl. §21–§23), tails of `HANDOFF.md` and `docs/greenfield/GATE_D_LESSONS.md`; inspect state.

## Never idle on a running job

**Never wait with `pgrep -f`/`pkill -f`: the pattern matches the WAITER'S OWN command line, so it
waits on itself forever.** 3 stalls, the last 10 idle hours. Wait on the lease, a file
lock with no process name: `/home/gianl/bin/wait-for-pod-run.sh [run_dir]`, backgrounded, or run the
wrapper backgrounded so its exit notifies. Any watcher matches a FILE, never a process pattern.
Never claim anything about a file without re-reading it.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
review. **Hard ceiling: live <2,500,000,000,000 bytes.** No full-size backup, no moving
cost elsewhere. Soft delete off; report live and soft-deleted separately. Rsync writes only `repos/`.
Before any >100-GB artifact state need/size/replacement. Live 1,991,594,668,825 B at 09-07 06:51Z.

## Achieved

**CLOSED:** Gate D §21.6 (DB 567, tokens 20/20 exact, event 1 == `4da05468…`); Gate G §22 (WS32_2D
promoted); §23 Step B (DB 568 C=2048, 569 C=512); §21.2 enforcement `48372a34` (sealer re-derives
items 3–4 from each rank's own arrays); B′ §23.8/9 (DB 570, host BF16 main rotary table
legacy-faithful, tokens == DB 567); Step C (DB 571/572 — capacity costs time, not
numerics; decode p50 129.9/143.0/160.3 ms, peak 26.4/27.8/29.7 GB/chip at cap
8,192/131,072/262,656); §23.5 L7 depth 1.0 (DB 573, pin `df1695af`, passkey 891482 == gold, legacy
ids matched as a diagnostic only; prefill 16,354 s = 128.4 ms/prompt token vs 128.1 predicted;
decode p50 142.68 ms). 2 adversarial rounds resolved.

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

Never block the owner. When a question arises, take the decision you would recommend — the one best
serving a correct, provable, finished project — record it with its reasoning and the alternative in
`HANDOFF.md`, and proceed: scope, sequencing, cost/benefit, what to build or drop, pod time. The
reviewer, not the owner, gates persistence, install, execution, destructive apply. Never ask.

## Efficiency/review

Smallest decisive test first; stop on first invariant failure; prefer offline adjudication to runs.
Adversarial reviewer = separate **gpt-6-astra** agent; resolve all P0–P2, cap hardening at 3 rounds
(2 used) — past that state the residual risk and move to the runs. Cron `sync-glm.sh` every 5 min;
verify origin + mirror before protected work. Never EU.

## Resume — 2026-09-07 07:20Z

Worktree `95a96c6a`, pushed. L7 depth 0.0 RUNNING at pin `679e2392` since 06:52Z (~5 h; reuses the
`df1695af` HLO pins, only the wrapper changed). Next: (1) L7 depths 0.05, 0.95; (2) L8 256K E0
acquisition at 262,656 then its run (127 chunks + 2048 tail, ~11.5 h); (3) §18 proof, base vs
effective throughput (Gate H). In the next gap between runs apply the 3 deferred review items at
the tail of `HANDOFF.md`, which also records the worker-0 ssh flake the sync step now retries.
