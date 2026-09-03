# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then §18

FULL ACCESS. Keep this <4000 chars; compaction-safe authority. At start/compaction read it and
`docs/glm-tpu-revolution.md` in full, then tails of `HANDOFF.md` and
`docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; inspect live state. If stuck, reread the full
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Priority

Close **Gate D**: correct 78-layer decoder at 8K with exact tokens/DSA order, state/cache integrity,
local collectives, HBM, trace and wall. Plan adjudication, long contexts and §18 follow.

## Hard invariants

Native-JAX greenfield only; legacy `tpu-inference` is oracle/utilities only. Use only
`db-v4-64-od` and `gs://driftbench-dsv4-uc`; never create infra. Serialize TPU work under both
leases; every protected run ends with authenticated 8/8 zero-work census. Evidence is append-only,
fail-closed and SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not proof. Never
weaken history. Optimizations default off. Targets: useful <=200, strong <=125, stretch <=100 ms/token.

## Efficiency/review contract

- Smallest decisive test first; instrument boundaries and stop on first invariant failure. Never
  run 8K while a row/chunk probe can decide the hypothesis.
- Use the independent Sol adversarial reviewer before persistence, install and execution; resolve
  every P0–P2. Fable-max may replace it when available.
- Cron `/home/gianl/bin/sync-glm.sh` runs every 5 min. Commit/push/same-region mirror; verify origin
  plus US-CENTRAL2 mirror before protected work. Never use EU.
- Report result, percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-03 23:03Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; current
pushed V7 pin `d393a3bf948f732c2e62526ee9dfcd1e61b2d7a3`. Gate D is open.

Accepted V6 proves the original DB518 layer-0 producer bitwise exact for all 2,048×128 lanes (SHA
`96d261cb…887c`), with exact remote set and 8/8 cleanup; record
`docs/artifacts/gate-d-original-db518-v6-exact-success.json`. Classification remains producer exact,
consumer/decoder unproven.

V7 tag `…20260903T222145752160015Z` compiled once but stopped before numerical invocation. CPU-sealed
StableHLO expected `3cd10543…a976`; TPU emitted `87255de0…683d`. An exact diff proves the only
portable-IR changes are one empty `sdy.mesh` plus replicated empty-mesh annotations on the three
ENTRY arguments; TPU optimized HLO also uses backend-specific fused square/reduce/rsqrt. Diagnostic
ledger generation `1788475897109185`, SHA `f1d3e3b0…5996`; pre/failure censuses 8/8. Record:
`docs/artifacts/gate-d-original-db518-v7-tpu-hlo-placement-failure.json`. No numerical/performance claim.

Sol blocked the first V8 batch because exact optimized normalization and while/dot/rotary witnesses
could remain dead while an alternate graph fed the root. The local correction now requires the
inverse chain, selected while slot, exact dot and role-separated live cosine/sine path to reach the
returned BF16 root. Both demonstrated bypasses reject; archived TPU forms replay; focused suite
102/102. Sol delta review: no P0–P2; persistence, install-only tree `419cc7a2…3d6` and execute-once
tag `…20260903T232020879638235Z` approved conditional on the merged pin. Exact next: commit/push,
same-region mirror, install-only, then the one bounded V8 run. Do not run 8K.
If exact, build a separate real layer consumer that takes the same completed buffer and whose HLO
proves no input-RMS recomputation; only then authorize the exact 8K decoder.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
