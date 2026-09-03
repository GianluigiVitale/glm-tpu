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

## Resume checkpoint — 2026-09-03 23:40Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; current
pushed V8 pin `73074bddcb9fe5312f3a369e877019378be6b13d`. Gate D is open.

Accepted V6 proves the original DB518 layer-0 producer bitwise exact for all 2,048×128 lanes (SHA
`96d261cb…887c`), with exact remote set and 8/8 cleanup; record
`docs/artifacts/gate-d-original-db518-v6-exact-success.json`. Classification remains producer exact,
consumer/decoder unproven.

V7 diagnosed exact empty-mesh placement and TPU-specific fused normalization. V8 pin `73074bdd…b13d`
then archived both boundary graphs and failed before normalization/key invocation: completed-buffer
arg0 has no sharding annotation while the four independent key inputs are replicated; optimized TPU
while also has an exact seventh passthrough scalar slot. Diagnostic terminal generation
`1788478038656464`, SHA `a212559e…a9d5`; pre/failure censuses 8/8. Records:
`docs/artifacts/gate-d-original-db518-v{7-tpu-hlo-placement,8-key-placement}-failure.json`.

Local V9 binds mixed placement per argument, the 7-slot TPU while, and the output-live inverse and
dot→while→rotary paths. Both archived V8 graphs pass offline; hostile placement/bypass tests reject;
focused suite 103/103. Sol returned no P0–P2 and approved persistence, install-only staging
`5fcb8bde…b1a2`, and one fresh tag `…20260903T233327590939916Z` after the merged-pin condition.
Exact next: commit/push/mirror, install-only, then one bounded V9 run. Do not run 8K.
If exact, build a separate real layer consumer that takes the same completed buffer and whose HLO
proves no input-RMS recomputation; only then authorize the exact 8K decoder.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
