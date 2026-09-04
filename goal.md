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

## Resume checkpoint — 2026-09-03 23:58Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; current
pushed V9 pin `33c85baf508f5aa6d7e67050326205cebe99a84a`. Gate D is open.

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

V9 protected probe completed in 12 s with 0/2,048 rows and 0/262,144 lanes mismatched, exact bits
SHA `96d261cb…887c`, one normalizer/key invocation and no intermediate host transfer. Publication
failed closed only because JSON changed contract tuples to lists; diagnostic generation
`1788479258358293`, SHA `82c0eb80…e7661d`; pre/post 8/8 clean. Result is diagnostic, not accepted.

Local V10 changes publication only: strict JSON comparison plus exact mirror-verifier repin; actual
V9 artifacts replay, hostile mutations reject, focused suite 105/105. Sol returned no P0–P2 and
approved staging `efbfebca…51af8` plus fresh tag `…20260903T235409426942147Z`. Exact next:
commit/push/mirror/install and one run. Do not run 8K.
If exact, build a separate real layer consumer that takes the same completed buffer and whose HLO
proves no input-RMS recomputation; only then authorize the exact 8K decoder.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
