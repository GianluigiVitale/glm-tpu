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

## Resume checkpoint — 2026-09-03 21:55Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; accepted
result code pin `75cbab626f7f2357b05dfab2c988330fabdca484`. Gate D remains open.

The original DB518 layer-0 producer uncertainty is closed. After V5 correctly failed publication
on stale inherited mirror authority, V6 bound and mutation-tested the full rewrite authority tuple.
Sol approved the batch with no P0–P2; origin and US-CENTRAL2 replay passed; install reported
`launcher_invoked=false`. Protected tag
`greenfield_original_db518_prompt_key_chunk0_20260903T213501747141067Z` is bitwise exact across all
2,048×128 lanes (both SHA `96d261cb…887c`), HLO contracts pass, remote set is 21/21, terminal is
generation `1788472134581783` / SHA `53fe14e4…ca38`, and pre/post censuses are 8/8 clean. Record:
`docs/artifacts/gate-d-original-db518-v6-exact-success.json`.

Classification: `ORIGINAL_DB518_PRODUCER_BOUNDARY_REPRODUCED_BITWISE;CONSUMER_UNPROVEN;
DECODER_UNPROVEN;GATE_D_OPEN`. The large chunk still lowers input RMS differently and misses 256
token-374 rows. Exact next: compile/complete a separate M2048 normalization helper, feed only its
finished BF16 device buffer into a separate M64 key-control helper, and require DB518 exactness. Only
then feed the same buffer into a consumer whose HLO proves no input-RMS recomputation. Stop on the
first failed boundary; do not run 8K yet. Review the full source/tests/certificate/install/run batch
with the same Sol thread before persistence or TPU execution.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
