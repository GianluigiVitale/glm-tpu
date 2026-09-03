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

## Resume checkpoint — 2026-09-03 18:32Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; pushed
tip `82e33fc8839ce076780161da1af1ab040be00633`. Gate D remains open.

V6 passed staged Sol reviews, push/full US-CENTRAL2 replay and immutable install
(`launcher_invoked=false`).
Protected tag `greenfield_layer1_prompt_chunk0_geometry_20260903T180438114586827Z` compiled once
but its HLO admission rejected **before `compiled(...)` invocation**. It published a diagnostic and
ended 8/8 clean. Terminal `1788459348320561`/`00eeae78…ec77`; HLO `b7713bdf…b0ea`, StableHLO
`d1d97229…bda9`. Artifact:
`docs/artifacts/gate-d-chunk0-v6-hlo-admission-failure.json`. Tag is burned; no numerical or
performance claim.

Cause: XLA duplicated exact layer-0 normalization into selected output 1 of a tiled tuple fusion.
V6 blocked only the standalone node and traversed every tuple output, falsely finding a bypass.

Current uncommitted V7 makes ancestry output-sensitive for fusion `get-tuple-element` and while
state slots (fixed point including condition), recognizes only exact logical-M2048 32x64 tiled
normalization with ordered four-slice `ConcatBitcast`, collects every exact normalization witness,
and requires blocking all witnesses to cut every gather→root5 path. It rejects wrong tuple index/
output, arithmetic/rounding/extent/caller drift, reordered slices, malformed while, decoys and a
parallel raw bypass. Exact archived V6 HLO passes offline; targeted mutations reject. Forced-CPU
suite 37/37. Sol found and closed unordered-slice and missing-condition defects, then returned
`APPROVE V7 VALIDATOR CORE`, no P0–P2.

Exact next: finish V7 hash chain and append-only source certificate; create immutable install-v7
staging for launcher-v7/capsule-v6 and a fresh tag. Sol-review full source, persistence and
install-only; commit/push/mirror; install without launch; separately review preflight/execution;
run one chunk0. Do not run 8K until DB518 layer-0 control is bitwise exact.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
