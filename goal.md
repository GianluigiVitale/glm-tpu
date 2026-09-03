# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then §18

FULL ACCESS. Keep this <4000 chars; compaction-safe authority. At start/compaction read it and
`docs/glm-tpu-revolution.md` in full, then tails of `HANDOFF.md` and
`docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`. Inspect live git/run/lease/pod state. If stuck, reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md` in full
and adjudicate hypotheses against local evidence.

## Priority

Close **Gate D**: correct 78-layer decoder at 8K with exact raw tokens and DSA set/tie order,
state/cache integrity, local repeated collectives, measured HBM, fresh trace and steady wall.
PP8/PP16/WS32 adjudication, 128K/256K, speculation and §18 follow; do not divert now.

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
  every P0–P2. Fable-max may replace it when available; otherwise Sol is sufficient.
- Cron `/home/gianl/bin/sync-glm.sh` runs every 5 min. Commit/push/same-region mirror; verify origin
  plus US-CENTRAL2 mirror before protected work. Never use EU.
- Report result, percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-03 18:05Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; last pushed
tip `9265301c…c442`. Gate D remains open.

Protected v5 tag `greenfield_layer1_prompt_chunk0_geometry_20260903T165338499173468Z` executed on
one four-chip host, published a diagnostic and ended 8/8 clean. Ledger generation
`1788455638916604`, SHA `ed164a0d…19678`; arrays `1d946006…ff587`; HLO `16b137…f49`. DB518 layer-0
control improved from 2,048 rows/74,299 lanes to 256 rows/1,042 lanes. Row 0 is exact, but all
failures are positions congruent to {4,9,14} mod 24 carrying token 374, so layer-1 has no
standing. Diagnosis: `docs/artifacts/gate-d-chunk0-v5-token374-control-diagnosis.json`.

Strongest bounded cause: v5 host-gathered the M2048 embedding and passed a direct BF16 chunk;
accepted DB518 device-gathered 37 unique embeddings into the input RMS lowering. Current v6 moves
the exact in-bounds row gather onto device (`jnp.take(..., mode="clip")`) before the unchanged
pipeline. A whole-module validator runs after compile but **before invocation**; publisher repeats
it. It proves exact typed gather/RMS/BF16 paths and dominance/no-bypass into root 5 (layer-0 key
control). Root 6 legitimately has a residual path and is judged only after root 5 is exact.
Sixteen hostile mutations; CPU 35/35; Sol approved with no P0–P2. No protected-fix claim yet.

V6 certificate: `docs/artifacts/gate-d-layer1-prompt-chunk0-geometry-v6-source.json`; immutable
staging tree `d4db809e…ced54`; proposed fresh tag `…20260903T180438114586827Z`. Exact next: review
certificate/staging/install-only; commit/push; replay same-region mirror; provision install-v6 and
install launcher-v6/capsule-v5 without launch. Then separately review fresh vacancy/locks/pod/
census and execute one chunk0. Do not run 8K until layer0 control is bitwise exact.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
