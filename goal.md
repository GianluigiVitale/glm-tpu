# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then finish §18

FULL ACCESS. Keep this <4000 chars; compaction-safe authority. At
start/compaction read it and `docs/glm-tpu-revolution.md` in full, then only the last ~300 lines of
`HANDOFF.md` and last entries of `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; never reread
whole histories. Inspect live git/run/lease/pod state. If stuck, reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md` in full
and adjudicate its hypotheses against local evidence.

## Priority

Close **Gate D**: a correct 78-layer decoder at 8K with exact raw tokens and DSA
set/tie order, state/cache integrity, local repeated collectives, measured HBM, fresh trace and
steady wall. PP8/PP16/WS32 adjudication, 128K/256K, speculation and §18 follow; do not divert now.

## Hard invariants

Native-JAX greenfield only; legacy `tpu-inference` is oracle/utilities only. Use only
`db-v4-64-od` and `gs://driftbench-dsv4-uc`; never create infra. Serialize TPU work under both
leases; every protected run ends with authenticated 8/8 zero-work census. Evidence is append-only
and fail-closed with exact SHA/code/plan/input binding. CPU/HLO/labels/throughput alone are not
proof. Never weaken historical evidence. Optimizations default off. Targets: useful <=200 ms/token,
strong <=125, stretch <=100.

## Efficiency/review contract

- Smallest decisive test first; instrument boundaries and stop on first invariant failure. Do not
  run an 8K decoder while a row/chunk probe can decide the hypothesis.
- One batched review covers source, tests, certificate, install and one fresh-tag command. Use Sol
  `gpt-5.6-sol`, thread `01a05206-57b1-7dc3-a49c-a913529b3937`, read-only, with exact verdict lines.
  Fable-max may replace it when available; otherwise Sol is sufficient.
- Cron `sync-glm.sh` runs every 5 min. Keep work committed/pushed and same-region mirrored; use
  `verify_gate_d_same_region_git_mirror.py` immediately before protected work. Never sync to EU.
- Report result, percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-03

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; pushed tip
`4572efae…9fce`. The v2 four-chip chunk0 run `…130504318767505Z` executed and ended 8/8 clean, but
its mandatory DB518 layer-0 control failed on 2048/2048 rows (74,299 lanes, median 35/row, max value
delta 0.03125). Diagnostic ledger `fd634ca0…92b54`, arrays `ae2026ec…49da`, runner
`3784cdc2…241f`, HLO `c0398830…76b4`; see
`docs/artifacts/gate-d-chunk0-v2-internal-wk-materialization-diagnosis.json`. The layer-1 arm has no
standing and Gate D is open.

High-confidence diagnosis: raw-FP8 wk materialization inside the large executable matched DB519's
rejected all-row signature; causality and the protected fix remain unproven. The uncommitted v3
correction separately completes raw→BF16 decode and BF16→FP32 promotion for wk0/wk1, then passes FP32 into
the main executable. Its publisher structurally proves exact ENTRY/root/live-output boundaries and
rejects raw/scale/BF16 rematerialization; hostile mutations are tested. Forced-CPU suite 34/34;
boundary suite 20/20. Adversarial Sol approved after two real P1 fixes, with no remaining P0–P2.
Required narrow Fable-max attempt hit its usage limit without review.

Exact next: finalize immutable source certificate/staging hashes, one final Sol command/certificate
check if bytes changed, commit/push, wait for same-region cron, verify origin+US-CENTRAL2 mirror,
install v3 without launch, verify 8/8 preflight, then execute one fresh never-used chunk0 tag. The
control must be exact before row0 may adjudicate geometry. Do not launch full 8K yet.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
