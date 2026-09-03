# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then finish §18

FULL ACCESS: autonomous. Keep this file <4000 chars; it is the compaction-safe authority. At
start/compaction read it and `docs/glm-tpu-revolution.md` in full, then only the last ~300 lines of
`HANDOFF.md` and last entries of `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; never reread
whole histories. Inspect live git/run/lease/pod state. If stuck, reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md` in full
and adjudicate its hypotheses against local evidence.

## Priority

Close **Gate D**: a correct 78-layer greenfield decoder at 8K with exact raw tokens and DSA
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
  A fresh Fable-max audit may replace it when usage is available; when exhausted, Sol is sufficient.
- Cron `sync-glm.sh` runs every 5 min. Keep work committed/pushed and same-region mirrored; use
  `verify_gate_d_same_region_git_mirror.py` immediately before protected work. Never sync to EU.
- Report result, realistic percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-03

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`, last pushed
tip `a43f6a2…`; pod READY/HEALTHY and no known workload. Three accepted schedule runs exist: only the
last two keep event 0 exact and leave 7 identical event-1 swaps. Offline substitution proves the
greenfield layer-1 **prompt cache is sufficient** for those swaps, not that decode-side arithmetic
is exact (`DECODE_SIDE_EXACTNESS_NOT_PROVEN`). Legacy layer-1 prompt cache differs on 8155/8155 rows.

Active test: one 4-chip chunk-0 legacy-prefill probe; row 0 is decisive, rows 1+ diagnostic only.
Sol rounds 29–31 cleared science/input/sealed-runtime/census controls. Round 32
blocked mutable execution/vacancy/run-dir provenance. The capsule fixes these. First re-review found
two residual P1s; the delta now creates/retains fd 7 in the launcher and independently hash-binds
three reference slices plus exact layer-0 control. Forced-CPU suite: 29 passed. Nothing installed;
no TPU run; tags `…022131838688366Z` and `…113530293108901Z` are burned. Gate D open.

Exact next: one Sol delta review of source/certificate/commands; correct findings;
commit/push/verify origin+US-CENTRAL2 mirror; run reviewed installer only; compose/authorize one new
tag; preflight 8/8 zero work; execute once. If row 0 exact, integrate the proven prefill geometry and
run the next smallest exactness check before 8K. If nonexact, use archived arrays/HLO to localize it.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
