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

## Resume checkpoint — 2026-09-04 07:45Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; current
pushed pin `b835e0f5b101e17c686b290cd5843b60d2765dca`. Pod READY and the latest dependency-free
pgrep/libtpu-lock/container census is 8/8 zero-work; strict Ray/Python census is unavailable until install.

The accepted completed-normalization→key boundary is exact: 0/2,048 rows and 0/262,144 lanes.
V9 then invoked one normalizer, exact key control and separate real layer consumer with no
intermediate host transfer. The consumer's layer-1 key differs from legacy in all 2,048 rows and
45,519 lanes; decisive row 0 differs 46/128. The first open boundary is therefore inside layer-0
consumer arithmetic, not input normalization.

Sealed legacy HLO `e7371f48…7216` proves prompt hidden reductions are exact `bf16[2048,6144]`
global-32 StrategyND. The model-free exact-M2048 fingerprint is implemented. V4 successfully
refreshed workers 1–7, proved all eight repos exact and copied runtime source, then failed before
bootstrap/JAX/TPU because its second embedded Python loader had the same quote-loss bug fixed in
the first. Preserve `gate-d-m2048-v4-install-runtime-loader-quoting-failure.json`; its transfer
trees are exactly cleaned 7/7 and tag retired. Local V5 corrects and compiles every embedded loader,
accepts only the new clean `b835e0f5` worker prestate, and passes 31/31 plus all hash/syntax/history
checks. V5 cert SHA `06932679…4399c7`; fresh tag
`greenfield_m2048_strategy_nd_20260904T072057405273776Z`. Exact next: one Sol batch, then
commit/push/US-CENTRAL2 replay, install-only, live preflight and one execution if approved. Do not
run 8K until this boundary is exact.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
