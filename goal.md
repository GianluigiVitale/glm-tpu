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

## Resume checkpoint — 2026-09-04 08:20Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; current
pushed pin `80bcd0edab9f4a1d7b0085af89dc4159cbf2254c`. Latest dependency-free
pgrep/libtpu-lock/container census is 8/8 zero-work; no TPU workflow is active.

The accepted completed-normalization→key boundary is exact: 0/2,048 rows and 0/262,144 lanes.
V9 then invoked one normalizer, exact key control and separate real layer consumer with no
intermediate host transfer. The consumer's layer-1 key differs from legacy in all 2,048 rows and
45,519 lanes; decisive row 0 differs 46/128. The first open boundary is therefore inside layer-0
consumer arithmetic, not input normalization.

Sealed legacy HLO `e7371f48…7216` proves prompt hidden reductions are exact `bf16[2048,6144]`
global-32 StrategyND. The model-free exact-M2048 fingerprint is implemented. V5 repaired both
remote loaders, refreshed workers 1–7, proved 8/8 exact repos and copied runtime source. It then
failed closed before capsule/JAX/TPU because recreated workers lack `/usr/local/libexec`; worker 0
alone verified its three sealed runtimes. Preserve
`gate-d-m2048-v5-install-missing-libexec-parent-failure.json`; census is 8/8 clean, transfers
cleaned 7/7, tag retired. Local V6 prepares only the fixed root-owned
`/usr/local -> libexec -> glm-tpu` chain in order, accepts only clean `80bcd0ed` worker prestates,
repins the full provenance chain and passes 34/34 focused tests. Certificate SHA
`3b821143…4a7f71`; fresh tag `greenfield_m2048_strategy_nd_20260904T081918746174545Z`.
Exact next: one Sol batch; only a clean verdict permits commit/push/US-CENTRAL2 replay and
install-only, followed by a separately authenticated one-shot execution. Do not run 8K first.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
