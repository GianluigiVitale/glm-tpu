# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-05 14:16 UTC

## Authority and isolation

- Branch/worktree: `rewrite/topology-first-decode` at
  `/home/gianl/glm-tpu-topology-rewrite`.
- Collective implementation/result pins: `fcd8426735119fee34ab8adc9e8c14b762adc2f8`
  through documentation pin `b12af9633c8b14648db8d2a2ccd9a3c577a04817`.
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`.
- Legacy oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`.
- Read `AGENTS.md`, `goal.md`, and `docs/glm-tpu-revolution.md` before this file.

Those tracked branch-local files are authoritative. `HANDOFF.md` is status only. The main checkout,
old worktrees, inherited campaign documents, and legacy `AGENTS.md`/`CLAUDE.md`/`HANDOFF.md` files
have no authority here. The old incremental TP32 sequence and pipeline-parallelism ban are
superseded. Never edit or delete the owner's untracked main-checkout files.

## Implemented and verified

- Frozen, hashed geometry/topology/plan types; runtime physical discovery; deterministic PP8/PP16
  physical rings; optimized-HLO contract with partition-id-to-physical-device mapping.
- Dependent 75-operation benchmark for control, all-reduce, reduce-scatter, all-gather,
  collective-permute, all-to-all, and fused tuple all-reduce over physical 2/4/8/32-chip groups.
- Rank-dependent nonlinear recurrence, exact HLO counts/groups/pairs/shapes, bitwise checksums,
  200 warmups, 1,000 samples, eight-host fleet agreement, append-only DB/archive, and strict cleanup.
- Greenfield tests last passed 49/49. No model path, checkpoint loader, transport executable, or
  greenfield model-throughput result exists yet.

## Protected evidence

Topology DB 405 / `greenfield_topology_20260805T125842425591441Z` proved 32 v4 chips in `2x4x4`,
eight processes, actual local ordering, topology hash `294e777...559`, PP8 hash `d5943ab8...c14`,
PP16 hash `6383e57c...f21`, remote `SUCCESS`, and 8/8 clean census.

Protected collective runs DB 406–414 all archived to the approved bucket and ended 8/8 clean:

- DB 406 `...T133905344573798Z`: dominant `bf16[2,6144]` control/all-reduce, all group sizes.
  Fleet-max p50 for 75 all-reduces: g2 `0.836`, g4 `1.083`, g8 `1.471`, g32 `3.941` ms.
- DB 407 `...T134254891049866Z`: dominant-payload all-gather, g2/g4/g8/g32 p50
  `0.798/1.012/1.475/4.503` ms.
- DB 408 and 409: protected exact-HLO validation of collective-permute and fused tuple reduction.
- DB 410 `...T135644529157649Z`: dominant-payload collective-permute, all-to-all, and fused tuple
  matrices. Collective-permute p50 is `0.651/0.653/0.653/0.791` ms for 75 operations.
- DB 411 `...T135850389312854Z`: supported six-operation `bf16[1,6144]` matrix.
- DB 412 `...T140125151247631Z`: supported six-operation `bf16[1,2048]` matrix.
- DB 413 `...T141114991474088Z`: supported six-operation `f32[1,6144]` matrix; all-reduce
  g2/g4/g8/g32 p50 `0.819/1.078/1.479/3.920` ms.
- DB 414 `...T141333601664455Z`: five-operation `int32[1,2048]` routing-metadata matrix;
  collective-permute p50 `0.617/0.641/0.640/0.768` ms.

Small decode reduce-scatter is deliberately **not** reported as measured. TPU-v4 optimized XLA
rewrote 75 requested reduce-scatters to 75 all-reduces even after disabling the decomposition flag
and making result segments non-equivalent. Diagnostics `...T134618415607642Z`,
`...T135045281384327Z`, and `...T135140118884391Z` failed closed before timing and preserve HLO.

## Interpretation and current boundary

The legacy 75 full-pod MoE combines cost `106.495 ms/token` (`1.420 ms` attributed per layer), but
an exact dependent chain of 75 full-pod all-reduces costs only `3.941 ms` p50; 75 nearest-neighbor
permutes cost `0.791 ms`. Therefore 12 KiB ICI payload bandwidth is not the legacy floor. Most of
the old time is arrival skew, layout/reshard, barrier waiting, and legacy decomposition around each
collective. Replacing only the collective primitive cannot recover 100 ms; the stage-local layout
and device-resident pipeline are the structural fix. These are synthetic mechanism results, not
token speed.

## Exact next sequence

1. Implement and protect the PP8 then PP16 device-resident stage-transfer chain: exact point-to-
   point HLO, no host/Ray/Python dispatch, no inactive-stage model-equivalent compute, warmed full
   distributions, fresh traces, DB/archive, and clean fleet.
2. Prove one exact real MoE layer whose combine remains inside its 4/2-chip stage.
3. Continue Gates B–H in the binding specification; do not load the full 753B checkpoint before
   the three architectural prerequisites pass.

The pod was rechecked at this update and all eight hosts reported `CENSUS_OK`.
