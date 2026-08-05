# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-05 14:40 UTC

## Authority and isolation

- Branch/worktree: `rewrite/topology-first-decode` at
  `/home/gianl/glm-tpu-topology-rewrite`.
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`.
- Legacy oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`.
- Collective result pins: `fcd8426735...` and `b12af9633...`.
- Transport latency pin: `577aa4bf706976f3d25f552526d0b6eaa5d4920e`; trace pin:
  `8aee3351a61f89141762dda237f582b8deed3a1c`.
- Read `AGENTS.md`, `goal.md`, and `docs/glm-tpu-revolution.md` before this status file.

Those tracked branch-local files are authoritative. Main checkout, old worktree, campaign, and
legacy `AGENTS.md`/`CLAUDE.md`/`HANDOFF.md` files have no authority here. The incremental TP32 plan
and pipeline-parallelism ban are superseded. Never edit/delete the owner's untracked main files.

## Implemented and verified

- Frozen, hashed geometry/topology/plan types; runtime physical discovery; deterministic PP8/PP16
  groups/rings; exact optimized-HLO parser with executable-partition-to-physical-device mapping.
- Complete protected 75-operation collective floor over physical 2/4/8/32-chip groups.
- Device-resident PP8/PP16 transport: closed physical lanes, exact point-to-point HLO, one compiled
  global program, deterministic checksums, warmed distributions, and fresh fleet XPlanes.
- Greenfield tests last passed 54/54. No model layer, checkpoint loader, decoder, serving path, or
  greenfield model-throughput result exists yet.

## Protected evidence

Topology DB 405 / `greenfield_topology_20260805T125842425591441Z` proved 32 v4 chips in `2x4x4`,
actual local ordering, topology hash `294e777...559`, PP8 `d5943ab8...c14`, PP16
`6383e57c...f21`, remote `SUCCESS`, and 8/8 clean census.

Collective DB 406–414 all have approved archives and clean census. Dominant `bf16[2,6144]` p50 for
75 all-reduces is g2/g4/g8/g32 `0.836/1.083/1.471/3.941 ms`; 75 full-ring permutes are `0.791 ms`.
The matrix covers required bf16 shapes, `f32[1,6144]`, and `int32[1,2048]` metadata. Small decode
reduce-scatter is not falsely timed: TPU XLA rewrites it to all-reduce, and three diagnostics fail
closed with saved HLO.

Transport DB 415 / `greenfield_transport_20260805T142953361259007Z` passed 16 cases at `577aa4b`:

- One invocation carries every rank-dependent payload through every stage and back to its exact
  lane origin. HLO contains exactly 8 PP8 or 16 PP16 neighbor `collective-permute` operations with
  exact bf16/int32 shapes and pairs. It contains no all-reduce/gather/to-all/reduce-scatter, host
  staging, Ray/Python stage dispatch, or model-equivalent compute.
- `bf16[1,6144]` fleet-max profiler-free p50: PP8 `0.331145 ms` vs `0.300895` control
  (`0.030250 ms` net); PP16 `0.407385` vs `0.331230` (`0.076155 ms` net). Other payloads are
  approximately `0.024–0.077 ms` net. This is synthetic mechanism latency, not token speed.
- Approved archive `SUCCESS`, DB linkage, full distributions, provenance, and 8/8 cleanup pass.

Trace DB 416 / `greenfield_transport_trace_20260805T143832942547470Z` passed at `8aee335`:

- Both plans have 8 fresh XPlanes, 64 TPU cores, and exactly 20 selected steps/core.
- Every PP8 core reports exactly 8 permute starts + 8 dones/step; PP16 reports 16 + 16;
  forbidden collective count is zero. Trace-contaminated timing is excluded from latency claims.
- Archive, hashes, DB snapshot, authenticated pre/post zero census, and remote `SUCCESS` pass.

Rejected diagnostic `...T143608529930365Z` has no DB/status claim: valid distinct rank traces were
downloaded into worker-0's original local trace tree, causing a duplicate-host parser refusal. The
fixed proof uses an isolated canonical fleet directory; it did not reinterpret the failed result.

## Interpretation

Legacy attributes `106.495 ms/token` to 75 full-pod MoE combines (`1.420 ms/layer`), while exact
75-op full-pod all-reduce needs `3.941 ms` and an entire PP8 residual ring costs only `0.331 ms`.
The old loss is therefore arrival skew, reshard/layout work, barrier waiting, and surrounding
decomposition—not raw 12 KiB ICI or stage-transfer bandwidth. Primitive replacement alone is not
enough; stage-local model layout remains the structural requirement.

## Exact next sequence

1. Prove one exact real GLM MoE layer whose routed/shared combine remains inside its four-chip PP8
   stage: captured legacy-oracle inputs/outputs, exact router/expert ownership, adversarial expert
   concentration, numerical contract, physical HLO, TPU result, DB/archive, and clean fleet.
2. Add the mandatory PP16 two-chip form against the same oracle after PP8 is exact.
3. Continue Gates B–H exactly as specified. Do not pack/load the full 753B checkpoint until the
   topology, transport, and exact topology-local MoE prerequisites all pass.

The pod ended the latest proof with all eight hosts `CENSUS_OK`.
