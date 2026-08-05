# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-05 13:21 UTC

## Authority and location

- Branch: `rewrite/topology-first-decode`
- Worktree: `/home/gianl/glm-tpu-topology-rewrite`
- Starting harness commit: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`
- Legacy oracle starting/current pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`
- Branch operating file: `AGENTS.md`
- Compact contract: `goal.md`
- Full contract: `docs/glm-tpu-revolution.md`

Those two contract files supersede the inherited incremental TP32 instructions and old ban on
pipeline parallelism. Legacy execution code and evidence remain intact on `main` and in Git history.
Files in the main checkout or any other worktree have no authority over this branch.

## Current state

The isolated engine foundation now exists. Immutable model/physical-topology/stage-plan contracts
serialize canonically and carry SHA-256 geometry/topology/plan hashes. Runtime discovery refuses
guessed topology or local ordering. The optimized-HLO contract parser now proves physical replica
groups, collective counts/pairs/shapes, global-id semantics, repeated-region policy, and explicit
diagnostic-only handling of full-pod repeated collectives. Focused HLO coverage passes 12/12.
The complete greenfield L0/HLO suite passes 40/40.

Protected Gate-A topology capture passed at code `75c8bb14cbd290930ff024937310f3aec6175090`:

- run `/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z`;
- archive `gs://driftbench-dsv4-uc/results/greenfield_topology_20260805T125842425591441Z`;
- DB 405, exact topology contract `07ccfc470a079e66177497036c5fac2a928067c96d392d0e117d45f2e5986f99`;
- observed `2x4x4`, 32 JAX-visible v4 chips, 8 processes, 4 local chips/process;
- topology hash `294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559`;
- PP8 ring process order `[0,2,4,6,7,5,3,1]`, group hash `d5943ab8...23c14`;
- PP16 all-lane-adjacent ring, group hash `6383e57c...40f21`;
- TPU-VM suffixes are not JAX ranks: launch-to-JAX is
  `{0:1,1:6,2:0,3:7,4:2,5:4,6:3,7:5}`;
- eight fleet records agreed on-device; pre/post census was 8/8 `CENSUS_OK`; remote `SUCCESS` exists.

This proves physical inventory and local groups only. It is not PP8/PP16 transport, HLO, model, or
throughput evidence. No greenfield performance claim exists. Legacy execution remains oracle-only.

## Exact next sequence

1. Implement the genuinely dependent 75-operation collective benchmark with anti-elision checksum,
   exact HLO count, group sizes 2/4/8/32, required shapes/dtypes, and warmed distributions.
2. Prove device-resident PP8 and PP16 stage-transfer skeletons with no host/Ray/Python stage dispatch,
   no inactive-stage model-equivalent compute, exact point-to-point HLO, and fresh traces.
3. Prove one exact real MoE layer whose combine is confined to its 2/4-chip stage.
4. Continue through Gates B–H exactly as specified. Do not pack/load the full 753B model before the
   first three architectural proofs pass.

Every result requires exact code/plan provenance. Update this file with current evidence and next
action; detailed measurements belong in `docs/greenfield/PERFORMANCE_LOG.md`.
