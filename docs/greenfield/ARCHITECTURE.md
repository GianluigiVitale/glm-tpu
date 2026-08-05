# Greenfield architecture state

The binding design is `../glm-tpu-revolution.md`. This file records implemented structure, not a
replacement specification.

## Isolation

- Branch: `rewrite/topology-first-decode`
- Worktree: `/home/gianl/glm-tpu-topology-rewrite`
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`
- Starting legacy-oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`
- Greenfield modules: `glm_tpu/greenfield`; legacy execution is never imported.
- Branch-local `AGENTS.md`, root `goal.md`, and `docs/glm-tpu-revolution.md` supersede inherited TP32
  instructions. The owner's untracked main-worktree `AGENTS.md` remains untouched and has no authority
  in this worktree.

## Implemented contracts

`ModelGeometry`, `PhysicalDevice`, `PhysicalTopology`, `StageAssignment`, and `ExecutionPlan` are
frozen, typed, canonical-JSON serializable, round-trippable, and content-addressed with SHA-256.
Validation refuses incomplete schedules, topology ambiguity, layer gaps, device reuse, cross-host
local groups, wrong named-plan geometry, and incomplete memory classes.

Runtime discovery reads device id, JAX process, physical coordinates, core-on-chip, platform, kind,
and observed local order. TPU v4 exposes no usable `local_hardware_id`, so every process contributes
its actual `jax.local_devices()` order and the fleet gathers a complete mapping. No device ordering
is inferred from ids.

PP8 and PP16 groups are derived from physical coordinates. Stage order is a deterministic
Hamiltonian ring. Every stage boundary, including last-to-first token return, requires a distinct
physical-neighbor match for every transfer lane. PP16 first chooses adjacent two-chip pairs along
the physical length-two axis where available.

The HLO contract parser reads optimized textual XLA HLO and records physical replica groups,
source-target pairs, result/operand shapes, source metadata, counts, channels, and global-id
semantics. Policies reject missing/wrong groups, repeated full-pod residual synchronization, dead
batch-32 rows, and count drift. Full-pod collective diagnostics require an explicit non-promotable
policy; XLA all-to-all's absence of a global-id flag is a narrow named exemption while its exact
replica groups remain mandatory. Because XLA numbers groups in executable partition order, every
policy records the exact partition-to-physical-device assignment and reports both logical and
physical groups/pairs.

The dependent collective benchmark supports control, all-reduce, reduce-scatter, all-gather,
collective-permute, all-to-all, and fused tuple all-reduce over physical groups of 2/4/8/32 chips.
Rank-dependent state, nonlinear cross-iteration feedback, optimization barriers, bitwise output
checksums, and optimized-HLO count assertions prevent elision. Protected configs require a chain of
75, at least 200 warmups, at least 1,000 measured samples, and report complete distributions. TPU
optimized HLO is authoritative: small decode reduce-scatter currently fails closed because TPU-v4
XLA rewrites it to all-reduce, while the other supported operations preserve the exact contract.

## Still unimplemented

No transport executable, model layer, checkpoint packer/loader, decoder, or serving path exists.
Protected TPU dependent-chain matrices now cover the dominant payload, required bf16 live-
residual/intermediate shapes, `f32[1,6144]`, and small `int32` routing metadata. FP8 is a checkpoint
weight-storage format here, not a numerically valid residual/reduction or stage-transfer payload;
the numerical contract uses bf16/f32 for those paths. The matrices show that 75 full-pod
all-reduces have a `3.941 ms` fleet-max
p50 and 75 full-ring nearest-neighbor permutes `0.791 ms`, versus `106.495 ms/token` attributed to
the legacy MoE combine region. This isolates legacy arrival/layout/barrier behavior rather than raw
small-payload ICI as the dominant loss. It is mechanism evidence, not model throughput. The next
executable is the synthetic device-resident PP8/PP16 stage chain—not a model port.
