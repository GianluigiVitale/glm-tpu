# Greenfield architecture state

The binding design is `../glm-tpu-revolution.md`. This file records implemented structure, not a
replacement specification.

## Isolation

- Branch: `rewrite/topology-first-decode`
- Worktree: `/home/gianl/glm-tpu-topology-rewrite`
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`
- Starting legacy-oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`
- Greenfield modules: `glm_tpu/greenfield`; legacy execution is never imported.
- Root `goal.md` and `docs/glm-tpu-revolution.md` supersede inherited TP32 instructions. The owner's
  untracked main-worktree `AGENTS.md` remains untouched and is absent from this worktree.

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

## Still unimplemented

No transport executable, HLO linter, collective benchmark, model layer, checkpoint packer/loader,
decoder, or serving path exists. Gate A is therefore only partially complete. The next executable
must be a synthetic device-resident PP8/PP16 stage chain—not a model port.
