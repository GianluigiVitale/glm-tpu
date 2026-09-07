# AGENTS — topology-first greenfield rewrite

This file applies only to branch `rewrite/topology-first-decode` in worktree
`/home/gianl/glm-tpu-topology-rewrite`.

## Authority and resume order

Read these files before acting, in this order:

1. `goal.md` in full (compact, compaction-safe contract).
2. `docs/glm-tpu-revolution.md` in full (binding engineering specification).
3. `HANDOFF.md`, repository state, and live process/fleet state (current status only).
4. Relevant `docs/RESEARCH_LOG.md`, protected artifacts, and observability documentation.

Before implementing a component that may overlap prior work, also read
`docs/greenfield/REUSE_INVENTORY.md` and its machine-readable
`configs/greenfield-reuse-inventory.json`. Update the registry when a source is reused, adapted,
rejected, or reserved for a later challenger. It does not override `goal.md` or the binding
specification.

If documents conflict, `goal.md` and `docs/glm-tpu-revolution.md` win. `HANDOFF.md` reports mutable
status and must not change the goal. `KICKOFF.md` and `PLAN.md` are summaries. `docs/suggestions.md`,
`docs/RESEARCH_LOG.md`, old campaign documents, and protected artifacts are evidence/advice, not
current execution instructions.

Files in `/home/gianl/glm-tpu`, any other worktree, or the legacy `tpu-inference` repository do not
govern this branch. In particular, the old incremental TP32 sequence and old prohibition on
pipeline parallelism are superseded. Never edit or delete the owner's untracked files in the main
checkout. Preserve historical evidence.

## Working contract

- Build the default-off native-JAX engine only in this worktree and under the isolated greenfield
  tree. Legacy execution is an oracle only and must never be imported.
- Work autonomously, commit and push small reviewable changes, and update `HANDOFF.md` when evidence
  or the exact next action changes.
- Use only the existing `db-v4-64-od` 32-chip pod and `gs://driftbench-dsv4-uc`; never create compute.
- Serialize TPU workflows. Prove ownership before cleanup and finish protected runs with an
  authenticated eight-host zero-work census.
- The topology, device-resident transport and exact topology-local MoE-layer gates have passed, and
  `WS32_2D` is promoted (§22), so the complete sealed checkpoint is loaded by protected runs. A NEW
  plan or transport re-enters that order: mechanisms first, full checkpoint only after its gates.
- CPU, synthetic, or HLO evidence proves mechanisms only. Performance claims require the protected
  profiler-free wall, correctness, provenance, DB, archive, trace, HBM, and cleanup contract.
- Continue until every Section 18 Definition-of-Done item in the binding specification has direct
  evidence.

Before editing or launching, verify that `git rev-parse --show-toplevel` is exactly
`/home/gianl/glm-tpu-topology-rewrite` and the branch is `rewrite/topology-first-decode`.
