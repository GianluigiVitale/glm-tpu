# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-05 11:49 UTC

## Authority and location

- Branch: `rewrite/topology-first-decode`
- Worktree: `/home/gianl/glm-tpu-topology-rewrite`
- Starting harness commit: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`
- Compact contract: `goal.md`
- Full contract: `docs/glm-tpu-revolution.md`

Those two contract files supersede the inherited incremental TP32 instructions and old ban on
pipeline parallelism. Legacy execution code and evidence remain intact on `main` and in Git history.

## Current state

The branch has just been created. No greenfield engine code or TPU performance claim exists yet.
The existing legacy engine is an oracle only. A protected legacy compute-row 128K smoke was already
running from the main worktree when this branch was created; do not compete for the TPU. Let its
owner workflow archive and clean up, then require an eight-host zero-work census before Gate-A metal.

## Exact next sequence

1. Commit and push the greenfield contract reset without changing the live main checkout.
2. Inventory reusable validation/observability/provenance/cleanup utilities without importing the
   legacy model-execution path. Record starting pins and a read-only evidence map.
3. Implement immutable geometry/execution-plan types, synthetic topology fixtures, physical
   topology discovery, explicit PP8/PP16 groups, plan hashing, and validation.
4. Implement the HLO contract/linter and dependent local-collective benchmark.
5. Prove device-resident PP8 and PP16 stage-transfer skeletons with no host/Ray/Python stage dispatch.
6. Prove one exact real MoE layer whose combine is confined to its 2/4-chip stage.
7. Continue through Gates B–H exactly as specified. Do not pack/load the full 753B model before the
   first three architectural proofs pass.

Every result requires exact code/plan provenance. Update this file with current evidence and next
action; detailed measurements belong in `docs/greenfield/PERFORMANCE_LOG.md`.
