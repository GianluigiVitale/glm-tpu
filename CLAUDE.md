# CLAUDE — greenfield branch compatibility pointer

Continue autonomously to full project completion under §18 amended by §24. Efficient multi-token
prefill is now required. Original depth0.05 sealed DB575; DO NOT launch the next serial
128K/256K run. Next: ranked efficiency audit, bounded batched layers, short decoder, efficient L7/L8.
Read `HANDOFF.md`, `goal.md` and `docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md` before action.
Historical D/G and serial L7 DB573–575 remain valid, not proof of changed prefill.

This file applies only to branch `rewrite/topology-first-decode` in
`/home/gianl/glm-tpu-topology-rewrite`.

`AGENTS.md` is the branch-local operating file. Start or resume by reading, in order:

1. `AGENTS.md` in full.
2. `goal.md` in full.
3. `docs/glm-tpu-revolution.md` in full.
4. `HANDOFF.md` and current repository/runtime state.
5. Relevant historical evidence in `docs/RESEARCH_LOG.md` and protected artifacts.

`goal.md` and the tracked branch copy of `docs/glm-tpu-revolution.md` are authoritative. They
supersede the legacy incremental TP32 plan and the former pipeline-parallelism ban. `KICKOFF.md` and
`PLAN.md` summarize the same greenfield contract; they do not narrow it.

Build the new model-execution path only under the isolated greenfield tree. The legacy
`tpu-inference` engine is a correctness/measurement oracle and source of isolated validation
utilities, never an execution dependency. Preserve all historical evidence and protection tools.

`WS32_2D` is the promoted plan (§22); the PP8/PP16 pipeline sequence is retained history.

Use only the existing `db-v4-64-od` 32-chip pod and `gs://driftbench-dsv4-uc`. Never create compute.
Serialize TPU workflows, prove exact ownership before cleanup, and keep every optimization
default-off until its required gates pass. The topology, device-resident stage transport and exact
topology-local MoE-layer gates have passed, so protected runs load the complete sealed checkpoint; a
NEW plan or transport re-enters that order, mechanisms first.

Work autonomously, commit and push reviewable greenfield changes, update `HANDOFF.md` and the
greenfield performance log as evidence changes, and continue until Section 18 of the full
specification is directly proved.

Do not use instruction, handoff, goal, or kickoff files from `/home/gianl/glm-tpu` or another
worktree; those describe historical campaigns and have no authority here.
