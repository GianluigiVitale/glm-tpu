# CLAUDE — greenfield branch compatibility pointer

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

Use only the existing `db-v4-64-od` 32-chip pod and `gs://driftbench-dsv4-uc`. Never create compute.
Serialize TPU workflows, prove exact ownership before cleanup, and keep every optimization
default-off until its required gates pass. Do not load the full checkpoint before the topology,
device-resident stage transport, and exact topology-local MoE-layer gates pass.

Work autonomously, commit and push reviewable greenfield changes, update `HANDOFF.md` and the
greenfield performance log as evidence changes, and continue until Section 18 of the full
specification is directly proved.

Do not use instruction, handoff, goal, or kickoff files from `/home/gianl/glm-tpu` or another
worktree; those describe historical campaigns and have no authority here.
