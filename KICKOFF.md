# KICKOFF — GLM-5.2 TPU-v4 topology-first rewrite

Continue autonomously to full project completion under §18.
Gate D/G closed; L7 depth 1.0 sealed, depth 0.0 still running under a detached monitor. Exact recovery: latest
`HANDOFF.md`, `goal.md`, and `docs/greenfield/WS32_ORPHAN_RECOVERY.md`.

Build a new default-off native-JAX inference engine for `zai-org/GLM-5.2-FP8` on the existing
8-host/32-chip TPU-v4 pod. Optimize protected batch-one 256K latency, not aggregate throughput.

The complete, binding specification is `docs/glm-tpu-revolution.md`; read it in full. `goal.md` is
the compact compaction-safe pointer. Both supersede the old incremental TP32 campaign and old
pipeline-parallelism prohibition.

That contest is decided: `WS32_2D` is PROMOTED (§22/Gate G, 2.00x faster than `PP8_LP4` at 2K under
identical protected conditions, both exact; `PP16_LP2` rejected with evidence). `PP8_LP4` and
`PP16_LP2` are retained history. Keep repeated layer communication inside the promoted group.

The topology/group, device-resident transport and exact topology-local MoE-layer gates have passed,
so protected runs load the complete sealed checkpoint; a NEW plan or transport re-enters that order,
mechanisms first. Finish only when every Definition-of-Done item in Section 18 has direct protected
evidence.
