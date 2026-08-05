# KICKOFF — GLM-5.2 TPU-v4 topology-first rewrite

Build a new default-off native-JAX inference engine for `zai-org/GLM-5.2-FP8` on the existing
8-host/32-chip TPU-v4 pod. Optimize protected batch-one 256K latency, not aggregate throughput.

The complete, binding specification is `docs/glm-tpu-revolution.md`; read it in full. `goal.md` is
the compact compaction-safe pointer. Both supersede the old incremental TP32 campaign and old
pipeline-parallelism prohibition.

Start with `PP8_LP4`, challenge it with `PP16_LP2`, and protect or evidence-reject `WS32_2D`.
Distribute capacity with depth, keep repeated layer communication topology-local, move only live
residual/compact metadata between stages, and never reconstruct hidden state over all 32 chips
inside a transformer layer.

Before loading the full checkpoint, prove physical topology/groups, device-resident PP8/PP16
transport, and one exact topology-local MoE layer. Finish only when every Definition-of-Done item in
Section 18 has direct protected evidence.
