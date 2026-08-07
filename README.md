# glm-tpu — topology-first GLM-5.2 TPU-v4 rewrite

This branch contains an isolated, default-off native-JAX inference engine for
`zai-org/GLM-5.2-FP8` on the existing 8-host/32-chip TPU-v4 pod. Its primary objective is minimum
protected batch-one latency at 256K context.

The implementation starts with `PP8_LP4`, must challenge it with `PP16_LP2`, and must protectively
measure or evidence-reject `WS32_2D`. Repeated layer communication stays in the smallest useful
topology-local group; only the live residual and compact metadata cross pipeline stages.

Read [AGENTS.md](AGENTS.md), [goal.md](goal.md), and
[docs/glm-tpu-revolution.md](docs/glm-tpu-revolution.md) before changing code. Current evidence and
the next action are in [HANDOFF.md](HANDOFF.md). Before implementing a component, consult the pinned
[reuse inventory](docs/greenfield/REUSE_INVENTORY.md) so existing protection, parity, kernel,
checkpoint, benchmark, and architecture work is reused or deliberately classified.

The inherited `tpu-inference` implementation is a correctness and measurement oracle only. This
branch does not extend or import its model-execution path. Historical files under `docs/` remain
available as evidence; they do not override the greenfield contract.
