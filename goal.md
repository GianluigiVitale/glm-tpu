# Goal — GLM-5.2-FP8 TPU v4 topology-first greenfield engine

FULL ACCESS: work autonomously to completion. Keep below 4,000 characters. After each
start/compaction, read this and `docs/glm-tpu-revolution.md` **in full**; the spec is authoritative.
Inspect live code/processes and relevant handoff/research evidence before acting.

## Scope and precedence

The specification supersedes incremental TP32, old exact-next, and the old pipeline-parallel ban.
Preserve the legacy engine/evidence/oracles/protection tools; stop extending its architecture.
Build an isolated, default-off native-JAX engine for `zai-org/GLM-5.2-FP8` on the existing
`db-v4-64-od` pod (8 hosts/32 TPU-v4 chips) minimizing protected profiler-free single-stream
latency at 256K. Legacy `tpu-inference` is an oracle only; never import its execution path.
Never create a VM, host, or TPU. Use only `gs://driftbench-dsv4-uc` and serialize TPU workflows.

## Required architecture search

- `PP8_LP4`: 8 host-aligned stages × 4 local chips; implement first.
- `PP16_LP2`: 16 topology-adjacent stages × 2 chips; mandatory challenger.
- `WS32_2D`: all-chip 2D weight-stationary plan; protected result or evidence-backed rejection.
- `LEGACY_TP32_DCP8`: measurement oracle only.

Evidence chooses. Distribute weights with pipeline depth; keep repeated communication in
the smallest useful local group; transfer only live residual/compact metadata; never reconstruct
hidden state across all 32 chips inside a layer. `decode_batch1` has one row, no batch-32 dead rows.

## Mandatory order

Use a dedicated branch/worktree and isolated code/scripts/tests/docs. Record starting pins; never
weaken historical evidence. Before loading the full 753B model, prove:

1. runtime physical topology and explicit local replica groups;
2. device-resident PP8/PP16 transfer chains with exact HLO, no host/Ray/Python stage dispatch or
   inactive-stage model compute, and warmed latency distributions;
3. one exact real MoE layer whose combine never leaves its local 2/4-chip stage.

Then pass Gates A–H in order: plan/memory/HLO linter; final-layout checkpoint manifest/packer/direct
loader; reference kernels; exact dense/DSA/IndexShare/MoE layers; complete 2K/8K decoder; protected
128K smoke; protected 256K E0; identical-condition plan adjudication; only then speculation.

## Proof and performance contract

Every optimization defaults off. Require exact DSA set/ties, raw tokens/quality, state/load/cache,
checkpoint bytes/checksums, per-chip peak HBM, code/plan hashes, physical groups/counts, fresh
8-host XPlanes, profiler-free wall, `bench/results.db` linkage, same-region archive and authenticated
zero-work cleanup. CPU/synthetic/HLO/labels/aggregate throughput/contaminated wall are not
performance proof. Stop on full-pod repeated collectives, host staging, dead rows, unknown HBM,
DSA drift, or device-only win with wall regression.

Useful gate: `<=200 ms/token`, `>=4.5` wall tok/s; strong: `<=125 ms`, `>=8 tok/s`; stretch:
`<=100 ms`, `>=10 tok/s`. Separate base decode/speculative throughput; claim 20–50 tok/s only from
protected local evidence.

## Definition of done

The fastest correct plan serves at 256K independent of legacy, uses a plan-aware packed checkpoint,
has local repeated collectives/no full-pod hidden reconstruction, and passes exactness,
quality, integrity, HBM, HLO, PP8/PP16 measurement, WS32 adjudication, 128K smoke, 256K E0,
DB/archive, and clean-fleet gates. Continue until section 18 has direct evidence.
Use logged batches; persist exact next before compaction. Adversarially review every new
code/evidence batch. Prefer one persistent Fable 5 Max Claude CLI chat with
`--dangerously-skip-permissions`; if Fable reports 100% usage, use one independent
Sol agent here with the same scoped diff/evidence and explicit verdict. Correct blockers; never
re-review cleared code. Verify, commit/push, then let the locked same-region cron sync. Never use
Opus, review workflows, or other subagents.
