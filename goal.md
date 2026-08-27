# Goal — GLM-5.2-FP8 TPU v4 topology-first greenfield engine

FULL ACCESS: work autonomously to completion. Keep below 4,000 characters. After start/compaction
read it and `docs/glm-tpu-revolution.md` **in full**, then inspect live state and evidence.

## Scope and precedence

Supersede incremental TP32, sequencing and the pipeline ban. Preserve legacy evidence/oracles;
never extend its execution path. Build an isolated, default-off native-JAX engine for
`zai-org/GLM-5.2-FP8` on
`db-v4-64-od` (8 hosts/32 TPU-v4 chips), minimizing protected profiler-free 256K stream latency.
Legacy `tpu-inference` is oracle only; never import its execution path.
Never create a VM, host, or TPU. Use only `gs://driftbench-dsv4-uc` and serialize TPU workflows.

## Required architecture search

- `PP8_LP4`: 8 host-aligned stages × 4 local chips; implement first.
- `PP16_LP2`: 16 topology-adjacent stages × 2 chips; mandatory challenger.
- `WS32_2D`: all-chip 2D weight-stationary plan; protected result or evidence-backed rejection.
- `LEGACY_TP32_DCP8`: measurement oracle only.

Evidence chooses. Distribute weight capacity with pipeline depth; keep repeated communication in
the smallest useful local group; transfer only live residual/compact metadata; never reconstruct
hidden state across all 32 chips inside a layer. `decode_batch1` has one row, no batch-32 dead rows.

## Mandatory order

Use a dedicated branch/worktree and isolated greenfield code/scripts/tests/docs. Record starting
pins; never weaken historical evidence. Before loading the full 753B model, prove:

1. runtime physical topology and explicit local replica groups;
2. device-resident PP8/PP16 transfer chains with exact HLO, no host/Ray/Python stage dispatch or
   inactive-stage model compute, and warmed latency distributions;
3. one exact real MoE layer whose combine never leaves its local 2/4-chip stage.

Then pass Gates A–H in order: plan/memory/HLO linter; final-layout checkpoint manifest/packer/direct
loader; reference kernels; exact dense/DSA/IndexShare/MoE layers; complete 2K/8K decoder; protected
128K smoke; protected 256K E0; identical-condition plan adjudication; only then speculation.

Do not wait for Gates E–H to upstream DSA. Re-audit `tpu-inference` main/PRs, then cut the smallest
TPU DSA scorer/indexer PR from protected v4 evidence. Keep it independent; respond/rebase until
merged. Greenfield remains the oracle.

## Proof and performance contract

Every optimization defaults off. Require exact DSA set/tie order, raw tokens/quality,
state/load/cache integrity, checkpoint bytes/checksums, per-chip peak HBM, code/plan hashes,
physical collective groups/counts, fresh 8-host XPlanes, profiler-free steady wall,
`bench/results.db` linkage, same-region archive, and authenticated zero-work cleanup. CPU/synthetic/
HLO-only/aggregate or contaminated wall data are not performance proof. Stop on full-pod
repeated layer collectives, host-staged transport, dead rows, unknown HBM margin, DSA drift, or a
device-only win with wall regression.

Minimum useful gate: `<=200 ms/token` and `>=4.5` wall tok/s. Strong base target: `<=125 ms` and
`>=8 tok/s`; stretch: `<=100 ms` and `>=10 tok/s`. Base decode and speculative effective throughput
must be separate; claim 20–50 tok/s only from protected local evidence.

## Definition of done

The fastest correct plan serves 256K independently, uses a plan-aware checkpoint/local collectives,
and passes exactness, quality, integrity, HBM/HLO, PP8/PP16/WS32, 128K, 256K, DB/archive and clean
fleet gates. Continue until section 18 has direct evidence.
Work in bulk. Never use Fable/Opus. Before deploy: at most one fresh Sol audit of current
diff/evidence; no workflows/subagents/cleared-code review. Fix, commit, push. After failure,
diagnose locally; optional narrow Sol review. Checkpoint each batch: commit+push+verify mirror only
in `gs://driftbench-dsv4-uc` (`US-CENTRAL2`); fail closed on location drift, never EU. Keep sync
outside TPU timing.
