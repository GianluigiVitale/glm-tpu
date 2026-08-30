# Goal — GLM-5.2-FP8 TPU v4 topology-first greenfield engine

FULL ACCESS: work autonomously. Keep <4,000 chars. At start/compaction read this,
`docs/glm-tpu-revolution.md` and `docs/suggestions.md` **in full**, then inspect live evidence.

## Scope and precedence

Build an isolated default-off native-JAX `zai-org/GLM-5.2-FP8` engine on `db-v4-64-od`
(8 hosts/32 v4 chips), minimizing protected 256K single-stream latency; supersedes old rules.
Preserve legacy evidence/oracles/tools, not execution. Create no infrastructure; use only
`gs://driftbench-dsv4-uc`; serialize TPU work.

## Required architecture search

- `PP8_LP4`: host-aligned 8×4; first.
- `PP16_LP2`: adjacent 16×2; mandatory.
- `WS32_2D`: all-chip 2D; result or evidenced rejection.
- `LEGACY_TP32_DCP8`: oracle only.

Evidence chooses. Distribute weights with depth; communicate locally; move live state only; no
32-chip hidden reconstruction/layer. `decode_batch1`: one row, no dead rows.

## Mandatory order

Isolated worktree; record pins; preserve evidence. Before full model prove
topology/groups; device-only PP8/PP16 chains with exact HLO, no host/Ray/Python dispatch/inactive
compute, warmed distributions; one exact MoE layer with local 2/4-chip combine.
Pass Gates A–H: plan/memory/HLO; final-layout manifest/packer/loader; reference kernels; exact
dense/DSA/IndexShare/MoE; complete cutoff-active short decoder; protected 128K;
protected 256K E0; identical plan adjudication; then speculation.

## Active priority: observe Gate D; find/fix root cause

Gate D is open; `context<=top_k` is not ranking proof; never rerun tombstoned graphs. Build
default-off typed snapshots/watchpoints, first-divergence bisection, bit/dtype/shape/layout/owner
data, coherent cache/query/head/key/scorer state, causal HLO fingerprints and artifact diffs.
Prefer device buffers plus one bounded transfer. Host consumers require unchanged executable,
outputs and DSA. Bind source/code/plan SHAs; append-only, fail closed, offline-first; test attacks.

Observability: `docs/10-observability.md` (legacy oracle);
`docs/greenfield/{REUSE_INVENTORY,EVIDENCE_MAP,GATE_D_LESSONS}.md` and
`configs/greenfield-reuse-inventory.json` are indexes. Tools:
`configs/greenfield-gate-d-observability.json`, `glm_tpu/greenfield/observability.py`,
`scripts/greenfield/audit_observability.py`. Search `glm_tpu/greenfield/{validation,
benchmarking,sharding}/`, `scripts/greenfield/{capture,compare,inspect,probe,trace}*` and
`tests/greenfield/`. Evidence: `docs/artifacts/`, `HANDOFF.md`, `bench/results.db`, bucket
`oracles/`/`results/`. Read/register first.

Localize the first causal divergence; define a legal one-row local mechanism; falsify it on the
smallest coherent state; compile/run only if it passes. Gate D requires root cause and fix.

## Proof and performance contract

Optimizations default off. Require exact DSA sets/ties, tokens/quality, state/load/cache/checkpoint
checksums, per-chip HBM, code/plan hashes, groups/counts, fresh 8-host XPlanes, profiler-free wall,
DB/archive/authenticated cleanup. CPU/synthetic/HLO/labels/throughput/contaminated wall are not
performance proof. Stop on full-pod collectives, host staging, dead rows, unknown HBM, DSA drift or
wall regression. Useful: `<=200 ms/token`, `>=4.5 tok/s`; strong:
`<=125 ms`, `>=8 tok/s`; stretch: `<=100 ms`, `>=10 tok/s`. Separate base/speculative throughput.

## Definition of done and workflow

Finish only with §18 direct evidence: independent 256K service; packed checkpoint; exactness/integrity,
HBM/HLO, local collectives, plan adjudication, 128K/256K, DB/archive and clean fleet.
Log batches/exact next. Review each new batch in Fable 5 Max CLI
(`--dangerously-skip-permissions`); at 100% use one Sol on the same scope. Fix blockers; don't
re-review cleared code. Use smallest checks, reuse proof, preflight locks/tags; serialize TPU
without weakening correctness. Verify, commit/push, then locked same-region sync. Never use Opus,
workflows or other subagents.
