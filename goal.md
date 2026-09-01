# Goal — GLM-5.2-FP8 TPU v4 topology-first greenfield engine

FULL ACCESS: autonomous work. Keep <4000 chars. At start/compaction read this and
`docs/glm-tpu-revolution.md` and `docs/suggestions.md` **in full**; inspect live evidence.

## Scope/precedence

Default-off native JAX `zai-org/GLM-5.2-FP8` on `db-v4-64-od` (8 hosts/32 chips); minimize
protected 256K latency. Preserve legacy evidence/oracles/tools only. No new infra; only
`gs://driftbench-dsv4-uc`; serialize TPU work.

## Architecture search

- `PP8_LP4`: host-aligned 8×4 first.
- `PP16_LP2`: adjacent 16×2 mandatory.
- `WS32_2D`: all-chip 2D; result/rejection evidence.
- `LEGACY_TP32_DCP8`: oracle only.

Evidence chooses. Distribute weights by depth; communicate locally; move live state; no 32-chip hidden
reconstruction/layer. `decode_batch1`: one row, no dead rows.

## Order

Worktree; pin. Pre-model prove
topology/groups; device-only PP8/PP16 chains with exact HLO, no host/Ray/Python dispatch/inactive
compute, warmed distributions; one exact MoE layer with local 2/4-chip combine.
Pass Gates A–H: plan/memory/HLO; final-layout manifest/packer/loader; reference kernels; exact
dense/DSA/IndexShare/MoE; complete cutoff-active short decoder; protected 128K;
protected 256K E0; identical plan adjudication; then speculation.

## Gate D: observe; find/fix root cause

Gate D open; `context<=top_k` is not ranking proof; never rerun tombstoned graphs. Build
default-off typed snapshots/watchpoints, first-divergence bisection, bits/dtypes/shapes/layouts/
owners, coherent cache/query/head/key/scorer state and causal HLO fingerprints/artifact diffs.
Prefer device buffers plus one bounded transfer. Host use requires unchanged executable/outputs/DSA.
Bind source/code/plan SHAs; append-only, fail closed, offline-first; test attacks.

Oracle: `docs/10-observability.md`. Indexes:
`docs/greenfield/{REUSE_INVENTORY,EVIDENCE_MAP,GATE_D_LESSONS,GATE_D_OBSERVABILITY_PLAYBOOK}.md`,
`configs/greenfield-reuse-inventory.json`. Core/config/CLI:
`glm_tpu/greenfield/{observability,gate_d_admission}.py`,
`configs/greenfield-gate-d-{observability,mechanism-admission}.json`,
`scripts/greenfield/{audit_observability,admit_gate_d_mechanisms}.py`. Search
`glm_tpu/greenfield/{validation,benchmarking,sharding}/`, `scripts/greenfield/{capture,compare,inspect,probe,trace}*` and
`tests/greenfield/`. Evidence: `docs/artifacts/`, `HANDOFF.md`, `bench/results.db`, bucket
`oracles/`/`results/`. Read/register first. On any stall/failure, reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`
in full; adjudicate against local evidence before acting.

Localize first causal divergence; define/falsify a legal one-row local mechanism on the smallest
coherent state; compile/run only if it passes. Gate D requires root cause/fix.

## Proof/performance contract

Optimizations off. Require exact DSA sets/ties, tokens/quality, state/load/cache/checkpoint checksums,
per-chip HBM, code/plan hashes, groups/counts, fresh 8-host XPlanes, profiler-free wall,
DB/archive/authenticated cleanup. CPU/synthetic/HLO/labels/throughput/contaminated wall are not
performance proof. Stop on full-pod collectives, host staging, dead rows, unknown HBM, DSA drift or
wall regression. Useful: `<=200 ms/token`, `>=4.5 tok/s`; strong:
`<=125 ms`, `>=8 tok/s`; stretch: `<=100 ms`, `>=10 tok/s`. Separate base/speculative throughput.

## Definition/workflow

Finish only with §18 evidence: independent 256K service; packed checkpoint; exactness/integrity,
HBM/HLO/local collectives; plan adjudication; 128K/256K; DB/archive; clean fleet.
Log batches/exact next. Review each new batch in Fable 5 Max CLI
(`--dangerously-skip-permissions`); at 100% use one Sol on the same scope. Fix blockers; don't
re-review cleared code. Use smallest checks, reuse proof, preflight locks/tags; serialize TPU
without weakening correctness. Verify, commit/push, then locked same-region sync. Never use Opus,
workflows or other subagents.
