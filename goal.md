# Goal — GLM-5.2-FP8 TPU v4 topology-first greenfield engine

FULL ACCESS: autonomous. Keep <4000 chars. At start/compaction read this,
`docs/glm-tpu-revolution.md`, and `docs/suggestions.md` **in full**; then inspect live evidence.

## Scope and architecture

Build a default-off native-JAX `zai-org/GLM-5.2-FP8` engine on `db-v4-64-od` (8 hosts/32 chips),
minimizing protected 256K latency. Legacy is oracle only; never import execution.
Never create infra. Use only
`gs://driftbench-dsv4-uc` and serialize TPU work.

Plans: `PP8_LP4` (8 host-aligned stages ×4, first), `PP16_LP2` (16 adjacent stages ×2),
`WS32_2D` (protected result or evidence rejection), and `LEGACY_TP32_DCP8` (oracle only).
Distribute weights by depth; keep repeated communication in the smallest local group; move live
state only; no 32-chip hidden reconstruction/layer. `decode_batch1` has one row/no dead rows.

## Mandatory order

Worktree/pins. Pre-model prove topology/groups; device-only PP8/PP16 chains with exact HLO/no host
dispatch/inactive compute; one exact real MoE local combine. Gates A–H: plan/memory/HLO; packed
manifest/loader; references; exact layers; complete short decoder; protected 128K/256K E0; plan
adjudication; then speculation.

## Gate D: observe, find and fix root cause

Gate D open. `context<=top_k` is not ranking proof; never rerun tombstoned graphs. Use typed
snapshots/watchpoints, first-divergence bisection, bits/dtypes/shapes/layouts/owners, coherent
cache/query/head/key/scorer state, HLO fingerprints and one bounded host transfer. Bind source/code/
plan SHAs; append-only, fail closed, offline-first; attack-test every authority boundary.

Read/register first: `docs/10-observability.md`, `HANDOFF.md`, `bench/results.db`, `docs/artifacts/`,
and `docs/greenfield/{REUSE_INVENTORY,EVIDENCE_MAP,GATE_D_LESSONS,GATE_D_OBSERVABILITY_PLAYBOOK}.md`;
inspect observability/admission core, scripts, configs and validation/benchmarking/sharding tools.
When stuck reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`
in full and adjudicate it against local evidence.

## Resume checkpoint — 2026-09-01 23:30Z

Worktree `/home/gianl/glm-tpu-gate-d-pp16-numerical`, branch
`tooling/gate-d-compensated-pp16-numerical`; parent `f762a28899b812b43f321c4e56bfc2f94f0a9811` is on
origin and the same-region mirror. This batch persists corrected numerical orchestration/install
sources, tests and docs: the immutable publisher prints the only `NUMERICAL_RESULT` status authority
after terminal replay; the wrapper dispatches only from it, local terminal/receipt is a supplementary
check; substitution regressions executed; publisher→wrapper→launcher→installer→analyzer repinned;
Bash/Ruff/74 focused tests pass; Sol closed both P1s.

**STOP: no install/launcher/TPU run is authorized.** Exact next after commit/push/mirror: generate
and persist the orchestration source certificate from the analyzer; separately review the literal
install-only commands, install without launch; separately review one fresh tag/command; then
run/adjudicate the bounded discriminator once.

## Proof and finish

Optimizations off. Require exact DSA/tokens/quality; state/load/cache/checkpoint integrity; per-chip
HBM; code/plan hashes; groups/counts; fresh 8-host XPlanes; wall; DB/archive and 8/8 cleanup.
CPU/synthetic/HLO/labels/throughput alone are not proof.
Stop on global repeated collectives, host staging, dead rows, unknown HBM, DSA drift or wall
regression. Useful `<=200 ms/token, >=4.5 tok/s`; strong `<=125 ms, >=8`; stretch `<=100 ms, >=10`.

Finish only with §18 evidence: independent 256K service, packed checkpoint, exactness/integrity,
local collectives, PP8/PP16 measurements, WS32 adjudication, 128K/256K, DB/archive, clean fleet.
Review new batches in Fable 5 Max CLI; when unavailable/100% use the one existing Sol on the same
scope. Never Opus, workflows or extra subagents. Use smallest tests; verify, commit/push, then locked
same-region sync.
