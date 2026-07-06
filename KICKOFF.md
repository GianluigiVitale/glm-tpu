# GLM-5.2 → TPU v4 PORT — /goal KICKOFF (≤4k)

SOLO, FULLY AUTONOMOUS. Port **GLM-5.2 (FP8)** to **TPU v4** the way DeepSeek-V4-Flash was done. Unlimited fleet.

**DO NOT STOP, DO NOT ASK, DO NOT CHECK IN** until BOTH: (1) GLM-5.2-FP8 runs correctly on the pod, AND (2) the
reproduced benchmarks match the HF card within noise. Never pause at phase boundaries, for reassurance, or on a
blocker — self-correct, try other approaches, keep going. **When in doubt, automatically pick the option you'd
recommend to the user and proceed — log it, never ask.** ~10 h from now you must be DONE or still actively
working; do not stop for ANY reason. Only a genuine hard block (a credential you lack, an owner-gated
irreversible push) pauses THAT thread — keep working every other track meanwhile.

## Read FIRST, in full (don't reinvent — DSV4 is your base)
`~/glm-tpu`: `HANDOFF.md`→`CLAUDE.md`→`PLAN.md`→`docs/00-feasibility-memo.md`. Then the DSV4 base you reuse:
`~/moe-tpu/{CLAUDE.md,HANDOFF.md,docs/09,11,12,13,15,16}` + fork `~/tpu-inference@dsv4-flash-v4`. `CLAUDE.md
§"What transfers"` maps every reusable piece.

## Model + Goal
`zai-org/GLM-5.2-FP8` (FP8 ~744 GB, fits 1024 GB HBM). `GlmMoeDsaForCausalLM`, 753B/40B, MLA + **DSA**
(index_topk=2048, 32 indexer heads, IndexShare, interleaved-RoPE), MTP, 1M ctx. Full spec docs/00; HF_TOKEN in
`~/glm-tpu/.env`. Goal: (1) correct forward vs the HF/GPU reference (dense-MLA first); (2) reproduce the HF-card
benchmarks within noise, FULL PROVENANCE; (3) **the headline — a DSA lightning-indexer/top-k sparse-MLA Pallas
kernel** (none exists on TPU; PR #2324 DISABLES the indexer; your DSV4 CSA kernel is the closest — adapt it).

## Phases (thresholds PLAN.md; first steps HANDOFF)
Stage 1 dense-MLA correctness (port PR #2324 registry + v4 MLA workarounds; first benchmark within ~1–2 pts) →
Stage 2 the DSA kernel (selected-set-exact + bit-faithful; passkey ≥95% to ≥128K) → Stage 3 IndexShare + MTP.

## Methodology (EXACTLY DSV4's)
FIX the root cause, don't patch. Every change gated (byte-identical off) + additive + CPU test + sub-cube
zero-recompile + pod **3/3**. Correctness before performance — validate each component vs the reference at a tiny
config, then real dims. **Spawn adversarial reviewers** after each change. Code PR-compatible with
`vllm-project/tpu-inference` (Googler-maintained AGENTS.md — no code-agent PRs; **owner submits**).

## COST — 2 HARD RULES + prefer-streaming
Pod `db-v4-64-od`, us-central2-b, 32 v4 chips. **HARD RULES: (1) the bucket MUST be same-region —
`gs://driftbench-dsv4-uc` (us-central2), NEVER the EU `gs://driftbench-storage` (the bill driver); (2) NEVER
create/request a new compute machine or any TPU — only these 32 v4 chips / 8 hosts.** ONE FP8 copy (no bf16).
**Prefer streaming** (stage HF→GCS once via `~/moe-tpu/scripts/stage_base_to_gcs.py`, then runai_streamer
GCS→HBM; FP8 stays RESIDENT + dequant per-tile in-kernel — NOT load-time bf16, which OOMs 753B). BUT if a
per-host local copy proves required you ARE authorized to
**create + attach a ~1000 GB disk to each of the 8 hosts** and copy from the bucket (disks on the existing hosts
OK; new machines/TPUs NOT). Clean up after.

## Benchmarks + PROVENANCE (machinery pre-built in `bench/`)
Reproduce the HF-card benchmarks (card: GPQA-Diamond, AIME; + fast sanity gates MMLU-Pro/GSM8K, not on the card
— all cached; agentic SWE-bench/Terminal-Bench later). The SQLite provenance DB + harness exist — wire the model
into `bench/run_bench.py::make_generate()`; it stores per item prompt+**raw output verbatim**/gold/extracted/
correct/timestamp/tokens/latency + run provenance. **Never run anything untraceable.** Honest nulls; signed Δ.

## First task
Stage 1 per HANDOFF: fork `glm-5.2-v4` off `dsv4-flash-v4`; stage `zai-org/GLM-5.2-FP8` → us-central2; register
the arch; dense-MLA correctness; wire the bench + reproduce a benchmark. Commit+push often (no force-push).
