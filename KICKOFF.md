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
`~/moe-tpu/{CLAUDE.md,HANDOFF.md,docs/09,11,12,13,15,16}` + fork `~/tpu-inference@dsv4-flash-v4` (CSA/HCA Pallas
kernel + head-shard + v4 enablement). `CLAUDE.md §"What transfers"` maps every reusable piece.

## Model
`zai-org/GLM-5.2-FP8` (FP8 ~744 GB, fits 1024 GB HBM; BF16 ~1.5 TB doesn't). `GlmMoeDsaForCausalLM`, 753B/40B,
78L (3 dense+75 MoE), 256+1 experts top-8 sigmoid noaux_tc, MLA + **DSA** (index_topk=2048, 32 indexer heads,
index_head_dim 128, IndexShare, interleaved-RoPE), MTP, 1M ctx. HF_TOKEN in `~/glm-tpu/.env`.

## Goal (correct → faithful → fast)
1. Correct — forward matches the HF/GPU(SGLang/vLLM) reference (fp32+bf16); dense-MLA first.
2. Faithful — reproduce the HF-card benchmarks within noise, FULL PROVENANCE (below).
3. **Fast — the headline: a DSA lightning-indexer/top-k sparse-MLA Pallas kernel.** None exists on TPU (PR #2324
   DISABLES the indexer). Your DSV4 CSA kernel is the closest — adapt it.

## Phases (thresholds PLAN.md; first steps HANDOFF)
Stage 1 dense-MLA correctness (port PR #2324 registry + v4 MLA workarounds; first benchmark within ~1–2 pts) →
Stage 2 the DSA kernel (selected-set-exact + bit-faithful; passkey ≥95% to ≥128K) → Stage 3 IndexShare + MTP.

## Methodology (EXACTLY DSV4's)
FIX the root cause, don't patch. Every change gated (byte-identical off) + additive + CPU test + sub-cube
zero-recompile + pod **3/3** (halts are probabilistic). Correctness before performance — validate each component
bit-faithfully vs the reference at a tiny config, then real dims. **Spawn adversarial reviewers** vs the
reference after each change. Code PR-compatible with `vllm-project/tpu-inference` (Googler-maintained AGENTS.md —
no pure code-agent PRs; **owner submits**).

## COST (the bill is watched)
Pod `db-v4-64-od`, us-central2-b, 32 v4 chips. Weights ONLY in **`gs://driftbench-dsv4-uc` (US-CENTRAL2)** —
NEVER `gs://driftbench-storage` (EU = the bill driver + egress). ONE FP8 copy (no bf16 conversion). **Don't
download to any host/local disk** — stage HF→GCS once (streaming, zero disk;
`~/moe-tpu/scripts/stage_base_to_gcs.py`), then runai_streamer GCS→HBM + on-device FP8→bf16 dequant. **Operate
ONLY inside the 32 v4 chips — no new TPUs.** A us-central2 CPU staging VM is authorized ONLY if host 0 lacks
disk/NIC — delete after. Clean up as you go.

## Benchmarks + PROVENANCE (non-negotiable)
Reproduce the HF-card benchmarks (tractable first: GPQA-Diamond, AIME, an MMLU-family loglikelihood; agentic
SWE-bench/Terminal-Bench later). **Build `bench/results.db` (SQLite) FIRST**, store LITERALLY EVERYTHING: per
run — utc ts, model+revision, harness+fork commits, env/flags, pod; per item — benchmark, item_id, prompt
verbatim, gold, **raw output verbatim**, extracted answer, correct, score, tokens, latency, seed; per summary —
n, metric, card value, signed Δ. **Never run anything untraceable.** Honest nulls; signed Δ vs the card.

## First task
Stage 1 per HANDOFF: fork `glm-5.2-v4` off `dsv4-flash-v4`; stage `zai-org/GLM-5.2-FP8` → us-central2; register
`GlmMoeDsaForCausalLM`; dense-MLA correctness on the pod; provenance DB + first benchmark. Commit+push both repos
(no force-push). Owner-gated (draft only): upstream PRs, spend beyond the staging VM.
