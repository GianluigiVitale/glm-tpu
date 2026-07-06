# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-06 — **Repo initialized. Nothing ported yet.** This is the starting point for the GLM-5.2
port; the actual porting begins in the next chat with Stage 1.

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `PLAN.md` (phases + thresholds) →
`docs/00-feasibility-memo.md` (the config-verified GO study). Then read the **DSV4 base** you are building on:
`~/moe-tpu/CLAUDE.md`, `~/moe-tpu/HANDOFF.md`, and `~/moe-tpu/docs/{09,11,12,13,15,16}` + the fork branch
`~/tpu-inference@dsv4-flash-v4` (the CSA kernel + head-shard + v4 enablement you adapt). **Don't reinvent —
CLAUDE.md §"What transfers" maps every reusable piece to where it lives.**

---

## State

- `glm-tpu` scaffolded + on GitHub (`GianluigiVitale/glm-tpu`, private, branch `main`). `setup.sh --folder=glm-tpu`
  is wired (`~/setup.sh`). `~/glm-tpu/.env` has the HF token (gitignored — never commit it).
- The pod `db-v4-64-od` (us-central2-b, 32 v4 chips) was **emptied for GLM** (Gemma/ASPt server stopped, HF
  caches cleared, DSV4 cluster down). The DSV4 keepers are safe in `gs://driftbench-dsv4-uc/models/`.
- **No GLM code exists yet.** No fork GLM branch yet. No weights staged yet.
- **The benchmark + provenance machinery is PRE-BUILT** (`bench/`, CPU-tested): the SQLite provenance DB
  (`bench/provenance.py` → `results.db`) that stores every question/timestamp/verbatim-reply/pass-fail + full
  run provenance, the benchmark registry + extractors/scorers, and a pluggable harness (`run_bench.py`, wire
  `make_generate()` to the engine in Stage 1). Datasets cached (MMLU-Pro, GSM8K, AIME-2026); **GPQA-Diamond is
  gated — [OWNER] request access** (see `bench/README.md`). So Stage-1 benchmarking is data-ready + traceable
  from run #1 — do NOT run anything that isn't stored in the DB.

## Target (confirmed)

- **Model:** `zai-org/GLM-5.2-FP8` (FP8-native, ~744 GB — the benchmark checkpoint that fits 1024 GB HBM; the
  BF16 `zai-org/GLM-5.2` is ~1.5 TB and does NOT fit). `GlmMoeDsaForCausalLM`, 753B/40B, 78L, 256+1 experts
  top-8, MLA+DSA (index_topk=2048, 32 indexer heads, IndexShare), 1M context. Full spec in `docs/00`.
- **Hardware:** the 32-chip v4 pod ONLY (no new TPUs). us-central2 for all storage (cost — see CLAUDE.md).

---

## The exact next task (Stage 1 — dense-MLA correctness)

Do these in order; validate each against a reference; commit + push as you go.

1. **Fork GLM branch.** In `~/tpu-inference`, create a GLM working branch off the DSV4 work
   (`git checkout dsv4-flash-v4 && git checkout -b glm-5.2-v4`). Re-`pip install -e` into `~/vllm-env`.
2. **Stage the FP8 weights → us-central2 (cost-optimal, no local disk, no new VM).** Adapt
   `~/moe-tpu/scripts/stage_base_to_gcs.py`: stream `zai-org/GLM-5.2-FP8` HF → `gs://driftbench-dsv4-uc/models/
   GLM-5.2-FP8/` via `gcloud storage cp -` (resumable, skip-if-present, size-verify). Use the token from
   `~/glm-tpu/.env` (`HF_TOKEN`). **Do NOT** download to local
   disk or per-host; **do NOT** put it in the EU `driftbench-storage` bucket. Only create a us-central2 helper VM
   if host 0 genuinely lacks the disk/NIC (it should not) — and delete it after.
3. **Register `GlmMoeDsaForCausalLM`** through the vLLM/torchax path (port PR #2324's registry entry +
   `_PP_DISABLED_MODELS`; see `docs/00` §2). Run **dense MLA** (DSA bypassed) first.
4. **Port the v4 MLA workarounds** (FP8→bf16 tile dequant in `kernels/mla/v2/kernel.py`, v4 block sizes,
   fp32 softmax) — mostly done on the DSV4 branch; re-verify against GLM's MLA dims (qk_head_dim 256,
   v_head_dim 256, kv_lora_rank 512, q_lora_rank 2048).
5. **Load + correctness on the pod.** runai_streamer GCS→HBM + on-device FP8→bf16 dequant; head-shard for HBM
   fit (`docs/12` reused); a prefill MC gate (like DSV4's 6/6) then a first benchmark.
6. **Provenance DB from run #1.** Build `bench/results.db` (schema in CLAUDE.md §Benchmarks) and score the
   first benchmark (GPQA-Diamond or an MMLU-family loglikelihood) storing every question/answer/timestamp.
   **Threshold to proceed to Stage 2:** within ~1–2 pts of the GPU/SGLang reference at ≤8K context.

Stages 2 (the DSA kernel — the critical path) and 3 (IndexShare + MTP) are in `PLAN.md`.

## Landmines carried over from DSV4 (expect these)

- Flaky FP8-dequant-during-load crash (~60%, environmental) → retry-loop / relaunch.
- Multi-host serving-time recompile race under the hybrid (model>1) mesh (`~/moe-tpu/docs/15`) — will re-appear;
  the `ExecutableCompileObserver` + drive-real-layout fixes are the template.
- `GLM_*` env vars must be baked into the raylet env (Ray drops non-`VLLM_*`).
- Relaunch after any pod crash before re-running.

## Owner-gated (draft, don't do)

Upstream PRs to `vllm-project/tpu-inference` (owner submits); any spend beyond the authorized us-central2 staging
VM; force-push; external comms.
