# RESEARCH LOG — GLM-5.2 on TPU v4

Append a dated entry every session (newest first). Mirror the DSV4 log (`~/moe-tpu/docs/RESEARCH_LOG.md`):
what you did, what you validated it against, the exact numbers, and the honest nulls.

---

## 2026-07-06 — Repo initialized (starting point; nothing ported yet)
- Created `glm-tpu` as the GLM-5.2 porting harness (the moe-tpu sibling): `CLAUDE.md`, `HANDOFF.md`, `PLAN.md`,
  `docs/00-feasibility-memo.md` (the config-verified GO study), dir scaffold, `.env` gitignored.
- `setup.sh --folder=glm-tpu` wired; repo on GitHub (`GianluigiVitale/glm-tpu`, private).
- Target confirmed: **`zai-org/GLM-5.2-FP8`** (FP8-native, ~744 GB — fits 1024 GB HBM; BF16 ~1.5 TB does not).
  `GlmMoeDsaForCausalLM`, 753B/40B, MLA + DSA (index_topk=2048, 32 indexer heads, IndexShare), 1M context.
- Cost geography established: pod + storage must be **us-central2** (`gs://driftbench-dsv4-uc`); the EU
  `gs://driftbench-storage` bucket was the June bill driver — never put GLM weights there.
- **Benchmark + provenance machinery pre-built (CPU, no model yet)** so Stage 1 starts ready: `bench/`
  (`provenance.py` = the SQLite DB storing every question/timestamp/reply/pass-fail + run-level provenance;
  `benchmarks.py` registry; `extract.py`; `run_bench.py` with a pluggable `make_generate()`; `download_data.py`;
  `test_bench.py` **all CPU tests pass**). Datasets **cached** (13 MB): MMLU-Pro (12032), GSM8K (1319),
  AIME-2026 (30). Pipeline validated end-to-end on real data via `--stub`. **GPQA-Diamond (198) now cached** too (access granted). `CARD_TARGETS` holds the 18 HF-card benchmark targets +
  2 non-card sanity gates (mmlu_pro, gsm8k, value=None).
- NEXT: Stage 1 (see HANDOFF) — fork GLM branch, stage FP8 → us-central2, register the arch, dense-MLA
  correctness, then wire `make_generate()` to the engine and reproduce the first benchmark (data + DB ready).

## 2026-07-07 — Stage 1a+1b(sub-cube): staging DONE, dense-MLA + FP8 parity GREEN (machine-gated), adversarially reviewed

- **Weights staged**: `zai-org/GLM-5.2-FP8` → `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` (150/150 files,
  755.7 GB, size-verified, 0 failures, ~45 min @ ~300 MB/s aggregate). Key-set fingerprint committed
  (`configs/glm-5.2-fp8-keyset.json`): 118,629 params; **`indexers_proj` does NOT exist** (quant-config red
  herring); indexer weights only on full-schedule layers 0,1,2,6..74 + MTP 78.
- **Fork commits** (branch `glm-5.2-v4`, off dsv4-flash-v4 @17d635a1): `9bb2c23e` (registry + DISABLE_DSA_INDEXER
  + GLM construction patches + 2 latent MLA-wrapper bug fixes: identity scales for kv auto, W_UV_scale axis),
  `8ad980e7` (FP8-on-v4: GMM in-VMEM dequant routing gated tpu_generation()==4; ceil scale columns for
  non-block-aligned fused parts; ragged-aware 2D-scale expansion in xla_quantized_matmul), `3f745ddc`
  (review fixes incl. kv_cache_spec indexer-cache skip [engine-boot HIGH], gate default ON, Ray env
  propagation, + 2 REAL gmm_v2 tiling bugs found by the real-dims run: dequant-buffer VMEM modeling + the
  at-floor tile_k-shrink dead branch).
- **Parity harness** (`parity/glm_engine_{common,parity}.py`, machine-gated, exit-code): production stack
  (VllmModelWrapper → vLLM GlmMoeDsa → fork MLA wrapper → mla.v2 Pallas + fused-MoE GMM) vs transformers
  5.12 `GlmMoeDsaForCausalLM` (fp32+bf16 controls), synthetic native-key ckpt (HF loads 0 missing/unexpected).
  1 chip (multi-chip goes through the real runner on the pod — hand-built metadata is single-shard-only).
  - mini bf16 (5L): final_hs 0.344 < floor 0.375, logits 0.106 < 0.127, top-1 1.000 vs bf16 ref → **PASS**
  - mini fp8 block-64 + DISABLE_WEIGHT_REQUANTIZATION=1, HF twin = quant-dequant roundtrip (same effective
    weights): top-1 1.000/1.000, below floor → **PASS**
  - **real-dims fp8 block-128 (3L: H6144, 64h, nope192/rope64/v256, lora 2048/512, idx 32×128)**:
    final_hs 0.619 < floor 0.722, logits 0.910 < 0.958, top-1 vs fp32 0.906 > bf16-ref 0.875 → **PASS**
- **4-lens adversarial review** run on the Stage-1 diff (docs/reviews/stage1-*.md): 3 HIGHs found + fixed
  (engine-boot kv_cache_spec crash; default-on indexer crash path; broken CPU test), numerics lens confirmed
  the NaN root cause + HF dense-equivalence at topk>=T is exact; overclaiming lens drove the machine gate,
  real-dims run, Ray env propagation, and this log entry.
- **Stage-1 serving config settled**: FP8 resident checkpoint-exact (`DISABLE_WEIGHT_REQUANTIZATION=1`),
  dense MLA (indexer gate default-on), kv-cache auto (bf16, identity scales), runai_streamer from the
  us-central2 bucket. Known open items: PR #2324's cross-shard MLA all-gather + v4 MLA block sizes port
  (needed for the pod's TP topology — attention weights cannot replicate at 753B), decode/2-step parity,
  pod 3/3 runs.
