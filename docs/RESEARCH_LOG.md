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

## 2026-07-07 — Long-context passkey/NIAH harness ported from DSV4 (Stage-2 threshold instrument; CPU-validated)

- **`bench/glm_longctx.py`** — port of `~/moe-tpu/bench/dsv4_longctx.py` with ONE protocol change:
  **generation-based retrieval** (greedy decode ≤20 tok + exact-match of a 6-digit passkey) instead of DSV4's
  loglikelihood-candidate scoring (that was for a Base model on a prefill-only validated path; GLM-5.2 has a
  working decode path). **Prompt style = RAW COMPLETION, not the chat template** (documented in the module
  docstring): the GLM template ends `<|assistant|><think>` → a ≤20-tok budget would be all thinking preamble;
  the cue "...The secret passcode is" invites the direct continuation (the canonical Mohtashami & Jaggi
  protocol). Tokenized `add_special_tokens=True` ([gMASK]<sop>), stops at the GLM EOS ids
  [154820, 154827, 154829].
- Filler is token-counted with the REAL tokenizer at SENTENCE granularity (~4-8 tok/unit) → the needle's token
  position lands within a fraction of a percent of the requested depth; `--lengths` accepts up to **1M**
  (=1048576, max_position_embeddings) with k/K / m/M suffixes; 1M-token construction takes 0.21 s. CLI mirrors
  DSV4 (`--lengths --depths --trials`); Stage-2 gate (≥95% per length to ≥128K) printed per length.
- **Provenance**: every trial stored via `record_item` under `benchmark="passkey_L{L}_d{d}"` (verbatim prompt up
  to `--prompt-chars-cap`, above that head+tail+sha256 + the seed for deterministic reconstruction; raw output
  ALWAYS verbatim) + per-cell `finalize` rows + an aggregate `longctx_passkey` summary row (new CARD_TARGETS
  'passkey' kind).
- **`bench/engine.py`** — the Stage-1 `LLM(...)` recipe factored out of `run_bench.make_generate` (pure TP×EP /
  no DP attention, runai_streamer, kv auto, DISABLE_WEIGHT_REQUANTIZATION via the launcher raylet env) so
  run_bench + glm_longctx build the IDENTICAL engine. vllm import stays INSIDE `build_llm` — `run_bench --stub`
  re-verified offline (no vllm in sys.modules; stub smoke recorded to the DB; log line byte-identical via
  `log_extra`).
- **CPU tests `bench/test_longctx.py` 9/9** (mock whitespace tokenizer, no model/network/vllm): length targeting
  (L−8 ≤ n ≤ L), **depth placement within ±2% at every ladder length** (measured 0.7500 vs 0.75 at 128K/1M),
  monotonic + endpoints, needle-exactly-once + raw-completion prompt shape, determinism (seed→identical trial),
  extractor (comma/think-block/no-truncation/never-guess), `parse_lengths` (1M accepted, 2M rejected), an
  end-to-end `--stub` pipeline test asserting all rows land in a temp provenance DB, and the prompt-storage cap.
  `test_bench.py` still green. NOT run on the engine/TPU (pod run is a next-session task, after Stage-1 serving
  is up).

## 2026-07-07 — Stage-2a (first slice): XLA-reference DSA indexer + RoPE-layout verdict = INTERLEAVED (E1+E2 on real weights)

- **`parity/glm_indexer_reference.py`** — pure-jnp, layout-parameterized transcription of the indexer forward
  (docs/01 §1.1): `indexer_scores(..., *, interleaved)` + exact `topk_indices` (`lax.top_k`; `approx_max_k`
  banned per §1.4). Math line-cited against BOTH refs (HF modeling.py:166-262; vLLM deepseek_v2.py:678-742).
  Hadamard+fp8 skipped per the HF-documented equivalence (modeling.py:211-215); scale pre-relu like HF
  (fold-into-w is the 2b Pallas form, §1.1). fp32 throughout.
- **`parity/test_indexer_reference.py`** — machine-gated CPU test (JAX_PLATFORMS=cpu forced pre-import),
  **ALL PASS, exit 0**: (a) score parity vs a line-cited torch transcription using HF's own rope functions
  (installed transformers 5.12 module asserted byte-identical to reference/), T=64 real dims, BOTH layouts
  same-side-same-layout: max|Δ| 6.1e-6 / 6.3e-6 ≤ 1e-5 fp32; cross-layout max|Δ| 2.69 (distinguishable);
  (b) selected-set equality vs torch.topk k∈{8,16,64} modulo boundary tie-groups + vs the actual
  `GlmMoeDsaIndexer.forward`; (c) determinism (2× eager + jit identical) + engineered-tie semantics
  (62 tie rows; value-multisets exact).
- **`parity/glm_indexer_rope_experiment.py`** — the §1.2 offline discriminator on REAL GLM-5.2-FP8 weights
  (layer 0 full-indexer layer from `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` shard 1, fp8 block-128
  dequant via weight_scale_inv; h = input_layernorm(embed), q_resid = q_a_layernorm(q_a_proj(h));
  ground truth = the same layer's dense MLA attention row-mass per key from kv_b_proj, main rope
  interleaved/uncontested). 1024 tokens natural text ([gMASK]<sop> prefix).
  - **E1 (T=1024): INTERLEAVED wins all 5 metrics** — recall@64 **0.898 vs 0.680**, mass@64 0.475 vs 0.436,
    recall@256 0.965 vs 0.897, mass@256 0.793 vs 0.781, Spearman **0.984 vs 0.928**. Inter-layout top-64
    overlap 0.674 (the layouts genuinely differ). Robustness rerun T=512: interleaved wins all 5 again
    (recall@64 0.948 vs 0.834, Spearman 0.984 vs 0.940).
  - **E2 (corroborating)**: within-pair |log2 norm-ratio| far lower under interleaved pairing (wk 0.31 vs
    0.70; wq_b 0.19 vs 0.79) — rope-pair norm matching exists only under the (2i,2i+1) hypothesis.
  - **VERDICT: interleaved** — matches the §1.2 prior (vLLM honoring `indexer_rope_interleave: true`);
    HF's non-interleaved `apply_rotary_pos_emb` call at modeling.py:239 is a DSV3.2 copy-paste. Per §1.2,
    2a hard-codes interleaved (comment citing docs/01 §1.2) with non-interleaved reachable only behind a
    parity-harness debug flag.
  - **Residual uncertainty**: E3 (behavioral: passkey + logprob divergence-from-dense at ctx 4K-16K under
    both layouts, + gate-S TPU-vs-torch selected sets) still required on the pod once the 2a sparse path
    exists — E1/E2 are score/weight-level, single-layer (0), and cannot see output-level effects at
    ctx > 2048. Layers 1-2 (other shards) can be added to E1 if more evidence is wanted.

## 2026-07-07 (later) — the "T>128 multi-page divergence" was a HARNESS false alarm; engine exonerated; ALL sub-cube gates green

- **Root cause (found by a read-only kernel-analysis agent, empirically confirmed)**: `make_mini_config`'s
  `index_topk` default bound the module-import value (mini 128); `use_real_dims()` rewrites the global but not
  the already-bound default → the written checkpoints carried `index_topk: 128` under `--real-dims` → the HF
  REFERENCE ran top-128 SPARSE DSA past position 128 while the engine ran dense. The 128 cliff == index_topk
  (== PAGE by coincidence). Explains everything: block-size independence, identical outputs across kernel
  configs, engine paths agreeing with each other. **The mla.v2 multi-page path was verified correct by hand**
  (write/attend/mask/page-walk all traced clean; PR #2324's kernel patch touches none of it).
- Fix: late-bound default + a written-artifact assert (`cfg["index_topk"] >= T`). T=136/256 real-dims fp8 now
  **PASS** (top-1 vs fp32 0.89/0.91 ≥ the bf16 ref's own 0.88/0.89).
- **Two-step gate corrected**: bit-exactness across different chunkings is not a sound bar at real dims — a
  borderline MoE routing decision legitimately flips under a different summation order (scattered positions
  from pos 3; BOTH paths equally close to HF). Gate is now two_step-vs-HF ≤ 1.5× bf16 floor. T=300 real-dims
  two-step PASS; mini T=32 remains bit-exact.
- Round-3 adversarial review (4 lenses) on the parallel-agent deliverables: 2 HIGHs in the passkey harness
  (token-length targeting −17.2% under the real BPE tokenizer; the `[gMASK]<sop>` add_special_tokens claim is
  factually wrong), extractor silent-wrong regressions, dataset revision-pinning, doc-truthfulness items —
  fix batch delegated; none pod-blocking. Reports in docs/reviews/round3-*.
- **Sub-cube validation is COMPLETE. Next: pod bring-up** (sync workers @ fork a429be54, launch, engine build).

## 2026-07-07 (round-3 fix batch) — review findings fixed; CORRECTION to the long-context-harness entry above

- **CORRECTION (appended, history not rewritten):** the 2026-07-07 long-context-harness entry above claims the
  raw prompt is "Tokenized `add_special_tokens=True` ([gMASK]<sop>)". That was **factually wrong**: the GLM-5.2
  tokenizer's post_processor is plain ByteLevel — `add_special_tokens=True` adds NOTHING (verified on the real
  tokenizer.json; review round3-unknown finding 2), so the harness was sending a bare BPE stream with no special
  tokens at all. Fixed in `bench/glm_longctx.py` by prepending the ids **explicitly**
  (`PROMPT_PREFIX_IDS = [154822 [gMASK], 154824 <sop>]`); the module docstring now documents the no-op.
- **Second correction to the same entry:** "the needle's token position lands within a fraction of a percent" was
  true, but the LENGTH claim was not — per-sentence token counts are not additive under BPE (boundary merges), so
  the "128K" cell really tested **~108K tokens (−17.2%)**. `build_trial` now targets the CONCATENATED prompt
  iteratively (Newton on the measured effective rate, 2-3 whole-prompt tokenizations): real-tokenizer check
  L=131072 → 130,418 (99.50%), L=1,048,288 → 1,043,047 (99.50%), depth 0.5000 exact, 1M build 5.0 s. New CPU test
  uses a boundary-MERGING mock tokenizer (old code measures 79.2% under it → the old bug is now caught).
- **Round-3 fix batch** (reports in `docs/reviews/round3-*`; commits this session, none pod-blocking):
  - longctx: `<|assistant|>` (154828) added to the RAW-protocol stop ids (a chat-token emission burned the
    20-token budget); `--protocol {raw,chat}` — chat = the model's own template with `enable_thinking=False`
    (renders `…<|assistant|><think></think>`, verified on the real tokenizer; 32-token budget) as the documented
    fallback if raw completion proves unreliable on the pod.
  - longctx: `--lengths` capped at **1048288** (= max_position_embeddings 1048576 − 256 max_len headroom − 32
    answer budget) and `max_len = min(max(lengths)+256, 1048576)` — a bare `1M` target used to crash at engine
    build (derived max_model_len 1048832 > 1M).
  - extract.py: the last `\boxed{}` and the last "final answer is X"/"option X" phrase now compete BY POSITION
    (a boxed candidate revised later in prose was silently kept before); thousands-commas stripped only when the
    ENTIRE token is a plain grouped number — `(1,200)` / `0,001` no longer rewritten (silent-wrong fixes, tests
    cover both flip directions).
  - benchmarks.py: dataset revisions PINNED (HfApi shas fetched 2026-07-07: gpqa `633f5ee8…`, mmlu-pro
    `b189ec76…`, gsm8k `740312ad…`, aime_2026 `d2de22f3…`) — recorded per item (meta) + per run (env);
    GPQA per-item shuffle seed now = sha256(question) content hash, row-order independent — **gold letters
    CHANGE vs the previous idx-seeded scheme** (acceptable: no real runs were recorded under it; pinned canary
    added to the tests).
  - docs: 02-pr-series hunk map regrouped (`9bb2c23e`'s indexer-forward-gate hunk → PR-G1, not G2; `a429be54`
    added to the series incl. the `TPU_MLA_V4_KV_PAGES/QUERIES` debug overrides); the "AGENTS.md" policy
    correctly attributed to **vllm-project/vllm** (`~/vllm-build/AGENTS.md`; tpu-inference has none) with its
    duplicate-work-check + AI-attribution-trailer requirements added to the PR checklist; HANDOFF ENVS list
    corrected to the launcher's 13 envs verbatim and the T>128 item marked RESOLVED.
  - All bench CPU tests green after the batch: `test_bench.py` 5/5 suites, `test_longctx.py` 11/11.

## 2026-07-07 06:40-07:00 UTC — 🎉 STAGE-1 POD MILESTONE: GLM-5.2-FP8 GENERATES CORRECTLY ON 32 v4 CHIPS

- **Engine built in 615.5 s** (run `stage1 pod smoke #16`, ~/glm-run/smoke16.log): 753B FP8 streamed
  GCS→HBM via runai + the sharding-derived EP filter (32/256 experts/host — mesh-aware non-contiguous
  chunks), FP8 kept resident (23.06/30.75 GiB per chip), checkpoint-exact block scales
  (DISABLE_WEIGHT_REQUANTIZATION=1), dense MLA via mla.v2 + cross-shard all-gather, MoE GMM per-tile
  dequant, KV 65,536 tokens (128 blocks × 512), pure TP-32 mesh (model=32).
- **GSM8K smoke n=4: acc 75.0** — 3 correct with reasoned CoT (`18`,`3`,`540`), 1 honest truncation
  miss at the 512-token cap. Config: max_len 4096, mbt 512, max_seqs 4, blocks 128, gmu 0.90. All items
  verbatim in bench/results.db.
- **Pod-bring-up bugs fixed en route** (fork commits 4e24a6c6..10efa393 + launcher/bench commits):
  driver must not import tpu_inference (holds libtpu lockfile against its own EngineCore); leaked
  1.5-day EngineCore held w-5 accel0 (sudo pkill in launcher now); HEAD_IP derived not hardcoded
  (pod re-created since DSV4; w-0 = .21); pkill patterns bracket-escaped (self-match killed the ssh
  carrier); EP expert filter (host CPU-RAM OOM at 376 GB — each host now loads 32/256 experts);
  full-res-N block scales (TP shard width 64 < quant block 128); selective K-axis scale expansion
  (16 K-blocks vs 32 shards); vocab-sharded lm_head must keep vocab-sharded logits in BOTH
  compute_logits variants (the forced token-sharded layout all-gathered the 1.77 GiB lm_head into
  scratch); get_page_size probes TPU version via metadata not jax.devices() (driver-side TPU init);
  KV blocks capped (auto-sizer overcommitted HBM).
- **Perf status (honest)**: ~0.95 tok/s single-stream XLA decode (unoptimized — the Pallas decode
  path is Stage-2+ work; DSV4 was the same pre-kernel). GPQA-198 needs batched generation first.
- Remaining for the Stage-1 gate: batched bench runs (GSM8K n≥32, GPQA-Diamond) vs card within noise,
  3/3 clean pod runs, adversarial review of the bring-up series.

## 2026-07-07 — bench: BATCHED generation in run_bench.py (the GPQA-198 enabler)

- `make_generate()` now also exposes `generate.generate_batch(prompts) -> [(text, n_gen_tokens), ...]`:
  ONE `llm.generate` call over a list of token-id prompts (same greedy protocol; per-prompt
  SamplingParams differ only in the room clamp, exactly as the sequential path computed it), so vLLM's
  scheduler runs up to `--max-seqs` sequences concurrently (~0.95 tok/s single-stream → ×batch aggregate).
  Outputs are mapped back to items by request order/id (defensive int(request_id) re-sort); over-long
  prompts keep their slot as ("", 0) — recorded empty + scored wrong, same as the sequential SKIP.
- `run_benchmark()` auto-uses it when present (`--batch-size N` chunks; 0 = ALL items in one call);
  `--stub` has no generate_batch and keeps the sequential path byte-for-byte. Per-item provenance is
  unchanged (verbatim prompt/raw output/extracted/correct/n_gen_tokens); per-item latency is NOT
  individually measurable inside a batch, so latency_ms is stored NULL (never faked) and the real batch
  wall time goes into the summary note (`batched:N chunks=C batch_wall_ms=… gen_tok=…`).
- CPU-validated (JAX_PLATFORMS=cpu, no TPU touched — the pod is owned by another session):
  `test_bench.py` + a new `test_batched_run` (fake generate_batch: ordering, per-item provenance,
  chunking [2,1], NULL latency, summary note, sequential fallback) — pytest 17/17 across both suites;
  `--stub gsm8k,gpqa_diamond --limit 2` records 4 items correctly (run 27), vllm never imported.
- KV sizing for GPQA-198 at max_len 8192 (block=512 → 16 blocks/seq): MLA latent KV ≈ (512+64)×78×2 B
  ≈ 87.8 KiB/token per chip (replicated across TP ranks) → 128 blocks = 65,536 KV tokens ≈ 5.4 GiB/chip,
  the proven smoke sizing inside the 30.75−23.06 = 7.69 GiB/chip free; coverage max_seqs ≤ 128/16 = 8.
  260k KV tokens (max_seqs 32) needs ~22 GiB/chip — only possible with a sharded latent cache (Stage-2+).
