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

## 2026-07-07 — docs/05: KV-scaling design (fixing the 32x-replicated MLA latent cache) — READ-ONLY

- Written `docs/05-kv-scaling-design.md` (no TPU touched). Problem: MLA cache spec `P(BATCH)` with BATCH
  product 1 on the pure-TP mesh → full latent cache on every chip (~97.5 KiB/token padded / 87.8 unpadded;
  128 blocks × 512 = 65,536 tokens ≈ 6 GiB/chip) → max_seqs 8 @4K, 128K ctx impossible (11.9 GiB/seq).
- KEY FINDING: upstream commit `f940073e` (#2398) already landed DCP (`--decode-context-parallel-size`) as
  the sanctioned mechanism — `dcp` mesh axis, cache `P(BATCH, CONTEXT)` striped over the in-page token dim,
  scheduler `block_size *= dcp`, weight shardings invariant (every axis tuple includes 'dcp') — **but the
  MLA attention path was never finished**: `mla_attention` cache specs are still `P(BATCH)`, so dcp>1 today
  would all-gather every layer's cache over dcp each step (~17 GiB ICI/step at dcp=4). Storage-only support.
- Options evaluated: (A) head-sharded attention specs (kills 32x redundant FLOPs + q all-gather; no kernel
  change; no memory win); (B) page-sharded cache — REJECTED (same combine machinery as DCP + table/ownership
  complexity DCP avoids); (C) finish DCP for MLA = per-shard kernel + LSE output + XLA softmax-combine over
  'dcp' + strided position walk + owner-shard new-KV write routing; (D) transpose layout (+11%, env-only) and
  fp8 KV (x2, gated on PR #2324's NaN-under-EP + streaming-loader conflict).
- Recommended sequence: S0 transpose → S1 option A → S2 DCP=4→8 (128K×4 seqs, passkey ladder runnable) →
  S3 fp8 KV → Stage-2 DSA composes with DCP as a pure storage layer (local gather + 2.5 MiB segment
  all-gather; no distributed softmax on the sparse path). 1M endpoint: dcp≥16 + fp8 (2.9 GiB/chip/seq).
- Validation plans per step in the doc (dcp=1 byte-identity, new `parity/mla_dcp_parity.py`, two-step engine
  parity, zero-serving-compile re-enumeration, pod 3/3 + longctx ladder). No code changed this session.

## 2026-07-07 — docs/04: per-decode-step HOST-path audit (READ-ONLY; no TPU touched)
- New `docs/04-decode-hostpath-audit.md`: one decode step traced end-to-end through EngineCore.step →
  RayDistributedExecutor (compiled Ray DAG **forced on**, channel shm, built lazily on step 1) →
  RayWorkerWrapper.execute_model_ray → tpu_runner (prepare/upload/jit-dispatch) → sample →
  the async result protocol (DAG returns an **int result_id**; a second classic actor-RPC round
  `get_execute_model_output` fetches the real output, and its `jax.device_get(next_tokens)` is THE
  per-step blocking sync). Key findings: async scheduling defaults ON (batch queue depth 2) ⇒ exposed
  host floor ≈ **2–7 ms/step** multi-host (sync mode ≈ 8–20 ms); per-step payloads are tiny (SchedulerOutput
  0.9 KB @ 8 reqs, ModelRunnerOutput 0.6 KB — measured by pickling on CPU with the tree's vllm); worker
  H2D is ~5–9 small device_puts ≈ 10 KB (blob-packed already); the vllm-impl `state_leaves` is the raw
  params dict ⇒ per-call pytree flatten on model_fn+compute_logits (est. 0.5–3 ms/call — measure first).
  vs DSV4's 61.2 ms/step flat-in-ctx floor: host was ~3–11% then, <1% of GLM's current 1,053 ms/step —
  but returns to 3–30% after Stage-2 device wins. Top-3 fixes: (1) `enable_continue_decode`+`max_decode_steps`
  (already in the fork; amortizes BOTH Ray rounds over ≤10 on-device steps, needs async off, decode-only);
  (2) protect the async pipeline (assert the "Asynchronous scheduling is enabled" log line; zero
  "does nothing" cycles — DSV4 burned 5,189); (3) piggyback step-N output on step-N+1's DAG round +
  pre-flatten the params dict + fold positions into the blob. §5 lists what the 20-step profiler trace
  must confirm (the inter-step bubble, 0 backend compiles, instant device_get after copy_to_host_async).

## 2026-07-07 — docs/03: throughput roofline + suspect ranking (READ-ONLY; the FLOP/roofline side docs/04 defers to)

- **`docs/03-throughput-analysis.md` committed.** Baselines re-derived from the logs: single-stream
  1.087 s/step (0.92 tok/s), batch-8 ≈1.37 s/step (5.85 tok/s aggregate, gsm8k_n32 pass 1); decode steps
  carry 7–74 real tokens into the single 512-token compiled program (`compile_ranges_endpoints=[512]`,
  crash-dump `total_num_scheduled_tokens=7`). Roofline floors on v4-64: single-stream ≈4.7–7 ms/step,
  batch-8 ≈9–12 ms, pessimal read-everything 19.2 ms → we sit **155–230× / ~120× / ~71×** above (not
  the guessed 1000×). Suspects: (a) TPU_MIN_TOKEN_BUCKET=512 CONFIRMED #1 (real constraint is
  divisibility by the 32-wide token shard → **bucket 32 is legal**; the ">50% padding garbage" claim
  is nowhere in code/recon docs); (d) CONFIRMED sleeper — ~504 identical pad rows land in the SAME 8
  experts every layer (~38 GFLOP per owned expert per layer on 1–4 chips, psum-serialized), while
  gmm_v2 skips empty groups so the weight READ is only ~F2 (~9–15 ms, NOT all-256); (b) the mla
  cross-shard gather moves ~3 GB/step/chip at bucket 512 (plus a GSPMD head→token reshard both ways —
  q is BORN head-sharded from the W_UK_T einsum) but the kernel grid visits only real tokens at decode;
  (c) host 2–7 ms (docs/04), (e) logits/sampling ~2–5 ms, (f) zero mid-run recompiles — all ruled down.
  Static budget accounts for ~135–380 ms of 1,370 → honest ×3–8 residual → **probes before fixes**:
  Probe A bucket-32 A/B (exact launcher+run_bench commands in §6), Probe B 20-step PHASED_PROFILING_DIR
  xplane capture, Probe C FORCE_MOE_RANDOM_ROUTING. Fix ladder: bucket 32 (~10–30×, env-only) →
  max_seqs 16 (~2×, KV already covers 16×4096 exactly) → continue_decode (docs/04) → head-sharded
  attention specs (docs/05 S1) → Stage-2 DSA+DCP. Also flagged: both gsm8k runs died on pass 2 with a
  device-fatal TPU_EXECUTE_ERROR (reliability item, separate from throughput).

## 2026-07-07 — bench: card-protocol fidelity (--protocol card|greedy) + docs/07 (CPU-only; no engine runs)

- **`docs/07-card-protocol-fidelity.md` committed** — per-benchmark map of the HF card's ACTUAL protocol
  (README footnote 1, quoted verbatim): reasoning tasks = `temperature=1.0, top_p=0.95`, max gen 163,840;
  AIME/HMMT/IMOAnswerBench additionally get the exact `Explanation:/Exact Answer:/Confidence:` SYSTEM
  prompt and a GPT-5.5 (medium) judge; **GPQA gets NO benchmark-specific protocol** (no prompt format, no
  judge — the Exact-Answer prompt is explicitly scoped to AIME/HMMT/IMOAnswerBench); **no k/averaging, no
  seeds, no effort level are published for any reasoning task**. Comparability caveats for any published Δ
  are in docs/07 §4 (judge substitution, n=30 sampling variance, generation-cap deviations via n_truncated,
  GPQA prompt = harness choice).
- **Implemented `--protocol card|greedy`** (default greedy = pre-protocol harness, byte-identical —
  DB-verified same prompts vs old stub runs, same SamplingParams): card mode sets the card sampling params
  per request, the card system prompt (byte-derived from reference/hf-repo/README.md, test-pinned) with the
  bare-question user turn for AIME, the 163,840 cap (min-ed with window room + optional CLI cap), per-sample
  seeds (`--seed`, sample s = seed+s, recorded per row), and `--samples N` (avg@N, item_id `#sN` rows —
  labeled harness variance knob, NOT a card spec). GPQA card mode = greedy MC machinery + card sampling only
  (documented as such). `extract_exact_answer` + `drop_confidence_lines` added to extract.py (judge
  SUBSTITUTE = exact-match on the card's answer format; fallback chain can never grab the Confidence
  percentage); mmlu_pro/gsm8k refused in card mode (not on the card). Provenance: runs.env_json now records
  protocol/samples/base_seed + per-benchmark card params + verbatim system prompt + card quote + labeled
  substitutes. test_bench.py: +3 test groups (adversarial exact-answer, README byte-pin, card-run
  plumbing/avg@N provenance); all green; `--stub` green in both protocols (gsm8k greedy, aime card
  samples=2, gpqa card). Pod commands for AIME n=30 / GPQA n=198 card mode: docs/07 §6.

## 2026-07-07 — Stage-3 MTP design (docs/08) + glm_mtp contract skeleton (CPU-only; no TPU runs)

- **`docs/08-mtp-design.md` committed** — full Stage-3 design for GLM-5.2 MTP spec decode on the fork.
  Core finding: vLLM maps `glm_moe_dsa → deepseek_mtp` (`DeepSeekMTPModel`) and builds the MTP block
  from the SAME `DeepseekV2DecoderLayer` as the target, so the fork's OOT MLA wrapper, FP8-on-v4
  linears, indexer patch and EP filter apply to the draft for free; tpu-inference already routes
  `method=="mtp"` through Eagle3Proposer (target `full_hidden_states` as the MTP hidden input, own
  draft KV, advancing positions) and `draft_step_fun` has the MTP branch. Checkpoint layer 78 =
  enorm/hnorm/eh_proj/shared_head.norm + FULL attn+indexer+MoE, NO embed/shared_head.head → must be
  shared from target (GPU proposer does this unconditionally). Five gaps: G1 one-line arch resolution
  (`DeepSeekMTPModel` → `_VLLM_PREFERRED_ARCHITECTURES`, else the eagle3 impl-equality check raises);
  G2 shared-weight NAME MAP (lm_head → layers.78.shared_head.head; hard-error on miss; target sharding
  preserved via `_tensor_is_in_cpu` skip; lm_head-probe V3); G3 draft load re-streams 755 GB → filter
  files by index.json weight_map; G4 DSA index-share across draft steps (step-0 stash emitted from the
  jitted draft fn, re-seeded into the wrapper context for static step≥1 traces — GPU's `set_skip_topk`
  toggle can't survive jit); G5 verification items (glm patch engages on draft load, single KV group,
  logits-probe). KVShare = training-side robustness; inference artifacts = index share + acceptance
  4.56→5.47 (≤5 drafts). DSV4's `_disable_ds_v4_mtp_buffer` stub is a torchax in-place-buffer issue in
  DSV4's TARGET forward — structurally absent from the GLM path. Gates: M0 CPU contracts → M1 draft
  parity vs COMPOSED HF reference (HF has NO MTP module — GlmMoeDsaDecoderLayer + transcribed glue) →
  M2 greedy spec-decode ≡ non-spec greedy (exact tokens; tie-flip protocol) → M3 acceptance-length
  (≥~4.5 @ k=5) + provenance-DB recording from fork-aggregated SpecDecodingStats → M4 DSA-mode MTP
  after 2a.2. Effort ≈ 2–3 weeks; dense-MTP (2–4× decode) does not wait for DSA.
- **Skeleton `tpu_inference/models/vllm/glm_mtp/` committed on `glm-5.2-v4`** (contract-level only:
  draft_config.py, shared_weights.py, index_share.py). **M0 self-check PASS** (CPU, vllm-env):
  config surgery → `DeepSeekMTPModel`/n_predict=1 on the real config.json; layer 78 = FULL indexer via
  the beyond-`indexer_types` fall-through (offset 3, freq 4); 39/39 layer-78 keyset keys classified to
  loader routes (11 stacked / 6 expert / 22 direct); expected-shared set exactly {embed_tokens,
  shared_head.head}; rewrite transcription == installed vLLM's `_rewrite_spec_layer_name` on all keys.

## 2026-07-07 — Observability-first instrumentation: flight recorder + crash triage + stats knob (CPU-only)

- **Blindness being closed** (docs/suggestions.md philosophy: instrument BEFORE debugging): pod runs
  die with roaming single-host fatal `Error Interrupt`/core-halts (probeA2: .15+.17, A3: .15, A5: .20,
  A6: .15) and nothing recorded WHAT each worker was executing at death, nor per-step throughput
  (tqdm only).
- **Flight recorder (fork, worktree branch `glm-5.2-v4-obs` off local `glm-5.2-v4`** — NOTE:
  `origin/glm-5.2-v4` did not exist on the remote; based off the local branch head 183f18ce and pushed):
  new `tpu_inference/runner/flight_recorder.py` + 5 minimal hooks in `tpu_runner.py`
  (`__init__`/`execute_model` post-hook/`load_model`/`capture_model`/`_execute_continue_decode` tail).
  Gated `GLM_FLIGHT_RECORDER=1` (default off = None recorder = zero behavioral change). One JSON line
  per serving step to `/tmp/glm_flight_<host>_<pid>.jsonl`: ts, step, num_reqs, real/padded tokens,
  decode_only, prefill chunks, finished, req-id-set hash (xor-crc32), kv_len min/max, padded_num_reqs;
  lifecycle events (model_loaded, warmup_done, continue_decode summary). Host-side only, outside jit,
  per-line O_APPEND `os.write` (SIGKILL-durable), 50 MB rotation keep-2, fail-open after 3 errors.
  **Measured 15 µs/step** (16-req decode batch, CPU) vs the <100 µs target. CPU tests:
  `tests/runner/test_flight_recorder.py` (11 pass: gate-off writes nothing, field values, hash
  stability, rotation, fail-open, cost smoke).
- **Triage tool (harness): `scripts/triage_crash.sh <run_log> [--no-fetch]`** — shellcheck-clean;
  extracts halted-IP(s)+first fatal ts, deduped error lines, per-host last step+composition from
  `[OBSERVE_COMPILES]` (Ray-dedup caveat documented), jit names near the fatal window, RESOURCE/ICI
  dedup, then fetches `tail -100` of the flight files from all 8 hosts (gcloud --worker=all) and
  aligns last steps (newest file per host wins; earliest-stopping host = diverged). **Validated on
  probeA5.log**: halted 192.168.0.20 first fatal E0707 11:01:39.201727; log-side last step 233 vs
  cluster max 386 (delta −153). Fetch path tested against a stubbed gcloud (diverged host flagged,
  stale files ignored, no-file host reported).
- **Throughput knob (harness `bench/engine.py`): `GLM_LOG_STATS=1`** → passes
  `disable_log_stats=False` into `LLM(...)` (offline LLM force-defaults it True — verified in
  installed vLLM) → 10 s tok/s + running/waiting. Default unchanged.
- **Launcher**: `GLM_FLIGHT_RECORDER=${GLM_FLIGHT_RECORDER:-0}` baked into the raylet ENVS
  (worker-side read; the vLLM Ray executor does not forward custom envs).
- **Doc: `docs/10-observability.md`** — instrument inventory (recorder, triage, stats,
  DSV4_OBSERVE_COMPILES, vLLM dump_input) + run-books (crash → triage; next run →
  GLM_FLIGHT_RECORDER=1 + GLM_LOG_STATS=1) + limits (async dispatch semantics: the signal is the
  silence after the last line, not the line itself).

## 2026-07-07 — Round-5 review fixes applied: Stage-2 DSA kernels + S1 + 2a path (branch `glm-5.2-v4-r5fix`; CPU-only, TPU untouched)

Applied the round-5 adversarial-review findings (docs/reviews/round5-agent*.md: indexer-kernel, sparse-MLA,
dsa-path+S1) in a dedicated worktree, branch **`glm-5.2-v4-r5fix`** off `glm-5.2-v4` (main checkout never
touched — live pod imports it). One commit: **`c8a51543`** (9 files, +598/−29; NOT pushed per task).

**Fixes applied (fork):**
- **[dsa-path 1, CRITICAL] fp8-resident `indexer.wq_b` read without scales** → new
  `_linear_weight_f32` adapter in `glm_dsa_indexer.py`: dequantizes fp8 codes with their separate
  `weight_scale` (per-tensor / axis-0-block / 2-D-block / kernel-formatted scales, via `dequantize_tensor`
  ceil-block expansion); loud `ValueError` on fp8-without-scale. GLM-5.2-FP8 quantizes wq_b (not in
  `modules_to_not_convert`) and gate D0 (topk ≥ T) is structurally blind to garbage scores. + 4-case fp8 test.
- **[dsa-path 2, HIGH] `xla_ref` silently wrong off single-prefill** → runtime guard
  (`jax.debug.callback`, positions == arange(T)) inside `glm_dsa_xla_ref_attention` refuses decode / chunked /
  batched-multi-sequence use with a loud error; docstrings now say SINGLE-SEQUENCE single-chunk. + 3-case test.
- **[dsa-path 3, MED] S1 head-sharded unsafe on DP meshes** → `mla_attention` raises
  `NotImplementedError` when GLM_MLA_HEAD_SHARDED=1 on a mesh with data/attn_dp product > 1. + test.
- **[dsa-path 5, MED] "born head-sharded" is a propagation hope** → belt-and-braces
  `with_sharding_constraint(q_NTA, P(ATTN_HEAD, None, None))` before the shard_map (no-op if GSPMD already
  chose it); on-TPU HLO collective-count check still on the checklist.
- **[dsa-path 6, LOW] "XLA folds the adapter" comment wrong at runtime** → honest comment (casts execute
  every step, ~0.75 GB/step at fp32 over 22 full layers; production must precompute in PWAL). Ray env-forward
  warning added at the GLM_MLA_HEAD_SHARDED read (cheap half of finding 7).
- **[indexer 1, MED] mixed-dtype q/k contract** → `assert q.dtype == kv_cache.dtype` in both scorers
  (shared `_check_scoring_shapes`) + explicit `q.astype(kv_cache.dtype)` at the `dsa_topk_indexer` boundary
  + mixed-dtype test (direct scorer calls raise; wrapper cast bit-equals pre-cast on both paths).
- **[indexer 3f] `page_size % 128` guard** when `interpret=False`; **[indexer 5] `n_valid` clamped** to
  `min(topk, S)` (kv_lens-contract-violation hardening) + tests. **[indexer 4, doc]** pad-page-id contract
  tightened to "constant/repeated id" (DMA-elision); signed-zero top-k note; on-TPU checklist extended
  (1-sublane matmul LHS risk + SMEM table scaling).
- **[sparse-MLA F1, MED] false "degrades to output 0" claim** → made TRUE instead of reworded:
  `_finalize` (and the XLA oracle, keeping Gate K meaningful) now explicitly zeroes fully-masked rows
  (`where(m > _MASK_VALUE, out, 0)`) — exact-identity for any live row; a seg_valid==0 padded slot now
  yields exactly 0 even with NaN/inf garbage in the gathered segment. + 2 tests (direct + gather composition).
- **[sparse-MLA F3, LOW] effective `seg_block` 128-multiple assert** after the `min(seg_block, seg_len)`
  bypass, gated on `interpret=False`. **[indexer 6 / glue]** "decode always selects the self token"
  justification corrected (real guarantee: n_valid ≥ 1 ⇐ kv_len ≥ 1); seg_valid==0 padding-row
  discard note added to the gather/kernel contracts.
- **[sparse-MLA F4c/R5, doc]** `MLA_TRANSPOSE_KV_CACHE` incompatibility + reshape-is-a-bitcast condition +
  index-clamp/-1-interleave robustness limits documented in `gather_kv_segment`; Gate-K bars flagged as
  interpreter measurements (re-measure on real MXU before upstreaming).
- **[sparse-MLA R1] packed-cache layout discriminating test ported into the suite**:
  `test_gather_layout_vs_upstream_v1_writer` writes via the UPSTREAM mla/v1 reference writer (independent
  (row,col) convention; the v2 Pallas fused writer is pinned bit-exact to it on TPU) and asserts the port's
  row-major reshape + gather round-trips token-identifiable payloads at kv_packing ∈ {2,4,32} ×
  page_size ∈ {8,32,64}.

**Doc fixes (this repo, docs/01-dsa-kernel-design.md):**
- **Gate S amended (indexer finding 2)**: two tiers — **S1 (fp32 algorithmic): 0 mismatches modulo exact
  ties — ACHIEVED**; **S2 (production bf16): boundary-band criterion** (every mismatch within
  |s − s_kth| ≤ ε, ε ≈ 2⁻⁸ relative — bf16 churn measured 1–2 non-tie indices/2048 typical, ~2% adversarial;
  "0 modulo ties" vs fp32 is unachievable by construction). Tests' `_assert_topk_set_equiv` = the S2 form.
- **§3.1 write-then-attend bullet CORRECTED (sparse-MLA F2)**: mla.v2 does NOT write-then-attend (write is
  fused in its own pallas_call; the read-only kernel.py:319-320 contract is DSV4's) → Stage-3 must scatter
  the step's latents BEFORE `gather_kv_segment`. §3.1 also now states indices are descending-score,
  not position-sorted.

**Skipped (with reasons):** [dsa-path 4] mla.v2 `_INTERPRET` shim prefill bug — upstream kernel, needs
dedicated debugging + on-TPU A/B (prefill at H_local=2 has no off-TPU evidence; keep GLM_MLA_HEAD_SHARDED
off until the on-TPU gate); [dsa-path 7 full fix] env plumbing through vllm config — PR-shaping work, doesn't
change the operative mitigation (launcher bakes GLM_* into the raylet env); [indexer 3a-e, 4-perf] on-TPU-only
verifications (documented in the kernel's real-TPU checklist); [sparse-MLA F4a/b runtime asserts] inputs are
jit tracers — documented as contracts instead.

**Tests (CPU, vllm-env, JAX_PLATFORMS=cpu):** the five DSA/MLA files **93/93 PASS** (incl. +12 new cases):
`test_dsa_indexer_kernel` 24, `test_dsa_sparse_mla` 42, `test_glm_dsa_indexer` 17, `test_mla_attention` 5,
`test_mla_head_sharded` 5. Full `tests/kernels/` sweep as tasked: 1238 failed / 137 passed / 870 skipped in
33 min — the failures are the PRE-EXISTING CPU baseline (TPU-only Pallas kernel tests — spmm, transpose,
RPA, GMM, … — that compile-fail off-TPU; sampled failures reproduce identically at pristine HEAD via a
stash round-trip; no DSA/MLA file among them; the diff imports nothing they use).
Commit: fork `glm-5.2-v4-r5fix` @ c8a51543 (not pushed, per task); glm-tpu docs committed + pushed.

## 2026-07-07 12:40 UTC — CORE-HALT ROOT CAUSE CLOSED: JAX_SHARE_BINARY_BETWEEN_HOSTS=1; GSM8K waveA acc 93.75

- **The batched-serving fatal Error-Interrupt/core-halt class is FIXED by `JAX_SHARE_BINARY_BETWEEN_HOSTS=1`.**
  Evidence: 7 consecutive batched runs died (bucket 512 AND 32/64, async on AND off, admissions or not);
  the flight-recorder triage of probeA5 showed the halted host **153 steps behind** the cluster (per-host
  independent compilation → binary/latency skew → collective desync → ICI fatal). Wave A2, differing from
  crashed wave A only by sharedbin+recorder, ran CLEAN to completion. The forensic finish+page-edge
  correlation was the *symptom locus* (steps where program composition shifts), not the cause.
- **GSM8K wave A (items 0-15, n=16): acc 93.75**, 2 misses = truncations at the 1024-token cap
  (run recorded with full provenance; batched 16-way, 8.8 tok/s aggregate incl. tail).
- Launcher default flipped to JAX_SHARE_BINARY_BETWEEN_HOSTS=1 (override via env).
- Wave B (items 16-31) running at TPU_MIN_TOKEN_BUCKET=32 — tests whether sharedbin also fixes the
  small-bucket runs (expected same root cause) AND unlocks the 10-30x decode throughput (docs/03).
