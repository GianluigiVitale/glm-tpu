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

## 2026-07-07 — Stage-3 dense-MTP M1 BUILT + CPU parity PASS (fork branch `glm-5.2-v4-mtp`; CPU-only, TPU untouched)

Implemented docs/08 gaps G1-G3 for DENSE MTP (DSA off; index-share/G4 stays M4) in a worktree
(`~/tpu-inference-mtp`, branch `glm-5.2-v4-mtp` off `origin/glm-5.2-v4`):

- **G1** — `DeepSeekMTPModel` added to `_VLLM_PREFERRED_ARCHITECTURES` (models/common/model_loader.py):
  draft and target now both resolve `"vllm"`; Eagle3Proposer's impl-equality check passes.
- **G2** — shared-weight NAME MAP: `glm_mtp/shared_weights.py` gains `build_mtp_shared_params`
  (draft-name-keyed: embed→embed, layers.78.shared_head.head←lm_head); `Eagle3Proposer.load_model`
  uses it for `method=="mtp"` + `DeepSeekMTPModel` drafts (all other methods keep the old
  name-equality set byte-identically). Wrapper sharing factored into `_apply_shared_params` with
  **must_share_all hard errors** for MTP (shape mismatch / missing draft param / missing target key
  all raise — no more warn-and-skip garbage-logits risk); sharing now re-registers a fresh Parameter
  (works for plain-CPU AND torchax draft params; the shared array keeps the target's sharding).
  V3 closed: `jit_compute_logits_func`'s lm_head probe falls back to the MTP `shared_head.head`
  (`_mtp_shared_head_weight`) so a vocab-sharded shared head keeps vocab-sharded logits (no 1.77 GiB
  lm_head all-gather); target probe path unchanged.
- **G3** — draft-load filters (`glm_mtp/draft_load_filter.py` + RunaiIncrementalModelLoader): FILE-level
  (safetensors index weight_map → only files carrying `model.layers.78.*`; ~1-2 of 150 files instead of
  755.7 GB re-streamed; gs:// index pulled via ObjectStorageModel) + NAME-level iterator skip (EP-filter
  pattern; composes with the EP expert filter — layer 78 carries all 256 experts so coverage verification
  still passes). Both fail OPEN; target loads byte-identical (`_mtp_spec_layers=None`).
- **Spec-config wiring verified E2E on CPU**: `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'`
  → vLLM surgery (arch `DeepSeekMTPModel`, `n_predict=num_nextn_predict_layers=1`) → `use_eagle()` →
  Eagle3Proposer → draft build. Also fixed a CPU-harness-only artifact: `_free_cpu_storage` in
  unquantized.py (JAX CPU backend aliases the torch buffer via `jnp.asarray`; freeing must be best-effort).

**Gate M1 (CPU, mini dims, fp32): PASS** — `parity/glm_mtp_parity.py` (new; `glm_engine_common.py` extended
with MTP-layer weights/config, fp32 checkpoints, eh_proj kept bf16-class under fp8). Candidate = production
draft build through the wrapper's real jitted `draft_step_fun` + `compute_logits` (GLM_DSA_MODE=xla_ref,
index_topk>=T ⇒ bit-exact dense; MTP block dense-MLP — the MoE GMM kernel is TPU-only). Reference = COMPOSED
HF (`GlmMoeDsaDecoderLayer` at the MTP index from a twin config + transcribed enorm/hnorm/eh_proj/
shared-head glue, incl. the position-0 embed mask). Result: **max rel Δ 3.1e-6 (hidden) / 3.2e-6 (logits),
~2000x under the bf16 control floor (6.4e-3); top-1 agreement 100%**. Plus: M0-style LOAD AUDIT on the real
built model (dense AND MoE-MTP minis) — the only params not fed by the checkpoint are exactly
{embed_tokens, shared_head.head}, then hard-error-shared; V1 confirmed (draft indexer built, topk buffer on
CPU). **Byte-identity hash test PASS**: target forward sha256 identical with the MTP speculative config
attached vs speculative off.

CPU suites: mtp `test_glm_mtp_stage3.py` 20/20; DSA `test_glm_dsa_indexer` + kernels 56/56 + suite batch
43/43; MLA 8/9 (`test_process_weights_after_loading` failure is PRE-EXISTING at pristine HEAD — mesh-axis
rename drift in the test, reproduced identically on the untouched main checkout).

M2 (pod, owner): greedy spec-decode == non-spec baseline for k∈{1,5}; needs engine-scale checks of V2
(draft KV grouped with target MLA layers; eagle3 `prepare_inputs` last-group assumption degenerates
correctly for a single unified group), the G3 file filter against the real gs:// index, FP8 draft load
(quantized path skips vLLM weights-tracking), and precompile of the draft programs per bucket.

## 2026-07-07 — Round-6 review fixes applied (CPU-only, TPU untouched) + a CORRECTION to the 12:40 entry

Four round-6 adversarial reports committed to `docs/reviews/round6-{dcp,paged-indexer,observability,mtp-card}.md`;
fixes applied per repo/worktree:

- **CORRECTION (honest-nulls; round6-observability F5).** The 12:40 entry's evidence line "the
  flight-recorder triage of probeA5 showed the halted host **153 steps behind**" is WRONG twice over:
  (1) probeA5 (run 11:01) PREDATES the flight recorder (commit 11:45) — zero `flight_recorder` lines in
  probeA5.log; the number came from §3's `[OBSERVE_COMPILES]` log lines. (2) Those per-host numbers are
  Ray-dedup artifacts: Ray's canonicalizer strips digit-bearing tokens, so cross-host `step=N` lines
  dedupe and "per-host last step" measures who last won a ~5 s dedup window — the same triage showed
  healthy hosts −12/−24/−33/−36, impossible under sync scheduling's ≤1-step skew, so −153 carries no
  skew meaning (the script itself labels §3 a lower bound). The sharedbin=1 conclusion rests on the
  waveA2 A/B alone — and "CORE-HALT ROOT CAUSE CLOSED" was OVERBROAD: waveB (12:53, .20) and waveB2
  (13:15, .18) both crashed WITH sharedbin=1 (the step-417/kv-513 investigation is the follow-up).
- **Bench HIGH (round6-mtp-card F1): `--protocol card` could never run on the pod** — card mode always
  set `SamplingParams(seed=...)`; the fork's `TpuPlatform.validate_request` rejects every
  `RANDOM_SEED` request per request, AFTER the ~45-min engine build (proven live on this stack; the CPU
  tests checked "seeds reach the generator", never "the engine accepts them"). Fix in `bench/run_bench.py`:
  `_per_request_seed_support()` probes the platform once at `make_generate` (fail-safe: any probe error
  = unsupported); `_sp` omits the seed when unsupported + a loud banner; `items.seed` now records THE
  SEED THE ENGINE GOT (NULL on this backend, and NULL for greedy — F10's cross-protocol ambiguity);
  summary note gains `seed_passthrough=on/off`; per-sample seeds remain labels (`#sN` ids + base_seed).
  docs/07 updated (seeds-are-labels §3 note, §6 banner note). New tests: engine-side platform probe in a
  SUBPROCESS (asserts TpuPlatform rejects seeded + accepts seedless params — keeps test_longctx's
  "vllm never imported" contract intact) + seed-provenance gating. `pytest bench/` 23 → **25 passed**.
- **Triage/launcher (round6-observability F1/F4/F7/F8/F3-countermeasure):** `triage_crash.sh` §6 fetch
  now uses `--output-directory` per-worker files + remote per-line `FR|host|` tags — `gcloud
  --worker=all` on shared stdout interleaves the 8 streams and the old awk misattributed hosts
  (reproduced by the review: false "NO flight data" w-4/w-5/w-7 AND a false "diverged" flag); the output
  header prints the sync/async step-attribution rule (recorder logs at DISPATCH: sync lag 0; async — the
  vLLM DEFAULT — halted step = last line − 0..1) with the run's own `async_scheduling` grepped from the
  log; per-host `ts` column + recorder-death mark (quiet ≫ before cluster-max ts → suspect the RECORDER,
  not the worker); step-less hosts print their newest lifecycle event (crashed-in-load signal) instead of
  "NO flight data"; stale-ts-reuse bug fixed. Launcher stop phase prunes all but the 8 newest
  `/tmp/glm_flight_*` per host (F8 — /tmp pressure at crash time fed the recorder's fail-open).
  shellcheck-clean; §6 awk smoke-tested on synthetic per-worker data; `--no-fetch` re-validated on
  probeA5.log. docs/10: mode-dependence table, corrected divergence heuristic ("peers keep going" is
  wrong for mid-collective halts — expected spread 0–1), kv_len_* = total-known-tokens caveat (F6:
  decode-only lines only; `num_prefill_chunks` will misclassify under Stage-3 MTP).
- **Fork worktrees (each committed + pushed on its feature branch):**
  `glm-5.2-v4-r6fix` @ 413db9e1 (NEW, off origin/glm-5.2-v4) — recorder F2 (failed rotation falls back
  to the original path; every death logs `BLACK BOX DEAD at ts=…`), F3 (3 CONSECUTIVE errors, counter
  resets on success), F4a (`recorder_init` records `async_scheduling` + `GLM_ASYNC_SCHED`); recorder
  tests 11 → 15 pass. `glm-5.2-v4-2a2` @ 755b1719 — paged-indexer finding 1 (MEDIUM latent): pad mask no
  longer opt-in; paged entry points take required `query_start_loc` and derive req-ids + mask internally
  (`token_request_ids`), per-request block tables only (loud ValueError otherwise), `write_indexer_keys`
  requires `valid`; DSA/MLA 5-file suite 93 → 95 pass. `glm-5.2-v4-dcp` @ 70aa6825 — F1 claim accuracy:
  gate-off kernel trace is dataflow-identical (scalar address hoists above paired dma_starts; DMA
  order/operands unchanged ⇒ outputs bit-identical), NOT byte-identical; byte-identity holds only for
  the mocked-kernel `mla_attention` trace; comments/docstrings only, 47/47 pass.
- **Not fixed here (out of scope / other branches):** round6-paged-indexer finding 2 (the S1 DP-guard
  `set(str)` axis-name bug) lives on `glm-5.2-v4-r5fix`; finding 3 (2a2↔r5fix cross-merge) is the
  integration step; mtp-card F2–F9 are MTP-gate/extractor hardening items for the M-milestones and the
  bench extractor backlog (F4–F6 documented in the report, LOW).

## 2026-07-07 15:05 UTC — OOB KERNEL FIX VALIDATED ON HARDWARE; GSM8K n=32 CLEAN AT 32.9 tok/s; GPQA-198 LAUNCHED

- **The mla.v2 pack_new_kv OOB fix (63427f86, merged 02e44b36) HOLDS on the pod**: GSM8K n=32 batched at
  TPU_MIN_TOKEN_BUCKET=32, max_seqs 16 — the exact config that previously died 100% of the time at the first
  512-page crossing — ran CLEAN: **acc 87.5 (n=32, 6 truncated at the 1024 cap)**, 20,280 gen tokens in
  616.7 s = **32.9 tok/s aggregate (~35x the original 0.92 tok/s single-stream)**. Run `gsm8k_n32_fix`,
  full provenance.
- Combined GSM8K evidence: waveA2 (items 0-15) 93.75; n=32 87.5 with 6/32 truncation-misses — the cap, not
  the model, drives most misses; next GSM8K runs use --max-new 2048.
- The prior "JAX_SHARE_BINARY_BETWEEN_HOSTS=1 fixed it" claim was CORRECTED in-log (Ray-dedup artifact +
  composition luck); the real cause was the OOB read (docs/reviews/round7-pg2-oob.md pending).
- **GPQA-Diamond n=198 flagship run launched** (bucket 32, max_seqs 8, max-new 4096, ~4-5 h ETA).
- Stage-2 code-complete on branch glm-5.2-v4-2int (118 tests): GLM_DSA_MODE=pallas_decode end-to-end;
  round-7 adversarial reviews of pg2+2int running; staging merge (glm-5.2-v4-next) + DSA perf follow-ups
  + dense-MTP M1 in flight.

## 2026-07-07 22:3x UTC — INDEPENDENT CLEAN-CLONE REPRODUCTION of the Stage-2 + staging CPU suites (CPU-only; TPU untouched)

- **Method (independence):** two FRESH `git clone`s (not worktrees) of `~/tpu-inference` at the pinned
  commits, run with `JAX_PLATFORMS=cpu` exported before python and `PYTHONPATH=<clone>` forcing import
  resolution off the clean tree (the `~/vllm-env` interpreter carries an EDITABLE `tpu_inference` pointing
  at the working checkout — verified pre-run that `tpu_inference.__file__` resolved into each clone, and
  `jax.default_backend()=='cpu'`). Interp: Python 3.12.13; jax/jaxlib 0.10.1, vllm 0.1.dev1+ga30addc75.tpu,
  torch 2.10.0+cpu, pytest 9.0.3. No XLA_FLAGS needed at either commit (the mesh-test files append
  `--xla_force_host_platform_device_count=8` themselves; the 8-device tests ran, 0 silent skips).
- **Stage-2 CPU suite @ `c1456937` (glm-5.2-v4-sparse-prefill): 124 passed / 0 failed / 0 skipped** in 289 s
  → `docs/artifacts/stage2-cpu-suite-c1456937.log`. FINDING: the tasked 7-file list named
  `tests/kernels/mla_v2_pack_new_kv_oob_test.py`, which DOES NOT EXIST at c1456937 (it entered via the pg2
  merge 02e44b36, not an ancestor of the sparse-prefill branch — forked at the obs merge 87abdf53); ran the
  6 files that exist. Per-file P: dsa_indexer_kernel 24, dsa_sparse_mla 42, glm_dsa_indexer 32,
  glm_dsa_pallas_decode 10, glm_dsa_sparse_prefill 12, mla_head_sharded 4. DEVIATION vs the relayed "140+":
  124 tests exist across these files and ALL pass — a COUNT shortfall, not a failure (consistent with
  c1456937's own commit-message "108 existing + 11 new"; the "140+" likely counted a different file set).
- **Staging cross-suite @ `999f0307` (glm-5.2-v4-next): 250 passed / 123 skipped / 0 failed** in 342 s
  → `docs/artifacts/next-cross-suite-999f0307.log`. REPRODUCES the merge agent's "250P/123S/0F" EXACTLY.
  The merge agent's file list was not recorded anywhere, so it was reconstructed: the 8 GLM-touched files
  (merge-base 02e44b36→999f0307) + the pg2 oob test gave 243P/0S/0F standalone; the missing 123 skips are
  exactly `mla_v2_test.py` (1P/103S) + `mla_tuned_vs_baseline_test.py` (0P/20S) — TPU-Pallas benches
  skipped on CPU — and `test_mla_attention.py` adds 6P → the combined 12-file run hits 250/123/0 on the
  nose. `backends/test_flash_attn_mla.py` is EXCLUDED (5P/5F on CPU: fp8/w8a8 forward paths need TPU), so
  it was not in the merge agent's 0-failure set either. 999f0307 is itself the commit that fixes the
  cross-suite mesh-skip ordering, so the 8-device mesh tests PASS here (44P test_mla_dcp + 4P
  mla_head_sharded), not skip.
- Both logs carry a full header (commit rev-parse, interpreter, versions, env vars, wall time, per-file
  P/S/F, claim-check). Committed as durable artifacts. Bottom line: 0 failures on either branch; the
  staging 250/123/0 is reproduced exactly; the Stage-2 "140+" is 124 (all green) — a counting deviation.

## 2026-07-08 — bench: DCP plumbing (runbook §5a) + _run_env NameError FIX + longctx attention_path (CPU-only; TPU untouched)

- **GLM_DCP → `decode_context_parallel_size` landed in `bench/engine.py build_llm`** (docs/11 §5a prereq,
  bench-owned). Contract: unset/`0`/empty = kwarg ABSENT from the `LLM(...)` args (byte-identical engine
  build — unit test asserts every OTHER kwarg equal between the two builds); `GLM_DCP=N` = kwarg present
  with int N + `dcp=N` in the engine-built log line. Kwarg name verified against installed vLLM
  (`EngineArgs.decode_context_parallel_size`, arg_utils.py; the gpqa198.log config dump echoes
  `decode_context_parallel_size=1` default). Provenance: `GLM_DCP` is caught by the existing `GLM_*`
  os_env sweep in BOTH `_run_env`s (asserted in tests; demonstrated live in results.db run 49).
- **BUG FOUND+FIXED: `run_bench._run_env` crashed EVERY run since the audit commit 3ba7702** —
  `env["attention_path"] = ...` assigned before `env` existed → NameError at `pv.start_run` (stub and pod
  runs alike; nothing exercised `_run_env` in the CPU suite, so tests stayed green). Fix: the field now
  rides in the returned dict via a NEW shared `engine.attention_path()` helper (GLM_DSA_MODE → `dense-mla`
  / `dsa-sparse:<mode>`, semantics identical to the audit's intent), and `glm_longctx._run_env` records the
  SAME field (it previously had none). Regression tests: `test_run_env_provenance_fields` (run_bench) +
  env_json assertions in the longctx stub test. Live proof: results.db run 49 (STUB, gsm8k n=2,
  attention_path=dense-mla, os_env.GLM_DCP="2").
- **Passkey/longctx readiness dry-run (CPU `--stub`, 8K/32K/128K × depths .25/.5/.75 × 2):** 18/18 trials
  recorded to a scratch DB. Prompt lengths on target (ctx ≤ L, within 1%: 8190/32765/131071 + the 2
  explicit `[gMASK]<sop>` prefix ids); needle depth within ±0.02% at every rung; 8K prompts stored
  verbatim, 32K/128K as head+tail+sha256 (cap 65536 chars, seed-reconstructible); per-cell + aggregate
  summary rows present; env_json carries attention_path/stop_ids/prefix. Protocol drift check vs
  run_bench: NONE — both harnesses share `engine.EOS_IDS`; longctx raw protocol still prepends
  `[gMASK]<sop>` ids explicitly and stops on EOS+`<|assistant|>` (round-3 fixes intact,
  `test_raw_prompt_protocol` green). New oracle test covers the HIT path (stub only exercised misses).
- **Adversarial review (independent agent): PASS on all 4 intents, every claim demonstrated executably** —
  `LLM.__init__` **kwargs→EngineArgs forwarding read from the installed vLLM (llm.py:305-345); HEAD-vs-diff
  byte-identity of the unset-GLM_DCP build shown with a capture-fake LLM; the HEAD NameError reproduced;
  0 drift across 8 GLM_DSA_MODE values; oracle-regex attack refuted over 420 trials; no sys.modules leak
  under either pytest order or hostile ambient GLM_DCP/GLM_DSA_MODE. Two low findings FIXED post-review:
  (a) negative GLM_DCP now raises a readable ValueError at the harness boundary + test (vLLM's tp%dcp
  check passes 32%-2==0 in Python; only pydantic ge=1 caught it, opaquely, deep in the build); (b) the
  longctx stub test now save/restores GLM_DSA_MODE instead of pop-without-restore. Noted, accepted as-is:
  Python int underscore/sign forms ("4_0"→40, "+4") parse — pathological inputs; fail-fast covers the rest.
- Suites green: test_bench (13 fns) + test_longctx 12/12, both direct and pytest (25 passed, both orders).

## 2026-07-08 — PR-G5 cut (`pr-g5-mla-pure-tp`): the deferred TP-topology MLA PR, now that pod validation exists (CPU-only; TPU untouched)

- **Branch `pr-g5-mla-pure-tp` @ `132a11f9`** cut off `97938b62` in `~/tpu-inference-prs` and pushed to the
  fork: squashed re-cut of `cd8eeb6c` + `a429be54` + `7ae390f2` (the #2324 TP-topology port group per
  docs/02 §G5) — cross-shard all-gather (+post-gather TuningKey), v4 blocks/fp32 scores, v4-gated
  page-512 via the tpu_info driver-safe probe, EP-head o_proj constraint, MLA-without-DP platform check,
  TPU_MIN_TOKEN_BUCKET (+Ray propagation). Deferred-until-pod-validated per docs/02; the validation now
  cited: results.db runs 26/28/47 (GSM8K n=32 clean at 32.9 tok/s aggregate, acc 87.5, pure TP-32) +
  RESEARCH_LOG 07-07 06:40/12:40/15:05 entries.
- **Two deliberate deltas vs the dev-branch hunks** (documented in pr-g5.md + the commit message):
  (a) TPU_MLA_V4_KV_PAGES/QUERIES debug overrides dropped (scaffolding; every pod run used the defaults —
  verified in the provenance env records); (b) the gather now runs over the token-shard axes whose
  descriptors are REPLICATED (MLP_TENSOR minus ATTN_DATA, derived from the effective specs) instead of all
  MLP_TENSOR axes — identical on the validated TP×EP topology (attn-DP axes size 1), but required upstream:
  gathering over attn_dp would corrupt the pure-DP-attention meshes that are main's ONLY accepted MLA
  config today (descriptors co-shard there). Unit-tested algebra incl. the hybrid DP×TP (#2988/Kimi) case.
- **CPU tests**: 17 new (test_mla_cross_shard 6 — mock-kernel spec-algebra, 8-dev mesh;
  test_flash_attn_mla_page_size 9; test_tpu_runner_min_token_bucket 2) + platform suite 39/39 (1 new test;
  the pre-existing MLA-check test rewritten to the new contract). Adversarial vs pristine base: 7
  behavioral failures + a direct probe of the cross-shard bug (8-way vs reference max |diff| 3.91; base
  TuningKey saw the 1/8-shard shape). Existing suites identical to base (incl. the 5 pre-existing
  Pallas-needs-TPU fails in test_flash_attn_mla.py). Forward-port to upstream tip `6a837025` (re-fetched
  2026-07-08): 1 mechanical TuningKey conflict hunk; tip still lacks every piece (verified by reading it).
- **Duplicate-work re-sweep (live, 2026-07-08)**: #2324 unchanged since 07-04 (head `c2822bd7`, needs
  rebase) — G5 is a port of 5 of its hunks, disclosed hunk-by-hunk with the axis/gating/test deltas;
  commit carries `Co-authored-by: yiqiliu2`; owner must run the #2324 conversation before submission.
  **NEW finding: #2988 is an ALTERNATIVE fix for the same cross-shard bug** (rewrites the mla_attention
  default token specs MLP_TENSOR→ATTN_DATA; textual+semantic conflict — if it lands first our gather
  correctly degrades to a no-op; maintainers must pick a default). #2930/#2955/#3056/#2767 adjacent,
  no overlap.
- **lm_head vocab-sharded logits guards (761ea755/87ace031/10efa393 heritage) audited, NOT ported**:
  they guard the DSV4-branch STEP-1 token-sharded-logits machinery, which does not exist at the base or
  tip — base already keeps vocab-sharded logits for the vocab-sharded lm_head (wrapper out-sharding
  P(MLP_DATA, MLP_TENSOR) at :660 + lm_head P(MLP_TENSOR, None) in unquantized.py:169). Porting = dead
  code referencing nonexistent variables. Documented as an audit note in pr-g5.md; the guards belong to
  the (unsubmitted) DSV4 logits-layout series.
- Docs: `docs/pr-descriptions/pr-g5.md` written (full PR body draft: description, hunk-level #2324/#2988
  disclosure, test commands+results, pod evidence with run ids, risk, AI disclosure, submitter
  checklist); README table + header updated. No S1 head-shard gate, no DCP (follow-on PRs per the task
  directive).

## 2026-07-08 — Round-8 finding 1 CLOSED: stranded round-6 fix `755b1719` merged to staging (CPU-only; TPU untouched)

- **`glm-5.2-v4-next` @ `15246fc8`** (pushed) merges `glm-5.2-v4-2a2` @ `755b1719` — paged-indexer hardening (write_indexer_keys `valid` REQUIRED keyword-only; compute_topk_indices_paged / topk_indices_for_layer_paged take required `query_start_loc`, derive the token→request map + pad mask internally, per-request block tables only) now lands on the 2int/sparse-prefill/MTP state; auto-merge textually clean + one semantic-conflict fix (test_glm_dsa_pallas_decode.py history prepopulation now passes an explicit all-ones `valid` — provably unpadded rows); production wiring reconciled per round8-freeze-conformance.md finding 1 (mla_attention.py:1059/:1153 pass `valid=tok_valid` by keyword — untouched, still compatible); frozen kernels untouched; suites (CPU, JAX_PLATFORMS=cpu): indexer 34 / pallas_decode 18 / sparse_prefill 12 / mtp_index_share 14 / dsa_indexer_kernel 24 / dsa_sparse_mla 42 — all ≥ the 32/18/12/14/24/42 gates.

## 2026-07-08 04:20 UTC — GPQA-Diamond n=198 COMPLETE (dense path): raw 52.5 TRUNCATION-DOMINATED; 86.2% on completed items

- Run 48 (~/glm-run/gpqa198.log, 8h52m, 703,918 gen tokens, 22.0 tok/s aggregate, ZERO interrupts —
  the OOB fix's longest hardware validation yet). attention_path=dense-mla (Stage-1 number; DSA bypassed).
- **Raw acc 52.5 vs card 91.2 (Δ −38.7) is an ARTIFACT of --max-new 4096**: 140/198 items (71%) truncated
  mid-reasoning (GLM-5.2 thinks long; the card evaluates at a 163,840-token cap).
- Honest split: **completed items 50/58 = 86.2%** (within ~1σ of the card for n=58; caveat — the completed
  subset skews toward easier/short-reasoning items, so this likely OVERSTATES slightly); truncated items
  54/140 = 38.6% (salvaged partial answers, above the 25% MC floor).
- Decision: rerun at max-new 16384 on the staging branch AFTER the single-chip kernel gates + byte-identity
  smoke (runbook order). All 198 items with verbatim outputs in results.db run 48.

## 2026-07-08 05:25 UTC — 🎉 STAGE-2 ON-METAL GATES ALL PASS (single v4 chip, kernels silicon-validated)

- **GATE 2a ACCEPT** after three v4 Mosaic lowering fixes (sanctioned freeze-break, all in the w/output
  BlockSpecs + the documented broadcast-multiply fallback; commit on glm-5.2-v4-next). Round-5's flagged
  highest-risk spec (the (1,H) w-tile) was indeed rejected — exactly as predicted — and the documented
  fallback landed.
- **GATE 2b ALL PASS on real MXU**: A1 pallas-vs-XLA fp32 2.4e-7; **A2 selected-set-EXACT vs the HF-math
  oracle (0 non-tie mismatches)**; A3 bf16 0 out-of-band (S2 ε=2^-8); GATE B sparse-MLA fp32 9.5e-7 /
  bf16 1.95e-3; GATE C pack_new_kv OOB geometry no-OOB + byte-identical at kv_len 511/512/513.
- One probe bug found ON METAL and fixed: the HF oracle itself ran at default MXU precision (bf16 passes,
  ~4.5e-3 self-error) — the first A2 "failure" was the ORACLE's error, not the kernel's. Oracle now pinned
  to matmul precision 'highest' (no-op on CPU).
- Verbatim logs: docs/artifacts/kernelprobe-2a-20260708*.log (REJECT trail + ACCEPT),
  kernelprobe-2b-20260708-metal2.log + .results.txt. CPU regression: 144/144 DSA tests on -next.
- The audit's steps 1+2 are green. Next: staging switch smoke + GPQA rerun @16K cap; then passkey/throughput.

## 2026-07-08 07:55 UTC — STAGING SWITCH VALIDATED: -next byte-identical to run 26 (4/4 sequential); GPQA rerun @16K launched

- Pod fleet on glm-5.2-v4-next @ 886eaceb. Sequential smoke (matched run-26 config): **raw_output byte-identical
  4/4 items** — the whole merged Stage-2+3 stack is provably inert with gates off, on hardware. (First smoke
  compared batched-vs-sequential and differed on 3/4 — a confounded comparison, documented, not a defect:
  concurrent MoE batching changes summation order.)
- GPQA-Diamond rerun launched at max-new 16384 (the 4K-cap truncation artifact fix), bucket 32, max_seqs 16.

## 2026-07-08 08:05 UTC — PRECISION AUDIT (the A2-lesson generalization): every matmul/oracle/scoring path in the passkey+throughput+parity instruments audited for the silent bf16-MXU default-precision hazard

- Scope: could `jax.default_matmul_precision` DEFAULT (fp32 dots via bf16 MXU passes, ~4.5e-3 self-error;
  `preferred_element_type=f32` does NOT prevent it) skew any number the gates rely on — including a
  FLATTERING wrong number? CPU-only audit; TPU untouched.
- **IMMUNE (no numeric comparison path at all):** bench/glm_longctx.py (tokenize + greedy generate +
  string exact-match; numpy only for host RNG/needle placement/means); bench/dsa_throughput.py (tok/s =
  host wall-clock floats over integer token counts; no device math in the harness);
  bench/report_throughput.py (A/B join on DB floats; output-identity = integer token-id list equality);
  gate D0 4a (runbook §step-4: verbatim raw_output string compare via results.db) and 4b (npz
  bit-compare of the SAME program off-vs-sparse — no oracle). bench/*.py imports no jax/torch outside tests.
- **ALREADY PINNED (why the on-metal 2b numbers are trustworthy):** the fork oracles pin precision
  INTERNALLY — indexer_scores_xla (qk dot via _score_precision → HIGHEST for fp32, head-sum einsum
  explicitly HIGHEST) and dsa_sparse_decode_xla (all three einsums precision=HIGHEST for fp32), matching
  the kernels' own HIGHEST-fp32 rule; hierarchical_topk/topk_indices contain no matmuls. That is exactly
  why GATE B measured 9.5e-7 and A1 2.4e-7 on metal: kernel AND twin both ran 3-pass fp32. bf16 rows use
  DEFAULT on both sides deliberately (bf16×bf16→f32 is single-pass exact given bf16 inputs; Mosaic
  rejects an fp32 contract precision on bf16 operands). No fork change needed.
- **SAFE BY PLATFORM:** parity/glm_engine_parity.py (fp32+bf16 controls are HF **torch on CPU**; the
  logits matmul is host numpy; the TPU side is the DUT, not a reference), parity/glm_mtp_parity.py
  (JAX_PLATFORMS=cpu + explicit CPU mesh; fp32-vs-fp32 both exact on host),
  parity/glm_indexer_rope_experiment.py + test_indexer_reference.py (force+assert CPU backend).
- **FIXED (the one real gap):** parity/glm_indexer_reference.py — the HF-math oracle itself carried NO
  internal pinning (wq_b/wk/weights_proj matmuls + both einsums at caller-context precision); it was
  protected only by probe_2b's caller-side 'highest' wrapper, while the fork tests
  (tests/kernels/test_dsa_indexer_kernel.py, tests/layers/vllm/test_glm_dsa_{indexer,sparse_prefill}.py)
  import the same module with NO wrapper — a latent on-metal A2 repeat. `indexer_scores` now wraps its
  whole body in `jax.default_matmul_precision("highest")` (no-op on CPU; idempotent under probe_2b's
  wrapper), and test_indexer_reference.py gained test (d): lower under a hostile 'bfloat16' caller
  context and assert all 5 dot_generals carry HIGHEST in the HLO.
- Tests: parity/test_indexer_reference.py ALL PASS (a-d); probe_2b --interpret ALL GATES PASS
  (kernelprobe-2b-audit-interpret.results.txt); bench test_longctx+test_dsa_throughput 29/29; fork
  tests/kernels/test_dsa_indexer_kernel.py 24/24 against the pinned reference.
- Production note (not a defect): the fork's serving-path scorer
  (layers/vllm/custom_ops/glm_dsa_indexer.py score_block + projections) runs at DEFAULT precision by
  design — it is the DUT, never an oracle; D0 is structurally insensitive (topk ≥ ctx) and passkey
  measures the production system as-is.

## 2026-07-08 — MTP M2 prep (runbook §7): GLM_SPEC_K knob + mtp_m2_check.py landed, §7 refreshed for zero-turnaround (CPU-only; TPU untouched)

Everything M2 needs on the pod is now committed and CPU-tested — the pod session only runs the
three run_bench commands + the checker.

- **Engine knob (`bench/engine.py`):** `GLM_SPEC_K=k` → `speculative_config={"method": "mtp",
  "num_speculative_tokens": k}` in the `LLM(...)` args. Kwarg + dict shape verified against the
  installed vLLM (`~/vllm-build/vllm/engine/arg_utils.py:616` — `EngineArgs.speculative_config:
  dict[str, Any] | None`, consumed by `create_speculative_config` → `SpeculativeConfig(**dict)`);
  `"mtp"` is a valid `SpeculativeMethod` (`config/speculative.py` MTPModelTypes), the glm_moe_dsa
  surgery maps to `DeepSeekMTPModel`/`n_predict=1`, and k=5 passes the `k % n_predict == 0`
  module-reuse check (speculative.py:773). Fork routing re-verified: method `"mtp"` →
  `Eagle3Proposer` (`tpu_runner.py:711` + `eagle3.py` mtp branches). Unset/0/empty = kwarg ABSENT
  → byte-identical engine args (same contract as GLM_DCP); negative = readable harness-boundary
  ValueError. Provenance: GLM_SPEC_K is captured by the existing GLM_* os_env sweep in
  `run_bench._run_env` (verified by test assertion — no new provenance code) and the engine-built
  line prints `spec=mtp:k=<k>`. Unit test `test_bench.py::test_spec_engine_kwarg` (monkeypatched
  vllm.LLM): absent/present k=1/k=5/0-off/negative-raise + byte-identity of every other kwarg.
- **M2 comparison instrument (`bench/mtp_m2_check.py`, stub-tested):** given two run ids (either
  order — roles ORIENTED from runs.env_json `os_env.GLM_SPEC_K`, never argument position; or
  `--latest` for the back-to-back §7c pair), asserts per-item exact identity of
  (raw_output, n_gen_tokens, finish_reason) — the DB's faithful projection of the generated token
  sequence (identical ids ⇒ identical triple; any triple mismatch ⇒ sequences differ; the
  converse text-alias gap is documented in the module docstring, honest — raw token-id capture is
  an M3 instrumentation item). Hard-fails (exit 2) on non-comparable pairs: ambiguous roles,
  non-greedy protocol/temperature (the M2 theory bar is greedy-only), differing item sets or
  prompts; WARNS on generation-relevant env drift (max_new/max_len/GLM_DSA_MODE/fork_git/...).
  Exit 1 on mismatch with per-item first-divergence offset + excerpts (feeds the docs/08 tie-flip
  protocol). Acceptance stats: `--log <spec run log>` scrapes vLLM's interval `SpecDecoding
  metrics:` lines (format pinned to `~/vllm-build/vllm/v1/spec_decode/metrics.py`; needs
  GLM_LOG_STATS=1, already in the §7 BASE env) → per-interval + run-aggregate acceptance
  (aggregate mean-acceptance-length recovered from the 2-dp interval values — labeled
  approximate; per-item attribution honestly reported as unavailable until M3). tok/s A/B from
  the two summary notes (batch_wall_ms/gen_tok). New suite `test_mtp_m2_check.py` (10 tests):
  identity pass both argument orders + CLI exit 0 + --latest, mismatch detection (text divergence
  offset + count-only divergence), role/protocol guards, item-set/prompt guards, vacuous-PASS
  guards (stub/all-empty), NULL-vs-empty mismatch, usage-error exit codes, env-drift warnings,
  log-scraper parse + aggregates (incl. the all-nan degenerate interval), --log CLI wiring.
  Tracked bench suites all green (test_bench.py 14 incl. the new engine-kwarg test +
  test_mtp_m2_check.py 10; full `pytest bench/` green alongside a concurrent session's in-flight
  --ids/merge_runs/report_passkey work, which is NOT part of this commit).
- **Runbook §7 refreshed** for the n=8 SEQUENTIAL bring-up pair (--limit 8 --batch-size 1: one
  request in flight, simplest batch shaping) before the N≥32 batched docs/08 bar: exact command
  pair = baseline (spec OFF) → `GLM_SPEC_K=1` → `mtp_m2_check.py --latest --log ~/glm-run/m2_k1.log`,
  then `GLM_SPEC_K=5` vs the SAME baseline (explicit ids). **7a branch-state verification (git
  merge-base, 2026-07-08): `glm-5.2-v4-mtp-g4` IS merged into `-next` (merge `534cd74d7`) AND the
  OOB fix `02e44b36` is an ancestor — both coexist on origin/`-next` from `886eaceb4` onward**
  (local `-next` @ `cda8a707b` adds the det merge, unpushed at prep time — workers pull from
  origin). The old 7a prereq (merge OOB into `glm-5.2-v4-mtp`) is OBSOLETE — M2 runs from the
  step-2 staging engine, and the old 7b GLM_SPEC_CONFIG one-liner is superseded by the landed
  GLM_SPEC_K knob. Branch-map rows in docs/11 + HANDOFF updated to match.

  **Round-9 adversarial review of this prep (fresh reviewer on the staged diff — all claimed vLLM/fork
  verifications independently re-verified and confirmed): 2 MED + 8 LOW findings, all addressed.**
  MED-1 (exit-code contract): a typo'd `--db`/`--log` path crashed with exit 1 — indistinguishable from
  an M2 mismatch — and `sqlite3.connect` silently CREATED the missing DB file; fixed (path checks +
  sqlite3.Error/OSError → exit 2, no side-effect file; tested). MED-2 (vacuous PASS): two `--stub` runs
  or an all-SKIP pair passed "identity" over empty outputs; fixed (stub/model=STUB guard + all-empty-
  output guard → CompareError; tested). LOW: per-position regex now accepts vLLM's all-`nan`
  num_drafts=0 interval line (byte-exact reconstruction test); garbage `GLM_SPEC_K=abc` now gets the
  readable knob-naming error; NULL-vs-"" raw_output no longer conflated (a real difference is a
  mismatch); `run_tok_s` docstring corrected to END-TO-END tok/s (wall incl. prefill; newest-summary-row
  = single-benchmark runs only); the mal-recovery bias mechanism + the DEBUG-logged final idle flush
  documented in `parse_spec_log`; missing/unparseable env_json protocol now warns. Not fixed (accepted):
  reviewer could not verify live `SpecDecoding` emission on this offline-LLM+Ray stack (checker prints a
  loud hint when zero lines match) nor anything pod-side — that IS the §7 run this preps.
  [Relocated: the parallel PR-G6 session's append interleaved with this entry's commit, leaving this
  paragraph under the PR-G6 heading; moved here where it belongs.]

## 2026-07-08 12:00 UTC — PR-G6 CUT: the DSA kernels (headline contribution) — branch `pr-g6-dsa-kernels` pushed (CPU-only; TPU untouched)

- **Cut from base `97938b62`** in `~/tpu-inference-prs` (same discipline as G1–G5), commit
  `0ac6eb9e4`: 5 new files — `tpu_inference/kernels/dsa/{__init__,indexer_kernel,sparse_mla_kernel}.py`
  + `tests/kernels/test_dsa_{indexer_kernel,sparse_mla}.py` — taken at the silicon-validated `-next`
  state (`886eaceb`, identical through head `cda8a707`; the newer round-9 det commits touch the
  integration layer only, verified by path diff). Kernels import only jax/pallas — fully
  self-contained against the base (no mla.v2 dependency; checked). **KERNELS ONLY** — serving wiring
  (indexer module, cache writers, dispatch, IndexShare, sparse prefill) is declared a follow-on PR.
- **Delta vs -next: yapf 0.43.0 only** (the repo pre-commit pin; -next files weren't yapf-clean),
  verified **AST-identical per file** (`ast.dump` equality) — the code is semantically exactly what
  the 2a/2b silicon gates ran. isort/ruff still unavailable locally (owner: `pre-commit run
  --all-files`).
- **CPU tests: 66/66** (`JAX_PLATFORMS=cpu`, interpret; 24 indexer + 42 sparse-MLA, ~95 s), rerun
  post-yapf. Upstream-conditions run (glm-tpu HF-math oracle absent via `GLM_TPU_ROOT=/nonexistent`):
  **62 pass + 4 graceful skips** — the suite is CI-safe without the harness repo. New-module tests
  fail structurally on the pristine base (package absent).
- **Forward-port:** upstream tip re-fetched (`99a662a1`, 2026-07-08); merge-tree trial merge
  **conflict-free** (all-new files; no `kernels/dsa/` path on tip).
- **Live duplicate-work sweep (2026-07-08 ~11:45 UTC, GitHub API):** no open PR ships DSA kernels
  (9 queries + 50-PR title scan + files/diff reads of #2324/#2988/#3073/#3062/#3096); **#2324
  re-verified to disable the indexer** (`TPU_DISABLE_DSA_INDEXER`, "until a JAX-native DSA lands" —
  exact lines quoted in pr-g6.md). **Load-bearing counterfinding:** merged main carries
  `kernels/experimental/deepseek_v4/` (#2903/#2905/#2980, 2026-06-22..24) — a DeepSeek-V4
  KV-compressor StreamIndex top-k + topk-consuming sparse-MLA Pallas stack. Adjacent mechanism, zero
  file overlap, but it falsifies an unscoped "first public TPU Pallas indexer/top-k" claim →
  pr-g6.md scopes the claim to the exact-top-k, uncompressed-latent DSv3.2/GLM (`GlmMoeDsa`) DSA
  variant and adds a position-don't-compete checklist item.
- **pr-g6.md written** (G-series template): validation story with artifact paths + exact deltas
  (2a three-REJECT→ACCEPT trail incl. the archived first-metal A2 oracle-precision failure; 2b metal2
  A1 2.4e-7 / A2 selected-set-exact 0 non-tie / A3 bf16 0 out-of-band @ ε=2⁻⁸ / B 9.5e-7 fp32,
  1.95e-3 bf16), honest limits (decode-shaped; prefill masked-XLA in the follow-up; v4-validated
  block configs, no perf numbers claimed — microbench outstanding; bf16 S2 boundary-band semantics
  spelled out; k=64 probe-shape caveat; metal results-file branch-header WARNING disclosed), AI
  disclosure, owner-submits checklist. README table G6 row updated (was "SKIPPED — too fresh").
- **Post-cut adversarial review (fresh reviewer, full claims audit vs primary artifacts + independent
  test/AST/merge re-runs): 0 CRITICAL / 1 HIGH / 3 MED / 5 LOW — all addressed.** HIGH-1: the shipped
  indexer module docstring still described the PRE-Mosaic-fix kernel ((1,H)-matmul-LHS w tile, [1,P]
  output block, stale VMEM table) and listed the silicon-validated items under "Remaining for real-TPU
  validation" — fixed in a second, docstring-only commit `5bf3e5927` (code-AST-identical, verified via
  docstrings-stripped `ast.dump`; 66/66 re-run; pushed — no force-push, per repo rule), together with
  the same-class staleness the review pattern exposed (test docstring's "bars MUST be re-measured on
  TPU before upstreaming" — done on metal 2026-07-08, now recorded; the unscoped "no public JAX/Pallas
  DSA kernel" sentence; fork-side markers on the `mla/dsv4` references; seg_block sweep list). MED:
  bf16 churn statistic provenance split (metal k=64: 0 out-of-band, 0–2/64 in-band; CPU round-5:
  1–2/2048); **chain-of-custody disclosure added — `886eaceb` was committed 05:19 UTC, AFTER the metal
  runs (2a 05:05, 2b 05:09), the audit rerun @886eaceb is CPU-interpret, so no artifact pins the metal
  numbers to the commit bytes** (corroboration: 2a try-1 reproduces the pre-fix-only (1,H) BlockSpec
  reject) → the owner clean re-run from the PR branch is now a REQUIRED checklist item; "approx-shaped
  StreamIndex" mischaracterization of deepseek_v4 dropped (it's exact streaming top-k at
  compressed-block granularity — compressor-based is the real differentiator). LOW: stale try4 log
  banner disclosed; `cda8a707` correctly labeled local/unpushed; README tip-fetch header updated.
  These fixes exist only on the PR branch — fold-back to `-next` is an owner checklist item.
- **Log-threading note:** the parallel MTP-M2 session's commit `94b6664` appended its round-9-review
  paragraph after this entry's heading (concurrent appends); relocated to its own entry above.

## 2026-07-08 13:10 UTC — D0 CLOSED: sparse path deterministic + correct in the 753B engine; passkey ladder started

- **Across-boot determinism post-fix: 4/4 byte-identical** (runs 56 vs 57, two fresh engine boots,
  GLM_DSA_MODE=pallas_decode sequential). The round-9 root cause (selection-ORDER summation amplifier:
  descending-score gather order x fp non-associativity x cross-boot executable skew) is FIXED by the
  canonical ascending-position gather (fork 215f8ddb, merged cda8a707); same-boot probe had already shown
  4/4 (graph input-pure). Round-9 adversarial review: fix correct as merged; harness edits sound.
- D0 gate summary (audit step 1): sparse serves the full 753B correctly — accuracy identical to dense
  (3/4 same items, same miss), deterministic, kernel-level fp32 exactness = the silicon GATE B result.
- Audit step 2 started: passkey ladder 8K/32K x depths {.25,.5,.75} x 12 trials on the SPARSE path
  (attention_path=dsa-sparse:pallas_decode), gate = every (length,depth) cell >= 95%.

## 2026-07-08 15:30 UTC — PASSKEY 8K/32K: SPARSE 100% ALL CELLS (matches dense); F1 hardware-proven

- **Sparse path (attention_path=dsa-sparse:pallas_decode): 72/72 needles, 100% in every (length, depth)
  cell at 8K and 32K** — at 32K the DSA kernel attends to only the 2048 indexer-selected positions (16x
  sparsification) and retrieval is perfect. Dense baseline: also 72/72 (runs recorded back-to-back,
  same seeds). Report gate: PASS per-cell; coverage verdict honestly exits 2 (<128K — not the gate yet).
- The F1 static-elision fix is hardware-proven: the 33K sparse engine that E1000-OOM'd at compile now
  builds and serves (fork d8fddbda). mbt raised 512->2048 for prefill-heavy runs (~4x prefill speedup).
- 128K sparse requires the SPARSE x DCP composition (docs/05 §6 owner-gather) — build launched
  (glm-5.2-v4-sdcp). Meanwhile: DCP dense bring-up (dcp=4, first DCP on hardware) running on the pod;
  dense 128K passkey next.
