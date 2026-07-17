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

## 2026-07-08 17:00 UTC — GSM8K n=32 @2048 cap: 96.9%; DCP has a mesh-axis bug (blocks 128K/256K gates)

- **GSM8K n=32, max-new 2048: acc 96.875% (31/32, 1 truncation)** — dense path, truncation-free quality
  signal at scale. Full provenance.
- **DCP FAILURE (blocking the 128K passkey + 256K throughput gates):** GLM_DCP=4 (vLLM
  decode_context_parallel_size=4) + GLM_MLA_DCP=1 serves SHORT contexts correctly (GSM8K smoke acc==dense)
  but FAILS all passkey lengths 8K-128K (pred=None, haystack-filler outputs = model sees only ~1/4 of
  context). Root cause hypothesis H-MESH: vLLM stripes the KV block tables for dcp=4 but the fork's mesh
  stays (…,model=32,dcp=1) — decode_context_parallel_size is NOT wired into the mesh dcp axis, so attention
  reads the logical stripe as if contiguous. Root-cause+fix agent running (Opus).
- HBM reality: a single 128K MLA sequence needs ~11.9 GiB/chip of latent KV at dcp=1 (replicated across TP)
  > ~7.7 GiB free after the 23 GiB model — so 128K genuinely REQUIRES DCP (dcp>=2 → <=5.95 GiB/chip fits).
  The dcp=1 fast path is ruled out by HBM; DCP must be fixed. Interim: extending sparse evidence to 64K@dcp=1.

## 2026-07-08 17:40 UTC — dcp=1 ceiling confirmed at 32K; ≥64K REQUIRES the DCP fix (E1000 at 64K@dcp=1)

- 64K@dcp=1 (max_seqs 1, gmu 0.92, 136 blocks / 69,362-token pool): **CompileTimeHbmOom (E1000)**. The KV
  pool fits (~6.1 GiB replicated) but the dense-MLA attention program's 64K working buffers exceed the
  ~2.5 GiB headroom after the 23 GiB model. Lowering gmu trades KV pool for program scratch but can't win
  at dcp=1 — the design's answer is DCP (shards KV ÷dcp → room for scratch AND longer ctx).
- **VERIFIED CLEAN-PATH CEILING: 32K@dcp=1, sparse 100% all cells.** The 128K passkey + 256K throughput
  gates are BOTH hard-blocked on the DCP mesh-axis fix (agent running on Opus). No further dcp=1
  long-ctx attempts — they cannot reach the >=128K gate.
- Pod idle until: DCP fix lands (-> 128K passkey), OR owner greenlights the full GPQA-198 @16K (parked).

## 2026-07-08 18:20 UTC — DCP unblock ruled out; 128K needs real on-metal DCP-kernel work (VMEM + kv_packing)

- Gate-OFF gather test (GLM_DCP=4, GLM_MLA_DCP unset, 128K): FAILED at compile —
  `MLA-...-p_2048-... RESOURCE_EXHAUSTED: Allocation (size=17301504) would exceed memory` = 16.5 MB VMEM
  buffer > v4's 16 MB, because dcp=4 makes the KV page = block_size(512) x dcp(4) = 2048 tokens. Hits
  the MLA decode kernel regardless of the attention gate. AND gate-off gather re-materializes the full
  cache (defeats the ÷dcp memory saving) — so it can't reach 128K even if it compiled.
- Opus root-cause (refuted my H-MESH): mesh split IS correct (model8 x dcp4); the real failures are
  (1) MLA kernel VMEM at the dcp logical page 2048, (2) the sharded DCP path's kv_packing=32 multi-block
  bitcast read (zero test coverage — interpret path asserts kv_packing==1). Both are on-metal kernel work.
- **HONEST FRONTIER:** sub-128K is DONE (32K sparse 100%, GSM8K n=32 96.9%, kernels silicon-validated).
  128K passkey + 256K throughput require the DCP-kernel fix (VMEM page-tiling + kv_packing correctness) —
  a genuine ~1-day on-metal effort. No config shortcut exists (fp8-KV would fit 128K@dcp=1 but isn't
  supported on the sparse path yet). Design agent launched; NO more pod trial-and-error until a concrete fix.

## 2026-07-08 (later) — DCP kv_packing>1 SUSPECT REFUTED on CPU: kernel is CORRECT; the real blocker is VMEM (dcp=2 is the config) — branch `glm-5.2-v4-kvpack`

Ran the kv_packing>1 multi-block DCP investigation on CPU (worktree `~/tpu-inference-kvpack`, branch
`glm-5.2-v4-kvpack` off `origin/glm-5.2-v4-next`; TPU untouched). The 18:20 hypothesis — "the sharded
DCP path's kv_packing=32 multi-block bitcast read is where the bug hides (zero test coverage)" — is
**REFUTED**. There is **no kv_packing>1 correctness bug in the MLA v2 decode kernel**; the on-metal
128K/dcp=4 blocker is **VMEM**, and **dcp=2 is the viable v4 config**.

- **Closed the coverage gap (the fix):** `kernels/mla/v2/kernel.py` `load_bkv` interpret branch dropped its
  `assert kv_packing == 1` — the packed read is now CPU-executable at kv_packing>1 via the plain C-order
  reshape `bkvc_x2_ref[sem,b,:bkv_sz_per_kv_packing].reshape(bkv_sz, D)` (logical token t at physical
  (t//pack, t%pack)). **Proven byte-for-byte equal to the REAL `ref.bitcast(uint32)…pltpu.bitcast` round
  trip** under the interpreter for kv_packing∈{2,4,8,32}, fp32 AND bf16, incl. the +2 buffer padding and
  [2,batch] leading dims (independent reviewer reproduced it; the one apparent kv_packing=2 mismatch in an
  early probe was a bf16 marker-rounding artifact, not a bitcast bug). No-op in production (`_INTERPRET=False`
  leaves the real bitcast path untouched); byte-identical at kv_packing==1. 1-line semantic change + comment.
- **The reproduction test (it PASSES):** new `tests/kernels/test_mla_v2_kvpack_dcp_cpu.py` (39 cases) runs the
  REAL DCP kernel over context-striped packed slices + the production `dcp_lse_merge`, vs a pure-numpy
  full-context attention **ground truth** (independent of the kernel — a wrong kernel cannot self-certify).
  Covers: kv_packing∈{4,8,32}, dcp∈{2,4}, deep multi-page decode (needle on the last stripe of the last
  page), 4-way `decode_batch_size=4` BATCHED_DECODE (the v4 decode path), mixed straddling batches, BOTH
  strided-mask implementations (two-step `flash_attention_step1` AND one-step `batch_flash_attention`), and
  production-scale geometry (pack=32, dcp=2, P_l=128, 3 pages). **All pass** — read + stripe + global-position
  mask + LSE combine reproduce full attention exactly. dcp=1 packed read == numpy too (isolates the read).
- **VMEM is NOT the DCP-kernel blocker (CORRECTION — an earlier draft of this entry overclaimed it; the
  adversarial reviewer caught it).** The DCP kernel runs INSIDE the `shard_map` (attention_interface.py:985)
  over the `P(BATCH, CONTEXT)`-sharded cache, so it receives the shard's LOCAL slice: the local page is
  `block_size` (~512 tokens) at ANY dcp — the scheduler's `block_size *= dcp` and the ÷dcp CONTEXT sharding
  cancel. So the per-block buffer `bkvc = 2·decode_batch_size·(512/32 + 2)·32·512·2 B ≈ 4.5 MB` fits at dcp=2
  AND dcp=4. The **16.5 MB `RESOURCE_EXHAUSTED size=17301504`** seen on metal was the **gate-OFF** path
  (`GLM_MLA_DCP` UNSET → the regular `mla_ragged_paged_attention` on the UN-sharded, gathered 2048-token page
  = 512·dcp) — a DIFFERENT kernel path, not the DCP kernel. My "VMEM at page 2048" arithmetic used the global
  page; under DCP the kernel never sees it.
- **So the honest conclusion is scoped, not closed:** the DCP kernel's kv_packing>1 multi-block LOGIC (packed
  read semantics, striping, global-position mask, LSE combine) is CORRECT **as modeled by the Mosaic
  interpreter** (proven; mutation-verified; reviewer-reproduced incl. int8/fp8 + permutation traces), and it
  FITS VMEM at dcp=2 and dcp=4. What CPU CANNOT certify: the production read lowers to the **native**
  `tpu.bitcast` (mosaic/lowering.py:4274) + `tpu.memref_bitcast` (:1885), which the interpreter does not run
  (it substitutes JAX's Python reference model); and the real kernel INSIDE `shard_map` (a JAX limitation).
  A SILENT 1/dcp truncation is a correctness signature — NOT what a VMEM OOM produces — so if the on-metal
  symptom is accurate, the true root cause lives precisely where CPU is blind: native-bitcast sublane
  ordering, the packed-buffer DMA layout, or the runner-side cache-spec/block-table plumbing (is the
  `P(BATCH,CONTEXT)` sharded local cache actually created? do block tables arrive in P_g-token units?).
- **THE FALSIFIER (decisive; coordinator's primary ask): the "~1/dcp whole-shard-drop" does NOT reproduce on
  CPU at kv_packing==1 on a REAL dcp mesh → the metal bug is NOT the CPU-fixable combine/accumulation.**
  New `test_falsifier_real_mesh_combine_no_shard_drop` (in the kvpack test file): the REAL per-shard kernel
  `(out, lse)` (kv_packing==1, num_bkv>1 — needle spans ≥2 LOCAL blocks per shard, keys on EVERY shard) fed
  through the PRODUCTION `_dcp_lse_combine` inside an actual `jax.shard_map` over the 'dcp' axis on 8
  simulated CPU devices, incl. pod-mirroring **model×dcp** meshes (dcp,model)∈{(2,1),(4,1),(2,4),(4,2)}.
  **All pass**: combined == replicated full-context numpy reference (and ≠ shard-0-only, so non-vacuous). This
  targets exactly the two CPU-fixable suspects — (5a) the mesh `pmax/psum` over a "degenerate 'dcp' axis"
  dropping a shard, and (5b) the num_bkv>1 online-softmax m/l → per-shard lse — and **falsifies both**.
- **Convergent exoneration of the read:** the bitcast `load_bkv` is SHARED with the WORKING non-DCP dense path
  (dense GSM8K n=32 @ kv_packing=32/bf16 = 96.9% on metal); `history_only` bypasses only the WRITE, not the
  READ. So the native packed read is already proven correct on silicon by dense — consistent with the CPU
  fidelity proof. The residual metal-only DCP bug is therefore NOT the read and NOT the combine/lse; it is
  localized to the DCP-specific XLA glue that CPU cannot exercise on metal: the owner-scatter WRITE at
  kv_packing=32 (`attention_interface.py:906-922`, tested on CPU only at pack≤2 in `test_mla_dcp.py`), the
  runner-side cache-spec/block-table plumbing (is the `P(BATCH,CONTEXT)` sharded local cache actually created;
  do block tables arrive in P_g-token units), or a metal-only miscompile — **an on-metal dump, not a CPU fix.**
- **Concrete on-metal instrument added (owner runs on the pod, 1 chip, no serving stack):**
  `tests/kernels/test_mla_v2_kvpack_bitcast_tpu.py` — runs the EXACT load_bkv `ref.bitcast/pltpu.bitcast`
  round trip on real silicon with distinct-value data and asserts it equals the C-order layout
  (`JAX_PLATFORMS=tpu pytest tests/kernels/test_mla_v2_kvpack_bitcast_tpu.py`). Dense working already implies
  this passes; if it did FAIL, native `tpu.bitcast` ordering would be the root cause. The higher-value metal
  escalation is a dcp=2 AND dcp=4 / kv_packing=32 multi-page needle run with a per-shard-lse + post-scatter
  cache dump, compared to the numpy full-context reference, to catch the write-routing/plumbing suspect.
- **dcp=2 vs dcp=4:** both FIT VMEM (~4.5 MB local-page buffer) and HBM (128K: dcp=2 ≈5.95, dcp=4 ≈2.98
  GiB/chip). The reported truncation was seen at dcp=4; since the CPU-reproducible logic is correct at BOTH,
  run dcp=2 first (conservative) but expect the SAME metal behavior — the differentiator is the on-metal glue,
  not dcp. No config shortcut; escalate to the on-metal dump.

### 2026-07-08 (later still) — on-metal discriminator pinned MULTI-CHUNK PREFILL; CPU write path EXONERATED

The pod discriminator (dcp=2, gate-ON passkey) narrowed the DCP failure precisely: **L=3200 FAILS at
max_batched_tokens=2048 (two prefill chunks) but PASSES at mbt=6144 (single chunk)**; single-chunk prefill is
correct to ≥5000 tokens / multi-block per shard. So the bug is specific to MULTI-CHUNK prefill (≥2 chunks) —
NOT the packed read, NOT decode, NOT the combine (all already CPU-exonerated). The prime suspect was the
owner-scatter KV write for chunk ≥2 (attention_interface.py:906-922), which starts at a nonzero global
position. I built the CPU reproduction and it **exonerates the write path on two independent counts**:

- **The owner-scatter arithmetic is CORRECT** (`tests/layers/common/test_mla_dcp.py::
  test_dcp_chunked_prefill_write_matches_single_chunk`, 7 cases): writing a prompt in TWO chunks via the REAL
  scatter (P(BATCH,CONTEXT) striped cache, 8 CPU devices) lands EVERY token in the same owner-shard slot as a
  single-chunk write — across mid-block chunk boundaries (nonzero offset inside a stripe), the token-sharded
  all_gather path (model>1, like the pod's model=8), unequal chunks, dcp 2/4, pack 1/2. NON-VACUOUS (asserts
  the scatter writes ≥ total slots). The mechanism the scatter would need to be buggy — a chunk-local index
  that resets to 0 — is reproduced ONLY when a chunk-local `seq_lens` is force-fed
  (`test_dcp_scatter_misroutes_iff_seqlen_is_chunk_local`), which is NOT what happens.
- **The runner feeds CORRECT metadata** (`tpu_runner.py:2531-2533`): attention
  `seq_lens = num_computed_tokens + num_scheduled_tokens` = the POST-chunk total (3200 for chunk 2), and
  `positions = num_computed + arange` (:2486) = global. So the scatter's recomputed
  `pos = seq_lens − q_len + local` = the true global position, and the kernel's causal mask (same `seq_lens`)
  is right too. No chunk-local value anywhere.
- **Read path also correct on CPU:** under DCP `chunk_prefill_size` is None, so chunked-prefill tokens route
  through the kernel's MIXED case (not the skipped PREFILL case) — already covered by the kvpack mixed-batch
  test (a q_len=9 / kv_len=70 chunk starting at pos 61) which matches the numpy full-context reference.
- **VERDICT:** the write (scatter + its metadata) and the decode/mixed reads are all CPU-correct for
  multi-chunk prefill. The real multi-chunk failure is therefore a METAL-ONLY effect in the multi-CALL
  composition — most likely the sharded-cache write-back/persistence between the two prefill steps (does
  chunk 1's P(BATCH,CONTEXT) scattered cache survive intact into chunk 2's step on metal?) or a native-op
  read during chunk-2 prefill — NOT the scatter arithmetic. On-metal disambiguation: dump the DCP cache
  AFTER the 2-chunk prefill (before decode) and diff against the single-chunk-prefill cache; if they differ,
  it is write-back/persistence (metal plumbing); if they match, it is the chunk-2 read/decode on metal.
- **Suites green (CPU, JAX_PLATFORMS=cpu):** DCP suites 86/86 (`test_mla_v2_lse_dcp_cpu` 3 + new kvpack 39 +
  `test_mla_dcp` 44); DSA+MLA 108/108 (`test_dsa_indexer_kernel`, `test_dsa_sparse_mla`, `test_mla_attention`,
  `test_mla_head_sharded`, `test_glm_dsa_pallas_decode`, `test_glm_dsa_mtp_index_share`). ≥ the -next counts
  (added a file; the kernel change is a no-op there).
- **EXACT pod validation (dcp=2, GLM_MLA_DCP=1, 128K passkey), bring-up gate first:**
  ```bash
  # relaunch with DCP baked into the raylet env:
  EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh
  cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
  # 5b bring-up: GSM8K n=32 at dcp=2 must reproduce the Stage-1 rows:
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
  HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
  REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
  GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_MLA_DCP=1 GLM_DCP=2 \
  ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 32 \
    --max-len 4096 --max-new 1024 --max-seqs 16 --num-gpu-blocks 0 --gmu 0.90 \
    --max-batched-tokens 512 --batch-size 0 --note "dcp2 bring-up GSM8K n=32" \
    > ~/glm-run/dcp2_gsm8k.log 2>&1
  # 5c the 128K cell at dcp=2 (page_g=1024 -> bkvc ~8.5MB < 16MB; ~5.95 GiB/chip KV, max_seqs 1):
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
  HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
  REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
  GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_MLA_DCP=1 GLM_DCP=2 \
  nohup ~/vllm-env/bin/python -u glm_longctx.py --lengths 131072 \
    --depths 0.25,0.5,0.75 --trials 8 --max-seqs 1 --gmu 0.90 \
    --note "passkey dcp2 128K" > ~/glm-run/passkey_dcp2_128k.log 2>&1 &
  ```
  Gate P = ≥95% per depth. Commit: fork `glm-5.2-v4-kvpack` (kernel interpret path + repro test).

## 2026-07-08 19:30 UTC — DCP BUG PINNED (on-metal): MULTI-CHUNK PREFILL, not the kernel; + fp8-KV route confirmed

- **On-metal discriminator (observability-first, suggestions.md):** dcp=2 gate-ON passkey —
  L=512 ✓, L=1600 ✓ (single prefill chunk); L=3200 ✗ at mbt=2048 (TWO chunks) but ✓ at mbt=6144
  (SINGLE chunk); L=5000 ✓ single-chunk. **The DCP failure is MULTI-CHUNK PREFILL** — KV written for
  prefill chunk >=2 (nonzero query_start_loc offset) is misrouted under the P(BATCH,CONTEXT) stripe.
  Single-chunk prefill under DCP is correct to >=5000 tok / multi-block. Fix target: the owner-scatter
  write (attention_interface.py:906-922) at nonzero chunk offset — CPU-testable (XLA scatter, not the
  native bitcast). Five CPU hypotheses (bitcast/position/blocktable/kvlen/combine) were all falsified
  first on an 8-device CPU dcp mesh — the bug was invisible to CPU because it needs multi-CHUNK prefill.
- **fp8-KV route (independent, agent-verified):** the fp8 MLA latent path pre-exists + is v4-validated;
  GLM_KV_CACHE_DTYPE=fp8 knob wired (default bf16 byte-identical). fp8 latent 6.1 GiB + 23 weights =
  29.15 < 30.75 → 128K@dcp=1 fits WITHOUT DCP (immune to the multi-chunk bug). Needle retrieval survives
  fp8 (97.5-100%); content precision degrades ~cumulatively (validate generation quality on pod).
- Two converging paths to 128K passkey: (A) fix DCP multi-chunk-prefill scatter [running]; (B) fp8-KV
  dense 128K @dcp=1 [pod now]. Sparse-128K gate then needs (A) OR fp8 indexer-cache too under (B).

## 2026-07-08 20:05 UTC — fp8-KV route hits a v4 Mosaic compile bug at 128K; DCP scatter fix is the cleaner path

- fp8-KV dense 128K@dcp=1: engine build FAILED — `Mosaic failed to compile TPU kernel: failed to legalize
  operation 'arith.cmpi'` in the fp8 dequant read (_upcast_kv_for_v4). The fp8-KV path was validated by the
  agent only at small v4-8 shapes; it does not compile at GLM's 128K/v4 config here. Checking whether it
  works at 8K (code-path vs size-specific). fp8-KV is a FALLBACK; the primary is the DCP fix.
- Primary path = fix the DCP multi-chunk-prefill owner-scatter (CPU-testable, clear target, agent running).

## 2026-07-08 20:20 UTC — fp8-KV fails at 8K too (same Mosaic arith.cmpi); DCP is the primary, cache-dump probe next

- fp8-KV 8K: SAME `arith.cmpi` Mosaic-legalize failure as 128K → it's a code-path bug in the fp8 dequant
  read (_upcast_kv_for_v4) on THIS v4 stack, NOT size-specific. fp8-KV is not a quick fallback (would need
  its own Mosaic-level kernel fix). Route retired for now.
- **Primary = DCP.** Its entire CPU logic is proven correct (scatter arithmetic 7 cases incl. mid-block +
  model>1 all_gather; runner metadata global; combine; bitcast). The multi-chunk-prefill failure is a
  metal-only multi-CALL effect — chunk-1's striped cache likely not persisting into chunk-2's step
  (input_output_aliases / P(BATCH,CONTEXT) write-back across scheduler steps). A gated cache-dump hook
  (2-chunk vs 1-chunk cache diff) is being built to pin write-back-vs-read in one pod run — the disciplined
  observability step before any fix.
- Not reward-hacking / not faking: report_passkey still refuses to call <128K a gate pass; no 128K claim
  until it's real. Proven so far: sparse passkey 100% @32K, kernels silicon-validated, GSM8K n=32 96.9%.

## 2026-07-08 21:10 UTC — DCP ROOT CAUSE CONFIRMED (cache-dump verdict): cross-step striped-cache persistence loss

- The gated cache-dump probe (2-chunk vs 1-chunk prefill, dcp=2, layer 0, all 8 hosts' shards reassembled)
  returned **DIFFER: max|Δ|=5.44, exactly 1024/2048 rows stale = precisely ONE dcp stripe (half at dcp=2).**
- **Root cause, empirically confirmed:** chunk-1's writes to the P(BATCH,CONTEXT)-striped MLA KV cache are
  NOT carried into chunk-2's execute_model step — one dcp shard is lost across the scheduler-step boundary.
  This is the exact "sees 1/dcp of context" symptom. It is a RUNNER cache-persistence / input_output_aliases
  issue under DCP striping — NOT the kernel (all kernel/scatter/combine CPU logic already proven correct).
- Single-chunk prefill has one step → no boundary → correct (matches the on-metal discriminator). This is why
  every CPU test (single execute_model call) passed and only multi-CHUNK on metal fails.
- Fix target: preserve the DCP-striped kv_caches buffer across chunked-prefill execute_model calls (aliasing/
  donation round-trip of the striped layout). Then dcp=2 128K passkey. Observability-first paid off again:
  the dump pinned write-back-vs-read in one probe after 5 CPU hypotheses were falsified.

## 2026-07-08 21:55 UTC — Independent review caught a MISTARGETED fix (saved a pod cycle); redirected to GLM's real path

- The DCP persistence fix (get_kv_cache_out_sharding in get_flax_model) is mechanistically CORRECT but
  on the WRONG code path for GLM. Independent adversarial review FINDING 1 (HIGH), verified: GlmMoeDsa is
  in _VLLM_PREFERRED_ARCHITECTURES (model_loader.py:52-76) → served via get_vllm_model/VllmModelWrapper,
  NOT get_flax_model. The fix + its CPU test exercise a path GLM never takes → NO-OP for GLM. Reviewer 1
  said "SHIP" (correct about code quality) but missed the path; reviewer 2 traced the chain of custody and
  caught it. Without the review we'd have run an ~18-min pod build to "validate" a no-op.
- REDIRECT: same mechanism (donated striped cache out-sharding mismatch), correct location =
  VllmModelWrapper.jit_step_func (step_fun_jit + draft_step_fun), whose cache out_shardings=None → XLA
  picks a layout that won't match P(BATCH,CONTEXT) → same reshard + one-stripe donation drop. Fix: set the
  vLLM step-fn cache out_sharding to the DCP-striped spec under the gate. Agent redirected; CPU test must
  exercise the vLLM wrapper step fn, not get_flax_model. The flax fix stays (valid for the flax MLA path).
- The observability guards (glm-5.2-v4-obsguard, 18 tests) are done and merge-ready alongside the real fix.
- Discipline held exactly: no merge of an unreviewed aliasing change; the review is a HARD gate and it paid.

## 2026-07-09 00:45 UTC — DCP persistence FIXED (A/B/C proves it); retrieval still fails → read/content; fp8-KV re-blocked

- **A/B/C on-metal localizer (dcp=2, 2-chunk):** A(postfwd.step1)==B(prefwd.step2)==C(postfwd.step2),
  max|Δ|=0, BOTH stripes fully populated (|sum| 25195/25131). → the out-sharding fix (merged) CURED the
  stale-stripe persistence bug: carry clean, cache temporally consistent. But 2-chunk retrieval STILL
  fails (pred=None) → remaining bug is NOT persistence. Now distinguishing (1) cache populated-but-WRONG
  vs (2) READ path, via the logical-position 2chunk-vs-1chunk correctness diff (dcp_cache_diff.py, merged).
  CPU/write evidence leans MATCH→read-path. 1-chunk reference dump running.
- **fp8-KV RE-BLOCKED:** after the fp8→f32→bf16 rewrite merged, the engine STILL fails with the identical
  `arith.cmpi` (%11167) at 8K — so the cmpi is NOT the astype; the fp8→float lowering on v4 emits it
  regardless of cast form (the CPU Mosaic-routing test wasn't faithful to real v4 libtpu). Deeper Mosaic
  limitation; fp8-KV deprioritized (future: dump the real failing MLIR to find the true cmpi source, or a
  manual bitcast dequant). DCP is the closer path.

## 2026-07-09 01:20 UTC — DCP localized to the WRITE: owner-scatter corrupts the 2nd logical page

- Correctness diff (2-chunk vs 1-chunk post-prefill cache, logical-position, bf16-cast fixed): **DIFFER,
  max|Δ|=5.44, exactly 1024 positions wrong starting at position 1024** (= the 2nd logical page; P_g=1024
  = block_size 512 × dcp 2). Page 0 (0..1023) correct; chunk-2 region (2048..3186) correct; ONLY 1024..2047
  wrong. A/B/C showed it's written-wrong-and-stable → the bug is chunk-1's OWN owner-scatter WRITE, not the
  read or the carry. NOT persistence (that was fixed), NOT read path.
- Root cause narrowed to the DCP owner-scatter (attention_interface.py:902-922) per-page position/owner
  arithmetic for a MULTI-PAGE prefill chunk. CPU-reproducible (XLA scatter). Agent fixing; the earlier
  scatter CPU test missed this exact geometry (2048-tok/2-page chunk).
- The correctness-diff tool (dcp_cache_diff.py) needs a bf16 cast in reassemble_layer (its tests used
  float32); I ran the diff inline with the cast. Fold the cast into the committed tool.

## 2026-07-09 02:45 UTC — DCP bug LOCKED: the owner-scatter mislowers on TPU (kernel exonerated)

- **GLM_DCP_SCATTER_ONLY probe (kernel SKIPPED, pure owner-scatter output): DIFFER @ page 1** — max|Δ|=5.44,
  1024/3187 wrong, first_diff=1024, identical to the full-path signature. So with the kernel out entirely,
  the 2-chunk scatter output is STILL wrong at the 2nd local page. Kernel EXONERATED.
- Combined with the CPU ground-truth (arithmetic correct, 264 adversarial cases can't reproduce it), the
  root cause is LOCKED: the owner-scatter `.at[...].set(mode="drop")` (attention_interface.py:915-931)
  MISLOWERS on TPU when writing into the DONATED, P(BATCH,CONTEXT)-sharded, TILED cache buffer — the 2nd
  local-page tile isn't committed. A metal-only XLA/GSPMD scatter-into-sharded-donated-buffer interaction.
- Fix directions handed to the agent: (1) drop the donation for the DCP scatter, (2) per-page
  dynamic_update_slice, (3) shard_map-LOCAL scatter (each shard scatters its local slice — matches the
  kernel's per-shard read), (4) mode=promise_in_bounds / segment-sum. On-pod scatter-only diff = the exact
  DIFFER→MATCH falsifier. Observability chain: multi-chunk → persistence(fixed) → A/B/C(carry ok) →
  correctness-diff(page 1 write) → scatter-only(scatter EXEC, not kernel). Each probe halved the search.

## 2026-07-09 04:15 UTC — DCP bug is UPSTREAM of the scatter (onehot fails too): the new-KV VALUES are wrong

- Full-path 2-chunk fix attempts ALL fail (pred=None): GLM_DCP_NO_DONATE, GLM_DCP_SCATTER_IMPL=flat, and
  GLM_DCP_SCATTER_IMPL=onehot. onehot is scatter-primitive-FREE → the WRITE is exonerated. The pre-scatter
  new-KV VALUES for the 2nd-page tokens (positions 1024..2047) are already corrupt.
- New root-cause locus: the MLA forward's new-KV (kv_c/k_pe) for a MULTI-PAGE prefill chunk under DCP ×
  the 32-way token all_gather (MLP_TENSOR cross-shard q/k gather) × dcp — NOT the cache write. All the
  write-side levers (no_donate/flat/onehot/shardlocal) are dead ends because the data is wrong before them.
- Next probe: GLM_DCP_DUMP_NEWKV (dump the gathered new-KV feeding the scatter, 2chunk-vs-1chunk logical
  diff) to confirm page-2 values differ + split all_gather-ordering vs projection/positions. Observability
  chain: multichunk → persistence(fixed) → A/B/C(carry ok) → correctness(page-1 write) → scatter-only
  (scatter exec) → onehot(NOT the write) → upstream new-KV values. Each probe eliminated a layer.

## 2026-07-09 08:35 UTC — DCP write bug is packing-INDEPENDENT + new-KV values CONFIRMED correct; audit's decoupling + 128K fit-check

- **new-KV values are CORRECT (correction to the 04:15 entry):** the warmup-skip probe fix (ONLY_PREFILL)
  captured the REAL 2048-token chunk-1 (dist=[0,0,1], val.shape=(2048,640), pos 0..2047); the 2chunk-vs-1chunk
  new-KV `val` diff over all 2048 owned positions = **0 (MATCH)**. So the "values wrong upstream" call (which
  rested on onehot also failing) was WRONG — onehot failed for another reason. The corruption is the physical
  WRITE into the kv_packing-packed, CONTEXT-sharded cache's 2nd tile on metal, downstream of correct values.
- **Packing-INDEPENDENT:** MLA_KV_PACKING_SIZE=2 (min valid for bf16) 2-chunk passkey ALSO fails (pred=None),
  same as 32. kv_packing=1 rejected (bf16 needs >=2). So it's not a packing-size artifact — it's the
  fundamental CONTEXT-sharded packed WRITE across the tile boundary on TPU (CPU can't test: interpret asserts
  kv_packing==1).
- **Independent audit (adopted):** blockers are ALL in the KV/cache layer, NOT the DSA kernel (which passed
  silicon). GSM8K n=32 relabeled SMOKE (Wilson ~84-99%, not "scale"). fp8-KV SHELVED (2 v4 Mosaic cmpi =
  systematic). Passkey@128K (correctness, batch=1) vs throughput@256K (batch/DCP/fp8) = decoupled gates.
- **Replication CONFIRMED empirically:** the dcp=2 engine sized a 65,536-token KV pool at ~6.5 GiB/CHIP
  (23.06/30.75 model, 7.69 free) — the MLA latent cache is replicated per-chip, so 128K = ~12-13 GiB/chip >
  free → needs DCP or fp8. The audit's pod-pool (270 GB) math doesn't apply. Running the explicit
  128K@dcp=1/bf16/batch=1 fit-check to settle it with a real OOM (or a surprise).

## 2026-07-09 09:20 — 128K@dcp=1 fit-check: the wall is FRAGMENTATION, not capacity (KICKOFF premise corrected)

The audit's decoupling hypothesis (128K correctness may not need DCP) was tested with 4 on-pod
fit-checks. Result overturns a load-bearing KICKOFF claim.

**Runs:**
- (a) gmu0.95 / override260 / chunk2048: KV pool sized **132,862 tokens** (>=128K → FIT); failed
  allocating a 162M buffer.
- (b) gmu0.88 / override256 / chunk2048: rejected — pool 131,072 < max_len 131,840 ("serve one request").
- (c) gmu0.92 / auto / chunk512: KV pool auto-sized to **1,796,703 tokens** (!); failed allocating 2.15G.
- The (c) error is definitive: *"Attempting to allocate 2.15G. There are **5.54G free**. The largest
  contiguous region is **2.09G due to fragmentation**."*

**Corrected accounting (empirical):** the auto-pool held **1.79M tokens** in the free space → 128K
tokens of MLA latent costs only **~150 MiB/chip**, NOT the 6.2–11.9 GiB the KICKOFF asserted
("replicated per-chip → 128K needs DCP/fp8"). That premise was WRONG for the correctness (batch=1)
case. There is **5.54 GiB free** with a 128K prompt — capacity is a non-issue.

**The true blocker for 128K dense@dcp=1:** HBM **fragmentation**. The oversized auto KV pool carves the
arena so the 2.15 GiB attention compile-scratch can't find a contiguous slot (2.09G largest vs 2.15G
needed — short by 60 MiB of *contiguity*). Fix = right-size the pool (cap num_gpu_blocks ~300 blocks ≈
150K tokens) so it stops fragmenting + lower gmu so more physical HBM stays unreserved/contiguous for
the scratch. Run (d) tests this: gmu0.90 / override300 / chunk512.

**Implication for the gate:** 128K passkey CORRECTNESS decouples from the blocked DCP-write and
fp8-KV(2×cmpi) paths — it's a fragmentation-tuning problem on the dense path, not a cache-capacity
wall. DCP/fp8 remain needed only for the *throughput@256K / large-batch* gate (many concurrent seqs),
which is a genuinely different resource regime. KICKOFF §"Why the cache is the blocker" to be rewritten.

## 2026-07-09 11:15 — 128K fragmentation: tuning table + XLA compile-scheduler flags (the real lever)

Followed the fragmentation diagnosis (prev entry) with a controlled sweep. **The failing buffer is a
compile-scratch that scales EXACTLY as 0.625 MiB × num_gpu_blocks** (187.5M@300blk, 168.75M@270,
161.25M@258). 128K needs ≥258 blocks (258×512=132,096 ≥ max_len 131,840), so the buffer floor at 128K
is ~161M and CANNOT be shrunk by trimming the pool (bounded by max_len). The fix must GROW the largest
contiguous free region past ~161M.

**Sweep (all dcp=1, bf16, batch=1, dense-MLA, DSA off):**
| config | contiguous free | buffer | gap |
|---|---|---|---|
| gmu0.90 pool270 chunk512, no flags | 118M | 168.75M | −51M |
| gmu0.80 pool270 chunk512, no flags | 171M (noisy) | 168.75M | ~0..−48M |
| gmu0.77 pool270 chunk512, no flags | 128M | 168.75M | −41M |
| gmu0.80 pool270 chunk512, FLIGHTREC off | 120M | 168.75M | −48M |
| **rerun=5 + rwb_fusion=false, pool258 chunk256** | **147.76M** | **161.25M** | **−13.5M** |
| scheduler=false + rerun=5 + rwb_fusion=false, pool258 chunk256 | 105M (WORSE) | 161.25M | −56M |

**Findings (empirical, on-pod):**
1. `gpu_memory_utilization` is mechanically INERT for fragmentation once `num_gpu_blocks_override` is set
   (0.77 gave more total-free but LESS contiguous than 0.80 → allocation ORDER governs, not total free).
   Confirmed the research agent's account (gmu is a vLLM KV-budget cap, not an allocator setting).
2. Flight recorder is NOT the fragmenter (off → slightly worse). Rejected cleanly.
3. **`LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"` (scheduler ON)
   is the best lever so far: contiguous 118→147.76M.** Baked into all 8 raylets via EXTRA_ENVS (verified
   in /proc/<raylet>/environ); env_override.py:22 prepends `--xla_tpu_use_dynamic_smem_negotiation=true`.
4. **Disabling the latency-hiding scheduler makes it WORSE (147.76→105M)** — rerun=5 needs the scheduler
   ENABLED to reduce the reservation. Rejected.
5. Still 13.5M short at the best config. Next decisive lever under investigation: **raise KV block_size**
   (512→1024) → halves num_blocks → buffer ~80M ≪ 148M contiguous (CPU agent verifying kernel safety).

**Harness robustness fix:** two earlier runs (fit64, fit128g) were SIGTERM'd after engine-build but
before generation (driver was a child of the tool-shell; teardown killed the process group). Now launch
the driver via `setsid nohup ... </dev/null` — its own session survives; DRIVER_EXIT is captured.
NOTE: the "64K PASS" I briefly read earlier was a FALSE grep match on the substring in `hlo_passes.cc`
log lines, NOT needle results — the 64K engine BUILT clean at dcp=1 (no OOM, confirming the buffer model)
but the needle was not captured. A proper 64K verify (setsid) is running now.

## 2026-07-09 12:25 — 64K passkey VERIFIED 100% @ dcp=1/bf16 (ceiling 32K→64K); GLM_MLA_ALIAS_KV fix for 128K

**64K dense passkey, dcp=1, bf16, NO DCP/fp8, batch=1** (pool 140 blocks, chunk 256, gmu 0.90, best
anti-frag flags), run_id=100, setsid-hardened driver ran clean (DRIVER_EXIT=0):
- depth 0.25: 2/2 (pred 487136/614386 == gold)
- depth 0.50: 2/2 (pred 669915/971123 == gold)
- depth 0.75: 2/2 (pred 208080/217556 == gold)
- **6/6 = 100%**, avg_prompt_tok 65,211. First VERIFIED passkey above 32K. Extends the proven ceiling
  32K→64K on real hardware; confirms the buffer-scaling model (64K → ~84M scratch < ~148M contiguous).

**128K root-cause candidate — GLM_MLA_ALIAS_KV** (fork a248bd6b0, reviewed SAFE-TO-TEST): the 161 MB
warmup transient is one bf16 MLA KV layer allocated FRESH because, with the donated replicated cache and
step-fn `out_shardings=None` (vllm_model_wrapper.py), XLA may pick a cache output layout that can't bind
the mla.v2 pallas input_output_alias → fresh full-cache output. Gate pins the dcp=1 MLA cache out-sharding
to P(BATCH) (== replicated at dcp=1, byte-identical; matches mla_attention's dcp-off return). Adversarial
review: byte-identical off, numerically identical on at dcp=1; refined P(BATCH,CONTEXT)→P(BATCH) to avoid a
wasteful reshard on a dcp>1/DCP-off mesh. EFFICACY UNPROVEN (out_shardings pins sharding not layout) — the
metal A/B decides: flags-only FAILED at −13.5M (161.25M buffer vs 147.76M contiguous); if flags+alias now
BUILDS, the alias eliminated the 161M request. Testing now.

## 2026-07-09 12:50 — GLM_MLA_ALIAS_KV is INERT for the 161M copy (honest null); next robust fixes

128K A/B (same config, only GLM_MLA_ALIAS_KV added): the **161.25M buffer PERSISTS** ("Attempting to
allocate 161.25M … 157.50M contiguous"). The sharding-pin did NOT eliminate the fresh cache output —
exactly the reviewer's efficacy caveat (out_shardings pins SHARDING, not physical LAYOUT; donation
aliasing needs a layout match too). Contiguous rose 147.76→157.50M (allocation-order side effect), so
still ~3.75M short — but chasing that with fragmentation noise would be a FRAGILE, non-reproducible pass
(violates 3/3 robustness + no-reward-hacking). NOT claiming the gate. Gate kept (default off,
byte-identical); it stands as a documented no-op pending the layout escalation.

Robust root-cause options now (the 161M = one bf16 MLA KV layer, irreducible at bf16/128K unless the
fresh copy is bound or the layer is halved):
1. **Layout-pin** — bind the donation with jax layout control (Format/DLL on the step-fn output, or
   jit-level input_output_aliases), so the kernel cache output reuses the donated input → 161M vanishes.
   The reviewer's named escalation for an inert sharding-pin. (agent investigating)
2. **fp8-KV** — halves the layer to ~80M → fits with ~70M margin. Shelved on a 2nd v4 Mosaic arith.cmpi
   in the write/quantize path; re-examining whether it's clearable branchless like the read path. (agent)
3. Accept 64K as the demonstrated dcp=1 ceiling; 128K via DCP (write bug) for throughput.

## 2026-07-09 13:15 — 96K also frag-fails; contiguous-free is a LOTTERY (~114–157M) → 128K needs a robust fix

96K probe (pool 200, chunk 256) confirmed the buffer model EXACTLY: 200 blocks × 0.625 MiB = **125.0M**
buffer requested. But it FAILED — this run's largest contiguous was only **114M** (vs 147–157M on the 128K
runs). So the largest-contiguous-free is **run-to-run variable (~114–157M)** — a fragmentation lottery, not
a fixed floor. Implications:
- **Reliable dcp=1/bf16 ceiling ≈ buffer < ~114M → <182 blocks → ~88–90K context.** 64K (87.5M buffer)
  builds reliably; 96K (125M) exceeds the unlucky floor; **128K (161M) is far above even the lucky 157M max.**
- **128K CANNOT be reached by scheduler/gmu tuning** (161M ≫ best-ever 157M contiguous). It needs a robust
  buffer fix: (1) ELIMINATE the fresh 161M cache copy (layout-pin donation aliasing), or (2) HALVE it via
  fp8-KV (161→80M). Both under active CPU investigation. The marginal-tuning route is CLOSED.
- Verified ceiling stands at **64K (6/6, 100%)**. Solid deliverable independent of the 128K outcome.

## 2026-07-09 13:40 — Two robust 128K fixes designed (source agents); primary = L2 donation-chain

Two independent CPU agents (read-only, no TPU) traced the 161M-copy root cause and a backup:

**PRIMARY — donation-chain gap (attention_interface.py:1146).** The mla.v2 cache write is meant IN-PLACE
via a 3-link donation chain: L1 step-fn donates kv_caches (vllm_model_wrapper.py:928), L3 kernel wrapper
donates cache_kv (kernel.py:3085), pallas MUST-aliases operand→output (kernel.py:2883,2933). But the MIDDLE
jit(shard_map) (L2, attention_interface.py:1146) did NOT donate its cache arg → XLA is forced to COPY the
161M cache fresh each step to honor L2's preserve-contract. This is why the earlier L1 out-sharding pin was
inert (it fixed sharding, not the L2 donation). **Fix: gated `donate_argnums=(4,)` at L2** (GLM_MLA_ALIAS_KV,
dcp-off only). Byte-identity OFF proven on CPU (md5-identical HLO to omitting it). Applied to fork; focused
adversarial review of donation correctness (index, use-after-donate, MTP/multichunk) in flight before pod.

**BACKUP — fp8-KV, and the shelving was based on an INFERENCE not an observation.** Agent found the "2nd
write cmpi" was MISATTRIBUTED: the KV bf16→fp8 quantize runs in XLA (quantize_kv clip = max/min, cmpi-free),
NOT in Mosaic. The only remaining in-Mosaic fp8 convert is the attention OUTPUT cast (kernel.py:2323), fp8
ONLY because Q-activation-quant defaults ON (q_dtype=fp8). Fix: keep Q/output bf16 on v4 via
`DISABLE_MLA_Q_ACTIVATION_QUANTIZATION=1` (+ a latent q_scale over-scale bugfix) — KV cache stays fp8 (the
161→80M win). CRUCIAL: fp8-KV has NEVER been rebuilt on v4 since the branchless read fix merged (8802ebab7)
→ "still 2 cmpi" is unproven; it may already work. This is the clean capacity halving if the donation fix
proves inert. CPU structural check: 0 surviving f8 float-convert ops in the isolated Mosaic module.

## 2026-07-09 14:00 — Both aliasing fixes INERT on metal → pivot to fp8-KV (concrete robust land)

**bf16 aliasing route exhausted (honest):** L1 out-sharding pin AND L2 donate_argnums=(4,) BOTH inert —
the 161.25M cache copy persists on metal unchanged (157.50M contiguous). Two carefully source-derived,
adversarially-reviewed, byte-identical-off fixes, both no-ops for the copy. Classic suggestions.md blind
spot: source reasoning ≠ the real HLO. Added the GLM_DUMP_STEP_HLO instrument, but the model-forward step
fn compiles INLINE during warmup (AOT lower skipped for nested-jit bodies) so it bypasses the compile hook
(only sample/rng/logits helpers dumped). Full model-forward HLO needs the XLA_FLAGS firehose — deferred.
The L1+L2 donation change is numerically SAFE (8K passkey 1/1 correct with GLM_MLA_ALIAS_KV=1) and
byte-identical off, so it's kept (gated) as a no-op pending the HLO.

**PIVOT: fp8-KV (Agent 2) — the concrete robust land.** Halves the per-layer cache 161→80M (< the ~114M
lottery floor → 128K builds). Implemented the Q-bf16 fix (flash_attn_mla.py): on
DISABLE_MLA_Q_ACTIVATION_QUANTIZATION=1, keep Q/output bf16 (removes the ONLY remaining in-Mosaic fp8
convert — the output cast) + move q_scale inside the quant branch (q_scale=None when Q unquantized, else
the kernel over-scales — a latent bug fixed). KV stays fp8. CPU structural check CONFIRMS: Q→bf16 output
cast = 0 surviving f8 float-converts (vs 1 for the old Q→fp8) → the v4 legalizer never sees the blocker.
Byte-identical when GLM_KV_CACHE_DTYPE unset. Adversarial numerics review in flight; then 128K fp8 build
(pool 258, chunk 256, gmu 0.90, DISABLE_MLA_Q_ACTIVATION_QUANTIZATION=1 + GLM_KV_CACHE_DTYPE=fp8). Gate is
"passkey ≥95% @128K" — fp8-KV retrieval survives (agent: 97.5-100%), a legitimate long-context config.

## 2026-07-09 14:15 — fp8-KV: the REAL write cmpi located via MLIR loc → pack_new_kv i8 lax.select

fp8-KV 128K build FAILED: `Mosaic failed to compile ... failed to legalize 'arith.cmpi'
(vector<8x128x4xi8>, predicate=ne)`. MLIR source loc is exact:
`loc(select_n(pack_new_kv.<locals>.merge_loop_body kv_utils.py:204:25))`. So the Q-bf16 fix DID clear
the output-cast fp8 convert (agent 2 correct), but a DIFFERENT cmpi remains: the `lax.select`s in
`pack_new_kv`'s merge_loop_body (kv_utils.py 191/199/204/205/213/224) run on the packed registers in
their NATIVE dtype — for bf16 (16-bit) Mosaic legalizes the select; for fp8 (i8) it emits an i8
`arith.cmpi ne` the v4 backend can't legalize. THIS is why bf16 built @64K but fp8 didn't — it's a
`select` on packed fp8 bytes, NOT a float convert. (CPU can't reproduce the v4 cmpi in a toy; agent-2
caveat holds — verification is numeric-parity + structural + the pod build.)

Fix direction: eliminate sub-16-bit lax.select in pack_new_kv — bitcast the i8 operands to uint32
(shift_roll already does this internally) for the select, back after; masks are constant within each
packed word so semantics are preserved; MUST stay bit-identical for bf16 (shared kernel). Reference
`pack_new_kv_reference` (kv_utils.py:267) gives the numeric oracle. Launching an ultracode workflow:
parallel fix candidates (bitcast-widen / branchless-mask / uint32-merge), each numeric-parity-verified
vs the reference (bf16 AND fp8, interpret mode) + adversarially reviewed, before the pod build.

## 2026-07-09 14:25 — bf16 in-place copy-elimination is a DEAD END (definitive); fp8-KV is the robust lever

Backup CPU agent (faithful HLO repro) settled the bf16 route: the KV donation chain is NOT severed — it
BINDS at the JAX level (outer list-donation → per-element may-alias, 0 copies through nested pjit +
torchax list getitem/setitem). The torchax boundary was a red herring. Both prior fixes were
ARCHITECTURALLY inert, provably: (1) L2 inner-jit donate_argnums materializes NOTHING at the outer
executable boundary (only the OUTERMOST donation counts, already at L1) — CPU-proven; (2) L1 out_shardings
pins SHARDING not LAYOUT, and at dcp=1 input & output are both replicated so there's no mismatch to fix
(this is why the same pin WORKED for dcp>1 striped but not dcp=1). The residual 161M is a TPU-only XLA
buffer-assignment DECLINE of a may-alias (physical layout) — no Python source line. Only untried
copy-elim lever = end-to-end jax.experimental.layout Format pin (speculative, fragile, metal-only, "long
shot"). CONCLUSION: the robust 128K bf16 levers are cache-SIZE reduction (fp8-KV / DCP / block_size), NOT
copy elimination. → Validates the fp8-KV pivot. The GLM_MLA_ALIAS_KV L1+L2 changes stay (gated, byte-id
off, 8K-correct) as documented no-ops; not the fix. Layout-pin kept as a one-shot last resort only.

## 2026-07-09 15:20 — CRITICAL PROCESS BUG: workers were STALE (8802ebab) all night → fix-tests invalid

Discovered via the MLIR loc: fp8 build kept failing at `kv_utils.py:204` even AFTER the pack_new_kv fix —
but current line 204 is an iota, the fixed select moved to 234. Root cause: the 8 hosts have PER-HOST
local ~/tpu-inference checkouts (ext4, not shared). Only worker 0 (this VM) had tonight's commits; workers
1-7 were all stuck at **8802ebab** (verified: `head=8802ebab, _dtype_safe_select count=0`). Compounding:
my `git push -q` (no args) was NOT updating origin/glm-5.2-v4-next either (origin stuck at 8802ebab until
an explicit `git push origin glm-5.2-v4-next`). So EVERY on-pod test of a worker0-only fix tonight (L1
sharding pin, L2 donation, fp8 Q-bf16, kv_utils) actually ran STALE 8802ebab worker code — the fixes never
executed on the pod. (64K/bf16 results remain VALID — they need no recent commit. The bf16 aliasing
dead-end verdict also stands — it's CPU-proven architecture, not the on-pod inert runs.)

**FIX + STANDING RULE:** before ANY pod test of a fork change, `git push origin glm-5.2-v4-next` THEN
`TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh` and confirm all 8 hosts show
the SAME hash. Done now: all 8 @ **8a76ae5e3**. Re-running fp8 128K with the pack_new_kv i8-select fix
actually deployed. This footgun (memory: "cross-host drift causes silent divergence") likely explains
several "inert" results — re-validate any worker0-only fix that mattered.

## 2026-07-09 16:15 — fp8-KV kernel COMPILES on v4 (i8 cmpi+muli fixed); 128K now HBM-capacity bound

Milestone: with workers synced (0f15e3ad0) the fp8-KV MLA kernel FULLY COMPILES on v4 — the pack_new_kv
i8 select_n cmpi AND the mask i8 arith.muli are both cleared (KV cache sizes to 131,840). fp8-KV is no
longer Mosaic-blocked. Remaining: a compile-time HBM OOM at 128K. Breakdown (deepsea_compiler_util):
arguments **28.90G** (FP8 weights ~23 + fp8 KV pool ~5.9G, in/out shared via donation) + program **2.15G**
(overlays **2.05G**) + reserved **1.25G** = ~32.3G > 30.75, over by ~0.3–1.5G (varies by program variant).
fp8 DID halve the KV vs bf16 (bf16 KV would be ~11.8G → weights+KV alone 34.8G, infeasible), but
program+reserved (3.4G) overhead eats the margin. The 2.05G "overlays" = compiled program code, plausibly
one variant per token bucket (TPU_MIN_TOKEN_BUCKET=32 → [32,64,128,256]=4 variants). Testing
TPU_MIN_TOKEN_BUCKET=256 + chunk256 (1 bucket) to cut overlays ~1.5G. If insufficient, 128K@fp8/batch=1 is
genuinely HBM-bound and needs DCP (fp8+dcp=2 → KV/2 → fits, but the DCP multi-chunk packed-write bug) or a
smaller footprint. fp8 at a shorter ctx (fits) validates the path + extends the ceiling regardless.

## 2026-07-09 16:40 — fp8-KV @ dcp=1 ceiling is ~80–122K (overlays fragmentation + capacity); 128K NEEDS fp8+DCP=2

Definitive HBM analysis (agent, byte-reconciled). The fp8-KV MLA kernel compiles + sizes on v4 (i8
cmpi+muli fixed). Two structural limits cap fp8/dcp=1 below 128K:
1. **Capacity @128K:** args 28.90G (weights 22.76 + fp8 KV 6.14G @258blk) + overlays 2.05G + reserved
   1.25G = ~32.3G > 30.75 → over ~316M. The fp8 KV can't shrink below 256 blocks (128K prompt). Weights
   fixed. **overlays 2.05G = the UNROLLED 78-layer backbone's machine code — NO flag shrinks it** (not
   per-bucket: TPU_MIN_TOKEN_BUCKET=256 confirmed inert; only a lax.scan model rewrite would, risky).
   **reserved 1.25G = fixed libtpu carve-out** (no flag). gmu inert w/ num_gpu_blocks_override.
2. **Overlays-fragmentation below 128K:** @112K (232blk) args 28.28G fits but the 2.05G overlays buffer
   finds no CONTIGUOUS slot ("2.10G free, largest contiguous 1.5G"). So even <122K fails on overlays
   fragmentation until the KV pool is small enough (~<175blk / ~87K) to leave 2.05G contiguous. Realistic
   fp8/dcp=1 ceiling ≈ 80–96K (lottery).

**CONCLUSION — the ≥128K gate REQUIRES fp8 + DCP=2** (or DCP alone): sharding the replicated MLA cache
halves per-chip KV (6.14→3.07G → args ~26G, +2.5G margin) — fits capacity AND defrags the overlays. It's
ALSO required for the 256K throughput gate. The blocker is the KNOWN **DCP multi-chunk-prefill packed-WRITE
bug** (2nd kv_packing tile mis-commit, owner-scatter attention_interface.py:951-1078; observability wired:
GLM_DCP_SCATTER_IMPL flat/barrier/onehot + GLM_DCP_DUMP_NEWKV DIFFER→MATCH falsifier). This is now THE
critical path to 128K. fp8-KV kernel fixes (pack_new_kv) are a real standing contribution regardless.

## 2026-07-09 18:25 — DCP multi-chunk bug CONFIRMED REAL on synced workers (E1); pageloop fix next

With all 8 workers CERTIFIED synced (0f15e3ad0), bf16 + GLM_MLA_DCP=1 + GLM_DCP=2 at 16K multi-chunk
(chunk 256, pool 64) ran to clean DRIVER_EXIT=0 but the needle FAILED: pred=None gold=578768 correct=False
(garbage — needle at depth 0.5 ≈ pos 8000 lands in the corrupted pos-1024+ region). So the DCP
multi-chunk owner-scatter bug is GENUINE, not a stale-worker artifact — this is the trustworthy
re-localization (agent E1) the earlier confounded overnight probes lacked. (DCP=2 32K first attempt OOM'd
on auto-pool over-sizing 3M tokens → capped pool 64 fixed that.)

DCP agent audit: 8802ebab (07-09 06:54) CONTAINS all DCP code (persistence, SCATTER_IMPL variants, probes
— all ancestral); the overnight VARIANT sweeps (flat/onehot/barrier) were likely SPMD-confounded (Franken
mix across hosts) — one conclusion already retracted. The runbook's "top candidate" shardlocal is ORPHANED
on branch dcppersist2, NOT on glm-5.2-v4-next/workers. Corrected geometry: the corrupt region is the 2nd
P_g=1024 PAGE (dim-0 tile), NOT a kv_packing=32 tile — `cache.at[_page_safe,_row,_sub,:].set(_val,
mode=drop)` (attention_interface.py:1077) writes TWO physical pages in one scatter; the 2nd dim-0 page tile
mis-commits into the donated/sharded/tiled buffer. Fix #1 = **pageloop**: per-physical-page
lax.dynamic_update_slice (different XLA op class; one page-tile/op). Implementing + CPU-verifying vs
test_mla_dcp_scatter_gt.py before pod. THIS is the ≥128K gate blocker.

## 2026-07-09 19:10 — pageloop write-fix INERT on metal → DCP bug is NOT the multi-page scatter write

bf16 DCP=2 16K multi-chunk with GLM_DCP_SCATTER_IMPL=pageloop: needle STILL pred=None correct=False —
IDENTICAL to the default scatter. pageloop was CPU-verified bit-identical to default across 20+ geometries
(incl. exact pod tiling, dcp=2/4, bf16+fp8) and uses a genuinely different XLA op class (per-page
dynamic_update_slice, not multi-tile scatter) — yet metal is unchanged. So the "2nd-page mis-commit via
multi-tile scatter" hypothesis is FALSIFIED. pred=None (total garbage) points at the read/combine or carry
path, not the write. Next: the write-vs-value / single-vs-multi-chunk discriminator (E3), which the earlier
overnight localization (contaminated by stale workers) got wrong. Testing single-chunk DCP=2 first.

## 2026-07-09 19:40 — DCP=2 fails SINGLE-chunk too → bug is FUNDAMENTAL (read/combine), not multi-chunk write

Observability (not needle-guessing) reframed the DCP bug: bf16 DCP=2 SINGLE-chunk 4K (chunk 4096, one
prefill; 4080 tok = 4 logical pages at P_g=1024) needle pred=None correct=False — DCP is broken at ANY
>1-page context, NOT just multi-chunk. The overnight "single-chunk always correct" was a STALE-WORKER
artifact (those tests likely used <=1-page prompts or confounded SPMD). This EXPLAINS pageloop being inert:
a per-page WRITE fix can't help a READ/COMBINE bug. Prime suspects now: the per-dcp-shard strided-position
causal mask, per-shard kv_lens, or the cross-dcp LSE (log-sum-exp) softmax merge (attention_interface.py
~1099-1155) — all in the DCP READ path, NOT the owner-scatter write. Next (observability-first, correct
reference = DENSE not 1chunk): cache-dump DCP vs DENSE at a 2-page context — MATCH ⇒ write correct ⇒
read/combine bug; DIFFER ⇒ write. pageloop kept (CPU-correct, gated) but is NOT the fix.

## 2026-07-09 20:10 — DCP bug is METAL-ONLY (packed/tiled read of context-sharded cache); CPU logic PROVEN correct

DCP-read agent built 3 CPU harnesses with the REAL mla.v2 kernel (interpret) at the exact failing geometry
(single-chunk 1/2/4/8 logical pages, dcp=2, prefill+decode): dcp2 == dcp1 == numpy-dense to ~5e-6. So the
DCP read/combine/scatter LOGIC (strided global-pos mask, per-shard kv_lens, local strided read, LSE merge)
is PROVEN correct — NOT a logic bug. The corruption is a **metal-only lowering defect in the DCP-specific
packed/tiled/context-sharded cache access**. Test-coverage gap explains the slip: interpret forbids
kv_packing>1 (kernel.py:2159 assert) so production packed KV (bf16 pack=2 / fp8 pack=4) reads/writes are
NEVER CPU-covered; the packed metal read (kernel.py:1242-1276, reshaped_cache + pltpu.bitcast word-shuffle)
is bypassed by interpret. Context-sharding splits the packed tile dim by dcp → reading tile>=1 of a
multi-page seq in packed layout is the untested surface.

Ranked (agent): H1 kv_packing>1 packed read of the sharded multi-page cache (highest); H2 donated/tiled
2nd-tile mis-commit (flat/onehot/no_donate "all failed" — but STALE-WORKER contaminated, so NO_DONATE
untested fairly); H3 LSE combine (lowest, CPU-clean). Discriminators (all on-pod, CPU can't see):
GLM_DCP_NO_DONATE=1 (cheap — if fixes → H2); pack=1 vs pack>=2 (H1, but pack=1 invalid for bf16/fp8);
SCATTER_ONLY page-0 vs page-1 dump (write vs read). Fix if H1 = read-side pageloop analog in kernel _fetch_bkv
(unreshaped per-page DMA when dcp>1). Testing NO_DONATE=1 (fair, synced) next — potential cheap unblock.

## 2026-07-09 20:50 — H2 REFUTED fairly: GLM_DCP_NO_DONATE=1 does NOT fix (pred=None @4K single-chunk, synced)

NO_DONATE on all 8 raylets (verified), workers d59626964: DCP=2 4K single-chunk needle still pred=None.
The donated-tile mis-commit hypothesis is eliminated on a FAIR test. Lead suspect = H1 (packed metal read
of context-sharded cache, page tiles >=1). Next discriminator: 1-logical-page needle (900 tok < P_g=1024,
still exercises both dcp shards + LSE combine + strided mask) — PASS ⇒ defect is page>=1 access (H1
boundary confirmed); FAIL ⇒ DCP broken at any size on metal (combine/mask), H1 also wrong.

## 2026-07-09 21:30 — ROOT CAUSE FOUND (DCP): block-granularity DOUBLE-multiplication; + two audit workflows

**THE DCP BUG (H1/H2 both disproven; agent traced the real cause):** `kv_cache_manager.get_kv_cache_spec`
pre-multiplies block_size by dcp (the "TODO(xiang) hack") AND vLLM's engine core multiplies spec.block_size
by dcp_world_size AGAIN (single_type_kv_cache_manager.py:66-70) → engine allocates ONE block id per
512*2*2=2048 tokens while the TPU stack consumes the table at _p_g=1024 tokens/entry → every table entry
>=1 dereferences an unallocated/stale page. Predicts EVERY symptom: dcp-only, single-chunk, >1024-token
threshold, pageloop-inert, NO_DONATE-inert (empirically confirmed BEFORE the theory landed — a real
prediction), CPU-invisible (harness built its own tables). CPU-verified: engine probe 2048→1024 tokens/id
with the fix; numpy sim corrupts at EXACTLY L=1025 unpatched, round-trips patched; existing tests
unchanged. Patch (kv_cache_manager spec fix + create_kv_caches *dcp + Guard-2 addressing repair) applied
to worktree; adversarial review in flight; 1-page (900tok) pod test = live prediction (should PASS).

**Code-audit workflow (29 agents, 8802ebab..HEAD): no BLOCKER/MAJOR.** Confirmed MINORs: (1) the L2
donate_argnums comment asserts a mechanism JAX structurally precludes (inner-jit donation dropped —
independently reproduced; comment must be corrected, code is harmless); (2) **fp8-KV has NEVER produced a
validated on-metal token** (fp8_80k built then died before generating) → MUST run a cheap fp8 needle at
dcp=1 (<=80K) before trusting the coupled fp8+DCP 128K gate — else an independent fp8 corruption could be
misattributed to DCP; (3) pageloop is silently expensive (per-layer sort+scan) — debug-only, never default.

**Observability-audit workflow (25 agents): 2 BLOCKER + 6 MAJOR in the tools themselves.** Highlights:
GLM_DCP_CACHE_DUMP driver-only → silent zero-files (tonight's incident; no loud check);
**dcp_cache_diff zero-fills missing shards → a partial dump flips DIFFER→MATCH** (the decisive verdict
could have been WRONG all along); diff breaks silently on short block tables (under the granularity bug
the diff itself was wrong); bf16/fp8 dumps stored as |V2 void crash every offline reader (tests were
float32-only); GLM_DCP_DUMP_NEWKV on a host-subset → SPMD divergence w/ silent fail-open import guard;
GLM_DUMP_STEP_HLO writes helper HLO while silently omitting the step fn; Guard-2 blanket-except fail-open
persists; guards never announce armed/inert. Gap proposals (worker code-hash cross-check, engine-vs-TPU
geometry assert, raw-decode-text capture, dump-written assertion, fail-loud guard pattern, HBM probe)
queued as the observability PR series. suggestions.md vindicated: half of tonight's cost was blindness
the tools were supposed to prevent — and some tools could actively mislead.

## 2026-07-09 21:45 — LIVE PREDICTION CONFIRMED: 1-page DCP=2 needle CORRECT (first DCP retrieval ever on metal)

bf16 DCP=2, 900-token prompt (1 logical page, BOTH dcp shards + LSE combine + strided mask exercised):
**pred='655242' == gold, acc 100%, DRIVER_EXIT=0.** The granularity theory called this in advance
(<=1024 tokens → only block-table entry 0 → valid). Five independent confirmations now: 1-page PASS,
NO_DONATE-inert (predicted before the theory landed), pageloop-inert, single-chunk >1024 FAIL, CPU sim
corrupting at exactly L=1025. The DCP read/combine/write machinery is CORRECT — the block-table
granularity contract was the bug all along. Patch under adversarial review → commit → sync → re-test the
previously-failing DCP=2 4K single-chunk needle.

## 2026-07-09 22:05 — ✅ DCP FIXED: post-granularity-fix 4K DCP=2 needle CORRECT (was pred=None)

The DIFFER→MATCH moment: bf16 DCP=2, 4K single-chunk (4 logical pages), previously pred=None — with the
block-granularity fix (1f700c507, all 8 workers synced): **pred='493718' == gold, acc 100%,
DRIVER_EXIT=0.** Same prompt/gold as the failing run; only the fix changed. The multi-week "DCP
packed-write metal bug" NEVER EXISTED — it was the engine-vs-TPU block-table granularity contract
(double ×dcp) end to end. Every prior symptom is explained; the DCP read/combine/scatter machinery was
correct all along.

**128K plan (updated):** bf16 + DCP=4 now FITS 128K WITHOUT fp8 (22.76 weights + 2.05 overlays + 1.25
reserved + 12.6/4=3.15 KV ≈ 29.2 < 30.75), decoupling the dense 128K gate from the still-unvalidated-on-
metal fp8 write path (audit: fp8 has never generated a token). Sequence: (1) 16K multi-chunk DCP=2 bf16
(validate multi-chunk post-fix); (2) 128K bf16 DCP=4 passkey ladder (THE dense gate); (3) fp8 needle @
dcp=1 (isolate fp8), then fp8+DCP=2 as the alternative config; (4) DSA sparse at 128K (the SPARSE gate);
(5) 256K throughput A/B.

## 2026-07-09 22:32 — 16K multi-chunk DCP=2 CORRECT (was pred=None). DCP comprehensively validated → 128K ladder

pred='578768' == gold (the identical gold that failed pre-fix), 8-chunk prefill, acc 100%. Post-fix DCP
record: 900tok 1-page ✓, 4K single-chunk ✓ (was None), 16K multi-chunk ✓ (was None). Launching THE dense
gate: 128K passkey ladder, bf16 + GLM_DCP=4 (fits: ~29.2/30.75), pool 68 ids (2048 tok/id at dcp=4),
chunk 2048, depths .25/.5/.75.

## 2026-07-09 23:10 — Owner review: n=3 ladder = SMOKE not gate; depths corrected; sparse+DCP landmine; MTP frozen

Owner flags (all adopted):
1. **The running 128K ladder (3 needles) is a GO/NO-GO SMOKE, not the gate.** 3/3 → Wilson LB ~44%. The
   ≥95% gate needs **n≥73 zero-failure** (~200 to survive one miss). Small-n is the recurring systematic
   weakness (GSM8K n=32, GPQA truncation, now n=3) — saved as a durable memory + gate definition below.
2. **Depths 0.25/0.5/0.75 omit the mechanism cells.** Depth ~0.0–0.05 is THE diagnostic for DSA (needle at
   max range must survive top-2048 selection out of 128K); 0.75 passes nearly by construction. Gate ladder
   = depths {0.0,0.05,0.25,0.5,0.75,0.95,1.0} × 11 trials = **77 needles** (Wilson LB 95.3% at 77/77),
   ~17–19h pod (overnight). Smoke-2 first: depths 0.0+1.0 × 1 (~1h) before committing the long run.
3. **Sparse@128K is NEW DISTRIBUTED CODE, not "32K with a bigger number":** DCP shards context → per-chip
   indexer scores only its slice → local top-2048 ≠ global top-2048; the sparse gather needs cross-chip
   rows. Design (owner's, adopted): per-shard top-min(k, shard_len) → all-gather candidates (4×2048,
   ~64KB/req) → global lax.top_k(2048) — EXACT by the union argument (the indexer docstring's proof extends
   verbatim) → each shard attends its locally-resident selected rows → cross-shard flash/LSE merge (the
   dense-DCP _dcp_lse_combine machinery + the kernel's online (m,l,acc) state). FREE regression: the
   distributed selection at 32K must reproduce the dcp=1 single-chip selection SELECTED-SET-EXACT (Gate-2b
   standard). Design work runs NOW while the dense ladder bakes.
4. **MTP FROZEN** (code-complete, pod-validation pending) until the headline gates close. No new surface.

**Upstream check (owner's ask):** the granularity hack came in via upstream PR #2398; upstream/main has
since removed the spec pre-multiplication (NOTE(weiyu0824) — same diagnosis) BUT allocates the physical
page at storage_block_size = block_size (NO ×dcp) while vLLM's engine still multiplies ids ×dcp →
**the same mismatch class plausibly lives on upstream/main today** (factor dcp, threshold 512 tok).
Verification + standalone bug-report draft delegated (owner submits; independently-mergeable credential —
review-bandwidth lesson from #2324).

## 2026-07-10 00:05 — 128K SMOKE 3/3 GO (first 128K retrievals ever); upstream scooped us by 3h; sparse-DCP design done

**128K SMOKE (bf16+DCP=4, granularity fix): 3/3 needles CORRECT** at 130,420 prompt tokens (d=0.25:
253647✓, d=0.5: 915875✓, d=0.75: 876150✓, DRIVER_EXIT=0). Recorded as SMOKE (n=3, Wilson LB ~44%) — GO
for the real gate. Gate = depths {0.0,0.05,0.25,0.5,0.75,0.95,1.0} × 11 = 77 needles (Wilson LB 95.3% @
77/77). Smoke-2 first: mechanism depths {0.0,0.05,0.95,1.0} × 1.

**Upstream verdict (owner's ask): the bug WAS upstream — and weiyu0824's PR #3129 merged TODAY 12:04 PT
(3h before our fix) implementing the same two-sided contract.** Shipped broken in v0.20.0–v0.24.0.
Memo: docs/upstream/dcp-block-granularity-report.md. Reframe 1f700c507 as convergent-with-#3129 (adopt
upstream structure on next sync). Upstreamable: geometry assert (would have caught 5 releases), 2 residual
granularity holdouts (routed-experts telemetry, KV-connector), and — the big one — **upstream dcp>1 is NOT
real context parallelism** (dcp folds into head-TP; kernels see the full cache): our owner-scatter +
LSE-merge CP attention is fork-only → a genuinely novel feature PR.

**Sparse+DCP distributed top-k: designed + prototyped, 52/52 CPU tests** (Gate-2b-DCP selected-set AND
tie-order exact at 32K geometry, dcp=2/4; mutation audit: dropping the position-sort → 5529 mismatches
caught; k/2 local width survives random data but fails the hot-shard case → top-min(k,local_len) is
load-bearing). Design: per-shard score (striped indexer k-cache is co-resident with latents — the
enabling invariant) → local top-min(k,S_local) → all-gather (64KiB/tok@dcp=4) → position-sort → top_k =
elementwise-exact vs single-chip AND gather-order-invariant by construction → owned-subset attention →
_dcp_lse_combine (dense machinery reused; sparse kernel needs emit_lse — (m,l) currently discarded).
5-file implementation plan, gated GLM_DSA_DCP; structural conflicts flagged (gather_kv_segment full-cache
flatten, ATTN_HEAD includes dcp, scatter class). Stage-A implementation next (CPU, parallel to the gate).
**MTP FROZEN** per owner directive.

## 2026-07-10 00:55 — SMOKE-2 4/4 (mechanism depths 0.0/0.05/0.95/1.0 ALL correct); GATE n=77 LAUNCHED

Smoke-2: depth 0.0 (911889✓), 0.05 (505695✓), 0.95 (638503✓), 1.00 (610927✓) — full-range retrieval at
130,420 prompt tokens works. Combined smokes 7/7 over all 7 depths. Stage A (GLM_DSA_DCP primitives:
distributed top-k, emit_lse, local gather) committed ed2a021be — 132/132 CPU tests, byte-identity proven,
all 8 workers synced. EU-bucket backup violation by an agent caught + objects deleted + re-backed-up to
gs://driftbench-dsv4-uc. **THE DENSE GATE IS RUNNING: 128K passkey, n=77 (7 depths × 11 trials),
bf16+DCP=4, ~19h.** 77/77 → Wilson LB 95.3% → the ≥95%@128K dense slot fills with a defensible n.

## 2026-07-10 03:40 — Stage B reviewed SAFE-FOR-METAL-LADDER; gate 16/16; Stage C in flight

Stage B (4f390a61e) adversarial review: **SAFE** — gate-off jaxpr byte-identity independently re-proven
CROSS-CHECKOUT (fresh processes, non-vacuous); a NEW selection battery (stripe boundaries ±1, hot-shard,
tie bands straddling stripes, topk>kv_len, dcp 2/4/8, ALL gather permutations) all elementwise-exact;
owner-scatter algebraically identical to gate-off for every step type; the reviewer BUILT the missing
mixed-batch + multi-chunk-prefill e2e cases itself (pass; being folded into the suite); refusal semantics
proven un-bypassable (NaN poison survives even swallowed callbacks — IndexShare can't slip through).
Non-blocking: export GLM_DCP_SCATTER_IMPL=pageloop on the sparse metal ladder (dense fallback side);
run the ladder with GLM_DSA_SCORER=xla first; MTP+DCP composition untested (MTP frozen anyway).
Post-granularity-fix note: docs/11's scatter-impl DIFFER→MATCH guidance is largely OBSOLETE (the scatter
was never the bug — the block table was); flag for a docs cleanup pass.
Dense gate: 16/16 (depth 0.00 closed 11/11 = 100%). Stage C (masked-prefill LSE, the last bridge to the
sparse ladder) implementing in parallel.

## 2026-07-10 07:10 — SPARSE-DCP STACK CODE-COMPLETE (Stages A+B+C); dense gate 36/36 and rolling

**Stage C landed (6f8855c3f, rebased onto the reviewer's test extension):** the distributed masked-prefill
flash scan with LSE combine — the last code bridge to the sparse 128K ladder. Merged suite **24/24 in one
process** (the Stage-B reviewer's ctx>topk refusal test CONVERTED to assert the real path — finite outputs,
== dcp=1 gate-off, caches bitwise); prefill e2e at (1,2)/(1,4)/(2,2) with non-page-aligned chunks; zero
regressions; gate-off jaxpr SHA == HEAD. The suite SIGABRT was proven PRE-EXISTING at HEAD (equal-load
pure-HEAD control aborts identically — jaxlib per-process XLA-CPU compile-volume threshold; bounded by a
documented clear-caches fixture). Stage-C adversarial review launched (the method's hard gate before its
metal ladder).

Dense gate: **36/36 zero failures** — depths 0.00/0.05/0.25 all closed 11/11 (the mechanism cells perfect).
On-pod ladder once the gate frees the pod (workers sync to 6f8855c3f first): (1) emit_lse unit @dcp=1;
(2) sparse decode @dcp=2 ≤2048-token prompts vs dcp=1; (3) chunked prefill @dcp=2 ~4-6K + step-HLO honesty
(expect ONE candidate all-gather pair per full layer + combine pmax/2psum, NO whole-cache collectives);
(4) 32K selected-set dump dcp=2 vs dcp=1 (Gate-2b on metal); (5) 32K sparse passkey dcp=2/4; (6) 64K;
(7) **128K SPARSE gate at n>=73** (env: GLM_DSA_DCP=1 GLM_MLA_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop
GLM_DSA_SCORER=xla, all raylet-baked). MTP stays frozen.

## 2026-07-10 08:30 — Stage C reviewed SAFE-FOR-METAL-LADDER: the sparse-DCP stack is FULLY certified

Independent adversarial review of 6f8855c3f: **SAFE** — per-shard masked scan proven bitwise/exact on
fresh experiments (empty-shard lse exactly -inf; l=1 → lse==m bitwise; one-hot sweep over EVERY global
position at dcp 2/4/8; float64 oracle ≤2.4e-7; poison probes clean); causal boundary proven correct with
MUTATION-POWER checks (kv±1 / wrong-owner / dropped-block-term all caught, maxdiff 1.15-2.57); gate-off
jaxpr SHA re-proven cross-checkout; prefill owner-scatter bitwise at P_g and P_g+1 boundary chunks; the
converted test PROVEN to execute the new branch (NaN-poison monkeypatch); the swallowed-callback drill
confirmed obsolete (all remaining refusals are trace-time raises); the clear-caches fixture proven not to
mask a real leak. 3 non-blocking notes (LSE-floor asymmetry theoretical-only; a (j) boundary-power test
suggestion; a cosmetic count grouping). Both Stage B and C now carry the same verdict. The metal ladder
(runbook §8) is cleared to start the moment the dense gate frees the pod + workers sync to 6f8855c3f.

## 2026-07-10 12:20 — Owner review: gate arithmetic pinned; ladder rung-4 gap closed; cheap-signal sequencing

Owner points (all adopted):
1. **The last 11 needles ARE the gate, not a victory lap.** 77/77 → Wilson LB ~95.3% (clears ≥95% by a
   hair — the one-sided zero-failure margin). A SINGLE miss → 76/77 → LB ~91% → the gate FAILS. If a miss
   occurs: EXTEND to n≈130 total (129/130 recovers LB >95%) — never round, never re-run-until-green.
2. **Ladder rung 4 must compare SELECTED INDEX SETS across sharding configs** (32K prompt: dcp=2
   GLM_DSA_DCP vs dcp=1 gate-off), elementwise incl. tie order — the free regression against the
   silicon-validated dcp=1 selection. Output-only comparisons can hide a subtly-wrong global merge that
   still produces plausible passkey hits (the failure mode that passes benchmarks and dies on
   reproduction). GAP FOUND: no metal dump hook existed for the stashed topk indices (CPU tests read the
   stash in-process). GLM_DSA_DUMP_TOPK + runner/dsa_topk_diff.py (selected-set differ w/ coverage
   refusal + cross-shard replication assert) being implemented now — a ladder prerequisite.
3. **Cheap signal before the ~10h sparse gate:** rungs 4 (selected-set-exact @32K) and 6 (64K smoke) are
   the discriminators that decide whether the distributed top-k is right; the 128K n>=73 run only launches
   after they're green.
Gate at 68/68 (six depths closed 11/11); depth 1.0 in its final trials.

## 2026-07-10 10:30 — ✅✅ THE DENSE 128K PASSKEY GATE: 77/77 = 100%, ZERO FAILURES (Wilson LB ~95.3%)

**GATE CLOSED.** GLM-5.2-FP8 on 32× TPU v4, bf16 + GLM_DCP=4, 130,420-token prompts (run_id 124):
7 depths {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0} × 11 trials = **77/77 needles retrieved exactly**,
DRIVER_EXIT=0, ~470s/needle, ~13.5h wall. Every depth closed 11/11 including the mechanism cells (0.0 =
retrieval at maximum range). Wilson one-sided 95% lower bound ≈ 95.3% ≥ 95%: the **dense passkey ≥95% to
≥128K slot is FILLED at a defensible n**. Full provenance: results.db run 124 (raw outputs, seeds,
latencies) + gs://driftbench-dsv4-uc/results/. This stands on the granularity fix (1f700c507) + DCP=4 —
three days from "128K impossible (HBM wall + DCP corrupts)" to a closed gate. NEXT: the sparse ladder
(runbook §8) — rungs 4+6 are the cheap discriminators before the ~10-19h SPARSE 128K gate (n≥73).

## 2026-07-10 10:50 — RUNG 1 PASS ON METAL: emit_lse compiles + exact on v4 (top sparse-stack risk retired)

Single-chip probe (probe_lse_unit.py, artifact rung1-lse-unit-metal-rung1-metal.txt): GATE L1
byte-identity (emit_lse=False == default, BITWISE) PASS; GATE L2 out-invariance (True's out == False's,
BITWISE) PASS; GATE L3 lse vs XLA oracle max-abs 0.0–1.9e-6 (fp32 bar 5e-5, bf16 bar 1e-2) PASS; empty
rows (seg_valid=0) below the LSE floor on both PASS. All geometries (R 8/64, sv full/700/1/0, seg_block
512/768). The [1,H,128] lane-broadcast lse out-block — the #1 Stage-A/C metal risk — lowers fine on v4.
RUNG 2 next: 2040-token needle, dcp=1 sparse (reference) vs dcp=2 GLM_DSA_DCP (new distributed decode),
GLM_DSA_DUMP_TOPK armed both sides, GLM_EXPECT_CODE_HASH pinned.

## 2026-07-10 12:40 — RUNG 2 PASS: first distributed sparse attention on metal; selections BITWISE identical

**RUN B = the first GLM_DSA_DCP execution on silicon** (dcp=2, 2040-token needle ×2): engine compiled
(2686.9s — scan-with-LSE, all-gather-in-cond, owner-scatter-in-cond all lowered on v4), needles 2/2 with
predictions IDENTICAL to the dcp=1 reference (202296, 173606). Selected-set verdict by DIRECT dump
comparison: **840/840 (step,evt) events BITWISE EQUAL including order** — the distributed selection
reproduces the single-chip selection exactly on metal (the Gate-2b standard, strongest form).
CAVEAT/HONESTY: the new dsa_topk_diff CLI reported DIFFER on the same dumps — a bug in the DIFFER's own
alignment/live-row masking (the raw arrays are equal; direct np.array_equal over all 840 pairs). The
instrument gets the same discipline as everything else: fix + regression-test before rung 4 relies on it.
Also noted: JAX dedupes the replicated-value debug callback to ONE process (all dumps land on a single
host; --allow-missing-procs is the designed escape; cross-proc replication assert not exercisable).
RUNGS 3+4+5 COMBINED next: 32K needles at dcp=2 sparse (chunked masked prefill for real) vs dcp=1, dumps
both sides, direct-comparator verdict + needle correctness; then 64K (rung 6); then THE SPARSE GATE.

## 2026-07-10 13:30 — ⚠ RETRACTION + RUNG 2 REOPENED: the differ was RIGHT; my verification script was the bug

**Retracting the 12:40 "selections BITWISE identical" claim — it was FALSE.** My ad-hoc comparator had a
key-matching bug (`next(k for k in keys if "ind" in k)` matched 'step_index', not 'topk_indices') → it
compared step numbers with step numbers, 840 trivially-equal pairs. The differ-fix agent refused the "fix",
read the raw npz bytes at zero abstraction, and proved **798/840 pairs GENUINELY differ**. All 15 differ
tests pass; A-vs-A and B-vs-B on the real dumps MATCH — the instrument is correct. The suggestions.md
lesson cuts both ways: ad-hoc verification scripts are instruments too, and overriding a tool's verdict
requires the same rigor as building the tool.

**True rung-2 state:** prefill events EQUAL (42/42); 634 decode events same-SET-different-ORDER (dcp2
index-sorted vs dcp1 score-ordered — hypothesis: STASH-POINT inconsistency, the DCP path stashing the
post-canonical-position-sort list while dcp=1 stashes raw score-ordered top-k; sets equal → attention
unaffected → identical needles); **164 events with REAL SET DIFFERENCES, exactly and only where
truncation bites (nc>k)** — adjudication needed: per-shard bf16 score-precision boundary swaps (the
Gate-2b S2 boundary-band class, expected + acceptable) vs a genuine merge bug (unacceptable). The
identical needle predictions are exactly the "plausible-but-wrong selection passes passkey" failure mode
the owner flagged — the instrument did its job. RUNG 2 IS NOT PASSED until the 164 are adjudicated.

## 2026-07-10 14:10 — RUNG 2 ADJUDICATED: FAILS on a REAL evt00 defect (first indexer layer score-blind under DCP)

Full forensics (agent, code + data): (1) ORDER diffs = benign cross-config artifact — BOTH paths stash
score-ordered pre-sort (mla_attention.py:2125-2130; canonical position sorts happen post-stash); two
independent autoregressive runs diverge by ulps from layer 1 → near-tie order shuffles, SETS preserved;
the CPU suite passed elementwise legitimately (paired inputs → bitwise-identical scores). Differ criterion
for cross-config runs → canonicalized-SET comparison. (2) evt01-20 set diffs: textbook S2 boundary band
(symdiff 2-10 positions = <=0.49% of k, ONLY at kv_len>2048, 57% within 10 ranks of the k-th boundary) —
final adjudication needs scores in the dump (extension prescribed: topk_scores f32 key + band-width
report). (3) **evt00 = REAL DEFECT: the dcp2 run's FIRST full indexer layer selects score-blind — exact
arange(min(kv_len,2048)) on all 38 rows (constant-score signature), at truncation DROPPING the most
recent d positions that dcp1 ranks #1-15** (e.g. step0039: kv_len 2052, dcp1 head [1,2049,2044,2048...],
dcp2 = identity). evt01's mid-rank anomaly = cascade contamination from evt00 (prediction: vanishes with
the fix). Suspects (file:line in report): the layer-0 striped indexer k-cache reading zeros under DCP
(_dcp_idx_write / _glm_dsa_dcp_owner_scatter / _glm_dsa_indexer_cache_index slot resolution for
DeepseekV32IndexerCache) vs degenerate q/w in _dcp_score_select. NEW TRIPWIRE adopted: any valid row that
is an exact ascending identity = automatic FAIL (would have caught evt00 alone, even below truncation).
The needles passed identically throughout — the exact "plausible-but-wrong selection survives the
benchmark" trap; the selected-set rung earned its keep. RUNG 2 = FAILED until evt00 is fixed.

## 2026-07-10 15:40 — 32K dcp=2 SPARSE SMOKE 3/3 (Stage-C masked prefill works end-to-end on metal)

First 32K distributed sparse serving: depths 0.0/0.5/1.0 all correct (249708/731442/108407), 16-chunk
prefill through the Stage-C distributed masked scan, DRIVER_EXIT=0, engine 2631s. Labeled a SMOKE — the
evt00 score-blind defect is PRESENT in this run (rung 2 failed); retrieval survives because the other 21
full layers + IndexShare select correctly. Dumps gathered for post-fix comparison. dcp=1 32K reference
launching now (independent of the fix — the reference side of rung 4).

## 2026-07-10 15:30 — 32K adjudication data: dcp=1 retroactive check CLEAN; tie-saturation hypothesis rises

On-pod (w-2) set-level analysis of the 32K dumps (2205 events/side): **ascending-identity rows = 0 in BOTH
dcp=1 and dcp=2** — (a) the owner's retroactive check passes: the dcp=1 silicon validation never carried
the score-blind signature; (b) the "layer-0 indexer cache reads zeros under DCP" hypothesis is WEAKENED
(a dead cache would arange at 32K too; 32K evt00 is nearly clean, median symdiff 56). The rung-2 arange
signature is kv≈2050-SPECIFIC → new leading hypothesis: **ReLU tie-saturation** — the indexer ReLU zeroes
fully-negative rows; at a weak early layer the k-th boundary sits inside a large exact-0.0 tie class;
cross-run ulp noise flips tie populations; in the all-tied extreme the merge tie-break emits exact arange.
The layer-0 cache dump remains the discriminator (dead cache vs saturation).
ALSO: evt01-20 at 32K show set symdiffs of median 944-3666/4096 — FAR beyond a narrow band → cross-run
selected-set-exact is UNACHIEVABLE at truncation scale by construction if the tie class is that wide; the
rung-4 criterion must be the score-band-quantified form (needs the topk_scores dump extension) or a
paired-input on-line A/B (same-run dual selection compare). The needles (3/3 both configs) are consistent:
the churn lives in the ~0-score tail that contributes nothing to attention. TRAP pinned per owner: the
ascending-identity tripwire is a DETECTOR, never a mitigation — no tie-perturbation "fixes".

## 2026-07-10 16:10 — evt00 ROOT-CAUSE REPORT: constant-score row PROVEN as the mechanism; 15:30 hypothesis CORRECTED; rung-4b instrument LANDED (f0c63c302)

The bug-hunt agent's final report (full CPU verification: **349 passed, 0 failed**; dcp suite 28/28 incl.
4 new tests). Mechanism **proven end-to-end**: zeroing ONE layer's indexer weights_proj in the REAL
forward (positive control) reproduces `arange(min(kv,topk))` + `-1` tail **bit-for-bit** through the real
DCP merge — ascending identity is the byte-exact output of a CONSTANT score row through both selection
paths. Exonerated with evidence: (a) event misalignment — 840/840 rung-2 pairs metadata-aligned bitwise;
(b) DCP merge / kv-cache slot resolution — NEW multilayer test drives 3 full + 1 shared indexer layers
through the runner-real INTERLEAVED slot map (indexer k_cache registered before attn; layer-0 cache at
kv_caches[0], the production shape the 1-layer harness couldn't reach): dcp=2 == gate-off ELEMENTWISE at
every step/layer; (c) worker code skew — hash-pinned, dirty=0.

**CORRECTION to the 15:40 and 15:30 entries** (the record over the narrative, again): the 15:40 claim
"the evt00 score-blind defect is PRESENT in this run" is DISPROVEN — a fresh scan of the 32K dcp=2
dumps shows evt00 **fully healthy** — 57/57 truncated decode rows score-rich (sink + recency heads like
`[0, 1, 32550, 32588, …]`), zero ascending rows. The "ReLU tie-saturation" framing (k-th boundary inside
a wide 0.0 tie class) predicted PARTIAL degeneracy and does not match: rung-2b evt00 was TOTAL arange on
38/38 scored steps while evt01 in the SAME step was healthy, and 32K evt00 is clean. Leading verdict:
the 2560-shape dcp=2 run's layer-0 score inputs (w·relu(q·k)) were exactly constant — the all-zero class:
**either the layer-0 striped indexer k-cache read zeros at decode, or that executable's q_idx/w_idx were
degenerate** — shape/state-specific pod behavior (never-written stripe or buffer-donation aliasing of the
first cache slot), CPU-blind. Index-only dumps cannot name the zero input; the new instrument can.
(The 15:30 evt01-20 finding STANDS: 32K cross-run set symdiffs 944–3666/4096 = boundary tie churn →
selected-set-exact is unachievable cross-run at truncation scale; kth_band is the criterion.)

**Landed as f0c63c302** (applies my adversarial review + byte-identity check vs the agent's validated
tree; workers synced 8× f0c63c30 dirty=0): merge_topk_candidates(return_values=True) → optional f32
selected-slot scores (indices math byte-identical); armed-only threading through both selection branches
(gate-off jaxpr identity re-proven); stash/dump `topk_scores` payload; dsa_topk_diff gains (1) the
SCORE-BLIND TRIPWIRE — arange rows (kv_len>2) are DIFFER **even when both runs agree elementwise** (two
degenerate runs must never MATCH green; detector only, never a mitigation), (2) `kth_band` per diff event
(0.0 = tie churn at the k-th boundary; large = real drop — the rung-4 criterion), (3) topk_scores in the
cross-proc replication contract. Review note: the tripwire's false-positive risk at 2<kv_len≤topk is
disproven by the slot-order test (the stash is score-descending; position-argsort is only the tie-break).

NEXT (owner order): (1) rerun the EXACT rung-2 shape (max_seq_len=2560, dcp=2) with GLM_DSA_DUMP_TOPK
raylet-baked on ALL 8 hosts at f0c63c302 — expect tripwire FAIL + constant-0.0 score rows at evt00, which
fingerprints the zero input; then the layer-0 indexer k-cache dump names the buffer. Non-repro ⇒
state-dependent ⇒ flight-recorder + repeated launches. (2) Rung 4 with scores on BOTH sides (existing 32K
dumps lack topk_scores) — adjudicate via kth_band; tripwire must stay silent. (3) One unarmed metal smoke
re-confirms gate-off 3/3.

## 2026-07-10 16:20 — ARMED RERUN (2560-shape, dcp=2, f0c63c302): evt00 NON-REPRO — honest null with a caveat

The instrumented rerun of the exact rung-2b shape (2040-tok needles ×2, max_len 2560, blocks 8, mbt 2048,
dcp=2; GLM_DSA_DUMP_TOPK + GLM_EXPECT_CODE_HASH raylet-baked and verified in /proc on all 8 hosts;
fingerprints 8× f0c63c30240d dirty=0): needles 2/2 (110391/865371), 840 dump events on w-2 WITH the new
topk_scores payload. Analyzer verdict (validated first against the old dumps: old dcp=2 → SCORE-BLIND
detected at evt00; old dcp=1 → clean): **all 21 events score-rich — arange rows 0, constant-score rows 0,
all-zero rows 0, on 4098 live rows/event.** The evt00 defect did NOT reproduce on this launch.

CAVEAT (the reason this is a bound, not an exoneration): this run's traced program differs from the bad
run's in three ways — (1) the armed dump code now threads scores (merge return_values + tuple cond),
(2) GLM_DCP_ASSERT_SHARDING/CACHE_SANITY=1 were on, (3) the binary is f0c63c302 vs 33384c16a. A
buffer-donation/layout-lottery bug can vanish under ANY of those perturbations. Classification firms up
as **state/program-dependent zero-input at cache slot 0** — the class the forensics predicted
(never-written stripe or donation aliasing), NOT a deterministic property of the shape. Detection is the
durable mitigation: the score-blind tripwire now fails ANY dump run that carries the signature (even two
agreeing runs), and armed topk_scores runs fingerprint the input directly. A forensic repro attempt at
the OLD binary (33384c16a, asserts off) is queued as a separate thread — worth one run for the record;
the ladder proceeds on current code regardless.

RUNG 2 RE-RUN in flight: the dcp=1 armed twin (same shape/protocol, GLM_DCP=1) launched — then
dsa_topk_diff (tripwire + kth_band armed, scores on BOTH sides) delivers the rung-2 verdict on f0c63c302.

## 2026-07-10 17:05 — THE INSTRUMENT DELIVERS: ×¼ SCORE STRIPES caught at rung-2 re-run (dcp=1 side, evt01) — cache-geometry fingerprint of the evt00 defect class

Rung-2 re-run on f0c63c302, both sides armed, needles identical (110391/865371 both configs). Differ:
798/840 DIFFER, tripwire 0 both sides. Set-level adjudication: 42 EQUAL / 718 ORDER-ONLY (the benign
cross-config tie-shuffle) / **80 REAL set diffs, all where truncation bites — but kth_band max = 103.5,
NOT boundary churn.** Per-position cross-run score join at step 20: evt00 pristine (p95 0.015), evt10-20
= MoE-routing drift envelope (ulp flips → expert swaps; p95 10-21), and **evt01 = a structural defect:
positions scoring 36-40 in the dcp=1 run where dcp=2 scores 147-155.** Ratios 3.86-4.09 ≈ **exactly ×4**.
Band scan at steps 20/30 (clean scale separation): the depressed set is EXACTLY four 128-wide,
128-aligned windows — **{576, 832, 1600, 1856}+[0,128) = in-block offsets 64 and 320 of the ODD 512-token
blocks only** — while dcp=2's map is smooth there. Pure cache-geometry structure; magnitude ≈ ×¼ = a
quantization-scale/exponent class (the DSA indexer k-cache is fp8 with per-tile scales — a wrong/stale
scale tile yields exactly this signature; all-zero/garbage scales yield the constant-score → arange
signature of the ORIGINAL evt00). UNIFIED HYPOTHESIS: stripes of the indexer k-cache (payload or scale
sub-buffer) are written wrong/not-at-all in a launch-state-dependent way; expression depends on prior
HBM contents — zeros → score-blind arange (original run), ~×¼ garbage → depressed stripes (this run),
benign memory → clean-looking (the non-repro). Both configs can express it (dcp=2 originally, dcp=1 now).
An index-only dump could NEVER have caught today's form: needles pass, sets differ by 2-10 at truncation
only, tripwire silent — **only the score payload exposes it.** (Analyzer gap noted: my constancy check
missed the stripe class; the differ's kth_band caught it — add a within-run bimodality check.)
IN FLIGHT: dcp=1 run-B on the SAME ray cluster (determinism probe: stripes stable per launch-state or
per-config?). NEXT: indexer-cache dump run (GLM_DCP_CACHE_DUMP_LAYERS targeting evt01's slot) to read the
fp8 payload + scale tiles directly and name the buffer + write path. RUNG 2 remains OPEN (correctly).

## 2026-07-10 17:55 — RUN-B: stripes GONE (same binary/config/session) → per-engine-instance lottery; the stale-HBM read hypothesis now leads

dcp=1 run-B (identical driver, same ray session, back-to-back with run-A): needles 2/2 identical,
**evt01 CLEAN at steps 20/30 — zero depressed positions** where run-A had the four ×¼ stripes. Same
executable + same inputs ⇒ the A/B difference can only be MEMORY STATE: HBM is not scrubbed between
engine instances, so a read-of-unwritten-memory defect expresses whatever the previous occupant left.
Timeline fits: run-A's engine inherited the dcp=2-armed engine's HBM (DIFFERENT cache layout → stale
bytes ≠ expected keys → visible ×¼ stripes); run-B inherited run-A's (SAME layout, same prompt → stale
≈ fresh → invisible). Original evt00-arange run inherited dense-gate-layout HBM (→ zeros/garbage →
constant scores). CRITICAL IMPLICATION: if the write path skips those cache stripes, the bug is present
in EVERY run and merely invisible when stale≈fresh — "clean" runs are clean by luck.
DE-LOTTERIED PROBE (next): GLM_DCP_CACHE_DUMP on the first ~12 kv-cache slots (covers evt00/evt01
indexer k-caches under either registration mapping), TWO identical dcp=1 runs, byte-diff the dumps:
prefill compute is deterministic ⇒ written regions identical across runs; regions that VARY across runs
are NEVER WRITTEN. Correlate hole geometry with the run-A stripe map ({576,832,1600,1856}+[0,128) =
in-block offsets 64/320 of odd 512-blocks). This decides write-hole vs read-geometry in one pass.

## 2026-07-10 19:40 — WRITE-PATH HOLE NAMED: evt01's indexer k-cache has NEVER-WRITTEN sublane stripes; scrambler experiment confirms both predictions

The de-lotteried probe (runs C/D identical-config cache dumps; run-E dcp=2 as HBM scrambler; run-F dcp=1
post-scramble) delivered on BOTH registered predictions:
(1) **run-F scoring striped at evt01** — {512,768,1536,1792}+[0,64): 64-token windows, 256-periodic,
odd logical blocks (run-A's family; A had offsets 64-191/320-447 at 128 wide — same structure, different
phase/width per engine instance/executable);
(2) **F-vs-C byte-diff: layer2 [IDX] — evt01's indexer k-cache and ONLY it — differs on all 8 hosts at
identical coordinates: i-rows {0,1}∪{8,9} of 16 (= the first 2 sublanes of each 8-sublane tile of the
(16,32)-token page layout) in two physical blocks; 32732/32768 elements.** Deterministic prefill ⇒
regions that vary across identical runs were NEVER WRITTEN. C-vs-D "IDENTICAL" is explained: both
inherited same-layout same-prompt predecessor HBM → stale bytes coincided (the byte-diff pair design's
blind spot; the scrambler run closes it). The scoring stripes and cache holes agree exactly at in-page
offsets [0,64)∪[256,320) once the sequential-block-table assumption is dropped (physical blocks {2,4}
hold logical pages 1,3).
VERDICT: the DSA indexer k-cache WRITE path (GLM_DSA_DCP gate on, dcp=1 mesh, GLM_DCP_SCATTER_IMPL=
pageloop) skips sublane stripes of alternating logical pages on ONE cache buffer per instance —
partial-sublane-tile store, layer/row lottery per executable, invisible whenever stale HBM ≈ fresh keys.
This unifies every observation to date (original evt00 arange = zeros-flavored stale; A/F ×¼-flavored;
B/C/D clean-by-luck). CPU-DETERMINISTIC REPRO NOW POSSIBLE: sentinel-initialized cache through the real
prefill write; assert no sentinel below kv_len — no HBM lottery on CPU, the sentinel IS the stale byte.
NEXT: read the write path (dsa indexer cache update / owner-scatter / pageloop), sentinel CPU test,
root-cause fix, adversarial review, land, re-run rung 2.

## 2026-07-10 21:10 — CPU logic EXONERATED (56/0); flat-impl runs clean so far; byte-diff pair G2→H2 in flight

CPU sentinel adjudication (agent, scratchpad; landed as e1b666382, test-only): **all four
GLM_DCP_SCATTER_IMPL formulations × dcp={1,2} meshes give position-exact sentinel-free coverage** of the
exact pod geometry under the verbatim serving shard_map — the traced write logic is EXONERATED; the metal
stripes are lowering/allocation-level. Adversarial-read notes (documented, none reachable from the vLLM
allocator): pageloop clamp-vs-drop divergence on out-of-contract bt ids; the OOB-sentinel-largest
assumption would break if BATCH ever sharded dim 0 (guarded by the no-DP refusal); at serving num_seqs
(>=8 via MIN_NUM_SEQS) n_pages_touched saturates total_pages+1 so unique-truncation is dead at this
shape. Sharpest metal pointer: every pageloop live iteration full-page-stores via
dynamic_update_index_in_dim after a jnp.where merge into the DONATED striped cache — a wrong
sublane-granular masked partial store lowers to EXACTLY the observed rows-{0,1}∪{8,9} stripes.
METAL DISCRIMINATOR (in flight): GLM_DCP_SCATTER_IMPL=flat relaunch; run-G (dcp=2 flat) clean; **run-H
(dcp=1 flat, post-scramble — run-F's exact situation) evt01 CLEAN at steps 20/30 where pageloop's run-F
was striped.** One draw ≠ proof → the rigorous readout is byte-diffing two scrambled flat runs
(G2 scrambler → H2, in flight): H-vs-H2 IDENTICAL everywhere ⇒ flat writes everything ⇒ pageloop v4
lowering indicted for the indexer path ⇒ fix = flat for the DSA owner-scatter (its own validation ladder:
CPU bit-identity already test-gated, then dcp=2 selected-set + needles) — NOTE the dense-path history is
the mirror image (plain scatter mislowered, pageloop was the fix); nothing generalizes across paths
without metal evidence, measure per path.
OPERATIONAL NEAR-MISS (logged for the record): landing the test-only commit auto-synced workers to
e1b666382 while the running ray session pins GLM_EXPECT_CODE_HASH=f0c63c302 — H2 would have died at init.
Caught before launch; workers re-pinned to f0c63c302 until the probe pair completes. The pin worked as
designed — against its own operator.

## 2026-07-10 23:30 — RUNG 2 CLOSED; root cause = pageloop v4 lowering; fix landed (flat default, 4f7d9a001)

**The verdict chain, complete:** CPU sentinel suite exonerated all four scatter formulations logically
(56/0) → metal scrambler-byte-diff protocol indicted pageloop (never-written sublane row-stripes
{0,1}∪{8,9}-class in the evt01 indexer k-cache, all 8 hosts, F-vs-C) and cleared flat (H-vs-H2
byte-IDENTICAL everywhere, all 12 dumped slots, each behind its own scrambler) → **the defect is
pageloop's v4 lowering of the per-page jnp.where merge + full-page dynamic_update_slice RMW into the
donated striped cache.** The stale-HBM expression lottery (zeros → evt00 arange; foreign-layout garbage
→ ×¼ stripes; same-layout → invisible) explains every observation since rung 2 first failed. The dense
path's history is the exact mirror (its scatter mislowered; pageloop is ITS fix) — per-path metal
evidence, now enforced by SEPARATE envs: dense GLM_DCP_SCATTER_IMPL (default pageloop) vs DSA
GLM_DSA_DCP_SCATTER_IMPL (default flat) so a dense-tuned bake can't re-break DSA silently.
**RUNG 2 VERDICT (flat, runs G dcp=2 / H2 dcp=1, both score-armed):** needles 6/6 across G/H/H2; differ:
tripwire 0 both sides, replication 0, prefill events EQUAL (42), 723 ORDER-ONLY, 75 truncation-only set
diffs with **band/drift ratio max 2.02, p90 1.17, median 0.21** — pure k-th-boundary churn under
cross-run MoE drift (the run-A stripe class measured ratio ~500 on the same yardstick). CLOSED under the
band-quantified criterion; runbook §8 rung 2 updated with the criterion revision + verdict.
**Fix commit 4f7d9a001** (on top of the sentinel suite e1b666382): flat default + own env + doctrine
docstrings; CPU 28/28 full DCP suite + 16/16 coverage on the new default; test baselines flipped
(non-vacuous: scatter/pageloop/barrier each diffed against the flat default). Independent adversarial
review in flight; workers sync + hash re-pin after its verdict. NOTE for the ladder: the rung45 32K dumps
are pageloop-era — rungs 4-6 re-run under the flat default with scores armed (they double as the fix's
pod 3/3). Fifty-some pod-runs of forensics, and the instrument that broke the case was the one the owner
prescribed two days ago: scores in the dump.

## 2026-07-11 00:20 — Review verdict landed (74d8c3225); rung 3 first attempt CONFOUNDED by disk-full; HLO honesty HARD GATE PASSES

**Review of 4f7d9a001: SAFE-TO-SYNC, plus a real catch** — my runbook edit had removed the dense
pageloop bake and claimed "dense default = pageloop"; in CODE the dense default is the PLAIN scatter
(dense's metal-proven-bad op) and pageloop only ever came from the bake — and the dense fallback FIRES
INSIDE SPARSE SERVING (ctx≤topk prefills). Next launch would have silently regressed the dense path.
Fixed: bake restored + doctrine corrected in docs/11; GLM_DSA_DCP_SCATTER_IMPL added to bench/engine.py's
driver-only-env warn list; test-header nit fixed (74d8c3225). Workers synced 8× 74d8c3225 dirty=0.
**Rung 3 attempt 1 (5K chunked prefill, dcp=2, flat, HLO armed): CONFOUNDED, not counted.** d=0.25
correct; d=0.75 pred=None — but BOTH w-0's and a worker's raylet file-system monitors were throwing
>95%-full errors during that decode (object-store ops degrade; the run's own HLO dumps + accumulated
forensics archives filled two hosts). Infra failure, not a sparse-stack verdict. Hygiene done: rung45
(pageloop-era, void) deleted; rung2fix archives — irreplaceable lottery draws runA+runF backed up to
gs://driftbench-dsv4-uc/dumps/rung2fix-w2/ (1682 objects) and w-0's shard set to /dumps/rung2fix-w0/ —
then cleared everywhere; protocol runs B/C/D/E/G/H/H2 are regenerable and their verdicts are logged.
**HLO honesty (rung 3's other deliverable, valid from the completed compile): HARD GATE PASSES — ZERO
whole-cache collectives** (no all-gather with any dim ≥4096 in the 2048-token chunk program). Census:
five shapes ×78 layers (attention-level, incl. the watchlist's predicted replicated-q reshard
bf16[64,2048,512] — throughput note) + 2 singletons; detailed sparse-branch attribution deferred
(program archived: scratchpad/rung3_hlo_2048tok.txt.gz). Rung-3 needle rerun in flight on clean disks.

## 2026-07-11 01:10 — RUNG 3 CLOSED: 5K chunked prefill 2/2 @dcp=2/flat + HLO hard gate (zero whole-cache collectives)

Attempt 3 (post-relaunch, clean disks): both needles exact (843616, 773976) — the d=0.75 miss of attempt
1 confirmed as the disk-full confound (attempt 2 died separately at TPU init: stale SliceBuilder grpc
after the wedged engine's SIGTERM; the launcher's stop-hygiene relaunch cleared it — the runbook's
"relaunch after any pod crash" rule, again). Stage-C masked prefill at 3 chunks + decode under the flat
default on a NEW shape. Rung 3 = CLOSED (needles + step-HLO honesty: no whole-cache collectives; census
archived). NEXT: rung 4 — 32K selected-set, dcp=2 vs dcp=1, topk_scores armed both sides, tripwire +
kth_band criterion; then the 32K/64K smokes; then the 128K sparse gate.

## 2026-07-11 05:30 — RUNGS 4+5 CLOSED at 32K (flat default); the criterion held in every part

Rung-4 protocol (events-filtered dumps 0,1,2,10,19,20 after the 32K prefill events blew w-2's disk —
ENOSPC even fingered the earlier d=0.5 pred=None, which passed exactly on the clean rerun): dcp=2 3/3
(422181/663295/648060), dcp=1 twin 3/3 IDENTICAL predictions. Adjudication (vectorized, w-4): tripwire 0
both sides; replication 0; **first-chunk rows EXACTLY equal 36864/36864** (the kv<=topk region — set-
equality by construction holds ELEMENTWISE on metal); past truncation the sets churn by cross-run MoE
drift (band/drift max 3.26 where defined — the drift envelope at 32K is itself large, p95 up to ~41
score units over 105 compounding steps); **structured-band stripe detector: 0/285 decode events** (the
within-run check the drift cannot fake — contiguous >=32-position runs at 10x median diff: none, either
side). Note for the record: the 2.5K-era "prefill EQUAL" clause of the criterion generalizes at 32K to
"the un-truncated region exactly equal" — chunks 2+ truncate and churn like decode; first chunks match
exactly as the union argument demands. Throughput note for the 256K A/B: dcp=2 needles ran ~715s vs
dcp=1 ~73s at 32K — the replicated-q reshard + selection collectives cost is real; correctness first,
perf stage later. NEXT: rung 6 (64K smoke) → 128K mechanism-depth smoke → THE GATE (n>=73).

## 2026-07-11 09:55 — RUNG 6 CLOSED: 64K sparse 3/3 (first sparse retrieval above 32K on any hardware)

Take 1 hit the WATCHLIST'S OWN prediction — CompileTimeHbmOom by 98MB (bf16[2048,512,256] chunk
transient) — and the runbook's prescription (chunk 1024) fixed it on the first try. Take 2: 3/3 EXACT
(205323 / 189158 / 860638) at depths 0.0/0.5/1.0 incl. both mechanism cells, dcp=2, flat default,
64K max-len. Needle time ~2715s (63 chunks × ~43s — the dcp≥2 prefill collective cost; see below).
OWNER DIRECTIVES ACCEPTED (from review): (1) permanent metal WRITE-PROBE guard at engine init
(sentinel scratch cache through the compiled owner-scatter, refuse startup on any hole) — build+review
after the gate launches; (2) upstream report for the pageloop-v4 sublane-store defect staged under
docs/upstream/ for OWNER submission (bug #2 after the granularity bug); (3) 256K A/B must be dense-vs-
sparse at IDENTICAL dcp so the O(L·k) win is measured THROUGH the reshard/collective penalty.
GATE-FEASIBILITY FLAG: at 64K/dcp=2 pace, a 128K needle could cost ~90min ⇒ n=77 ≈ 5 days — NOT viable
if it holds at dcp=4. The 128K mechanism smoke (dense-gate geometry: dcp=4, chunk 2048, pool 68,
max-len 131840) measures the true per-needle cost and decides: gate as-is vs the head-split perf stage
FIRST (the replicated-q reshard ×78 layers ×chunks is the suspected dominator — bf16[64,T,512]
all-gathers in the HLO census).

## 2026-07-11 11:20 — 128K sparse: chunk 1024 MANDATORY (CompileTimeHbmOom at 2048 for BOTH dcp=2@64K and dcp=4@128K); w-2 disk mystery solved; smoke take-3 in flight

Two CompileTimeHbmOoms establish: the sparse chunk transient (bf16[T,512,256]-class per layer) caps
max_num_batched_tokens at 1024 for >=64K regardless of dcp — the dense gate's chunk-2048 geometry does
NOT carry over to sparse. Gate plan must use chunk 1024 (prefill = 128 chunks/needle at 128K).
DISK ROOT CAUSE (three incidents today): the ORIGINAL pageloop-era dump sets (topk_r345 2x2205 files
~17G + old partials) sat in w-2:/tmp the whole time — individually small files invisible to top-N size
listings; found via sudo du -xsh + prefix counting. Purged (decisive sets in GCS: dumps/rung2fix-w2 A+F,
dumps/rung2fix-w0, dumps/rung4 dcp1+dcp2; w-0 ~/dumps/rung2 originals intact). w-2 now 52G free; the
7/8-node join failure was the full disk. RULE ADOPTED: gate-class runs are UNARMED (a 128K armed gate
would write ~230GB); the selection instrument's job ended with rungs 2-6, all closed.
IN FLIGHT: 128K mechanism smoke take-3 (depths 0.0/0.05/0.95/1.0 ×1, dcp=4, chunk 1024, pool 68,
UNARMED). Its per-needle time decides: ~15-20min → gate (n=77) tonight ~19-26h; ~90min → the head-split
perf stage (compose ('model','expert') head sharding back into the dcp bodies — the replicated-q reshard
×78 layers is the suspected dominator) comes FIRST, else the gate costs ~5 days. Post-smoke sequence
regardless: RESULTS ROW + backup, then gate-or-perf per the measurement.

## 2026-07-11 14:45 — 128K smoke take-4: the COMPILE itself is the story so far (2h+ single-pass, still burning)

Take-3 stalled during w-0 disk pressure (killed; possibly prematurely — lesson: the stall discriminator
is the compile worker's TIME+ growth, not driver-log quiet). Take-4 (clean disks, 8/8 nodes): the w-1
compile worker has burned 2h+ CPU single-threaded on the sparse 128K/chunk-1024 jit_step — vs ~45min for
every prior program incl. the dense 128K gate. Whatever pass is exploding (suspect: the candidate-arena/
masked-prefill structures at 128 chunks) is ITSELF evidence for the head-split perf stage: that change
shrinks the per-shard program. DECISION RULE armed in the watcher: summary → read pace, decide gate-vs-
perf; error → diagnose; compile-idle-without-output → hung, kill+pivot to perf stage. If total compile
exceeds ~3h, pivot regardless — the gate cannot ride a program this fragile.
STATE SNAPSHOT for continuity: rungs 1-6 CLOSED; fix tip 74d8c3225 synced 8×; upstream package staged
(docs/upstream/, owner files); dumps archived in GCS (rung2fix-w2 A+F, rung2fix-w0, rung4 dcp1+dcp2,
rung2-originals); gate protocol: UNARMED, chunk 1024, pool 68, dcp=4, depths {0.0,0.05,0.25,0.5,0.75,
0.95,1.0}×11, watchdog, extend-to-n≈130 on one miss. Pending after gate: 256K A/B same-dcp both sides;
GSM8K n≥200; GPQA (owner-gated); write-probe guard (owner directive); MTP unfreeze last.

## 2026-07-11 15:50 — PIVOT EXECUTED: 128K smoke take-4 killed at 3.7h compile (rule fired); head-split perf stage delegated

w-1's XLA compile of the sparse 128K/chunk-1024 jit_step reached 223 CPU-minutes at 100% with no end in
sight — the 3h pivot rule (14:45 entry) fired. Take-4 killed; pod idle. THE GATE IS DEFERRED behind the
head-split perf stage, which attacks all three symptoms at once: the ×78-layer bf16[64,T,512]
replicated-q all-gathers (HLO census), the ~10× dcp step cost (32K: 715s vs 73s/needle), and the
super-linear compile. Perf-stage agent briefed and running (scratchpad; gated GLM_DSA_DCP_HEADSPLIT,
byte-identical off, CPU equivalence at dcp=1/2, full suites; diff + local commit as deliverable).
SEQUENCE ON ITS RETURN: adversarial review → land → sync+pin → 32K A/B (headsplit on/off: step time +
needles + selected-set sanity) → re-try the 128K mechanism smoke (expect sane compile) → THE GATE
(unarmed, chunk 1024, pool 68, dcp=4, 7 depths × 11, extend-to-n≈130 on one miss) → 256K A/B same-dcp
→ GSM8K n≥200 → GPQA (owner-gated) → write-probe guard → MTP unfreeze. Dense 128K gate (77/77, run 124)
and sparse-to-64K (rungs 1-6) remain BANKED and pushed.

## 2026-07-11 17:40 — HEAD-SPLIT DELIVERED (scratchpad 0ed33e3e on 74d8c3225); adversarial review in flight

Implementation: GLM_DSA_DCP_HEADSPLIT=1 (trace-time, default OFF) head-shards ONLY the two attend
shard_map bodies (_dcp_decode Stage-B, _dcp_prefill Stage-C) via P(None,('model','expert'),None) in /
P(('model','expert'),None,None) out; _dcp_lse_combine untouched (elementwise in heads); indexer/scoring
DELIBERATELY unsplit (score-map psum would cost more than the replicated boundary saves — indexer params
are replicated P() — and head-partial fp32 summation would reorder near-ties, breaking the rung-4
elementwise criterion); owner-scatter/selection/dense untouched. Arithmetic: dcp=2 per layer/chunk —
replicated-q 65MB+8MB → 2.4MB (~30×), LSE psum 134MB → 8.4MB (16×), attend FLOPs/chip ÷16; per-shard
head extent 64→4 (the suspected compile feeder). Tests 54/0: dcp suite 28 (incl. extended jaxpr-hash),
NEW headsplit suite 10 (byte-identity off; on-vs-off bitwise at (2,1)/(2,2)/(4,2) decode + Stage-C +
mixed, selections/caches bitwise everywhere; outputs rtol 2e-5 ONLY at the H_local=1 CPU cells —
XLA-CPU single-head contraction-order artifact, production never reaches H_local=1), coverage 16.
Diff: scratchpad/headsplit.diff. REVIEW IN FLIGHT (attack list incl. independent reproduction of the
H_local=1 artifact, sharding-axis-set equality vs sharding.py, preseeded-path safety).
ON SAFE-TO-LAND: apply+byte-check vs scratchpad tree → fast suites → commit+push → sync+pin 8× → add
GLM_DSA_DCP_HEADSPLIT to bench/engine.py warn lists (implementer couldn't, outside checkout) → POD:
32K A/B on/off (step time expect ~715s→~100s; needles exact; scores-armed selected-set sanity; HLO
census: bf16[64,T,512] all-gather class GONE) → 128K mech smoke (compile time = the pivot's success
metric) → THE GATE (unarmed, chunk 1024, pool 68, dcp=4, 7 depths×11). MTP note: preseeded+headsplit
must be CPU-tested when MTP unfreezes.

## 2026-07-11 19:05 — HEAD-SPLIT LANDED: edc7d726b, reviewed SAFE-TO-LAND (all attacks verified incl. independent H_local=1 repro), synced 8×

Byte-identity vs the reviewed scratchpad tree confirmed on all 3 files; 54/54 twice (implementer +
reviewer independently). GLM_DSA_DCP_HEADSPLIT added to bench/engine.py warn lists. NEW PIN: edc7d726b.
NEXT POD SEQUENCE: relaunch with HEADSPLIT=1 + scores armed (events filter) + GLM_DUMP_STEP_HLO → 32K
dcp=2 run = the A/B ON side (OFF baseline = rung-4 take-2: 715s/needle, needles exact, dumps in GCS
dumps/rung4/dcp2): criteria = needles exact + step time (expect ~715s→~100s) + selected-set sanity vs
the OFF dumps + HLO census (bf16[64,T,512] all-gather class GONE; if step time disappoints, profile the
emit_lse kernel at H_local=4 BEFORE blaming collectives — reviewer item 3) → 128K mech smoke (compile
time = pivot success metric; chunk 1024, pool 68, dcp=4, UNARMED) → THE GATE. MTP unfreeze checklist
now includes the reviewer's preseeded-trace test battery (hash/equivalence/refusal/tolerance).

## 2026-07-11 21:10 — HEADSPLIT A/B at dcp=2 FAILED (0/3 pred=None); kernel EXONERATED at H=4 and H=8 on metal; deciding shape = dcp=4

A/B ON-side (32K, dcp=2, H_local=4): 3/3 pred=None at ~521s/needle — clean serving, garbage output; no
disk/mosaic errors. The ladder caught it pre-gate. Single-chip metal units: probe_lse_unit PASS at H=8
AND at H=4 (artifacts rung1-lse-unit-metal-headsplit-h{8,4}.txt) → the sparse-decode/emit_lse kernel is
NOT the defect; the fault is in the head-sharded shard_map composition on metal at H_local=4 (spec
slicing/reassembly under GSPMD — CPU-blind, reviewer risk 1). HEADSPLIT is gated default-OFF: production
unaffected. DECISION: the gate runs at dcp=4 = H_local=8, a DIFFERENT shape — test it directly via the
128K mechanism smoke with HEADSPLIT=1 (also the compile-time metric). If dcp=4 passes needles+compile →
add a trace-time refusal for H_local<8 (known-broken shape) and proceed to THE GATE; if it also fails →
headsplit OFF everywhere, hunt the composition defect before any gate (compile infeasibility stands).

## 2026-07-11 23:55 — 128K smoke (headsplit, dcp=4): needles 1-2 EXACT incl. both hard mechanism cells; pace verdict = GATE INFEASIBLE AS-IS

d=0.0 → 705269 ✓, d=0.05 → 824794 ✓ (the two cells where the needle must survive top-2048 selection out
of 128K at max distance) — headsplit at H_local=8 (dcp=4) is CORRECT on metal; the dcp=2 H_local=4
failure is shape-specific (kernel exonerated at H=4 AND H=8 by single-chip units; the fault is the
GSPMD composition at that shape — refusal to be added, hunt deferred). BUT: needle 2 took 7512.9s ≈
needle 1's 7514.9s ⇒ ~125 min/needle STEADY STATE ⇒ n=77 ≈ 160h. INFEASIBLE.
DOMINATOR NAMED: prefill at 17 tok/s (vs ~140 tok/s dense at 128K; decode 1.1 tok/s but only ~20
tok/needle) — 99.7% of needle time is sparse PREFILL: the per-chunk distributed-selection cost (the
[T, dcp·k] candidate score+position all-gather per FULL indexer layer per chunk ≈64MB at dcp=4/T=1024,
×21 layers ×125 chunks, + merge + per-shard segment gather). The attend-side headsplit worked; the
selection arena is the next wall. PLAN: (1) smoke-5 finishes needles 3-4 (banked regardless); (2) perf
stage 2 design delegated — the candidate-arena diet (bf16 scores on the wire w/ tie-band analysis vs the
rung-4 criterion, fused score+position packing, chunk-2048 retry now that headsplit shrank the attend
transient, gather formulation review vs the 32K A/B HLO); (3) gate DEFERRED until ≤~30min/needle
(n=77 ≈ 38h) is in reach. Honesty: the sparse stack is CORRECT to 128K on the mechanism cells; what
remains is making its prefill cheap enough to afford the statistics.

## 2026-07-12 00:40 — PERF STAGE 2 DESIGNED + S1 IMPLEMENTED (scratchpad ba8aeb31); the cost model REFUTED both prior hypotheses

Cost model (numbers in scratchpad/perf2_design.md, validated: predicts 20-22s/chunk at dcp=2/32K vs
22.3s measured): collectives ≈0.1-0.5s and scoring einsum ≈1-2s per chunk CANNOT explain 60s — the brief's
5.4e14 scoring-flop figure was off ~10³ (owner note: my arithmetic, refuted by the agent — logged as the
honest correction). THE DOMINATOR: the Stage-C masked attend walks the WHOLE local stripe every chunk —
per-token page-gather duplication + unconditional f32 materialization ≈319 GB HBM/layer/chip + f32 dots
over all 68 blocks (hence the flat 60s pace regardless of kv fill). Indexer-scoring head-split re-examined
QUANTITATIVELY and rejected (≤1.2s saved vs ~3GB psum — the win isn't there; supersedes the earlier
qualitative verdict).
S1 IMPLEMENTED: GLM_DSA_DCP_PREFILL_ATTN=segment (default masked = jaxpr byte-identical) — Stage-C attend
over the SELECTED top-k segment (O(T·topk)), reusing the metal-validated Stage-B decode pipeline;
SEG_TBLOCK=512 lax.map tiling (bitwise tile-invariant). Expected 3-4.5×; composing S2 (existing
GLM_DSA_SCORER=pallas, zero code, needs metal validation + S2-band check) → 4.5-6× ⇒ 75-100 tok/s ⇒
needle ≤25-30min ⇒ GATE ≈ 32-38h. 62/62 CPU green. ADVERSARIAL REVIEW IN FLIGHT (attack list: in-chunk
causal semantics vs the walk, byte-identity off, tile invariance, headsplit composition, full battery).
ON SAFE-TO-LAND: land+sync+pin → 32K A/B at dcp=4 (NEVER dcp=2/H_local=4) → 128K mech smoke (segment+
headsplit; ≥50 tok/s target) → xprof residual → THE GATE. Smoke-5 needles 3-4 (d=0.95/1.0) still in
flight on the OLD config — they complete the 4-cell correctness picture regardless.

## 2026-07-12 04:05 — 128K MECHANISM SMOKE 4/4 EXACT (sparse, dcp=4, headsplit) — correctness at the gate geometry BANKED

d=0.0→705269 ✓, 0.05→824794 ✓, 0.95→289958 ✓, 1.0→891482 ✓ — all exact, ~7513s each (masked prefill,
the S1-superseded config). The four cells that test the mechanism hardest are green at 128K. SMOKE ≠
GATE — but correctness at the gate geometry is now established twice over (dcp=4+headsplit, flat
scatter, chunk 1024). S1 (segment prefill, 29305e185) landed+reviewed while it ran; engine now free.
NEXT: sync+pin 29305e185 → relaunch segment+headsplit → 32K dcp=4 segment sanity (needles + tok/s)
→ 128K mech smoke under segment (≥50 tok/s target = gate ~32-38h) → THE GATE.

## 2026-07-12 05:20 — S1 ON METAL: 6.1× — 104.1 tok/s prefill, needles exact; GATE ARITHMETIC RESTORED

32K dcp=4 segment+headsplit (pin 29305e185): 3/3 EXACT with the SAME passkeys as every prior config
(422181/663295/648060 — cross-config answer stability now spans dcp=1/dcp=2-masked/dcp=4-masked/
dcp=4-segment), ~306s/needle, **prefill 104.1 tok/s vs 17 masked (6.1×)** — the cost model's dominator
call (the masked whole-stripe walk) confirmed by the fix working. S2 (pallas scorer) not even needed for
the target. Projection at 128K: ~21min/needle ⇒ n=77 ≈ 27h. NEXT: 128K mech smoke under segment (4
needles ×1, correctness+pace at gate length) → THE GATE (unarmed, chunk 1024, pool 68, dcp=4,
segment+headsplit, 7 depths×11=77, extend-to-n≈130 on one miss).

## 2026-07-12 08:00 — SEGMENT AT 128K: request-boundary defect — req 1 EXACT, reqs 2-3 pred=None; the smoke did its job

seg_128k_smoke: needle 1 (d=0.0) EXACT (705269, cross-impl answer stability) at 3225.7s (~54min steady
⇒ segment gate ≈ 69h — the 32K 104 tok/s does NOT hold at 128K; the O(S) scoring term grows — noted,
secondary). THEN needles 2 (d=0.05) and 3 (d=0.95) pred=None — same engine, sequential requests, no
disk/raylet errors (w-0 was at the monitor edge 5.0G, freed to 6.7G — cannot fully exclude, but the
1-good-then-bad pattern is structural, not pressure-shaped). MASKED at the identical geometry was 4/4
across the same sequential-reuse pattern ⇒ the defect is segment-specific and request-boundary-shaped:
128K pool = 68 blocks, request 2 reuses request 1's freed blocks in allocator order — hypothesis: a
gather step in the segment chain addresses pages arithmetically (pos//page_size) instead of through the
block table, correct only on the fresh-pool identity layout. 32K 3/3 doesn't contradict (small pool may
re-allocate identically). CPU repro agent launched (adversarial permuted block tables, 2 sequential
requests, segment-vs-masked bitwise). Needle 4 left running (sharpens the signature free). FALLBACK IS
REAL: masked = correct 4/4 at ~125min/needle (gate ~160h — the ugly backstop). GATE BLOCKED pending the
fix; the ladder's cheap-rung discipline caught this BEFORE 77 needles were burned.

## 2026-07-12 09:10 — CPU EXONERATED with a mutation canary (59/59); D0 kills the APC suspect; D2(i) TBLOCK=256 in flight

Repro agent verdict: NOT-REPRODUCED — the segment chain resolves every page THROUGH the block table
(code-read: gather_kv_segment_local sparse_mla_kernel.py:534-541; owner-scatter mla_attention.py:675),
and an 8-test adversarial battery (reversed/rotated block reuse over STALE data, pool-permutation
bitwise-invariance, APC-hit shape, headsplit composition) passes — non-vacuously: a mutation canary
planting EXACTLY the hypothesized arithmetic-page-map bug is caught grossly by the same fixture.
59/59; tests staged (scratchpad 42c3649c, perf2_adv_tests.diff) for landing with the eventual fix.
Smoke final: 1 exact + 3 pred=None (d=0.05/0.95/1.0) — request-boundary signature cemented. D0 (free):
enable_prefix_caching=False, 0 hits ⇒ APC suspect DEAD. Remaining suspects (metal-only): (1) the
donated-cache RMW→flat-1D-gather interplay inside lax.map (the pageloop-class precedent), (2) the
headsplit×segment Mosaic composition at these shapes. DISCRIMINATOR LADDER RUNNING: D2(i) TBLOCK=256
(flips ⇒ the slab/gather lowering — and a working config); next D2(ii) HEADSPLIT=0+segment; then D1
armed seg-vs-masked pair (write-side vs attend-side localization). Each 128K probe ≈3.5h. Masked
backstop stands (4/4 correct, ~160h gate).

## 2026-07-12 12:40 — D2(i) VERDICT: TBLOCK is BEHAVIOR-CHANGING on metal (CPU-bitwise-invariant) — the lax.map slab/gather lowering indicted; failure re-framed as DECODE-STREAM death

TBLOCK=256: request 1 went EXACT→TRUNCATED ('70526' = first 5 digits of 705269 — retrieval CORRECT,
generation died mid-answer); request 2 pred=None again. Two conclusions: (1) a CPU-bitwise-invariant
tile size changes metal behavior ⇒ the segment slab/gather lowering (lax.map + flat 1-D gather against
the DONATED cache) is the defect class — the pageloop family, third sighting; (2) pred=None ≈ the same
corruption expressing at token 0: the needle is likely RETRIEVED but the decode stream dies — cumulative
layout-lottery damage (request # and TBLOCK both shift layouts). D2(ii) IN FLIGHT: segment+HEADSPLIT=0
(a pass = working fast config immediately; a fail = the slab lowering alone suffices). Then D1 armed
pair for byte-level localization if needed. The evidence chain for upstream report #3 is accumulating.

## 2026-07-12 14:00 — D2(ii) VERDICT: headsplit×segment METAL COMPOSITION is the defect; segment-alone is CORRECT at 128K (reqs 1-2 exact) — GATE CONFIG FOUND

segment + HEADSPLIT=0: requests 1 and 2 both EXACT (705269/824794), needle 3 in flight — the request-
boundary failure is GONE with headsplit off. Combined with D2(i) (TBLOCK behavior-changing on metal,
CPU-bitwise-invariant): the defect = the head-sharded shard_map specs wrapping the segment attend's
lax.map/flat-gather — mislowers on v4, layout-lottery expression (request # and tile size both shift
layouts; the third member of the pageloop defect family). CPU cannot see it (59/59 incl. the bitwise
composition cell). Cost check: headsplit adds NOTHING under segment at 128K (3228s vs 3225s/needle —
segment already removed the big attend; the O(S) scoring/top_k terms dominate the residual) ⇒ dropping
it is FREE. **GATE CONFIG: dcp=4, GLM_DSA_DCP_PREFILL_ATTN=segment, HEADSPLIT unset (off), flat scatter,
chunk 1024, pool 68 — ~54min/needle ⇒ n=77 ≈ 69h (~3 days).** Doctrine: headsplit stays default-OFF with
a known-broken-composition note (its solo win was at 32K decode-side; revisit post-gate with the D1
armed protocol + upstream report #3). Plan: on D2(ii) 3/3 → RESULTS row + docs → LAUNCH THE GATE as 7
sequential per-depth runs (11 trials each, ~10h/run — crash-resilient checkpoints, same statistics,
77 cells total, extend-to-n≈130 on any miss). During the gate (pod busy): CPU threads — upstream #3
evidence staging, the write-probe guard, the composition hunt.

## 2026-07-12 14:10 — 🚀 THE 128K SPARSE GATE IS RUNNING (n=77)

D2(ii) closed 3/3 EXACT (705269/824794/289958) — the gate config validated end-to-end. LAUNCHED:
~/glm-run/gate_sparse128k.sh (setsid orchestrator, pid logged) — 7 sequential depth runs × 11 trials
(mechanism cells 0.0/0.05/0.95/1.0 FIRST, then 0.25/0.5/0.75), dcp=4 + segment + NO headsplit + flat
scatter + chunk 1024 + pool 68 + pin 29305e185, UNARMED. ~10h/depth ⇒ ~69h total. Retry policy: one
infra retry per depth after ray relaunch (no-SUMMARY only); a needle MISS is a recorded result — the
77/77 → Wilson LB 95.3% bar, ONE miss ⇒ extend to n≈130, never rerun-until-green. results.db backed up
to GCS after each depth. Watchdog armed (completion / miss / orchestrator-death). During the gate:
CPU-side threads queue — upstream #3 (headsplit×segment composition) evidence, the write-probe guard,
MTP-unfreeze test battery prep.

## 2026-07-12 20:30 — OWNER PIVOT: gate STOPPED (5/5 banked); state FROZEN; EFFICIENCY CAMPAIGN opened

Owner directive: 54min/needle is unacceptable — maximize efficiency (no bottlenecks, max throughput, min
latency), audit + ideas + adversarial review first, then apply + verify; observability-first per
docs/suggestions.md. EXECUTED: (1) gate orchestrator + drivers stopped — 5/5 depth-0.0 needles banked in
results.db (valid data, resumable); (2) FROZEN: tag freeze-correct-128k-20260712 on BOTH repos (fork
29305e185) + results.db + partial gate logs → GCS; (3) the efficiency-audit WORKFLOW launched (4 phases:
Map ×4 readers / Ideas ×5 lenses / adversarial Verify ×14 / Synthesize → scratchpad/eff_campaign_plan.md);
(4) MEASUREMENT-FIRST: free two-point analysis of existing logs — 32K 9.6s/chunk vs 128K 25.8s/chunk ⇒
~4s fixed + ~0.34s/1K-fill: THE O(S) TERM IS 84% OF CHUNK TIME AT 128K; scoring flops explain only ~0.6s
⇒ lax.top_k full-sort over [1024,S_local]×21 layers is the prime suspect (the cost model's flagged
unknown). xprof measurement run LAUNCHED (PHASED_PROFILING_DIR, single-device traces, 32K needle, current
config) to confirm op-level. Correctness state preserved: everything to date (dense 77/77, sparse 4/4+3/3
at 128K, rungs 1-6) stands; the campaign is gated+verified per the standing method.

## 2026-07-12 20:45 — AUDIT DELIVERED (38 candidates, 14 adversarially verified) + P0.a ADJUDICATED: the 258-wide block table IS the O(S) monster

Workflow verdict (24 agents; full plan scratchpad/eff_campaign_plan.md): the top finding was a
DISCOVERY, not a tune — the vLLM block table is cdiv(max_model_len, spec block_size 512)=258 entries
wide while the engine allocates ids at 512·dcp granularity (65 live at dcp=4). P0.a jaxpr dump PROVES
the DSA path pays the padded width: paged_indexer_scores' lax.map = 258 SERIALIZED page-steps/layer/chunk
(×21 layers — the measured ~21s O(S) residual), hierarchical_topk = 33 groups over 132,096 cols vs 9
over 33,280 (top_k itself is already hierarchical — the monolithic-sort hypothesis dies; the SERIALIZED
page loop is the killer). W2.1 (owned-width slice, exact semantics — dead tail is zero-id + kv_len-
masked) projects chunk 25.4→~9.5s ⇒ ~107 tok/s @128K (2.7×) + decode 2-3×; gate 69h→~26h. Implementation
agent launched (gated GLM_DSA_BT_WIDTH, default full/byte-identical; bitwise CPU proof; pod protocol incl.
armed selections-bitwise + 2-request + 4-depth smoke). Wave-1 quick wins queued (launcher bucket default
fix, compile-cache verification). Wave-3 owner decisions flagged: chunk-2048, APC (protocol-changing),
dcp=8 probe. xprof 32K run still in flight (P0.b confirms op-level + decode classes).

## 2026-07-12 21:30 — OWNER RATIFIED the Wave-3 decisions (recorded in auto-memory + here)

(1) CHUNK 2048: ADOPT (option a) — pending one compile probe + a 128K smoke; the 5 banked chunk-1024
gate needles are DISCARDED by consequence; the gate runs at 2048 if the probe passes, else 1024.
(2) APC: option c — OFF for the gate (write-path coverage stays whole; 2 of 3 silicon bugs lived
there); ADOPT post-gate for benchmarks after its own cache-hit-under-DSA validation.
(3) dcp=8: DEFER (option b) to the 256K stage.
SEQUENCING LOCKED: W2.1 lands+reviews → sync+pin → pod cycle A (32K armed A/B, W2.1 on/off, chunk 1024
— single variable) → pod cycle B (chunk-2048 compile probe + combined W2.1+2048 128K 4-depth smoke) →
THE GATE (dcp=4, segment, owned width, chunk per probe, APC off, n=77 fresh).

## 2026-07-12 22:15 — P0.b XPROF VERDICT: the plan's top candidate REFUTED by measurement — GATHERS are 61.9% of the step

Op-level trace (mid prefill step, 10.36s, device 99.7% busy, no host gaps): (d) GATHERS 61.9% — the
post-top_k REORDER gathers (indexer_kernel.py:609-612, [1024,8192] outputs, 21×4/step) = 3465ms/33.4%
at ~2 GB/s effective (the KV payload gather nearby runs 440 GB/s — a ~200× layout/lowering pathology);
per-layer selected-index gather (mla_attention.py:766, ×78) = 1322ms/12.8%; segment KV s32 INDEX gather
(sparse_mla_kernel.py:534) 2.3× the payload it addresses. (c) collectives 11.4%; (b) top_k 8.9%
(f32[1024,32768], S-dependent → ~2.8s at 128K); (a) the serialized scoring loop I indicted = 3.8%.
The 32K trace ran at table width 64 (width scales with max_model_len) — W2.1 remains EXACT + worthwhile
(kills the width-scaled ~14% at 128K + the 258-anomaly) but its 2.7× projection is DEAD. Honest note:
the audit's adversarially-verified top candidate was mis-sized; only the instrument caught it —
suggestions.md discipline, again. Residual: trace-predicted 128K step ≈17s vs measured 25.4s — ~8s still
unattributed at 128K (128K-specific trace queued). Decode trace missing (prefill_only phase captured).
RETARGET: the campaign's prize is the GATHER machinery (~62% → the reorder-gather class first).

## 2026-07-13 01:00 — GATHER DOMINATOR DELIVERED (39087f8f): scalar-gather pathology root-caused; 3 gated bitwise-exact fixes, 1.9-2.3× projected

ROOT CAUSE of the 0.5 GB/s gathers: slice_sizes=(1,1) lane-dimension scalar gathers — XLA:TPU emits
sequential per-element gather_custom_fusion (the 440 GB/s neighbor fetches contiguous 640-wide rows;
slice width IS the whole pathology). FIXES (each env-gated, default byte-identical, trace-time refusal):
GLM_DSA_MERGE_IMPL=v2 — merge as ONE stable two-key lax.sort (u32 sign-flip value key + position key;
bitwise == v1 incl. tie order + concat-invariance), 3 gathers + top_k + sort → 1 sort;
GLM_DSA_OWNED_SEG_IMPL=v2 — sort the key directly (0 gathers); GLM_DSA_SEG_GATHER_IMPL=v2 — page-id
one-hot reduce with exact OOB semantics (only the fast payload gather remains). Projection bounded by
the measured class: −5.0…−5.9s of the 10.36s step ⇒ ~4.5-5.4s/chunk (1.9-2.3×), S-INDEPENDENT, composes
with W2.1 (67749faa, its own review finishing). 62 new adversarial tests + batteries green (the ~250-test
single-process abort classified pre-existing compile-volume class, subdivided runs green both trees).
SIDE HYPOTHESIS registered: the segment request-boundary bug may live in the v4 lowering of the s32
index gather — SEG_GATHER v2 may change its signature (observe in cycle A). NOTE: the two agents collided
in the shared scratchpad tree (worktree isolation saved both; W2.1 on branch w21-btwidth-shared, gather on
gather-dominator; both apply cleanly to 29305e185 — integration check is in the gather review's scope).
COMBINED PROJECTION if both land + chunk-2048: 32K chunk 10.4→~4.5s and 128K step composing W2.1's width
cut ⇒ prefill ~200+ tok/s territory — the gate in ~half a day. Reviews in flight; pod cycle A next.

## 2026-07-13 04:30 — BOTH EFFICIENCY DIFFS LANDED (5c6e1f0c8 W2.1 + a98c77c9c gather dominator), reviews SAFE ×2, synced 8× a98c77c9

Gather review highlights: NaN-class bit-probes bitwise; jax-source proof that FILL_OR_DROP is an HLO
mask+select (backend-independent — the CPU/TPU OOB trap does NOT apply); both cherry-pick orders clean;
integrated cross-product (owned + all-v2) 16/16; the reused-permutation decode site correctly left v1.
block-perm test file committed (was untracked, load-bearing). POD CYCLE A: arm A0 = baseline (defaults,
armed dumps, 2 sequential 32K requests, step times); arm A1 = GLM_DSA_BT_WIDTH=owned + MERGE/OWNED_SEG/
SEG_GATHER=v2 (the combined config) — GATES: selections BITWISE == A0 (all four changes exact; any
v2-only diff ⇒ per-gate flip-back, G1 first = TPU TopK tie-order residual), needles exact, chunk
10.4→expect ~4.5-5.5s (32K), scan-trip census as the owned positive control. LATER: the 3-arm request-
boundary observation (default / seg-gather-v2-only / all-v2 — diagnostic for the OPEN segment bug, never
a fix), xprof re-capture (one-hot fusion + sort cost), 128K smoke, chunk-2048 probe (ratified), fresh gate.

## 2026-07-13 07:20 — CYCLE A: PASS — the combined config (owned + all-v2) is 2.7× on metal with selection health intact

A0 (baseline, armed): 2/2 exact, ~350s/needle. A1 (BT_WIDTH=owned + MERGE/OWNED_SEG/SEG_GATHER=v2,
armed): 2/2 exact SAME passkeys, ~130s/needle — **2.7× wall at 32K armed**. Selections A1-vs-A0:
tripwire 0, replication 0; shallow events near-clean (evt00/01: 3+3 set-diff rows, ALL in request-2 late
decode, band/drift 0.02–1.32 = boundary-tie churn); deep evt20 diffs = the cross-program drift envelope
(ratio p90 4.8, max 12.5 — A0/A1 are DIFFERENT programs; the bitwise expectation only binds same-program
pairs — criterion applied is the owner-ratified band-quantified standard). All four gates HOLD on metal.
CYCLE B LAUNCHING: chunk-2048 compile probe at the winning config (ratified) + 128K 4-depth smoke;
fallback chunk 1024. Then THE FRESH GATE at the final config.

## 2026-07-13 10:30 — CYCLE B: 4/4 EXACT at 595s/needle — 12.6× END-TO-END; chunk-2048 compiled; 🚀 THE FRESH GATE LAUNCHES

Chunk 2048 at the winning config (owned + all-v2 + segment, dcp=4): COMPILED (the pre-campaign OOM
buffer was the masked walk's — segment removed it, exactly as the re-analysis predicted). 128K smoke
4/4 EXACT (705269/824794/289958/891482 — the same passkeys across now FIVE configs), 595s/needle vs
7513s pre-campaign = **12.6×**; prefill ≈220 tok/s at 128K — SPARSE NOW BEATS DENSE (140 tok/s). The
campaign's promise delivered: measure → find (scalar gathers + dead width + serialized walk + masked
walk) → fix exactly → verify on metal. FRESH GATE: n=77, 7 depths×11, chunk 2048, APC off (ratified),
pin a98c77c9c, mechanism depths first, per-depth GCS checkpoints, extend-to-n≈130 on one miss.
ETA ≈ 12.7h.

## 2026-07-13 09:00 — HOLISTIC AUDIT (20 agents, 44 findings, 12 confirmed) + GATE2 IS DEAD: d=0.95 = 11/11 MISSES, new signature

AUDIT VERDICT (full report scratchpad/eff_audit_report.md): the four transforms are REAL and CPU-exact
(12.6× measured; all five semantic claims independently re-derived; 127 shipped + 6 new combo tests
green) — but NOT PR-ready and the gate is NOT passing. CONFIRMED: F1 BLOCKER gate2 dead — d=0.95 went
0/11 with a NEW signature (fluent-filler retrieval, NOT the D2 decode-death pred=None... the driver
reports pred=None but outputs are fluent filler = the needle NOT RETRIEVED), Wilson-unrecoverable;
F2 recurrence UNATTRIBUTED (gate moved 3 variables at once vs cycle B; standing suspect: the
donated-cache payload gather in the attend lax.map, sparse_mla_kernel.py:610); F4 orchestrator RELAUNCH
env-dropout = false provenance on retry (verified not-fired in gate2); F6 headsplit×segment known-broken
combo armable with no refusal + overclaiming docstrings; F7 masked backstop has ZERO metal tokens on the
owned/v2 program it would actually run; F8 armed bitwise metal coverage only at T=1024 (gate ran T=2048).
EXECUTED NOW: d=1.0 first-trials harvested as the free depth-vs-engine discriminator, then orchestrator
KILL; disks cleaned (w-2 50G). NEXT (audit top-3, owner-aligned): F3 attribution ladder BEFORE any code
change (10-min same-instance d=0.0/d=0.95 interleave + re-issue cycle B's exact passing prompt + the two
skipped bisect arms + xla_dump compile diff; NEVER lead with armed probes — trace-time env = different
program); then the safety/truth commit (headsplit refusals, docstring corrections, combo-matrix test
port, RELAUNCH fix, miss-abort watchdog); then re-gate with one armed T=2048 cell + a masked-backstop
smoke first. The campaign's perf stands; its correctness debt is now the whole job.

## 2026-07-13 09:40 — d=1.0 DISCRIMINATOR: 2/2 EXACT on the very next engine ⇒ WHOLE-ENGINE-INSTANCE LOTTERY (rate ~1/7); gate killed; probe loop hunting

The dead d=0.95 engine (0/11, fluent-filler) sits between two perfect engines (d=0.05 11/11 before,
d=1.0 2/2 after) ⇒ per-ENGINE-INSTANCE expression, depth incidental, cumulative-degradation dead. Rate
estimate 1/7 engine draws ⇒ every 7-engine gate expects ≥1 dead depth — NO GATE PASSES UNTIL FIXED.
Gate2 killed (22 exact needles + the 0/11 + 2/2 all recorded in results.db — honest rows). ATTRIBUTION
PROBE LOOP running: 14 × single-needle 128K engines at the full config, GLM_DCP_CACHE_DUMP armed
(host-side — the traced program is UNCHANGED, respecting the audit's F3 warning about armed-topk
program perturbation), per-probe dump archival; a caught bad engine gets byte-diffed against a good
one (deterministic prefill ⇒ byte-equal unless the WRITE side corrupts) — the pageloop protocol,
adapted. ~2.5h for ~2 expected bad draws. THEN: write-side vs read-side verdict → the F3 bisect arms
on the guilty side → fix → safety/truth commit → masked-backstop smoke + armed T=2048 cell → re-gate.

## 2026-07-13 15:40 — POSTMORTEM: 4h of probe INFRA-FAILs = w-4 disk at 0 (raylet died on every join, 7/8 nodes) — and this CONFOUNDS the lottery hypothesis itself

w-4 held 55G of parked dump archives (rung4 + relics; my own parking decisions) → 0 free → its raylet
died on every one of 14 relaunches → every probe INFRA-FAILED (the fixed loop correctly classified them,
the watcher's grep pattern didn't — repaired). CRITICAL REFRAME: w-4's disk was degrading through the
EXACT window of gate2's d=0.95 0/11 AND probe-1 (the "bad engine" specimen) — the engine-instance-lottery
hypothesis is now CONFOUNDED by disk-pressure-degraded engines (the same class as every prior pred=None).
The clean experiment runs NOW with all 8 disks healthy (w-0 23G / w-2 ~40G / w-4 23G / rest 50G+):
14 fixed-seed probes — bad engines recur ⇒ real lottery (specimen p1 stands); 14/14 good ⇒ the "lottery"
was operational all along and the fixes are disk quotas + engine health-probe + the write-probe guard,
NOT kernel code. OPS DEBT NOW UNDENIABLE (4 disk incidents this campaign): a disk-watchdog hook + quota'd
dump archiver join the safety commit. Dumps parked across workers were a self-inflicted wound.

## 2026-07-16 23:55 — VM LOST + FULL RECOVERY: pod recreated (all 8 disks wiped); everything COMMITTED survived; the 14-probe experiment must restart

The worker-0 VM — and the whole pod (new hostnames t1v-n-6c15e171-w-*) — was recreated between 07-13 and
07-16; all 8 host disks wiped. The glm-tpu bucket mirror turned out to be EMPTY (gs://driftbench-storage/
repos/glm-tpu/ never existed): setup.sh [6/7] `source ~/.local/bin/env` aborts under `set -e` whenever the
uv installer skips writing that file (PATH already carries ~/.local/bin) — so [7/7], the 5-min sync cron,
silently never ran on the old VM either. FIXED in the bucket setup.sh (fallback PATH export;
setup.sh.bak-20260716 kept). **GIT PUSH DISCIPLINE HELD:** a 26-agent recovery audit confirms
origin/glm-5.2-v4-next @ a98c77c9 is the newest commit on all 39 fork refs (git log --all since 07-12
23:19 over every ref: empty; no dangling commits), pr-g1..g6 intact at their 07-07/08 cuts, tag
freeze-correct-128k-20260712 → 29305e185 verified on the live remote.
RESTORED today: glm-tpu @ fdb6e7f + fork @ a98c77c9 (fresh clones, editable install); vLLM@LKG a30addc7
rebuilt into ~/vllm-build (moe-tpu build_vllm_lkg.sh); venv from the bucket tarball; **workers 1-7
provisioned 7/7 OK @ a98c77c9** (new scripts/provision_worker_glm.sh; bulk artifacts now mirrored
same-region at gs://driftbench-dsv4-uc/artifacts/); ~/glm-tpu/.env (HF token); sync cron re-armed and
verified; results.db restored from the 08:31:37Z GCS copy — its LAST row is run 165's aggregate (gate2
d=0.95 0/11): the per-depth GCS checkpoint discipline captured the gate's death seconds after it landed.
LOST (never committed — rebuild/rerun): the d=1.0 discriminator rows (2/2 exact; survives only as the
07-13 09:40 log entry), the 14-probe fixed-seed experiment (zero rows — it was starting at the last
entry), the safety/truth commit (F6 headsplit×segment refusal, docstring corrections, combo-matrix test
port, F4 RELAUNCH env fix, miss-abort watchdog), the write-probe guard (owner directive; spec survives in
docs/upstream/pageloop-v4-sublane-drop-REPORT.md), the disk-watchdog + quota'd dump archiver,
scratchpad/eff_audit_report.md (F1-F8 detail; the 07-13 09:00 summary above survives), all XLA caches
(first engines recompile from scratch) and the ~/glm-run orchestrators.
SILVER LINING: all 8 disks now sit at ~83G free — the clean-disk precondition of the 14-probe experiment
holds by construction. NEXT (unchanged in substance from 07-13 15:40): (1) re-run the 14 fixed-seed
single-needle 128K probes (full gate config, cache-dump armed host-side) — bad engines recur ⇒ real
lottery; 14/14 good ⇒ the "lottery" was disk-pressure all along; (2) verdict → write-side hunt vs
ops-fixes-only; (3) the safety/truth commit + disk-watchdog + dump quota land BEFORE dumps re-accumulate;
(4) masked-backstop smoke + one armed T=2048 cell (audit F7/F8); (5) re-gate n=77 (chunk 2048, per-depth
checkpoints, extend-to-n≈130 on one miss).

## 2026-07-17 04:15 — THE SAFETY/OPS-DEBT COMMIT LANDED (fork a98c77c9→845f4ffeb, synced 8×) + the discriminator REDESIGNED by its own review

The queued-then-lost safety commit is rebuilt, adversarially reviewed (6 lenses, 2 BLOCKER + 8 MAJOR
found and fixed), and landed. Fork commits: **c9d87919** — F6 metal-verdict refusals
(`_glm_dsa_dcp_headsplit_axes` now refuses H_local<8 AND headsplit×segment at trace time, evidence
cited; `GLM_DSA_DCP_HEADSPLIT_UNSAFE=1` = the documented isolation-diff override; docstrings
de-overclaimed) + the dense `GLM_DCP_SCATTER_IMPL` unknown-value loud refusal (the silent else-fallback
was the metal-BAD plain scatter — gap exposed by the write-probe work); **fd0bf456** —
`GLM_WRITE_PROBE` startup sentinel through the REAL owner-scatters at KV-cache init (dcp_guards idiom;
refuse-to-serve on any never-written/wrong-value/clobbered coordinate; mutation canaries prove
detection); **1152db21** — the permanent combo-matrix suite (full gate config owned+all-v2+segment
bitwise-selections/caches vs defaults, F8 T=2048 cell closed; the lost scratchpad combos rebuilt);
**fb5000ba** — review fix: the DSA probe leg covers BOTH row widths (128 indexer + 640 latent — the
width-specific defect class) + the dense typo-refusal test; **845f4ffeb** — strip hygiene + hook
exception attribution. Suites: headsplit 12/12, segment 12/12, decode 28/28, block-perm 8/8,
scatter-gt 41P/4S, write-probe 25/25, combo 8/8, bench 15/15.
REVIEW HIGHLIGHTS (docs/suggestions.md vindicated again): (1) BLOCKER — the original 14-probe design
COULD NOT answer its own question: identical back-to-back fixed-seed engines inherit stale≈fresh HBM
(the C-vs-D blind spot) and mask the never-written class ⇒ **scrambler interleave** added (each counted
draw preceded by a different-content/-layout 32K engine); (2) BLOCKER ×4 — a disk-tainted MISS still
flipped the headline verdict ⇒ taint-ordering fixed (INFRA misses never counted); (3) N=14 had an 11.6%
false-negative vs a 1/7 lottery ⇒ **N=20 valid draws** (95% power), honest confidence wording; (4) dump
coverage 0,1,2 = ~3% of the defect-class buffer family ⇒ all 21 indexer k-cache slots + mla0;
(5) dump_archiver could purge unarchived bytes on a mid-stream tar failure ⇒ pipefail + upload verify;
(6) gate orchestrator folded tainted misses into the Wilson counter ⇒ depth-INFRA abort semantics.
PROCESS NOTE: a reviewer's mutation audit transiently broke the shared tree under other reviewers'
concurrent suites (combo 4/8, pair 10/20 at 01:22) — clean simultaneous repro fully green (8/8, 20/20);
adjudicated NO-DEFECT; standing rule: mutation audits run in isolated worktrees.
OPS SCRIPTS (in-repo now, never ~/glm-run-only again): disk_watchdog.sh (check + watch + alert flag),
dump_archiver.sh (GCS-or-delete, quota'd, --last-step-only), **probe_lottery.sh** (the redesigned
discriminator: 20 scrambled draws, gate2-verbatim config from run-165 env_json, fixed seed = gate2's
first d=0.95 needle, INFRA-vs-verdict classification, per-probe GCS archival),
gate_sparse128k.sh (F4 single env source, live miss-abort at 2, depth-INFRA taint, per-depth
checkpoints, GLM_WRITE_PROBE armed). Workers synced 8× 845f4ffeb dirty=0; default-trace byte-identity
of the whole stack confirmed by the review (SAFE-TO-LAND on that lens). NEXT: launch probe_lottery.sh
(first engine pays the cold XLA compile), then per its verdict → F7/F8 cells → RE-GATE.

## 2026-07-17 03:15 — Draw-1 INFRA: pod recreation had orphaned the Ray firewall rule (fixed; the classifier worked)

probe_lottery draw 1 came back INFRA:ray_nodes_1 (both scrambler and probe): workers 1-7 joined rc=0
but never appeared — /tmp/rayjoin.log: "Failed to connect to GCS at 192.168.0.29:6379". Root cause:
`allow-ray-pod-internal` carries a PER-POD-INSTANCE target tag, and the 07-16 recreation minted a new
tag (tpu-t1v-n-6c15e171-w-5201142156555843955) — the rule matched nothing, so the private-fabric Ray
ports were closed. Fixed with `gcloud compute firewall-rules update allow-ray-pod-internal
--target-tags=<current tag>` (tag read from the metadata server:
`curl -H "Metadata-Flavor: Google" .../instance/tags`). **STANDING RULE: after ANY pod recreation,
re-point this rule** — it is now part of the recreation checklist alongside reprovisioning. The
orchestrator's INFRA-vs-verdict classifier caught the condition in-protocol (no verdict pollution,
draw not counted) — the postmortem fix doing its job on its first live incident. Loop self-heals on
the next draw.

## 2026-07-17 08:20 — 🎯 THE LOTTERY IS REAL AND CAUGHT ON CAMERA: per-HOST NaN poisoning of ONE buffer (layer-1 indexer k-cache), byte-deterministic bad program, per-instance host set

**Discriminator (probe_lottery run 035115Z, pin 845f4ffeb): draw 1 CORRECT, draw 2 MISS** — same
fixed-seed needle (gate2's first d=0.95 cell, prompt_tok=127363 identical), scrambler-interleaved,
disks clean, all-green infra. MISS signature = gate2's exactly: pred=None, fluent-filler
("The grass is the sun...") — run 171; probe1 answered '289958' crisply (run 169). The bad engine was
also 2.6× slower (3690s vs 1394s/needle).

**BYTE-DIFF FORENSICS (probe1 CORRECT vs probe2 MISS, all 22 dumped slots × 8 hosts × 4 shards):**
- slots layer0 (idx0) + layer1 (mla0): byte-EQUAL everywhere. Slots ≥4: massive FINITE divergence
  (50-90% of elements, all hosts) = downstream cascade, no NaN anywhere.
- **THE SOURCE: slot layer2 = model-layer-1's DSA indexer k-cache.** NaN census: probe1 → host w4
  poisoned (all 4 chips, 92%); probe2 → hosts w3, w4, w7 poisoned (91-92%). **The poison unit is a
  whole HOST; the per-instance lottery is WHICH hosts.** 1 bad host → retrieval survives (7/8 model
  replicas' attention contributions dominate the o_proj psum); 3 bad hosts → fluent filler. Gate2's
  engine-lottery, mechanism in hand.
- **Geometry:** within a poisoned replica: page 0 CLEAN, pages 1-63 100% NaN, pages 64-67 (beyond
  fill) clean zeros. At dcp=4 each 2048-token chunk writes exactly one logical page, and **chunk 0
  runs the ctx≤topk DENSE FALLBACK while chunks ≥1 run the sparse path** — the poison is the SPARSE
  path's layer-1 indexer-k value computation, whole-chunk, from chunk 1 onward. Uniform across all
  sublanes/packs/128 cols. NOT the pageloop sublane-stripe class; NOT stale HBM (fill = canonical
  quiet-NaN 0x7fc0, 4.1M elements, single bit pattern).
- **Determinism:** w4's poison is byte-IDENTICAL across the two engines; probe2's w3 and w7 poison is
  byte-IDENTICAL on shared stripes. Device placement identical across engines (dump device metadata).
  ⇒ exactly TWO program behaviors exist — good and ONE deterministic bad — and each host draws one
  per engine instance. Prime suspect class: **per-host compilation split** (JAX_SHARE_BINARY broadcast
  vs local compile, or a compile-time autotuning/HBM-pressure-dependent choice) yielding a v4
  MISCOMPILE of one fused op in the sparse-path layer-1 indexer-k chain — the 07-07 per-host-binary
  core-halt family, now expressing as numerics. Driver-log fingerprint attribution is Ray-dedup-
  poisoned (known artifact) — per-host log forensics is the next instrument.
- NO deliberate NaN writer exists in the DSA path (grepped) — the NaN is computed+written faithfully.

**Consequences:** (1) gate2's death, the d=1.0 2/2 recovery, and probe1-with-w4-poisoned-yet-correct
are all the SAME mechanism at different draw counts; (2) an ENGINE HEALTH PROBE at init (2-chunk
mini-prefill + NaN check on the layer-1 idx cache — ~1 min) can DETECT a bad engine before any gate
depth burns → detect-and-relaunch unblocks the gate operationally while the compiler bug is hunted;
(3) the loop continues collecting draws (rate + host histogram). Dumps banked:
gs://driftbench-dsv4-uc/dumps/probe_lottery_20260717T035115Z/probe{1,2}/ (3.7 GiB each, 8 hosts).

## 2026-07-17 08:55 — F3 log forensics: the per-host BINARY-split hypothesis is REFUTED (ground-truth fingerprints); the discriminator is per-host RUNTIME STATE

Per-host libtpu logs (no Ray dedup; /tmp/tpu_logs, both engine sessions, copied to ~/glm-run/hostlogs/
+ evidence bundle): **probe1 = 8× concurrent LOCAL compile of all 7 jit_step_fun_impl variants with
BYTE-IDENTICAL executable fingerprints (code AND data segments) on every host — poisoned w4 ==
clean w2 for every serving program.** probe2 = pure persistent-cache-hit run (zero compiles; init
293s vs probe1's 1761s) — poisoning occurred in BOTH modes. JAX_SHARE_BINARY_BETWEEN_HOSTS=1 produced
ZERO observable behavior in any log (flag may be a no-op in this stack — the 07-07 core-halt
attribution deserves a re-look; the persistent XLA cache at ~/.cache/vllm/xla_cache is what actually
uniformizes warm engines). Cache-state divergence exists but is tiny and non-correlating ({w1,w4}
extra compute_logits entry ≠ poison sets).
**KILLER FACT: w4 computed its byte-identical NaN poison while provably executing the same
executables as the clean hosts.** ⇒ the split is per-host RUNTIME STATE consumed by the sparse-path
layer-1 indexer-k chain. NaN is ABSORBING (canonical 0x7fc0), so byte-identical poison across engines
is consistent with varying per-host garbage inputs collapsing to NaN. Pages 64-67 = clean zeros on
the same replica ⇒ the buffer was zero-initialized and pages 1-63 were WRITTEN with computed-NaN
values (not never-written).
Free extra datum: draw-3's SCRAMBLER (32K, chunk 1024, different seed) also MISSED — the class
expresses at 32K too.
NEXT INSTRUMENTS (the forensics agent's prescription): (1) dump the layer-1 indexer-k chain INPUTS
(hidden, wk, scales) on the first sparse chunk of a poisoned host — input-borne vs computed; (2) one
draw with per-host --xla_dump_to for buffer-assignment diffs (fingerprints don't cover allocation);
(3) one draw with VLLM_DISABLE_COMPILE_CACHE=1 to fingerprint what cache-hit engines actually load.
OPERATIONAL UNBLOCKER (gate path, mechanism-independent): the ENGINE HEALTH PROBE — 2-chunk
mini-needle at init + NaN scan of the layer-1 idx cache (~1 min) ⇒ detect-and-relaunch bad engines
before any depth burns. The loop continues (rate + host histogram; every draw is now warm-cache).
