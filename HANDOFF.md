# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-07 — **Stage 1 (dense-MLA correctness) is sub-cube GREEN at T ≤ 128 and BLOCKED on an
open T > 128 divergence (IN PROGRESS, see §Open items). Weights are staged; the pod has not been touched yet.**

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `PLAN.md` (phases + thresholds) →
`docs/00-feasibility-memo.md` (the config-verified GO study) → `docs/RESEARCH_LOG.md` (latest entry) →
`docs/01-dsa-kernel-design.md` (the Stage-2 DSA kernel design — already written) → `docs/recon/*.md`
(9 recon summaries: PR #2324 full map, fork layout, DSV4 docs 09/11/12/13/15/16, GLM references) →
`docs/reviews/stage1-*.md` (the 4-lens adversarial review of the Stage-1 diff). Then the **DSV4 base**:
`~/moe-tpu/CLAUDE.md`, `~/moe-tpu/HANDOFF.md`, `~/moe-tpu/docs/{09,11,12,13,15,16}` + the fork branch
`~/tpu-inference@dsv4-flash-v4`. **Don't reinvent — CLAUDE.md §"What transfers" maps every reusable piece.**

---

## State (what is DONE and validated)

- **Weights staged (DONE):** `zai-org/GLM-5.2-FP8` → `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` —
  **150/150 files, 755.7 GB, size-verified, 0 failures** (~45 min @ ~300 MB/s aggregate). Key-set
  fingerprint committed at `configs/glm-5.2-fp8-keyset.json`: **118,629 params**; `indexers_proj` does
  **NOT** exist (the quant-config entry is a red herring); indexer weights ship ONLY on the `full`-schedule
  layers (0,1,2,6,10,…,74) + MTP layer 78 — IndexShare reuse is structurally required in Stage 2 (docs/01 §5).
- **Fork branch `glm-5.2-v4`** (off `dsv4-flash-v4` @ `17d635a1`), 4 commits, all adversarially reviewed:
  - `9bb2c23e` — register `GlmMoeDsaForCausalLM` (+`_PP_DISABLED_MODELS`), `TPU_DISABLE_DSA_INDEXER` gate,
    `_maybe_patch_for_glm_moe_dsa` (indexer only on `full` layers; cpu device for the topk buffer), and
    **2 latent upstream MLA-wrapper bug fixes**: kv_cache_dtype=auto NaN (identity scales) + W_UV_scale
    sharding axis (`P(None, ATTN_HEAD)`).
  - `8ad980e7` — FP8-on-v4: gmm_v2 routes FP8 RHS through in-VMEM dequant (gated `tpu_generation()==4`);
    checkpoint-exact block-scale linears (ceil scale columns for non-aligned fused parts; ragged-aware
    2D-scale expansion in `xla_quantized_matmul`).
  - `3f745ddc` — review fixes (kv_cache_spec indexer-cache skip [engine-boot HIGH], indexer gate default-ON,
    Ray env propagation, test repair 6/6) + **2 real gmm_v2 tiling bugs** the real-dims run caught
    (dequant-buffer VMEM modeling; the at-floor tile_k-shrink dead branch).
  - `cd8eeb6c` — port of **PR #2324's TP-topology MLA fixes** (cross-shard all-gather on MLP_TENSOR inside
    the shard_map, v4 block sizes (1,1,1)/(1,8,8) + s_dtype=f32, page_size 512 for kv_lora>256, EP-head
    gather before o_proj, MLA no longer hard-requires DP attention). 1-chip regression re-PASSED after it.
  - ⚠️ **UNCOMMITTED WIP in the fork** (4 files, part of the T>128 investigation — do NOT commit blind):
    move the cross-shard all-gather BEFORE the TuningKey (key must see post-gather shapes); gate page-512 to
    v4 only; `TPU_MIN_TOKEN_BUCKET` env in `tpu_runner.py` (default 0 = byte-identical; launcher sets 512)
    + its Ray propagation. Validate, then commit with the T>128 fix.
- **Parity harness green (machine-gated, exit-code, 1 chip)** — `parity/glm_engine_{common,parity}.py`:
  production stack (VllmModelWrapper → vLLM GlmMoeDsa → fork MLA wrapper → mla.v2 Pallas + fused-MoE GMM)
  vs transformers 5.12 `GlmMoeDsaForCausalLM` (fp32 + bf16 controls), synthetic native-key checkpoint
  (HF loads 0 missing/unexpected). Exact commands (from `~/glm-tpu`; single-chip pin is REQUIRED — the
  harness hand-builds single-shard metadata; multi-chip goes through the real runner on the pod):

  ```
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_DISABLE_DSA_INDEXER=1 \
  ~/vllm-env/bin/python parity/glm_engine_parity.py                      # mini bf16 (5L, T=32)
  … glm_engine_parity.py --fp8                                           # + DISABLE_WEIGHT_REQUANTIZATION=1
  … glm_engine_parity.py --fp8 --real-dims                               # 3L, real GLM dims, fp8 block 128
  … glm_engine_parity.py --two-step            [--split-at N]            # cached prefill+decode, reversed BT
  … glm_engine_parity.py --fp8 --two-step
  ```

  Results (all PASS, re-confirmed after cd8eeb6c — the v4-blocks + fp32-scores path is now exercised):
  - mini bf16: final_hs 0.344 < floor 0.375, logits 0.106 < 0.127, top-1 1.000 vs bf16 ref.
  - mini fp8 block-64 (HF twin = quant-dequant roundtrip, same effective weights): top-1 1.000/1.000.
  - **real-dims fp8 block-128** (H6144, 64h, nope192/rope64/v256, lora 2048/512, idx 32×128):
    final_hs 0.619 < 0.722, logits 0.910 < 0.958, top-1 vs fp32 0.906 > bf16-ref 0.875. Also PASS at T=128.
  - **two-step** (prefill [0,split) + continuation via the paged KV cache, reversed block table):
    `exact_bit_match=True` in both bf16 and fp8 at T=32.
- **Stage-1 serving config settled** (baked in `scripts/launch_glm_32chip.sh` ENVS): `NEW_MODEL_DESIGN=1
  MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 TPU_DISABLE_DSA_INDEXER=1` (dense MLA),
  `DISABLE_WEIGHT_REQUANTIZATION=1` + **`REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn` pinned** (FP8 stays
  checkpoint-resident; explicitly NOT `bfloat16` — that OOMs at 753B), `TPU_MIN_TOKEN_BUCKET=512`
  (32-way token-shard divisibility, PR #2324's validated value), **kv-cache dtype auto** (bf16 + identity
  scales; FP8-KV NaNs under EP per PR #2324), **pure TP×EP, NO DP-attention** (no `additional_config`
  sharding — the 753B attention weights cannot replicate; PR #2324's validated topology),
  `load_format=runai_streamer` from the us-central2 bucket (`RUNAI_STREAMER_CONCURRENCY=32`, 32 GiB limit).
- **Launch/sync/bench wiring EXISTS but is POD-UNTESTED** (never executed on the 8 hosts):
  `scripts/launch_glm_32chip.sh` (DSV4-cloned 3-phase launcher, env-baked raylets, `--dry-run`, EXTRA_ENVS
  hook), `scripts/sync_workers.sh` (ff-only fetch/checkout/pull on all 8 hosts, drift echo),
  `bench/run_bench.py make_generate()` (in-process `vllm.LLM`, GLM's own chat template → token ids, greedy,
  stop ids [154820,154827,154829], full env provenance into `results.db`; `--stub` + CPU tests pass —
  datasets + AIME-2026 verified and extractor gaps fixed in `ac37c8d`), and `bench/glm_longctx.py` +
  `bench/engine.py` (`e27c3c5` — the Stage-2 passkey/NIAH threshold instrument, generation-based
  retrieval, raw-completion prompts, 1M-capable, provenance-backed; CPU tests 9/9; shares the exact
  Stage-1 engine recipe with run_bench via the factored `build_llm`). None of it has touched the pod.
- **Stage-2 design is written** (`docs/01-dsa-kernel-design.md`, commit `8d069ba`): indexer math + the
  RoPE-interleave E1/E2/E3 resolution experiments, indexer k-cache KVCacheSpec, gathered `[R,2048,640]`
  decode segment, gates S/K/D0/D/P, IndexShare via the wrapper context, VMEM/HBM budget, 2a/2b/2c phasing.

## Open items (honest status — what is NOT validated)

1. **T > 128 divergence — IN PROGRESS, blocks everything downstream.** The real-dims fp8 parity FAILS as
   soon as the sequence exceeds 128 tokens (bisected on 1 chip): **T=128 PASS** / **T=136 FAIL** (layer-0
   max|Δ| jumps 6.2 → 108; final_hs 3.04 vs the 1.5×floor bar 1.22; top-1 0.860 < 0.875). T=300 two-step
   (split 130, reversed BT) fails catastrophically (top-1 0.42, two_step rel 0.14). Evidence in the session
   scratchpad (`bis_128.log`/`bis_136.log`/`reg_c2.log`); **NOT yet in RESEARCH_LOG — log it with the fix
   (staleness is a bug)**. Facts constraining the root cause: it reproduces on 1 chip where the PR #2324
   cross-shard all-gather is a no-op (mesh product 1), so it is NOT the gather itself; divergence starts at
   layer 0 (the first MLA attention); this regime had ZERO coverage before this session (review stage1-a
   finding 5 — multi-page/bucketed prefill untested), so it cannot be attributed to (or excluded from) the
   cd8eeb6c port without a pre-port bisect. Prime suspects: the mla.v2 multi-page paged-KV walk under the
   newly-exercised v4 block sizes, page/bucket boundary handling, or the harness's hand-built metadata at
   T past one page. Harness note: an odd token count hit the fused-MoE GMM assert
   (`num_tokens*topk % 16`, 131×8) — pick T/split multiples of 2; the pod path pads buckets anyway.
2. **Pod bring-up — NOT STARTED, gated on (1).** Nothing has run on the 8 hosts: no multi-host load, no
   TP×EP topology validation (the W_UV_scale fix, EP-head gather, cross-shard all-gather and
   TPU_MIN_TOKEN_BUCKET are all only meaningful at mesh product > 1 and are so far validated only by
   1-chip no-regression), no engine boot through the real runner/KV-spec path on real weights, no 3/3 runs.
3. **RESEARCH_LOG is partially behind** — the 2026-07-07 long-context-harness entry is logged (`e27c3c5`),
   but the two-step green, the cd8eeb6c port, the launcher/bench-wiring commits and above all the **T>128
   finding** are not yet logged.
4. **Bench numbers:** none yet. `results.db` has stub runs only; no real benchmark has been scored.

## Target (confirmed — unchanged)

- **Model:** `zai-org/GLM-5.2-FP8` (FP8-native, ~744 GB; bf16 ~1.5 TB does NOT fit). `GlmMoeDsaForCausalLM`,
  753B/40B, 78L, 256+1 experts top-8 (sigmoid noaux_tc ×2.5), MLA (lora 2048/512, nope 192, rope 64, v 256,
  64 heads) + DSA (index_topk=2048, 32×128 indexer heads, IndexShare freq 4), 1M context. Spec in `docs/00`.
- **Hardware:** the 32-chip v4 pod `db-v4-64-od` ONLY (us-central2-b; no new TPUs). All storage us-central2.

## The exact next task (in order)

1. **Root-cause + fix the T>128 divergence on the sub-cube** (no pod time until green). Bisect layer-0
   attention at T∈{128,136}: dump per-site (q/k/rope/kernel-out) diffs, test the pre-cd8eeb6c tree at T=136
   to attribute the port, check page-boundary/bucket handling in the mla.v2 call, and validate the WIP
   TuningKey/page-512 changes. Then re-run the FULL parity matrix (bf16 / fp8 / real-dims / two-step at
   T ≤ 128 AND well beyond) — commit fork WIP + harness together, update RESEARCH_LOG (incl. the missing
   entry for the port/two-step), push both repos.
2. **Pod bring-up** (gated on 1): `bash scripts/sync_workers.sh` → `bash scripts/launch_glm_32chip.sh`
   (use `--dry-run` first — the script is pod-untested) → engine boot with the Stage-1 serving config →
   a cold-cache compile observation pass (the `DSV4_OBSERVE_COMPILES` detector exists on the branch —
   drive serving-region backend compiles to 0 BEFORE trusting anything, docs/recon/doc15) → prefill MC
   gate → **3/3 runs** (one pass ≠ done).
3. **First benchmark with provenance:** `bench/run_bench.py` (make_generate is wired) → GPQA-Diamond or a
   generation-scored MMLU-family MC into `results.db`. Stage-1 threshold: within ~1–2 pts of the GPU/SGLang
   reference at ≤8K context (PLAN.md).
4. **Stage 2 (the DSA kernel — the critical path):** execute `docs/01` phase 2a — indexer KVCacheSpec
   (replace the Stage-1 skip), XLA blocked scoring + exact top-k, settle the RoPE-interleave conflict
   (E1→E2→E3; run output-level tests at ctx > 2048 or they cannot discriminate), IndexShare via the wrapper
   context. Gates S/D0/D + passkey. Upstream PR packaging: `docs/02-pr-series-draft.md` (drafted; owner submits).

## Landmines (carried over from DSV4 + new ones — expect these)

- **Flaky FP8-dequant-during-load crash** (~60% on DSV4, environmental) → retry/relaunch loop; 2+
  consecutive crashes = full cluster relaunch.
- **Multi-host serving-time recompile race** under any mesh with cross-host collectives
  (`~/moe-tpu/docs/15`, summarized in `docs/recon/doc15-multihost-race.md`): dummy-precompile-vs-real
  XLA-layout misses live-compile at serving, staggered across hosts, and worker-3 loses the launch-group
  race. `VLLM_XLA_CHECK_RECOMPILATION` is BLIND to it. Use the `ExecutableCompileObserver`
  (`DSV4_OBSERVE_COMPILES=1|trace`, on the branch) + the drive-real-layout fix pattern; note GLM Stage 1
  is pure TP×EP (no DP-attention) — a different mesh than DSV4's hybrid, so re-enumerate, don't assume.
- **`GLM_*`/`TPU_*` env vars must be baked into the raylet env** (the launcher does this; vLLM's Ray
  executor carries only `VLLM_*` + `TpuPlatform.additional_env_vars` — keep that list in sync).
- **Relaunch after any pod crash** before re-running (leaked EngineCore / ~1800 s placement-group timeout).
- `OMP_NUM_THREADS=1` everywhere (copy_ OpenMP-after-fork segfault).
- **Token-count divisibility:** fused-MoE GMM asserts `num_tokens*topk % 16 == 0`; the MLA-without-DP mesh
  token-shards 32-way → buckets < 32 fail shard_map divisibility (hence `TPU_MIN_TOKEN_BUCKET=512`).
- The parity harness is **single-chip-only by design**; don't "fix" it to multi-chip — pod validation goes
  through the real runner.

## Owner-gated (draft, don't do)

Submitting PRs to `vllm-project/tpu-inference` (see `docs/02-pr-series-draft.md` — the owner submits,
human-defended, AI-assist disclosed; coordinate with PR #2324 / yiqiliu2 first); provisioning any new
machine/VM/TPU (never — only same-region bucket + disk-attach to the 8 existing hosts, §COST in CLAUDE.md);
force-push; external comms.
