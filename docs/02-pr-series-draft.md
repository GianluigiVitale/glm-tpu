# docs/02 — Decomposing the GLM-5.2 fork work into upstreamable PRs (draft blueprint)

**Status:** decomposition + draft descriptions (2026-07-07; Stage-1 state — sub-cube parity green at
T ≤ 128, the T>128 divergence OPEN, pod bring-up pending). Mirrors the proven DSV4 pattern in
`~/moe-tpu/docs/13-pr-f-upstream-decomposition.md` (read in full; summarized in
`docs/recon/doc13-pr-decomp.md`). The deliverable is **merge-ready, individually-gated branches** —
one concern per PR, a CPU test where the math is pure, a functionally-RUN gate otherwise, and a draft
description (Motivation / Changes / Tests-run-with-numbers / Risk / AI-assisted disclosure). **The OWNER
submits to `vllm-project/tpu-inference`** — the project's `AGENTS.md` forbids pure code-agent PRs: a human
must understand and defend every line. No PR is pushed/opened by an agent, ever.

Source commits (fork `GianluigiVitale/tpu-inference`, branch `glm-5.2-v4`, cut from `dsv4-flash-v4`
@ `17d635a1`):

| commit | subject | files |
|---|---|---|
| `9bb2c23e` | register GlmMoeDsaForCausalLM + Stage-1 dense-MLA enablement | envs.py, mla_attention.py, model_loader.py, vllm_model_wrapper.py |
| `8ad980e7` | FP8-on-v4: per-tile GMM dequant + checkpoint-exact block-scale linears | gmm_v2.py, linear.py, quantization/fp8.py |
| `3f745ddc` | Stage-1 hardening: review fixes + real-dims VMEM tiling | envs.py, gmm_v2.py, mla_attention.py, vllm_model_wrapper.py, tpu_platform.py, kv_cache_manager.py |
| `cd8eeb6c` | PR #2324 TP-topology MLA port | attention_interface.py, flash_attn_mla.py, tpu_platform.py |
| *(WIP, uncommitted)* | all-gather-before-TuningKey, page-512 v4 gate, TPU_MIN_TOKEN_BUCKET | attention_interface.py, flash_attn_mla.py, tpu_platform.py, tpu_runner.py |

## The packaging realities (different from DSV4's — read before cutting branches)

1. **The GLM delta is small (4 commits) but `3f745ddc` is a cross-cutting hardening commit** whose hunks
   belong to four different concerns. Unlike DSV4's 49-commit branch-order stacking, the GLM PRs are built
   by **hunk-level regrouping** (cherry-pick + split, or re-commit per concern). That makes the docs/13
   lesson binding: *clean cherry-pick + py_compile ≠ works* — every PR branch must be **functionally run**
   (the parity harness at minimum) before it is called ready. Both DSV4 decomposition bugs
   (`7f405408` missing definer; `1882077f` cross-PR dep) were caught only by running.
2. **Everything currently sits on top of the DSV4 stack**, not on `origin/main`. Several touched files are
   heavily DSV4-modified on the branch (`kv_cache_manager.py`, `vllm_model_wrapper.py`, `tpu_runner.py`,
   `gmm_v2.py`, `layers/common/quantization/fp8.py`), so each PR must be **re-cut onto current upstream
   `main` and its context re-verified by actually creating the branch** (the docs/13 §"Independent…
   verified by actually creating branches" rule). Files expected clean vs upstream: `mla_attention.py`,
   `attention_interface.py`, `flash_attn_mla.py`, `envs.py`, `model_loader.py` (additive frozensets),
   `linear.py`. Files needing context care: the four DSV4-heavy ones above (e.g. GLM's kv_cache_manager
   skip sits one branch below the DSV4-only `is_cache_for_ds_v4` block; upstream main lacks that block).
3. **PR #2324 (author: yiqiliu2) coordination is REQUIRED, not optional** — see §Coordination at the
   bottom. PR-G5 is literally a port of #2324 hunks; PR-G1/G3 overlap it partially.
4. **Numbering:** PR-G1…PR-G6 label concerns, dependency-ordered. G2/G3/G4 are the independent trio
   (submit first / in parallel); G1 follows; G5 is stacked last and is **not ready** until the open T>128
   divergence is fixed and the pod validates the topology; G6 is the future DSA-kernel series.

---

## PR-G2 — MLA wrapper latent-bug fixes: kv_cache_dtype=auto NaN + W_UV_scale sharding axis (INDEPENDENT — submit first)

- **Concern:** two latent bugs in `VllmMultiHeadLatentAttentionWrapper.process_weights_after_loading`,
  reachable by **any** model on the vLLM MLA path (DSV3.2/Kimi/GLM family), no GLM dependency:
  (a) with `kv_cache_dtype=auto`, `quantize_tensor(None, W_UK_T/W_UV)` resolves dtype None → float64
  (`finfo` max 1.8e308) → scales collapse to 0.0, weights clip to inf → **NaN forward** (adversarially
  reproduced on CPU — `docs/reviews/stage1-a.md` item (a), `stage1-b.md` finding 8, `stage1-correctness.md`);
  (b) `W_UV_scale` is `[1, H, v]` (head axis 1) but was `device_put` with `P(ATTN_HEAD,)` on axis 0 →
  **IndivisibleError** whenever head-TP > 1 (reproduced on a 4-device CPU mesh, `stage1-b.md` finding 4).
- **Commits/hunks:** `9bb2c23e` (mla_attention.py identity-scales + resharding) + `3f745ddc`
  (ndim-agnostic scale shape — repairs the pre-existing `tests/layers/vllm/test_mla_attention.py`, now
  6/6 — and `t2j_dtype` act-dtype mapping, fp16-safe).
- **Files:** `tpu_inference/layers/vllm/custom_ops/mla_attention.py` (+ the test file).
- **Gate/risk:** bug fixes on existing paths. The auto-dtype branch was *always*-NaN, so no working config
  regresses; the quantized-KV branch makes the byte-identical `quantize_tensor(self.kv_cache_quantized_dtype,…)`
  call. Head-TP=1 behavior equivalent (verified).
- **Tests carried:** `tests/layers/vllm/test_mla_attention.py` 6/6 (CPU); the 1-chip engine parity
  (bf16 + fp8, machine-gated) as the run gate. **Add before submission:** a 4-CPU-device sharding unit test
  (`XLA_FLAGS=--xla_force_host_platform_device_count=4`; old spec → IndivisibleError, new spec → OK — the
  exact repro from the review) and an auto-dtype no-NaN assert.
- **Draft description skeleton:** Motivation = the NaN/IndivisibleError repros above (upstream-reachable);
  Changes = identity scales in the activation dtype when no quantized KV dtype is set + the corrected scale
  spec; Tests = 6/6 CPU + parity numbers; Risk = none on previously-working configs.
- **AI-assisted disclosure** (verbatim, required in every PR body): *"Portions of this change were
  developed with AI assistance (Claude); every line has been reviewed and is defended by the human
  submitter."*

## PR-G3 — gmm_v2: FP8-RHS in-VMEM dequant on TPU v4 + VMEM-model/tiling fixes (near-independent)

- **Concern:** v4's MXU has no FP8 path (Mosaic E2001 rejects any FP8 RHS matmul), so FP8 block-quant MoE
  cannot run at all on v4; and the tiling search mis-modeled the dequantize-in-VMEM path.
- **Commits/hunks:** `8ad980e7` (route FP8 RHS through the existing dequantize-in-VMEM path, gated
  `tpu_generation()==4` + FP8 dtype — FP8 stays resident in HBM, only the current tile dequantizes in VMEM)
  + `3f745ddc` (two REAL tiling bugs found only by the real-dims run: `_gmm_vmem_estimate` now models the
  f32 dequant copy + product intermediate — the search had picked tile_k=6144 → 25M > 16M VMEM; and
  `calculate_tiling`'s tile_k-shrink loop ran only when tile_n stepped BELOW its floor — when tile_n
  bottomed exactly AT the floor with the estimate still over budget, tile_k stayed at size_k).
- **Files:** `tpu_inference/kernels/megablox/gmm_v2.py`.
- **Gate/risk:** the routing is v4-gated → v5e/v6e byte-identical; the tiling fixes are generation-agnostic
  but no-ops for any config that already fit.
- **Standalone value:** every FP8 block-quant MoE checkpoint on v4 (GLM-5.x, DSV3.2-family FP8, Kimi FP8).
- **Independence caveat:** same file as DSV4 PR-2's `9eaf1270` (v4 VMEM budget) — **verify by creating the
  branch on upstream main and RUNNING a GMM parity/microbench**, not just compiling.
- **Coordination:** PR #2324 attacks the same wall differently (`MOE_DEQUANT_FP8_BEFORE_GMM` = dequant in
  HBM outside pallas + a `scale_n_block_size` 2D-scale mode). Ours keeps FP8 resident (HBM-cheaper at 753B).
  Name the overlap in the PR body and let the maintainers pick / reconcile.
- **Tests carried:** fp8 engine parity (mini block-64 + real-dims block-128, top-1 1.000/0.906 — see
  RESEARCH_LOG 2026-07-07); **add** a CPU/small-TPU unit test on the VMEM estimator (assert the chosen
  tiling fits the modeled budget at the GLM real dims). **AI-assisted disclosure** as in PR-G2.

## PR-G4 — Blockwise-FP8 linear: checkpoint-exact ragged block scales (near-independent)

- **Concern:** with `DISABLE_WEIGHT_REQUANTIZATION=1` (serve the checkpoint's own FP8 blocks), fused
  linears whose parts are not block-aligned load wrong scales: per-part scale-column counts must be
  `ceil(part/block)` — fused parts are block-quantized separately, and a non-aligned part (GLM's
  `kv_a_proj_with_mqa`, 576 @ block 128) ends in a partial block that owns a full scale column. And
  `xla_quantized_matmul`'s 2D-block-scale branch used a uniform-blocks reshape that cannot represent a
  ragged tail — replaced with repeat+clip scale expansion + a hard assert on interior-ragged fused concats.
- **Commits/hunks:** `8ad980e7` (linear.py + layers/common/quantization/fp8.py hunks).
- **Files:** `tpu_inference/layers/common/linear.py`, `tpu_inference/layers/common/quantization/fp8.py`.
- **Gate/risk:** only the `DISABLE_WEIGHT_REQUANTIZATION` / blockwise path; block-aligned checkpoints
  produce identical scale layouts (ceil == div when aligned).
- **Independence caveat:** `layers/common/quantization/fp8.py` is also touched by DSV4 PR-5a — re-cut onto
  upstream main and verify context (the DSV4 dequant methods may not exist there).
- **Tests carried:** the fp8 parity twin (HF twin loads the SAME effective weights via quant-dequant
  roundtrip, so the diff isolates the engine's fp8 handling — top-1 1.000); **add** a CPU unit test:
  ragged 2D-scale expansion vs an explicit per-block reference on a 576@128 shape (bit-exact).
  **AI-assisted disclosure** as in PR-G2.

## PR-G1 — Register GlmMoeDsaForCausalLM + Stage-1 dense-MLA enablement (stacks conceptually on G2; submit after the trio)

- **Concern:** make GLM-5.2 loadable and correct-by-default on TPU with the DSA indexer gated OFF (dense
  MLA) until a TPU sparse path exists: registry + PP-disable, the IndexShare-aware construction patch
  (upstream vLLM builds an `Indexer` on every layer; GLM-5.2 ships indexer weights only on `full`-schedule
  layers → loader fails without it), the `TPU_DISABLE_DSA_INDEXER` gate (default ON for TPU — vLLM's
  `SparseAttnIndexer` has no TPU forward and raises), the engine-boot KV-spec fix (vLLM's self-registered
  `DeepseekV32IndexerCache` defines neither `kv_sharing_target_layer_name` nor `attn_type` →
  `get_kv_cache_spec` AttributeError on every GLM/DSV3.2 config; the dense stage gives it no KV slot), and
  Ray propagation of the new env vars.
- **Commits/hunks:** `9bb2c23e` (model_loader.py, envs.py, vllm_model_wrapper.py) + `3f745ddc` (gate
  default-ON, kv_cache_manager.py skip, tpu_platform.py env list, `_layer_is_shared` list-form fix).
- **Files:** `tpu_inference/models/common/model_loader.py`, `tpu_inference/envs.py`,
  `tpu_inference/models/vllm/vllm_model_wrapper.py`, `tpu_inference/runner/kv_cache_manager.py`,
  `tpu_inference/platforms/tpu_platform.py`.
- **Gate/risk:** additive frozenset entries; the construction patch is arch-sniffed
  (`_maybe_patch_for_glm_moe_dsa`) and yields early for non-GLM; the KV-spec skip keys on the vLLM
  `DeepseekV32IndexerCache` class; non-GLM configs byte-identical. Known review-flagged residue to state
  honestly in the PR: the env attribute (`DISABLE_DSA_INDEXER`) reads `TPU_DISABLE_DSA_INDEXER`
  (PR #2324 lineage; the only key≠env-name entry in envs.py), and the monkeypatch duplicates upstream's
  schedule derivation — see the root-fix note below.
- **Root-fix note (companion upstream-vLLM PR, cleaner long-term):** the shared-layer failure is an
  upstream **vLLM** bug — `deepseek_v2.py` already computes `_skip_topk` per layer but still constructs an
  `Indexer` on every `is_v32` layer; the GPU path has the identical IndexShare load failure. The clean fix
  is a one-liner there (`indexer = None` when skipped). Offer it as a separate tiny vLLM PR; the
  tpu-inference wrap stands until it lands (review `stage1-b.md` finding 5).
- **Tests carried:** the machine-gated 1-chip engine parity suite (bf16/fp8/real-dims — the numbers in
  RESEARCH_LOG 2026-07-07); **add per the review list:** `tests/models/common/test_model_loader.py`
  (arch resolves to impl "vllm", PP rejected), `tests/test_envs.py` (env-name parse), a CPU
  `_layer_is_shared` test against the real 78-entry `indexer_types` (hoist the closure to module level
  first), and a KVCacheManager spec test with a GLM config (the engine-boot crash class).
  **AI-assisted disclosure** as in PR-G2.
- **Dependency:** functionally assumes G2 (without the auto-NaN fix, the registered model NaNs under the
  default kv-cache dtype). Declare G2 a prerequisite for the whole-model gate (the docs/13 mHC-stub
  precedent: gates that need a prerequisite PR must say so).

## PR-G5 — TP-topology MLA fixes for v4 multi-host (cross-shard all-gather, v4 blocks/page, EP-head layout, MLA-without-DP, TPU_MIN_TOKEN_BUCKET) — STACKED LAST, #2324-COORDINATED, NOT READY YET

- **Concern:** run MLA under the pure TP×EP topology (no DP-attention — mandatory at 753B where attention
  weights cannot replicate): all-gather q/q_rope/k/k_rope over MLP_TENSOR inside the MLA shard_map +
  dynamic-slice the output back (the kernel derives causal positions from replicated descriptors + LOCAL
  iota — any shard beyond the first applied the wrong mask), v4 block sizes (1,1,1)/(1,8,8) + fp32 scores,
  page_size 512 when kv_lora_rank > 256 (v4 VMEM, gated to v4), the EP-head layout constraint before
  o_proj, drop the hard MLA→DP-attention requirement, and `TPU_MIN_TOKEN_BUCKET` (buckets below the 32-way
  token-shard product fail shard_map divisibility; default 0 = bucket table byte-identical).
- **Commits:** `cd8eeb6c` **+ the currently-uncommitted WIP** (all-gather moved before the TuningKey so the
  key sees post-gather shapes; page-512 gated to v4; tpu_runner min-bucket env + Ray propagation).
- **Files:** `tpu_inference/layers/common/attention_interface.py`,
  `tpu_inference/layers/vllm/backends/flash_attn_mla.py`, `tpu_inference/platforms/tpu_platform.py`,
  `tpu_inference/runner/tpu_runner.py`.
- **⚠️ NOT ready to cut:** (a) the open **T>128 parity divergence** is under investigation in exactly this
  code region (even though it reproduces at mesh product 1, where the gather is a no-op); (b) every
  multi-host-only behavior here is **pod-unvalidated** (1-chip no-regression only). Bar before drafting:
  T>128 green on the sub-cube + pod 3/3 with zero serving-region backend compiles.
- **Gate/risk:** all pieces are v4-gated, mesh-product-1-no-op, or default-off; v5/v6 tuned paths unchanged.
- **Coordination (the load-bearing part):** this PR is a port of **PR #2324** (yiqiliu2 — GLM-5.1-FP8 on
  v4-64, DSA disabled), axis-adapted to this branch's head-major q layout. #2324 is unmerged/stalled
  (reviewer declined; missing `ready` CI label). **Do not double-submit someone else's work.** The owner
  should first comment on #2324: either (a) help land #2324 itself and rebase this PR to only the deltas
  (TuningKey ordering, v4 gating, the head-major axis adaptation), or (b) with yiqiliu2's and the
  maintainers' consent, submit the port with explicit attribution (`Co-authored-by:` yiqiliu2 + a body
  paragraph crediting #2324 and listing the deltas). Re-verify #2324's state at submission time.
- **Tests carried:** 1-chip parity regression PASS post-port (now exercising v4 blocks + fp32 scores);
  pending: the pod TP×EP bring-up gates. **AI-assisted disclosure** as in PR-G2.

## PR-G6 (future) — the DSA sparse-attention series (Stage 2; the headline contribution)

Not yet built. Will follow `docs/01-dsa-kernel-design.md` phasing, each phase its own gated PR in the
docs/13 stacked style: (i) indexer k-cache KVCacheSpec registration (replaces the Stage-1 no-slot skip),
(ii) XLA indexer scoring + blocked exact top-k + IndexShare wrapper-context carriage (gate S/D0/D),
(iii) Pallas indexer-scoring kernel (gate S vs (ii) + microbench), (iv) sparse-MLA gathered-segment decode
kernel (gate K/D/P, TP=4==TP=1, 3/3 pod). This is the piece that positions the fork against #2324 upstream:
#2324 runs DSA *disabled*; no public JAX/Pallas DSA inference kernel exists on TPU. Each PR carries its
parity harness numbers + a CPU test where pure-math, and the same disclosure line.

---

## Submission order (recommended)

1. **PR-G2** (MLA wrapper bug fixes — cleanest, upstream-valuable, CPU-tested).
2. **PR-G3 + PR-G4** (FP8-on-v4 enablement pair) — after re-cutting onto upstream main and functionally
   running each branch (the docs/13 rule); can go in parallel with G2.
3. **PR-G1** (registration + enablement) — declares G2 as a whole-model-gate prerequisite; optionally
   paired with the tiny companion vLLM PR (indexer=None on shared layers).
4. **PR-G5** — only after the T>128 fix + pod 3/3, and only after the #2324 conversation.
5. **PR-G6 series** — as Stage 2 lands, phase by phase.

## CI gate (model after the Kimi `.buildkite/models/...` gate; v4 queue = part of the contribution)

- **Kernel gate:** gmm_v2 FP8-dequant + tiling unit tests (guards G3) and `mla_v2` tests on v4.
- **Correctness gate (cheap):** a trimmed mini-config GLM engine parity (the machine-gated
  `glm-tpu/parity/glm_engine_parity.py` pattern — exit-code, bf16 floor bars) — guards G1/G2/G5.
- **Accuracy gate (full model, nightly):** the Stage-1 benchmark once it lands (GPQA-Diamond / MMLU-family
  MC vs the provenance DB). Never trust a load-only UnitTest as correctness.

## Rules restated (non-negotiable)

- **Every PR body carries:** *"Portions of this change were developed with AI assistance (Claude); every
  line has been reviewed and is defended by the human submitter."*
- **The owner submits and defends** every PR (vLLM/tpu-inference `AGENTS.md`: no pure code-agent PRs).
  Agents draft branches + descriptions only; no push to upstream, no PR opened, no comments posted.
- Each branch: re-cut onto current `origin/main`, **functionally run** (parity, not just py_compile),
  gates + numbers quoted in the description, model-unchanged byte-identical when its gate is off.
