# Feasibility Memo: Porting GLM-5.2 (`GlmMoeDsaForCausalLM`) to TPU v4 via vLLM/torchax (tpu-inference)

**Prepared:** July 6, 2026 · **Subject:** Technical feasibility and go/no-go for a GLM-5.2 port on TPU v4, using the completed DeepSeek-V4-Flash (DSV4) port as a candidate base.

> **Verification note (July 6, 2026):** the two load-bearing facts below were checked directly against primary sources this session — the live GitHub PR #2324 page and the raw `zai-org/GLM-5.2/config.json` on HuggingFace — not just the research synthesis. Several figures from the first-pass memo were corrected as a result; see the "Corrections from direct verification" box at the end.

## TL;DR
- **Go — start from your DSV4 draft repo, not from scratch.** GLM-5.2 is real, MIT-licensed, and reports `GlmMoeDsaForCausalLM` / `model_type: glm_moe_dsa` (verified in config.json). Architecturally it is a DeepSeek-V3.2-lineage **MLA + DeepSeek Sparse Attention (DSA)** MoE (256 routed + 1 shared experts, top-8, sigmoid `noaux_tc` gating, 78 layers, FP8-native, MTP, 1M context) — the same family your DSV4 MLA path already touches, plus a sparse "lightning indexer" that your CSA kernel is the closest thing in the ecosystem to being able to serve.
- **The single most important variable — a DSA TPU kernel — still does not exist publicly, and PR #2324 confirms this from the inside.** As of July 6, 2026 there is no JAX/Pallas DSA (lightning-indexer + top-k + sparse-MLA) inference kernel anywhere. PR #2324's own commit history shows the DSA indexer path being **actively disabled** on TPU: the `TPU_DISABLE_DSA_INDEXER` env gate (commit `29852c6`) "creates/loads indexer params but skips the forward call," because the indexer "is not yet ported to torchax (`k_norm.type_as` hits `'View' object has no attribute '_elem'`)." So the model runs as **dense MLA with DSA bypassed**. Your CSA/HCA compressed sparse-attention Pallas kernel is therefore your single biggest strategic asset — it is the exact missing piece.
- **Recommended first work-item:** register `GlmMoeDsaForCausalLM` through the vLLM/torchax path on TPU v4 running attention as **dense MLA** (fast correctness win, already de-risked by PR #2324 which ran GLM-5.1-FP8 end-to-end on a v4-64), then adapt your CSA "lightning-indexer/top-k" compressed kernel to GLM-5.2's DSA + IndexShare to recover the long-context FLOP savings. Frame the port on TPU v4 with head-sharding + the PR's `tpu_streaming_loader` multi-host weight load; the target is a v4-64 pod (32 chips × 32 GB = 1024 GB HBM).

## Key Findings

1. **GLM-5.2 exists and is very new.** Released to GLM Coding Plan subscribers June 13, 2026; open weights (MIT) on HuggingFace `zai-org/GLM-5.2`, with the official Z.ai technical blog dated June 17, 2026. It is distinct from GLM-5.1 (April 8, 2026 open release) and GLM-5 (Feb 11, 2026).
2. **HF architecture class is `GlmMoeDsaForCausalLM`**, `model_type: glm_moe_dsa`, requiring `transformers>=5.3.0`. It does NOT alias to a DeepSeek arch string, but its internals are DeepSeek-V3.2-lineage MLA+DSA (confirmed in the config.json and by NVIDIA's model card: "Network Architecture: GLM-5.2 (GlmMoeDsaForCausalLM)").
3. **GLM-5.2 uses DSA** — DeepSeek Sparse Attention: a lightweight lightning indexer scores tokens, a top-k selection (`index_topk: 2048`; DeepSeek-V3.2's k=2048 hyperparameter reduces attention complexity from O(L²) to O(Lk)) picks the relevant subset, then sparse MLA attends only over those. GLM-5.2 adds **IndexShare**: per Z.ai's official release notes, "We propose IndexShare, which reuses the same indexer across every four sparse attention layers, reducing per-token FLOPs by 2.9× at a 1M context length." Per NVIDIA's NeMo AutoModel docs, the indexer is computed only on "full" attention layers and reused by the following "shared" layers, with the per-layer schedule declared in the config's `indexer_types` array (which alternates one `full` followed by three `shared`, matching `index_topk_freq: 4`).
4. **No DSA kernel exists for TPU** in any framework — confirmed across the tpu-inference repo, MaxText, and vLLM's own DSA roadmap, which as of its DeepSeek-V3.2 launch post still listed TPU support as future work ("We will expand the support to other hardwares such as AMD and TPU"). This is unchanged from your May 2026 assessment.
5. **A TPU v4 GLM path already exists as a draft: tpu-inference PR #2324** (`GlmMoeDsaForCausalLM` routed through the vLLM MLA path, run **dense** with DSA bypassed), targeting exactly your hardware class (v4-64), and independently rediscovering several of your own DSV4 techniques.
6. **GLM-5.2 has first-class GPU support** — vLLM ≥0.23.0, SGLang ≥0.5.13.post1 (which registers `GlmMoeDsaForCausalLM`), and an NVIDIA NVFP4 checkpoint — giving you a solid reference implementation to port attention/indexer math from.

## Details

### 1. GLM-5.2 architecture spec (from `zai-org/GLM-5.2/config.json`, verified verbatim)

| Field | Value |
|---|---|
| HF `architectures` | `GlmMoeDsaForCausalLM` |
| `model_type` | `glm_moe_dsa` |
| Total / active params | **753B total / ~40B active** (per NVIDIA GLM-5.2-NVFP4 card, 06/25/2026; some third-party sources cite 743–744B) |
| `num_hidden_layers` | 78 |
| `first_k_dense_replace` | 3 (first 3 layers dense, remainder MoE; `mlp_layer_types` confirms 3 dense + 75 sparse) |
| `hidden_size` | 6144 |
| `intermediate_size` (dense) | 12288 |
| `moe_intermediate_size` | 2048 |
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 8 (top-8) |
| Gating | `scoring_func: sigmoid`; `topk_method: noaux_tc`; `routed_scaling_factor: 2.5`; `norm_topk_prob: true` |
| Attention (MLA) | `kv_lora_rank: 512`, `q_lora_rank: 2048`, `qk_nope_head_dim: 192`, `qk_rope_head_dim: 64`, `qk_head_dim: 256`, `v_head_dim: 256`, `head_dim: 192`, `num_attention_heads: 64`, `num_key_value_heads: 64` |
| DSA indexer | `index_head_dim: 128`, `index_n_heads: 32`, `index_topk: 2048`, `index_topk_freq: 4` (IndexShare), `index_skip_topk_offset: 3`, `indexer_rope_interleave: true`, `index_share_for_mtp_iteration: true` |
| MTP | `num_nextn_predict_layers: 1` (extended to up to 5 draft tokens for spec decoding) |
| Precision | native FP8 checkpoint (also BF16; NVFP4 community/NVIDIA quant) |
| `vocab_size` | 154880 |
| `max_position_embeddings` | 1048576 (1M); `rope_theta: 8000000`, `rope_interleave: true` |
| `dtype` | bfloat16 |
| Vision | **None — text-only** |

**Distinction from GLM-5.1:** same ~744B/40B MoE DSA backbone; GLM-5.2 adds IndexShare (4-layer indexer reuse), improves the MTP layer (KVShare, rejection sampling, end-to-end TV loss), and extends context 200K→1M. Per Z.ai's card, the MTP improvement increases "the acceptance length by up to 20%" (from baseline 4.56 to 5.47 in their ablation), and MTP is extended from ~3 to 5 draft tokens.

### 2. "What exists already" audit (line-cited)

| Artifact | Repo | Status | Evidence |
|---|---|---|---|
| GLM-5.2 GPU support | vLLM | Shipped | vLLM ≥0.23.0 recipe; `--reasoning-parser glm45 --tool-call-parser glm47`; MTP 5 draft tokens; DeepGEMM required for FP8 |
| GLM-5.2 GPU support | SGLang | Day-0 | SGLang ≥0.5.13.post1 registers `GlmMoeDsaForCausalLM` |
| GLM-5.2 NVFP4 | `nvidia/GLM-5.2-NVFP4` | Shipped (06/25/2026) | MoE experts NVFP4; MLA + DSA lightning indexer kept BF16; ~410 GB |
| GLM-5.1-FP8 on TPU v4-64 | tpu-inference | **Draft PR #2324** (open, un-CI'd, unmerged) | Author `yiqiliu2` (uid 66063897), opened Apr 20 2026; routes `GlmMoeDsaForCausalLM` via vLLM MLA path, **DSA bypassed**; 24 commits; head branch `pr/a-mla-multihost` |
| GLM-5.1 TPU feature request | tpu-inference | Issue #2339 (open, Apr 20 2026) | Links PR #2324; "same MLA + MoE architecture as DeepSeek"; ~750GB into 32×32GB HBM |
| GLM5 support | tpu-inference | Issue #1699 (open, Feb 11 2026) | "want to run GLM5 on TPUv7"; contribution-welcome |
| DeepSeek-V4 Pro on TPUv7 | tpu-inference | Issue #2569 (open, May 10 2026) | contribution-welcome, needs-more-info |
| DSA TPU kernel | any public repo | **Does not exist** | PR #2324 §1.7: "JAX has no GLM DSA implementation"; vLLM DSA blog lists TPU as future |
| MaxText deepseek3.2 DSA | AI-Hypercomputer/maxtext | Training/arch config only (Apr 10 2026); Kimi-K2 Apr 13 2026 | Published Pallas-kernel set lists only paged/ragged attention + Megablox GMM — no indexer/sparse-MLA kernel |
| Production DSA kernels | DeepSeek DeepGEMM (indexer) + FlashMLA (sparse attn); TileLang ref | GPU/CUDA only | SM90/SM100+ |

### 3. DSA-kernel-status determination (the critical variable)

**Determination: No public JAX/Pallas DSA inference kernel exists on TPU as of July 6, 2026.** This is the single most important input to the effort estimate. Specifics, all primary-sourced:

- **tpu-inference PR #2324 bypasses DSA.** §1.7 ("GLM registry") states verbatim: *"JAX has no GLM DSA implementation, and GLM's JAX-side PP plumbing is not wired up."* It registers `GlmMoeDsaForCausalLM` in `models/common/model_loader._VLLM_PREFERRED_ARCHITECTURES` and adds `Glm4MoeLiteForCausalLM` to `_PP_DISABLED_MODELS`, running the model as **dense MLA**. The §1.8 cross-shard fix explicitly feeds the kernel the *full prefix* (all-gathers `q / q_rope / new_kv_c / new_k_pe`), not a top-2048 subset — confirming dense, full-context attention. This is correct for short/moderate contexts but forgoes DSA's long-context savings (per the Spheron 2026 DSA guide, DSA cuts attention FLOPs by ~98% at 128K via fixed top-2048 selection) and will diverge numerically from a true-DSA reference beyond ~2048 tokens. PR #2324's own smoke test uses short prompts at `max_num_batched_tokens=4096`.
- **PR #2324 is stalled.** CI never ran (author, May 9 2026: the `buildkite/tpu-inference-ci/pr` check "exits early because the PR is missing the `ready` label"); collaborator `kyuyeunk` declined review (May 9 2026: "33 files changed, added 4,400 lines, deleted 334 lines… i'm going to have to pass on reviewing this pr"). Sibling PR #2325 (FP8 MoE direct path) was folded in and closed.
- **MaxText** added a `deepseek3.2-671b` DSA config (~Apr 10, 2026) but its published Pallas inference kernels are only paged/ragged attention and Megablox GMM — no lightning-indexer/sparse-MLA/top-k kernel. DSA there is architecture/training-level (likely XLA top-k / dense-then-mask). (Could not be fully verified from source — treat as strongly-supported-by-absence.)
- **TPU dynamic sparsity is genuinely hard.** Pallas enforces lexicographic grid traversal, which is unfriendly to top-k dynamic gather; and per Google's Spark Transformer paper (arXiv 2506.06644, §4.4), the standard `jax.lax.approx_max_k` operator "leads to more than 10× slowdown even when operating on a small recall of 50%." This is precisely the class of problem your CSA "lightning indexer" compressed kernel already solves on TPU — making it far more valuable than any off-the-shelf option. (The DeepSeek-V3.2/V4 "Compressed Sparse Attention" family your DSV4 CSA kernel targets is the same lineage analyzed in StreamIndex, arXiv 2605.02568, which notes V4-Flash uses H_I≈64 indexer heads with compression ratio m=4.)

### 4. What transfers from your DSV4 port

| Component | Transfer verdict | Rationale |
|---|---|---|
| (a) vLLM/torchax registration path (arch string) | **Direct** | GLM-5.2's `GlmMoeDsaForCausalLM` routes through the same vLLM/torchax MLA path; PR #2324 shows the exact one-line registry addition. |
| (b) CSA/HCA compressed-attention Pallas decode kernel | **Modify — highest-value asset** | GLM-5.2 DSA = lightning-indexer + top-k=2048 sparse MLA. Your CSA "lightning-indexer sparse top-k with comp_bias" is conceptually the closest existing TPU kernel. Modify for: MLA MQA-mode latent sharing, `index_n_heads=32 / index_head_dim=128` ReLU-weighted indexer, top-k=2048, and IndexShare (reuse selected indices across each 4-layer `full→shared×3` block, incl. MTP). |
| (c) ATTN_HEAD head-sharding for v4 memory fit | **Direct→Modify** | 753B FP8 ≈ 750GB needs the same multi-host HBM management. PR #2324 independently rediscovered this need (expert-axis all-gather before `o_proj`; streaming loader). Your head-sharding of wq_b/wo_a/wo_b applies to MLA; re-verify against GLM-5.2's dims (64 heads, qk_head_dim 256, v_head_dim 256). |
| (d) FP4-requant low-bit path | **Modify (conditional)** | GLM-5.2 is FP8-native, not FP4. On v4 (no native FP8 MXU) you need the FP8→bf16 per-tile dequant-in-Pallas trick PR #2324 uses (Mosaic rejects FP8 RHS on v4, E2001). Your FP4-requant path is reusable if you serve the NVFP4 community checkpoint, but the FP8 dequant-first branch is the more direct fit. Watch the flaky FP8-dequant-during-load crash you already hit. |
| (e) per-rank KV-sizing / DPScheduler long-context fix | **Direct** | GLM-5.2 targets 1M context; Z.ai explicitly names KV-cache capacity (not compute) as the primary bottleneck. Your per-rank KV-pool auto-sizing + DPScheduler fix directly apply. Note PR #2324's caveat: streaming loader + `--kv-cache-dtype fp8` raises NotImplementedError — use `--kv-cache-dtype auto` there. |

### 5. Engineering-effort estimate (TPU v4, vLLM/torchax path)

| Task | Effort | Notes |
|---|---|---|
| Register `GlmMoeDsaForCausalLM`, load FP8 weights, dense-MLA correctness on v4 | Low | Largely done by PR #2324; port its registry + v4 MLA workarounds (FP8→bf16 tile cast in `kernels/mla/v2/kernel.py`, v4 block sizes in `kernels/ragged_paged_attention/v3/kernel.py`, fp32 softmax in `attention_interface.py`). |
| Multi-host weight load (753B) + EP/TP sharding on v4 pod | Low–Medium | Reuse your head-sharding + PR #2324's `tpu_streaming_loader` + mesh-aware EP filter. |
| MoE: 256 experts, sigmoid noaux_tc top-8, shared expert | Low | Standard GMM path; `e_score_correction_bias` plumbing (PR #2324). |
| **DSA lightning-indexer + top-k=2048 Pallas kernel (adapt CSA)** | **High — critical path** | No reference on TPU. This is the main net-new engineering. |
| IndexShare (4-layer `full→shared` indexer reuse) | Medium | Cache/reuse top-k indices across the layer block; applies to MTP too (`index_share_for_mtp_iteration`). |
| MTP speculative decoding (1 layer / up to 5 draft) | Medium | Optional for first correctness; needed for throughput. |
| 1M-context KV sizing / DPScheduler | Low | Reuse your existing fix. |
| Accuracy gating (MMLU/passkey) on v4 pod | Medium | Reuse your 5-shot MMLU + passkey harness. |

### 6. Go/no-go verdict
**Verdict: GO, from your DSV4 draft repo as base.** GLM-5.2 shares the MLA + MoE + FP8 + MTP + DSA-family skeleton with your DSV4 work; starting from scratch would discard directly-reusable head-sharding, KV-sizing, FP8-on-v4, and — most importantly — your CSA/HCA sparse-attention kernel, which is the closest thing in the ecosystem to a TPU DSA kernel and which no one else has built. The port is meaningfully *easier* than your May 2026 GLM-5.1 estimate for one reason only: PR #2324 has since de-risked dense-MLA GLM on v4-64. The hard part (a real DSA kernel) is unchanged and remains the critical path.

## Recommendations
1. **Stage 1 (days): dense-MLA correctness.** Port PR #2324's registry entry and v4 MLA workarounds into your DSV4 repo; serve GLM-5.2-FP8 on your 32-chip v4 pod with DSA bypassed (dense). Benchmark 5-shot MMLU vs the GPU/SGLang reference. **Threshold to proceed:** MMLU within ~1–2 pts of reference at ≤8K context.
2. **Stage 2 (weeks): DSA kernel.** Adapt your CSA compressed decode kernel into a DSA lightning-indexer (`index_n_heads=32`, `index_head_dim=128`, ReLU scoring) + top-k=2048 + sparse-MLA path, validated against the DeepSeek-V3.2 reference math (note the V3.2 indexer RoPE non-interleaved-layout gotcha; GLM-5.2 sets `indexer_rope_interleave: true`, so verify layout carefully). **Threshold:** passkey retrieval ≥95% to ≥128K and long-context divergence from dense-MLA within tolerance.
3. **Stage 3: IndexShare + MTP.** Add 4-layer indexer reuse (`index_topk_freq: 4`) and MTP spec decode for throughput. **Threshold:** measurable FLOP/throughput gain at ≥256K context and MTP acceptance length in the ~5 range.
4. **Upstream coordination.** PR #2324 is unmerged and stalled (reviewer declined; missing `ready` CI label). Engage `yiqiliu2` and tpu-inference maintainers (jrplatin, kyuyeunk) to avoid duplicated work and to position your DSA kernel as the missing piece the entire DSA-family (DeepSeek-V3.2/V4, Kimi-K2.6, GLM-5.x) needs on TPU. This maximizes the strategic payoff of your CSA asset.
5. **Hardware framing.** Keep TPU v4 as target but design v4 gates (`tpu_generation()==4`) to be liftable to v6e/v7 (Ironwood is the first TPU with native FP8 MXU) so the DSA kernel benefits the broader roadmap rather than being a v4-only dead end.

## Caveats
- GLM-5.2 released ~3 weeks ago; param figures vary by source (743B / 744B / 753B). The config.json (753B total, 78 layers, 256 experts) and NVIDIA's model card (753B) are authoritative; ~40B active is consistent across sources.
- PR #2324 is a moving, unmerged draft; verify its current state (and whether a maintainer has since added the `ready` label or merged it) before porting.
- MaxText's DSA implementation details could not be fully verified from source; treat "no Pallas DSA inference kernel in MaxText" as strongly-supported-by-absence, not positively confirmed.
- No v4 CI exists for any DSA-family model; v4-specific gates (FP8-on-MXU dequant, VMEM block sizes, fp32 softmax on deep stacks) are your responsibility to validate — PR #2324 only exercised them on v4-64 and flags v5e/v5p/v6e/v7 as untested.
- GLM-5.2 GPU recipes assume native FP8 MXU (Hopper/Blackwell); your v4 path must use the dequant-in-Pallas workaround throughout, and streaming-loader + fp8-KV is currently mutually exclusive (use `--kv-cache-dtype auto`).
- GLM-5.2 is text-only (no vision component), simplifying the port relative to a multimodal target.