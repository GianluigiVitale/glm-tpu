# PLAN — GLM-5.2 on TPU v4 (phased, with thresholds)

Built on `docs/00-feasibility-memo.md` (the config-verified GO study) and the completed DeepSeek-V4-Flash port
(`~/moe-tpu`). The three poles are the same as DSV4 — **correctness → v4 enablement → fit** — but the *headline*
is the **DSA sparse-attention kernel** (which no one has on TPU). Advance only when a stage's threshold is met.

## Stage 1a — staging + registration (days; cost-critical)
- Fork GLM branch off `dsv4-flash-v4` (`glm-5.2-v4`); editable-install.
- **Stage `zai-org/GLM-5.2-FP8` (~744 GB) HF → `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/`** (us-central2, same
  region as the pod). Streaming, zero local disk (adapt `~/moe-tpu/scripts/stage_base_to_gcs.py`). Never the EU
  bucket (must be same-region — the two hard rules are same-region bucket + NO new machines/TPUs). Prefer
  streaming GCS→HBM (no local copy); if a per-host local copy proves mandatory, you may create + attach a
  ~1000 GB disk to each of the 8 hosts (see CLAUDE.md §COST).
- Register `GlmMoeDsaForCausalLM` (vLLM/torchax path; PR #2324's registry entry).
- **Done when:** the FP8 weights are one same-region copy on GCS and the model constructs on the engine.

## Stage 1b — dense-MLA correctness (days)
- Run attention as **dense MLA (DSA bypassed)** — the fast correctness win PR #2324 already de-risked on v4-64.
- Port the v4 MLA workarounds (FP8→bf16 tile dequant, v4 block sizes, fp32 softmax) — re-verify vs GLM's MLA
  dims (qk_head_dim 256, v_head_dim 256, kv_lora_rank 512, q_lora_rank 2048, 64 heads).
- Validate the forward **bit-faithfully vs the HF/GPU reference** (fp32 + bf16 controls) at a tiny config, then
  real dims; head-shard for HBM fit (reuse `~/moe-tpu/docs/12`); runai_streamer + on-device dequant on the pod.
- Build the **provenance DB** and reproduce a first benchmark (GPQA-Diamond or a generation-scored MMLU-family MC).
- **Threshold → Stage 2:** the reproduced benchmark is within ~1–2 pts of the GPU/SGLang reference at ≤8K
  context, with 3/3 clean pod runs and every item stored in `bench/results.db`.

## Stage 2 — the DSA kernel (weeks; the critical path, the contribution)
- Adapt the DSV4 CSA compressed-decode Pallas kernel into a **GLM DSA** path: lightning indexer
  (`index_n_heads=32`, `index_head_dim=128`, ReLU-weighted scoring, **interleaved-RoPE** — verify the layout,
  it differs from V3.2) → **top-k=2048** selection → sparse MLA over the selected set.
- Validate **selected-set-exact vs the reference** (0 mismatch vs `torch.topk`, tie-breaks aside) and
  **bit-faithful to the XLA decode** (fp32 ~3e-8 / bf16 bit-identical) — the DSV4 kernel-parity bar.
- Watch the TPU dynamic-sparsity hazard (Pallas lexicographic grid vs top-k gather; `approx_max_k` is >10×
  slow — memo §3). The CSA `comp_bias` masking approach is the proven way around it.
- **Threshold → Stage 3:** passkey retrieval ≥95% to ≥128K context; long-context divergence from dense-MLA
  within tolerance; the DSA path scores match the dense path where they should and diverge only past top-k.

## Stage 3 — IndexShare + MTP (throughput)
- **IndexShare:** reuse the selected top-k indices across each 4-layer `full→shared×3` block
  (`index_topk_freq=4`), including the MTP iteration (`index_share_for_mtp_iteration`).
- **MTP:** speculative decode, 1 layer / up to 5 draft tokens.
- **Threshold:** measurable FLOP/throughput gain at ≥256K context; MTP acceptance length ~5.

## Cross-cutting (every stage)
- Every change gated + additive + CPU-tested + sub-cube-validated + pod 3/3; adversarial reviewers vs the
  reference; PR-compatible code drafted into a stacked PR series (mirror `~/moe-tpu/docs/13`).
- Full benchmark provenance (every question/answer/timestamp in `bench/results.db`); honest signed Δ vs the HF
  card; nulls first-class.
- v4 gates liftable to v6e/v7 (Ironwood = native FP8).
- Cost: one same-region (us-central2) FP8 copy kept resident; prefer streaming, per-host disks only if required; clean up as you go (§COST).

## Benchmark targets (HF card — reproduce with provenance; verify at runtime)
Reasoning/agentic: HLE 40.5, HLE+Tools 54.7, AIME 2026 99.2, HMMT Nov-25 94.4 / Feb-26 92.5, GPQA-Diamond 91.2,
IMOAnswerBench 91.0, CritPt 20.9. Coding/agentic (heavier harnesses): SWE-bench Pro 62.1, Terminal Bench 2.1
81.0–82.7, NL2Repo 48.9, DeepSWE 46.2, ProgramBench 63.7, FrontierSWE 74.4, MCP-Atlas 76.8, Tool-Decathlon 48.2,
SWE-Marathon 13.0, PostTrainBench 34.3. Start with the tractable short-generation card benchmarks (GPQA-Diamond, AIME) + the MMLU-Pro/GSM8K sanity gates; the
agentic coding benchmarks need full agentic harnesses (later).
