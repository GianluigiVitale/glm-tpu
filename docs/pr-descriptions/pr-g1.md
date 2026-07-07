# PR-G1 — [Model] Register GlmMoeDsaForCausalLM (GLM-5.x) with dense-MLA Stage-1 enablement

**Branch:** `pr-g1-register-glm-moe-dsa` (worktree `~/tpu-inference-prs`)
**Commit:** `e03c237f`
**Base:** `97938b62` (fork main, 2026-06-13).
Forward-port to `vllm-project/tpu-inference` main @ `0d59fee9` (2026-07-07): **conflict-free
textually** (`git merge-tree`: 0 conflict hunks; `envs.py`/`model_loader.py`/
`vllm_model_wrapper.py`/`tpu_platform.py`/`kv_cache_manager.py` all drifted upstream but the
hunks merge cleanly). Semantic re-verification on the tip requires a newer vLLM than the local
env pins — run the CPU test suite once on the tip after forward-porting.
**Status:** ready for owner review. **Declares PR-G2 (auto-dtype NaN fix) a prerequisite for
the whole-model gate** — without it the registered model NaNs under the default kv-cache dtype.

---

## PR body draft

# Description

Make GLM-5.x (`GlmMoeDsaForCausalLM`, DeepSeek Sparse Attention family) loadable and
correct-by-default on TPU, with the DSA indexer forward gated OFF until a TPU sparse kernel
consumes its indices. Attention is dense MLA either way at this stage: vLLM's DSA indexer only
*selects* tokens; with no TPU sparse-attention consumer, running it is pure overhead — and in
fact impossible (below).

- **model_loader**: add `GlmMoeDsaForCausalLM` to `_VLLM_PREFERRED_ARCHITECTURES` (DSA has no
  flax_nnx implementation) and `_PP_DISABLED_MODELS`.
- **vllm_model_wrapper — `_maybe_patch_for_glm_moe_dsa`** (arch-sniffed, yields early for
  non-GLM): GLM-5.2 **IndexShare** ships indexer weights only on `full` indexer-schedule layers
  (`shared` layers reuse the previous full layer's top-k indices; the HF reference sets
  `indexer=None` on them). vLLM's DeepSeek-V2 code builds an `Indexer` on **every** DSA layer,
  so the loader fails on every shared layer. The patch wraps the `Indexer` constructor to
  return `None` exactly where the config's schedule says `shared` — schedule derived from
  `indexer_types` (explicit list), `index_topk_pattern` (`"FSS…"` string or
  `["full","shared",…]` list), or the freq/offset formula, all three verified against the real
  78-layer GLM-5.2 schedule. Construction happens under `device_type="cpu"` (torch rejects
  `"tpu"` device strings in the `topk_indices_buffer` alloc; same approach as the existing
  DeepSeek-V4 patch). GLM-5.1-style configs (freq==1, per-layer indexer weights) are unaffected.
- **envs**: `DISABLE_DSA_INDEXER` (reads `TPU_DISABLE_DSA_INDEXER`, PR #2324 lineage),
  **default ON**: vLLM's `SparseAttnIndexer` has no TPU forward (`forward_native` raises
  `NotImplementedError`), so the out-of-box default would crash on every full-schedule layer.
  `mla_attention.py` gates the indexer call on it. The Stage-2 Pallas indexer flips it.
- **kv_cache_manager**: skip vLLM's self-registered `DeepseekV32IndexerCache` in
  `get_kv_cache_spec` — it defines neither `kv_sharing_target_layer_name` nor `attn_type`, so
  engine boot raised `AttributeError` on **every GLM/DSV3.2 config**. The dense stage allocates
  it no KV slot; the Stage-2 sparse path will register a real spec.
- **tpu_platform**: propagate `TPU_DISABLE_DSA_INDEXER` + `DISABLE_WEIGHT_REQUANTIZATION` to
  Ray workers (the raylet env allow-list).

Honest residue (review-flagged): the env attribute name (`DISABLE_DSA_INDEXER`) differs from
its variable (`TPU_DISABLE_DSA_INDEXER`) — kept for PR #2324 lineage/compatibility; and the
constructor wrap duplicates upstream vLLM's schedule derivation. The clean long-term fix is a
one-liner in upstream **vLLM** (`deepseek_v2.py` already computes `_skip_topk` per layer but
still constructs an `Indexer`; the GPU path has the identical IndexShare load failure) — we
intend to offer that as a separate tiny vLLM PR; this wrap stands until it lands.

**Prerequisite:** the MLA wrapper auto-dtype fix (submitted separately) is required for
end-to-end correctness under the default kv-cache dtype; this PR declares it a prerequisite
for the whole-model gate.

## Why this is not duplicating an existing PR

Searches run (2026-07-07): open PRs matching `GLM`, `DSA`, `MLA`, `kv_cache_dtype`. Findings:
the only GLM PR is **#2324** (GLM-**5.1**-FP8 multi-host), which runs DSA *disabled* via env
gating but does **not** register `GlmMoeDsaForCausalLM`, does not handle GLM-5.2 IndexShare
construction, and does not fix the `DeepseekV32IndexerCache` boot crash. The
`TPU_DISABLE_DSA_INDEXER` env name is adopted from #2324 deliberately so the two PRs converge.
No other open PR registers this architecture or touches these code paths.

# Tests

```bash
JAX_PLATFORMS=cpu python -m pytest \
  tests/models/vllm/test_glm_moe_dsa_patch.py \
  tests/models/common/test_model_loader.py \
  tests/test_envs.py \
  tests/layers/vllm/test_mla_attention.py -q
# -> 56 passed, 1 failed (pre-existing: test_getattr_without_cache asserts
#    JAX_PLATFORMS == "" and fails identically on the pristine base under
#    JAX_PLATFORMS=cpu; unrelated to this change)
JAX_PLATFORMS=cpu python -m pytest tests/runner/test_kv_cache_manager.py -q -k get_kv_cache_spec
# -> 19 passed (incl. the new DSA indexer-cache skip test)
```

New tests: registration/resolution to the vLLM impl + PP fallback (4), env parse + default-ON
(1), the IndexShare schedule helper against the real 78-entry GLM-5.2 `indexer_types` in all
three config forms + MTP out-of-range fallback + GLM-5.1 freq==1 (6), patch swap/restore +
shared-vs-full constructor routing + cpu device patch (3), and a KVCacheManager
`get_kv_cache_spec` regression test constructing the **real vLLM `DeepseekV32IndexerCache`**
(1). Adversarial check: on the pristine base the KV-spec test reproduces the exact engine-boot
crash (`AttributeError: 'DeepseekV32IndexerCache' object has no attribute
'kv_sharing_target_layer_name'`), the registration tests fail, and the patch-test module
doesn't import.

Machine-gated functional run (TPU v4, dev branch): 1-chip GLM engine parity (registered model,
native checkpoint keys, dense+MoE+full/shared indexer layers) vs HF `GlmMoeDsaForCausalLM` —
mini bf16 PASS, mini fp8 PASS, real-dims fp8 PASS (final hidden 0.619 < bf16 floor 0.722;
logits 0.910 < 0.958; top-1 vs fp32 0.906 > bf16-ref 0.875) — RESEARCH_LOG 2026-07-07. The
re-cut branch needs the submitter's 1-chip parity re-run.

# Risk

Additive frozenset entries; the construction patch is arch-sniffed and yields early for
non-GLM; the KV-spec skip keys on the vLLM `DeepseekV32IndexerCache` class name; the env
default only affects DSA-family models (the gate is inside `if self.indexer and
self.is_sparse`). Non-GLM configs take byte-identical paths.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commit carries a `Co-authored-by: Claude` trailer.)

---

## Submitter checklist (not part of the PR body)

- [ ] Submit after the independent trio (G2/G3/G4); state the G2 prerequisite in the PR body.
- [ ] Re-run the 1-chip machine-gated parity harness on the re-cut branch (owner TPU time).
- [ ] After forward-porting to current main, re-run the CPU suite there (needs the tip's vLLM pin).
- [ ] Consider pairing with the tiny companion **vLLM** PR (`indexer=None` on skipped layers in
      `deepseek_v2.py`) — see `docs/reviews/stage1-b.md` finding 5.
- [ ] `pre-commit run --all-files`; add DCO `Signed-off-by` when pushing.
