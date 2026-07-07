# 08 — Stage-3 design: MTP speculative decoding for GLM-5.2 on tpu-inference

**Date:** 2026-07-07. **Status:** DESIGN (read-only analysis + CPU-testable skeleton; no TPU touched).
**Scope:** turn GLM-5.2's checkpoint MTP layer 78 into a working vLLM-v1 speculative decoder on the
tpu-inference torchax path (`glm-5.2-v4` fork), with gates from draft-forward parity through
acceptance-length benchmarking. Companion to `docs/01-dsa-kernel-design.md` (Stage 2; §5 IndexShare)
and the Stage-1 parity harness (`parity/glm_engine_*.py`).

---

## 0. Verdict up front

MTP for GLM-5.2 on this stack is **mostly plumbing, not architecture**: vLLM already maps
`glm_moe_dsa → deepseek_mtp` and builds the draft from the exact same `DeepseekV2DecoderLayer` class
the target uses — which means every fork patch that makes the target run on TPU (MLA wrapper OOT
registration, FP8-on-v4 block-scale linears, indexer-on-full-layers-only patch, EP weight filter)
**applies to the draft for free**. tpu-inference's runner already routes `method == "mtp"` through its
Eagle3 proposer with MTP-specific branches, and the torchax wrapper already has an MTP-aware
`draft_step_fun`. The real Stage-3 work is five concrete gaps (§5): a one-line architecture-resolution
fix, lm_head/embed weight-sharing name maps, draft-load streaming cost, the DSA index-share carriage
across draft steps (only needed once `GLM_DSA_MODE≠off`), and verification of KV-group assumptions.
Estimated effort: **~2–3 weeks calendar**, with dense-MTP (the throughput win) landable in the first
half and DSA-mode MTP deferred behind Stage-2a.2's indexer k-cache.

---

## 1. Ground truth: what checkpoint, config, and card give us

### 1.1 Layer-78 weight inventory (`configs/glm-5.2-fp8-keyset.json`, verified fingerprint)

Layer 78 (`model.layers.78.*`) ships:

* **MTP glue:** `eh_proj.weight` (bf16 — **no** `weight_scale_inv`), `enorm.weight`, `hnorm.weight`,
  `shared_head.norm.weight`.
* **Full MLA attention stack** identical to a target layer: `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/
  `kv_b_proj`/`o_proj` (+ FP8 `weight_scale_inv` each), `q_a_layernorm`, `kv_a_layernorm`,
  `input_layernorm`, `post_attention_layernorm`.
* **Full DSA indexer:** `self_attn.indexer.{wq_b, wk (+scale), weights_proj, k_norm.weight, k_norm.bias}` —
  layer 78 is a **full** indexer layer (checkpoint-confirmed; see §1.2).
* **Full MoE:** 256 routed experts (`mlp.experts.N.{gate,up,down}_proj` + scales), `mlp.gate.weight`,
  `mlp.gate.e_score_correction_bias`, shared expert.

Layer 78 does **NOT** ship: `embed_tokens` and `shared_head.head`. Both **must** be shared from the
target model (`model.embed_tokens.weight` / `lm_head.weight` exist only at top level, which the MTP
loader deliberately skips — §2.2). This matches DeepSeek-V3 MTP convention and vLLM GPU behavior
(`llm_base_proposer._maybe_share_lm_head`: *"MTP model → share_lm_head = True … Always share
[shared_head.head] explicitly"* or logits are NaN).

### 1.2 Config facts (`reference/hf-repo/config.json`)

| Key | Value | Consequence |
|---|---|---|
| `num_nextn_predict_layers` | 1 | one MTP layer (idx 78); ≤5 draft tokens = 5 sequential passes of the SAME layer |
| `index_share_for_mtp_iteration` | `true` | draft step 0 computes indices; steps 1+ **reuse step-0 indices** (§4) |
| `indexer_types` | len **78** (layers 0–77) | layer 78 is out of range → falls to the formula |
| `index_skip_topk_offset` / `index_topk_freq` | 3 / 4 | formula for 78: `max(78−3+1,0) % 4 == 0` → **full** — consistent with the checkpoint shipping indexer weights on 78 |
| `index_topk` | 2048 | MTP indexer top-k identical to target layers |

The fork's `_maybe_patch_for_glm_moe_dsa._layer_is_shared` already lands on "full" for layer 78 via
this exact fall-through (its inline NOTE anticipates Stage 3); the skeleton makes the derivation
explicit and asserted (§8).

### 1.3 Model-card / feasibility-memo facts (docs/00 §1)

GLM-5.2 improved its MTP layer with **KVShare, rejection sampling, and end-to-end TV loss**
(training-side techniques), raising ablation acceptance length **4.56 → 5.47 (+20%)** and extending
speculation from ~3 to **up to 5 draft tokens**. The GPU recipes serve with
`--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` (vLLM warns that >1 step reuses
the same MTP layer — for GLM-5.2 that regime is what the model was *trained* for, so the warning is
expected and benign).

**HF `modeling_glm_moe_dsa.py` has NO MTP module** (`GlmMoeDsaModel` ends at layer 77; layer-78
weights are simply ignored by HF). Consequence for gate M1: there is no off-the-shelf HF forward to
diff against — the parity reference must be *composed* (HF `GlmMoeDsaDecoderLayer` for the block +
hand-transcribed enorm/hnorm/eh_proj/shared-head glue; §7 M1).

---

## 2. The vLLM mapping we inherit (GPU reference semantics)

### 2.1 Config surgery (`vllm/config/speculative.py:301-313`)

`SpeculativeConfig.hf_config_override` rewrites the draft config:
`model_type: glm_moe_dsa → deepseek_mtp`, `architectures → ["DeepSeekMTPModel"]`,
`n_predict = num_nextn_predict_layers (=1)`. With `method="mtp"` and no `model`, the draft model
path = the target checkpoint and draft quantization inherits the target's (FP8). So the draft is
"the same checkpoint viewed through a different architecture class."

### 2.2 The draft model (`vllm/model_executor/models/deepseek_mtp.py`)

* `DeepSeekMTP` → `DeepSeekMultiTokenPredictor` → `ModuleDict{"78": DeepSeekMultiTokenPredictorLayer}`.
* Each MTP layer = `enorm` + `hnorm` + `eh_proj` (**plain `nn.Linear`**, bf16 — fine under torchax,
  replicated by `shard_model_to_tpu`) + `SharedHead(norm + ParallelLMHead)` +
  `mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix, config, topk_indices_buffer)` — **the same
  decoder-layer class as the target**, so the fork's `VllmMultiHeadLatentAttentionWrapper`
  (`@MultiHeadLatentAttentionWrapper.register_oot`) is instantiated inside the draft automatically,
  as are the FP8 block-scale linear methods and the wrapped `Indexer` constructor.
* Forward: `inputs_embeds` masked at position 0, `enorm(embeds) ⧺ hnorm(prev_hidden) → eh_proj →
  mtp_block → hidden + residual` (returned **pre-`shared_head.norm`**); `compute_logits` applies
  `shared_head.norm` then `shared_head.head` — matching how tpu-inference splits
  `model_fn` / `compute_logits_fn`.
* `load_weights` name rewrite (`_rewrite_spec_layer_name`): layer-78 block weights →
  `model.layers.78.mtp_block.*`; `enorm/hnorm/eh_proj/shared_head` stay at `model.layers.78.*`;
  `embed_tokens` would be lifted to top level *if present under 78* (GLM: absent). Everything not in
  a spec layer (`get_spec_layer_idx_from_weight_name` → None) — including the top-level
  `model.embed_tokens.weight` and `lm_head.weight` — is **skipped**: the loader never populates
  embed/head; sharing is the proposer's job (GPU) / our shared-params map (TPU, §5 G2).
  Stacked mappings used: `gate_up_proj`←(gate,up), `fused_qkv_a_proj`←(q_a, kv_a_with_mqa),
  `wk_weights_proj`←(indexer wk, weights_proj) + `_try_load_fp8_indexer_wk` for the FP8 wk dequant.
  A final validation demands ≥1 loaded weight per MTP layer (GLM passes trivially).
* `is_v32 = hasattr(config, "index_topk")` → allocates a torch `topk_indices_buffer` at
  `device=current_platform.device_type`. On TPU this only survives construction because
  `_maybe_patch_for_glm_moe_dsa` patches `device_type→"cpu"` during load (the patch keys off the
  **target** `model_config.architectures`, which stays `GlmMoeDsaForCausalLM` during the draft load,
  so it engages — verify item V1). The buffer itself is dead weight on TPU: the fork carries indices
  as JAX arrays in the wrapper context, never through the torch buffer.

### 2.3 `index_share_for_mtp_iteration` — exact GPU semantics (so we replicate, not guess)

`vllm/v1/spec_decode/llm_base_proposer.py`:

1. At init, if the draft has a `topk_indices_buffer`, the **target model's buffer is shared into the
   draft** (storage channel), and `self._share_mtp_indices = hf_config.index_share_for_mtp_iteration`.
2. Before draft **step 0**: `set_skip_topk(False)` — the MTP layer's own indexer (layer 78 is full)
   computes fresh top-k indices for the draft tokens and writes them into the shared buffer.
3. After step 0: `set_skip_topk(True)` — steps 1..k−1 skip the indexer entirely
   (`mla.py: if self.indexer and self.is_sparse and not self.skip_topk`) and their sparse attention
   reads the **step-0 rows** of the buffer.

So precisely: **the MTP iterations 1+ reuse the MTP layer's own step-0 indices** (computed from the
target's final hidden states + first draft token); the buffer sharing with the target is the transport
mechanism, and rows stay request-aligned across steps (decode = 1 token/request/step). Reuse across a
+1 position shift is safe by construction: an index set selected at position p is a valid causal
subset at p+1..p+4, and GLM-5.2 was trained for exactly this reuse (KVShare/TV-loss). TPU carriage of
the same semantics via `ctx.dsa_topk_indices` is specced in §4.

---

## 3. What tpu-inference gives us today (line-verified)

* **`method == "mtp"` is already routed**: `SpeculativeConfig.use_eagle()` returns True for `"mtp"` →
  `tpu_runner._init_speculative_decoding` builds `Eagle3Proposer` (`tpu_runner.py:698`);
  `speculative_decoding_manager.propose_eagle3_draft_token_ids` has an MTP branch that feeds the
  target's **`full_hidden_states`** (the step-fn output, post-final-norm `[T, 6144]`) as the single
  aux hidden state (`speculative_decoding_manager.py:181-184`) — exactly what DeepSeek MTP's
  `previous_hidden_states` wants (GPU passes the same post-norm model output).
* **`Eagle3Proposer` MTP branches** (`spec_decode/jax/eagle3.py`): `_prepare_hidden_states_and_input_ids`
  uses `aux_hidden_states[0]` w/o the eagle3 combine; `_select_inputs_for_loop_speculation` recurses on
  the draft layer's own output hidden states (`residual[0]`, pre-`shared_head.norm`) — the correct
  h^(k−1)→h^(k) MTP recursion; positions/seq_lens/block_tables advance per step
  (`constant_draft_positions` only for `use_gemma4_mtp()`, which is False for GLM → own-KV semantics,
  correct). The k-step loop passes `spec_step_idx=i+1` (harmless: `idx % num_mtp_layers = 0`).
* **`VllmModelWrapper` draft path**: `is_draft_model=True` → `load_weights` builds the model from
  `speculative_config.draft_model_config` (`vllm_model_wrapper.py:540`), applies `shared_params` by
  **exact-name match** (`:562-576`), and `jit_step_func` returns `draft_step_fun` with an MTP branch
  (`spec_step_idx` static kwarg; single-tensor output, `hidden_prenorm = hidden_states`) (`:702-742`).
* **Rejection sampling exists on TPU**: `RejectionSampler` (`tpu_runner.py:1686`), bonus/target logits
  split via `SpecDecodeMetadata` (`speculative_decoding_manager.get_spec_decode_metadata`),
  `max_logits_per_req = num_speculative_tokens + 1` padding (`tpu_runner.py:810`), async-scheduling
  rejected-token subtraction (`_subtract_num_rejected_tokens`). Greedy target + greedy draft ⇒ the
  standard argmax-match acceptance rule ⇒ **output distribution identical to non-speculative greedy**
  (the M2 gate's theoretical basis).
* **Acceptance stats plumbing exists**: the fork's `dp_scheduler` aggregates vLLM `SpecDecodingStats`
  (`num_drafts`, `num_draft_tokens`, `num_accepted_tokens`, per-position) across DP ranks
  (`core/sched/dp_scheduler.py:997-1040`) → Prometheus/log stats. §6 wires this into the provenance DB.
* **Draft KV registration**: the wrapper merges the draft load's `static_forward_context` into the
  shared `vllm_config` (`vllm_model_wrapper.py:555-557`), so the draft MLA layer
  (`model.layers.78.mtp_block.self_attn.attn`) flows through `kv_cache_manager.get_kv_cache_spec`'s
  torchax branch like any target MLA layer and gets the same padded-latent MLA spec. Eagle3 passes the
  runner's full `layer_name_to_kvcache_index` to the draft fn; the draft looks up only its own name.
* **Why DSV4's MTP is stubbed (`_disable_ds_v4_mtp_buffer`) — and why it does not apply here**:
  DeepSeek-V4's **target** model writes `self._mtp_hidden_buffer[:n].copy_(hidden)` inside its own
  forward — an in-place mutation of a persistent torch buffer, which breaks under torchax
  `functional_call` (functionalized params/buffers; in-place `copy_` on an aliased buffer is
  untraceable and the buffer lives on the wrong device). tpu-inference nulls the buffer because DSV4
  MTP is unsupported there. **GLM's MTP has no target-side buffer at all** — the hidden-state handoff
  is `full_hidden_states` through the runner, and the only analogous torch buffer
  (`topk_indices_buffer`) is bypassed on TPU by design (JAX-array carriage). The DSV4 lesson to keep:
  *never route cross-model state through mutable torch buffers under torchax* — everything crosses as
  jitted function inputs/outputs.

---

## 4. DSA × MTP: the index-share design (wrapper-context carriage)

Only relevant when `GLM_DSA_MODE != off`; dense-MTP (M1–M3) skips this entirely, matching Stage-1's
dense serving. Design for M4:

**Within one draft forward** the existing Stage-2a machinery already does the right thing: layer 78 is
a full layer (`indexer is not None`, `skip_topk=False` at construction), so
`_glm_dsa_topk_indices` computes + stashes indices into `ctx.dsa_topk_indices` (per-forward context,
`vllm_model_wrapper_context.py:35`). Nothing persists to the next draft step — the context is rebuilt
per jitted call. That is correct for step 0 and *wrong* (recompute instead of reuse) for steps 1+.

**Cross-step carriage (the delta):**

1. `set_vllm_model_wrapper_context(...)` grows an optional `dsa_topk_indices: Dict[str, jax.Array]`
   seed argument (default empty dict — no behavior change anywhere else).
2. `draft_step_fun_impl` grows:
   * an extra **output**: `ctx.dsa_topk_indices.get("topk_indices")` (the step-0 stash; `None`
     when DSA off — traced as absent, not a dynamic branch);
   * an extra **input** `shared_topk_indices: Optional[jax.Array]`, seeded into the context.
3. Per-layer resolution in the wrapper (`mla_attention._glm_dsa_topk_indices`) becomes:
   `is_full_layer = indexer is not None and not skip_topk and not context_preseeded`. Since
   `spec_step_idx` is already a **static** argname, "step 0" (compute+stash+emit) and "step ≥1"
   (seeded, fetch-only) are two distinct traces — no runtime toggling, unlike the GPU's mutable
   `set_skip_topk` (which would not survive jit anyway).
4. `Eagle3Proposer._propose` threads the step-0 indices into the loop calls (one extra loop-carried
   array `[num_reqs, 2048] int32` — trivial vs the KV caches already carried).
5. Trace count: collapse `spec_step_idx` to a static **bool** `is_first_spec_step` for the MTP method
   (DeepSeekMTP only uses `idx % 1 == 0`), so 5 draft steps compile **2** draft programs per bucket
   instead of 5 — a real warmup saving on the pod.
6. Fetch-path error semantics: `fetch_shared_topk_indices` currently raises if nothing is stashed —
   the seeded path satisfies it by construction; keep the raise (mis-wiring stays loud).

**Indexer k-cache:** layer 78 is a full indexer layer, so once Stage-2a.2's indexer-key cache lands,
the draft layer needs an indexer k-cache slot exactly like target full layers (the current
`is_dsa_indexer_cache` skip in `kv_cache_manager` covers dense mode). Draft indexer keys are written
for draft tokens at their (advancing) positions — same write-then-score discipline as the target
(docs/01 §2). Positions p+1..p+4 reusing step-0 indices never *read* keys the cache lacks (reused
indices point strictly into the pre-existing prefix).

---

## 5. Gap list — the actual Stage-3 engineering

**G1 — architecture resolution (1 line + test).** `resolve_model_architecture(is_draft=True)` reads
the draft arch `DeepSeekMTPModel`, which is neither in the JAX registry nor in
`_VLLM_PREFERRED_ARCHITECTURES` → resolves `"flax_nnx"`, while the target resolves `"vllm"` →
`Eagle3Proposer.load_model`'s impl-equality check (`eagle3.py:107-110`) raises **before any load**.
Fix: add `"DeepSeekMTPModel"` to `_VLLM_PREFERRED_ARCHITECTURES` (`models/common/model_loader.py:53`).
(`get_model` would even fall back correctly without it; only the equality check breaks.)

**G2 — shared weights need a NAME MAP, not name equality.** The wrapper's shared-params hook shares
only when target key == draft key. Embedding works today
(`vllm_model.model.embed_tokens.weight` matches both sides and is in `Eagle3Proposer.load_model`'s
`shared_names`). The head does **not**: target `vllm_model.lm_head.weight` must land on draft
`vllm_model.model.layers.78.shared_head.head.weight`. Change: `Eagle3Proposer.load_model` builds
`shared_params = {draft_param_name: runner.state[target_key]}` from a per-method map — provided by the
skeleton (`glm_mtp/shared_weights.py: mtp_shared_param_map`). Two hardening notes:
* the wrapper's shape-mismatch guard currently *warns and skips* — for MTP head/embed that means
  garbage `torch.empty` logits at runtime; make missing/mismatched MTP shares a **hard error**
  (mirror the GPU comment: unshared `shared_head.head` ⇒ NaN logits).
* **sharding is preserved for free**: shared params are torchax views of the target's on-TPU arrays,
  so `shard_model_to_tpu`'s `_tensor_is_in_cpu` predicate skips them (`cleanup_sharding.py:87-92`) —
  the draft head keeps the target lm_head's vocab-sharded layout (no 1.9 GB/chip replication).
  Verify item V3: confirm no type-registered sharding fn re-places them first, and that the wrapper's
  `_logits_partition_spec` lm_head probe (which reads `self.model.vllm_model.lm_head`) is taught the
  MTP head path (draft has no `.lm_head` attr → probe currently falls back to "replicated" and would
  pick the token-sharded layout against a vocab-sharded shared head → the exact 1.77 GiB all-gather
  the probe exists to avoid).

**G3 — draft-load streaming cost.** The draft load re-streams all 150 files (755.7 GB) through the
host just to keep layer-78 (+nothing else — embed/head are skipped and shared). Two mitigations,
either sufficient: (a) filter `_prepare_weights`' file list by `model.safetensors.index.json`
weight_map to files containing `model.layers.78.` keys when loading the draft (RunAI loader already
has the sorted-file hook); (b) accept one extra pass (~45 min at staging bandwidth — tolerable but
annoying at every boot). Recommend (a); ~20 lines in `vllm_model_loader.py`. The **EP filter applies
unchanged**: `compute_local_expert_ids_from_sharding` probes the mesh off the draft's own
quant-method (same mesh, same `EXPERT`-axis slices → same local ids); `should_skip_weight` matches
`.experts.<id>.` names regardless of layer; the coverage verifier passes because layer 78 carries all
256 experts (every local id receives tensors even under the file filter). Draft memory: ~9.7 GB FP8
experts EP-sharded (~0.3 GB/chip) + ~0.2 GB replicated dense/attn/indexer + eh_proj 151 MB bf16 —
well inside post-load headroom; draft KV = 1 extra MLA layer ≈ 1/78 of target KV per token.

**G4 — DSA index share across draft steps** — §4 (M4 only; dense-MTP unaffected).

**G5 — verification items (no code until proven needed).**
* V1: `_maybe_patch_for_glm_moe_dsa` engages during the draft load (target-arch keyed) — assert in the
  bring-up test that the draft's `topk_indices_buffer` built on CPU and layer 78 got an Indexer.
* V2: KV-group behavior — the draft MLA layer's spec is identical to target MLA layers, so vLLM groups
  it WITH them (shared block table, own per-layer cache tensor). That is functionally correct
  (GPU MTP does the same); but `Eagle3Proposer.prepare_inputs` assumes "the LAST KV-cache group is the
  draft's" — with a single unified group the last group IS the shared one and the code degenerates
  correctly. Confirm on the mini-model harness (single group expected).
* V3: G2 sharding notes above.
* V4: `filter_speculative_logprobs` and async scheduling with MTP (`use_eagle()` gate covers it) — the
  eagle3 code paths are DP-aware; GLM Stage-1 serves pure TP×EP (dp=1), which is the simpler case.

**G6 — batch shaping.** `num_speculative_tokens=5` ⇒ decode schedules 6 tokens/request through the
target (`max_logits_per_req=6`), which *helps* the docs/03 bucket-512 padding problem (denser buckets)
but multiplies target decode-forward tokens ×6 — the throughput win comes entirely from acceptance
(expected ~5.47/6 useful). Precompile cost: +2 draft programs per bucket (§4.5) + the existing
spec-decode target shapes. Start bring-up with `num_speculative_tokens=1` (single trace, simplest
equivalence), then sweep 1..5 in M3.

---

## 6. KVShare semantics — what we implement vs what the card means

Evidence-based reading: **KVShare is a training-time robustness technique** (with rejection-sampling
training and end-to-end TV loss) that makes the MTP layer tolerate *shared/stale per-step state* so
inference can cheapen the speculation loop. Its inference-visible artifacts in the reference stack are
exactly two: (1) `index_share_for_mtp_iteration=true` — reuse of step-0 top-k indices across MTP
iterations (§2.3/§4); (2) the acceptance uplift (4.56→5.47) that makes 5 draft tokens worthwhile.
Neither vLLM nor SGLang shares the *latent KV cache* between target and MTP layer — the draft keeps
its **own** MLA KV (layer 78 has its own attention weights; its K/V live in its own cache tensor), and
positions advance per draft step. tpu-inference's only full-cache-share machinery
(`use_gemma4_mtp()` → `compute_mtp_kv_share_map` + `constant_draft_positions`) targets Gemma4's
weight-tied MTP and is **not applicable** to GLM. Decision: **own draft KV, advancing positions,
deepseek_mtp semantics** — which is what the existing `method=="mtp"` branches already do. Rejected
alternative (share target KV into the draft): no reference implementation, no trained guarantee, and
layer 78's K/V are produced by different projections — would be a silent-quality experiment, not a port.

Accepted-token KV persistence follows the standard eagle discipline already on TPU: draft KV written
at speculative positions is simply overwritten on the next step's re-run after rejection trimming
(`num_rejected_tokens` → `prepare_inputs` token filtering).

---

## 7. Phased plan with gates

**M0 — CPU contract checks (this commit).** Skeleton `tpu_inference/models/vllm/glm_mtp/` (§8) +
self-check runnable now, no TPU, no weights: (a) config surgery on the real `config.json` yields
`DeepSeekMTPModel`/`n_predict=1`; (b) layer-78 indexer schedule derivation == full; (c) **loader-name
coverage**: every layer-78 key in `glm-5.2-fp8-keyset.json` classifies into a DeepSeekMTP param
destination under the transcribed `_rewrite_spec_layer_name` + stacked/expert mappings, with exactly
`{embed_tokens, shared_head.head}` expected-missing (the shared set) — cross-checked against the
installed vLLM's own rewrite function when importable.
**Gate M0: self-check exit 0.** (Done — see §9.)

**M1 — draft-forward parity vs composed HF layer-78 (single chip, ~2–4 days).** Extend the Stage-1
harness (`parity/glm_engine_common.py` machinery, same synthetic-checkpoint discipline: mini bf16 +
real-dims fp8-block-128): reference = HF `GlmMoeDsaDecoderLayer` instantiated from the target config
(+ indexer active per DSA mode) wrapped with transcribed enorm/hnorm/eh_proj/shared_head math (fp32
and bf16 controls); candidate = the loaded `DeepSeekMTP` through `draft_step_fun` + wrapper
`compute_logits`. Feed matched `(input_ids, previous_hidden_states, positions)` including a
position-0 row (embed-masking branch). **Gate M1 (Stage-1 thresholds discipline): candidate-vs-fp32
divergence ≤ bf16-control floor; top-1 agreement ≥ bf16 control; plus the M0-style load audit — zero
unexpected/missing params after sharing.**

**M2 — greedy spec-decode equivalence (pod, ~3–5 days incl. bring-up).** Serve GLM-5.2-FP8 with the
Stage-1 recipe + `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'`, greedy, for
k ∈ {1, 5}. **Gate M2: for N≥32 prompts (bench suite prompts + long generations), the generated token
sequences EQUAL the non-speculative greedy baseline exactly.** Theory guarantees distribution-level
equality; token-level equality additionally requires the target's argmax to be stable across batch
shapes (spec step evaluates 6-token rows vs 1-token rows — low-bit logit wobble can flip only exact
ties). Protocol: assert exact match; on any mismatch, dump the step logits and classify — a tie flip
(top-2 gap below bf16 eps) is a documented pass-with-note, anything else is a bug. Also gate:
non-spec serving numbers unregressed (byte-identical config path when `speculative_config is None`).

**M3 — acceptance-length + throughput benchmark (~1–2 days).** Instrumentation for the provenance DB:
after each bench run, scrape vLLM v1 metrics (`LLM.get_metrics()` /
`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}[_per_pos]`, DP-aggregated by the fork's
`dp_scheduler`) and record run-level rows: `acceptance_len = 1 + accepted/drafts`, per-position
acceptance vector (5 entries), drafts/sec, plus the usual env provenance. Sweep k=1..5 on the bench
decode workloads + one long-generation workload. **Gate M3: mean acceptance length ≥ ~4.5 at k=5 on
generic decode (card ablation 5.47; greedy + our workloads may differ), and end-to-end decode tok/s
uplift ≥ 2× vs non-spec at equal batch (docs/03 row 8 predicted 2–4×).** Below ~3.5 acceptance →
investigate before shipping (likely hidden-state/positions off-by-one — the classic MTP bug).

**M4 — DSA-mode MTP (after Stage-2a.2's indexer k-cache; ~3–5 days).** Implement §4 carriage +
draft indexer k-cache slot; gates: (a) step-0-indices parity vs recompute-per-step on the harness,
(b) M2 rerun under `GLM_DSA_MODE=xla_ref` (equivalence vs DSA non-spec greedy), (c) M3 rerun —
acceptance must not degrade vs dense-MTP beyond noise (index reuse is what the model was trained for).

**Effort summary:** G1+G2+M1 ≈ 1 week; M2+M3 ≈ 1 week of pod time (amortized with other pod work);
M4 ≈ 1 week attached to Stage-2a.2. Dense-MTP value (2–4× decode) does not wait for DSA.

---

## 8. Skeleton shipped with this doc (CPU-tested now)

`tpu-inference@glm-5.2-v4: tpu_inference/models/vllm/glm_mtp/` — deliberately *contract-level* (no
runner/proposer edits yet; those land with M1/M2 diffs):

* `draft_config.py` — `mtp_layer_indexer_is_full(hf_config, layer_id)` (the explicit §1.2 derivation
  with the beyond-`indexer_types` fall-through asserted), `classify_mtp_checkpoint_key(...)` (the
  transcribed loader-name contract), and a `__main__` self-check against the real
  `config.json`/keyset (cross-checks the transcription against installed vLLM where importable).
* `shared_weights.py` — `mtp_shared_param_map(num_hidden_layers)`: the G2 target-key → draft-param
  map ({embed→embed, lm_head→layers.78.shared_head.head}) with the must-share hard-error contract in
  the docstring.
* `index_share.py` — the §4 seeding contract (`is_first_spec_step` collapse, seed/emit helper
  signatures) as typed stubs with the design invariants documented, so the M4 diff is mechanical.

Run: `JAX_PLATFORMS=cpu ~/vllm-env/bin/python -m tpu_inference.models.vllm.glm_mtp.draft_config \
--config ~/glm-tpu/reference/hf-repo/config.json --keyset ~/glm-tpu/configs/glm-5.2-fp8-keyset.json`

## 9. Self-check results (2026-07-07, CPU)

See RESEARCH_LOG entry of this date: M0 self-check PASS — config surgery OK (`DeepSeekMTPModel`,
n_predict 1), layer-78 = full indexer layer via fall-through, 39/39 layer-78 keyset keys classified
to loader destinations, expected-missing set exactly `{model.embed_tokens, shared_head.head}`,
rewrite transcription bit-identical to installed vLLM's `_rewrite_spec_layer_name` on all keys.
