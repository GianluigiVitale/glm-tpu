# Round-6 adversarial review — Stage-2a.2 paged indexer k-cache (2a2) + round-5 fixes (r5fix)

**Reviewer scope.** Two worktrees, attacked independently and cross-checked:

- `~/tpu-inference-2a2`, branch `glm-5.2-v4-2a2`, commit `3ca5c85c` — DSA indexer
  k-cache: KVCacheSpec registration via `model_version` discriminator, write-then-score
  paged path, hierarchical top-k.
- `~/tpu-inference-r5fix`, branch `glm-5.2-v4-r5fix`, commit `c8a51543` — round-5 fixes
  incl. the fp8 `wq_b` dequant adapter, xla_ref positions guard, S1 DP-mesh refusal.

All verification CPU-only (`JAX_PLATFORMS=cpu` set before interpreter start; no TPU
touched). vLLM machinery checked against the fork actually installed
(`vllm 0.1.dev1+ga30addc75` at `~/vllm-build/vllm` — the version every claim below is
pinned to; upstream drift re-opens them).

**Test runs (both branches, executed by this review):**

- 2a2: `test_glm_dsa_indexer.py` (28) + `test_dsa_indexer_kernel.py` +
  `test_dsa_sparse_mla.py` + `test_mla_head_sharded.py` → **87 passed**, plus
  `test_mla_attention.py` → **6 passed** (93 total across the 5 DSA/MLA files).
- r5fix: the same 5 files → **93 passed**.

Zero failures, zero errors, in `vllm-env`. Matches the commit messages' claims.

---

## Summary verdict

The core engineering of both commits **survives attack**. I could not break the
group-formation trick against the real vLLM machinery (I drove it further than the
author's own tests — through the real scheduler `KVCacheManager`, coordinator,
`find_longest_cache_hit`, and prefix-cache hit/reuse — and it behaves as designed), and
the write-then-score paged path's tests are genuinely strong (shuffled physical pages,
bit-exactness gates, 1-token chunkings, mixed ragged decode). The round-5 fp8 CRITICAL
is **confirmed real, not theoretical** against the actual GLM-5.2-FP8 checkpoint keyset.
The genuine findings are: one **confirmed latent foot-gun** in the paged write API
(finding 1), one **confirmed real bug** in the new r5fix DP-mesh guard under the
non-production axis naming (finding 2), and a set of merge/wiring hazards that are
process- rather than code-level (findings 3–6).

---

## Findings

### 1. MEDIUM (latent until wiring) — `valid` defaults to `None`: the pad-clobber protection is opt-in, and the runner's pad positions are exactly the garbage that clobbers

- **Where:** `tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py:517-540`
  (`write_indexer_keys`, `valid=None` default), `:607-616`
  (`compute_topk_indices_paged`, same default), `:689-699`
  (`topk_indices_for_layer_paged`, same default) — 2a2 branch.
- **Failure scenario:** the runner's `positions_cpu` is a persistent zeros buffer
  (`tpu_inference/runner/tpu_runner.py:781`); the padded tail of a ragged batch carries
  zeros or **stale positions from previous steps**. A future Stage-3 caller that wires
  `compute_topk_indices_paged` and forgets `valid=` gets: pad token → `token_request_ids`
  clips its req id to the last request → `indexer_cache_slots` computes an **in-range
  live slot** (position 0, or a stale position, of that request) → the scatter runs with
  `mode="drop"` but the index is *in bounds*, so the pad row's key **overwrites a live
  cached key**. Silent corruption of decode-time indexer scores for that request; no
  gate catches it (scores stay plausible).
- **Evidence:** `write_indexer_keys` only diverts slots out-of-range when `valid` is
  passed (`slots = jnp.where(valid, slots, num_slots)`); with `valid=None` the scatter
  applies everything in range. The author's own test
  (`test_padded_token_writes_are_dropped`, tests/layers/vllm/test_glm_dsa_indexer.py:741)
  *proves the hazard is live*: it asserts the unmasked write **does** clobber
  (`cache_bad != cache_ref`). So the mechanism is correct when used correctly — the
  finding is that correctness depends on a keyword argument every future caller must
  remember, in a codebase where the ragged pad is the norm.
- **Refutation attempted:** none possible — the hazard is pinned by the author's own
  test. What I *can* refute is present-tense impact: **no caller exists yet** (the
  runner wiring is Stage 3), so nothing is broken today.
- **Recommendation:** before wiring, make `valid` required (no default) on
  `compute_topk_indices_paged` / `topk_indices_for_layer_paged`, or derive it internally
  from `query_start_loc` (the `token_request_ids` helper already computes exactly the
  right mask — force the two to travel together).

### 2. LOW-MEDIUM (confirmed bug, inert in the production config) — the r5fix DP-mesh refusal guard degenerates to a character set under `ShardingAxisName2D` and never fires

- **Where:** `tpu_inference/layers/common/attention_interface.py:564-567` (r5fix).
- **Failure scenario:** `set(ShardingAxisName.ATTN_DATA) | set(ShardingAxisName.BATCH)`
  assumes tuple-form axis names. Under the **default** axis naming
  (`ShardingAxisName2D`, selected whenever `NEW_MODEL_DESIGN`/`USE_2D_TP` are unset —
  `sharding.py:106-116`), `ATTN_DATA = 'data'` is a plain string, so
  `set('data') = {'d','a','t'}` and the guard computes
  `mesh.shape.get('d',1)*... == 1` on **any** mesh — including `data=2`. The refusal
  the guard exists for (head-sharded specs pairing global token axes with DP-local
  metadata → silent wrong outputs) is then bypassed silently.
- **Evidence (executed):**
  ```
  ShardingAxisNameBase ATTN_DATA: ('data','attn_dp','attn_dp_expert') -> _dp_axes: ['attn_dp','attn_dp_expert','data']  -> product on data=2 mesh: 2
  ShardingAxisName2D   ATTN_DATA: 'data'                              -> _dp_axes: ['a','d','t']                        -> product on data=2 mesh: 1
  ```
  The r5fix test (`test_head_sharded_refuses_dp_mesh`) passes **only** because its
  `base_sharding` fixture pins `ShardingAxisNameBase`
  (tests/layers/common/test_mla_head_sharded.py:166-170) — the 2D naming is untested.
- **Mitigating fact (verified):** the GLM launcher bakes `NEW_MODEL_DESIGN=1`
  (`~/glm-tpu/scripts/launch_glm_32chip.sh:94`), which selects the tuple form, so the
  guard **works in the config this pod actually runs**. It is dead code, not wrong
  code, everywhere else.
- **Recommendation:** normalize str-vs-tuple the way `utils.get_mesh_shape_product`
  already does (it accepts `str | list`), e.g.
  `_dp_size = get_mesh_shape_product(mesh, list(<normalized ATTN_DATA>) + ...)` — one
  line, and add a `ShardingAxisName2D` parametrization to the refusal test.

### 3. MEDIUM (process) — the two branches each lack the other's critical fix; merging conflicts textually AND semantically

- **Where:** `glm_dsa_indexer.py` on both branches; tests.
- **Evidence:** `grep _linear_weight_f32` in 2a2 → **0 hits** (its paged path, run
  against the real FP8 checkpoint, would score raw fp8 `wq_b` codes — the exact round-5
  CRITICAL, unfixed on that branch); `grep compute_topk_indices_paged` in r5fix →
  **0 hits**. `git merge-tree` of the two heads produces **2 conflict hunks**
  (`glm_dsa_indexer.py` import region + test file), plus one **semantic** conflict the
  textual merge will not flag: 2a2 changed the `k_cache=` rejection in
  `compute_topk_indices` to `TypeError` ("…is compute_topk_indices_paged") while r5fix's
  `test_k_cache_is_2a2_gated` still expects `NotImplementedError` matching `"2a.2"`.
  Whichever direction merges, one suite breaks until reconciled.
- **No refutation needed** — this is arithmetic. The good news: once merged, the fp8
  dequant applies to the paged path automatically (both paths consume
  `params_from_vllm_indexer`; the paged functions take already-adapted `params`).

### 4. LOW (latent, documented-elsewhere-but-not-here) — the paged helpers' flat-cache view silently assumes `MLA_TRANSPOSE_KV_CACHE=0`

- **Where:** `glm_dsa_indexer.py:475-481` (2a2): "the mla.v2 layout
  `[num_pages, page_size // packing, packing, 128]` … row-major-identical to the flat
  form — reshape, don't transpose."
- **Failure scenario:** with `MLA_TRANSPOSE_KV_CACHE=1`, `mla.v2.get_kv_cache_shape`
  returns `(pages, kv_dim, page_size)` (kernels/mla/v2/kernel.py:103-109) — **not**
  row-major-identical to `[pages, page_size, head_dim]`. A Stage-3 caller reshaping the
  device array into the paged helpers would interleave key components silently.
  The spec/page-budget math stays consistent either way (same byte product through the
  same `get_attention_page_size_bytes`), so this corrupts *content*, not accounting.
- **Mitigating facts (verified):** env default is False (`envs.py:411-412`), the
  launcher does not set it, and r5fix documents the incompatibility for the 2c gather —
  but the 2a.2 module docstring does not.
- **Recommendation:** at wiring time, one loud
  `assert not envs.MLA_TRANSPOSE_KV_CACHE` next to the reshape.

### 5. LOW (operational, on-TPU-verifiable only) — `GLM_DSA_MODE` now shapes the KV-cache **spec**, so per-host env divergence graduates from "wrong mode" to "inconsistent cache topology"

- **Where:** `tpu_inference/runner/kv_cache_manager.py:707-716` (spec loop) + `:167`
  (`_dsa_indexer_spec_for_mode` reads the env at spec time).
- **Failure scenario:** vLLM's Ray executor forwards only `VLLM_*` + an allow-list; if
  `GLM_DSA_MODE` reaches some raylets and not others (the exact class r5fix documents
  for `GLM_MLA_HEAD_SHARDED`), workers submit **different layer sets** (99 vs 78
  specs). `get_kv_cache_configs`' merge loop unions layer names and only asserts
  equality *for layers present on both* — no loud cross-worker check fires; divergence
  surfaces later as differing per-worker tensor lists / SPMD program skew (hang class).
- **Status:** the operative mitigation exists (launcher bakes `GLM_*` into the raylet
  env on all hosts); this finding is a reminder that 2a.2 raised the stakes on that
  discipline. Only an on-TPU multi-host bring-up can verify it end-to-end.

### 6. INFO — spec-decode (MTP) rewrite-in-place is argued-correct but untested

- **Where:** `compute_topk_indices_paged` causality (`kv_lens = positions + 1`).
- **Analysis (mine, not the author's):** with MTP lookahead, rejected draft tokens
  leave **stale keys** at positions beyond the accepted length. Per-token
  `kv_len = position + 1` recovers correctly *iff* every subsequently processed token
  writes its own slot before anything scores a kv_len covering it — which
  write-then-score guarantees, since the real token at a formerly-stale position
  overwrites that slot in its own step, and no earlier step's kv_len reaches it.
  So the semantics hold — but no test exercises the *overwrite* pattern (write pos P,
  later re-write pos P with a different key, assert scoring uses the new key). The
  existing suites only ever write each slot once.
- **Recommendation:** add a 5-line "rejected-draft rewrite" case to the decode test
  before Stage-3 MTP lands. Also note the checkpoint has an **indexer on MTP layer 78**
  (`glm-5.2-fp8-keyset.json`) — Stage 3 must decide its spec/IndexShare treatment.

---

## Attack results in detail (what I tried to break, and what held)

### Attack 1 — the KVCacheSpec `model_version` discriminator vs the REAL vLLM machinery

**CONFIRMED, and beyond the author's own test.** Read directly from the installed
fork's source and then executed:

- `MLAAttentionSpec.merge` (`vllm/v1/kv_cache_interface.py:386-412`) asserts
  `len(model_version_set) == 1` and **never compares `head_size`** (it copies
  `specs[0]` fields) — both halves of the author's claim are true in this vLLM. The
  silent-absorb hazard the discriminator exists for is real: without it,
  `is_kv_cache_spec_uniform` returns True and a 128-wide spec merges first-spec-wins
  into the 640 group. The author's test pins this so upstream drift will be caught.
- `is_kv_cache_spec_uniform` catches exactly `AssertionError`
  (`vllm/v1/core/kv_cache_utils.py:874-897`) → the discriminator routes to
  `UniformTypeKVCacheSpecs.from_specs` (same `block_size`; both specs are
  `MLAAttentionSpec` whose registered `uniform_type_base_spec` is `FullAttentionSpec`
  — `single_type_kv_cache_manager.py:1388-1390`) → **one group**.
- **Executed with the GLM-shaped layout** (78×`MLAAttentionSpec(head=640)` interleaved
  with 21×indexer(head=128, `model_version="glm_dsa_indexer"`), block 32, 1 GiB):
  1 group, 99 per-layer `KVCacheTensor`s each `shared_by` exactly one layer, group page
  `78*40960 + 21*8192 = 3,366,912 B`, `num_blocks = 318 = mem // group_page` — the
  commit's page-budget math is exactly what vLLM computes
  (`kv_cache_utils.py:1273-1290`, uniform-type single-group branch). The `+5.4%`/token
  claim checks out (`172032/3194880 = 5.38%` over the MLA-only budget).
- **Scheduler side — the part the author's test did NOT cover — driven end-to-end:**
  the engine rewrites the worker config through `generate_scheduler_kv_cache_config`
  (`kv_cache_utils.py:1698-1717`, called from `engine/core.py:283`), replacing the
  `UniformTypeKVCacheSpecs` with an arbitrary inner spec — this matters because
  `UniformTypeKVCacheSpecs` itself has **no registered manager** (a naïve coordinator
  construction with the raw worker config asserts `No manager registered`; I hit this
  before finding the rewrite). With the rewrite: `UnitaryKVCacheCoordinator` +
  `FullAttentionManager`, and a real `KVCacheManager` round-trip — allocate 100 tokens
  on request 1, `cache_blocks(96)`, then `get_computed_blocks` for an identical
  request 2 returns a **96-token prefix hit on shared blocks [1,2,3]** through the real
  `find_longest_cache_hit`. Per-layer page sizes never reach the scheduler (it sees
  only `block_size`), so "find_longest_cache_hit with per-layer page sizes" cannot
  misalign — the arbitrary-inner-spec choice is safe because both specs agree on
  `block_size` and manager class. `get_max_concurrency_for_kv_cache_config` on the
  worker config: 0.0776 = 318·32/131072 ✓.
- **Worker side:** `initialize_kv_cache` (runner kv_cache_manager.py:797-1013) resolves
  per-layer specs via `hasattr(group_spec, 'kv_cache_specs')`, computes per-tensor
  `num_blocks = tensor.size // per-layer page == config num_blocks` for **both** page
  sizes, applies the same BATCH-divisor rounding to all layers, and allocates the
  indexer cache through the same `mla.get_kv_cache_shape` that computed the spec's
  `page_size_padded` — spec and allocation cannot drift by construction
  (`get_attention_page_size_bytes` shares the shape function, kv_cache.py:155-168).
- **`model_version="glm_dsa_indexer"` inertness — exhaustively grepped:** the only
  consumers in this vLLM are (a) `fp8_ds_mla` page-size special case (requires
  `cache_dtype_str == "fp8_ds_mla"`), (b) `SlidingWindowMLASpec.__post_init__`'s
  `assert model_version in (None, "deepseek_v4")` — a *different class*, never
  instantiated here, and (c) `_annotate_eagle_groups_deepseek_v4` — filters on
  `== "deepseek_v4"` and runs only on the DSV4 multi-group path. All inert. ✓

**Misalignment constructions attempted (all fail to produce silent corruption):**

- *Different block sizes:* impossible — the indexer spec is built from the same
  `block_size` variable inside the same loop as the MLA spec.
- *`--kv-cache-dtype fp8_ds_mla`:* the indexer spec would inherit the GPU 656 B/token
  `real_page_size_bytes` while its TPU `page_size_padded` is 4096 →
  `assert page_size_padded >= real_page_size` (`kv_cache_interface.py:178`) fires at
  group formation → **loud startup crash**, not misalignment. (The main MLA spec has
  the same pre-existing property on TPU.)
- *`index_head_dim` not a multiple of 128 (e.g. 96):* spec pads to 128 (test-pinned);
  the write helper would then see 96-wide keys vs a 128-wide cache → **loud shape
  error**, not silent. GLM-5.2 config confirms `index_head_dim: 128` exactly.
- *`MLA_TRANSPOSE_KV_CACHE=1`:* accounting stays aligned; content layout breaks —
  finding 4 (default off).
- *Per-host env divergence:* finding 5 (operational).

### Attack 2 — write-then-score 'drop' scatter at ragged batches

**Mechanism CONFIRMED correct when used as designed; API default is the hazard
(finding 1).** Specifics verified by reading + the passing suite:

- `mode="drop"` semantics: pad slots are set to `num_slots` (one-past-the-end) so the
  scatter drops them; JAX drops only *out-of-bounds* scatter indices, which is why the
  divert-then-drop two-step is load-bearing — a pad row with an in-range garbage slot
  and no `valid` **does** land (author's test asserts exactly this, both directions:
  masked == real-rows-only, unmasked ≠).
- `token_request_ids` handles right-padded `query_start_loc` (repeated totals):
  hand-checked `searchsorted(side="right")-1` + clip on `qsl=[0,3,5,5,5]` — req ids
  `[0,0,0,1,1]`, pads → clipped id 3 with `valid=False`; block-table row 3 exists in a
  real `[max_num_reqs, B]` table, so even the pre-divert gather is in-range.
- Slot content at ragged batches: real tokens' slots come from `positions` +
  `block_tables[req_ids]` — the runner's block table always covers
  `position // page_size` for scheduled tokens (allocation invariant incl. lookahead
  slots), so real-token writes cannot go out of range. The unclamped
  `take_along_axis(bt, pos // page_size)` for garbage pad positions resolves to *some*
  in-range page under jit gather semantics — harmless **only** because `valid` diverts
  the slot afterward; this ordering (clip-gather first, divert second) is correct as
  written.
- Cross-request clobber via shared prefix pages: not possible — prefix-cache-shared
  full blocks are never re-written (their token positions are < num_computed_tokens of
  every sharer, so no scheduled token maps to them), and a reallocated page is fully
  re-written slot-by-slot before any kv_len covers it.

### Attack 3 — multi-chunk == single-chunk; is `kv_len = position + 1` the serving semantics?

**CONFIRMED for everything servable today.** The test suite is genuinely adversarial:
shuffled non-contiguous physical pages, three chunkings including `[1]*8+[56]`, bitwise
equality on **indices and cache state**, decode == full-history row with topk both < and
> history, a mixed two-request ragged decode through `req_ids`, bf16-cache ==
bf16-rounded-reference, and merge-block invariance. The score path is additionally
pinned bit-exact against the 2a.1 `weighted_scores` at matched blocking and ulp-checked
against the independent 2b `indexer_scores_xla` formulation.

Semantics cross-check against serving (my analysis):

- Standard decode / chunked prefill: positions are token ordinals; per-token
  `kv_len = pos + 1` is exactly the GPU reference's per-token causal history
  (vLLM `Indexer` inserts-then-scores with cu_seqlen/context-lens; same discipline).
- Prefix-cache resume: works **because of** attack 1's one-group design — a hit block
  shares the same page ids across the MLA latent and indexer tensors, and the donor
  request wrote the indexer keys at those slots in the same forward pass that computed
  the latents. MLA-computed ⟺ indexer-computed per block, by construction.
- Mid-chunk causality: keys of the whole chunk land before scoring, and later-position
  keys are masked per-token — the "1-token chunks == one chunk" bitwise gate is the
  strongest possible pin of this.
- Spec-decode lookahead: argued-correct via rewrite-in-place (finding 6 — add the
  missing overwrite test). One boundary the semantics do NOT cover: models where
  positions ≠ token ordinals (mrope-style multimodal position ids) — irrelevant for
  text-only GLM-5.2, worth one contract line.
- Width contract divergence (paged always returns `topk` wide, 2a.1 clips to
  `min(topk, T)`) is documented and test-pinned; padded rows produce
  `n_valid = 1` rows (not all `-1`) — the r5fix contracts already require the glue to
  discard pad-row outputs.

### Attack 4 — fp8 dequant adapter vs the REAL GLM-5.2-FP8 checkpoint

**The round-5 CRITICAL was REAL, not theoretical.** From
`~/glm-tpu/configs/glm-5.2-fp8-keyset.json` (the staged checkpoint's key inventory):

- Every indexer-bearing layer class (layers 0-2, the 18-layer class within 6..74,
  and MTP layer 78) ships `self_attn.indexer.wq_b.weight` **plus**
  `self_attn.indexer.wq_b.weight_scale_inv` → `wq_b` is fp8 **block-quantized** in the
  real checkpoint.
- `modules_to_not_convert` (541 entries, checked): **zero** `wq_b`, `wk`, or
  `weights_proj` entries — only `indexer.k_norm`(±bias) and `indexers_proj` appear.
  So the fork's `VllmFp8LinearMethod` will leave `wq_b` fp8-resident under the
  launcher's `REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn` / `DISABLE_WEIGHT_REQUANTIZATION`
  configs; without `_linear_weight_f32` the adapter reads raw codes — silent garbage
  that gate D0 (topk ≥ T selects everything) cannot see. The fix is necessary.
- `indexer.wk` is **also** fp8 (+`weight_scale_inv`) — but handled a layer up:
  `GlmMoeDsaForCausalLM` maps to `deepseek_v2.py` (registry.py:124), whose `Indexer`
  builds `wk_weights_proj` with `quant_config=None` and load-time-dequantizes fp8 wk
  via `_try_load_fp8_indexer_wk` (deepseek_v2.py:636-642, 750-785). The adapter
  therefore sees a plain bf16 fused weight (its ≥16-bit float fast path). ✓
- `weights_proj` is bf16 (no scale key). ✓
- Scale-format coverage: the fork's `process_weights_after_loading` (fp8.py:173-261)
  leaves `layer.weight` `[in,out]`-transposed fp8 + `layer.weight_scale` (renamed from
  `weight_scale_inv`, transposed, ue8m0→f32 pre-converted at fp8.py:184). Semantics
  check: `weight_scale_inv` is the **multiplicative** dequant factor
  (`dequantize_tensor` computes `w*scale` — quantization/__init__.py:162 — identical
  convention to upstream `scaled_dequantize`); no inversion bug. The four
  test-pinned formats (requant axis-block, checkpoint 2-D block, kernel-format
  singleton-expanded, per-tensor) plus len-1 ParameterList plus loud missing-scale /
  non-fp8-dtype errors cover the reachable shapes of this fork's fp8 pipeline. An
  int8-requant config would fail loudly (`_is_8bit_float` → ValueError), not silently.
- Residual (finding 3): the fix exists only on r5fix; the 2a2 paged path is exposed
  until the branches merge.

### Attack 5 — the `positions == arange(T)` runtime guard vs jit

**Stronger than the author demonstrated.** The r5fix tests exercise the guard only
**eagerly**; the serving forward is torchax-**jitted**. Executed under `jax.jit` on CPU
(this review):

- valid `arange(T)` → traces and runs clean, no false positive;
- decode-like `[37]*T` and chunk-continuation `arange(1,T+1)` → the `ValueError` from
  `_check_xla_ref_positions` (mla_attention.py:56-77) surfaces as
  `JaxRuntimeError: INTERNAL: CpuCallback error` — **loud even without consuming the
  result**, because `jax.debug.callback` (mla_attention.py:123) is an ordered effect
  that cannot be DCE'd.

So the guard does **not** break jit tracing, and it does fire under jit — on CPU. What
CPU cannot settle (see on-TPU list): TPU host-callback exception delivery, and the
multi-host failure mode — a raise on one host while the other 7 proceed into
collectives is structurally the Error-Interrupt/core-halt desync class this pod just
diagnosed (RESEARCH_LOG 2026-07-07). Acceptable for a validation-only mode, but it
belongs on the on-TPU checklist, and the operator should expect "loud" to possibly mean
"fatal runtime error / halt", not a tidy Python exception.

---

## Positive assurance (what this review CONFIRMS)

1. **Group formation** works against the real vLLM: one group, shared block table,
   per-layer tensors at native page sizes, `num_blocks = mem // (78·40960 + 21·8192)`
   — reproduced through `get_kv_cache_groups` → `get_kv_cache_config_from_groups` →
   `generate_scheduler_kv_cache_config` → `KVCacheManager`/`UnitaryKVCacheCoordinator`
   → `find_longest_cache_hit`, including a real 96-token prefix-cache hit between two
   requests. The scheduler machinery never sees the per-layer page sizes, so no
   misalignment is reachable from that side.
2. **The silent-merge hazard is real and correctly pinned** — `MLAAttentionSpec.merge`
   in the installed vLLM compares neither `head_size` nor page size; the discriminator
   is the correct minimal lever, and it is inert everywhere else in this vLLM
   (exhaustive grep of `model_version` consumers).
3. **Spec ↔ allocation consistency is by construction**: both sides call the same
   `get_kv_cache_shape` via `get_attention_page_size_bytes`, so dtype packing,
   head-dim alignment and transposition cannot diverge between the budget and the
   device array.
4. **Write-then-score** is bit-faithful: scatter/gather round-trip, paged == direct
   scoring bit-exact at matched blocking, decode == full-history, multi-chunk ==
   single-chunk bitwise **including cache state** at pathological chunkings, causality
   pinned per-token. `kv_len = position + 1` is the correct serving semantics for
   decode, chunked prefill, prefix-cache resume, and (by rewrite-in-place argument)
   MTP verification.
5. **The fp8 wq_b CRITICAL was a real checkpoint-verified bug** and the r5fix adapter
   covers the real formats (wq_b fp8-block; wk load-time-dequantized upstream;
   weights_proj bf16), with the correct multiplicative scale convention.
6. **The xla_ref guard survives jit** (verified here, not just eagerly) and the
   `GLM_DSA_MODE=off` byte-identity gate now also proves the paged functions never
   execute at `off` (monkeypatched-to-raise in the hash test).
7. **Both branches' DSA/MLA suites pass**: 93 (2a2, incl. `test_mla_attention`) and
   93 (r5fix), CPU-only, this review's own runs.

## What only on-TPU runs can verify

1. **Real bring-up of the spec path**: profiling-run memory, `num_blocks` at pod scale,
   `create_kv_caches` compile/shard for the 128-wide MLA-layout array on the serving
   mesh (dims 0-2 identical to the proven 640-wide cache; only the unsharded head dim
   differs — low risk, still unproven).
2. **`jax.debug.callback` raise semantics on TPU + 8-host Ray**: does the guard surface
   as a catchable error, a fatal `XlaRuntimeError`, or a cross-host desync? (Finding
   in attack 5; validation-mode-only exposure.)
3. **`GLM_DSA_MODE` raylet-env consistency** across all 8 hosts at spec-computation
   time (finding 5).
4. **Mosaic-compiled behavior** of the 2b/2c kernels (r5fix's own deferred list:
   interpret-blind specs, Gate-K on real MXU, mla.v2 `_INTERPRET` prefill shim).
5. End-to-end paged-indexer serving parity once Stage-3 wiring lands (block-table
   plumb, `valid` mask, per-layer cache index lookup — none of which exist yet).

## Cross-branch integration checklist (from findings 1, 3, 4)

- [ ] Merge 2a2 ↔ r5fix; resolve the 2 textual hunks + the TypeError/NotImplementedError
      test conflict; re-run both suites.
- [ ] Make `valid` non-optional (or internally derived) on the paged entry points.
- [ ] `assert not envs.MLA_TRANSPOSE_KV_CACHE` at the paged-cache reshape.
- [ ] Fix the `set(str)` axis-name bug in the S1 DP guard; parametrize the refusal test
      over both axis-name schemes.
- [ ] Add the rejected-draft slot-overwrite test before MTP.
- [ ] Decide MTP layer 78's indexer treatment (it exists in the checkpoint).

---
*Round-6 adversarial review; CPU-only (`JAX_PLATFORMS=cpu`), no TPU touched, nothing
committed. vLLM claims pinned to `0.1.dev1+ga30addc75`.*
