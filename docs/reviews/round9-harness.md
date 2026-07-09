# Round 9 — adversarial review: unreviewed main-thread hand edits (harness + kernel-fix quality)

- **Targets** (the last sessions' unreviewed hand edits):
  1. glm-tpu `50f385d` — `parity/glm_engine_parity.py` Stage-2 indexer k-cache slots
     (unpadded mini width, two-step reset, `[1,1,1]` decode marking).
  2. glm-tpu **untracked** `bench/det_probe.py` — same-boot determinism probe.
  3. glm-tpu `170d2b8` — `scripts/kernel_probe/probe_2b_parity.py` `_make_reference_case`
     'highest' oracle wrapper (+ `parity/glm_indexer_reference.py` internal pinning).
  4. fork `886eaceb` (glm-5.2-v4-next @ `cda8a707`, clean tree) —
     `tpu_inference/kernels/dsa/indexer_kernel.py` three Mosaic lowering fixes
     (silicon-validated; reviewed here for CODE QUALITY only, per the task brief).
- **Method:** read-only source cross-examination against the fork's runner/wrapper
  (`persistent_batch_manager.py`, `mla_attention.py`, `glm_dsa_indexer.py`,
  `kv_cache_manager.py`, `kv_cache.py`, `mla/v2/kernel.py`) and the installed vLLM
  (`~/vllm-build/vllm/model_executor/models/deepseek_v2.py`); CPU suites executed with
  `JAX_PLATFORMS=cpu` set before python. **TPU never touched** (a pod run is live);
  the one probe artifact my interpret run wrote was deleted afterwards.
- **Stance:** adversarial — every hand-edit claim was attacked before acceptance;
  refutation attempts are logged per finding.

## Verdict summary

| Target | Verdict |
|---|---|
| (1) engine-parity indexer slots + `[1,1,1]` | **SOUND** — matches the runner's real semantics byte-for-byte; mapping cannot shift; unpadded width is *required*, not merely allowed |
| (2) det_probe determinism verdict | **SOUND for what it measures**, but the verdict's *scope* is narrower than the D0 divergence it is meant to discriminate (Findings 4–6); file is **uncommitted** (Finding 7) |
| (3) probe_2b 'highest' oracle wrapper | **FULLY COVERED — no escape found** (attack log in §3) |
| (4) indexer_kernel Mosaic fixes | **Correct and clean**; lane-mask cost negligible at decode, material only under `GLM_DSA_SCORER=pallas` prefill scoring (Finding 1); module docstring now stale/contradictory on VMEM (Finding 2) |
| CPU suites | **ALL GREEN** — fork DSA kernels 135/135; parity reference a–d ALL PASS; probe_2b `--interpret` gates A/B/C PASS; bench 58/58 (§5) |

Findings: 2 MEDIUM (both documentation/verdict-scope, no code defect), 5 LOW, plus
positive assurance. **No correctness bug found in any of the four targets.**

---

## 1. `parity/glm_engine_parity.py` @ 50f385d — Stage-2 indexer cache slots

### 1.1 Attack: does `[1,1,1]` match the runner's real decode semantics? — YES, exactly

The runner's constructor is `persistent_batch_manager._reorder_batch`
(fork `runner/persistent_batch_manager.py:52-84`): the 3-vector is
`[num_decode, num_decode, num_reqs]`, and when
`total_num_scheduled_tokens == num_reqs` (every request scheduled exactly 1 token)
it emits `[num_decode, num_decode, num_reqs]` = **`[1, 1, 1]`** for a 1-request
decode step. For a single multi-token request the two-pointer path computes
`num_decode = 0` → **`[0, 0, 1]`** — which is precisely the harness's `else` arm.
So `np.array([1, 1, 1] if n_new == 1 else [0, 0, 1])` reproduces the real runner
for BOTH chunk shapes, not just the decode one.

Consumer check: `glm_dsa_is_pure_decode` (`mla_attention.py:174-201`) tests
`dist[0] == dist[2] and dist[2] > 0` → `[1,1,1]` → True → `_sparse_decode_branch`
(the 2c kernel), `[0,0,1]` → False → prefill dispatch (dense fallback at mini
ctx ≤ topk — matching the commit message's "trivially dense" prefill note).
The wrapper's own docstring (lines 183-194) independently confirms a 1-token
final chunk taking the decode branch is safe (keys written every step, latent
history written by prior chunks, `kv_len = pos + 1`).

The mla.v2 dense kernel sees the same vector; its classification
(`sequences[0:i]` decode-only) is also correct for a 1-token continuation, and the
DP caveat does not bite: the guard in `_glm_dsa_pallas_decode_attention`
(`mla_attention.py:960-973`) checks mesh **axis sizes** (product over
ATTN_DATA/BATCH axes), not the `enable_dp_attention` config flag — the harness
mesh is `(1,1,1,1,N,1)` so `_dp_size == 1` and the guard passes. Refutation
attempt (guard-on-flag would break the harness): **failed — code checks sizes.**

- **Residual (Finding 3, LOW/INFO):** the `[1,1,1]` arm is NOT gated on
  `GLM_DSA_MODE` — a Stage-1 dense rerun with `--split-at T-1` now routes the mla.v2
  kernel through its decode-only classification instead of mixed. That is *more*
  faithful to the real runner, but it changes the executed kernel branch vs any
  previously recorded dense two-step log at `split == T-1`; bit-identity with such
  old logs is not guaranteed. Default `split = T//2` (n_new > 1) is unaffected.

### 1.2 Attack: unpadded `idx_shape` vs what the write/score paths expect — REQUIRED, not just allowed

The consumers all derive the head width from the cache array itself:

- `_glm_dsa_indexer_cache` (`mla_attention.py:864-891`): `head_dim = cache.shape[-1]`,
  then `reshape(shape[0], -1, head_dim)`.
- `write_indexer_keys` (`glm_dsa_indexer.py:734-761`): scatters `keys_TD [T, IDX_HD]`
  into `flat [num_pages*page_size, head_dim]` — **a 128-padded mini cache (head_dim
  128 vs keys 64) would raise a scatter shape error**, i.e. the padded alternative
  is not merely different, it is impossible. Refutation attempt (padded cache would
  also work): **failed.**
- `paged_indexer_scores` / `indexer_scores_pallas` take `page_size`, `D` from
  `k_cache.shape`; `sm_scale = D**-0.5` from q. Self-consistent at D=64.

At real dims the runner's Stage-2 spec is `head_size = align_to(index_head_dim, 128)`
(`kv_cache_manager._dsa_indexer_spec_for_mode`, docstring "GLM-5.2: 128, zero padding
waste") — identical to the harness's unpadded 128. The divergence exists only at mini
dims, where the harness's choice is the only self-consistent one. (Corollary, noted
for the record: running the REAL runner at mini dims with GLM_DSA_MODE on would crash
in `write_indexer_keys` on the 64-vs-128 scatter — a latent fork mini-dims
incompatibility, not a harness bug and not reachable in production dims.)

Packed-layout claim verified: `get_kv_cache_shape` (mla/v2/kernel.py:119-142) is
`[pages, align(S,pk)//pk, pk, align(D,128)]` with the packing dim holding adjacent
tokens, so `(num_pages, PAGE//2, 2, IDX_HD)` reshaped to `[pages, PAGE, IDX_HD]` is
row-major identical — the harness comment is accurate. `PAGE=128 % pk=2 == 0` holds.

### 1.3 Attack: could the added slots shift the MLA cache indices? — NO

- Build order: the n_layers MLA caches are created (indices 0..n_layers-1) and
  `mapping[inner.layer_name] = i` fixed BEFORE the indexer append loop; appended
  slots use `mapping[...] = len(kv_caches) - 1` at append time. No aliasing:
  MLA keys are `...attn` layer names, indexer keys are `...indexer.k_cache` prefixes.
- Two-step reset: `_n_idx = len(kv_caches) - n_layers`; the rebuild is
  `[n_layers MLA] + [_n_idx indexer]` — same partition, and since every indexer slot
  shares ONE shape, intra-group order is immaterial; every mapped index points at a
  correctly-shaped zeroed cache. Slice assignment (`kv_caches[:] =`) preserves the
  list identity the `_forward` closure and the wrapper's in-place item writes rely on.
- `GLM_DSA_MODE=off`: `_n_idx == 0` and the `np.zeros(idx_shape=None)` expression sits
  inside a `range(0)` comprehension body — never evaluated. Refutation attempt
  (None-shape crash on dense two-step): **failed.**
- Object identity for the mapping key: vLLM's `DeepseekV2MLAAttention.self.indexer`
  is passed as `mla_modules.indexer` → wrapper `self.indexer` → `VllmMLAAttention`'s
  `indexer=` — one object; `self_attn.indexer.k_cache.prefix` (harness) ==
  `self.indexer.k_cache.prefix` (wrapper lookup). And the fork's
  `_maybe_patch_for_glm_moe_dsa` (vllm_model_wrapper.py:173-191) forces
  `indexer=None` exactly on `shared` layers, so the harness's skip-None loop
  registers slots for FULL layers only — mirroring the real runner, which creates
  KVCacheSpecs only for existing `DeepseekV32IndexerCache` modules.

### 1.4 Residual caveats (informational)

- **Finding 8 (LOW):** the harness "enables GLM_DSA_MODE runs" at mini dims, but two
  mini-dims combos remain silicon-unvalidated: (a) `GLM_DSA_SCORER=pallas` — the 2b
  kernel's q/cache blocks then carry lane dim D=64 (< 128); the compiled-mode assert
  only checks `page_size % 128`, so Mosaic acceptance of D=64 is unprobed (default
  scorer 'xla' unaffected); (b) the 2c kernel at mini latent width 384 (vs the
  GATE-B-validated 640). Neither affects the CPU/interpret gates; worth remembering
  before reading a mini on-metal D0 as kernel validation.
- **Nit:** `import os as _os` is redundant — module-level `os` is already imported
  and used three lines later (`GLM_IDENTITY_BT`). Style only.

---

## 2. `bench/det_probe.py` — determinism verdict correctness

### 2.1 Refuted attacks (positive assurance)

- **Sampling params vs run_bench's `_sp`:** byte-equivalent for the greedy protocol.
  `SamplingParams(temperature=0.0, max_tokens=…, stop_token_ids=engine.EOS_IDS)` —
  `top_p` defaults to 1.0 (run_bench passes 1.0 explicitly), `ignore_eos` defaults
  False (run_bench passes False), seed omitted (run_bench also omits it on TPU per
  the round-6 F1 probe). Same `EOS_IDS` constant from the same module. **No drift.**
- **Tokenizer path:** `tok.apply_chat_template([user msg], add_generation_prompt=True,
  tokenize=True, return_dict=False)` is the exact `_chat_prompt_ids` call for the
  no-system-prompt greedy protocol (`chat_template=None` ⇒ tokenizer's own — the same
  engine tokenizer run_bench uses); `TokensPrompt(prompt_token_ids=ids)` ≡
  `{"prompt_token_ids": ids}`. Items come from the same `B.load_items(REGISTRY["gsm8k"])`
  builder, so prompts are byte-identical to the bench items. **No drift.**
- **Prefix-caching contamination (the sharpest attack):** if APC were on, pass 2 would
  reuse pass 1's KV pages and the "identical" verdict would be circular (a
  nondeterministic prefill would still replay identically within one boot).
  **Refuted:** `engine.build_llm` hardcodes `enable_prefix_caching=False`, and
  det_probe builds through it — pass 2 recomputes prefill from scratch.
- **Greedy vs global RNG:** temperature 0.0 ⇒ argmax; the TPU sampler's advancing
  global key chain cannot influence outputs. Text comparison is deterministic
  (detokenization is a pure function; stop-token stripping identical in both passes).

### 2.2 Findings — verdict SCOPE, not verdict math

- **Finding 4 (MEDIUM, verdict scope): the probe removes the batching variable the
  diverging runs had.** `max_seqs=1` + one prompt per `llm.generate` call is
  single-stream; run_bench drives `generate_batch` at `max_seqs=8`, where batch
  composition/scheduling changes bucketing and summation orders. If runs 53 vs 55
  were batched, a "SAME-BOOT identical 4/4" here does NOT acquit within-boot
  batch-order nondeterminism — it only discriminates *single-stream* runtime
  nondeterminism from boot state. The `[DET]` verdict line should carry this scope
  (e.g. "single-stream"), or a second batched phase (submit all 4 in one
  `llm.generate` twice) should be added — that phase is 4 lines and would make the
  discriminator complete.
- **Finding 5 (LOW): 512-token horizon.** Default `--max-new 512` vs the bench's
  2048; a divergence first occurring after token 512 is invisible and would be
  reported as `identical=True`. The flag exists — the run just has to use it; the
  verdict line could print the horizon (`identical@512tok`) to prevent over-reading.
- **Finding 6 (LOW): engine-config and attention-path provenance not printed.**
  The probe builds a DIFFERENT engine than the bench (`max_batched_tokens=512` vs
  4096, `gmu=0.90` vs 0.94, `num_gpu_blocks=64`, `max_len=4096`) — different compiled
  buckets, so its verdict certifies these executables, not the bench's. That is a
  legitimate cheap-probe choice, but nothing in the output records it, and
  `engine.attention_path()` (built for exactly this) is never printed — a log reread
  cannot tell dense-mla from dsa-sparse probes. One print line fixes both.
- **Finding 7 (LOW, process): `bench/det_probe.py` is UNTRACKED** (as are
  `merge_runs.py`, `report_passkey.py`, their tests, and modified
  `benchmarks.py`/`run_bench.py`). The repo's own rule is that `git push` is the only
  durable backup on this preemptible VM; the instrument behind a determinism verdict
  should be committed so the verdict is reproducible. (Result rows are print-only,
  not in results.db — defensible for a diagnostic probe, but the script itself must
  survive the VM.)

---

## 3. `probe_2b_parity.py` 'highest' oracle wrapper @ 170d2b8

**Claim under attack:** `_make_reference_case` wraps `_make_reference_case_inner`
in `jax.default_matmul_precision("highest")` and this covers the whole oracle.

### 3.1 Coverage audit — every matmul-bearing expression in `_inner`

| Site | In context? | Notes |
|---|---|---|
| `ref.indexer_scores(...)` | YES (and self-pinned) | since 170d2b8 the reference pins HIGHEST *internally* (`glm_indexer_reference.py:176-179`) — double coverage |
| `q_resid @ wq_b.T`, `h @ wk.T`, `h @ weights_proj.T` | YES | eager dispatch inside the `with` — precision baked at dispatch |
| LayerNorm mean/var, `rope_cos_sin`, `apply_rope` | YES | element-wise VPU fp32 — precision-insensitive anyway |
| `_build_paged_cache` | YES (called at `_inner:198`) | numpy RandomState + `jnp.asarray` — no matmul |
| return-expression `jnp.asarray(np.stack(...))` | YES | conversions only |

### 3.2 Escape attempts — all failed

1. **Jit-cache escape** (a cached trace from outside the context reused inside):
   nothing in the oracle path is `@jax.jit`-ted (only the unused `dsa_topk_indexer`
   wrapper is); and `default_matmul_precision` participates in JAX's trace context,
   so even a jitted oracle would re-trace under the pinned value.
2. **Async-execution escape** (computation executing after `__exit__`): precision is
   a property of the lowered op, fixed at trace/dispatch time — deferred execution
   runs the already-pinned computation.
3. **Post-context consumers**: `ref.topk_indices` (gate A lines 255, 284) is
   `lax.top_k` over precomputed rows + `causal_mask_scores` (`jnp.where`) — zero
   matmuls (verified by reading `glm_indexer_reference.py:241-267`);
   `_tie_aware_mismatches` is numpy. **No matmul executes on oracle data outside a
   pinned region.**
4. **DUT-side leak** (does A1 accidentally need the context?): `indexer_scores_xla`
   pins its head-sum einsum `precision=HIGHEST` and its q·k dot HIGHEST-for-fp32
   internally (`indexer_kernel.py:183-189, 247-258`) — that is why on-metal A1 hit
   2.4e-7 without any caller context. bf16 uses DEFAULT deliberately (Mosaic
   precedent). Correct as designed.
5. **Enforcement**: `test_d_precision_pinned` lowers `indexer_scores` under a HOSTILE
   `default_matmul_precision("bfloat16")` context and asserts all ≥5 `dot_general`s
   in the HLO carry HIGHEST — re-executed this review, PASS (§5).

**Verdict: fully covered; no escape.** One oracle-robustness nit (LOW): in
`_tie_aware_mismatches`, a hypothetical bogus `-1` index inside the first `k_eff`
slots would numpy-wrap to `row[-1]` and could be misclassified as an in-band tie
instead of erroring; filtering `i < 0` into `out` would make the oracle
fail-closed. Unreachable while `hierarchical_topk`'s `n_valid` contract holds
(`k_eff = min(topk, L) == n_valid`, so `-1` fill starts exactly at `k_eff`).

---

## 4. `indexer_kernel.py` Mosaic fixes @ 886eaceb — code-quality review

(Silicon-validated: GATE 2a ACCEPT, A1 2.4e-7, A2 exact, A3 0 out-of-band, B/C PASS.
This section reviews the CODE, per the brief; **no changes made**.)

### 4.1 Lane-mask select cost — quantified

Per (r, b) grid step the select is iota `[H, R]` + compare + select + lane-reduce:
~4 VPU passes over `ceil(H/8) × ceil(R/128)` vregs, recomputed every step
(w_col depends only on r, so it is R×nb-fold redundant overall → O(H·R²·nb) wasted
work across the grid). Against the step's MXU dot `[H,D]×[D,P]` (32·128·128):

- **Decode** (`R = num_reqs ≤ max_num_seqs`, e.g. 256): `[32, 256]` ≈ 8 vregs/op —
  **noise** (≲1% of step cost). The w block is 32·R·4 B = 32 KiB VMEM. Not worth
  touching for Stage 2's decode-first story.
- **Prefill scoring under `GLM_DSA_SCORER=pallas`** (the wrapper calls the 2b kernel
  with **per-token rows**: `R = num_tokens`, up to `max_num_batched_tokens` — 4096 in
  the bench config): `[32, 4096]` ≈ 128 vregs × ~4 ops per step is comparable to the
  MXU dot's cycle count → **up to ~2× step-time inflation**, plus a 512 KiB resident
  w block. **Finding 1 (LOW today / MEDIUM if the pallas scorer is promoted to
  default for prefill): flagged, with exact alternatives below.** Default scorer is
  'xla', so nothing currently shipping pays this.

Cheapest exact alternatives (in ascending layout risk — candidates for a future
sanctioned change, NOT applied):

1. **Per-request hoist via scratch:** `pl.when(b == 0)` computes w_col once into a
   `[8, 128]`-padded VMEM scratch (`pltpu` scratch persists across grid steps; the
   grid is r-outer/b-inner, so row r's sweep reuses it). Reduces the select to once
   per request row — O(H·R) total instead of O(H·R²·nb). No new block layouts;
   lowest risk.
2. **w as `[R, 1, H]` with a `(1, 1, H)` full-tail block** at index map `(r, 0, 0)` —
   the same last-two-dims-full rule that legalized the o_ref fix (REJECT #2). Zero
   select cost; needs a 2a-style Mosaic acceptance probe.
3. **`[R_pad, H]` with an `(8, H)` block** at `(r // 8, 0)` + in-kernel **sublane**
   dynamic slice `pl.ds(r % 8, 1)` — sublane ds is the op class the o_ref store
   already uses (E2003 bans only the LANE dim); select cost O(8·H), R-independent.
4. **Scalar-prefetch w into SMEM** + H-step `fori_loop` of scalar×row FMAs —
   eliminates the w block entirely; most invasive restructuring.

### 4.2 Finding 2 (MEDIUM, doc honesty): module docstring now contradicts the shipped kernel

The 886eaceb diff updated the in-body comments but left the module docstring stale
in three places that now assert falsehoods:

- The **VMEM budget table** (lines 131-143) still lists `w [1, H] fp32 … 128 B` and
  `out [1, P] fp32 (x2) … 1 KiB`. Shipped: w is `[H, R]` (4·H·R B — R-dependent) and
  the output block is `[1, max_blocks, P]` fp32.
- The claim **"Context-independent (ctx grows the GRID, never a tile) … the DSV4
  ctx-scaling VMEM wall cannot occur here"** (lines 145-149) is now FALSE at the
  margin: the output block is nb·P·4 B — 512 KiB at 128K ctx (page 128), **4 MiB at
  the 1M-ctx target (×2 if Pallas double-buffers the output window)** — the wall
  class the sentence rules out is back, merely far away. There is a
  `vmem_limit_bytes` knob but no guard; a `max_blocks`-budget assert next to the
  existing `page_size % 128` assert would make the failure readable instead of a
  Mosaic OOM.
- The **"Remaining for real-TPU validation"** list (lines 151-160) still names the
  `(1, H)` w tile as MATMUL LHS ("highest-risk … fallback: broadcast-multiply") and
  the `[1, P]` output tile — both are resolved (the fallback IS now the shipped
  design, and the output tile no longer exists). A reader doing the next freeze
  review would chase ghosts.

Since the kernels are frozen post-validation, the docstring rewrite is a
comment-only touch of a frozen file — it should ride the next sanctioned
freeze-break (or an explicitly comment-only exemption), not silently.

### 4.3 Finding 9 (LOW): the zero-sign invariant paragraph should acknowledge the rewrite

Docstring lines 111-115 warn that "any future rewrite of the head-sum must preserve
the zero signs or tie ORDER … can differ between paths". 886eaceb IS such a rewrite,
and the analysis is favorable but nowhere recorded:

- Sum-of-products zero sign is order-independent in IEEE (+0 unless ALL terms are
  −0), and both paths form the same products `w_h · s_h` — preserved vs the XLA twin.
- The one exception: the mask-select `where(lane==r, w, 0.0)` + lane-sum maps a gate
  that is exactly `−0.0` to `+0.0` (−0 + 0 = +0), flipping the resulting relu-zero
  products' signs vs the twin. Reachable only if `weights_proj` emits an exact
  negative zero (measure-zero; affects tie ORDER only, never the selected set —
  gates compare sets). One sentence in the zero-sign paragraph closes the loop.

### 4.4 What is RIGHT about the fix (positive assurance)

- The o_ref restructure writes every row exactly once (both `pl.when` arms store at
  `ds(b, 1)`; the grid covers all b; the block's index map depends on r only and the
  b-inner iteration order means the block stays resident through the sweep and
  flushes on r change). No uninitialized rows, no double writes.
- `w_map = (0, 0)` full-array residency: one DMA, no per-step refetch.
- The interpret and compiled paths share one kernel body — the CPU suites exercise
  the same select/store code silicon runs (no interpret-only fork of the math).
- fp32 discipline preserved: relu before the signed head-sum, `-inf` fill and the
  `kv_abs < seq_len` mask untouched, `precision` still HIGHEST-for-fp32 on the q·k
  dot. Only the head-sum summation ORDER changed — covered by the tie-aware set
  gates and measured at 2.4e-7 on metal (128× headroom under the 5e-4 bar).
- Host-side `.T` and the final `.reshape(R, nb*P)` keep the public contract
  byte-compatible for both callers (`mla_attention.py`, `dsa_topk_indexer`) — no
  call-site changed, and `_check_scoring_shapes` still validates the UNtransposed
  `[R, H]` w.

---

## 5. Suite executions (this review, `JAX_PLATFORMS=cpu` exported before python)

| Suite | Result |
|---|---|
| fork `tests/kernels/test_dsa_indexer_kernel.py` + `test_dsa_sparse_mla.py` + `mla_v2_pack_new_kv_oob_test.py` | **135 passed** (129 s) |
| `parity/test_indexer_reference.py` (script mode, its designed entry point) | **ALL PASS** — (a) ≤6.3e-6 both layouts, cross-layout 2.685 distinguishable; (b) sets == torch.topk k=8/16/64 + actual `GlmMoeDsaIndexer.forward`; (c) deterministic, 62 tie rows; (d) all 5 dot_generals HIGHEST under hostile bf16 context |
| `probe_2b_parity.py --interpret` | **A/B/C PASS** — B fp32 8.345e-7 ≤ 2e-6, bf16 1.953e-3 ≤ 2.5e-3; C byte-identical at kv_len 511/512/513 × q_end 10/11 (artifact deleted post-run to keep this review read-only) |
| `bench/` pytest | **58 passed** |

- **Finding 10 (LOW, harness hygiene):** `pytest parity/test_indexer_reference.py`
  reports **1 ERROR** — `test_b_topk_sets(score_pairs)` takes its predecessor's
  return value as a parameter, which pytest resolves as a missing fixture (plus a
  `PytestReturnNotNoneWarning` on `test_a`). The file is a script-gated suite
  (`main()` chains a→d; exit-code contract), so the intended invocation passes — but
  the `test_*` naming invites a pytest run that shows red for a structural, not
  mathematical, reason. Either rename to `check_*` with `test_` pytest shims, or
  convert `score_pairs` to a real fixture. As-is, a future "run everything under
  pytest" CI gate would false-alarm.

## 6. Consolidated findings (severity-ordered)

| # | Sev | Where | What |
|---|---|---|---|
| 1 | LOW→MED (conditional) | `indexer_kernel._indexer_scores_kernel` | O(H·R) lane-mask select per grid step (O(H·R²·nb) total redundancy); negligible at decode R≤256, up to ~2× step cost + 512 KiB VMEM at `GLM_DSA_SCORER=pallas` prefill R=4096. Four exact alternatives listed (§4.1); hoist-via-scratch is the low-risk one |
| 2 | MED (docs) | `indexer_kernel.py` module docstring | VMEM table, "context-independent / no VMEM wall" claim, and the "remaining risks" list all predate 886eaceb and now misdescribe the shipped kernel; output block is ctx-proportional (4 MiB @1M ctx) with no guard |
| 3 | LOW/INFO | `glm_engine_parity._step_am` | `[1,1,1]` arm not gated on GLM_DSA_MODE — dense `--split-at T-1` reruns change mla.v2 branch vs old logs (more runner-faithful, but bit-comparisons to pre-50f385d dense logs at that split are void) |
| 4 | MED (verdict scope) | `det_probe.py` | single-stream (`max_seqs=1`, sequential) probe cannot acquit within-boot BATCH nondeterminism; verdict line should say so, or add a batched phase |
| 5 | LOW | `det_probe.py` | 512-token default horizon; late divergence invisible; print the horizon in the verdict |
| 6 | LOW | `det_probe.py` | no `attention_path()`/engine-config provenance in output; probe engine ≠ bench engine buckets |
| 7 | LOW (process) | glm-tpu working tree | `det_probe.py` + 5 sibling files uncommitted/untracked — violates the repo's durable-backup rule; the determinism instrument itself is unreproducible if the VM is preempted |
| 8 | LOW/INFO | `glm_engine_parity.py` | mini-dims + `GLM_DSA_SCORER=pallas` (D=64 lane) and mini 2c width 384 are silicon-unvalidated combos the new slots make reachable |
| 9 | LOW (docs) | `indexer_kernel.py` | zero-sign invariant paragraph predates the head-sum rewrite; the −0.0-gate → +0.0 select edge (tie order only) is unrecorded |
| 10 | LOW | `parity/test_indexer_reference.py` | pytest collection ERROR on `test_b` (chained-arg design vs `test_*` naming); script mode is green |

**Positive assurance:** the `[1,1,1]`/`[0,0,1]` marking is byte-for-byte the
runner's `_reorder_batch` semantics in both arms; the unpadded mini indexer width is
the only shape the fork's own write/score paths accept and equals the runner's spec
at real dims; the MLA cache index mapping cannot shift under the append or the
two-step reset; the probe oracle's 'highest' pinning has no escape (5 attack vectors
tried, all failed, HLO-asserted); the Mosaic fixes are correct, contract-preserving,
and cheap at the decode geometry Stage 2 targets. All four CPU suites green at
glm-tpu `50f385d` / fork `cda8a707` (clean tree). No correctness defect found.
