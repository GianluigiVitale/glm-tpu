# Round 6 — adversarial review of the upstream PR branch series (G1–G4)

**Reviewer:** adversarial, read-only, CPU-only (`JAX_PLATFORMS=cpu` set before every python; TPU never touched).
**Target:** worktree `~/tpu-inference-prs`, branches `pr-g1-register-glm-moe-dsa` (`e03c237f`),
`pr-g2-mla-wrapper-latent-fixes` (`8eab375b`+`1716f345`), `pr-g3-gmm-v2-fp8-v4-dequant` (`773b6ad4`),
`pr-g4-fp8-ragged-block-scales` (`1cfaf85b`), all cut from base `97938b62`; plus
`~/glm-tpu/docs/pr-descriptions/{README,pr-g1..g4}.md`.
**Method:** each branch exported standalone (`git archive`) to the scratchpad and its tests run with
`PYTHONPATH` shadowing the editable install (verified shadowing works — the editable finder is
*appended* to `sys.meta_path`, so PathFinder/PYTHONPATH wins); every "fails on base" claim re-run
against the pristine base tree; forward-port claims re-run with `git merge-tree` against **both**
the cited tip `0d59fee9` **and** the current tip `e6e47b1a` (upstream moved 6 commits today);
open-PR claims re-checked against the **live** GitHub diffs/API (`#2988`, `#2324`, `#2822`,
`#2609`, `#2896`).

**Bottom line:** the *code* on all four branches is solid — every test-count, base-failure,
merge-tree, and byte-identity claim I attacked reproduced exactly, and I found no debug residue,
no private-repo leakage, and clean yapf on every changed file. What did **not** survive attack is
part of the **duplicate-work narrative** (findings 1–2: PR #2324's live content contradicts two
"not addressed by any other open PR" statements) and two **edge-case code claims** (findings 3–4:
a premature tile_k give-up in G3 I can regress vs base under a tiny VMEM budget, and a G4 "fails
loudly" claim that a single-interior-ragged input silently defeats). None of the four blocks
submission; 1–2 require description rewrites first, 3–4 deserve small code tweaks while the
branches are still local.

---

## Findings (most severe first)

### 1. HIGH (description accuracy, G1) — pr-g1.md's duplicate-work claim about #2324 is false
- **Where:** `docs/pr-descriptions/pr-g1.md` §"Why this is not duplicating an existing PR" (also the
  commit body's implication of novelty for the registry entries), README.md row G1.
- **Claim under test:** "#2324 … does **not** register `GlmMoeDsaForCausalLM`".
- **Refuted.** The live #2324 diff (head `c2822bd7`, **last updated 2026-07-04 — i.e. before the
  2026-07-07 searches**, so this was wrong when written) contains:
  - `model_loader.py`: `"GlmMoeDsaForCausalLM"` added to **both** `_VLLM_PREFERRED_ARCHITECTURES`
    and `_PP_DISABLED_MODELS`, with a near-identical comment ("uses Dynamic Sparse Attention which
    does not yet have a flax_nnx implementation; route it through the vLLM path") — diff lines
    3653–3666 of `pr2324.diff`;
  - `envs.py`: the **identical** entry `"DISABLE_DSA_INDEXER": env_bool("TPU_DISABLE_DSA_INDEXER", …)`
    (default **False** there vs G1's default **True** at `tpu_inference/envs.py:322` — same hunk
    location, different default → textual conflict if #2324 lands first);
  - `mla_attention.py`: the **byte-identical** gate line G1 adds at
    `tpu_inference/layers/vllm/custom_ops/mla_attention.py:318`
    (`if self.indexer and self.is_sparse and not envs.DISABLE_DSA_INDEXER:`).
- **Failure scenario:** a Googler maintainer (several reviewed #2324) reads G1's "no other open PR
  registers this architecture or touches these code paths", opens #2324, and finds three of G1's
  five files carrying the same hunks. Credibility of the whole series drops; G1 also silently
  conflicts with #2324 on `envs.py` defaults.
- **What stays true:** #2324 does **not** handle GLM-5.2 IndexShare construction (no Indexer wrap
  anywhere in its diff), does **not** touch `kv_cache_manager.py`/`DeepseekV32IndexerCache`, and
  does **not** add `TPU_DISABLE_DSA_INDEXER` to the Ray allow-list. G1's unique substance
  (`vllm_model_wrapper.py:143-236` IndexShare wrap, `kv_cache_manager.py:62,633` spec skip,
  default-ON env, 15 new tests) is real.
- **Fix:** rewrite the section to state plainly that the registry/env/indexer-gate hunks are
  **shared with (ported from) #2324**, that G1 changes the env default to ON and adds the
  IndexShare + KV-spec fixes #2324 lacks, and that whichever lands first the other rebases
  trivially. The current CLAUDE.md/recon history ("port PR #2324 registry entry") shows the lineage
  was known — the description just denies it.

### 2. HIGH (description accuracy, G2) — "Bug 1 … is not addressed by #2988 **or any other open PR**" is false w.r.t. #2324
- **Where:** `docs/pr-descriptions/pr-g2.md` §Duplicate-work; README.md row G2 ("commit 1 … the
  un-duplicated substance").
- **Refuted.** #2324's `mla_attention.py` rework adds an explicit `kv_cache_dtype=auto` else-branch
  ("No KV quantization (kv_cache_dtype=auto): move to TPU as-is with scale=1.0 (identity: no
  dequantization needed)") that keeps `W_UK_T`/`W_UV` unquantized and device_puts a **replicated
  scalar 1.0** for both scales — the same latent auto-dtype NaN bug, fixed differently (scalar
  replicated scale vs G2's shaped identity scales). Evidence: `pr2324.diff` mla hunk lines 187–209.
- **Mitigations (why G2 still deserves to exist):** #2324 is a 5,887-line multi-purpose PR whose
  mla_attention context lines don't even match current main (its removed lines show
  `expand_dims(W_UK_T_scale, 0)` where main has axis 1 — it needs a large rebase), it ships the fix
  entangled with a TP-selective loader, and it has no regression test; G2 is minimal, on-tip,
  tested. The searches (`MLA`, `kv_cache_dtype`, `fp8 v4`, `block scale`) plausibly missed it
  because #2324's title matches none of them — but the README's own duplicate-work table *did* list
  #2324, so its mla_attention.py hunks should have been diffed.
- **Fix:** correct the sentence to "not addressed by #2988; **addressed differently inside draft-scale
  PR #2324** (scalar identity scale in a 5.9k-line PR that needs rebasing); this PR is the minimal
  tested vehicle" and mirror the droppable-commit logic: coordinate with #2324's author.

### 3. MEDIUM (code, G3) — the new tile_k loop gives up at the first align_to *plateau*, not the floor; regression vs base is constructible
- **Where:** `tpu_inference/kernels/megablox/gmm_v2.py:1022-1026` (branch) —
  `prev_tile_k = tile_k; … if tile_k == prev_tile_k: break`.
- **Mechanism:** `tile_k(n) = ceil(size_k/(n·128))·128` plateaus *transiently* long before the true
  floor: for `size_k=6144` the visited sequence is 6144, 3072, 2048, 1536, 1280, 1024, 896, 768,
  **768** (n=8,9) → break. 640, 512, 384, 256, 128 become unreachable, though the base loop (no
  break) reaches them.
- **Concrete repro (CPU, mocked v4 info):** `size_k=6144, size_n=192` (drives the below-floor
  tile_n branch base also executes), int8 block-128 RHS (so base and branch share the identical
  VMEM estimate — isolates the loop change), `vmem_limit=0.8 MiB`:
  - **base:** returns `tile_k=640`, est 796,672 ≤ 838,860 — **valid**;
  - **branch:** returns `tile_k=768`, est 929,792 > 838,860 — **silently over budget** → scoped-VMEM
    OOM at kernel compile where base succeeded. (At 0.6 MiB: base 384 fits; branch 768 over.)
  - Script: scratchpad `probe_g3_break.py`; also demonstrated on the branch alone at
    `size_n=4096, 0.8 MiB`.
- **Impact honestly bounded:** at the real v4 budget (~15.1 MiB) and sane shapes the search fits
  long before the first plateau (GLM dims fit at 3072); hitting the bug needs a tiny
  `vmem_limit_bytes` or enormous `tile_m·tn`. But it directly falsifies the PR-body claim
  "provably no-ops for any configuration that already fit", and the loop-restructure is exactly
  where a gmm reviewer will stare.
- **Fix:** break on the true floor (`if tile_k == num_lanes … break`, plus keep a
  compat-impossible guard), or skip non-decreasing n
  (`num_k_tiles = max(num_k_tiles+1, ceil(size_k_blocks / (size_k_blocks // num_k_tiles - 1) …))`-style),
  and add the 0.8 MiB case as a test. Secondary precision nit: "the loop body never executes when
  the estimate fits" is also falsifiable when `tile_k` is quant-block-*incompatible* while fitting
  (the OR-condition runs the body); that's arguably a fix, but the claim as written is wrong.
- **Also noted (test rigor):** "the tests fail on main before the change" is technically true but
  hollow — all 6 fail on base with `AttributeError: module … does not have the attribute
  'tpu_generation'` (mock target absent on base), not with behavioral failures. The behavioral
  failure is real (probing base directly returns `tile_k=6144` at GLM dims) but the description
  shouldn't lean on the mock artifact.

### 4. MEDIUM (code + description, G4) — the "hard assert rejects interior-ragged … instead of silently mis-scaling" claim is defeated by a single interior-ragged part
- **Where:** `tpu_inference/layers/common/linear.py:64-68` (the ceil-count assert) and the claim in
  `pr-g4.md`/commit body; test `tests/layers/common/test_fp8_ragged_block_scales.py:106-114` only
  covers the mismatched-count case (2, 6 cols vs 576).
- **Refuted by construction:** fused parts `[576, 512]` @ block 128 → per-part cols 5+4 = 9 ==
  `ceil(1088/128)` = 9 → **assert passes** (with exactly ONE ragged part anywhere, Σceil always
  equals ceil(Σ), so the count check cannot see it). Repro on the branch tree: repeat+clip then
  mis-assigns every column from feature 576 on — **1023/4352 outputs wrong, max abs err 2.57,
  no error raised**. On base this same input failed **loudly** (uniform reshape 2·128·9·120 ≠
  256·1088). So for this input class the change converts a loud failure into silent corruption —
  the opposite of the stated guarantee.
- **Impact honestly bounded:** no known checkpoint fuses `[ragged, aligned]` — GLM's ragged part
  (`kv_a_proj_with_mqa`, 576) is last, DeepSeek/Kimi are aligned; the shipped GLM path is correct
  (bit-exact test + the parity twin). This is a latent-future-model hazard plus an overclaim.
- **Fix options:** (a) weaken the comment/description to "rejects fused layouts whose total scale
  column count is inconsistent with a uniform grid; a lone interior-ragged part is undetectable at
  this call site (per-part sizes are not available here) and unsupported"; or (b) actually detect it
  by plumbing `output_sizes` into the matmul path / validating in
  `process_blockwise_fp8_linear_weights` (which *does* know the part sizes — assert there that only
  the last part is ragged, at `quantization/fp8.py:115`). (b) is small and makes the claim true.
- **Also untested:** the fused reorder path with `n_shards > 1` and a ragged part
  (`reorder_concatenated_tensor_for_sharding(weight_scale, [2,5], n_shards>1)`); both new
  process-weights tests use `n_shards=1`. Likely moot if `kv_a_proj_with_mqa` is replicated on the
  out axis (pod parity at TP=4 passed), but state it or test it.

### 5. LOW (hygiene, all branches) — trailer and note nits
- Every commit carries **two** co-author trailers: `Co-authored-by: Claude` (no email — GitHub will
  not parse a trailer without `Name <email>`) plus `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`. Keep one well-formed trailer. DCO `Signed-off-by` absent (acknowledged
  in the checklists — repo has a DCO file; must be added at push).
- G2 commit `1716f345`'s message embeds "DROP THIS COMMIT if #2988 merges first…" — right plan,
  wrong place: if the commit *does* merge, the instruction becomes permanent upstream history. Move
  it to the PR description / a review comment; keep the commit message to the fix + the #2988
  attribution note.
- G2 `mla_attention.py:149`: `from torchax.ops.mappings import t2j_dtype` is function-local inside
  the `_maybe_quantize` closure; the repo imports it at module top elsewhere
  (`runner/kv_cache_manager.py:21`). Hoist it (isort will want it there anyway).
- G2 description mechanism nit: "the 'quantized' weights overflow on the f32 cast" — empirically
  they **saturate to f32max (3.4e38)** with a numpy overflow RuntimeWarning; NaN then arises in the
  forward einsum (accumulation → ±inf) × the 0.0 scale. Scale collapse to exactly 0.0 confirmed;
  substance right, wording slightly off if a reviewer probes.
- G3 `_FP8_DTYPES = (e4m3fn, e5m2)` only: an FP8 RHS with `has_scale=False`, or e4m3/e4m3fnuz,
  still routes to the direct matmul on v4 → Mosaic E2001. Pre-existing wall (the gate doesn't
  widen it); a one-line comment would preempt the review question.
- Stale tip pin: all four descriptions cite tip `0d59fee9`; upstream is now `e6e47b1a` (+6 commits,
  same day). All forward-port statements **remain true at the new tip** (re-verified below), but
  re-pin before submission.

---

## CONFIRMED (positive assurance — everything below was attacked and held)

**Branch/test claims (all reproduced bit-for-bit):**
- G2: `8 passed` (6 pre-existing + 2 new) with the 4-device flag; both new tests **fail on the
  pristine base** — the 4-device test with the exact quoted `IndivisibleError` ("array axis 0 is
  partitioned 4 times, but the dimension size is 1 (full shape: (1, 8, 64))"). Direct repro:
  `quantize_tensor(None, ones) → scale.max() == 0.0` exactly as claimed. Identity-scale shapes
  match `quantize_tensor`'s ([H,1,kv_lora] / [1,H,v]); the quantized-KV branch call is argument-identical;
  the forward multiplies scales unconditionally (flash_attn_mla.py:156,209) so 1.0 scales are
  provably neutral.
- G3: `6 passed`; GLM-dims search returns `TileSizes(tile_m=64, tile_k=3072, tile_n=256)` under the
  15,099,494-byte budget (est ≈9.6 MiB) exactly as the description claims; base search returns
  `tile_k=6144` (est ≈19–20 MiB → the claimed ~20 MiB) when probed directly. The at-limit
  (`size_n == 2·MXU`) dead-branch bug is real and generation-agnostic (reproduced with int8 too).
- G4: `5 passed`; on base exactly `4 failed, 1 passed` with the aligned case passing — matching the
  description word-for-word. The ragged 576@128 expansion is bit-exact vs the explicit per-block
  reference; base's uniform reshape does fail outright at 576@5 (115·5 ≠ 576).
- G1: `56 passed, 1 failed` + `19 passed` exactly as claimed; the 1 failure
  (`test_getattr_without_cache`) fails **identically on the pristine base** under
  `JAX_PLATFORMS=cpu` (16 pass/1 fail there) — genuinely pre-existing and env-sensitive. On base:
  the 4 registration tests fail, the KV-spec test reproduces the **exact** engine-boot
  `AttributeError: 'DeepseekV32IndexerCache' object has no attribute
  'kv_sharing_target_layer_name'`, and the patch-test module ImportErrors. The KV-spec test
  constructs the **real** vLLM `DeepseekV32IndexerCache` as claimed.

**G1 technical spot-checks:**
- The hard-coded 78-entry schedule in the test == the real `indexer_types` in
  `~/glm-tpu/configs/glm-5.2-fp8-config.json`, and the freq/offset formula reproduces it for all 78
  layers (verified programmatically). MTP layer 78 → formula says `full`, consistent with the
  checkpoint shipping indexer weights there.
- The wrapped constructor's signature matches the pinned vLLM's `Indexer` call site exactly
  (8 positionals + `is_inplace_rope` kw).
- `SparseAttnIndexer.forward_native` in the pinned vLLM **does** raise `NotImplementedError` on
  non-CUDA/ROCm/XPU — the default-ON justification is accurate.
- `DISABLE_WEIGHT_REQUANTIZATION` exists in base `envs.py:364` — the allow-list addition is
  self-contained (though a reviewer may ask why it rides in G1; the answer — GLM FP8 multi-host
  needs it propagated — belongs in the PR body).
- The env attr/var name mismatch (`DISABLE_DSA_INDEXER` ← `TPU_DISABLE_DSA_INDEXER`) has no other
  precedent in envs.py, but is **the identical hunk #2324 carries** — the lineage argument is
  stronger than the description states (see finding 1).

**Forward-port / duplicate-work claims:**
- `git merge-tree` re-run against cited tip `0d59fee9` **and** current tip `e6e47b1a`: G1, G2, G4 —
  **zero** conflict markers at both tips; G3 — exactly **one** conflict hunk at both tips, in
  `should_dequantize_before_matmul`, matching the described guard→assert upstream refactor; the
  described resolution (insert the v4-FP8 `return True` after the assert) is coherent, and the tip
  drift diff confirms the dequant copy is now `lhs_cfgs.dtype` (bf16), making the branch's f32-based
  `2·tk·tn·4` term conservative exactly as stated.
- Byte-identity claims verified: G2's two files and G4's three files are byte-identical between
  base and tip.
- #2988 (open, non-draft, approved by gxd3, `ready` label, updated 06-25): its mla_attention change
  is the **same `P(None, ATTN_HEAD)` W_UV_scale fix** — *substantively identical* as claimed (not
  textually: `wuv_scale_sharding` vs `uv_scale_sharding`, different comment — no byte-equivalence is
  claimed in the docs, and none exists); it **ships no tests**, so offering the 4-device test stands.
  It additionally touches `attention_interface.py` (MLP_TENSOR→ATTN_DATA) which G2 rightly does not
  carry. Given the approval + ready label, plan on the drop-commit-2 path being exercised.
- #2822 (open, 06-26), #2609 (open, 05-13), #2896 (**draft**, Tokamax, 06-23) all exist with the
  scopes the G3/G4 descriptions attribute to them; #2324 does **not** touch G4's files
  (`layers/common/linear.py`, `layers/common/quantization/fp8.py`) — G4's no-overlap claim holds.

**Hygiene / self-containedness:**
- Each branch exported standalone runs its tests green with only its own diff — no cross-group
  symbol leakage (the only shared file, `mla_attention.py` in G1∩G2, has disjoint hunks; the G1+G2
  stack cross-merges with zero conflicts).
- No debug prints/breakpoints/TODO-XXX residue; no `glm-tpu`/`moe-tpu`/`driftbench`/`RESEARCH_LOG`/
  username references anywhere in the diffs; no `GLM_*` env leakage (`GLM_DIMS` is a test constant).
- License headers on all new files match repo convention (`Copyright 2026 Google LLC` Apache-2.0
  block, same as sibling tests). Em-dashes in comments have ample precedent in base source.
- `yapf --diff` (v0.43.0 == the repo's pinned pre-commit rev) is clean on **every** changed file of
  all four branches. isort/ruff not runnable here — the checklists' `pre-commit run --all-files`
  before push stands.
- Worktree clean, HEAD detached at base, no branch pushed anywhere (`origin` has none of the four) —
  consistent with the owner-submits policy. No AGENTS.md exists in the repo at base or tip
  (CONTRIBUTING.md carries the unit-test/CI expectations); the disclosure line in each PR body is
  present.

---

## What only on-TPU runs can verify (out of scope here, correctly gated in the checklists)

1. The actual Mosaic `E2001` rejection of an FP8 RHS on v4 and the dequantize-in-VMEM path
   compiling + matching the GMM reference at GLM dims (G3 gate re-run + microbench) — the CPU tests
   mock `tpu_generation`/`get_tpu_info` entirely.
2. The claimed 1-chip GLM parity numbers (final-hidden 0.619 < 0.722, logits 0.910 < 0.958, top-1
   0.906 > 0.875; G4 twin top-1 1.000) on the **re-cut** branches — all currently attributed to the
   dev branch, not these branches.
3. Engine-level sufficiency of the `DeepseekV32IndexerCache` spec-skip (that no other vLLM
   component demands a spec/slot for that layer at real engine boot, and DSV3.2-family configs boot
   too).
4. G2's `P(None, ATTN_HEAD)` under a real head-TP>1 pod mesh at load (CPU 4-device is a faithful
   but not identical proxy), and the interaction with the G3 VMEM headroom change (0.8 vs 0.9
   capacity fraction) flagged in pr-g3.md's Risk.
5. G4 fused-ragged behavior at `n_shards > 1` (or confirmation the ragged layer is never
   out-axis-sharded).
6. Whole-model default-path correctness with G1's default-ON env on the tip's newer vLLM pin
   (the local env cannot import the tip).

## Recommended pre-submission actions (in order)

1. Rewrite pr-g1.md's and pr-g2.md's duplicate-work sections per findings 1–2 (state the #2324
   overlaps honestly; they strengthen, not weaken, the case for these minimal PRs).
2. Fix the G3 plateau-break (finding 3) and add the small-budget regression test; soften or fix the
   G4 interior-ragged claim (finding 4, option (b) preferred).
3. Re-pin the descriptions to the current upstream tip; normalize the co-author trailer; move the
   "DROP THIS COMMIT" note out of the G2 commit message; hoist the `t2j_dtype` import.
4. Keep watching #2988 (approved + ready): expect to drop G2 commit 2 and offer its test.
