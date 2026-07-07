# PR-G3 — [Kernel] gmm_v2: FP8-RHS dequantize-in-VMEM on TPU v4 + VMEM tiling-model fixes

**Branch:** `pr-g3-gmm-v2-fp8-v4-dequant` (worktree `~/tpu-inference-prs`)
**Commit:** `d731fef0` (re-cut of `773b6ad4`: comment/commit-message accuracy fixes from the
round-6 adversarial review — same code)
**Base:** `97938b62` (fork main, 2026-06-13).
Forward-port to `vllm-project/tpu-inference` main @ `0d59fee9` (2026-07-07): **1 conflict hunk** —
upstream refactored `should_dequantize_before_matmul` (guard → assert, new
`should_dequantize_after_matmul` property). Resolution: insert the v4-FP8 `return True` after the
assert; the `calculate_tiling` hunks apply cleanly (region unchanged on the tip). On the tip the
dequant copy is bf16 (not f32), so the new `2*tk*tn*4` VMEM term becomes conservative — safe
direction, note it in review.
**Status:** ready for owner review; needs the owner's TPU gate re-run before submission.

---

## PR body draft

# Description

TPU v4's MXU has no FP8 matmul path: Mosaic rejects any FP8 RHS matmul with
`E2001 CompileTimeMosaicUnsupportedRhsType`, so FP8 block-quantized MoE checkpoints
(DeepSeek/GLM/Kimi style, block 128) **cannot run at all on v4** today. This PR routes an FP8
RHS through the *existing* dequantize-in-VMEM path when `tpu_generation() == 4`: FP8 weights
stay resident in HBM (the point, at 100s-of-GB model scale) and only the current tile is
dequantized in VMEM. v5e/v6e routing is byte-identical (gate = generation 4 AND FP8 dtype).

Exercising this path at real model dims exposed two latent bugs in the tiling search, both
generation-agnostic:

- `_gmm_vmem_estimate` did not model the f32 dequant copy + scale-product intermediate that the
  dequantize-in-VMEM path materializes, so the search picked `tile_k = size_k` (at GLM dims
  k=6144, n=256: ~20-25 MiB real usage vs 16 MiB v4 VMEM → scoped-VMEM OOM by construction).
  The new term applies only on the dequantize-in-VMEM path.
- `calculate_tiling`'s tile_k-shrink loop only ran when tile_n stepped BELOW its floor; when
  tile_n bottomed out exactly AT the floor (`size_n == 2*MXU`) with the estimate still over
  budget, tile_k silently stayed at `size_k`. The loop now always runs (its body is skipped
  when the estimate fits with a quant-block-compatible tile_k) and terminates by breaking once
  `align_to(size_k, n*lanes)//n` stops decreasing, instead of looping forever. Honest
  limitation (also commented at the break): that break fires at the first *plateau* of the
  sequence, which can sit above the 128 floor (size_k=6144 repeats 768 at n=8,9), so tile_k
  values past the first plateau are unreachable — a regime only entered when the estimate is
  still over budget at the plateau, i.e. at budgets far below any real TPU VMEM limit
  (measured: k=6144, n=192, int8 block-128 under an artificial 0.8 MiB limit returns
  tile_k=768/over-estimate where the previous floor-seeking loop found 640/fitting; at the
  real ~15.1 MiB v4 budget the search fits long before any plateau).

Standalone value: every FP8 block-quant MoE checkpoint on v4 (GLM-5.x, DeepSeek-V3-family FP8,
Kimi FP8).

## Why this is not duplicating an existing PR

Searches run (2026-07-07): open PRs matching `gmm`, `fp8 v4`, `block scale`, `DSA`. Findings:

- **PR #2324** (yiqiliu2, GLM-5.1-FP8) attacks the same v4 FP8 wall **differently**:
  `MOE_DEQUANT_FP8_BEFORE_GMM` dequantizes in HBM *outside* Pallas (plus a 2D
  `scale_n_block_size` mode). Ours keeps FP8 resident in HBM and dequantizes per-tile in VMEM —
  materially cheaper in HBM at large model scale. Named here so maintainers can pick/reconcile.
- **PR #2822** fixes a *different* bug in the same dequant path (kernel-side scale reshape when
  tile_k is not an exact multiple of the quant block; repeat+slice). Complementary: our
  `_is_tile_k_quant_block_compatible` keeps the search block-compatible, #2822 makes the kernel
  robust regardless. No overlapping hunks.
- **PR #2609** refactors the inner k-loop stepping (rhs_qbs < lhs_qbs scale application) —
  same file, different concern, no overlapping hunks.
- **PR #2896** (draft) migrates gmm_v2 to Tokamax — if it lands first, this fix needs porting
  to the Tokamax kernel; flagged for coordination.

# Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/kernels/gmm_v2_tiling_test.py -q
```
Result: **6 passed** (new file; pure-math with mocked v4 TPU info): v4 FP8 routing gate on/off-v4,
non-FP8 routing unchanged, GLM-real-dims tiling fits a reference VMEM model of the dequant path
(search now returns tile_k=3072, est. 9.6 MiB ≤ 15.1 MiB budget; the old search returned
tile_k=6144, est. ~20 MiB), fits-already no-op, termination on an unsatisfiable budget.
Adversarial check: all 6 tests fail on main before the change, but mostly for a trivial
reason (the `tpu_generation` mock target does not exist on main); the behavioral regression
was verified by probing main's `calculate_tiling` directly — it returns `tile_k = size_k =
6144` (est. ~20 MiB > the 15.1 MiB budget) at the GLM dims.

Machine-gated functional run (TPU v4, dev branch): fp8 GLM engine parity — mini block-64 top-1
1.000; real-dims block-128 (k=6144) top-1 vs fp32 0.906 > bf16-ref floor 0.875 (RESEARCH_LOG
2026-07-07). The re-cut branch needs the submitter's GMM parity/microbench re-run on v4
(the docs/13 "functionally run, not just compile" rule).

# Risk

- Routing is v4-gated: v5e/v6e byte-identical.
- Tiling fixes are generation-agnostic. For a config whose estimate fits with a
  quant-block-compatible tile_k, the shrink-loop body does not execute (tile choice
  unchanged). Two bounded behavior changes exist: (a) configs already on the
  dequantize-in-VMEM path get the new (real) VMEM term and may pick smaller tiles — the
  intended fix; (b) under budgets far below any real TPU VMEM limit the restructured loop
  stops at the first `align_to` plateau rather than the 128 floor (see Description), so an
  artificially tiny `vmem_limit_bytes` can end one plateau above what the old loop reached.
  At real budgets and model shapes the search fits before the first plateau.
- Interaction note: the dev branch also carried a v4 VMEM-budget headroom change
  (0.8 vs 0.9 capacity fraction on ≤16 MiB chips, from the DeepSeek-V4 PR series). This PR was
  validated with it present; if the v4 gate re-run without it shows small scoped-VMEM overflows
  from layout padding, fold that one-hunk headroom change into this PR.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commit carries a `Co-authored-by: Claude` trailer.)

---

## Submitter checklist (not part of the PR body)

- [ ] Re-run the v4 GMM parity/microbench on the re-cut branch (owner TPU time).
- [ ] Forward-port: resolve the single `should_dequantize_before_matmul` hunk on current main;
      re-verify the estimate term against the tip's bf16 dequant copy (term is now conservative).
- [ ] Watch #2896 (Tokamax migration) and #2822 (kernel-side ragged scale fix) for landing order.
- [ ] `pre-commit run --all-files`; add DCO `Signed-off-by` when pushing.
