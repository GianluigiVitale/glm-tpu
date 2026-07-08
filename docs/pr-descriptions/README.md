# PR branch series — status (2026-07-08)

Branches live in the worktree **`~/tpu-inference-prs`** (main checkout untouched), cut from
`97938b62` = fork `origin/main` (2026-06-13) = the exact merge-base of the dev stack — the
newest base the locally-installed vLLM (`a30addc75`) can import/run. The true upstream tip
(`vllm-project/tpu-inference` main, fetched read-only as `refs/remotes/upstream/main`:
`0d59fee9` on 2026-07-07 for G1–G4, re-fetched `6a837025` on 2026-07-08 for G5 and
`99a662a1` later on 2026-07-08 for G6) requires a
newer vLLM than the local env; each PR's forward-port status against it was verified with
`git merge-tree` / a scratch-worktree trial merge and is stated per description.

All CPU tests were run with `JAX_PLATFORMS=cpu` (TPU never touched). Every new test was
adversarially verified to FAIL on the pristine base. Commits carry `Co-authored-by: Claude`
trailers; yapf applied (isort/ruff unavailable locally — run `pre-commit run --all-files`
before pushing). **Branches are pushed to the fork (`origin`) as feature branches; no
upstream PR is opened — owner submits** (vllm-project AGENTS.md policy).

| PR | branch | commits | CPU tests | forward-port to tip | status |
|---|---|---|---|---|---|
| G2 MLA wrapper latent fixes | `pr-g2-mla-wrapper-latent-fixes` | `8eab375b` + `1716f345` | 8/8 (`test_mla_attention.py`, 4-dev flags) | conflict-free | ⚠️ commit 2 duplicates open **#2988** (same `P(None, ATTN_HEAD)` fix) — split into a droppable commit; ⚠️ commit 1's auto-dtype NaN bug is **also fixed (differently) inside #2324** — this PR is the minimal tested vehicle (see pr-g2.md) |
| G3 gmm_v2 FP8-on-v4 | `pr-g3-gmm-v2-fp8-v4-dequant` | `d731fef0` (re-cut of `773b6ad4`, comment/message accuracy) | 6/6 (`gmm_v2_tiling_test.py`, new) | 1 trivial hunk (upstream guard→assert refactor) | ready pending owner v4 gate re-run; coordinate #2324 (HBM-dequant alternative), #2822, #2609, #2896 (Tokamax); tile_k loop stops at the first align_to plateau — honest limitation stated in pr-g3.md |
| G4 blockwise-FP8 ragged scales | `pr-g4-fp8-ragged-block-scales` | `c349d1d6` (re-cut of `1cfaf85b`, comment/message accuracy) | 5/5 (`test_fp8_ragged_block_scales.py`, new) | conflict-free (files identical on tip) | ready pending owner parity re-run; lone interior-ragged fusion undetectable at the matmul call site — honest limitation stated in pr-g4.md |
| G1 register GlmMoeDsa | `pr-g1-register-glm-moe-dsa` | `e03c237f` | 56 pass + 19 pass (1 pre-existing env-sensitive fail, identical on base) | conflict-free textually (drifted files merge clean; re-verify semantics on tip) | ready; **declares G2 prerequisite** for the whole-model gate; ⚠️ 4 of 6 source files carry hunks **shared with #2324** (registry/env/gate/allow-list — disclosed in pr-g1.md); unique substance = IndexShare wrap + KV-spec fix + default-ON env |
| G5 TP-topology MLA (pure TP×EP, v4 multi-host) | `pr-g5-mla-pure-tp` | `132a11f9` (squashed re-cut of `cd8eeb6c`+`a429be54`+`7ae390f2`; debug env knobs dropped, gather axes made descriptor-aware — deltas in pr-g5.md) | 17 new (6 `test_mla_cross_shard.py` + 9 `…_page_size.py` + 2 `…_min_token_bucket.py`) + platform suite 39/39; 7 behavioral fails on base + max-diff-3.91 probe (adversarial) | 1 hunk (tip enriched the TuningKey in the restructured region; mechanical) | cut 2026-07-08 after pod validation landed (GSM8K n=32 clean @ 32.9 tok/s, runs 26/28/47 in results.db); ⚠️ **is a port of #2324 hunks** (`Co-authored-by: yiqiliu2` carried; owner must run the #2324 conversation first); ⚠️ **#2988 = alternative fix for the same cross-shard bug** (textual+semantic conflict on the default token specs — disclosed in pr-g5.md); lm_head-guard heritage (761ea755/87ace031/10efa393) audited: guards a DSV4-only machinery absent from base — documented, not ported |
| G6 DSA kernels (Stage-2 kernel layer) | `pr-g6-dsa-kernels` | `0ac6eb9e4` (5 new files from `-next` @ `886eaceb`; yapf-only, AST-identical per file) + `5bf3e5927` (docstring-accuracy pass from the post-cut adversarial review — code-AST-identical; fold back to `-next`) | 66/66 (`test_dsa_indexer_kernel.py` 24 + `test_dsa_sparse_mla.py` 42; 62+4 graceful skips without the glm-tpu HF-math oracle) | conflict-free (all-new files; merge-tree clean vs `99a662a1`, 2026-07-08) | cut 2026-07-08 after silicon gates landed (GATE 2a ACCEPT after 3 Mosaic fixes; GATE 2b ALL PASS on metal — A2 selected-set exact, artifacts `docs/artifacts/kernelprobe-*`); **KERNELS ONLY** — serving wiring is a follow-on; ⚠️ "first public" claim scoped to the DSv3.2/GLM DSA variant: merged `kernels/experimental/deepseek_v4/` (#2903/#2905) ships a DeepSeek-V4 compressor-based StreamIndex top-k + sparse MLA (disclosed in pr-g6.md); #2324 disables the indexer (`TPU_DISABLE_DSA_INDEXER`) — this PR is its missing kernel layer |

Submission order (docs/02): G2 → G3+G4 (parallel) → G1 → G5 (after the #2324 conversation) →
G6 (kernels) → G6 follow-on (serving wiring).

Duplicate-work searches run 2026-07-07 against `vllm-project/tpu-inference` open PRs
(GitHub search API; gh CLI unavailable): `MLA`, `GLM`, `DSA`, `fp8 v4`, `gmm`, `block scale`,
`kv_cache_dtype` — **plus a full live-diff read of #2324** (head `c2822bd7`, 2026-07-04; its
title matches none of the search terms, which is how its overlaps were initially missed).
Re-swept 2026-07-08 for the G5 cut (`MLA`, `dp attention`, `TP topology`, `all-gather`,
`page size`, `GLM`, `cross-shard` + live diffs of #2324 — unchanged since 07-04 — and #2988):
new load-bearing finding — **#2988 rewrites the same `mla_attention` default token specs**
(MLP_TENSOR→ATTN_DATA) as an alternative fix for the cross-shard-query bug G5 ports from
#2324; G5's upstream tip re-check ran against `6a837025` (2026-07-08). Adjacent, no-overlap:
#2930 (MLA tuned params, same region), #2955 (RPA-v3 v4 blocks), #3056/#2767 (DP-attention
sharding).
Re-swept 2026-07-08 for the G6 cut (`DSA`, `sparse attention`, `indexer`, `lightning
indexer`, `top-k attention`, `sparse MLA`, `NSA`, `deepseek sparse`, `GLM`; title scan of
the 50 newest open PRs; files/diff reads of #2324, #2988, #3073, #3062, #3096): **no open
PR ships DSA kernels**; #2324 confirmed to disable the indexer (`TPU_DISABLE_DSA_INDEXER`,
"until a JAX-native DSA lands"). Load-bearing counterfinding: **merged main already carries
`kernels/experimental/deepseek_v4/`** (#2903/#2905/#2980, 2026-06-22..24) — a DeepSeek-V4
KV-compressor StreamIndex top-k + topk-consuming MLA Pallas stack; adjacent mechanism, no
file overlap, but it forbids an unscoped "first public TPU Pallas indexer/top-k" claim
(pr-g6.md scopes the claim to the exact-top-k uncompressed-cache DSv3.2/GLM DSA variant).
G6's upstream tip re-check ran against `99a662a1` (2026-07-08).
Live findings folded into each description: **#2988** (G2 commit-2 overlap, load-bearing);
**#2324** (G1: shared registry/env/indexer-gate/allow-list hunks across 4 files — G1's
IndexShare wrap, KV-spec fix and default-ON env are the un-shared substance; G2: alternative
scalar-identity-scale fix for the commit-1 auto-dtype NaN bug, untested and needing a rebase
there; G3: alternative HBM-dequant approach; G4: no file overlap; also the G5 port source);
#2822/#2609/#2896 (G3-adjacent, no hunk overlap).
