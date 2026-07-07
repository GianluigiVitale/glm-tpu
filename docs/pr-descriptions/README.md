# PR branch series — status (2026-07-07)

Branches live in the worktree **`~/tpu-inference-prs`** (main checkout untouched), cut from
`97938b62` = fork `origin/main` (2026-06-13) = the exact merge-base of the dev stack — the
newest base the locally-installed vLLM (`a30addc75`) can import/run. The true upstream tip
(`vllm-project/tpu-inference` main @ `0d59fee9`, 2026-07-07, fetched read-only as
`refs/remotes/upstream/main`) requires a newer vLLM than the local env; each PR's forward-port
status against it was verified with `git merge-tree` and is stated per description.

All CPU tests were run with `JAX_PLATFORMS=cpu` (TPU never touched). Every new test was
adversarially verified to FAIL on the pristine base. Commits carry `Co-authored-by: Claude`
trailers; yapf applied (isort/ruff unavailable locally — run `pre-commit run --all-files`
before pushing). **No branch is pushed; no PR is opened — owner submits** (vllm-project
AGENTS.md policy).

| PR | branch | commits | CPU tests | forward-port to tip | status |
|---|---|---|---|---|---|
| G2 MLA wrapper latent fixes | `pr-g2-mla-wrapper-latent-fixes` | `8eab375b` + `1716f345` | 8/8 (`test_mla_attention.py`, 4-dev flags) | conflict-free | ⚠️ commit 2 duplicates open **#2988** (same `P(None, ATTN_HEAD)` fix) — split into a droppable commit; commit 1 (auto-dtype NaN) is the un-duplicated substance |
| G3 gmm_v2 FP8-on-v4 | `pr-g3-gmm-v2-fp8-v4-dequant` | `773b6ad4` | 6/6 (`gmm_v2_tiling_test.py`, new) | 1 trivial hunk (upstream guard→assert refactor) | ready pending owner v4 gate re-run; coordinate #2324 (HBM-dequant alternative), #2822, #2609, #2896 (Tokamax) |
| G4 blockwise-FP8 ragged scales | `pr-g4-fp8-ragged-block-scales` | `1cfaf85b` | 5/5 (`test_fp8_ragged_block_scales.py`, new) | conflict-free (files identical on tip) | ready pending owner parity re-run |
| G1 register GlmMoeDsa | `pr-g1-register-glm-moe-dsa` | `e03c237f` | 56 pass + 19 pass (1 pre-existing env-sensitive fail, identical on base) | conflict-free textually (drifted files merge clean; re-verify semantics on tip) | ready; **declares G2 prerequisite** for the whole-model gate |
| G5 TP-topology MLA (v4 multi-host) | — not cut — | `cd8eeb6c`+`a429be54` on dev branch | — | — | **SKIPPED per docs/02**: pod-unvalidated (bar: pod 3/3, zero serving-region compiles) + #2324 conversation required first |
| G6 DSA sparse series (Stage 2) | — not cut — | Stage-2a/b/c on dev branch | — | — | **SKIPPED**: too fresh (task directive); follows docs/01 phasing later |

Submission order (docs/02): G2 → G3+G4 (parallel) → G1 → G5 (after pod 3/3 + #2324) → G6 series.

Duplicate-work searches run 2026-07-07 against `vllm-project/tpu-inference` open PRs
(GitHub search API; gh CLI unavailable): `MLA`, `GLM`, `DSA`, `fp8 v4`, `gmm`, `block scale`,
`kv_cache_dtype`. Live findings folded into each description: **#2988** (G2 overlap, load-bearing),
#2324 (G3 alternative approach + G1 env-name lineage + G5 port source), #2822/#2609/#2896
(G3-adjacent, no hunk overlap).
