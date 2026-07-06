# docs/13 — Stacked-PR decomposition summary (DSV4 → template for GLM)

Source: `~/moe-tpu/docs/13-pr-f-upstream-decomposition.md` (710 lines, read in full).

## 1. Decomposition pattern
- 49 commits / 14 files on `dsv4-flash-v4` vs `origin/main`; one file (`deepseek_v4_attention.py`) carries 35/49 commits (+2652 lines), so forward/prefill/decode/head-shard PRs **cannot** be independent cherry-picks → a **STACKED PR series in dependency order**, each branch cut from the previous PR's HEAD (13:30-44). Refuted alternative: refactor into `dsv4/{cores,prefill,decode,sharding}.py` modules first — deferred but "more attractive" after the entanglement was proven (13:200-203).
- **Stack in BRANCH ORDER as contiguous chunks** — a definitive 44-commit audit showed branch order has no hidden dep bugs; only docs/13's original *groupings* were wrong (13:179-187). Independent isolated-file PRs are verified by actually creating branches onto current `origin/main` and submitted first/parallel (13:46-52).
- **Gates**: every PR names its test + result (CPU pytest where pure-math; sub-cube parity; pod accuracy). Rule: clean cherry-pick + py_compile ≠ works — each branch must be **functionally run** (13:200-201, 442). Whole-model gates need prerequisite PRs co-applied (mHC stub proof: 0.9896 FAIL → 0.999869 PASS with PR-3; owner submits PR-3 first, 13:454-466).
- **CI story**: model after the Kimi `.buildkite/models/...` gate on a (new) v4 queue: kernel gate (`mla_v2_test`), cheap mini-config engine parity, nightly 0-shot accuracy (WinoGrande); never trust load-only UnitTests (13:282-292, 701-709).
- **AI disclosure line** (required every PR body): "Portions of this change were developed with AI assistance (Claude); every line has been reviewed and is defended by the human submitter." Owner submits — vLLM `AGENTS.md` forbids pure code-agent PRs (13:294-298, 6-8).

## 2. PR list (dependency order; no PR-4, 13:67-70)
- **Independent**: PR-1 MLA-v2 kernel on v4 (VMEM tiles + fp8→bf16 upcast, `mla/v2/kernel.py`, 13:72-89); PR-3 mHC TPU ops (13:91-101); PR-5a FP8/FP4→bf16 dequant quant-methods + F8_E8M0 map (13:349-376); PR-E FP8 MoE-expert dequant, fold into PR-5a (13:378-429); PR-12 DPScheduler KV-capacity warning (13:257-265, 657-685).
- **Stack**: PR-2 attention forward base (13:125-163) → PR-2b compressed-decode cores (13:475-489) → PR-6 runner JIT-safe prefill+DP (13:205-214, 491-505) → PR-7b runner decode + Pallas kernel (13:507-521) → PR-8 blocked prefill (13:523-533) → PR-9 head-sharding + logits (13:535-551) → PR-5b compressor routing (on PR-2 **and** PR-5a, 13:468-473, 553-573) → PR-11 comp_kv tiling (on PR-7b, 13:623-655) → PR-10a/b recompile fixes (top of stack; some need hand-ports, 13:575-621). Branch-HEAD/gate table at 13:444-452.

## 3. Lessons (bugs found only by RUNNING)
1. **Missing definer commit**: PR-2 omitted `7f405408` (`dsv4_paged_dense_mqa`); py_compile+import passed, runtime `NameError` (13:133-142).
2. **Cross-PR dependency**: PR-6's `1882077f` refactors a function PR-7's `51e3d755` created → cherry-pick conflicts; logical prefill/decode split unachievable on one file (13:165-177).
3. **Hidden context deps**: PR-5b conflicts on bare PR-2 (needs PR-5a's context, 13:468-473); PR-10 fixes don't fold (latent `NameError` from an absent `hybrid_mesh` refactor; renamed functions need hand-ports, 13:583-599).
4. **Stub contamination**: stacked branches carry old mHC stub → whole-model gates degrade unless PR-3 co-applied (13:454-466).

## 4. GLM template
(a) Submit isolated-file independents first (kernel-enablement, standalone ops, quant methods, diagnostics), each with a CPU pytest; (b) build the attention/runner stack as **branch-order contiguous chunks**, each cut from the prior HEAD, functionally validated (run a parity script, not just compile); (c) declare prerequisite PRs for whole-model gates; (d) draft descriptions with Motivation/Changes(commits)/Tests-run-with-numbers/Risk/AI-disclosure; (e) design the v4 CI gate (kernel + mini-config parity + nightly accuracy); (f) owner submits.