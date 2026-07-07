# PR-G2 — [MLA] Fix two latent bugs in `VllmMLAAttention.process_weights_after_loading`

**Branch:** `pr-g2-mla-wrapper-latent-fixes` (worktree `~/tpu-inference-prs`)
**Commits:** `8eab375b` (auto-dtype NaN fix) + `1716f345` (W_UV_scale axis fix — ⚠️ overlaps open PR #2988, see below)
**Base:** `97938b62` (fork main, 2026-06-13 — the newest base the local vLLM install can run).
Forward-port to `vllm-project/tpu-inference` main @ `0d59fee9` (2026-07-07): **conflict-free**
(`git merge-tree` verified; `mla_attention.py` and the test file are byte-identical on the tip).
**Status:** owner decision needed before submission on two overlaps (see Overlap section):
commit 2 duplicates #2988; commit 1's bug also has an alternative (untested, needs-rebase)
fix inside #2324.

---

## PR body draft (paste below into the GitHub PR)

# Description

Two latent bugs in `VllmMLAAttention.process_weights_after_loading`, both reachable by **any**
model on the vLLM MLA path (DeepSeek-V3.x / Kimi / GLM family) — no new model support involved.

**1. `kv_cache_dtype=auto` → NaN forward** (commit 1). The absorbed projections were routed
through `quantize_tensor(None, W_UK_T/W_UV)`. dtype `None` resolves to float64 (numpy
semantics, `finfo` max ~1.8e308): the scales collapse to **0.0**, the "quantized" weights
overflow on the f32 cast, and every layer output turns NaN under the *default* KV-cache dtype.
Fix: when no quantized KV dtype is set, keep the weights in the activation dtype
(`t2j_dtype`-mapped, fp16-safe) with identity scales of the exact shape `quantize_tensor`
would produce — same shapes, forward path unchanged. The quantized-KV branch still makes the
byte-identical `quantize_tensor(self.kv_cache_quantized_dtype, …)` call.

**2. `W_UV_scale` sharded on the wrong axis** (commit 2). `W_UV_scale` is `[1, H, v]`
(`expand_dims` puts the head axis at axis 1) but was `device_put` with `P(ATTN_HEAD,)` on the
size-1 axis 0 → `IndivisibleError` at load whenever the head-TP mesh product > 1. Fix: shard
`P(None, ATTN_HEAD)`. **Note:** this is substantively identical to open PR #2988 — commit 2
should be **dropped** if #2988 merges first (its 4-CPU-device regression test can be offered
to #2988, which currently ships no test).

## Overlap with existing PRs (disclosed, verified against the live diffs)

Searches run (2026-07-07, `api.github.com/search`, equivalent to
`gh pr list --repo vllm-project/tpu-inference --state open --search "…"`):
`MLA`, `kv_cache_dtype`, `fp8 v4`, `block scale`; plus a live-diff check of the open GLM PR
**#2324** (head `c2822bd7`, updated 2026-07-04), whose title matches none of those terms but
which reworks the same function. Findings:

- **PR #2988** (dawnhan1111, "Fix tensor parallelism (model>1) for MLA models (Kimi K2.6)",
  open, non-draft) contains the **same `W_UV_scale` fix** (`P(None, ATTN_HEAD)`). Bug 2 is
  therefore split into its own commit, marked droppable.
- **Bug 1 (the auto-dtype NaN) is not addressed by #2988, but it IS addressed — differently —
  inside open PR #2324** (GLM-5.1-FP8 multi-host): its `mla_attention.py` rework adds an
  explicit `kv_cache_dtype=auto` else-branch that keeps `W_UK_T`/`W_UV` unquantized and
  device_puts a replicated **scalar** 1.0 for both scales (vs the shaped identity scales
  here). Why this PR still stands on its own: #2324 is a ~5.9k-line multi-purpose PR whose
  `mla_attention.py` hunks are written against pre-refactor context (its scale
  `expand_dims` axes predate current main — it needs a rebase), it ships the fix entangled
  with a TP-selective loader, and it carries no regression test for this path; this PR is the
  minimal, on-tip, tested vehicle for the same bug. As with commit 2 / #2988: coordinate with
  #2324's author, and if maintainers prefer the fix landing there, this PR's auto-dtype
  regression test can be offered to #2324 instead.
- No other open PR touches `process_weights_after_loading` in the MLA wrapper.

# Tests

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python -m pytest tests/layers/vllm/test_mla_attention.py -q
```
Result: **8 passed** (6 pre-existing + 2 new) on the branch.
Adversarial check: the 2 new tests **fail on main before the change** — the auto-dtype test
observes the 0.0 scale collapse (reproduced: `quantize_tensor(None, ones)` → `scale.max() == 0.0`),
and the 4-device test hits the `IndivisibleError`.

Machine-gated functional run (TPU v4, dev branch carrying these fixes): 1-chip GLM engine
parity vs HF reference — mini bf16 PASS, mini fp8 PASS, real-dims fp8 PASS (final hidden
0.619 < bf16 floor 0.722; logits 0.910 < 0.958; top-1 vs fp32 0.906 > bf16-ref 0.875).
The re-cut branch itself needs a 1-chip parity re-run by the submitter before merging
(RESEARCH_LOG 2026-07-07 has the harness details).

# Risk

- The auto-dtype branch was *always*-NaN before, so no working configuration can regress.
- The quantized-KV branch is byte-identical.
- Head-TP == 1 behavior of the sharding fix is equivalent (verified in the CPU test at 1 device).

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commits carry `Co-authored-by: Claude` trailers.)

---

## Submitter checklist (not part of the PR body)

- [ ] Re-check #2988 state; if merged, `git rebase --onto` dropping commit `1716f345` and offer
      the 4-device test to #2988 as a review comment.
- [ ] Watch #2324 (contains an alternative scalar-identity-scale fix for Bug 1): if it lands
      first, drop commit `8eab375b` too and offer its regression test there.
- [ ] Re-run the 1-chip machine-gated parity harness on the re-cut branch (owner TPU time).
- [ ] `pre-commit run --all-files` (yapf applied; isort/ruff not runnable in the local env).
- [ ] Add DCO `Signed-off-by` when pushing.
