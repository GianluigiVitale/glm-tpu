# PR-G5 — [MLA] Support the pure TP × EP topology (MLA without DP attention) on TPU v4 multi-host

**Branch:** `pr-g5-mla-pure-tp` (worktree `~/tpu-inference-prs`, pushed to the fork)
**Commit:** `132a11f9` (squashed re-cut of dev commits `cd8eeb6c` + `a429be54` + `7ae390f2`
from `glm-5.2-v4`; deltas from the dev branch are listed below and in the commit message)
**Base:** `97938b62` (fork main, 2026-06-13 — the newest base the local vLLM install can run).
Forward-port to `vllm-project/tpu-inference` main @ `6a837025` (2026-07-08, re-fetched at cut
time): **1 conflict hunk** — the tip enriched the batched-decode `TuningKey` (adds kv/q dtypes,
page/packing dims, `max_num_seqs = md.padded_num_reqs // dp_size`, `pages_per_seq`) in the exact
region this PR restructures. Resolution: keep the tip's richer TuningKey inside the non-v4
`else` branch, built from the POST-gather shapes. All other hunks (platform check, page size,
min bucket, EP-head constraint, tests) merge clean; the tip still hard-requires DP attention
for MLA, still returns a bare 1024 page size, and still has no cross-shard gather — verified by
reading the tip files, so the PR remains needed there.
**Status:** pod-validated on the dev branch (see Evidence); needs the owner's #2324
conversation before submission (this PR is a port of #2324 hunks — see Overlap) and a pod
re-run of the re-cut branch.

---

## PR body draft (paste below into the GitHub PR)

# Description

753B-class MLA models (GLM-5.x, DeepSeek-V3-family) cannot replicate their attention weights
(~13 GiB/chip at GLM-5.2 scale), so on a 32-chip v4-64 slice they must run **pure TP × EP**:
attention heads sharded over `ATTN_HEAD` (model × expert), tokens sharded over `MLP_TENSOR`
inside the MLA shard_map, **no DP attention**. Today `tpu_platform.py` hard-rejects that
topology (`MLA models require … DP attention`), and four independent breakages sit behind the
check. This PR — a port of open **PR #2324**'s TP-topology fixes (see Overlap) — makes the
topology work end-to-end:

1. **Cross-shard-query fix** (`attention_interface.mla_attention`). The MLA kernel derives
   each query's absolute position from the sequence descriptors (`kv_lens`/`cu_q_lens`,
   sharded `P(ATTN_DATA)` — i.e. replicated across model/expert/dcp) plus its **local** iota.
   With tokens sharded over `MLP_TENSOR`, every token shard beyond the first applies the wrong
   causal mask and corrupts the last real query's hidden state. Fix: all-gather
   q/q_rope/k/k_rope inside the shard_map body over exactly the token-shard axes whose
   descriptors are replicated, run the kernel on the full token axis, and dynamic-slice the
   output back to the shard's range (q/out on axis 1 — head-major; q_rope/k/k_rope on axis 0).
   No-op when that shard product is 1: single chip, and **pure attention-DP meshes keep their
   current behavior** (their descriptors are co-sharded with the tokens via `ATTN_DATA`, so
   the gather-axis selection excludes them — a deliberate delta from #2324, which gathers over
   all `MLP_TENSOR` axes; the two are identical on the validated TP × EP topology where the
   attn-DP axes have size 1). The `TuningKey` for the v5/v6 tuned-params lookup is built from
   the POST-gather shapes — the tensors the kernel actually sees.
2. **v4 MLA kernel parameters** (same function). v4 has 16 MB VMEM (vs 32 MB+ on v5/v6):
   fixed block sizes `(1,1,1)`/`(1,8,8)`, `decode_batch_size 4`, and fp32 softmax scores
   (bf16 scores can saturate on deep stacks) — #2324's validated values. v5/v6 keep the
   tuned-params lookup unchanged.
3. **v4 page size** (`flash_attn_mla.get_page_size`): 512 when `kv_lora_rank > 256` (the
   GLM/DeepSeek 512-latent overflows kernel scratch at page 1024), gated to v4 — v5/v6 keep
   their tuned 1024. The version probe uses `tpu_info` (env/GCE metadata), **not**
   `jax.devices()`: this runs in the driver's config path, and initializing the TPU slice
   there makes the driver hold the libtpu multi-process lockfile against its own EngineCore
   child.
4. **EP-head layout before o_proj** (`flash_attn_mla.forward`): when expert parallelism splits
   attention heads (`ATTN_HEAD` product > `model` size), pin the attention output to
   `P(None, 'model')` — keeps the expert-axis resharding point explicit before the
   RowParallelLinear; no-op when `ATTN_HEAD` reduces to `model` alone. (On this whole-model
   GSPMD path this is a layout hint rather than a numeric fix; #2324's original comment claims
   a numeric role on its older base.)
5. **`TPU_MIN_TOKEN_BUCKET`** (`tpu_runner` + Ray propagation): token buckets below the
   token-shard product fail shard_map divisibility at startup (16 < 32 on v4-64 TP × EP).
   The env floors the bucket table; unset/0 keeps the table **byte-identical**.
6. **Drop the hard MLA → DP-attention requirement** (`tpu_platform`); `NEW_MODEL_DESIGN=1`
   stays required.

Validated end-to-end with GLM-5.2-FP8 (753B) on a v4-64 pod at pure TP-32 (see Tests).

## Why this is not duplicating an existing PR (and what it ports)

Searches run 2026-07-08 against `vllm-project/tpu-inference` **open** PRs
(`api.github.com/search`, equivalent to `gh pr list --state open --search "…"`): `MLA`,
`dp attention`, `TP topology`, `all-gather`, `page size`, `GLM`, `cross-shard` — plus live-diff
reads of #2324 and #2988. Findings:

- **PR #2324** (yiqiliu2, "GLM-5.1-FP8 MLA multi-host + multi-host weight loader + FP8 MoE
  direct path", open, non-draft, head `c2822bd7`, unchanged since 2026-07-04, needs rebase).
  **This PR is a port of five of its hunks** — disclosed hunk-by-hunk: the
  `attention_interface` cross-shard all-gather + v4 blocks/fp32 scores, the `get_page_size`
  512 cap, the `flash_attn_mla` `P(None,'model')` EP-head constraint, the `tpu_platform`
  MLA-without-DP change, and `TPU_MIN_TOKEN_BUCKET`. Deltas of this port vs #2324, verified
  against its live diff: axis adaptation to the current head-major q layout (its gathers/slices
  are all token-major axis 0; the current `mla_attention` q/out are `[H, T, lkv]`), gather
  axes restricted to descriptor-replicated axes (it gathers all of `MLP_TENSOR` — wrong on
  attention-DP meshes, which its GLM-5.1 use case never ran), post-gather TuningKey placement
  (its base predates the tuned-params lookup; it hardcodes the v5/v6 blocks), the
  `tpu_info`-based version probe (its probe reads `hf_config` only, ours must not touch
  `jax.devices()` from the driver), v4-gating of the 512 page cap (its cap applies on every
  generation), min-bucket default 0 = byte-identical (its default is 512), and CPU regression
  tests (it ships none for these hunks). #2324 also carries ~4k lines of unrelated multi-host
  loader/FP8-MoE work. **Do not double-submit someone else's work:** the owner should first
  comment on #2324 and either (a) help land #2324 and rebase this PR to only the deltas above,
  or (b) with yiqiliu2's and the maintainers' consent submit this minimal, tested,
  currently-conflict-light vehicle — the commit already carries
  `Co-authored-by: yiqiliu2 <66063897+yiqiliu2@users.noreply.github.com>`.
- **PR #2988** (dawnhan1111, "Fix tensor parallelism (model>1) for MLA models (Kimi K2.6)",
  open, head `26aca67b`) attacks the **same cross-shard-query bug class differently**: it
  rewrites `mla_attention`'s default token specs from `MLP_TENSOR` to `ATTN_DATA` — i.e.
  replicates tokens across model/expert instead of sharding + gathering in-body. **Textual
  conflict:** its changed spec lines are inside this PR's hunk context. **Semantic conflict:**
  the two fixes are mutually exclusive as defaults (if #2988 lands first, the in-body gather
  keyed on the effective specs correctly degrades to a no-op — the gather-axis selection here
  reads the *effective* specs, so this PR remains safe but its gather becomes dead code on
  those meshes; if this PR lands first, #2988's spec rewrite is unnecessary). Maintainers
  should pick one default: #2988's is simpler; this PR's keeps upstream-of-attention
  activations token-sharded (no `[T, hidden]` replication at large T) and additionally fixes
  v4 blocks/page-size/bucket floor/platform gate, which #2988 does not touch. Named for
  reconciliation; this PR's other five pieces have no overlap with #2988.
- **PR #2930** ([MLA] tuned-params mapping for Mistral-Small-4 on v7x) adds entries near the
  TuningKey construction this PR moves into the `else` branch — merge-order coordination only,
  no semantic overlap (verified: it adds mapping-table entries + 9 lines in
  `attention_interface`).
- **PR #2955** (RPA default block sizes for Gemma4 on v4) tunes the *ragged-paged-attention v3*
  kernel defaults, not the MLA v2 path — same "v4 needs smaller blocks" theme, different file,
  no overlap.
- **#3056 / #2767** (2D DP-attention sharding; explicit attention-DP multihost meshes) touch
  `sharding.py`/`tpu_runner.py` but not the MLA interface/backend or the platform MLA check;
  the descriptor-aware gather-axis selection here is exactly what keeps this PR correct if
  those land (attention-DP axes are excluded by construction, with a unit test).

# Tests

```bash
# all CPU-only (JAX_PLATFORMS=cpu); no TPU touched
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  python -m pytest tests/layers/common/test_mla_cross_shard.py -q          # 6 passed (new)
JAX_PLATFORMS=cpu python -m pytest \
  tests/layers/vllm/backends/test_flash_attn_mla_page_size.py -q           # 9 passed (new)
JAX_PLATFORMS=cpu python -m pytest \
  tests/runner/test_tpu_runner_min_token_bucket.py -q                      # 2 passed (new)
JAX_PLATFORMS=cpu python -m pytest tests/platforms/test_tpu_platform.py -q # 39 passed
```

- `test_mla_cross_shard.py` (new): the Pallas kernel cannot run on CPU, so
  `mla_ragged_paged_attention` is monkeypatched with a jnp-only stand-in implementing the same
  contract (paged-cache scatter + causal ragged attention from descriptors + local iota).
  Pins: 8-way token-sharded `mla_attention` == unsharded reference (out + cache); product-1
  no-op; kernel AND TuningKey see post-gather shapes; v4 gets `(1,1,1)`/`(1,8,8)`/batch-4/fp32
  scores; explicit-but-equivalent shardings still fixed; gather-axis selection algebra
  (attention-DP co-shard ⇒ no-op; hybrid DP×TP ⇒ model-axis only; q/kv spec disagreement ⇒
  historical behavior).
- `test_flash_attn_mla_page_size.py` (new): v4 512 gate (incl. the 257 boundary), v5/v6 keep
  1024, no-`kv_lora_rank` configs keep 1024, fail-open on probe error, and a guard that the
  probe never calls `jax.devices()`/`jax.local_devices()`.
- `test_tpu_runner_min_token_bucket.py` (new): floor applies (min bucket 512); unset vs `0`
  byte-identical table; default table still starts at 16.
- `tests/platforms/test_tpu_platform.py`: new `…_mla_without_dp_attention` (full
  `check_and_update_config` passes with `use_mla` + `NEW_MODEL_DESIGN`, no DP attention); the
  pre-existing `…_mla_checks` updated to the new contract (`NEW_MODEL_DESIGN` still raises).

**Adversarial check (all run against pristine `97938b62`):** the v4 page-size cases + the
`jax.devices()` guard + the min-bucket floor + both platform MLA tests **fail on main before
the change** (7 behavioral failures; the deliberate no-change guards — v5/v6 page size,
fail-open, unset bucket table — pass on both, as designed). The cross-shard test file needs a
branch-only helper, so its behavioral content was probed directly on main: 8-way-sharded
output diverges from the reference at max |diff| **3.91**, and main's TuningKey logs
`max_num_tokens=2` (the 1/8 shard) instead of 16 — both the bug and the pre-gather-tuning bug
reproduce.

**Regression (existing suites, branch vs pristine base — identical results):**
`tests/layers/vllm/test_mla_attention.py` 6/6 both; `tests/runner/test_tpu_runner.py` 11/11
both; `tests/layers/vllm/backends/test_flash_attn_mla.py` 5 passed / 5 failed **identically on
base and branch** (the 5 forward tests need the Pallas kernel: `Unsupported TPU device kind:
cpu` — pre-existing, unrelated).

**Pod validation (machine-gated; dev branch `glm-5.2-v4` carrying these exact hunks):**
GLM-5.2-FP8 (753B, FP8-resident) on a v4-64 pod (8 hosts × 4 chips), pure TP-32
(`model×expert`, `dp_attention: false`, DSA indexer disabled, `TPU_MIN_TOKEN_BUCKET` 512/32):

- Engine bring-up + first correct generations: provenance **run 26** (`stage1 pod smoke #16`,
  GSM8K n=4 smoke, acc 75.0 — 1 miss = truncation at the 512-token cap; fork @ `10efa393`).
- First batched wave: **run 28** (GSM8K batched 16-way; 16 items recorded, 12 correct; the
  session's crash class was later root-caused to an unrelated mla.v2 kernel OOB read, fixed
  separately on the dev branch).
- Clean validated run: **run 47** (`gsm8k_n32 bucket32` validation, fork @ `02e44b36`): GSM8K
  **n=32 acc 87.5** (6/32 truncation-capped at 1024 tokens), 20,280 generated tokens in
  616.7 s = **32.9 tok/s aggregate**, `TPU_MIN_TOKEN_BUCKET=32`, zero crashes.

Every item (prompt, verbatim output, extracted answer, pass/fail, tokens, latency) is stored
in the harness provenance DB (`glm-tpu/bench/results.db`, runs 26/28/47); narrative entries in
`glm-tpu/docs/RESEARCH_LOG.md` (2026-07-07 06:40, 12:40, 15:05 UTC). The re-cut branch itself
needs a pod re-run by the submitter before merging (the "functionally run, not just compile"
rule): note the dev branch also carried the mla.v2 v4 fp8-upcast/VMEM kernel changes and an
OOB fix from other PRs of this series — at v4 pod scale this PR is validated **in combination
with them**, standalone it is validated CPU-only.

# Risk

- **Pure attention-DP MLA meshes (today's only supported MLA config): behavior-preserving by
  construction** — the gather-axis selection excludes descriptor-co-sharded (ATTN_DATA) axes,
  so `_gather_size == 1` and the body is byte-identical to main (unit-tested algebra).
- Hybrid DP×TP MLA meshes (`model>1` with DP attention — the #2988 Kimi case) change from
  **broken** (wrong causal masks over the model axis) to fixed; no working config regresses.
- v4 block/page changes are v4-gated; v5/v6 byte-identical (tuned lookup + 1024 pages kept).
- `TPU_MIN_TOKEN_BUCKET` unset/0 keeps the bucket table byte-identical (unit-tested).
- The platform change only *removes* a rejection; configs that passed before still pass.
- The EP-head constraint is a no-op unless `ATTN_HEAD` product > `model` size (impossible on
  the meshes accepted before this PR).
- ICI cost of the gather: ~2 × max_num_tokens × (lkv+r) bytes per attention call plus the
  head-major q gather — negligible against MoE GMM traffic at the validated batch sizes;
  measured end-to-end in the 32.9 tok/s pod number.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(The commit carries `Co-authored-by: Claude` and — attribution for the ported #2324 hunks —
`Co-authored-by: yiqiliu2` trailers.)

---

## Audit note: the dev branch's lm_head vocab-sharded logits guards (NOT ported — nothing to port)

The dev branch also hardened `vllm_model_wrapper.compute_logits` for the pure-TP topology
(commits `87ace031` + `10efa393` + `761ea755`-wrapper-hunks: only force token-sharded /
vocab-replicated logits when the lm_head is actually replicated; size-1-mesh-axis-aware
vocab-shard probe). Those guards fix an interaction with the **DSV4-branch STEP-1
prompt-logprobs machinery** (`_logits_partition_spec` / `logprobs_layout` / forced
token-sharding under `model_parallel`), which does **not exist** on `97938b62` or on the
upstream tip. Verified at base: `jit_compute_logits_func` out-sharding is
`P(MLP_DATA, MLP_TENSOR)` (vocab stays sharded, `vllm_model_wrapper.py:660-663`) and the
lm_head weight loads `P(MLP_TENSOR, None)` (`layers/vllm/quantization/unquantized.py:169-170`)
— layouts agree, so the 1.77 GiB lm_head all-gather the guards prevent **cannot occur on this
base**; porting them would add dead code referencing nonexistent variables. They become
relevant only if the DSV4 logits-layout series is ever upstreamed, and belong to that series.
(The pod runs cited above ran the dev branch, i.e. with the guards active in the DSV4
machinery.)

## Deltas vs the dev branch (for the reviewer of this re-cut)

1. `TPU_MLA_V4_KV_PAGES`/`TPU_MLA_V4_QUERIES` debug env overrides **dropped** (investigation
   scaffolding; every pod run used the hard-coded defaults — verified in the provenance DB
   env records).
2. Gather axes restricted from "all `MLP_TENSOR` axes" to "token-shard axes whose descriptors
   are replicated", derived from the **effective** specs (explicit-spec callers keep their
   layout; q/kv spec disagreement falls back to historical behavior). Identical computation on
   the pod topology (attn-DP axes size 1); keeps DP-attention meshes untouched — required for
   an upstream PR since DP attention is the only MLA config main accepts today.
3. `import os` added to `tpu_runner.py` (base lacks it); comment wording de-GLM-ified.
4. Everything else is the dev branch's final state of these hunks (incl. the round-2 review
   fixes: post-gather TuningKey, v4-gated page cap, softened o_proj comment, and the
   `tpu_info` driver-safe probe).

## Submitter checklist (not part of the PR body)

- [ ] **#2324 conversation first** (docs/02 §G5): comment on #2324, agree on path (a) or (b);
      keep/drop the `Co-authored-by: yiqiliu2` trailer accordingly.
- [ ] Check #2988 state at submission; if merged, rebase: the platform/page/bucket/EP-head
      pieces stand, the in-body gather becomes dead code under its ATTN_DATA default specs —
      coordinate which default survives.
- [ ] Forward-port: resolve the single TuningKey conflict hunk on current main (keep the
      tip's richer key, post-gather, in the `else` branch).
- [ ] Pod re-run of the re-cut branch (owner TPU time): the Stage-1 bring-up + GSM8K n=32
      gate; note the kernel-side v4 enablement (mla.v2 fp8 upcast + OOB fix) ships in other
      PRs of this series — co-apply for the full-model gate.
- [ ] `pre-commit run --all-files` (yapf applied; isort/ruff not runnable in the local env);
      add DCO `Signed-off-by` when pushing.
