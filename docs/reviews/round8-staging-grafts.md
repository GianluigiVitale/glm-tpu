# Round 8 — adversarial review: round-2 staging merges (the "grafts")

**Target**: `~/tpu-inference-next`, branch `glm-5.2-v4-next` @ `82fe3f76` (clean tree).
**Commits under review**:

| commit | what it is | parents |
|---|---|---|
| `f5f10b63` | merge glm-5.2-v4-2int @2e41b08a (PWAL-precomputed indexer params + GLM_DSA_SCORER gate) | `999f0307` + `2e41b08a` |
| `f095a8ca` | merge glm-5.2-v4-sparse-prefill (masked-XLA per-token sparse prefill) — **conflict in mla_attention.py** | `f5f10b63` + `c1456937` |
| `534cd74d` | merge glm-5.2-v4-mtp-g4 (Stage-3 dense-MTP M1 + G4 index share) — **conflict in mla_attention.py** | `f095a8ca` + `89e1d5b5` |
| `82fe3f76` | tests: pin 2D ShardingAxisName scheme in bare-mesh PWAL tests | `534cd74d` |

**Method**: all verification CPU-only (`JAX_PLATFORMS=cpu` exported before interpreter
start; Pallas `interpret=True`). The TPU was never touched. Suites run from the worktree
root under `~/vllm-env` (note: `python -m pytest` from any other CWD silently imports
`tpu_inference` from the *editable install at `~/tpu-inference`* — runs from other
directories used an explicit `PYTHONPATH=~/tpu-inference-next`). Review is read-only;
probe scripts and re-merge scratch live in the session scratchpad, not the worktree.

---

## Verdict

The two real conflict resolutions (`f095a8ca`, `534cd74d`) are exact structural grafts —
nothing from either parent was silently dropped anywhere in the four commits, and the
grafted code survives every adversarial experiment I could construct on CPU. **No
correctness defect found in the merge resolutions.** Three findings below: one real
TPU-bring-up resource hazard on the (default-off) `GLM_DSA_SCORER=pallas` × prefill
combination, one test-hygiene coupling, one doc drift. Two observations of inherited
(not merge-introduced) semantics that the MTP fidelity work should know about.

---

## Findings

### F1 — `GLM_DSA_SCORER=pallas` at prefill token shapes is a compiled-mode resource hazard (SMEM scalar-prefetch + megagrid); semantics verified, budget is not (MEDIUM, gated: default `xla` unaffected)

The `f095a8ca` graft routes the *single* `_score_and_select` formulation
(`tpu_inference/layers/vllm/custom_ops/mla_attention.py:1097-1125`) through
`indexer_scores_pallas` for **decode AND masked-prefill** scoring. On its home branch the
kernel was only ever *executed* at decode shapes (R rows ≈ padded requests). At prefill
the call is row-per-TOKEN:

- scalar-prefetch table `bt_flat = i32[T * max_blocks]` — the kernel docstring already
  flags SMEM residency as "remaining for real-TPU validation" at **256 KiB**
  (R=64, 128K-ctx pages of 128). At a T=8192 prefill bucket the same table is
  **32 MiB** (8192 × 1024 × 4 B) — two orders of magnitude past anything the DSV4
  scalar-prefetch idiom was ever validated at, and far past any plausible SMEM budget.
- grid `(T, max_blocks)` = **8.4 M sequential grid steps** per full layer per prefill
  step at those shapes.
- the `f32[T, max_blocks·page_size]` score map is **4 GiB** at T=8192/128K — this one is
  NOT pallas-specific (the XLA walk materializes the same map; it is the accepted
  O(T·ctx) cost of the design-(b) correctness-first prefill), listed for completeness.

Nuance on "new code no branch ever tested": the *compile-time* exposure is NOT new —
on the 2int home branch the scorer lived inside `_sparse_decode_branch`, and XLA compiles
both `lax.cond` branches into every token-bucket executable, so prefill-bucket
executables already contained the pallas call at T-shapes. What the graft adds is that
masked-prefill steps now *execute* it. Either way, the **on-metal 2b gate (task #9) must
include a prefill-bucket compile probe with `GLM_DSA_SCORER=pallas`**; if Mosaic rejects
the scalar-prefetch table at bucket shapes, the cheap containment is to split the scorer
gate by step type (pallas for the decode cond arm, XLA walk for the prefill arm) —
`_score_and_select` already receives both predicates.

Semantics at prefill shapes are NOT the problem — see assurance item (1): I attempted to
construct a failing shape and could not; the kernel contract is genuinely per-row.

### F2 — `test_indexshare_shared_layer_reuses_prefill_indices` is coupled to the default scorer (LOW, test hygiene)

`tests/layers/vllm/test_glm_dsa_sparse_prefill.py:688` asserts `score_calls["n"] == 1`
by counting `paged_indexer_scores` calls, without pinning `GLM_DSA_SCORER`. Under an
ambient `GLM_DSA_SCORER=pallas` (exactly how I cross-ran the suite) scoring routes
through the kernel, the counter reads 0, and the test fails while all 11 substantive
tests pass. Same environment-isolation class as the issues `82fe3f76` fixes. Suggest
`monkeypatch.setenv("GLM_DSA_SCORER", "xla")` next to the existing
`GLM_DSA_MODE` setenv (or count both scorers).

### F3 — stale comment in `glm_dsa_indexer.py` scorer-gate block (NIT, doc drift)

`tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py` (~line 163): "Only consulted
on the pallas_decode **sparse-decode** path" — since `f095a8ca` the gate is consulted for
prefill/mixed scoring too. The `mla_attention.py` docstrings were correctly updated in
the resolution; this one lags. (The neighboring claim "xla_ref keeps the reference
scorer" remains true.)

---

## Observations (inherited semantics, NOT merge defects — verified identical to the G4 parent)

**O1 — preseeded draft loop steps go dense after a mixed/prefill target step, even with a
real seed.** The sparse-reuse gate is
`sparse_step = is_decode_step ∧ any(seed ≠ -1)` (mla_attention.py:1245-1247), and
`is_decode_step` reads the **stale step-0 `request_distribution`** (the proposer's loop
`replace()` at `eagle3.py:728-734` rewrites positions/seq_lens/qsl/block_tables but not
the distribution; the F3 pad-guard comment depends on exactly this). Loop batches are
always 1 token/request, so after a step whose distribution was mixed the loop steps
refuse the seed and run dense. Verbatim in parent `89e1d5b5` (`sparse_step =
glm_dsa_is_pure_decode(...)` before the same `logical_and`) — not merge-introduced.
Consequence is draft-fidelity/acceptance-rate only (dense over-attends vs the trained
sparse pattern; the target model still verifies), but worth a line in the MTP fidelity
notes.

**O2 — `f5f10b63` contains zero resolution decisions.** The mechanical merge of its
parents reproduces the committed tree byte-for-byte (see item 4). The task brief's
"scorer gate grafted inside `_score_and_select`" decision actually lives in `f095a8ca`;
the preseeded-dispatch decision in `534cd74d`.

---

## Refutation attempts + positive assurance, per review item

### (1) The scorer gate reaches both scoring paths with correct per-token `kv_lens`

**Structure** (merged file): scoring is hoisted OUT of the attention branches into one
`lax.cond(is_decode_step ∨ ctx_gt_topk, _score_and_select, _dummy_indices)`
(mla_attention.py:1133-1135); both `_sparse_decode_branch` (:1158) and
`_sparse_prefill_branch` (:1184) consume the same once-computed `indices`. There is
exactly one scoring site in the file (grep: `indexer_scores_pallas` /
`paged_indexer_scores` appear only inside `_score_and_select`). Both backends receive
identical operands: `bt_tok = bt_req[req_ids]` (per-token block-table rows) and
`kv_lens_tok = positions + 1` (per-token causal). So the gate provably covers decode and
masked-prefill scoring with the same kv_lens semantics — there is no second path to miss.

**Contract refutation attempt** — I tried to construct a prefill shape that violates
`indexer_scores_pallas`'s grid/contract and failed; the kernel is per-ROW, not
per-request:
- `_check_scoring_shapes` demands `q [R,H,D] / w [R,H] / bt [R,max_blocks] / kv_lens [R]`
  — all satisfied with R:=T (token rows). No assert or index map depends on rows being
  distinct requests; the row tile is 1 (any T legal); the only compiled-mode assert is
  `page_size % 128`, T-independent.
- the genuinely-new semantic surface — *multiple rows sharing one physical page with
  DIFFERENT causal `kv_len`s* (same-request chunk tokens, impossible at decode shapes) —
  is handled per-row by the `kv_abs < kv_len[r]` mask.

**Probe** (`scratchpad/probe_prefill_pallas_scorer.py`, CPU/interpret): ragged 2-request
batch, T=8 token rows (req0: 5 history + 3 chunk tokens at pos 5-7 sharing one page;
req1: 4 from-scratch tokens; 1 pad row), post-write cache. Results: (a) -inf/validity
maps bit-identical across `indexer_scores_pallas` / `paged_indexer_scores` /
`indexer_scores_xla`; (b) finite fp32 scores equal (rtol 1e-6); (c) same-page per-token
causality holds (pos-5 token sees exactly 0..5, its page-mates' later keys -inf);
(d) cross-request page isolation (perturbing req0's pages leaves req1 rows bit-equal);
(e) `hierarchical_topk` selected sets equal across backends. ALL PASSED.

**Wrapper-level cross-run**: the entire sparse-prefill suite re-run under
`GLM_DSA_SCORER=pallas` — 11/12 pass including the per-token selected-set reference
gates, multi-chunk-equals-single-shot, mixed-step routing, and outer-jit tests; the
single failure is F2's instrumentation counter reading 0 (which is the gate *working*).
The pallas_decode suite covers the decode side of the same gate natively (scorer-
parametrized tests). Remaining risk is exclusively F1's compiled-mode resource budget.

### (2) Preseeded-MTP dispatch

**Preseeded step can NEVER reach the masked-prefill branch**: the dispatch split is a
trace-time Python `if preseeded:` (mla_attention.py:1233-1254) — the preseeded program
traces only `lax.cond(sparse_step, _sparse_decode_branch, _dense_fallback_branch)`;
`_sparse_prefill_branch` / `_prefill_or_mixed_branch` are *not in the traced graph*, and
`_score_and_select` cannot run either (`is_full_layer` is False under preseed, :1029).
The dummy -1 seed forces `sparse_step` False → dense. The discriminator
`jnp.any(seed != INVALID_INDEX)` is faithful: the step-0 stash is whole-batch
all-real-or-all-dummy (single `lax.cond` at :1133), a real live row can never be all -1
(`kv_len >= 1` ⇒ ≥1 valid index), and the proposer's pad-row clones
(`eagle3.py:236-239`) replicate live rows. Pinned by
`test_dummy_seed_takes_dense_fallback` (poisoned sparse kernel + dense marker) and
`test_steps_ge1_reuse_seed_poisoned_scorer` (scoring/query/head-weight fns poisoned on
seeded traces). `topk_context_preseeded` is context state
(`dsa_topk_seeded=bool(dsa_topk_indices)`, wrapper_context.py) — deliberately distinct
from "stash non-empty", so a target forward's own within-step stash never demotes later
full layers.

**Non-preseeded prefill during spec-decode target steps takes the right branch**: the
target step function (`vllm_model_wrapper.py:747`) sets the context *without*
`dsa_topk_indices` → `dsa_topk_seeded=False`; only `draft_step_fun_impl` (:817-822)
seeds, and only when the proposer passes `shared_topk_indices` (loop steps ≥ 1,
`eagle3.py:735-746`). A verify step schedules >1 token/request, and
`_reorder_batch` counts a request as decode strictly on `num_scheduled_tokens == 1`
(persistent_batch_manager.py:54-84) → `request_distribution[0] < [2]` →
`glm_dsa_is_pure_decode` False → `_prefill_or_mixed_branch` → masked-sparse iff some
live token's ctx > topk, else dense (= the sparse result at ctx ≤ topk). Verify-token
indexer keys ARE written (full layer, non-preseeded), rejected positions fall beyond
`seq_len` and are masked/overwritten later — the same argument as the draft-loop
skip-write comment. Index-share suite: 14P baseline AND 14P under
`GLM_DSA_SCORER=pallas` (the preseed × scorer-gate composition).

**Merge-resolution delta vs the G4 parent, accounted for**: parent's cond returned
`(cache, out, indices)` with scoring inside the decode branch; the resolution keeps
sparse-prefill's hoisted 2-tuple conds and pre-computed `indices`, and re-expresses the
preseeded gate on the hoisted `is_decode_step`. Net semantic change vs parent: a
non-preseeded full layer now stashes REAL indices on sparse-prefill steps (dummy only on
dense-routed steps) — that is sparse-prefill's intended supersession, and it is what
makes a prefill-shaped draft step 0 emit a usable seed (better than the parent's
dummy-on-any-non-decode). The F3 pad-row guard (`req_ids < request_distribution[2]`,
:1031-1043) survived intact, applied before every latent/key write on seeded traces
(`test_seeded_step_drops_pad_row_writes` passes).

### (3) The hoisted DP guard (round-1 merge c6492178)

`tpu_inference/layers/common/attention_interface.py` is **byte-identical** from
`c6492178` through `82fe3f76` (`git diff` empty), and none of the three round-2 parent
branches touched the file (no-op for every merge — nothing to drop). Guard placement
re-verified in the merged tree: the round-5 S1 guard is a standalone `if _head_sharded:`
block (:665-695) evaluated BEFORE the three-way spec dispatch, so it fires for both
head-sharded combos — `(HS, no-DCP)` via `elif _head_sharded:` (:722) and `(HS, DCP)`
via `if _head_sharded and _dcp:` (:697); for reference, in the round-1 parents the DCP
side (`70aa6825`) had NO guard and the 2int side (`025bdd25`) had it fused into its
single spec block — the hoist is exactly what makes the composed branch guarded. The
third combo `(no-HS, DCP)` is deliberately unguarded: it keeps the default token-sharded
in_specs (q over MLP_TENSOR, md `P(ATTN_DATA)`) — the DP-compatible layout — and the
guarded axes (data/attn_dp) are disjoint from the dcp context axis. The 2D-scheme
string-vs-tuple normalization (`_axes_tuple`) is present in both this guard and the
pallas-path M1 guard (mla_attention.py:951-953). Suites: `test_mla_head_sharded.py` +
`test_mla_dcp.py` = **48P** on the merged tree, including
`test_head_sharded_refuses_dp_mesh`.

### (4) Silently dropped hunks — none

Mechanical re-merge of every touched file in all three merges (`git merge-file -p`
against each merge's base `53c5e5ee`, git 2.34 has no `--remerge-diff`):

- `f5f10b63`: 3 files, zero conflicts, auto-merge == committed tree **byte-for-byte**.
- `f095a8ca`: 4 files; only `mla_attention.py` conflicts (2 hunks: the pallas-decode
  docstring, and the scorer body). Resolution = P2's hoisted `_score_and_select`
  structure + P1's `use_pallas_scorer` branch grafted verbatim inside it (comment
  included); P1's superseded in-branch scoring carried nothing else (its
  `hierarchical_topk` call and `shared_indices` else-arm both exist in the hoisted
  structure). Every non-conflict line == auto-merge.
- `534cd74d`: 14 files; only `mla_attention.py` conflicts (1 hunk: the dispatch tail).
  Resolution = P1's two 2-tuple conds + P2's preseeded gate re-based onto the hoisted
  `is_decode_step`, comment extended with the "never the masked sparse-prefill branch"
  clause. P2's 3-tuple `indices` output is correctly subsumed by the hoisted pre-cond
  `indices`.
- Reverse-direction hole closed: for each merge,
  `comm -23 <(diff base..P2 names) <(diff P1..M names)` is **empty** — no branch-side
  file is missing from any merge.

### (5) Suite re-runs (all CPU, `JAX_PLATFORMS=cpu`, worktree @ 82fe3f76)

| run | result |
|---|---|
| `test_glm_dsa_pallas_decode.py` | **18 passed** (169 s) |
| `test_glm_dsa_sparse_prefill.py` | **12 passed** (44 s) |
| `test_glm_dsa_mtp_index_share.py` | **14 passed** (69 s) |
| `test_glm_dsa_sparse_prefill.py` under `GLM_DSA_SCORER=pallas` | 11 passed + 1 failed (F2 instrumentation only: `assert score_calls["n"] == 1` sees 0) |
| `test_glm_dsa_mtp_index_share.py` under `GLM_DSA_SCORER=pallas` | **14 passed** |
| `test_glm_mtp_stage3.py` (534cd74d's other suite) | **21 passed** |
| `test_mla_head_sharded.py` + `test_mla_dcp.py` | **48 passed** |
| combined single session: index_share + mla_attention + pallas_decode + sparse_prefill | **50 passed** (4:39) |

**82fe3f76 specifically verified both ways**: (a) pre-fix repro — the `534cd74d`
versions of the two victim test files, collected in one session with the (unchanged)
index_share module, fail exactly the two claimed tests with exactly the claimed
signature: `ValueError: Resource axis: expert of P(('model','expert','dcp'),) is not
found in mesh: ('data','attn_dp','model')`; (b) post-fix, the same combined collection
passes 2/2, and the full 4-suite session passes 50/50. Mechanism inspected: the fixture
monkeypatches pin `ShardingAxisName._cls = ShardingAxisName2D` + fresh `_overrides` for
the test's duration only (resolver consults `_cls` per access, sharding.py:127-130), so
the pin is order-independent and cannot leak into the base-scheme suites.

---

## Caveats / not covered here

- Everything above is CPU + interpret mode. F1 aside, the compiled-Mosaic acceptance of
  the 2b scorer (the `(1,H)` matmul LHS, `[1,P]` output tile, SMEM table) and the 2c
  kernels remain the on-metal gates of task #9 — unchanged by this review.
- bf16 device caches: prefill selections under the pallas scorer inherit the documented
  S2 boundary-band (not index-exact) contract; the sparse-prefill suite's oracles ran
  fp32 (test caches) plus the suite's own bf16 outer-jit test. No new bf16-specific gate
  was added for prefill×pallas — acceptable while `xla` is the default, should be part
  of flipping the default.
- `unquantized.py`, `model_loader.py`, `glm_mtp/*`, `vllm_model_loader.py`,
  `vllm_model_wrapper*.py`, `eagle3.py` in `534cd74d` auto-merged clean (byte-identical
  to the G4 parent's home-branch-tested versions) and are covered by the stage3 +
  index_share suites; they were not line-audited beyond the dispatch/seeding paths
  quoted above.
