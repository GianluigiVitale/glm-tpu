# Round-7 adversarial review — Stage-2 end-to-end integration (`GLM_DSA_MODE=pallas_decode`)

**Reviewer scope.** `~/tpu-inference-2int`, branch `glm-5.2-v4-2int`, range
`c780caa6..53c5e5ee` (4 commits: 2a.2 paged indexer k-cache + its merge, the Stage-2
sparse-decode integration `c374eba7`, and the review-fix commit `53c5e5ee`). Attack
surface as tasked: (1) the `lax.cond` pure-decode dispatch semantics + the traced cost
of the non-taken branch, (2) latent write-then-attend vs the dense kernel's fused write,
(3) indexer-cache functional-update swap-back / lost updates, (4) IndexShare stash
across the cond boundary, (5) full battery re-run + one adversarial case of my own.

All verification CPU-only (`JAX_PLATFORMS=cpu` before interpreter start, Pallas
interpret; **no TPU touched**). vLLM claims pinned to the installed
`vllm 0.1.dev1+ga30addc75` (`vllm-env`).

**Worktree state caveat.** `~/tpu-inference-2int` HEAD *is* `53c5e5ee`, but the working
tree carries **uncommitted WIP** in exactly the two target files (`glm_dsa_indexer.py`,
`mla_attention.py`: `precompute_indexer_params` PWAL hook + a new `GLM_DSA_SCORER`
gate — the M4 follow-up, +156 lines). That WIP is **outside this review's range**. I ran
the battery twice: once on the dirty tree, once on a pristine `git archive 53c5e5ee`
extraction. Anything merged to the pod branch must be the committed range, not the tree.

**Test runs (executed by this review):**

- Dirty tree: `test_glm_dsa_pallas_decode.py` → **10 passed**;
  `test_glm_dsa_indexer.py` (incl. the `GLM_DSA_MODE` off/unset **byte-identity
  sha256 hash gate**) → **32 passed**; `test_dsa_sparse_mla.py` +
  `test_dsa_indexer_kernel.py` → **66 passed**. 108/108 green.
- Pristine `53c5e5ee`: `test_glm_dsa_pallas_decode.py` + `test_glm_dsa_indexer.py`
  → **42 passed** (result recorded after re-run on the archived tree).
- My adversarial case (finding 8, script `adversarial_pallas_decode.py`): **passed**.
- Traced-buffer census (finding 1/2, script `branch_census.py`): numbers below.

---

## Summary verdict

**Correctness survives attack.** I could not construct a wrong-answer or
cache-corruption case on CPU: the dispatch predicate is exactly "every scheduled
request has exactly one token", the pad-token discipline (`valid` mask +
`mode="drop"` slot diversion) held up against engineered slot-aliasing garbage, the
IndexShare stash is a legal cond-output tracer, and the multi-request
sparse-vs-dense-restricted parity is exact at fp32 tolerances. The review-fix commit
`53c5e5ee` correctly fixes the 2D-axis-name guard no-op that round 6 flagged.

**The real findings are cost, not math.** The sparse branch's work scales with the
**padded token bucket**, not the live request count — and both cond branches are
compiled into every bucket's executable. Under the launcher's
`TPU_MIN_TOKEN_BUCKET=512` this makes the *taken* decode branch do ~64× redundant
gather/scoring work per step at smoke scale (finding 1), and parks multi-GiB traced
buffers in prefill-bucket executables for a branch that never runs there (finding 2).
Plus one aliasing question only a TPU HLO dump can settle (finding 3) and a Stage-3
design collision (finding 4). None of these block a *gated* merge (`GLM_DSA_MODE`
default `off` keeps the dense path byte-identical — hash-gate re-verified), but
findings 1–3 should be treated as blockers for *enabling* `pallas_decode` on the pod.

---

## Finding 1 — MAJOR (perf, the TAKEN branch): sparse-branch cost scales with the padded token bucket, ~64× redundant at the launcher's config

`_glm_dsa_pallas_decode_attention` builds every per-token map at width
`T = padded_total_num_scheduled_tokens`. The runner pads *every* step — pure-decode
steps included — to the token bucket floor (`tpu_runner.py:736-752`), and the GLM
launcher sets `TPU_MIN_TOKEN_BUCKET=512` (required for mla.v2 shard_map divisibility,
docs/04). So a decode step with 8–16 live requests executes the sparse branch at
**T=512**:

- `gather_kv_segment` materializes `seg_kv [512, 2048, 640] bf16 = 1.25 GiB` **per
  sparse layer per decode step** (write + scattered source reads ≈ 2.5 GiB of HBM
  traffic each), replicated per chip (`P()` in-specs). Across ~78 sparse MLA layers
  that is O(100 GiB) of per-chip HBM traffic per decode step — the sparse path as
  integrated is almost certainly *slower than the dense path it replaces* until this
  is fixed. At the *real* row count (8 reqs) it would be 20 MB/layer — fine.
- `paged_indexer_scores`'s `lax.map` gathers a **per-token page copy**
  `k_cache[page_ids] [T, page, 128]` per block step: per full layer per decode step
  it reads `T × max_blocks×page × 128 × 2` bytes — at T=512, 8K-wide tables that is
  ~1.07 GiB vs ~17 MiB of per-request useful reads (the 512/8 padding ratio, again
  64×), ×22 full layers.
- The scoring walk also runs to `max_blocks` (max_model_len) for every token
  regardless of actual `seq_lens` — inherent to the fixed-shape map, but it compounds
  the above for short sequences.

**Refutation attempted:** re-read the bucket derivation (`get_token_paddings` with
`min_token_size=max(512, …)`; `get_padded_token_len` picks the first bucket ≥ n) —
there is no decode-specific smaller bucket; and there is no row-slicing anywhere in
the branch. The numbers are from the jaxpr census below, not estimates.

**Cheap fix (correctness-preserving):** on a pure-decode step, after the decode-first
reorder, token `i` belongs to request `i` (`query_start_loc` is an iota), so
everything from scoring onward can operate on the **static** leading
`min(T, num_seqs)` rows (`num_seqs = md.seq_lens.shape[0]` = padded max_num_reqs, a
trace-time constant), and the `[T, topk]` indices output can be rebuilt by -1-padding
rows ≥ `num_seqs`. That bounds the branch by max_num_reqs instead of the token
bucket. (The indexer *key write* must stay at full T — prefill steps need it.)

## Finding 2 — MAJOR (compile/HBM, the NON-taken branch): both cond branches are compiled per token bucket; the sparse branch at prefill shapes parks GiB-scale buffers

`lax.cond` stages *both* branches into every bucket's executable; the predicate is
runtime (`request_distribution` is a tracer — correctly so). So every **prefill**
bucket also contains the sparse branch traced at that bucket's T. Jaxpr census of the
sparse branch built from the real repo functions at GLM-5.2 geometry (lkv=512, r=64,
width=640, topk=2048, idx 64×128, page=512 — `branch_census.py`):

| bucket | `seg_kv` gather | score map (f32, 32K ctx) | top-k sort bufs | branch Σ (naive) |
|---|---|---|---|---|
| T=512 | **1.25 GiB** | 64 MiB (×2–3 copies) | ~96 MiB | ~3.9 GiB + cache-sized scatter |
| T=1024 | **2.50 GiB** | 128 MiB (×2–3) | ~192 MiB | ~7.4 GiB + " |
| T=2048 | **5.00 GiB** | 256 MiB (×2–3) | ~384 MiB | ~14 GiB + " |
| T=512, 128K ctx | 1.25 GiB | **256 MiB ×2–3** | 384 MiB | ~5.5 GiB + " |

(Σ is the sum of intermediates; XLA reuses by liveness, so the *peak* is dominated by
the largest simultaneous set — `seg_kv` + its gather source + kernel operands, i.e.
roughly the `seg_kv` column ×2–3.) XLA's temp reservation for a conditional is ≈ the
max over branches, and MoE activation peaks at these T are hundreds of MiB — so the
never-taken sparse branch **raises the executable's HBM reservation by GiBs** in
every bucket ≥1024 tokens (production chunked-prefill configs use 2048–4096). Effect:
either the KV-cache budget silently shrinks (if the profiling run sees the peak) or
compile-time/serving OOM at the large buckets.

**Refutation attempted:** XLA cannot DCE the branch (runtime predicate); branch temp
buffers *are* shared with the rest of the graph by liveness, which is why this is a
reservation-peak claim, not an allocation-sum claim. The smoke config
(`max_num_batched_tokens=512` → a single T=512 bucket) keeps this at ~1.25–3 GiB —
survivable, which is presumably why nothing has blown up yet.

**Fix:** the finding-1 slice fixes this too (branch bounded by max_num_reqs rows
regardless of bucket). A trace-time `if T > num_seqs: dense-only` is **not** safe
alone — with the 512 bucket floor, pure-decode steps land in T=512 > max_num_reqs.

**TPU-only:** exact per-bucket reservation:
`jit(step).lower(...).compile().memory_analysis()` + HLO buffer-assignment dump,
per bucket, `GLM_DSA_MODE=pallas_decode` vs `off`.

## Finding 3 — MAJOR-risk, TPU-only verifiable: latent-cache in-place aliasing through the cond

The per-layer latent cache enters `lax.cond` as an operand of **both** branches
(sparse: XLA scatter `flat.at[slots].set`; dense: Pallas kernel with
input–output aliasing inside `shard_map`), and the updated cache is a cond output
swapped back into the donated `kv_caches` list. The census shows the scatter
producing a **full-cache-sized** value inside the branch (`scatter (num_slots, 640)`
+ a chain of full-cache reshapes). If XLA copy-insertion fails to alias the donated
buffer through the conditional (conditionals are exactly where it is most
conservative), every layer copies its entire latent cache every step — with the KV
pool sized to fill HBM, that is ~2×(HBM/99) of traffic per layer per step (step-time
explosion) plus a transient double allocation. The indexer-cache scatter sits
*outside* the cond and should alias cleanly; the latent one is the question.

**Refutation attempted:** none possible off-TPU — CPU tests never see TPU buffer
assignment, and the interpret-mode kernel path doesn't even lower the dense branch.
**TPU gate:** dump HLO for the decode bucket and grep for `copy` ops on the 640-wide
cache buffer (and confirm `input_output_alias` covers the kv_caches list through the
conditional). This must be part of the first on-TPU checklist, before any benchmark
is trusted.

## Finding 4 — MEDIUM (design collision with Stage 3): spec decode permanently disables the sparse path

The predicate counts a request as decode iff `num_scheduled_tokens[req] == 1`
(`persistent_batch_manager._reorder_batch`, `tpu_runner.py:2571`). Speculative decode
(MTP — task Stage 3) schedules `1 + num_draft_tokens` per decode request, so **every
step of an MTP-enabled server classifies as mixed → dense fallback; the sparse path
never runs**. Not a correctness bug (dense handles it, and the indexer keys still get
written), but Stage 3 as planned would silently discard Stage 2's entire win. Needs
either a multi-token-per-request sparse decode variant (the 2c kernel is R-row-based
and could take `R = Σ tokens` with per-token indices) or a loud config-time warning
when `speculative_config` and `GLM_DSA_MODE=pallas_decode` coexist.

**Refutation attempted:** checked whether the reorder counts spec reqs as decode
(it does not — draft tokens are included in `num_scheduled_tokens`), and whether
the distribution is built anywhere else (compilation_manager warmups use `[0,0,0]`
→ dense — consistent).

## Finding 5 — MINOR (documented asymmetry, sharpened): 1-token prefill chunks make prompt latents chunk-boundary-dependent

M2's note is correct that a 1-token chunk classifying as decode is *safe* (keys are
written every step; prior chunks' latents were written dense; `kv_len = pos+1` covers
the full history — my finding-8 test exercises numerically identical geometry). The
sharper consequence it doesn't state: at ctx > topk that mid-prompt token's attention
output is top-k sparse, feeds the next layer's hidden state, and therefore **that
position's `kv_c` in every subsequent layer differs from the dense-prefill result —
permanently, in the cache, consumed by all later prompt tokens**. The same prompt
now yields different cache contents depending on scheduler chunking, and a
prefix-cache hit can splice the two regimes into one sequence. More DSA-faithful,
yes — but it breaks bitwise reproducibility of prefill and will confound
dense-vs-sparse A/B parity debugging. Recommend a step counter/log when a 1-token
chunk takes the sparse branch, so a numerics investigation can rule it in or out.

## Finding 6 — verified fixed, one residual test hole: the 2D-axis-name guard no-op

Confirmed at source (`sharding.py:71-116`): the lazy resolver's **default** scheme
(no `NEW_MODEL_DESIGN`/`USE_2D_TP`) is `ShardingAxisName2D` with **plain-string**
axis names, so the pre-fix S1 guard's `set(ShardingAxisName.ATTN_DATA)` decomposed
`'data'` into characters — a real no-op, exactly as `53c5e5ee` says. Mitigating
context: the pod launcher sets `NEW_MODEL_DESIGN=1` (tuple scheme), so the S1 guard
*did* function in production; the exposure was the default scheme. `_axes_tuple` in
both guards is correct for both schemes. Residual: the pallas-path guard is tested
under the default 2D scheme (test (f) runs with no scheme override), but the
*attention_interface* S1 guard's only DP-refusal test
(`test_mla_head_sharded.py::test_head_sharded_refuses_dp_mesh`) pins the **base**
scheme via the `base_sharding` fixture — the configuration that was broken (2D
strings) still has no regression test for that guard. One-line test to add.

## Finding 7 — MINOR (inherited cost, being addressed out-of-range): per-step indexer weight adaptation

`_glm_dsa_indexer_params` (fp8 `wq_b` dequant + fp32 casts/transposes) executes
inside the jitted step on the decode hot path — M4 documents it (~0.75 GB/step
across 22 full layers). The uncommitted worktree WIP (`precompute_indexer_params`
at PWAL) is the right fix but is **not in the reviewed range**; whoever merges must
not conflate the two. Until it lands, pallas_decode benchmarks will under-report
the achievable decode rate.

## Finding 8 — adversarial case they missed: constructed, run, PASSED (refutation evidence)

Their suite never combines, in one pure-decode batch: a truly sparse request
(ctx > topk) with a complete one (ctx < topk) — (a2) is single-request; and pads are
always benign (`hidden=0, pos=0`), whereas production pads are **stale ring-buffer
contents**. Constructed (`adversarial_pallas_decode.py`, scratchpad): topk=6; req0
driven to ctx 20 (crossing PAGE=8 boundaries at 8/16, truly sparse), req1 joining at
step 13 (ctx ≤ 8, complete, -1-padded gather rows in the same batch); pad rows carry
**nonzero garbage hidden ×3.0** and stale positions **engineered to alias req0's
next write slot**, the padded metadata row given req0's real block-table row (worst
case if a pad write leaks), plus one pad position 1000 that **overflows the
block-table width** (`pos // 8 = 125 > 7` — exercises OOB `take_along_axis` inside
`indexer_cache_slots` on a dropped token). Per step per live row: output == fp64
dense reference restricted to exactly the stashed selection; selection == the 2a.1
single-chunk reference. Post-run: **byte-level audit of both caches against the
exact expected histories** (any pad leak or clobber would show). Result: **all
checks passed** — the `valid`-mask + `mode="drop"` discipline and the searchsorted
clip behave exactly as documented under hostile padding.

## Attack log — refuted (no finding)

1. **Dispatch semantics.** Predicate `dist[0]==dist[2] ∧ dist[2]>0` == "every
   scheduled request has exactly 1 token" — matches both constructions
   (`_reorder_batch` fast path and two-pointer path; `tpu_runner.py:2564-2576`).
   1-token chunked-prefill tails → sparse (finding 5); any multi-token chunk
   anywhere → dense as a whole; warmup/dummy `[0,0,0]` → dense; spec decode → dense
   (finding 4). Under DP the distribution is per-rank-raveled — refused loudly at
   trace time by the new guard (test (f); axes normalization verified, finding 6).
2. **Double-write between branches.** None: `lax.cond` executes one branch; the
   sparse branch's latent scatter replaces exactly the write the dense kernel would
   have fused (round-5 F2), and the *indexer* write is pre-cond by design
   (decode-after-prefill needs keys from prefill steps — pinned by test (d), which
   also proves the pad-slot write is dropped). Pad columns: sparse writes zeros,
   dense may leave stale pad lanes — inert either way, the kernel's q-side zero-pad
   nulls cache pad columns (gate in the 2c suite).
3. **Indexer-cache swap-back / lost updates.** Each layer touches only its own two
   `kv_caches` indices; layers run sequentially inside one wrapper context; the
   idiom (read → functional update → list swap-back) is byte-for-byte the
   `VllmMLAAttention.forward` discipline, and the step function returns
   `ctx.kv_caches` with `donate_argnames=("kv_caches",)`
   (`vllm_model_wrapper.py:637-746`). No aliasing of layer names → no lost update.
   Production `query_start_loc` padding (repeat-total, `tpu_runner.py:2525`) matches
   `token_request_ids`' documented assumption; block_tables/seq_lens row counts
   agree (`attention_metadata.py:39-45`); the merged-spec design keeps ONE kv-cache
   group → one block table (re-verified via the 2a.2 spec tests in the 32-test run).
4. **IndexShare across the cond.** The stash writes a **cond output** (a legal
   tracer of the enclosing trace), not a branch-internal value; the shared layer
   fetches outside the cond and closes over it inside its own branch (an ordinary
   cond operand). Dense steps stash `[T, topk]` `-1` dummies whose only consumers
   are the equally-untaken sparse branches of shared layers in the *same* step
   (the predicate is step-global). Branch output shapes/dtypes match. The outer-jit
   test (e) pins exactly this regime (stash-of-cond-output under `jax.jit` with
   torchax weights); my finding-8 run exercised it for 20 more steps.
5. **`ctx <= topk` unbranched exactness** — held in the multi-request mixed case
   (finding 8), not just their single-request gates.

## What only the TPU can verify (ordered gate list for the first on-TPU session)

1. **Dense-write → sparse-read layout agreement.** On CPU the dense mla.v2 callee is
   *always* mocked (it cannot lower), so no test anywhere runs the real sequence
   "prefill writes latents via the fused kernel → decode step sparse-gathers them".
   The claim that the kernel writes `[kv_c | k_pe | pad]` token-major at
   `bt[pos//page]*page + pos%page` in the packed layout is Stage-1-era knowledge,
   not a Stage-2 test. Gate: one real prefill + one real decode step, dump the row
   at a known (page, slot), compare against `compute_indexer_keys`/`kv_c` refs —
   then a full dense-vs-sparse decode logit diff at ctx ≤ 2048 (must be ~exact).
2. **Finding 3:** HLO copy/aliasing audit of the cache through the cond.
3. **Finding 2:** per-bucket `memory_analysis()` with mode on vs off.
4. Mosaic (non-interpret) `dsa_sparse_decode` inside shard_map inside cond inside
   the donated-buffer jit — compiles and runs on the v4 mesh (the cond-branch
   output sharding join between the shard_map'd sparse out and the dense kernel out
   is also where GSPMD may insert resharding — check the HLO while there).
5. bf16 end-to-end numerics at real scale (CPU (e) covers a toy bf16 case only) and
   the S2 selection-boundary behavior on long real prompts.

## Verification appendix

- Suites: commands as in round 6, `vllm-env`, `JAX_PLATFORMS=cpu`. Dirty tree:
  10 + 32 + 66 = 108 passed. Pristine `53c5e5ee` archive: 42 passed
  (pallas_decode + indexer files; kernel files unchanged by the range).
- Scripts (scratchpad, session `8712a40e`): `branch_census.py` (jaxpr walk of the
  cond's sparse branch, real repo functions, GLM-5.2 geometry),
  `adversarial_pallas_decode.py` (finding 8).
- Nothing in either worktree or the TPU was touched; the only write is this file.

*Round-7 adversarial review; CPU-only, no TPU touched.*
