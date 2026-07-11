# Two upstream-worthy findings from the GLM port (owner briefing)

*Prepared 2026-07-11. Both packages live in this directory; the owner files —
nothing has been submitted.*

## 1. Block-table granularity double-multiplication (dcp>1 silent corruption) — fixed upstream as #3129; we hold convergent forensics

At dcp>1, the kv-cache spec pre-multiplied `block_size` by dcp (introduced by
upstream PR #2398) **and** vLLM's engine core multiplied it by dcp again, so
the engine allocated one block id per `B x dcp^2` tokens while the TPU stack
consumed the table at `B x dcp` tokens/entry — every table entry >= 1
dereferenced an unallocated/stale page. Silent corruption (never a shape
crash); shipped in releases v0.20.0–v0.24.0. We root-caused it independently
with live predictions (<=1-page prompts pass; corruption threshold exactly
L=1025 at dcp=2; donation- and scatter-impl-inert) and fixed it in the fork as
`1f700c507` — upstream's #3129 (weiyu0824, merged 2026-07-09 12:04 PT, ~3h
before our fix) implements the same two-sided contract, so **do not file the
corruption report**. Still upstream-worthy from our audit of post-#3129
`main`: two residual granularity holdouts (routed-experts slot reconstruction,
`tpu_runner.py:757/:422-431`; KV-connector insert/extract,
`kv_cache_manager.py:1278,1292`), the fact that **no dcp>1 test exists**, and
a loud engine-vs-TPU geometry assert that would have caught five releases.
Evidence: `docs/RESEARCH_LOG.md` 2026-07-09 21:30 / 21:45 / 22:05 and
2026-07-10 00:05; full memo `docs/upstream/dcp-block-granularity-report.md`.

## 2. Pageloop v4 lowering defect: silent sublane row-stripe drop in a donated paged cache — ours alone

On TPU v4, the per-page RMW scatter formulation (`fori_loop` of
`dynamic_index_in_dim -> jnp.where masked merge -> dynamic_update_index_in_dim`)
into a donated, shard_map-resident `[pages,16,32,128]` bf16 paged cache
silently never stores contiguous sublane row-stripes of some pages (observed:
rows `{0,1} u {8,9}` of 16 = first 2 sublanes of each 8-sublane tile,
identical coordinates on all 8 hosts; geometry varies per executable). The
holes retain stale HBM, so expression is a launch-state lottery (zeros ->
constant scores; foreign garbage -> structured wrong values; same-layout ->
invisible) and corrupted runs **passed end-to-end needle accuracy checks**.
The traced logic is exonerated on CPU (sentinel suite, all 4 formulations x
dcp={1,2}, 16/16 + 28/28), and the same writes as a flat 1-D scatter are
byte-complete on v4 under the scrambler byte-diff protocol — indicting the
lowering. Fix shipped: flat default behind its own env (`4f7d9a001`, review
`74d8c3225`; sentinel suite `e1b666382`,
`tests/layers/vllm/test_glm_dsa_idx_write_coverage.py`; doctrine in the
`_glm_dsa_dcp_owner_scatter` docstring). Evidence: `docs/RESEARCH_LOG.md`
2026-07-10 16:10 through 2026-07-11 09:55. Filing package (generic
"paged KV-cache owner-scatter" framing, no model-proprietary detail):
`pageloop-v4-sublane-drop-REPORT.md` + `pageloop-v4-repro.py`
(CPU-verified: prints COMPLETE on all impl x dcp combinations, jax 0.10.1).
Suggested venue: tpu-inference tracker first (the pattern is the repo's own
dcp pageloop scatter class), likely forwarded to JAX/XLA:TPU — note the
mirror-image history (dense path: plain scatter mislowers, pageloop is the
fix; this path: pageloop mislowers, flat is the fix) as evidence the defect
class is in XLA's lowering of masked partial stores into donated sharded
tiled buffers, not in any one formulation.
