# Upstream dcp>1 block-granularity status — investigation memo (DO NOT FILE)

**Date:** 2026-07-09
**Upstream ref audited:** `vllm-project/tpu-inference` `main` = `a3fe6eec1d6812abb7d95f82e9d143de27d0534f`
(verified against the live remote via `git ls-remote https://github.com/vllm-project/tpu-inference.git main` — SHAs match).
**vLLM engine side verified at:** `~/vllm-build` @ `a30addc7548a` (`v1/core/single_type_kv_cache_manager.py:66-70`,
`v1/kv_cache_interface.py:114-115`, `v1/engine/core.py:283-289`).

## Verdict: NOT a live bug on upstream `main` — do not file the corruption report

The dcp>1 block-granularity silent-corruption bug (the exact class we root-caused and fixed in our fork as
`1f700c507`) **was live upstream**, shipped in **five releases (v0.20.0 through v0.24.0)**, and was **fixed on
`main` hours before this investigation** by upstream PR **#3129** (`631d72c24`, "fix: Correct block size
calculation for context parallelism", weiyu0824, merged 2026-07-09 12:04 PT). Upstream `main` HEAD
(`a3fe6eec1`, 15:17 PT the same day) contains the fix. Filing a "live silent corruption at dcp>1" issue
against `main` would be factually wrong and would burn credibility. No fix diff is drafted, because upstream's
merged fix already implements the same contract as ours.

## Timeline

| Date | Event |
|---|---|
| 2026-04-30 | `f940073ea` **#2398** "Add DCP sharding axis and KV cache support" lands — introduces the buggy spec pre-multiply (`block_size *= parallel_config.decode_context_parallel_size` in `get_kv_cache_spec`). Our fork branched from this code. |
| 2026-05 → 2026-07-02 | Releases **v0.20.0, v0.21.0, v0.22.1, v0.23.0, v0.24.0** all contain #2398 and none contain the fix (`git tag --contains f940073ea` / `--contains 631d72c24`). Verified directly in `v0.24.0`: pre-multiply at `tpu_inference/runner/kv_cache_manager.py:445`, **no** `physical_block_size` scaling in `runner/kv_cache.py`, raw spec size in `maybe_reinitialize_input_batch` (kv_cache_manager.py:674-676). |
| 2026-07-09 12:04 PT | `631d72c24` **#3129** merged: removes the spec pre-multiply, adds the physical ×dcp, scales the runner block table ×dcp. |
| 2026-07-09 15:17 PT | `main` HEAD `a3fe6eec1` (audited here). |

Note: in the released (buggy) versions the failure was **silent corruption, not a shape crash** — the
allocated cache dim-1 was `spec.block_size` tokens = `B×dcp` (pre-multiplied), which always divides evenly by
the `dcp` CONTEXT axis (RPA layout dim-1 is raw `page_size`, `kernels/ragged_paged_attention/v3/kernel.py:261-274`;
MLA dim-1 is `page/packing` with `B` a multiple of packing). Engine ids covered `B×dcp²` tokens vs. physical
pages of `B×dcp` → corruption past the first `B×dcp` tokens of a request (matches our fork's >1024-token
threshold at dcp=2), exactly our bug class.

## Units audit of `main` (post-#3129): tokens covered per block-table entry

With `spec.block_size = B` (raw, per `NOTE(weiyu0824)` at `runner/kv_cache_manager.py:445-447`) and dcp = `d`:

| Consumer | Tokens per entry | Where | Consistent? |
|---|---|---|---|
| vLLM engine allocator | `B×d` | vLLM `v1/core/single_type_kv_cache_manager.py:66-70` (`self.block_size *= dcp_world_size`); `storage_block_size` property returns raw `block_size` (`v1/kv_cache_interface.py:114-115`) | baseline |
| Runner InputBatch / block table | `B×d` | `maybe_reinitialize_input_batch` rebuilds InputBatch with `spec.block_size × context_cnt` (`runner/kv_cache_manager.py:676-702`); block tables copied verbatim to device (`runner/tpu_runner.py:2572-2603`) — no other dcp transform | YES |
| Physical KV page (global) | `B×d` | `initialize_kv_cache` passes `layer_spec.storage_block_size` = `B` (`runner/kv_cache_manager.py:893,938-947`) into `create_kv_caches` → `get_kv_cache_shape_with_mesh` multiplies internally: `physical_block_size = block_size * context_cnt` (`runner/kv_cache.py:66-67`, used at :80 MLA and :92 RPA) | YES |
| Attention kernels (RPA + MLA) | `B×d` | Kernels derive page size from the cache array they receive. Both shard_maps declare dim-1 **unsharded**: RPA `kv_cache_spec = P(ATTN_DATA, None, ATTN_HEAD, None, None)` (`layers/common/attention_interface.py:406-407`); MLA `P(ShardingAxisName.BATCH)` in/out (`attention_interface.py:576,583`). The creation-time `P(BATCH, CONTEXT)` sharding (`runner/kv_cache.py:142,146`) is therefore resharded away at the first attention call (jit inserts the collective; no error), and each device's kernel sees full `B×d`-token pages with engine ids in `B×d` units | YES |
| Routed-experts slot reconstruction (opt-in) | **`B`** | `_reconstruct_slots_for_request` uses `runner.block_size` = raw `cache_config.block_size` (`runner/tpu_runner.py:757`) at :422-431, via callers :448 and :1608. Gated by `enable_return_routed_experts` (tpu_runner.py:212,1499; default False) | **NO — residual holdout** |
| KV-connector insert/extract (disagg transfer) | **`B`** | `insert_request_with_kv_cache` passes `self.runner.block_size` (`runner/kv_cache_manager.py:1278,1292`) as tokens-per-page into the jitted insert helpers (:1100-1180), against physical pages holding `B×d` tokens | **NO — residual holdout** |

The two holdouts are the same mismatch class but live in niche, opt-in paths (MoE routed-expert recording;
prefill→decode KV transfer) and were **not runtime-verified** by us — code-reading only.

Divisibility question (shape crash vs. silent): post-#3129 dim-1 is `B×d` (RPA) or `B×d/packing` (MLA), both
always divisible by `d` for valid `B` — so the transient `P(BATCH, CONTEXT)` sharding never crashes; there is
no loud guard anywhere. A wrong geometry fails silently, as the v0.20–v0.24 history demonstrates.

## Is dcp>1 reachable/tested upstream?

- Reachable: `--decode-context-parallel-size` → `parallel_config.decode_context_parallel_size` →
  `layers/common/sharding.py:219`, requires `tp % dcp == 0` (:227-232, TP axis is split: `tensor_parallelism //= dcp`),
  mesh gets a real `dcp` axis (:526, `MESH_AXIS_NAMES` :32-33), gated on `NEW_MODEL_DESIGN=True` (:324-329).
- Tested: **no dcp>1 test exists**. Every upstream test sets `decode_context_parallel_size = 1`
  (`tests/layers/common/test_sharding.py`, 10 sites); mesh tests only assert the axis exists
  (`tests/runner/test_tpu_runner_mesh.py:137-180`). No docs/examples mention it (`git grep` over `docs/`,
  `examples/`, `README.md` is empty).
- **No true context-parallel attention**: at consumption time the `dcp` mesh axis is folded into the
  head/tensor axes (`ATTN_HEAD = ('model','expert','dcp')`, `MLP_TENSOR` includes `'dcp'` —
  `layers/common/sharding.py:47,49`), i.e. every device processes the full context for its head slice; the
  experimental CP kernel (#2842, `kernels/experimental/rpa_v3_cp/kernel.py`) is **not referenced anywhere
  outside its own directory**. Our fork's `P(BATCH, CONTEXT)` gate / owner-scatter / LSE-merge machinery is
  fork-only.

## What this means for our fork

1. **Reframe `1f700c507` as convergent with upstream #3129**, not as a fork-only invention. Both implement the
   identical two-sided contract: spec advertises the raw per-shard size; the physical global page is allocated
   at `spec.block_size × dcp`; the runner block table is rebuilt at `×dcp` granularity. On the next
   upstream sync we should adopt #3129's structure (the ×dcp inside `get_kv_cache_shape_with_mesh`, and
   `maybe_reinitialize_input_batch` reading dcp from the mesh CONTEXT axis) and drop any duplicate local logic.
2. **Residual deltas still worth upstreaming** (in rough order of value/controversy):
   - **Startup geometry assert + CPU-only unit test** (small, low-controversy, independently mergeable — and
     it would have caught the bug that shipped in five releases). Sketch against `main`, e.g. at the end of
     `maybe_reinitialize_input_batch` (`runner/kv_cache_manager.py`):

     ```python
     # The engine allocates one block id per spec.block_size * decode_context_parallel_size
     # tokens (vLLM scales internally); the mesh CONTEXT axis must agree, and the
     # physical page must cover exactly that many tokens, else the block table
     # silently addresses the wrong pages.
     dcp = self.runner.vllm_config.parallel_config.decode_context_parallel_size
     assert context_cnt == dcp, (
         f"mesh CONTEXT axis ({context_cnt}) != decode_context_parallel_size ({dcp})")
     for group in kv_cache_config.kv_cache_groups:
         spec = group.kv_cache_spec
         if isinstance(spec, AttentionSpec):
             expected = spec.block_size * dcp
             got = get_kv_cache_shape_with_mesh(
                 self.runner.mesh, 1, spec.block_size, spec.num_kv_heads,
                 spec.head_size, t2j_dtype(spec.dtype), self.use_mla)
             # dim-1 tokens: RPA layout stores page_size directly; MLA stores page/packing.
             ...  # assert tokens-per-page == expected
     ```
     (Exact accessor per layout to be worked out in the PR; the load-bearing check is
     `context_cnt == decode_context_parallel_size` plus one shape probe.)
   - **The two granularity holdouts** (routed-experts slots `tpu_runner.py:422-431/448/1608`; KV-connector
     insert `kv_cache_manager.py:1278,1292`): one-line fixes (`block_size × dcp` or reuse the InputBatch's
     group block size), but each needs a repro/verification pass first — flag honestly as found-by-inspection.
   - **True CP attention** (owner-scatter/LSE-merge, persistent CONTEXT-sharded cache): the actual feature gap
     upstream. This is PR-#2324-successor territory, not a small fix; upstream's own experimental CP kernel
     (#2842) being unwired suggests they know the gap exists.
3. **Optional, zero-cost goodwill move** (owner's call, not done by this investigation): a short comment on
   upstream #3129 noting that released v0.20.0–v0.24.0 carry the pre-fix corruption at dcp>1 (silent past
   `block_size × dcp` tokens per request), so users on releases should treat dcp>1 as broken until the next
   release. That is a factual footnote to their own fix, not a new report.

## Method note

All claims above cite `upstream/main` (`a3fe6eec1`) file:line, checked via `git show`/`git grep` against the
fetched remote-tracking ref whose SHA was verified against the live GitHub remote. vLLM engine behavior was
verified in the local checkout at `~/vllm-build` (not from memory). Nothing was executed on TPU; no external
issue/PR/comment was filed.
