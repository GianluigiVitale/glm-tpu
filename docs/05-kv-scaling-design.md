# docs/05 — Scaling GLM-5.2 decode batch + context: fixing the replicated MLA latent KV cache

**Design note, 2026-07-07. READ-ONLY analysis (no TPU touched; JAX_PLATFORMS=cpu session).**
Bases: `attention_interface.py` = `~/tpu-inference/tpu_inference/layers/common/attention_interface.py`,
`kernel.py` = `~/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py`,
`sharding.py` = `~/tpu-inference/tpu_inference/layers/common/sharding.py`,
`kv_cache.py` = `~/tpu-inference/tpu_inference/runner/kv_cache.py`,
`kv_cache_manager.py` = `~/tpu-inference/tpu_inference/runner/kv_cache_manager.py`,
plus `docs/01-dsa-kernel-design.md` (Stage-2 DSA), `~/moe-tpu/docs/16` (per-rank KV sizing lessons),
upstream commit `f940073e` ("Add DCP sharding axis and KV cache support (#2398)").

---

## 0. TL;DR

1. **The latent cache is replicated 32×.** MLA cache spec is `P(BATCH)` (attention_interface.py:534,541)
   with `BATCH = ('data','attn_dp','attn_dp_expert')` (sharding.py:62) — product **1** on the Stage-1
   pure-TP mesh (model=32). Every chip holds the FULL cache: ~97.5 KiB/token/chip (padded layout; 87.8 KiB
   in the RESEARCH_LOG's unpadded estimate). 128 blocks × 512 = 65,536 tokens ≈ 6 GiB/chip → max_seqs 8 at
   4K ctx; a single 128K seq needs ~11–12 GiB/chip → **impossible**.
2. **DCP (`decode_context_parallel_size`) is upstream's sanctioned mechanism for exactly this** — the mesh
   axis, engine arg, cache sharding `P(BATCH, CONTEXT)` and scheduler `block_size *= dcp` all landed in
   #2398 — **but the MLA attention path was never finished**: `mla_attention` still declares the cache
   `P(BATCH)`, so running dcp>1 today would all-gather every layer's cache over the dcp axis every step
   (≈ the whole KV pool over ICI per decode step). Functional, catastrophic.
3. **Recommendation:** (S0) transpose-cache layout, +11% capacity, env-only → (S1) head-sharded attention
   specs (kills the 32× redundant attention FLOPs + the q all-gather; no kernel change) → (S2) **finish DCP
   for MLA** (per-shard kernel over the local cache slice + LSE softmax-combine over `dcp`; the
   distributed-flash pattern) — dcp=4 then 8 → (S3) fp8 KV as a ×2 multiplier, gated on the PR #2324
   NaN-under-EP root cause. DCP composes cleanly with the Stage-2 DSA kernel (§6): the sparse decode
   needs only a local gather + a 2.5 MiB segment all-gather, no distributed softmax.

---

## 1. Baseline anatomy (where the bytes and the traffic go today)

### 1.1 Cache bytes

GLM-5.2: 78 MLA layers, `kv_lora_rank=512`, `qk_rope_head_dim=64`, bf16 cache (`kv-cache-dtype auto`).
mla.v2 layout (`kernel.py:95-118`, `MLA_TRANSPOSE_KV_CACHE=False` default, envs.py:411-412):
`[pages, page_size/kv_packing, kv_packing, align(512,128)+align(64,128) = 640]` → **1280 B/token/layer**
→ **99,840 B ≈ 97.5 KiB/token/chip** (docs/01 §6). The transpose layout (`[pages, 576, page_size]`,
kernel.py:103-109) stores the unpadded 576 width → **89,856 B ≈ 87.8 KiB/token/chip** (the RESEARCH_LOG
2026-07-07 figure assumed this unpadded width).

Pod smoke #16 sizing: weights 23.06/30.75 GiB/chip resident FP8 → ~7.7 GiB free; KV pool 128 blocks ×
page 512 = **65,536 tokens ≈ 6.1 GiB/chip** (padded layout). Coverage at 8K ctx = 128/16 = **8 seqs**;
at 4K ≈ 14–16 KV-side (8 was the run's max_seqs). One 128K seq = 256 pages = **11.9 GiB — does not fit**;
1M ctx = 93 GiB. Because the cache is replicated, *every chip* pays this; 31 of 32 copies are redundant.

### 1.2 Compute + comm (the PR #2324 design, as ported in `cd8eeb6c`/`a429be54`)

`mla_attention` (attention_interface.py:526-668) token-shards q/q_rope/k/k_rope over `MLP_TENSOR`
(32-way), then **inside the shard_map all-gathers all four along the token axis** (:571-587) so every
chip sees the full prefix (the cross-shard-query causal fix), runs the kernel over **all 64 q-heads ×
all tokens × the full replicated cache**, and dynamic-slices the output back to its token shard
(:644-656). Net per layer:

- **32× redundant attention FLOPs** — every chip computes every head over every query.
- q reshard (head-sharded from the `W_UK_T` einsum, `P(ATTN_HEAD)` at mla_attention.py:137 → token-shard)
  + full-token all-gather + output slice.
- **Every chip reads the full active cache from HBM every decode step.** At pool-full 6.1 GiB and v4's
  ~1.2 TB/s HBM this is a ~5 ms/step floor — irrelevant today (0.95 tok/s ≈ 1050 ms unoptimized XLA
  step) but the wall the optimized Stage-2 decode would hit at long ctx (128K dense = 11.9 GiB → ~10 ms).

So: **capacity** is gated by replication; **speed** at Stage-1 is gated elsewhere (XLA decode, MoE, comm),
with the redundant-FLOPs/all-gather term mattering most at prefill and long ctx.

---

## 2. Option A — head-sharded attention (q over `ATTN_HEAD`, cache stays replicated)

**What changes (specs only; no kernel change):**

```python
# attention_interface.py mla_attention
in_specs = (
    P(ShardingAxisName.ATTN_HEAD, None, None),   # q      [H, T, lkv]  head-major, 64/32 = 2 heads/chip
    P(None, ShardingAxisName.ATTN_HEAD, None),   # q_rope [T, H, r]
    P(ShardingAxisName.MLP_TENSOR, None),        # k      (token-sharded; all-gathered inside, unchanged)
    P(ShardingAxisName.MLP_TENSOR, None),        # k_rope
    P(ShardingAxisName.BATCH),                   # kv_cache (replicated — unchanged)
    ... metadata unchanged (P(ATTN_DATA) == replicated)
)
out_specs = (P(ShardingAxisName.BATCH),          # cache: every chip writes the same full new_kv — consistent
             P(ShardingAxisName.ATTN_HEAD, None, None))  # out [2, T_full, lkv] head-sharded
```

Inside the body: drop the q/q_rope all-gathers and the output dynamic-slice; keep the k/k_rope
all-gathers (tiny: `[T, 576]` per step). Wrapper side: q arrives head-sharded already (`W_UK_T`/`W_UV`
are `P(ATTN_HEAD)`, mla_attention.py:137,166) so the current head→token reshard disappears; the
head-sharded output feeds the per-head `W_UV` einsum and the row-parallel `o_proj` **better aligned than
today** (the `cd8eeb6c` EP-head gather before o_proj becomes the standard row-parallel psum).

**Gains:** kills the 32× redundant attention FLOPs (prefill-dominant) and the q all-gather/out-slice.
**Does NOT touch memory** — cache still replicated, max_seqs/context unchanged. Decode-step gain is
modest at 4K (BW-bound on the cache read, which is unchanged) and grows with ctx.

**Risks/validation:** kernel with 2 local q-heads (bf16 q_packing=2 → 1 packed row, kernel.py:351-352)
— verify the v4 block config `(1,1,1)/(1,8,8)` + `TuningKey(actual_num_q_heads=2)` path on the sub-cube;
the `ATTN_HEAD` tuple `('model','expert','dcp')` (sharding.py:47) also permits 16-way heads × 2-way
tokens if 2 heads/chip underfills the MXU. Gate: specs are a no-op at mesh product 1 (1-chip
byte-identity), TP=4==TP=1 tokens on the sub-cube, pod smoke + GSM8K rows unchanged.

---

## 3. Option B — PAGE-sharded cache (`P(pages over model)`) — **rejected in favor of C**

Sharding dim 0 (pages) over the 32-way model axis gives each chip 1/32 of pages, but:

- `block_tables` hold allocator-assigned page ids with **no ownership structure**; a per-shard walker
  needs either dynamically-shaped per-shard page lists (impossible under jit) or a full-table walk with
  ownership masks (`page_id % 32 == shard`) — the walk cost stays global, only the DMA is saved.
- The vLLM allocator is unaware of ownership → per-shard capacity imbalance; per-rank starvation is the
  exact failure class of `~/moe-tpu/docs/16` LATEST-10 (per-rank KV starvation under DPScheduler).
- It needs the same cross-shard softmax-combine machinery as option C, plus table plumbing C does not need.

Every kernel change B needs (per-shard partial attention + LSE combine) is a strict subset of C's, and C
gets the runner/scheduler scaffolding from upstream for free. Build C.

---

## 4. Option C — decode-context-parallel (DCP): the sanctioned axis, half-built upstream

### 4.1 What already exists (upstream #2398, in our tree)

- Mesh axis `dcp` (last, minor-most): `MESH_AXIS_NAMES` (sharding.py:32-33); engine arg
  `--decode-context-parallel-size` read at sharding.py:206; **DCP reuses the TP axis**
  (`tensor_parallelism //= dcp`, sharding.py:214-219); requires `NEW_MODEL_DESIGN=1` (:311-315, we set it).
- **Weight shardings are UNCHANGED at any dcp**: every weight axis tuple includes `'dcp'`
  (`ATTN_HEAD=('model','expert','dcp')`, `MLP_TENSOR/EXPERT/VOCAB=(...,'model','dcp')`, sharding.py:47-54)
  so model×dcp ≡ the old 32-way tensor axis. On our pure model=32 mesh, dcp=4 → model=8×dcp=4.
- **Cache sharding**: MLA cache `P(BATCH, CONTEXT)` with `CONTEXT='dcp'` (kv_cache.py:132-135) — dim 1
  (the in-page token dim) split dcp-ways. With `block_size *= dcp` in `get_kv_cache_spec`
  (kv_cache_manager.py:457-458) the scheduler allocates **logical blocks of `512·dcp` tokens** and each
  chip physically holds a 512-token slice of every logical block: shard `s` owns in-block offsets
  `[s·512, (s+1)·512)`. Per-chip cache bytes ÷ dcp. **No page-table rebasing needed** — all shards walk
  the same `page_indices`; only the in-block offset→shard mapping is new.

### 4.2 What is missing (the actual work)

`mla_attention`'s cache in/out specs are still `P(BATCH)` (attention_interface.py:534,541 — #2398 changed
them from `P(MLP_TENSOR)` to `P(BATCH)` but never added `CONTEXT`). Under dcp>1, jit reshards the
`P(BATCH, CONTEXT)` array to match → **all-gather of each layer's whole cache over dcp, every layer,
every step** (at dcp=4 ≈ 216 MiB/chip/layer × 78 ≈ 17 GiB ICI traffic per decode step). So dcp>1 is
*storage* support only. Required changes:

1. **Specs**: cache in/out → `P(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT)`; q/k gathers already
   include `dcp` (they gather `MLP_TENSOR`, which contains `'dcp'`) so every shard sees the full token
   set and the post-gather `TuningKey` shapes are unchanged.
2. **Kernel: emit LSE.** Each shard runs the existing kernel over its local slice and returns, besides
   the normalized per-shard output, the per-row log-sum-exp `lse = m + log(l)` from the `(m, l)` scratch
   at finalize (the online-softmax state already exists — kernel.py:491-509,647-655). Small, localized
   Pallas change (one extra `[num_head_rows, T]` f32 output).
3. **XLA combine over `dcp`** (outside the kernel, inside the shard_map):
   `LSE = logsumexp_s(lse_s)`; `out = Σ_s out_s · exp(lse_s − LSE)` — an `all_gather` of the tiny lse
   array + a weighted `psum`. fp32. Empty shard (a short seq entirely on other shards) → `l=0, m=−inf` →
   weight 0; this needs the PR #2324-class NaN guards (`exp(−inf−(−inf))`) at the combine.
4. **Per-shard positions + kv_len.** The kernel's causal mask compares `q_span = kv_len − q_len + …`
   against `k_span` derived from the *linear local* kv index (kernel.py:473-478,634-641). Under DCP the
   local element `(logical_page_p, offset o)` has global position `p·(512·dcp) + shard·512 + o`, and the
   per-shard token count is `n_s(L) = Σ_p clamp(L − p·512·dcp − s·512, 0, 512)` (closed form). Pass
   per-shard `kv_lens_s` (computed in XLA from the global `seq_lens`) and add the strided `k_span`
   formula (two scalars: `dcp`, `axis_index('dcp')`). For **decode-only** the mask is trivial (attend
   everything cached) — per-shard `kv_lens_s` suffices; the strided k_span matters for **prefill/mixed**.
5. **New-token write routing** (the trickiest piece). The kernel currently writes the chunk's new KV into
   its local cache and attends it as the tail (kernel.py:716-746). Under DCP a token's slot exists only on
   its owner shard `(pos mod 512·dcp) div 512`. Two candidate mechanisms, to be settled by a micro-parity
   harness before integration:
   - **(a) `owns_new` scalar gate**: per-shard scalar-prefetch flag/count; non-owner shards run with
     `kv_len_s` = cached-only and skip the new-kv branch (owner attends+writes it; the combine includes
     each token exactly once — a disjoint-union softmax is exact).
   - **(b) write-then-attend split** (the `doc16` Fix-B shape): scatter the new latents to owner-shard
     slots in XLA *before* the kernel, then run the kernel history-only. Cleaner ownership, but the
     kernel's fused write is load-bearing for chunked prefill — verify it tolerates zero-length new-kv.
6. **Runner glue**: none beyond what #2398 landed (spec block_size, allocation via
   `layer_spec.block_size` at kv_cache_manager.py:900-909 already flows the ×dcp size into the sharded
   alloc). No DPScheduler in this topology (total_dp_size=1), so no per-rank starvation class; the pool
   stays a single global block pool. Note: allocation granularity becomes `512·dcp` tokens/seq —
   short-seq internal fragmentation grows with dcp (fine for the long-ctx target; keep dcp ≤ 8 for
   short-ctx serving configs).

### 4.3 What DCP buys (KV budget ≈ 6.1 GiB/chip, padded layout; ×1.11 with S0-transpose)

| Config | pool tokens | max_seqs @4K (KV-side) | 128K/seq per chip | 128K seqs | 1M/seq per chip |
|---|---|---|---|---|---|
| baseline (replicated) | 65,536 | ~14 (run at 8) | 11.9 GiB | **0** | 93 GiB |
| **dcp=4** | 262K | ~56 | 2.98 GiB | 2 | 23.3 GiB |
| **dcp=8** | 524K | ~112 | 1.49 GiB | 4 | 11.6 GiB |
| dcp=8 + fp8 KV | 1.05M | — | 0.74 GiB | 8 | 5.8 GiB |
| dcp=16 + fp8 (or dcp=32) | 2.1M | — | 0.37 GiB | 16 | **2.9 GiB — 1M fits** |

(dcp must divide the tensor axis: 4/8/16/32 all divide 32. max_seqs also caps at runner/bucket limits;
KV stops being the binding constraint from dcp=4.) Long-ctx dense decode HBM-read floor scales ÷dcp;
prefill attention FLOPs are unchanged by C (that is option A's job — A and C compose: heads over
`('model','expert')`, pages over `'dcp'`, both inside `ATTN_HEAD`'s tuple).

---

## 5. Option D — keep replicated, shrink bytes (transpose layout; fp8 KV)

- **S0 quick win — `MLA_TRANSPOSE_KV_CACHE=1`**: 640→576 B/row = **+11% capacity**, env-only, page 512
  satisfies the `page_size % 128` assert (kernel.py:104). The transpose path is a distinct kernel branch
  (DSV4 Phase-1 hit v4 VMEM issues in `_xpose_tiles`) → gate on 1-chip parity + sub-cube before pod.
- **fp8 KV (e4m3)**: ×2 (97.5→~49 KiB/token). Blocked today: PR #2324 saw **NaN under EP with fp8 KV**
  and forced `kv-cache-dtype auto` (docs/recon/pr2324-diff.md:26,38), and the streaming loader raises
  `NotImplementedError` with `--kv-cache-dtype fp8` (docs/00 memo:82). v4 compute is fine (the
  `_upcast_kv_for_v4` per-256-chunk transient upcast already exists, doc11 §3). Verdict: a *multiplier*
  to apply after DCP if the 1M endpoint demands it — requires root-causing the NaN (suspect the same
  empty-shard/−inf class as §4.2(3)) and a loader fix. Never the primary fix: 2× does not reach 128K
  (11.9→6 GiB/seq still ≈ the whole pool), and it costs accuracy-validation burden.

---

## 6. Composition with the Stage-2 DSA kernel (docs/01) — why DCP is the right end state

- **Sparse decode (docs/01 §3.1(a))**: indices are absolute token positions; under DCP each shard gathers
  its *owned* selected entries (owner = `(pos mod 512·dcp) div 512`, avg 2048/dcp rows) from its local
  slice, then one all-gather assembles the full `[R, 2048, 640]` segment (~2.5 MiB/seq) on every chip —
  the dense DSA kernel then runs **unchanged, with no distributed softmax**. DCP degrades to a pure
  storage layer under DSA: the §4.2 combine machinery is needed only for the dense fallback (ctx ≤ 2048,
  gate D0) and Stage-2-initial prefill (§3.3 bias-mask over the full paged history — which reuses the
  §4.2 strided-position walk + LSE combine as-is).
- **Indexer scoring (docs/01 §1.3)**: the blocked scan over the indexer k-cache becomes per-shard over
  local pages (scores are per-(q,token) — concatenation, not softmax); the blocked exact top-k merge
  (§1.4) already merges block-local top-2048 candidates — shards are just blocks. The indexer k-cache
  registers with the same `P(BATCH, CONTEXT)` spec (+5.4% bytes also ÷dcp).
- Head-sharding (A) stays orthogonal: the DSA decode kernel's q rows fold `[bq·H]` per shard-local H.

So the Stage-2-compatible end state is: **A (heads over model·expert) + C (pages over dcp) + DSA sparse
decode gathering from the dcp-local slice**, with fp8 KV as the optional last ×2 for the 1M endpoint.

---

## 7. Recommended sequence, expected outcomes, validation

Perf baseline honesty: the pod does 0.95 tok/s single-stream *unoptimized XLA decode*; attention is not
the current bottleneck, so S0–S2 are **capacity** plays whose tok/s effect materializes as batch
aggregate (×max_seqs) and at long ctx. Numbers below are KV-side expectations, not e2e promises.

| Step | Change | Effort | max_seqs @4K | Context ceiling | tok/s expectation |
|---|---|---|---|---|---|
| S0 | `MLA_TRANSPOSE_KV_CACHE=1` | hours + parity | 8→9 | ~72K pool | ~none (capacity only) |
| S1 | Option A specs | days | 8 (unchanged) | unchanged | prefill/warmup ↓ (≤32× attn-FLOP cut); decode +small, grows with ctx |
| S2a | DCP=4, decode distribution | ~1 wk | →32 (KV-side 56) | 32K–64K real; 128K×2 marginal | aggregate ×4 via batch; long-ctx read floor ÷4 |
| S2b | DCP=8 + prefill/mixed strided positions | +1 wk | →64+ | **128K×4 (passkey ladder runnable)** | aggregate ×8; 128K dense step read ~1.5 GiB/chip |
| S3 | fp8 KV (gated on NaN root cause) | 1–2 wk | ×2 on any row | 256K×4 / 1M in sight | ÷2 read floor |
| End | Stage-2 DSA over DCP (docs/01) | Stage-2 | — | **1M @ dcp≥16 (+fp8)** | attention reads ctx-independent (2048-entry segment) |

**Validation plan (every step, the house discipline):**

- **Byte-identity when off**: all spec changes are no-ops at mesh product 1 / dcp=1 — 1-chip
  `parity/glm_engine_parity.py` (bf16+fp8, two-step) must stay bit-identical.
- **New micro-parity `parity/mla_dcp_parity.py`** (S2, before engine wiring): sub-cube 4-chip, dcp∈{2,4} ×
  model∈{2,1} vs the dcp=1 reference — fp32 ≈1e-7 / bf16 floor; chunk patterns one-shot / prefill+1 /
  1-by-1 decode (the doc16/doc09 patterns), shuffled block tables, seq lengths straddling shard
  boundaries (511/512/513, 512·dcp±1), empty-shard rows (the −inf guard), and both §4.2(5) write
  mechanisms A/B-tested here first.
- **Engine**: two-step cached parity with `--dcp`; TP-equivalence (dcp=4 == dcp=1 tokens, the
  `dsv4_runner_decode_parity` pattern); zero serving compiles via `DSV4_OBSERVE_COMPILES` (the mesh
  changes shape → re-enumerate the collective-region programs, docs/15 — do not assume the pure-TP set).
- **Pod**: 3/3 clean runs (probabilistic multi-host halts); GSM8K n≥32 + GPQA-Diamond scores unchanged
  vs the Stage-1 rows (provenance DB); `bench/glm_longctx.py` passkey ladder 8K→32K→128K at dcp=8 —
  this is the Stage-2 gate-P instrument and the first run that *needs* S2b.
- **Numerics**: combine in fp32 (`s_dtype=jnp.float32` is already the v4 kernel style,
  attention_interface.py:605); assert `Σ_s exp(lse_s − LSE) ≈ 1` in the parity harness.

**Open questions carried into implementation:** (i) §4.2(5) write mechanism (a) vs (b); (ii) whether
2 q-heads/chip (S1) underfills the MXU enough to prefer 16-way heads × 2-way tokens; (iii) fp8-KV NaN
root cause (shared with the −inf-guard work); (iv) physical page 512 vs 128 as the striping granule
(finer striping ↓ short-seq imbalance, ↑ DMA count — measure, don't guess).
