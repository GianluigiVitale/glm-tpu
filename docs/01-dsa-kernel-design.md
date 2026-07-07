# docs/01 — Stage-2 design: the GLM-5.2 DSA kernel on TPU v4

**Lightning indexer + top-k=2048 + sparse MLA in JAX/Pallas, adapted from the DSV4 CSA kernel.**
Written 2026-07-07 against fork branch `glm-5.2-v4` (Stage-1 dense-MLA parity GREEN, sub-cube — see
`docs/RESEARCH_LOG.md` 2026-07-07). Design bases: `docs/recon/doc09-csa-kernel.md` (DSV4 CSA kernel as-built),
`docs/recon/doc11-mlav2-map.md` (mla.v2 structural map), the HF reference
(`reference/modeling_glm_moe_dsa.py`), and the vLLM GPU reference (`/home/gianl/vllm-build`).

All paths below: `kernel.py` = `~/tpu-inference/tpu_inference/kernels/mla/dsv4/kernel.py`,
`mla_v2.py` = `~/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py`,
`mla_attention.py` = `~/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py`,
`kv_cache_manager.py` = `~/tpu-inference/tpu_inference/runner/kv_cache_manager.py`,
`deepseek_v2.py` = `/home/gianl/vllm-build/vllm/model_executor/models/deepseek_v2.py`,
`sparse_attn_indexer.py` = `/home/gianl/vllm-build/vllm/model_executor/layers/sparse_attn_indexer.py`,
`modeling.py` = `~/glm-tpu/reference/modeling_glm_moe_dsa.py`,
`config` = `~/glm-tpu/configs/glm-5.2-fp8-config.json`.

---

## 0. Decisions up front

1. **Sparsity is expressed over dense fixed shapes, never as dynamic gather inside Pallas** — the proven DSV4
   pattern (doc09 §1 "comp_bias masking"). Pallas enforces lexicographic grid traversal and
   `jax.lax.approx_max_k` is >10× slow even at 50% recall (`docs/00-feasibility-memo.md:72`); token-granular
   in-kernel gather would mean one DMA per selected token.
2. **Decode = design (a): XLA gathers the selected 2048 latent entries into a fixed `[R, 2048, 640]` segment**,
   then a dsv4-derived tiled flash kernel attends it (2048 = 16×128, tiles cleanly). The segment is
   **context-independent** — VMEM stops scaling with ctx by construction (DSV4's `max_win` grew with ctx and
   needed `DSV4_DECODE_COMP_BLOCK` tiling to survive 12K; GLM's segment is pinned at 2048 forever).
3. **Prefill Stage-2-initial = design (b): bias-mask over the full paged history** (correct, wasted FLOPs),
   upgraded later by measured page-union compaction. Chunked prefill reads cached cross-chunk history from
   step one (the doc16 lesson).
4. **Top-k is exact, blocked, and lives in XLA**: per-block `lax.top_k(·, 2048)` + hierarchical merge
   (mathematically exact for top-k). Two-pass thresholding is the benchmarked fallback, `approx_max_k` is banned.
5. **Indexer k-cache: bf16, head 128, real KVCacheSpec** replacing the Stage-1 no-slot skip
   (kv_cache_manager.py:640-643), sharing the MLA block tables (DSV4 aux-cache precedent,
   kv_cache_manager.py:73-84). FP8+scale (GPU layout, 132 B) is a later split-array optimization — the packed
   GPU layout gains nothing on TPU (132 → 256 after 128-lane alignment).
6. **RoPE layout: prior = interleaved (vLLM behavior honoring `indexer_rope_interleave: true`), HF's
   non-interleaved call presumed a DSV3.2 copy-paste** — but it is settled empirically against real weights
   before 2a is trusted (§1.2, three experiments; output-level tests must run at ctx > 2048 because at
   ctx ≤ topk sparse ≡ dense regardless of layout).
7. **IndexShare plumbing is NOT deferrable to Stage 3**: the checkpoint ships indexer weights ONLY on the 21
   `full` layers (+MTP layer 78) — keyset fingerprint, `RESEARCH_LOG` 2026-07-07; `indexers_proj` was a
   quant-config red herring and does not exist. Shared layers physically cannot score; cross-layer index reuse
   lands in 2a. Stage 3 adds only MTP sharing + throughput claims.

---

## 1. Indexer forward on TPU (XLA first — phase 2a; Pallas — 2b)

### 1.1 The math (dims fixed by config)

Per **full** layer (`indexer_types`, config:26-105 — layers 0,1,2 then 6,10,…,74; 21 of 78), per token batch
`[T]`, hidden `h [T, 6144]`, and `q_resid = q_a_layernorm(q_a_proj(h)) [T, 2048]` — already computed by the MLA
wrapper as `q_c` (mla_attention.py:314-319) and passed to the indexer exactly as vLLM does
(`self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`, mla_attention.py:343-345):

```
q = wq_b(q_resid)                     # [T, 4096] -> [T, 32, 128]   (modeling.py:191, :231-232)
k = k_norm(wk(h))                     # [T, 128]; k_norm is nn.LayerNorm(128, eps=1e-6) WITH bias
                                      #   (modeling.py:192-193; config modules_to_not_convert lists
                                      #    indexer.k_norm.bias — LayerNorm, not RMSNorm)
rope on dims [0:64] of q and k        # rope-FIRST split, both refs (modeling.py:233,236;
                                      #   deepseek_v2.py:702-712); LAYOUT: §1.2
scores[t,h,s]  = relu(q[t,h]·k[s] * 128**-0.5)          # modeling.py:246-247
w[t,h]         = weights_proj(h)[t,h] * 32**-0.5        # [T, 32]  (modeling.py:250)
index_scores[t,s] = Σ_h w[t,h]·scores[t,h,s]            # [T, S]   (modeling.py:251)
causal mask (s > pos_t -> -inf)                          # modeling.py:254-259
topk_indices = top-k(index_scores, k=min(2048, S)).int32 # modeling.py:261-262
```

Notes that matter for TPU implementation:

- **Scale folding through relu is legal**: relu(αx)=α·relu(x) for α>0, so `softmax_scale` (and, later, a
  positive per-token fp8 q-dequant scale) can be folded into `w` post-relu — exactly what vLLM does
  (`weights = weights_proj_out * q_scale * softmax_scale * n_head**-0.5`, deepseek_v2.py:738-741). The XLA
  reference applies the scale pre-relu like HF; the Pallas kernel folds it into `w` — bit-equivalent in fp32,
  and the parity harness checks both formulations once.
- `w` can be **negative** (plain linear head): `index_scores` are signed; do not assume non-negativity.
- HF computes scores in **fp32** (`q.float()`, `k.float()`, modeling.py:246) — the XLA reference does the same;
  fp32 score accumulation is also the mla.v2 house style (doc11 §3).
- vLLM fuses `wk`+`weights_proj` into one GEMM (deepseek_v2.py:636-643) with FP8-wk→bf16 load-time dequant
  (deepseek_v2.py:746-779). On TPU these GEMMs are trivial (6144×160); keep them **separate** in 2a for
  parity clarity, fuse later only if profiled.
- Decode uses the same math with `T = R` new tokens scoring against the full cached prefix (self-position
  included — the new k is written to the indexer cache before scoring, mirroring GPU insert-then-score,
  sparse_attn_indexer.py:163-173).

### 1.2 The RoPE-interleave conflict — and the experiment that settles it

**The landmine** (`docs/recon/glm-reference.md:7`): the checkpoint config sets `indexer_rope_interleave: true`
(config:25). vLLM honors it — `is_neox_style=not getattr(config, "indexer_rope_interleave", False)` →
**interleaved** rope for the GLM indexer (deepseek_v2.py:1003-1008). HF's modeling file applies
**non-interleaved** half-split rope with an explicit comment "The indexer uses NON-interleaved (half-split)
RoPE — unlike the main MLA attention" (modeling.py:238-239), and the HF config class never reads
`indexer_rope_interleave`. One reference is wrong. (The main MLA rope is uncontested: `rope_interleave: true`,
both refs interleaved — modeling.py:302/:434, deepseek_v2.py:987.)

**Prior: vLLM/interleaved is right.** The flag exists only in GLM configs (DSV3.2 has none and genuinely uses
non-interleaved — HF's comment reads as inherited from the DeepSeek modeling file), zai set it `true`
deliberately, and vLLM's GLM-5.x enablement added explicit handling for it. But a wrong layout scrambles the
64 rope dims' pairing and silently degrades selection at long ctx only — this must be settled empirically
against **real GLM-5.2-FP8 weights** before any 2a result is trusted.

**Critical constraint on experiment design**: at `S ≤ 2048`, `topk = min(2048, S) = S` — **every** token is
selected regardless of scores, so sparse ≡ dense and no output-level test at short ctx can discriminate.
Discriminating tests must either inspect scores/selected sets directly (any ctx) or run at ctx > 2048.

Three experiments, cheapest first; run all three, require agreement:

- **E1 — score-vs-attention agreement (short ctx, weights-only, no pod).** Load layer-0..2 real weights
  (embed + first blocks + indexer; a few GB). On ~512–2048 tokens of natural text, compute `index_scores`
  under BOTH layouts, and the same layer's **dense MLA attention** row-mass per key (sum of per-head softmax
  probabilities) as ground truth — the DSA indexer is trained against the attention distribution, so the
  correct layout must show markedly higher agreement. Metrics: recall of indexer-top-`n` vs attention-top-`n`
  (n ∈ {64, 256}) and per-row Spearman, averaged over queries/layers. Runs on CPU/single-chip in fp32.
- **E2 — weight forensics (corroborating only).** The 64 rope input dims of `wk`/`wq_b` acquire
  frequency-band structure in training; compare column-pair similarity statistics under the two pairing
  hypotheses (interleaved pairs (0,1),(2,3),… vs half-split pairs (i, i+32)). Cheap, no forward pass; treat
  as a tiebreaker signal, never sole evidence.
- **E3 — behavioral (decisive, ctx > 2048).** With the 2a XLA sparse path on the pod: passkey retrieval and
  prompt-logprob divergence-from-dense at ctx 4K–16K under both layouts. The correct layout tracks the dense
  baseline (bounded divergence, §4 gate D) and holds passkey; the wrong one visibly degrades retrieval. Also
  compare TPU selected sets vs the torch HF-math reference run with the winning layout (gate S).

Record the verdict + numbers in `docs/RESEARCH_LOG.md` and hard-code the winner with a comment citing this
section; keep the loser reachable behind a debug flag for the parity harness only.

### 1.3 XLA-first scoring, blocked over history

`index_scores [T, ctx]` cannot be materialized at long ctx (§6): the per-head intermediate
`[T, 32, ctx]` fp32 is 64× worse. Score computation is **blocked over the key axis** from day one, in both
phases:

- Walk the indexer k-cache in kv blocks of `B` tokens (B = 4096–32768, tuned; pages resolved through the same
  per-rank-local `block_tables` as the MLA cache — §2). Per block: `logits [T, 32, B] = einsum(q, k_blk)` →
  relu → weighted head-sum → `s_blk [T, B]` fp32; apply causal/valid masking (absolute positions vs
  `pos_t`/`kv_len`, same masking discipline as kernel.py:124,180-183).
- 2a does this as a `lax.fori_loop`/scan of XLA ops (correctness-first, HBM round-trips accepted — the same
  role `_paged_decode_jit`'s blocked XLA core played for DSV4). 2b fuses q·k → relu → weighted-sum into a
  Pallas kernel: grid `(seq, kv_blk)`, q `[T, 32, 128]` resident in VMEM, per-head accumulation via a
  `fori_loop` over the 32 heads (each head's `[T,128]` q is a clean leading-dim slice — avoids the
  `[T*32, B] → [T, 32, B]` relayout Mosaic can't do, the same constraint that forced the dsv4 host-side row
  fold, kernel.py:40-43,343-348), emitting `s_blk [T, B]` (or per-block top-k candidates, §1.4) to HBM.
  Scoring output for a full 128K map at decode R=64 is only 32 MiB fp32 — HBM-friendly; VMEM per step holds
  one `[T, B_tile]` tile.

### 1.4 Blocked exact top-k (the selection)

**Primary: hierarchical exact top-k in XLA.** Split ctx into `nb = ceil(ctx/B)` blocks; per block take
`lax.top_k(s_blk, 2048)` (values+indices, indices rebased to absolute positions); merge: `lax.top_k` over the
concatenated `[T, nb·2048]` candidates (or pairwise-running merge over `[T, 4096]` to bound peak memory).
This is **exact**: every global top-2048 element is necessarily in its own block's top-2048. At 128K with
B=8192: 16 blocks → final top-k over 32768 candidates per row. `lax.top_k` is sort-based on TPU — the block
size B trades sort width against merge fan-in; microbench in 2a and record the tuned B.

**Fallback (only if the sort cost is measured too slow at 512K–1M): two-pass threshold.** Pass 1: bisect a
global threshold θ on the score range using blocked `sum(s ≥ θ)` counts (~12 fp32 bisection iterations, each a
cheap masked reduction over the blocked score map); pass 2: compact indices with `s > θ` plus tie-trim at the
boundary to exactly 2048. More HBM passes, no sorts. Do not build this until the primary is measured
inadequate.

**Banned:** `jax.lax.approx_max_k` (memo:72). **Tie semantics:** `lax.top_k` and `torch.topk` order equal
values differently; gates compare selected **sets** with tie-groups at the k-th boundary treated as
interchangeable (§4 gate S).

**GPU cross-reference:** the GPU pipeline scores in fp8 (`fp8_fp4_paged_mqa_logits`,
sparse_attn_indexer.py:324-333) and selects with `persistent_topk` — which special-cases topk ∈ {512, 1024,
**2048**} (sparse_attn_indexer.py:337-349), confirming 2048 is a first-class production point. The GPU fp8
selected sets are NOT the parity reference (they already deviate from HF bf16 math); the reference is the HF
torch math (§4).

---

## 2. The indexer k-cache (per-layer paged cache of 128-dim keys)

### 2.1 What is cached, and the Stage-2-initial format

One 128-dim key per token per **full** layer (21 layers; +MTP layer 78 in Stage 3). Stage-2-initial:
**bf16, head_size 128** (already 128-lane aligned, zero padding waste) = 256 B/token/layer, 5.25 KiB/token
total — **+5.4%** on top of the MLA latent cache (97.5 KiB/token, §6). The GPU stores fp8+scale packed as
uint8 head 132 (`head_dim + head_dim//128*4`, deepseek_v2.py:654-659); on TPU that packed layout aligns
132→256 B — **identical to bf16, zero benefit**. The later fp8 optimization must therefore use **split
arrays** (k `[…,128]` u8 + scale `[…,4]` B → 132 B/token effective, a further 1.94×) and add a scale multiply
in the scoring kernel — the same class of hook as mla.v2's `_upcast_kv_for_v4` sites (mla_v2.py:358,437,571,593).

### 2.2 KVCacheSpec registration (replacing the Stage-1 skip)

Stage 1 deliberately gives the indexer cache **no slot**: `is_dsa_indexer_cache` (kv_cache_manager.py:62-70)
matches vLLM's self-registered `DeepseekV32IndexerCache` (constructed unconditionally by the vLLM `Indexer`,
deepseek_v2.py:654-659; registered into `static_forward_context` at deepseek_v2.py:584-587) and the spec loop
`continue`s at kv_cache_manager.py:640-643. vLLM's own spec for it is
`MLAAttentionSpec(block_size, num_kv_heads=1, head_size=132, dtype=uint8)` (deepseek_v2.py:589-595) — a GPU
layout we must not inherit.

Stage-2 registration, following the `is_cache_for_ds_v4` precedent one branch above
(kv_cache_manager.py:631-638):

```python
if is_dsa_indexer_cache(attn_module):
    kv_cache_spec[layer_name] = self._create_attention_spec(block_size, 1, 128)
    continue
```

with `_create_attention_spec` (kv_cache_manager.py:116) emitting an `MLAAttentionSpec` in the runner's
kv-cache dtype (bf16 auto — the settled Stage-1 config). Design requirements and known risks:

- **Same `block_size`, shared block tables.** The indexer cache must be addressable by the **same**
  token→page mapping as the MLA latent cache: the scoring pass walks the same per-rank-local `block_tables`,
  and the sparse gather (§3) converts the same absolute indices. The DSV4 aux caches did exactly this by
  reporting a uniform spec so they merge into the one group (kv_cache_manager.py:73-84 comment). If vLLM's
  group formation refuses to merge specs with differing `head_size` (128 vs the MLA 640) even at matching
  `page_size_padded`, the runner comment at kv_cache_manager.py:600-612 ("vLLM also expects page sizes to be
  unified", the mamba precedent `update_mamba_page_size_padded`) shows the unification lever:
  `page_size_padded` in `_create_attention_spec` pads the *accounting* while the array keeps real dims. This
  merge behavior is a named **2a implementation checkpoint**; the fallback — reporting uniform head_size 640 —
  wastes 4/5 of indexer pages (+2.6 GiB/seq at 128K, §6) and is accepted only if the checkpoint fails.
- **Write path**: the indexer forward scatters the chunk's new keys into its cache pages at `slot_mapping`
  positions in XLA (the MLA path's slot metadata; GPU analogue `indexer_k_quant_and_cache`,
  sparse_attn_indexer.py:167-173) **before** scoring, so decode self-positions are present.
- **Memory accounting**: the docs/16 per-rank KV sizing (`required_kv_blocks`) must include indexer pages —
  at 1M ctx the KV pool dominates HBM and per-rank starvation is guaranteed if the 5.4% is unbudgeted
  (`docs/recon/doc16-chunked-prefill.md` §4).
- **Sharding**: the indexer cache follows the MLA cache recipe — cache `P(BATCH)`, metadata/q `P(ATTN_DATA)`,
  per-rank-LOCAL block ids (doc09 §1 DP mechanism; attention_interface.py:526-591 pattern). Indices are
  rank-local JAX arrays; nothing crosses ranks (the wrapper-context carriage in §5 is per-rank state).

---

## 3. Sparse MLA attention

### 3.1 Decode: two candidate designs

GLM's MLA decode operates on the shared latent: K = `[kv_c(512) | k_pe(64)]` = 576 wide (rope padded 64→128 →
cache width 640, the mla.v2 packing: `align(lkv,128) + align(r,128)`, mla_v2.py:211-212,230-233), V = the
nope part `kv_c[:, :512]` with W_UV applied post-kernel (mla.v2 contract, doc11 §4 — NOT dsv4's full-latent
K==V). Absorbed q per head: `[ql_nope(512) | q_pe(64→128 zero-pad)]`.

**(a) Fixed gathered segment (RECOMMENDED for decode).** XLA pre-step per decode step, per full/shared layer:

1. `topk_indices [R, 2048]` int32 (from §1 / §5), padded with −1 where ctx < 2048 (or: dense fallback, below).
   The −1 fill is a strict TAIL suffix, and the valid prefix is in **descending-score order, NOT
   position-sorted** (round-5 note): the gather and the flash kernel are permutation-invariant and require no
   sort, but any future kernel exploiting index monotonicity would silently break — restate this contract
   before reordering anything.
2. Gather: absolute index `t` → `page = block_tables[r, t // page_size]`, offset `t % page_size`. The mla.v2
   packed cache `[pages, page_size//packing, packing, 640]` (mla_v2.py:111-118) reshapes to a token-major
   `[pages·page_size, 640]` view (row-major (page, row, sub) = token order), so the gather is one fused XLA
   `take` producing `comp_kv [R, 2048, 640]`.
3. `comp_bias [R, 1, 2048]` fp32: 0 on valid selected entries, `_MASK_VALUE` on −1 padding (kernel.py:57).
   No causal mask needed in-kernel — selection is already causal (§1.1), the bias encodes everything, exactly
   the dsv4 comp-segment contract (kernel.py:18-26).
4. Pallas: the dsv4 tiled compressed path (`_dsv4_decode_compressed_tiled_kernel`, kernel.py:202-291) with
   `num_local_blocks = 0` — grid `(R, n_comp_blocks)`, comp tiles only. Deltas in §3.2. 2048 = 16×128:
   `comp_block ∈ {512, 1024, 2048}` all tile cleanly; even **untiled** the segment is 2.5 MiB — VMEM-safe
   (§6) and context-independent, so tiling is a tuning knob, not a survival requirement (unlike DSV4's
   ctx-scaling `max_win`, doc16 LATEST-8).
5. Post-kernel XLA: attended latent `[R, 1, 64, 512]` → W_UV `[64, 512, 256]` → o_proj (mla.v2's existing
   post-kernel stage, doc11 §1).

Cost: reads exactly `R·2048·1280 B` of latent per layer (168 MiB at R=64) vs dense `R·ctx·1280 B`
(10.7 GiB at 128K) — **64× less HBM traffic at 128K**, break-even exactly at ctx = 2048.

**(b) Bias-mask over the full paged KV.** Keep the dense paged walk (mla.v2 or a dsv4-style paged grid) and
add an additive per-(query, kv-block) bias built from the selected set (−inf outside). No gather memory, but
the kernel still reads and multiplies the **entire** history: at 128K it burns 64× the FLOPs and HBM traffic
of (a) precisely where DSA is supposed to win. Mask construction per kv block is also non-trivial (membership
test of block positions against 2048 sorted indices).

**Recommendation: (a) for decode.** (b) is rejected for decode but retained as the Stage-2-initial **prefill**
form (§3.3) where per-query gather explodes. Edge cases for (a):

- **ctx ≤ 2048**: skip the sparse path entirely and run the dense mla.v2 kernel — bit-exact with the Stage-1
  dense gate by construction (topk = all). This is also gate D0 (§4) for free.
- **Latent-cache write ordering (CORRECTED, round-5 F2).** The earlier claim "mla.v2/dsv4 both
  write-then-attend" is wrong for mla.v2: the `kernel.py:319-320` read-only contract is **DSV4's** (its layer
  does an explicit XLA scatter before its read-only kernel, deepseek_v4_attention.py:1876), whereas mla.v2's
  cache write is **fused inside** its attention `pallas_call` (`new_kv_c`/`new_k_pe` operands,
  `donate_argnames=("cache_kv",)`, mla/v2/kernel.py:2169,2505-2508). On a sparse decode step mla.v2 does NOT
  run, so nothing writes the new token's latent before `gather_kv_segment` — Stage-3 integration MUST add an
  explicit XLA scatter of the step's `(kv_c, k_pe)` into the token-major cache view BEFORE the gather (and
  gather from the post-scatter array so jit orders the ops). Only then does a selected self-index gather
  correctly.

### 3.2 Kernel deltas from `dsv4/kernel.py` (what changes, what is verbatim)

| Piece | DSV4 as-built | GLM DSA decode | Verdict |
|---|---|---|---|
| `_online_update` (kernel.py:60-72) | flash step, unnormalized acc | identical math | **verbatim** (PV `v` argument = value slice, below) |
| `_finalize` (kernel.py:75-80) | folds per-head sink once | GLM has **no sink**: pass `sink_rows = _MASK_VALUE` → `exp(sink−m) = 0`, l unchanged | **verbatim** (or trivially dropped) |
| `_qk` (kernel.py:83-87) | q·k over full latent | same, width 640 | **verbatim** |
| comp-tile body (kernel.py:264-285) | bias add + pad masking + online update | same; drop the local-paged branch (kernel.py:248-262) and `sliding_window` | **near-verbatim** |
| PV operand | K==V full latent | `v = kvc_tile[:, :512]` (nope-only), acc `[R_rows, 512]`; clean 128-multiple minor-dim slice | **small delta** — the one real math change |
| inverse-RoPE on output (doc09 delta c) | applied post-normalize | **dropped** (DSV4-only) | delete |
| host row fold (kernel.py:343-348) | `[bq,H]→[bq*H]`, row = sq·H+h | identical, H=64 both models | **verbatim** |
| 4-D comp reshape + host pad (kernel.py:445-451) | pad to comp_block multiple | 2048 already exact multiple → pad is a no-op; keep the code path | **verbatim** |
| scaffolding: `PrefetchScalarGridSpec`, BlockSpecs, scratch `(m,l,acc)`, `_MASK_VALUE`, precision select (kernel.py:350-354, 475-508) | | drop the cache/`block_tables` operands (no local paged walk in the sparse decode entry) | **adapted** |
| FP8 weights | n/a (kernel reads bf16 latent) | unchanged — Stage-1 keeps the KV cache bf16; FP8 weight dequant is upstream (GMM/MLA projections) | n/a |

Estimated **~70% of the compressed-tiled path reuses verbatim**; the new entry
(`glm_dsa_sparse_decode(ql, comp_kv, comp_bias, …)`) is a strict simplification of `dsv4_paged_decode`'s
compressed branch (kernel.py:401-491) — no block_tables, no sliding window, no sink, one segment.
Output contract: normalized attended latent `[R, bq, H, 512]`, W_UV + o_proj in XLA (kernel read-only,
kernel.py:35-38 discipline).

### 3.3 Prefill (chunked; analyzed separately)

Chunked prefill with chunk `C` (Stage-1 uses mla.v2's ragged mixed path, mla_v2.py:2171). Per chunk, per full
layer:

1. **Indexer keys first**: compute + scatter the chunk's `C` keys into the indexer cache, THEN score — every
   chunk after the first must see cached cross-chunk history (the doc16 §1 bug class; its Fix-B
   "write-then-attend unified path" is the default here, not a gate).
2. **Scoring**: `[C, ctx_end]` blocked over both axes — q blocks of `bq` × kv blocks of `B`; causality =
   absolute-position mask (intra-chunk causal falls out of positions; the GPU expresses the same thing as
   `cu_seqlen_ks/ke` chunk windows, sparse_attn_indexer.py:192-256). Blocked top-k per query as §1.4.
3. **Sparse attention**: per-query selected sets differ, so the gathered form explodes:
   `C·2048·1280 B` = 20 GiB gather traffic per layer at C=8192 — and per-query-block gather traffic
   (`bq·2048·1280`) exceeds one dense read of the history (`ctx·1280`) whenever ctx < bq·2048 (= 512K at
   bq=256). **Stage-2-initial: design (b)** — dense-shaped blocked flash over the paged history with an
   additive per-(q-block, kv-block) bias: scatter each q-block's indices into a `[bq, ctx]` 0/−inf mask,
   blocked over ctx (128 MiB per q-block at 128K fp32, streamed; or recomputed per kv-block from sorted
   indices). Correct, faithful, wasted FLOPs ≈ dense prefill — accepted because Stage-2's gate is decode
   speed + end-to-end correctness, and prefill FLOP waste does not corrupt results.
4. **Upgrade path (measured, not assumed)**: 2a instruments **selection overlap** — for adjacent queries and
   across the 4-layer share block, what fraction of pages does a q-block's union of selected sets touch? If
   page-coverage < ~30% at ≥256K, build the compacted per-q-block page-list LUT (the mla.v2 `page_indices`
   indirection fits this exactly — doc11 §4 closing note) + per-token bias within surviving pages. That is a
   Stage-3 throughput item with the same correctness gates.

---

## 4. Correctness gates (the DSV4 bar, doc09 §4)

Every phase is gated; a phase without its gate is not done. All harnesses live in `~/glm-tpu/parity/`,
machine-gated with exit codes, fp32 + bf16 controls, results in the provenance discipline.

- **Gate S — selected-set exact (AMENDED round 5: two tiers).** TPU indexer (2a XLA, then 2b Pallas) vs a
  torch reference implementing the HF math (modeling.py:198-262) with the §1.2-resolved rope layout:
  - **S1 (fp32, algorithmic) — ACHIEVED (2a + 2b CPU suites):** **0 index mismatches modulo exact ties** —
    compare as sets; where `score == score_kth` (tie group straddling the boundary), any member is
    acceptable; assert the tie by checking score equality at the boundary.
  - **S2 (production bf16) — boundary-band criterion.** The original "0 mismatches modulo ties" is
    **unachievable by construction** for bf16 scoring (bf16 k-cache, DEFAULT MXU precision) against an fp32
    reference: bf16 perturbs scores by ~2⁻⁸ relative (measured per-score median ≈6e-3, p99 ≈2.6e-2 on
    unit-variance scores), which swaps NEAR-equal — not exactly tied — boundary scores. Measured churn
    (H=32, D=128, Gaussian, ctx 4K–16K): 1–2 non-tie indices per 2048 (0.05–0.10%) typical; 26–42 positions
    sit inside the p99 noise band of the k-th score, so adversarial worst case is ~2%. S2 therefore requires:
    every mismatch lies in the k-th-score boundary band **|s − s_kth| ≤ ε, ε ≈ 2⁻⁸ relative to the score
    scale** (the suites' `_assert_topk_set_equiv` atol=1e-4 form is this criterion at unit variance) — or,
    equivalently strict, exactness vs a bf16-scored reference with matched rounding.
  Chunk patterns: one-shot prefill, prefill+1, 1-by-1 decode (the dsv4 harness patterns, doc09 §4), shuffled
  block tables. NOT the reference: the GPU fp8 pipeline (it deviates from HF bf16 math by construction, §1.4).
- **Gate K — kernel bit-faithfulness.** The sparse-MLA Pallas kernel vs an XLA oracle attending the **same
  gathered segment** (same indices, same bias): fp32 max-abs ~3e-8 (DSV4 achieved 4e-8 compressed); bf16
  target ≤ 3e-4 (DSV4 compressed standalone 5e-5–2.7e-4; dense achieved 0.0 bit-identical — pursue 0.0 where
  block order matches the oracle's accumulation order).
- **Gate D0 — free exactness.** At ctx ≤ 2048, sparse path output == dense mla.v2 path output bit-for-bit
  (all tokens selected; also byte-identical model when the DSA gate is off — the fork's additive-gate rule).
- **Gate D — bounded divergence-from-dense.** ctx 4K→128K prompt-logprob divergence sparse-vs-dense tracked
  fp32 and bf16; dense is not ground truth in the sparse-trained regime, so the gate is *bounded and
  monotone-sane* divergence + benchmark parity at Stage-1-validated tasks, not equality.
- **Gate P — passkey ≥95% at every length up to ≥128K** (PLAN Stage-2 threshold), 3/3 pod runs (the docs/15
  multi-host race rule: one pass ≠ done).
- **In-engine invariants:** TP=4 == TP=1 token-identical (the `dsv4_runner_decode_parity` pattern, doc09 §4
  as-built 4/4); zero recompiles on the sub-cube before pod runs.

---

## 5. IndexShare (mandatory wiring in Stage 2; Stage-3 = MTP + throughput)

- **Schedule**: `indexer_types` (config:26-105) — 21 `full` / 57 `shared`; the closed form
  `skip = max(layer_id − 3 + 1, 0) % 4 != 0` (config `index_skip_topk_offset: 3` :21, `index_topk_freq: 4`
  :23) matches vLLM's derivation (deepseek_v2.py:1023-1034) and HF. Layers 3,4,5 reuse layer 2's indices;
  6 recomputes; etc.
- **Already present in the fork**: `skip_topk` is plumbed into `VllmMultiHeadLatentAttentionWrapper`
  (mla_attention.py:248, :272-277) from vLLM's constructor (deepseek_v2.py:1060-1074), and the GLM
  construction patch builds `indexer=None` on shared layers to match the checkpoint
  (`_maybe_patch_for_glm_moe_dsa` / `_layer_is_shared`, vllm_model_wrapper.py:171-231). Shared layers have
  **no indexer weights in the checkpoint** (keyset fingerprint) — reuse is structurally required from the
  first sparse forward, not an optimization.
- **Where the indices live between layers: the vllm wrapper context.** The forward already threads per-step
  state through `get_vllm_model_wrapper_context()` (kv_caches + `layer_name_to_kvcache_index`,
  mla_attention.py:200-203). Add a `dsa_topk_indices: jax.Array [R_tokens, 2048] int32` slot: written by each
  `full` layer after §1.4, read by the next three `shared` layers, cleared per forward. Rationale over the
  GPU's runner-owned `topk_indices_buffer` (allocated in `DeepseekV2Model`, deepseek_v2.py:1235; threaded to
  the wrapper, mla_attention.py:277): that buffer is a **torch** tensor — under torchax the indices are JAX
  arrays and must not round-trip through torch (the comment at vllm_model_wrapper.py:220-227 already commits
  to this); the wrapper context has exactly per-forward lifetime and is rank-local (§2.2 sharding). The torch
  buffer keeps receiving the no-op sentinel.
- **Per-layer regather**: indices are token positions (layer-invariant); the `[R, 2048, 640]` segment is
  re-gathered per layer from that layer's own latent cache — indices reused, gather not.
- **Stage 3**: `index_share_for_mtp_iteration: true` (config:20) extends the reuse into MTP draft iterations;
  the MTP layer 78 ships its own indexer weights (keyset) — re-derive the share map explicitly when MTP lands
  (the noted TODO at vllm_model_wrapper.py:198-200). Throughput claims (the 2.9×-at-1M FLOP reduction,
  memo:16) are Stage-3 deliverables.

---

## 6. VMEM/HBM budget at 128K ctx on v4 (16 MiB VMEM, ~31 GiB usable HBM/chip, 32 chips)

Decode batch R = 64 sequences, bq = 1, H = 64 (R_rows = 64), bf16 caches, fp32 scores. Latent cache row =
640 × 2 B = 1280 B (mla.v2 packing); indexer key row = 128 × 2 B = 256 B.

| Item | Formula | @128K | Notes |
|---|---|---|---|
| MLA latent cache | 1280 B/tok/layer × 78 | **12.19 GiB/seq** | the baseline the rest is measured against |
| Indexer k-cache (bf16, Stage-2) | 256 B/tok/layer × 21 full layers | **672 MiB/seq** (+5.4%) | 1M ctx: 5.25 GiB/seq |
| Indexer k-cache (fp8 split, later) | 132 B/tok/layer × 21 | 347 MiB/seq (+2.8%) | GPU-packed 132→256 B after TPU lane alignment = no win; split arrays required |
| Indexer k-cache (fallback uniform head 640) | 1280 B/tok/layer × 21 | 3.28 GiB/seq | the §2.2 group-merge fallback; −2.6 GiB/seq argues for the real spec |
| Score map (blocked) | R × B × 4 B per block | 8 MiB @ B=32K; 32 MiB whole map | whole `[T,ctx]` map at prefill C=8192 = 4 GiB → **must** block (§1.3); per-head `[T,32,B]` fp32 intermediate is 64 MiB even at B=8K → 2b fuses it away |
| Top-k candidates | R × nb·2048 × 8 B | 16 MiB (B=8192, nb=16) | values+indices; pairwise merge caps at R×4096 |
| Gathered segment (per layer, transient) | R × 2048 × 1280 B | **168 MiB** | one layer alive at a time; re-gathered per layer |
| `comp_bias` | R × 64 rows × 2048 × 4 B | 32 MiB | fp32, row-folded |
| **Decode-step HBM reads, sparse** | 78×gather + 21×indexer-scan | 78×168 MiB + 21×(R·ctx·256 B) = 13.1 + 44.0 = **~57 GiB/step** | indexer full-history scan now dominates — IndexShare (21 not 78 scans) is already the mitigation; fp8 keys halve the 44 GiB |
| **Decode-step HBM reads, dense** | 78 × R·ctx·1280 B | **~835 GiB/step** | → **~15× traffic reduction** at 128K (and 64× on the attention itself) |
| FLOPs/token, dense MLA | 78 × 64H × ctx × (576+512) × 2 | 1.42 TFLOP | |
| FLOPs/token, sparse (attn + indexer) | 78×64×2048×1088×2 + 21×32×128×ctx×2 | 22.2 + 22.5 = **~45 GFLOP** | **~31×** at 128K; the indexer O(L) term crosses the attention term at ~131K — fp8 scoring and B-tuning target it |

**Kernel VMEM (sparse decode, per grid step)**: q `[64, 640]` bf16 = 80 KiB; segment tile
`[comp_block, 640]` bf16 = 640 KiB at comp_block=512 (untiled 2048: 2.5 MiB); bias tile `[64, comp_block]`
fp32 = 128 KiB; scratch acc `[64, 512]` fp32 = 128 KiB + m/l; ×2 for double-buffered pipelining →
**≤ ~6 MiB worst case, context-independent** — comfortably inside 16 MiB with no ctx-scaling failure mode
(the exact wall DSV4 hit at ~12K, doc16 LATEST-8, cannot occur here).

**Indexer-scoring kernel VMEM (2b)**: q `[R, 32, 128]` bf16 = 512 KiB resident; one k tile `[B_tile, 128]`
(B_tile=2048: 512 KiB, double-buffered); score tile `[R, B_tile]` fp32 = 512 KiB — ~2.5 MiB total.

Pod-level: 992 GiB HBM − ~750 GiB FP8 weights ≈ 240 GiB KV pool; one 128K seq costs 12.9 GiB
(latent+indexer) before replication factors — per-rank sizing (docs/16 `required_kv_blocks` + indexer term,
§2.2) is what actually gates concurrency, and 1M-ctx work re-derives it.

---

## 7. Phasing — 2a → 2b → 2c, each gated

**2a — XLA-reference indexer + XLA sparse oracle (correctness, slow).**
Register the indexer KVCacheSpec (§2.2 — replaces kv_cache_manager.py:640-643; group-merge checkpoint);
indexer forward in XLA (§1.1/1.3 blocked scoring + §1.4 blocked exact top-k); resolve the rope layout (§1.2
E1→E2→E3, verdict logged); wrapper-context index carriage + IndexShare reuse (§5); sparse attention as pure
XLA: gather + masked softmax oracle over the selected set (decode) and bias-masked blocked prefill (§3.3).
Flip `DISABLE_DSA_INDEXER` (envs.py:44, default True — the mla_attention.py:343 gate) only in this path; gate
off = Stage-1 byte-identical. **Gates: S, D0, D; passkey at 4K–16K.** This phase is the reference every later
phase diffs against — it is allowed to be slow (it is DSV4's `_compressed_decode_jit` role).

**2b — Pallas indexer scoring (+ the top-k stays XLA).**
Fused q·k→relu→weighted-sum kernel (§1.3), per-block score tiles or per-block candidates to HBM; blocked
top-k merge unchanged in XLA. New `TuningKey`-style entries are NOT needed (this is a new kernel, not mla.v2;
but note mla.v2's tuned tables assume 128 q-heads — doc11 §3 — when touching the dense path for GLM's 64).
**Gates: S (vs 2a — S1 fp32 exact-modulo-ties, S2 bf16 boundary-band; §4), microbench vs the 2a blocked XLA
core** (the DSV4 analogue ran 1.4×@512 → 4.3×@8K, doc09 §4).

**2c — sparse-MLA Pallas decode consuming indices.**
The §3.1(a) gathered-segment kernel (§3.2 deltas: no sink/no inverse-rope/no local blocks, PV over
`[:, :512]`); XLA keeps gather + bias build + W_UV/o_proj. Wire as the decode path on full+shared layers
behind a `GLM_DSA_PALLAS_DECODE`-style gate (the `DSV4_PALLAS_DECODE` discipline,
deepseek_v4_attention.py:1841,2006). **Gates: K (bit-faithful vs the 2a oracle on identical selected sets),
D, D0, P (passkey ≥95% to ≥128K), TP=4==TP=1, 3/3 pod runs.**

**Verbatim-reuse estimate from `dsv4/kernel.py`** (§3.2 table): `_online_update`, `_finalize` (sink=−inf),
`_qk`, the host row-fold, the 4-D tile reshape/pad, `_MASK_VALUE`, precision selection, the
scalar-prefetch/BlockSpec scaffolding — ~70% of the compressed-tiled path; the deltas are one PV value-slice,
deleted DSV4-isms, and a simpler entry signature. The indexer scoring kernel (2b) is new code but follows the
same grid/masking/scratch idioms; nothing in Stage 2 requires a Pallas construct the DSV4 port has not
already validated on v4.

**Explicitly out of Stage 2**: MTP + `index_share_for_mtp_iteration` (Stage 3); fp8 indexer k-cache split
arrays (post-2c optimization, §2.1); prefill page-union compaction (Stage 3, measured first, §3.3); v6e/v7
native-fp8 scoring (keep every v4 gate liftable — `tpu_generation()==4` discipline).
