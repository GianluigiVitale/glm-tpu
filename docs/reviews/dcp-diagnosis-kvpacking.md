# Adversarial diagnosis — dense-DCP 1/dcp truncation: is it the kv_packing>1 strided read?

Reviewer: adversarial diagnostician (read-only, JAX_PLATFORMS=cpu; TPU untouched).
Date: 2026-07-08. Branch under test: `tpu-inference@glm-5.2-v4-next` (086469076).
Claim under test (RESEARCH_LOG 2026-07-08 18:20, item (2)): the gate-ON DCP silent
truncation (GSM8K short correct; passkey >=8K multi-block = haystack filler, needle
invisible, "sees ~1/4 of context") is caused by "the sharded DCP path's kv_packing=32
multi-block bitcast read" in `kernels/mla/v2/kernel.py`.

Verdict up front: **the kv_packing>1 bitcast-read hypothesis is REFUTED, not merely
downgraded.** The exact same packed read serves the working non-DCP path; the three
position-arithmetic alternatives I was asked to test are RULED OUT by hand-trace; and
the "~1/4" signature points hard at a **cross-shard / mesh-metadata whole-shard drop**,
which is the candidate the on-metal fix must target. Critically, **no CPU reproduction
can validate a kv_packing>1 fix** — the interpret path asserts `kv_packing == 1` — so an
interpret repro that "shows a bitcast truncation" is self-contradictory and, if it shows
any truncation at all, is reproducing a *different* bug than the one being fixed.

---

## 0. What the code actually does (the geometry the whole diagnosis hinges on)

- MLA KV cache shape (`kernel.py:119-142`, `runner/kv_cache.py:66-77`):
  `(total_pages, page_size_per_kv_packing, kv_packing, align128(kv_dim))`, with
  `kv_packing = envs.MLA_KV_PACKING_SIZE` **default 32** (`envs.py:404-405`) and cache
  dtype **bf16** (`kv_cache.py:34`). So the log's "kv_packing=32" is a correct read of
  the env — NOT the `32//dtype_bits` formula in the generic docstring (`kv_cache.py:107`).
- DCP block size is the LOGICAL page `P_g = block_size * dcp`
  (`runner/kv_cache_manager.py:524-525`, applied unconditionally when
  `decode_context_parallel_size>1`). Token `o` in a logical page maps to
  `(row = o // kv_packing, sub = o % kv_packing)` in dims (1,2)
  (kernel write `_update_kv_cache` `kernel.py:1556-1559,1575-1578`; XLA scatter
  `attention_interface.py:911-913`).
- `_cache_spec = P(BATCH, CONTEXT)` under the gate (`attention_interface.py:662-663`),
  `CONTEXT='dcp'` (`sharding.py:63`). `P(CONTEXT)` shards **dim-1**
  (`page_size_per_kv_packing`), NOT a raw token dim and NOT the block-id dim-0.
- Worked instance (dcp=4, logical `block_size=2048`, bf16, kv_packing=32):
  dim-1 = `align(2048,32)//32 = 64`; sharded 4 ways → 16 rows/shard;
  `P_l = 16*32 = 512` tokens/shard; `P_g = 512*4 = 2048`. Shard `s` holds rows
  `[16s,16s+16)` = tokens `o` with `o//32 ∈ [16s,16s+16)` = **tokens `[512s, 512s+512)`**.
  The kv_packing=32 interleave and the dcp shard split are BOTH row-major and land on the
  same 512-token boundary, so the physical stripe is exactly the contiguous-chunk model
  the kernel assumes. **Packing does not perturb the stripe.**

---

## 1. The formula the prompt asked me to trace by hand — RULED OUT

Kernel k_span (`kernel.py:549-552` batched, `kernel.py:705-708` per-seq):

    k_span_global = (x // P_l) * (P_l * dcp) + s * P_l + (x % P_l)

where `x` is the LOCAL linear key index, `P_l = page_size` = the *local* per-shard page
(`kernel.py:444`, = `cache.shape[1]*cache.shape[2]` inside shard_map).

Hand-trace at dcp=4, P_l=512 (P_g=2048), global position 40000:
- Owner shard (scatter, `attention_interface.py:906`): `(40000 % 2048)//512 = 1088//512 = 2`.
- Logical block `B = 40000 // 2048 = 19`; in-chunk offset `1088 % 512 = 64`.
- Local index `x = B*P_l + 64 = 19*512 + 64 = 9792`.
- Kernel formula on `x=9792, s=2`: `(9792//512)*2048 + 2*512 + 9792%512`
  `= 19*2048 + 1024 + 64 = 38912 + 1024 + 64 = 40000`. ✓ Exact round-trip.

No off-by-a-factor-of-dcp. The forward map (local→global) is the exact inverse of the
scatter's owner/offset map, and both agree with the local-length partition (§3). **This
candidate is eliminated.**

## 2. Is the block table LOCAL or GLOBAL? — GLOBAL, and that is CORRECT here — RULED OUT

`page_indices` are block ids that index cache **dim-0** (`total_pages`), which the DCP
sharding does NOT touch (`P(BATCH, CONTEXT)` shards dim-0 by BATCH/replicated over dcp,
and dim-1 by dcp). Every dcp shard holds its dim-1 slice of *the same* physical page, so
a single global block id is valid on every shard (kernel DMA `kernel.py:1054,1111`;
scatter `attention_interface.py:908-910`; both use `pages_per_seq = num_page_indices //
max_num_seqs`, `kernel.py:460`). The stripe is intra-block (dim-1), not inter-block, so
"a global block id indexed into a local stripe" never happens. **Eliminated.**

## 3. Do ranks get LOCAL or GLOBAL kv_len? — BOTH, correctly split — RULED OUT

- Local length (`attention_interface.py:873-878`):
  `n_s = (L//P_g)*P_l + clip(L%P_g - s*P_l, 0, P_l)`, passed as `kv_lens_local`
  (kernel arg `kv_lens`, used for DMA size + `num_bkv`, `kernel.py:1027,1035,1927`).
- Global length passed separately as `kv_lens_global` (`attention_interface.py:936`),
  used ONLY for the causal mask `q_span`/`k_span` (`kernel.py:392,560,701`).

Hand-check L=40001, dcp=4, P_l=512: locals = {10240, 10240, 9793, 9728}, **sum = 40001 = L**
(exact partition). Shard 2 local length 9793 > x=9792 → the needle at global 40000 is
in-range on its owner. The kernel uses local length for the loop/DMA and global length for
the mask — the correct split. **Eliminated.**

Corollary (stale-tail safety): local rows beyond `kv_len_local` in the final block are not
DMA'd but are included in the einsum; the strided k_span maps every such row to a global
position `>= L`, so the mask kills them (checked: shard-2 local 9793 → global 40001 > q_span
40000 → masked). No leakage. Correct at multi-block.

## 4. The claim itself — kv_packing=32 multi-block bitcast read — REFUTED

The DCP gate-ON path (`history_only=True`) exercises exactly ONE piece of kv_packing>1
machinery: the packed cache **read** `load_bkv` (`kernel.py:1861-1886`). Everything else
packed is skipped: `history_only` bypasses `_pack_new_kv` and the cache write-back
(`kernel.py:2014-2018, 2034`). So the only way the claim can be true is if `load_bkv`'s
depack is wrong under DCP.

Three independent facts refute that:

4a. **`load_bkv` is shared with the working non-DCP path and is validated at the identical
    packing.** MLA_KV_PACKING_SIZE=32 + bf16 is a global default applied whether or not DCP
    is on (`kv_cache.py:66-77`). The dense non-DCP GSM8K n=32 run (RESEARCH_LOG 17:00, 96.9%)
    and the short-context DCP run both go through this same `load_bkv` bitcast at
    kv_packing=32/bf16 and produce correct output. If the depack were wrong, dense GSM8K
    would be wrong too. It is not. The read is silicon-validated.

4b. **DCP introduces no new packed-write code, and the one packed write it does use (the XLA
    scatter) matches the kernel's native convention.** Scatter (`attention_interface.py:
    911-913`): `row = in_page // kv_pack`, `sub = in_page % kv_pack`. Kernel native write
    `_update_kv_cache` (`kernel.py:1556-1578`): `word_in_page = (offset % page_size) //
    kv_packing`, whole-row (all `kv_packing` sublanes) copy. Same `o → (o//kv_pack,
    o%kv_pack)` row-major convention that `load_bkv`'s reshape inverts (`kernel.py:1874-1878`,
    linear index `= row*kv_packing + sub = in_page`). Write and read agree by construction.

4c. **The bug cannot be reproduced on CPU — so no CPU "bitcast repro" is even possible.**
    Interpret mode forces `assert kv_packing == 1` in every packed path:
    `load_bkv` (`kernel.py:1863`), `load_bq` (`1813,1837`), `_pack_new_kv`
    (`2027`), output store (`2208`); the module header states the interpret formulations are
    "only valid at q/kv packing == 1" (`kernel.py:36-41`). The kv_packing>1 bitcast is not
    compiled under interpret at all.

Consequence for the running kv_packing-fix agent (the core of my charge):
- A CPU/interpret repro runs at kv_packing==1, so it CANNOT contain the alleged bitcast.
  If it nonetheless shows truncation, that truncation is produced by kv_packing-**independent**
  DCP logic (position/len/mask/scatter/combine) — i.e., it is a *different* bug, and a bitcast
  fix will not touch it on metal.
- If the repro at kv_packing==1 shows NO truncation, it proves nothing about the metal
  failure and cannot falsify or validate a bitcast fix.
- Either branch makes "CPU-reproduced kv_packing bitcast truncation" self-contradictory.

## 5. What the "~1/4 of context" signature actually points at (ranked #1)

"Model sees only ~1/4 of context" with dcp=4 is the fingerprint of a **whole-shard drop**:
each query ends up attending only to the keys on its own stripe (1/dcp of the context)
because the cross-shard combine is not merging the other dcp-1 partials. Suspects, in the
order the on-metal fix should check them:

5a. **Cross-shard LSE combine / dcp mesh-axis reduction** (`_dcp_lse_combine`,
    `attention_interface.py:531-551`; `pmax`/`psum` over `axis_names = _dcp_axes =
    (ShardingAxisName.CONTEXT,) = ('dcp',)`, `attention_interface.py:645-646,949`). If, at
    runtime, `'dcp'` is not a genuine size-4 mesh axis inside the shard_map (the H-MESH worry
    the 18:20 entry *claims* it refuted), the `psum`/`pmax` reduce over a degenerate axis and
    every shard returns its own local normalized partial = its own 1/dcp of the keys. That is
    precisely "~1/4". This is invisible on a single-device CPU (the collective is identity).
    NOTE the "short works" evidence does not clear this: short GSM8K (well under one logical
    page at some configs, or exercised largely via the dense non-DCP run) does not stress the
    same multi-block partial-merge accumulation; and if the combine were a *total* no-op short
    would also break — so the failure is likely a partial/accumulation interaction, not a flat
    no-op (see 5b).

5b. **`num_bkv>1` online-softmax accumulation feeding the combine.** The combine is only
    exact if each shard's emitted `out = acc/l` and `lse = m + log(l)` reflect ALL local
    blocks (`kernel.py:2190,2202`). This accumulation across `bkv_idx` is the discriminator
    between short (num_bkv==1, works) and long (num_bkv>1, fails). If `m`/`l` carry or the
    empty-shard fake-iteration handling (`kernel.py:1928-1940`) is wrong specifically at
    num_bkv>1 under DCP, the per-shard `lse` is miscomputed and the combine weights collapse
    toward one shard → "~1/4". Packing-independent; reproducible on CPU ONLY with a real
    multi-device dcp mesh (not single-device interpret).

5c. **Runtime metadata contract at multi-block.** `md.*` (seq_lens, page_indices,
    query_start_loc) are `P(ATTN_DATA)` (`attention_interface.py:713-716`), replicated over
    dcp. The fix must confirm that at multi-block the page tables/kv_lens handed in are the
    `block_size*=dcp` (logical-P_g) ones, consistent with the striped allocation. A mismatch
    between the scheduler's logical block size and the metadata actually threaded to the
    shard_map would truncate by a dcp factor and is packing-independent.

## 6. A distinct, already-documented failure — not this one

The gate-OFF 16.5 MB > 16 MB VMEM OOM at logical page 2048 (RESEARCH_LOG 18:20) is a
**compile-time** RESOURCE_EXHAUSTED, a different failure from the gate-ON silent "~1/4".
Gate-ON survives it because inside the shard_map the kernel sees the LOCAL page (P_l=512 →
smaller bkv buffer), which is why gate-ON runs and mis-answers while gate-OFF fails to
compile. Do not conflate the two.

---

## 7. Ranking (for the gate-ON silent multi-block "~1/4" truncation)

| # | Candidate | Likelihood | Why | CPU-reproducible? |
|---|---|---|---|---|
| 1 | Cross-shard LSE combine / `'dcp'` mesh-axis reduction not reducing (5a) | **High** | Exact 1/dcp "sees ~1/4" fingerprint | Only with a REAL multi-device dcp mesh; NOT single-device interpret |
| 2 | num_bkv>1 per-shard `lse`/accumulation feeding the combine (5b) | **Medium-High** | Matches the short-works/long-fails discriminator | Only with real multi-device dcp mesh, kv_packing==1 |
| 3 | Runtime metadata (logical vs local block_size) at multi-block (5c) | **Medium** | Packing-independent dcp-factor truncation | Yes, integration test with real block tables |
| 4 | kv_packing>1 bitcast read `load_bkv` (THE CLAIM) | **Refuted/Low** | Shared with & validated by working non-DCP kv_packing=32 path; scatter convention matches; not exercised under interpret | **No** — interpret asserts kv_packing==1 |
| 5 | Position formula off-by-dcp (prompt cand.1) | **Eliminated** | Exact round-trip hand-trace §1 | n/a |
| 6 | Block-table global-vs-local (prompt cand.2) | **Eliminated** | Stripe is dim-1; block id indexes unsharded dim-0 §2 | n/a |
| 7 | kv_len global-vs-local (prompt cand.3) | **Eliminated** | Local formula partitions global exactly; split correct §3 | n/a |
| 8 | VMEM page-2048 OOM | **Different bug** | Gate-OFF compile error, not gate-ON silent 1/4 §6 | n/a |

## 8. Gate the running kv_packing-fix agent's CPU reproduction MUST pass to be the RIGHT bug

1. **It must run a real dcp>=2 mesh** (multiple CPU devices via
   `XLA_FLAGS=--xla_force_host_platform_device_count>=4`, an actual `shard_map` over a
   `'dcp'` axis) — NOT single-device `_INTERPRET`. A single-device repro cannot express the
   cross-shard combine (candidate #1/#2), which is the signature.
2. **It must reproduce "~1/4 of context / needle invisible" at kv_packing==1**, at
   num_bkv>1 (>= 2 logical blocks per shard). If it does, the bug is #1/#2/#3
   (combine/accumulation/metadata) — fix THERE; a bitcast change will not help on metal.
3. **If the repro only reproduces at kv_packing>1**, it is by definition NOT a CPU repro
   (interpret forbids kv_packing>1); it can only be shown on TPU, and even then must first be
   distinguished from #1/#2/#3 by an on-metal control that pins kv_packing behaviour while
   holding the mesh fixed.
4. **Falsifier for the claim:** on metal, dump per-shard `kv_lens_local`, `kv_lens_global`,
   the first/last `k_span` per bkv block, and the HLO collective count for the dcp axis at an
   8K passkey. If `kv_lens_local` sums to the global length, the k_spans cover the full
   [0,L), and the dcp `psum`/`pmax` collectives are present with axis size == dcp, then the
   arithmetic and combine are correct and the residual must be the packed read — only then is
   the claim viable. My prediction: the collective/metadata check trips first.

**Bottom line:** the fix effort is currently aimed at a read path that the working non-DCP
run already validates and that no CPU test can even exercise. Re-aim it at the cross-shard
combine / dcp mesh-axis reduction and the multi-block per-shard `lse` accumulation, and
require the reproduction to run on a real multi-shard mesh at kv_packing==1 before trusting
any "it reproduces on CPU" claim.
