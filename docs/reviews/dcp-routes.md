# Adversarial route map — the 128K passkey gate + 256K throughput gate

**2026-07-08. READ-ONLY diagnosis (JAX_PLATFORMS=cpu; TPU never touched).** Enumerates and ranks every
route to (G1) passkey ≥95% @128K and (G2) the ≥256K dense-vs-sparse throughput A/B, with HBM/VMEM math,
effort, and independence from the dense DCP kernel — so the fix effort has fallbacks if the primary fails.
Bases: fork worktrees under `~/tpu-inference-*`; live branch `glm-5.2-v4-next` @ `086469076`.

---

## 0. Headline verdict (read this first — it corrects the task's framing)

1. **The "VMEM sub-page tiling required for 256K regardless" premise (route c) is REFUTED.** Under the
   **gate-ON** DCP path the MLA decode kernel *never* sees the logical page `P_g = 512·dcp`. It sees the
   **per-shard local page `P_l = 512`, invariant at every dcp**, because the `P(BATCH, CONTEXT)` cache
   sharding splits the in-page token dim `dcp`-ways and hands each shard a 512-token slice. Proof in code:
   `attention_interface.py:869-871` reads `cache.shape` *inside* the shard_map and computes
   `_p_l = _psz_per_pack * _kv_pack` (= 512) as the local page and `_p_g = _p_l * _dcp_size` **only as a
   scalar for the global-position mask** (`kernel.py:545-552, 705-708`) — `_p_g` is never a buffer
   dimension. The `bkvc` VMEM scratch (`kernel.py:2658-2664`) is sized from the *local* page. So
   256K@dcp=8 gate-ON has the **same VMEM profile as the working 32K@dcp=1** (`bkvc` ≈ 4.7 MB ≪ 16 MB).
   **Tiling is not on the critical path for either gate.**

2. **The `p_2048` OOM in RESEARCH_LOG 18:20 is a GATE-OFF artifact, not a gate-ON blocker.** With
   `GLM_MLA_DCP` unset the cache spec stays `P(BATCH)` (`attention_interface.py:662-663`) so the striped
   cache is **all-gathered to the full 2048-token logical page on every chip** — which both (a) blows VMEM
   and (b) re-materializes the full cache, destroying the ÷dcp saving. The overflowing buffer is
   `bkvc_double_buf = (2, batch=4, 64+2, kv_pack=32, lkv=512)` bf16 = **17,301,504 B = 16.5 MB > 16 MB**,
   which reproduces the reported `Allocation (size=17301504)` **to the byte**. This path is doubly dead
   (VMEM *and* HBM); confirm route (d) closed.

3. **The single real gate-ON DCP blocker is a CORRECTNESS bug, not memory:** the sharded DCP path's
   `kv_packing=32` multi-block packed read (`kernel.py:1317-1330` bitcast merge; interpret pins
   `q_packing==1` at `kernel.py:1811-1837`, so CPU tests cannot exercise it). This is exactly what
   RESEARCH_LOG 17:00 saw: gate-ON dcp=4 serves short ctx correctly but returns `pred=None` at 8K-128K
   (each shard reads ~1/4 of context wrongly). This is route (a)'s target. **It is real; keep the primary
   pointed at it.**

4. **A second, quieter constraint governs which dcp actually works for DENSE: the ~1.6 GiB program
   headroom + the empirical "≤32K local context per shard" ceiling.** Dense 32K@dcp=1 compiles+serves;
   dense 64K@dcp=1 is E1000 CompileTimeHbmOom (RESEARCH_LOG 17:40). Under DCP each shard processes
   `ctx/dcp` local tokens, so **dense 128K needs dcp≥4 (32K local), dense 256K needs dcp≥8 (32K local)** —
   *not* the pool-minimal dcp=2/dcp=4. **This means the primary's stated `dcp=2` is pool-sufficient but
   program-risky; recommend validating route (a) at dcp=4.** The sparse path sidesteps this entirely
   (§below), which is why it is the more robust vehicle.

---

## 1. The two gates and their true binding constraints

| Gate | What it needs to FIT | What it needs to be CORRECT | What it needs to be FAST |
|---|---|---|---|
| **G1 128K passkey ≥95%** | 128K latent KV per chip ≤ free HBM | attention reads the whole 128K correctly | (not gated on speed) |
| **G2 ≥256K throughput A/B** | 256K latent KV per chip ≤ free HBM | dense==sparse selected-set | sparse decode measurably > dense |

Free HBM/chip = 30.75 − 23.06 (weights+runtime) ≈ **7.7 GiB**, of which a KV pool takes some and the
**attention program working set** needs the rest (~1.6 GiB at a 6.1 GiB pool — the margin that E1000'd at
64K@dcp=1). Both gates are **capacity-first**; G1 adds a correctness dependency, G2 adds a speed one.

Route (g) SPARSITY DOES NOT REDUCE CAPACITY — confirmed dead as a capacity lever: `glm_dsa_indexer.py:54`
"every token scores over its full cached history", so the indexer selection needs **all** positions in
cache; `index_topk=2048` only bounds what MLA *reads*, never what is *stored*. Sparsity's value is
**program-headroom** (below), not footprint.

---

## 2. HBM / VMEM math reference (all verified numerically this session)

**Per-token latent (padded mla.v2 layout, bf16):** `align(512+64,128)=640 × 2 B × 78 layers = 99,840
B/token/chip` (= bench const `KV_BYTES_PER_TOKEN`, `dsa_throughput.py:104`). Replicated across TP at
dcp=1; divided by dcp under gate-ON DCP.

| ctx | bf16 dcp1 | dcp2 | dcp4 | dcp8 | **fp8 dcp1** | fp8 dcp2 | fp8 dcp4 |
|---|---|---|---|---|---|---|---|
| **128K** | 12.19 GiB | 6.09 | 3.05 | 1.52 | **6.09** | 3.05 | 1.52 |
| **256K** | 24.38 GiB | 12.19 | 6.09 | 3.05 | **12.19** | 6.09 | 3.05 |

**dcp thresholds (which dcp actually clears each gate):**

| | pool-only (KV ≤ ~6 GiB) | **+ dense program (≤32K local/shard)** | + sparse program (ctx-independent) |
|---|---|---|---|
| 128K bf16 | dcp≥2 | **dcp≥4** | dcp≥2 |
| 256K bf16 | dcp≥4 | **dcp≥8** | dcp≥4 (dcp≥8 for headroom) |
| 128K fp8 | dcp≥1 (pool) | dcp≥2 (program) | dcp≥1 pool / dcp≥2 headroom |
| 256K fp8 | dcp≥2 | dcp≥4 | dcp≥2 |

**VMEM (`bkvc_double_buf`, the buffer that OOMs), v4 limit 16 MiB = 16,777,216 B:**
`2 · batch · (bkv_p·(page/kv_pack) + 2) · kv_pack · lkv · 2 B`, batch = `decode_batch_size=4`
(`attention_interface.py:832`).

| path | page seen by kernel | `bkvc` | fits 16 MB? |
|---|---|---|---|
| gate-ON DCP, any dcp | **512 (local)** | **4.72 MB** | ✅ (same as working 32K@dcp1) |
| gate-OFF gather, dcp=4 | 2048 (gathered) | **16.5 MB** | ❌ = reported `17301504` |

---

## 3. Ranked route table

Ranked by **robustness of reaching the gates**, with feasibility, math, effort, and independence from the
dense DCP kernel (the `kv_packing` bitcast fix).

### R1 — SPARSE × DCP (`glm-5.2-v4-sdcp`, `fd3dc45c`) — most robust; the G2 vehicle by definition
- **Reaches:** G1 (128K sparse passkey) **and** G2 (it *is* the sparse arm of the throughput A/B).
- **Feasibility:** HIGH. Code-complete, round-10 reviewed, 580-line test suite
  (`tests/layers/vllm/test_glm_dsa_sparse_dcp.py`). Sparse decode already 100% @32K@dcp=1 on metal
  (RESEARCH_LOG 15:30). Mechanism: replicated selection → per-shard **owner-masked XLA gather**
  (`gather_kv_segment`, `dcp_local_segment_indices`) → LSE-emitting 2c kernel → cross-shard softmax
  (`mla_attention.py` sdcp diff).
- **HBM:** pool per §2; **program is ctx-independent** — it materializes a `[R, 2048, 640]` segment, not
  the full history, so the 32K-local ceiling that binds dense does **not** apply. 256K@dcp=8 program ≈
  256K@dcp=4 program ≈ 8K program. This is why sparse is HBM-robust where dense is not.
- **Effort:** on-metal bring-up only (Mosaic lowering of `dsa_sparse_decode` inside shard_map inside cond;
  HLO aliasing audit — round7-2int gate list). ~1-2 pod sessions.
- **Independence from the dense DCP kernel:** **HIGH.** Reads the striped cache via **XLA gather**, not the
  mla.v2 packed multi-block bitcast — so it is **immune to route (a)'s `kv_packing` bug**. Shares only the
  DCP striping (dcpfix, pinned) + the LSE-combine helpers (`_dcp_lse_combine`, `attention_interface.py:531`).
- **Caveat:** the round7-2int F1/F2 cost findings (sparse branch scales with the *padded token bucket*, not
  live rows) must be honored — run at `TPU_MIN_TOKEN_BUCKET=32`; and MTP spec-decode forces the dense
  fallback (F4), so G2's sparse arm and Stage-3 MTP cannot be measured in the same run.

### R2 — DENSE DCP + kv_packing fix (route a), **at dcp=4** (corrected from dcp=2) — the correctness anchor + G2 A-baseline
- **Reaches:** G1 (128K dense passkey — the reference the sparse path is validated against) and G2's
  **dense A-baseline**.
- **Feasibility:** MED-HIGH, contingent on the on-metal `kv_packing=32` bitcast fix landing. This is the
  one genuine gate-ON blocker (§0.3).
- **HBM:** 128K@dcp=4 = 3.05 GiB/chip pool, 4.6 GiB headroom, **32K local/shard = the proven program
  size**. 128K@dcp=2 = 6.09 GiB/chip, only ~1.6 GiB headroom, **64K local/shard = the size that E1000'd at
  dcp=1** → **dcp=2 is the pool-minimal but program-risky config; prefer dcp=4.** 256K@dcp=8 = 3.05 GiB,
  32K local → same safe program.
- **VMEM:** non-issue (§0.1, §2) — local page 512 at any dcp.
- **Effort:** the bitcast/packed-read fix is on-metal kernel work (interpret can't cover packing>1),
  ~1 day, as the design agent scoped.
- **Independence:** this **is** the dense DCP kernel; R1 does not depend on it. If R2's bitcast fix proves
  hard, R1 still delivers both gates (dense A-baseline is a *nice-to-have* comparator for G2, not required
  for G1).

### R3 — fp8 MLA KV as a ×2 MULTIPLIER composed with DCP (route b, RE-SCOPED) — NOT a dcp=1 replacement
- **Task's premise ("fp8 → 128K@dcp=1 no-DCP") is REFUTED.** The arithmetic the agent is checking
  (6.1 fp8-pool + 23 weights = 29.16 < 30.75) is **pool-only and true but insufficient**: it ignores the
  program working set. **Direct counter-evidence:** 64K@dcp=1 bf16 already E1000'd with an *equivalent
  6.1 GiB pool* and ~1.6 GiB headroom (RESEARCH_LOG 17:40). fp8@128K@dcp=1 has the **same 6.1 GiB pool /
  same headroom** but a **128K-long program (≥ the 64K one that failed)** → it inherits the OOM. fp8-KV
  shrinks the *cache*, not the *attention program*, and the program is the binding constraint at dcp=1.
- **Correct role:** fp8 is a clean **×2 KV multiplier on top of DCP** (docs/05 S3): fp8+dcp=2 → 256K at
  6.09 GiB/chip; fp8+dcp=4 → 256K at 3.05 GiB with 32K-safe local ctx; fp8+dcp≥16 → 1M in sight.
- **Feasibility:** MED, gated on: (i) the PR #2324 **NaN-under-EP with fp8 KV** root cause
  (docs/05 §5; suspected same −inf/empty-shard class as the DCP LSE floor), (ii) the streaming-loader
  `fp8` support. Compute is ready: `quantize_kv` (`flash_attn_mla.py:196-200`) + the v4 per-tile upcast
  `_upcast_kv_for_v4` (`kernel.py:434, 513, 658, 680, 773`) already exist.
- **Effort:** 1-2 weeks (root-cause + loader + accuracy re-validation).
- **Independence:** orthogonal multiplier — composes with R1 or R2; not required for either gate at ≤256K
  bf16 (dcp alone fits). **Reserve for 256K-at-lower-dcp headroom or the 1M endpoint.**

### R4 — VMEM sub-page tiling of the MLA decode kernel (route c) — NOT NEEDED for either gate
- **Refuted as "required for 256K regardless" (§0.1).** Gate-ON kernel sees local page 512 at every dcp;
  256K@dcp=8 `bkvc` = 4.72 MB. There is no gate-ON config in which the kernel sees `P_g`.
- **When it would ever matter:** only to rescue the gate-OFF gather (route d, HBM-dead) or to raise
  `decode_batch_size` above 4 (the ×4 factor in the 17.3 MB buffer). Neither is on the path to G1/G2.
- **Scope if ever needed:** add an inner sub-page bkv loop so `bkv_sz < page_size` (today `bkv_p` is an
  integer *page* count ≥1, `kernel.py:442-443, 2615`); ~medium kernel change. **Do not spend effort here
  for these gates.**

### R5 — smaller vLLM block_size (route e) — settable, but a NON-fix for the gates
- **block_size IS a settable knob, NOT hard-locked.** `tpu_platform.py:290` overrides it **only** `if not
  cache_config.user_specified_block_size` — a user `--block-size N` is honored. The v4 default is 512
  (`get_page_size`, `flash_attn_mla.py:50-69`, returns 512 for v4 + `kv_lora_rank>256`). The scheduler then
  applies `block_size *= dcp` (`kv_cache_manager.py:525`).
- **Why it does not help the gates:** gate-ON local page = block_size already, and 512 fits VMEM at any dcp
  (§0.1). Smaller block_size would only shrink the gate-OFF gathered page — but gate-OFF is HBM-dead. It
  also *worsens* short-seq fragmentation (allocation granule = block_size·dcp). **Useful only as a tuning
  knob; not a route to G1/G2.** (Constraint if used: `align(block_size, kv_packing)` and page%128 in the
  transpose path, `kernel.py:128, 139`.)

### R6 — reduce cache width 640→576 (route f) — dead for DCP, and insufficient alone
- **Insufficient alone (correct):** 640→576 is ×1.11; 128K still ≈ 11 GiB/chip at dcp=1 ≫ 7.7 free. Needs
  a 2× lever (dcp), not 11%.
- **Dead for DCP specifically:** the unpadded-576 layout **is** `MLA_TRANSPOSE_KV_CACHE=1`, and the DCP
  gate hard-raises `NotImplementedError` with transpose on (`attention_interface.py:652-655`,
  `kernel.py:2547-2549` — "the transposed layout stripes a different dim"). The non-transpose layout must
  128-align the last dim (`kernel.py:141`), and 576 is not a multiple of 128 (640 is 5·128) — so you
  **cannot** store 576 in the DCP-compatible layout. **Closed.**

### R7 — gate-OFF gather (route d) — CONFIRMED DEAD (both VMEM and HBM)
- VMEM: `bkvc` = 17,301,504 B at gathered page 2048 (§0.2). HBM: re-materializes the full cache on every
  chip → no ÷dcp saving → 128K still 12.2 GiB/chip ≫ free. **Doubly dead. Do not pursue.**

---

## 4. The ONLY combos that fit 256K (G2), and why

256K bf16 latent = 24.38 GiB/chip at dcp=1. **No single lever reaches it; you need dcp AND a program that
does not scale with ctx.**

| combo | 256K KV/chip | dense program | verdict |
|---|---|---|---|
| dcp=4 bf16 | 6.09 GiB | 64K local/shard → E1000 risk | **capacity ok, dense program risky** |
| **dcp=8 bf16** | **3.05 GiB** | **32K local/shard → proven** | ✅ **dense A-baseline (bench default)** |
| **dcp=8 bf16 + SPARSE** | **3.05 GiB** | **ctx-independent (2048 seg)** | ✅ **the G2 sparse arm — recommended** |
| dcp=4 + fp8 | 3.05 GiB | 32K local (fp8 pool frees headroom) | ✅ if fp8 lands (R3) |
| dcp=2 + fp8 | 6.09 GiB | 64K local → risky | capacity ok, program risky |

So the **unique robust 256K combo is dcp=8 × sparse** (the bench's `GLM_DCP=8` + `GLM_DSA_MODE=
pallas_decode`, `dsa_throughput.py` header). Tiling is absent from every winning row (§0.1). fp8 only
*lowers the required dcp*; it never substitutes for dcp.

---

## 5. Recommended primary + fallback ordering

1. **PRIMARY (keep running): route (a) dense-DCP `kv_packing` bitcast fix** — it is the one real gate-ON
   blocker and the passkey correctness anchor. **Correction: gate the 128K validation at dcp=4, not
   dcp=2** (dcp=2 is 64K-local/shard = the E1000 program size; dcp=4 is 32K-local = proven). Keep dcp=2 as
   an HBM-tight stretch only after dcp=4 passes.
2. **CO-PRIMARY / strongest fallback: R1 sparse × DCP (`sdcp`)** — bring it up in parallel. It reaches
   **both** gates, is **independent of route (a)'s bitcast** (XLA gather, not the packed kernel), sidesteps
   the dense 32K-local ceiling (ctx-independent program), and **is the G2 sparse arm by definition**. If
   route (a) stalls on the on-metal bitcast, sparse×DCP alone still clears G1 (sparse passkey) and G2.
3. **256K (G2): dcp=8 × {dense baseline, sparse}** per §4. No tiling, no fp8 needed for bf16@256K.
4. **fp8 (R3): reserve as the ×2 multiplier** — pursue only to (a) buy 256K at dcp=4 headroom or (b) reach
   1M, and only after root-causing the #2324 NaN-under-EP. Do **not** pursue it as a dcp=1 shortcut — the
   64K@dcp=1 E1000 refutes that.
5. **Do NOT spend effort on:** R4 tiling (not needed), R5 block_size (non-fix), R6 576-width (dead for DCP),
   R7 gate-off (dead). Redirect any tiling effort to the R1/R2 on-metal bring-up.

**One-line summary for the fix effort:** the gates are blocked by a *correctness* bug (dense DCP packed
read) and a *capacity+program-headroom* wall (dcp≥4 for 128K dense, dcp≥8 for 256K), **not** by VMEM
tiling. Sparse×DCP is the independent, more-robust path that clears both; fp8 is a later multiplier, not a
DCP replacement.

---

## 6. On-TPU checks that would confirm/refute this analysis

1. **§0.1 (the load-bearing claim):** dump the MLA decode scope name on a gate-ON dcp=4 run — expect
   `MLA-…-p_512-…`, NOT `p_2048`. If it shows `p_2048`, my local-page conclusion is wrong and tiling
   re-enters. (Grep the compile log or `DSV4_OBSERVE_COMPILES`.)
2. **§0.4 (dense program ceiling):** dense 128K@dcp=4 (32K local) should compile where 128K@dcp=2 (64K
   local) E1000s — the discriminating test for the "≤32K local" rule.
3. **R1 independence:** sparse×DCP@128K passes with route (a)'s bitcast fix *absent* (it uses XLA gather).
4. **R3 refutation:** fp8@128K@dcp=1 E1000s at compile (same class as 64K@dcp=1 bf16), proving fp8 is not a
   dcp replacement.
