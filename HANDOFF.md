# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-09 — **STAGE 1 DONE on the 32-chip pod** (GLM-5.2-FP8 serves; GSM8K n=32 @ max-new
2048 = **96.9% (31/32)** — a **SMOKE test (n=32)**, NOT a scale claim (Wilson ~84–99%; needs n≥200); the
fatal OOB core-halt class CLOSED on hardware) + **STAGE-2 DSA KERNELS SILICON-VALIDATED** (single-chip GATE
2a/2b, **selected-set-EXACT** vs the HF oracle, sparse-MLA fp32 9.5e-7 / bf16 1.95e-3; **sparse passkey 100%
every depth @8K & 32K** — 16× sparsification at 32K, the SELECTION works; D0 sparse==dense in the 753B engine)
+ **STAGE-3 MTP code-complete (CPU)**.

**Honest claim line:** *first public DSA kernel on TPU — compiles + runs on v4 silicon, selected-set-exact
index parity, 100% sparse passkey @8K/32K; **≥128K NOT yet demonstrated.*** The remaining frontier is a
**KV/CACHE-LAYER problem, NOT the DSA kernel** (the kernel PASSED). Three blockers: (1) the DCP stale-stripe
**persistence** bug is **FIXED** (VllmModelWrapper step-fn / MTP `_propose` out-sharding, reviewed **SHIP**,
merged; A/B/C proved the carry is clean); (2) the residual DCP failure is the **multi-chunk-prefill packed-WRITE
of the 2nd `kv_packing=32` tile** on metal — kernel + scatter arithmetic CPU-exonerated, so the metal
write/new-KV path (the one CPU can't test) is under on-metal bisection; (3) **fp8-KV SHELVED** (two v4 Mosaic
`arith.cmpi` blockers). 128K passkey (CORRECTNESS, batch=1) and 256K throughput (needs batch/DCP/fp8) are
**different gates** — do not conflate. **Exact next work: `docs/11-pod-runbook.md` + "The exact next task" below.**

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `docs/11-pod-runbook.md`
(the next actions) → `docs/RESEARCH_LOG.md` (the dated narrative — incl. the full DCP saga) → `PLAN.md`
→ the design docs as needed: `docs/01` (DSA kernel), `docs/03` (throughput), `docs/04` (host path),
`docs/05` (KV/DCP), `docs/06` (pass-boundary crash forensics), `docs/07` (card protocol), `docs/08` (MTP),
`docs/10` (observability + the DCP debug triad), `docs/reviews/*` (the round-1–9 + DCP-fix review ledger).
DSV4 base: `~/moe-tpu/{CLAUDE,HANDOFF}.md`.

---

## State — what is DONE and hardware-validated

- **Stage-1 serving WORKS on the pod** (RESEARCH_LOG 07-07 06:40 / 15:05 + 07-08 17:00): 753B FP8 streamed
  GCS→HBM in ~10 min (runai + the sharding-derived EP filter, 32/256 experts/host), FP8 resident
  (23.06/30.75 GiB/chip), checkpoint-exact block scales, dense MLA via mla.v2 + cross-shard all-gather,
  pure TP-32 mesh. **GSM8K n=32 @ max-new 2048 = 96.875% (31/32, 1 truncation)** — a **SMOKE test (n=32)**,
  NOT a scale claim (Wilson ~84–99%; needs n≥200 to claim scale); the earlier n=32 @1024 was 87.5 (cap-driven
  misses, not the model). All
  provenance in `bench/results.db`.
- **The fatal core-halt class is CLOSED:** the OOB VMEM read in mla.v2 `pack_new_kv` at bkv-block-boundary
  decode (kv_len % 512 == 0, odd-slot alignment) — fork fix `63427f86`, merged `02e44b36`, validated on
  hardware (the 8h52m GPQA-198 run: 703,918 gen tokens, ZERO interrupts — the longest hardware validation
  yet). Forensics `docs/06`; the earlier `JAX_SHARE_BINARY_BETWEEN_HOSTS=1` attribution was CORRECTED in-log
  (Ray-dedup artifact, not the cause). Determinism also fixed (canonical ascending-position gather, fork
  `215f8ddb`): across-boot 4/4 byte-identical.
- **GPQA-Diamond n=198 (dense, DSA bypassed) — truncation-dominated** (run 48; RESEARCH_LOG 07-08 04:20):
  raw acc 52.5 vs card 91.2 is an ARTIFACT of `--max-new 4096` (140/198 = 71% truncated mid-reasoning; the
  card evaluates at a 163,840-token cap). Honest split: **completed items 50/58 = 86.2%** (within ~1σ of the
  card for n=58; caveat — the completed subset skews toward easier/short-reasoning items, so this likely
  OVERSTATES slightly). Decision: **rerun at `--max-new 16384`** on the staging branch, queued behind the DCP
  validation. All 198 items with verbatim outputs in results.db run 48.
- **Stage-2 DSA kernels SILICON-VALIDATED (single v4 chip; RESEARCH_LOG 07-08 05:25):** GATE 2a ACCEPT after
  three v4 Mosaic lowering fixes + the documented broadcast-multiply w-tile fallback (round-5's flagged
  highest-risk (1,H) spec was rejected exactly as predicted); GATE 2b ALL PASS on real MXU — A1 fp32 2.4e-7,
  **A2 selected-set-EXACT vs the HF-math oracle (0 non-tie mismatches)**, A3 bf16 0 out-of-band (S2 ε=2⁻⁸),
  GATE B sparse-MLA fp32 9.5e-7 / bf16 1.95e-3, GATE C pack_new_kv no-OOB byte-identical @ kv_len 511/512/513.
  (One probe bug — the oracle running at default bf16 MXU precision — was found ON METAL and fixed, oracle
  pinned to 'highest'; a follow-up precision audit then pinned every parity/passkey/throughput instrument.)
- **Stage-2 sparse serving VALIDATED in the 753B engine (RESEARCH_LOG 07-08 13:10 / 15:30):** D0 CLOSED —
  `GLM_DSA_MODE=pallas_decode` serves the full model correctly, accuracy == dense, deterministic. **Sparse
  passkey = 100% in every (length,depth) cell at 8K AND 32K** (72/72 needles; at 32K the DSA kernel attends
  only the 2048 indexer-selected positions — 16× sparsification, retrieval perfect); dense baseline also
  72/72. Sparse decode + sparse prefill (mbt raised 512→2048; the F1 static-elision fix is hardware-proven)
  + IndexShare all built. Verified clean-path ceiling at dcp=1 is 32K (64K@dcp=1 OOMs at compile).
- **128K — persistence FIXED, packed-WRITE bug being bisected on metal (the frontier; RESEARCH_LOG 07-08
  17:00 → 07-09 04:15 + `docs/reviews/dcp-*`). ALL of this is the KV/CACHE layer, NOT the DSA kernel (which
  PASSED):**
  - **(1) Stale-stripe persistence — FIXED.** Chunk-1's writes to the `P(BATCH,CONTEXT)`-striped cache were
    not carried into chunk-2's `execute_model` step (one dcp stripe dropped across the scheduler-step boundary;
    cache-dump DIFFER, half the rows stale at dcp=2). Fixed by pinning the **VllmModelWrapper step-fn + the MTP
    `_propose` cache `out_sharding` to `P(BATCH,CONTEXT)`** under the gate so the donated striped buffer
    round-trips with no cross-step reshard — GLM's REAL vLLM path (an earlier `get_flax_model` fix was a NO-OP
    for GLM, caught by review). Reviewed **SHIP** (`docs/reviews/dcp-vllm-fix-{on,off}gate.md`), merged to
    `glm-5.2-v4-next`; the on-metal **A/B/C localizer proved the carry is now clean** (`max|Δ|=0`, both stripes
    populated).
  - **(2) The residual DCP failure = the multi-chunk-prefill packed-WRITE of the 2nd `kv_packing=32` tile.**
    Correctness-diff + scatter-only probes localize it to **logical page 2 (positions 1024–2047)** of a
    multi-page prefill chunk. The kernel and the owner-scatter **arithmetic are CPU-exonerated** (264
    adversarial cases; write levers no_donate/flat/onehot/shard-local all dead ends), so the bug lives on the
    **metal-only write/new-KV path CPU cannot exercise** — under observability-first on-metal bisection
    (cache-dump → A/B/C → correctness-diff → scatter-only → new-KV dump; each probe halved the search). Single-
    chunk prefill is always correct (one step, no boundary).
  - **(3) fp8-KV — SHELVED.** Two independent v4 Mosaic `arith.cmpi` legalize blockers (the read-dequant was
    fixed branchless; a **2nd cmpi in the write/quantize path** persists even after the fp8→f32→bf16 rewrite) —
    systematically unsupported on v4 for now; the CPU Mosaic-routing test was not faithful to real v4 libtpu.
  - **Why the cache is the bottleneck:** the MLA latent KV cache is **replicated per-chip** (`P(BATCH)` on the
    pure-TP mesh), so one 128K sequence ≈ **11.9 GiB/chip** which + the 23 GiB/chip FP8 weights **exceeds the
    30.75 GiB/chip budget** → 128K genuinely NEEDS **DCP** (shard the cache, ÷dcp) or **fp8-KV** (halve it).
    **Passkey@128K (CORRECTNESS, batch=1) and throughput@256K (needs batch/DCP/fp8) are DIFFERENT gates** with
    different resource needs — an empirical batch=1/bf16/DCP-off fit-check may show correctness@128K decouples
    from DCP. **≥128K is NOT yet demonstrated** — `report_passkey` still refuses to call <128K a gate pass.
- **Stage-3 MTP code-complete (CPU-only so far; TPU untouched by it):** dense M1 draft-parity vs composed-HF
  (max rel Δ 3.1e-6 / top-1 100%, ~2000× under the bf16 floor) + G4 IndexShare; the engine knob `GLM_SPEC_K=k`
  (unset = byte-identical, unit-tested) + `bench/mtp_m2_check.py` (per-item identity gate + acceptance-stat
  scrape). M2 (pod greedy-equivalence, k=1→5) is runbook §7, prepped zero-turnaround (mtp-g4 + OOB fix coexist
  on `-next`).
- **Observability toolkit (docs/10):** per-step **flight recorder** (`GLM_FLIGHT_RECORDER=1`, JSON lines to
  `/tmp/glm_flight_*.jsonl`, ~15 µs/step, SIGKILL-durable), **`triage_crash.sh`** (log parse + 8-host flight
  fetch + step alignment), `GLM_LOG_STATS=1` (10 s tok/s), `DSV4_OBSERVE_COMPILES`, and the **DCP debug triad**
  — `GLM_DCP_CACHE_DUMP` (post-hoc 2-chunk cache diff) + the two fail-loud guards `GLM_DCP_ASSERT_SHARDING`
  (Guard 1: sharding round-trip) and `GLM_DCP_ASSERT_CACHE_SANITY` (Guard 2: post-prefill stale-stripe) — run
  all three together on any DCP diagnostic run. Read docs/10's field notes before interpreting any crash or DCP run.
- **Benchmark harness**: batched generation (`--batch-size`), `--protocol card|greedy` (card = temp 1.0 /
  top_p 0.95 / 163,840 cap / byte-pinned Exact-Answer system prompt; per-request seeds OMITTED on this backend
  — the round-6 F1 fix, with a loud build banner), pinned dataset revisions, GPQA content-hash shuffle,
  `glm_longctx.py` passkey ladder to 1M, and the `GLM_DCP` / `GLM_SPEC_K` / `GLM_KV_CACHE_DTYPE` engine knobs
  (all unset = byte-identical engine args, unit-tested).

## Fork branch map (`~/tpu-inference`, worktrees at `~/tpu-inference-<tag>`)

| branch | what | on origin? |
|---|---|---|
| `glm-5.2-v4` (`02e44b36`) | **pod mainline** — Stage-1 + observability + the OOB fix; the pod runs this for dense-MLA | yes |
| **`glm-5.2-v4-next`** | **integrated staging** — mainline + Stage-2 kernels (silicon-validated 2a/2b) + 2a2 + 2int + sparse-prefill + DCP + guards + **mtp-g4** (`534cd74d7`) + det + **the DCP persistence fix**, ALL gated off | yes (merge-base `886eaceb4`; latest tip in the log) |
| `glm-5.2-v4-dcppersist` (`c6da472e5`+`ce51b85ef`) | the DCP multi-chunk-prefill out-sharding fix (VllmModelWrapper step-fns + MTP `_propose`); reviewed **SHIP**; merged into `-next` | — |
| `glm-5.2-v4-obsguard` | the two DCP guards + the `GLM_DCP_CACHE_DUMP` hook (18 tests); merged into `-next` | — |
| `glm-5.2-v4-kvpack` | the DCP kv_packing CPU investigation (kernel interpret path + repro test; REFUTED the kv_packing suspect) | — |
| `glm-5.2-v4-sparse-prefill` / `-sdcp` | Stage-2 sparse-prefill + the sparse×DCP build (another session) | — |
| `glm-5.2-v4-2int`/`-2a2`/`-r5fix`/`-r6fix`/`-obs`/`-pg2`/`-dcp` | feature branches, all merged into `-next` | some |
| `glm-5.2-v4-mtp` (`6beacb5d`) | dense-MTP M1 — frozen; superseded for M2 by the mtp-g4 merge (never run M2 here: lacks the OOB fix) | no |
| `pr-g1..g6` | upstream PR series slices (`docs/02` + `docs/pr-descriptions/`; owner submits). **G5** = `pr-g5-mla-pure-tp` (`132a11f9`, TP-topology MLA). **G6** = `pr-g6-dsa-kernels` (`0ac6eb9e4`→`5bf3e5927`, the DSA kernels — the headline contribution), pushed | on fork |

## Review-round ledger (adversarial review is part of the loop — every change-set gets one)

| round | scope | reports | outcome |
|---|---|---|---|
| 1 | Stage-1 diff (4 lenses) | `stage1-{a,b,correctness,numerics}.md` | 3 HIGHs fixed (engine-boot kv_cache_spec, default-on indexer, CPU test) |
| 2 | PR #2324 TP-topology port | (fixes = fork `a429be54`) | gather-before-TuningKey, page-512 v4 gate, TPU_MIN_TOKEN_BUCKET plumbing |
| 3 | parallel-agent deliverables | `round3-*.md` | 2 HIGHs (BPE length −17.2%, `[gMASK]` claim), revision pinning, extractor fixes |
| 4 | pod bring-up series | `round4-*.md` | hardening `761ea755` |
| 5 | Stage-2 kernels | `round5-agent*.md` | fixes on `-r5fix` (`c8a51543`); their on-TPU checklists = runbook §3's expected-failure list |
| 6 | dcp, paged-indexer, observability, mtp-card, pr-branches | `round6-*.md` | fixes on `-r6fix`/`-dcp`/`-2a2` + bench F1 seed fix; the sharedbin honest-nulls CORRECTION |
| 7 | pg2 OOB fix + 2int integration | `round7-2int.md`, `round7-pg2-oob.md` | LANDED (108/108 green; F1–F3 = the `pallas_decode` on-TPU gate list, folded into runbook §3) |
| 8 | staging grafts + freeze-conformance | `round8-{staging-grafts,freeze-conformance}.md` | stranded r6 fix `755b1719` merged to `-next`; kernel-freeze conformance confirmed |
| 9 | determinism fix + MTP-M2 harness prep | `round9-{detfix,harness}.md` | det root cause (selection-order summation amplifier) fixed `215f8ddb`; M2 harness 2 MED + 8 LOW addressed |
| DCP | the DCP saga: routes, kv_packing diagnosis, persist-fix correctness/CPU-proxy, vLLM-path fix on/off-gate | `dcp-routes.md`, `dcp-diagnosis-kvpacking.md`, `dcp-persist-fix-{correctness,cpu-proxy}.md`, `dcp-vllm-fix-{on,off}gate.md` | caught the mistargeted flax fix (HIGH) → redirected to VllmModelWrapper; both on/off-gate reviews **SHIP**; validation gated behind an on-pod cache-dump MATCH |

## Operating landmines (learned the hard way — expect these)

- **GLM serves via `VllmModelWrapper` (the vLLM path), NOT `get_flax_model`.** `GlmMoeDsaForCausalLM` is in
  `_VLLM_PREFERRED_ARCHITECTURES` → a fix on the flax path is a NO-OP for GLM (the DCP mistargeted-fix
  landmine, caught only by tracing the chain of custody in review). Verify the code path before trusting a fix.
- **DCP fails on MULTI-CHUNK PREFILL only** (single-chunk always correct); the kernel/scatter/combine +
  scatter arithmetic are CPU-exonerated — the residual bug is the metal-only packed-WRITE/new-KV path.
  fp8-KV is SHELVED (two v4 Mosaic `arith.cmpi` legalize blockers) — it is not a shortcut around DCP.
- **Worktree policy: agents NEVER edit the main checkout** `~/tpu-inference` — the live pod imports it on
  every host. All feature work in worktrees on feature branches, merged via `-next`; review agents read-only.
- **The pkill self-match landmine:** un-escaped `pkill -f` patterns match the pkill/ssh command line itself;
  every launcher pattern is bracket-escaped. The stop phase also SIGKILLs the cohabiting ASPt stack (see the
  launcher's SHARED-POD POLICY — coordinate). Prefer exact-PID kills.
- **TPU access is serialized to the main session.** Helper agents set `JAX_PLATFORMS=cpu` BEFORE python starts.
- **`GLM_*` envs are trace-time + worker-side** → raylet-baked via the launcher (`EXTRA_ENVS`) AND exported on
  the driver (workers need them, not just the driver — a repeated footgun). `GLM_DSA_MODE` shapes the
  KV-cache spec — cross-host divergence = inconsistent cache topology, not just a wrong mode.
- **Relaunch after any pod crash** (leaked EngineCore / ~1800 s PG timeout) — but fetch flight files FIRST
  (`triage_crash.sh`); the launcher prunes them and a pod restart wipes /tmp.
- Flaky FP8-dequant-during-load crash (~60% on DSV4, environmental) → retry; 2 consecutive = full relaunch.
  `OMP_NUM_THREADS=1` everywhere. No pipeline parallelism. The parity harness is single-chip-only BY DESIGN.
  Buckets: the real constraint is 32-way divisibility — bucket 32 is legal and validated.
- **Durable backups**: `scripts/backup_bundle.sh` → `gs://driftbench-dsv4-uc/backups/glm-tpu/<ts>/` (verified
  bundles, ALL fork branches, results.db snapshot, logs, docs). Run it after every milestone; us-central2 ONLY.

## Target (unchanged)

`zai-org/GLM-5.2-FP8` (753B/40B, 78L, MLA + DSA index_topk=2048/32 heads/IndexShare freq 4, MTP layer 78,
1M ctx) on the 32-chip v4 pod `db-v4-64-od` ONLY. Weights staged at
`gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` (150/150 files, 755.7 GB, verified). Stage-1 gate: card-family
scores within ~1–2 pts at ≤8K ctx. Stage-2 gate: passkey ≥95% to ≥128K + bounded divergence. Stage-3: MTP
acceptance ~5.

## The exact next task

DONE and not to be redone: §0–§4 (GPQA triaged, GSM8K clean, `-next` staging switch byte-identical 4/4 on
hardware, DSA compile probe + D0 sparse==dense closed); the DCP **persistence** cache-dump has already flipped
DIFFER→MATCH (A/B/C, `max|Δ|=0`). Remaining, in order (recipes in `docs/11-pod-runbook.md`):

1. **Empirical 128K@dcp=1 fit check (do this FIRST — it may unblock correctness cheaply).** Run 128K passkey
   at **batch=1 / bf16 / DCP-off** and MEASURE the HBM: the cache-replication arithmetic predicts ~11.9 GiB/chip
   KV + 23 GiB weights > 30.75 → OOM, but confirm empirically. **If it FITS, correctness@128K decouples from
   DCP entirely** (batch=1 needs no sharding). Defer to the measured result over the arithmetic.
2. **Land the KV-capacity path for the 128K SPARSE gate.** Fix the **DCP multi-chunk-prefill packed-WRITE of
   the 2nd `kv_packing=32` tile** (fix directions: drop the scatter donation / per-page `dynamic_update_slice`
   / shard-local scatter / correct the upstream new-KV values under the 32-way token all_gather × dcp). The
   **on-pod scatter-only + new-KV-dump DIFFER→MATCH diff is the exact falsifier** — do NOT trust any passkey
   until MATCH. Then **dcp=2 (dcp=4 if OOM) 128K passkey ≥95%/depth**, then the 8K/32K/128K ladder.
3. **256K throughput A/B** (dsa-sparse vs dense) — **needs DCP or fp8-KV** (gated on step 2 landing, since
   throughput needs batch/sharding); `bench/dsa_throughput.py`, dcp≥4 — the 2nd empty gate.
4. **§7 — MTP M2** — greedy spec-decode == non-spec (`GLM_SPEC_K=1` then `5`, `bench/mtp_m2_check.py --latest`).
5. **§6 — AIME-2026 card n=30** + the **full GPQA-198 rerun @ `--max-new 16384`** (owner-gated "go").

Each step: env block, expected outcome, abort criteria, **3/3 rule** (multi-host is probabilistic; one pass ≠
done — the DSV4 worker-3 race). After each milestone: RESEARCH_LOG + HANDOFF + commit/push + backup bundle.

## Owner-gated (draft, don't do)

Upstream PR submission (`docs/02` + `docs/pr-descriptions/` — the owner submits, AI-assist disclosed;
coordinate with PR #2324/yiqiliu2), the full GPQA-198 @16K "go", any new machine/VM/TPU (never — only
same-region bucket + disk-attach to the 8 existing hosts), force-push, external comms.
