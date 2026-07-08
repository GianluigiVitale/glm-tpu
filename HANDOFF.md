# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-08 (night) — **STAGE 1 IS DONE on the 32-chip pod** (GLM-5.2-FP8 serves; GSM8K
n=32 @ max-new 2048 = **96.9%** truncation-free; the fatal OOB core-halt class CLOSED on hardware) +
**STAGE-2 DSA KERNELS SILICON-VALIDATED** (single-chip GATE 2a/2b, selected-set-exact vs the HF oracle;
**sparse passkey 100% every depth @8K & 32K**; D0 sparse==dense closed in the 753B engine) +
**STAGE-3 MTP code-complete (CPU)**. The frontier is **128K via DCP**: the multi-chunk-prefill
cache-persistence bug is fixed on GLM's real path (VllmModelWrapper step-fn / MTP `_propose`
out-sharding), independently reviewed **SHIP**, merged to `glm-5.2-v4-next` — **on-metal validation
(the cache-dump must flip DIFFER→MATCH, then dcp=2 128K passkey) is the IN-PROGRESS next action.**
**The exact next work is scripted: `docs/11-pod-runbook.md` — execute from the DCP-validation step
(§5d/§5) onward.**

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
  pure TP-32 mesh. **GSM8K n=32 @ max-new 2048 = 96.875% (31/32, 1 truncation)** — the truncation-free
  quality signal at scale; the earlier n=32 @1024 was 87.5 (cap-driven misses, not the model). All
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
- **128K via DCP — fix MERGED, on-metal validation IN PROGRESS (the frontier; RESEARCH_LOG 07-08 17:00→21:55
  + `docs/reviews/dcp-*`):** DCP (`GLM_DCP=N` + `GLM_MLA_DCP=1`) serves short contexts correctly but dropped
  a stripe on long ones. Observability-first root cause: the **multi-chunk-prefill KV-cache PERSISTENCE**
  bug — chunk-1's writes to the `P(BATCH,CONTEXT)`-striped cache are not carried into chunk-2's `execute_model`
  step (cache-dump: DIFFER, `max|Δ|=5.44`, exactly 1024/2048 rows stale = one dcp stripe at dcp=2). NOT the
  kernel / scatter / combine (all CPU-exonerated across the kvpack + mla_dcp suites). Fix: pin the
  **VllmModelWrapper step-fn + the MTP `_propose` cache `out_sharding` to `P(BATCH,CONTEXT)`** under the gate
  so the donated striped buffer round-trips with no cross-step reshard — on GLM's REAL vLLM path (an earlier
  `get_flax_model` fix was a NO-OP for GLM, caught by review 21:55). Independently reviewed **SHIP** (off-gate
  byte-identical on the working path; on-gate correct — `docs/reviews/dcp-vllm-fix-{on,off}gate.md`), merged to
  `glm-5.2-v4-next`. **IN PROGRESS:** the on-pod 2-chunk-vs-1-chunk cache-dump must flip **DIFFER→MATCH**
  (`n_diff_rows==0`, 3/3) — do NOT trust any passkey until MATCH — then **dcp=2 (dcp=4 if OOM) 128K passkey
  ≥95%/depth**. 128K genuinely NEEDS DCP (1 seq @dcp=1 = 11.9 GiB latent KV > free HBM). **fp8-KV route
  RETIRED** (it would fit 128K@dcp=1 but hits a v4 Mosaic `arith.cmpi` legalize bug at both 8K and 128K).
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
- **DCP failed on MULTI-CHUNK PREFILL only** (single-chunk was always correct); the kernel/scatter/combine are
  CPU-exonerated. fp8-KV is RETIRED (v4 Mosaic `arith.cmpi` compile bug) — it is not a shortcut around DCP.
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

**Execute `docs/11-pod-runbook.md` from the DCP-validation step onward** (§0–§4 are DONE: GPQA triaged,
GSM8K clean, `-next` staging switch byte-identical 4/4 on hardware, DSA compile probe + D0 sparse==dense closed):

1. **§5d — DCP cache-dump falsifier (do this FIRST).** Relaunch with `GLM_MLA_DCP=1` and the
   `GLM_DCP_CACHE_DUMP` hook + the two guards baked into the raylet env; the 2-chunk vs 1-chunk layer-0
   cache-dump must flip **DIFFER→MATCH** (`max|Δ|≤1e-3`, `n_diff_rows==0`), 3/3. Do NOT trust any passkey until
   MATCH is observed (a MATCH first, passkey second — the review's hard ordering).
2. **§5/§5c — 128K passkey at dcp=2** (dcp=4 if OOM), gate P = retrieval ≥95%/depth; then the 8K/32K/128K ladder.
3. **256K throughput A/B** (dsa-sparse vs dense), dcp≥4, `bench/dsa_throughput.py` — the 2nd empty gate.
4. **§7 — MTP M2** — greedy spec-decode == non-spec (`GLM_SPEC_K=1` then `5`, `bench/mtp_m2_check.py --latest`).
5. **§6 — AIME-2026 card n=30** + the **full GPQA-198 rerun @ `--max-new 16384`** (owner-gated "go").

Each step: env block, expected outcome, abort criteria, **3/3 rule** (multi-host is probabilistic; one pass ≠
done — the DSV4 worker-3 race). After each milestone: RESEARCH_LOG + HANDOFF + commit/push + backup bundle.

## Owner-gated (draft, don't do)

Upstream PR submission (`docs/02` + `docs/pr-descriptions/` — the owner submits, AI-assist disclosed;
coordinate with PR #2324/yiqiliu2), the full GPQA-198 @16K "go", any new machine/VM/TPU (never — only
same-region bucket + disk-attach to the 8 existing hosts), force-push, external comms.
