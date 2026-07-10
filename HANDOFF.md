# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-10 — **THE DCP BUG IS DEAD** (root cause: block-table granularity double-×dcp, fixed
`1f700c507` — the multi-week "packed-write metal bug" NEVER EXISTED) + **first 128K retrievals ever**
(smoke 3/3 + mechanism-depth smoke 4/4 at 130,420 prompt tokens, bf16+DCP=4) + **THE DENSE 128K GATE n=77
IS RUNNING** (run_id 124, **42/77 correct / 0 failures** at 05:53 UTC) + **64K dense 6/6 VERIFIED @dcp=1**
(run_id 100) + **fp8-KV kernel now COMPILES on v4** (pack_new_kv i8 fixes — but fp8 has NEVER generated a
validated token on metal) + **sparse-DCP Stages A+B+C CODE-COMPLETE, CPU-certified** (`6f8855c3f` tip) +
**upstream verdict: the DCP bug WAS upstream's, fixed by #3129 three hours before ours** (convergent).
**MTP FROZEN** by owner directive until the headline gates close.

**Honest claim line:** *first public DSA kernel on TPU — selected-set-exact on silicon, sparse passkey 100%
@8K/32K; dense retrieval now proven to 64K (6/6) and 128K-smoked 7/7; the ≥95%@128K DENSE gate is in flight
at n=77 (small-n is never a gate — n≥73 zero-failure required). The sparse-DCP distributed stack (Stages
A+B+C) is CPU-CERTIFIED ONLY — zero sparse-DCP tokens have run on metal.* **Exact next work: finish the
dense gate (watchdog armed), then `docs/11-pod-runbook.md` §8 — the sparse metal ladder.**

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `docs/11-pod-runbook.md`
(§8 = the next actions; §5d is history) → `docs/RESEARCH_LOG.md` (the dated narrative — the whole
fragmentation → fp8 → worker-staleness → granularity-root-cause saga is 2026-07-09 09:20 onward) →
`PLAN.md` → the design docs as needed: `docs/01` (DSA kernel), `docs/03` (throughput), `docs/05` (KV/DCP),
`docs/10` (observability), `docs/upstream/dcp-block-granularity-report.md` (the #3129 memo),
`docs/reviews/*`. DSV4 base: `~/moe-tpu/{CLAUDE,HANDOFF}.md`.

---

## State — what is DONE and hardware-validated

- **Stage-1 serving WORKS on the pod** (unchanged): 753B FP8 streamed GCS→HBM (runai + EP filter), FP8
  resident (23.06/30.75 GiB/chip), dense MLA via mla.v2, pure TP-32. **GSM8K n=32 @ max-new 2048 = 96.875%
  (31/32)** — a SMOKE (n=32), NOT a scale claim (needs n≥200). The fatal OOB core-halt class is CLOSED
  (fork `63427f86`/`02e44b36`, 8h52m zero-interrupt validation). GPQA-198 dense was truncation-dominated
  (completed 50/58 = 86.2%); the 16K rerun stays queued behind the gates. Provenance: `bench/results.db`.
- **Stage-2 DSA kernels SILICON-VALIDATED** (single v4 chip, GATE 2a/2b): selected-set-EXACT vs the
  HF-math oracle, sparse-MLA fp32 9.5e-7 / bf16 1.95e-3. **Sparse passkey 100% every (length,depth) cell
  @8K & 32K** (72/72; 16× sparsification at 32K); D0 sparse==dense in the 753B engine.
- **64K dense passkey VERIFIED 6/6 = 100% @ dcp=1/bf16** (run_id 100, avg_prompt_tok 65,211; pool 140
  blocks / chunk 256 / gmu 0.90 + anti-frag flags; RESEARCH_LOG 07-09 12:25). First verified retrieval
  above 32K. The ceiling analysis behind it (07-09 09:20→13:15): the 128K@dcp=1 wall is HBM
  **fragmentation**, not capacity — the attention compile-scratch scales as **0.625 MiB × num_gpu_blocks**
  (128K ⇒ ≥258 blocks ⇒ ~161M) while the largest-contiguous-free region is a **run-to-run lottery
  (~114–157M)**. Best anti-frag lever: `LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5
  --xla_tpu_rwb_fusion=false"` (contiguous 118→148M; disabling the scheduler makes it WORSE). Reliable
  dcp=1/bf16 ceiling ≈ 88–90K; **128K@dcp=1 is CLOSED to tuning** — it needs cache-size reduction.
  Both bf16 copy-elimination attempts (`GLM_MLA_ALIAS_KV` L1 out-sharding pin `a248bd6b0` + L2 donation
  `4eb10e6ad`) are **honest nulls** — CPU-proven architecturally inert; kept as gated no-ops.
- **fp8-KV kernel COMPILES on v4** (07-09 16:15): the pack_new_kv i8 `select_n arith.cmpi`
  (`8a76ae5e3` branchless `_dtype_safe_select`) + i8 `arith.muli` (`0f15e3ad0` int32 blend mask) + the
  Q/output-cast fp8 convert (`65a6bb147` `DISABLE_MLA_Q_ACTIVATION_QUANTIZATION` keeps Q/out bf16 + fixes
  a latent q_scale over-scale) are ALL cleared — fp8-KV is no longer Mosaic-blocked. BUT: 128K@fp8/dcp=1
  is genuinely HBM-bound (args 28.90G + program 2.15G incl. 2.05G overlays + reserved 1.25G ≈ 32.3G >
  30.75; the overlays are the unrolled 78-layer machine code — no flag shrinks them), realistic fp8/dcp=1
  ceiling ~80–96K. **⚠ fp8-KV has NEVER produced a validated on-metal token** (audit MAJOR) — a cheap fp8
  needle @dcp=1 MUST pass before any coupled fp8+DCP claim. fp8 is now the ALTERNATIVE 128K config, not
  the required one (see next).
- **✅ DCP ROOT CAUSE FOUND AND FIXED — block-granularity double-multiplication (`1f700c507`; 07-09
  21:30→22:32).** `get_kv_cache_spec` pre-multiplied block_size ×dcp (the #2398 "TODO(xiang) hack") AND
  vLLM's engine multiplied ×dcp AGAIN → the engine allocated one block id per 512×dcp² tokens while the
  TPU stack consumed the table at 512×dcp tokens/entry → every table entry ≥1 dereferenced an
  unallocated/stale page. Explains EVERY symptom (dcp-only, >1024-token threshold at dcp=2,
  pageloop-inert, NO_DONATE-inert, CPU-invisible). Five independent confirmations incl. a live prediction
  (900-token 1-page needle PASSED before the fix landed). **Post-fix on metal: 900tok 1-page ✓, 4K
  single-chunk ✓ (was pred=None), 16K multi-chunk ✓ (was pred=None)** — runs 119–121. The DCP
  read/combine/scatter machinery was CORRECT all along; the docs/11 §5d scatter-impl hunt is OBSOLETE
  (banner added). `GLM_DCP_SCATTER_IMPL=pageloop` (`f2effec06`+`d59626964`) is kept — CPU-bit-identical,
  the default for the sparse indexer-key scatter — but it was never the fix.
  **Consequence: bf16 + DCP=4 fits 128K WITHOUT fp8** (22.76 weights + 2.05 overlays + 1.25 reserved +
  12.6/4 ≈ 3.15 KV ≈ 29.2 < 30.75) — the dense gate is decoupled from the unvalidated fp8 write path.
- **Upstream verdict (`docs/upstream/dcp-block-granularity-report.md` — DO NOT FILE the corruption
  report):** the same bug WAS live upstream, shipped broken in **v0.20.0–v0.24.0**, and was fixed by
  **PR #3129** (weiyu0824, merged 2026-07-09 12:04 PT — **3 hours before our fix**; same two-sided
  contract). Treat `1f700c507` as convergent-with-#3129; adopt upstream's structure on the next sync.
  Still upstreamable from us: the engine-vs-TPU **geometry assert** (would have caught 5 releases), the
  2 residual granularity holdouts (routed-experts slot reconstruction, KV-connector insert/extract), and
  the big one — **upstream dcp>1 is NOT real context parallelism** (it folds into head-TP; kernels see
  the full cache): our owner-scatter + LSE-merge CP attention is fork-only, a genuinely novel feature PR.
- **128K SMOKED 7/7 (first 128K retrievals ever):** smoke 3/3 (d=.25/.5/.75, run 122) + smoke-2 4/4 at the
  MECHANISM depths 0.0/0.05/0.95/1.0 (run 123), all at 130,420 prompt tokens, bf16+DCP=4. Recorded as
  SMOKE (n=7, Wilson LB ~44% at n=3) — GO for the gate, not the gate.
- **Observability toolkit REPAIRED (`6f45e0944` + `165c76462`; from the 25-agent obs audit: 2 BLOCKER +
  6 MAJOR in the tools themselves):** dcp_cache_diff now REFUSES verdicts on partial dumps (a missing
  shard could silently flip DIFFER→MATCH) + fails loud on short block tables (the silent break that hid
  the granularity bug); dumps carry a `__dtype` tag (bf16/fp8 dumps used to crash every reader);
  `dcp_dump_check.py` catches "dump requested, nothing written" (the driver-only-env incident); armed
  guards RAISE instead of fail-open; `GLM_DUMP_STEP_HLO` (`3b2963082`) says loudly when the step fn
  compiled inline and was NOT dumped; **`code_fingerprint.py` logs per-worker git-hash at init and
  `GLM_EXPECT_CODE_HASH=<hash>` makes a mismatched worker RAISE** (the stale-worker night, below);
  `assert_engine_tpu_block_granularity()` fails at init on the exact contract class just root-caused.
  74 CPU tests. The 29-agent code audit of 8802ebab7..HEAD found no BLOCKER/MAJOR in the shipped code.
- **CRITICAL PROCESS LESSON — the stale-worker night (07-09 15:20):** the 8 hosts have PER-HOST local
  `~/tpu-inference` checkouts; workers 1–7 silently ran `8802ebab` all night while worker 0 had the
  fixes, AND bare `git push -q` was not updating origin. **Standing rule: before ANY pod test of a fork
  change — `git push origin glm-5.2-v4-next`, then `TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash
  ~/glm-tpu/scripts/sync_workers.sh`, and confirm all 8 hosts print the SAME hash** (belt-and-braces:
  bake `GLM_EXPECT_CODE_HASH` into EXTRA_ENVS). Every worker0-only "inert" result from that night was
  re-validated after sync.
- **Sparse-DCP distributed stack (GLM_DSA_DCP) CODE-COMPLETE — Stages A+B+C, CPU-CERTIFIED ONLY (no
  metal execution yet):** the 128K SPARSE gate is NEW DISTRIBUTED CODE, not "32K with a bigger number"
  (DCP shards the context → local top-2048 ≠ global top-2048). Design (owner's, 07-10 00:05): per-shard
  score on the local stripe → local top-min(k, S_local) → all-gather candidates (64 KiB/token @dcp=4) →
  position-sort → global top_k — **elementwise-exact vs single-chip incl. tie order, gather-order-
  invariant by construction** → owned-subset attention → `_dcp_lse_combine` (dense machinery reused).
  - **Stage A** `ed2a021be` — distributed top-k primitives, sparse-decode `emit_lse`, local striped
    gather; 132/132 CPU tests, byte-identity proven. (Design prototype: 52/52 incl. mutation audit.)
  - **Stage B** `4f390a61e` (+ reviewer test extension `e7c239c91`) — serving-path distributed sparse
    decode (shard_map write/select/attend + LSE combine); 14/14 new e2e; **adversarial review: SAFE-FOR-
    METAL-LADDER** (cross-checkout gate-off jaxpr byte-identity; new selection battery all
    elementwise-exact; refusal semantics un-bypassable — NaN poison survives swallowed callbacks).
  - **Stage C** `6f8855c3f` (tip) — distributed masked-prefill flash scan with LSE combine (the last
    code bridge to the sparse ladder); merged suite 24/24 in one process; prefill e2e at non-page-aligned
    chunks; gate-off jaxpr SHA == HEAD. **Stage-C adversarial review in flight** — it is the hard gate
    before the metal ladder.
- **MTP: FROZEN** (owner directive, 07-09 23:10) — code-complete, pod-validation pending; no new surface
  until the dense+sparse 128K gates close.
- **Benchmark harness** (unchanged + hardened): `glm_longctx.py` passkey ladder, `GLM_DCP`/`GLM_SPEC_K`/
  `GLM_KV_CACHE_DTYPE` knobs, `warn_worker_only_envs()`; **drivers launch via `setsid nohup … </dev/null`**
  (a tool-shell teardown SIGTERM'd two engine-built runs mid-serve); **never grep bare "PASS"** — it
  matches `hlo_passes.cc` log lines (a false 64K "pass" was caught this way; read the needle lines).

## RUNNING right now

- **THE DENSE 128K GATE — run_id 124** (`~/glm-run/GATE128K_n77.log`): 128K passkey, **n=77 = 7 depths
  {0.0,0.05,0.25,0.5,0.75,0.95,1.0} × 11 trials**, bf16 + GLM_DCP=4, granularity fix + Stage-A code
  (`ed2a021be`) on all 8 workers. Config: pool 68 blocks (2048 tok/id at dcp=4), chunk 2048, gmu 0.90,
  max-len 131,840, TPU_MIN_TOKEN_BUCKET=32, anti-frag LIBTPU flags baked. Launched 2026-07-10 00:09 UTC,
  ~19 h ETA (≈8 min/needle). **42/42 correct, zero failures as of 05:53 UTC** — depths 0.00/0.05/0.25
  closed 11/11 (the mechanism cells perfect), depth 0.50 at t=8. **77/77 ⇒ Wilson LB 95.3% ⇒ the
  ≥95%@128K dense slot fills with a defensible n; ANY failure ⇒ n must grow (~200 to survive one miss).**
  A watchdog (5-min poll) fires on the first `correct=False`, driver death, or DRIVER_EXIT.

## What is NEXT (in order)

1. **Close the dense gate** — let run 124 finish; record in results.db + RESEARCH_LOG; backup bundle.
2. **The sparse metal ladder — `docs/11-pod-runbook.md` §8** (sync workers to `6f8855c3f` + Stage-C
   review verdict first): emit_lse unit @dcp=1 → sparse decode @dcp=2 ≤2048-tok vs dcp=1 → chunked
   prefill @dcp=2 4–6K + step-HLO honesty → 32K selected-set dump dcp=2 vs dcp=1 (Gate-2b on metal) →
   32K sparse passkey dcp=2/4 → 64K → **the 128K SPARSE gate at n≥73**.
3. **256K throughput A/B** (dsa-sparse vs dense; `bench/dsa_throughput.py`, dcp≥4, fp8-KV after its
   dcp=1 needle validates) — the second empty headline gate.
4. **Benchmarks at scale**: GSM8K n≥200 (retire the n=32 smoke), GPQA-198 rerun @ `--max-new 16384`
   (owner-gated "go"), AIME-2026 card n=30.
5. **MTP M2** — stays FROZEN until 1–3 close.

## Fork branch map (`~/tpu-inference`, worktrees at `~/tpu-inference-<tag>`)

| branch | what | on origin? |
|---|---|---|
| `glm-5.2-v4` (`02e44b36`) | pod mainline — Stage-1 + observability + the OOB fix | yes |
| **`glm-5.2-v4-next`** (**`6f8855c3f`**) | **integrated staging + what the pod RUNS NOW** — everything above: Stage-2 kernels, DCP + **the granularity fix `1f700c507`**, fp8-KV v4 fixes, obs repairs, **sparse-DCP Stages A+B+C**, mtp-g4, det; all gated off by default | yes — **workers pull from origin: push THEN sync_workers, verify 8× same hash** |
| `pr-g1..g6` | upstream PR series slices (owner submits); G6 = the DSA kernels | on fork |
| older feature branches | merged into `-next`; `dcppersist2`'s "shardlocal" was orphaned/never on workers | — |

Commit inventory since `8802ebab7` (all on `-next`): `a248bd6b0`/`4eb10e6ad` (ALIAS_KV no-ops, honest
nulls) → `3b2963082` (step-HLO dump) → `65a6bb147`/`8a76ae5e3`/`0f15e3ad0` (fp8-KV v4 fixes) →
`f2effec06`/`d59626964` (pageloop + its test) → **`1f700c507` (THE granularity fix)** →
`6f45e0944`/`165c76462` (obs repairs) → `ed2a021be` (Stage A) → `4f390a61e`/`e7c239c91` (Stage B + tests)
→ `6f8855c3f` (Stage C).

## Review-round ledger (adversarial review is part of the loop — every change-set gets one)

Rounds 1–9 + the DCP-saga reviews: see the table in git history (`docs/reviews/*`). New since 07-09:
| scope | outcome |
|---|---|
| GLM_MLA_ALIAS_KV L1/L2 | SAFE-TO-TEST, byte-identical off; efficacy caveat CONFIRMED on metal (inert) — kept as documented no-ops |
| fp8-KV Q-bf16 + pack_new_kv fixes | numeric-parity vs `pack_new_kv_reference` (bf16 AND fp8) + structural 0-f8-converts; v4 COMPILE confirmed on pod |
| granularity fix `1f700c507` | adversarial review + 5 independent confirmations incl. a live prediction; CPU sim corrupts at exactly L=1025 unpatched |
| code audit 8802ebab7..HEAD (29 agents) | no BLOCKER/MAJOR; 3 MINORs (L2-donation comment, fp8-never-ran-on-metal, pageloop cost) — all addressed/logged |
| obs audit (25 agents) | 2 BLOCKER + 6 MAJOR in the TOOLS — all repaired in `6f45e0944` |
| sparse-DCP Stage B | **SAFE-FOR-METAL-LADDER** (reviewer authored + contributed the mixed-batch/multi-chunk suite `e7c239c91`) |
| sparse-DCP Stage C | review IN FLIGHT — hard gate before the §8 ladder |

## Operating landmines (learned the hard way — expect these)

- **WORKER STALENESS (the #1 footgun):** 8 per-host checkouts + origin not updated by bare `git push -q`.
  Push explicitly, `sync_workers.sh`, verify 8× the same hash, pin with `GLM_EXPECT_CODE_HASH`. A
  worker0-only fix "tested" on the pod tests NOTHING.
- **`setsid nohup … </dev/null` every driver** — tool-shell teardown SIGTERMs the process group.
- **The grep-"PASS" trap:** `hlo_passes.cc` log lines match; assert on real needle/result lines only.
- **GLM serves via `VllmModelWrapper`, NOT `get_flax_model`** — verify the code path before trusting a fix.
- **`GLM_*` envs are trace-time + worker-side** → raylet-baked via `EXTRA_ENVS` AND exported on the
  driver. `GLM_DSA_MODE`/`GLM_DSA_DCP` shape the KV-cache spec — cross-host divergence = inconsistent
  cache topology. `warn_worker_only_envs()` + `dcp_dump_check.py` now catch the classic misses.
- **Small-n is never a gate**: ≥95% needs n≥73 zero-failure (Wilson). Mechanism depths 0.0/0.05 and
  0.95/1.0 are REQUIRED cells for any DSA claim (0.75 passes nearly by construction).
- **gmu is inert for fragmentation once `num_gpu_blocks_override` is set**; allocation ORDER governs.
  Contiguous-free is a lottery — a pass that needs fragmentation luck is NOT a pass (3/3 rule).
- **Relaunch after any pod crash** (leaked EngineCore / ~1800 s PG timeout); fetch flight files FIRST.
  TPU access serialized to the main session; agents set `JAX_PLATFORMS=cpu` and work in WORKTREES.
- **Durable backups**: `scripts/backup_bundle.sh` → `gs://driftbench-dsv4-uc/backups/glm-tpu/<ts>/`
  (us-central2 ONLY — an agent's EU-bucket violation was caught + purged on 07-10).

## Target (unchanged)

`zai-org/GLM-5.2-FP8` (753B/40B, 78L, MLA + DSA index_topk=2048/32 heads/IndexShare freq 4, MTP layer 78,
1M ctx) on the 32-chip v4 pod `db-v4-64-od` ONLY. Weights at `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/`.
Stage-2 gate: passkey ≥95% to ≥128K (dense IN FLIGHT n=77; sparse next at n≥73) + bounded divergence.
Stage-3: MTP acceptance ~5 (FROZEN).

## The exact next task

The dense gate (run 124) is baking with a watchdog — do not touch the pod until it triggers. Then, in
order: (1) close/record the gate; (2) **§8 sparse ladder** (`docs/11-pod-runbook.md`) — Stage-C review
verdict → sync workers to `6f8855c3f` → emit_lse unit → decode parity → prefill+HLO honesty →
selected-set dump → 32K/64K passkey → 128K sparse gate n≥73; (3) 256K throughput A/B; (4) GSM8K n≥200 +
GPQA@16K. Each step: env block, expected outcome, abort criteria, 3/3 rule for gates. After each
milestone: RESEARCH_LOG + HANDOFF + commit/push + backup bundle.

## Owner-gated (draft, don't do)

Upstream PR submission (incl. the #3129-adjacent geometry-assert/holdout/CP-feature series — the owner
submits, AI-assist disclosed), the GPQA-198 @16K "go", any new machine/VM/TPU (never — only same-region
bucket + disk-attach to the 8 existing hosts), force-push, external comms.
