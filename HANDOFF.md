# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-18 19:45 UTC — **THE SPARSE 128K GATE IS RUNNING AND WINNING** (run dir
`~/glm-run/gate128k_20260718T110127Z`, n=77, PIN **a225d16b4**, **33/33 through d=0.0/0.05/0.95** — the
gate2 killer cell CLEARED 11/11; verdict ETA ~05-07 UTC 07-19). **CHECK ITS STATE FIRST (its
orchestrator.log + results.db); NEVER launch pod work while it runs.** The full 07-18 arc
(RESEARCH_LOG 09:05→19:15): t2j alias fix adversarially reviewed 3/3 (SAFE on the GLM path; scope
claim refuted — sibling sites parked on `~/wt-sibling-alias`, 4 commits, 38 tests green, land
post-gate); the stats review REFUTED the "~10% residual, gate open" claim (pooled-detector confound;
all protections were NaN-only) ⇒ built **GLM_LOAD_CHECKSUM** (categorical cpu-vs-device byte-verify of
every t2j-staged tensor at load tail, raises on mismatch — catches FINITE corruption; validation 4/4
clean, live on 8 hosts, verified=1882/host, 312 benign 0-d skips); xprof 128K needle: NO ≥40% dominator
(S2 pallas scorer NOT justified; chunk ≈9.35s at tiny kv ⇒ per-chunk cost dominates prefill; post-gate
efficiency targets: top_k 21.8% / gathers 16% / collectives ~24%). **RESIDUAL SPECIMEN (hypothesis
revised):** the one sick gate draw (d=0.05 try-1) was byte-verified CLEAN on every surface yet
fluent-filler-missed a 5K needle ⇒ the ~14%/draw residual is NOT H2D weight corruption — candidates:
CPU-side finite corruption pre-t2j, or engine-instance STATE (XLA program draw / device order /
KV-selection state). Specimen: db run 193 + specimen_d005_try1/. MTP stays FROZEN.

**Honest claim line:** *first public DSA kernel on TPU — selected-set-exact on silicon; dense 128K gate
CLOSED 77/77 (run 124, Wilson LB 95.3%); 12.6× efficiency campaign; the sparse ≥95%@128K gate is IN
FLIGHT at 33/33 with zero misses — d=0.95 (gate2's 0/11 death cell) cleared 11/11 on a byte-verified
engine, retroactively attributing gate2's death to the engine lottery, not the kernel. Not yet closed
until 77/77 banks.*

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `docs/RESEARCH_LOG.md`
**2026-07-17 08:20 onward** (lottery→root-cause→checksum→gate arc; efficiency campaign is 07-12 20:30 on) (the efficiency campaign → gate2 death → lottery → confound → recovery arc;
the sparse-ladder/stripe-forensics arc is 07-10 16:10 onward) → `docs/11-pod-runbook.md` §8 →
`PLAN.md` → design docs as needed (`docs/01` DSA kernel, `docs/03` throughput, `docs/05` KV/DCP,
`docs/10` observability, `docs/upstream/*` the two silicon-bug reports). DSV4 base:
`~/moe-tpu/{CLAUDE,HANDOFF}.md`.

---

## State — what is DONE and hardware-validated

- **Stage-1 serving WORKS** (unchanged): 753B FP8 streamed GCS→HBM, FP8-resident + per-tile dequant,
  dense MLA via mla.v2, pure TP-32. GSM8K n=32 smoke 96.9%. OOB core-halt class CLOSED.
- **✅ DENSE 128K GATE CLOSED: 77/77** (run 124, bf16+GLM_DCP=4, 7 depths × 11 incl. mechanism cells,
  Wilson LB 95.3%). Fully present in the restored results.db.
- **Sparse ladder rungs 1–6 CLOSED** (07-10/11): emit_lse metal unit; distributed sparse decode (after
  the pageloop-v4 sublane-drop root cause → `GLM_DSA_DCP_SCATTER_IMPL` default **flat**, 4f7d9a001);
  5K chunked prefill + HLO hard gate (zero whole-cache collectives); 32K selected-set dcp=2 vs dcp=1
  scores-armed (tripwire silent, kth_band criterion); 64K sparse 3/3 — first sparse retrieval above 32K
  on any hardware.
- **Sparse CORRECT at 128K gate geometry, twice:** mechanism smoke 4/4 EXACT (run 153, masked+headsplit)
  and cycle-B 4/4 EXACT (run 162, segment+owned+all-v2+chunk-2048). Same passkeys across five configs.
- **✅ EFFICIENCY CAMPAIGN (owner pivot 07-12 20:30) — 12.6× end-to-end at 128K** (needle 7513s → 595s;
  prefill 17 → ~220 tok/s; **sparse now beats dense prefill**). The four transforms, all gated,
  bitwise-exact, landed on `-next`, synced 8×:
  - **S1 segment prefill** `GLM_DSA_DCP_PREFILL_ATTN=segment` (29305e185) — kills the masked
    whole-stripe walk (the cost-model dominator); 6.1× alone at 32K.
  - **W2.1 owned block-table width** `GLM_DSA_BT_WIDTH=owned` (5c6e1f0c8) — kills the 258-wide padded
    table walk (the serialized page-loop).
  - **Gather dominator** `GLM_DSA_MERGE_IMPL=v2 / GLM_DSA_OWNED_SEG_IMPL=v2 / GLM_DSA_SEG_GATHER_IMPL=v2`
    (a98c77c9) — kills the slice_sizes=(1,1) scalar-gather pathology (61.9% of the step at 2 GB/s).
  - **Chunk 2048** (ratified) — compiles at the winning config (segment removed the OOM buffer).
  - Cycle A: A1-vs-A0 2.7× with selections bitwise-clean per the band criterion (runs 160/161).
- **❌ GATE2 (sparse 128K, n=77, chunk 2048, pin a98c77c9): DEAD.** d=0.0 11/11 (run 163), d=0.05 11/11
  (run 164), then **d=0.95 = 0/11 (run 165) with a NEW signature** — fluent-filler outputs (needle not
  retrieved), NOT the D2-era decode-death pred=None. Wilson-unrecoverable; killed honestly.
- **The lottery hypothesis and its confound:** d=1.0 went 2/2 EXACT on the very next engine ⇒
  per-ENGINE-INSTANCE expression (~1/7 draws) — BUT the postmortem found **w-4's disk was at 0 through
  gate2's d=0.95 window** (55G of parked dump archives; raylet died on every join, 7/8 nodes, 14/14
  probe INFRA-FAILs) ⇒ **the lottery hypothesis is CONFOUNDED by disk-pressure-degraded engines.**
  Neither confirmed nor refuted. The deciding experiment (below) was starting when the VM died.
- **Known-broken (refusals still NOT in code — part of the lost safety commit):** headsplit×segment
  metal composition (D2(ii)); headsplit at dcp=2/H_local=4. `GLM_DSA_DCP_HEADSPLIT` stays default-OFF
  and adds nothing under segment (3228s vs 3225s) — the gate config runs WITHOUT it.
- **07-13 holistic audit (20 agents, 44→12 confirmed):** perf transforms REAL and CPU-exact but **NOT
  PR-ready**; F1 gate2 dead; F2 recurrence unattributed (standing suspect: donated-cache payload gather
  in the attend lax.map, sparse_mla_kernel.py:610); F4 orchestrator RELAUNCH env-dropout (false
  provenance on retry); F6 headsplit×segment armable with no refusal; F7 masked backstop has zero metal
  tokens on the owned/v2 program; F8 armed bitwise coverage only at T=1024 (gate ran T=2048). Full
  report was scratchpad-only (LOST); summary in RESEARCH_LOG 07-13 09:00.

## THE RECOVERY (2026-07-16) — what a resumed session must know

- **Restored and verified:** glm-tpu @ fdb6e7f + this recovery work; fork @ a98c77c9 on
  `glm-5.2-v4-next` (audit: newest commit on all 39 origin refs — nothing committed was lost);
  vLLM@LKG a30addc7 rebuilt (`~/vllm-build`); venv from tarball; **workers 1–7 provisioned 7/7 OK @
  a98c77c9** via `scripts/provision_worker_glm.sh` (NEW — the fresh-pod recipe; same-region artifacts
  at `gs://driftbench-dsv4-uc/artifacts/`); `.env` HF token; sync cron re-armed (the old VM's cron
  NEVER ran — setup.sh bug, fixed in the bucket copy); firewall rule intact; weights intact at
  `gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/`.
- **results.db restored from the 08:31:37Z GCS copy — last row IS run 165's 0/11 aggregate** (the
  per-depth checkpoint discipline captured the gate death). Missing: the d=1.0 discriminator 2/2 rows
  (log-recorded only) and everything after.
- **Lost, must be rebuilt:** the safety/truth commit (F6 refusal, docstring fixes, combo-matrix tests,
  F4 RELAUNCH fix, miss-abort watchdog), the write-probe guard (spec: docs/upstream/
  pageloop-v4-sublane-drop-REPORT.md), disk-watchdog + quota'd dump archiver, ~/glm-run orchestrators,
  all XLA caches (first engines recompile ~45min+), the 14-probe results (never existed).
- All 8 disks now ~83G free — the clean-disk precondition of the discriminator experiment holds by
  construction. Keep it that way (dump quotas BEFORE anything gets armed).

## What is NEXT (in order)

1. ~~The safety/ops-debt commit~~ **DONE 07-17** (fork 845f4ffeb synced 8×; ops kit in scripts/).
2. ~~The discriminator~~ **DONE 07-17/18 — the lottery is SOLVED: silent per-host WEIGHT-LOAD
   corruption** (~56-60% of engine inits; RESEARCH_LOG 07-17 08:20 → 07-18 06:35 is the full
   six-instrument + corpus arc). Root-cause candidate with fix in flight: the **t2j alias race**
   (zero-copy numpy view of torch storage × resize_(0) eager free × async H2D staging) — unifies
   DSV4's "flaky dequant crash". Refuted en route: per-host binaries, PWAL copies, runtime clobber,
   scale-as-primary, streamer concurrency (A/B), F8_E8M0 gating.
3. ~~Land the t2j fix + validation~~ **DONE 07-18** (629c20e84, reviews 3/3; GLM_LOAD_CHECKSUM
   a225d16b4 landed after the stats review demanded a categorical finite-corruption instrument;
   validation 4/4 clean with the full stack armed).
4. **THE RE-GATE IS RUNNING** (gate128k_20260718T110127Z, 33/33 through the first three mechanism
   depths). On PASS 77/77 ⇒ bank + close; ONE miss ⇒ extend n≈130; 2-miss abort ⇒ forensics FROM THE
   SPECIMEN (armed instruments), never rate experiments.
5. **256K throughput A/B** (dsa-sparse vs dense at IDENTICAL dcp; `bench/dsa_throughput.py`); fp8-KV
   only after its own dcp=1 needle validates (fp8 has still NEVER produced a validated metal token).
6. **Benchmarks at scale:** GSM8K n≥200, GPQA-198 rerun @ 16K (owner-gated), AIME-2026 n=30.
7. **MTP M2** — stays FROZEN until 2–5 close. Then the PR-series re-cut (pr-g1..g6 predate the
   campaign; none of the 24+5 post-8802ebab commits are re-cut — future work, audit says NOT PR-ready).

## Fork branch map (`~/tpu-inference`)

| branch | what | on origin? |
|---|---|---|
| `glm-5.2-v4` (`02e44b36`) | pod mainline — Stage-1 + obs + OOB fix | yes |
| **`glm-5.2-v4-next`** (**`a225d16b4`**) | **integrated staging + what the pod RUNS** — the campaign stack (granularity fix, fp8-KV v4 fixes, obs, sparse-DCP A/B/C, scatter-flat, segment S1, W2.1, gather-dominator v2s) + the 07-17 safety commit (F6 refusals, GLM_WRITE_PROBE, combo suite, dense-impl refusal) + the 07-17/18 integrity layer (GLM_PWAL_NAN_CHECK raises; GLM_LOAD_NAN_CHECK full-weight init scan + reject dumps; 629c20e84 t2j alias fix; 04507ba1d CPU stage-splitter; a225d16b4 GLM_LOAD_CHECKSUM byte-verify), all gated off by default | yes — synced 8× dirty=0 |
| `pr-g1..g6` | upstream PR series (07-07/08 cuts — predate the campaign; owner submits) | yes |
| `dsv4-flash-v4` | the DSV4 base (LKG pin source) | yes |

Commit inventory 8802ebab7..a98c77c9 (24 commits): ALIAS_KV no-ops → step-HLO → fp8-KV v4 fixes →
pageloop → **1f700c507 granularity fix** → obs repairs → Stage A/B/C → topk-scores instrument +
sentinel tests → **4f7d9a001 scatter-flat** → 74d8c3225 → **edc7d726b headsplit** → **29305e185
segment S1** (= tag freeze-correct-128k-20260712) → **5c6e1f0c8 W2.1** → **a98c77c9 gather dominator**.

## Operating landmines (learned the hard way — expect these)

- **THIS VM IS POD WORKER-0** (`t1v-n-6c15e171-w-0`): a `--worker=all` git/ssh command mutates the
  LOCAL dev checkout too (a sync reset ran here mid-session; both branch pointers happened to match —
  check `git branch -vv` after any pod-wide git). And `ls -td ~/glm-run/gate128k_*` races the
  `gate128k_outer.log` FILE — always use the explicit run-dir name.
- **WORKER STALENESS (#1 footgun):** push origin explicitly → `sync_workers.sh` → verify 8× same hash →
  pin `GLM_EXPECT_CODE_HASH` (re-pin collides with a mid-flight engine — wait for it).
- **DISK PRESSURE is a first-class failure mode** (4 incidents; it killed gate-class runs and confounded
  the lottery): gate-class runs are UNARMED (a 128K armed gate writes ~230GB); dumps get quotas and GCS
  archival + purge; check `df` on ALL 8 hosts before any long run. The postmortem rule: a probe watcher
  must classify INFRA-FAIL vs verdict (grep patterns tested).
- **`setsid nohup … </dev/null` every driver** (foreground setsid forks+returns instantly — watch the
  real child pid). **Never grep bare "PASS"** (hlo_passes.cc). Armed topk dumps land on ONE host (w-2).
- **`GLM_*` envs are trace-time + worker-side** → raylet-baked via EXTRA_ENVS AND exported on the
  driver; `GLM_DCP_SCATTER_IMPL=pageloop` must STAY raylet-baked on every dense/GLM_MLA_DCP launch
  (dense's code-default plain scatter is metal-proven-BAD; the dense fallback fires INSIDE sparse
  serving at ctx≤topk). DSA owner-scatter default flat — leave `GLM_DSA_DCP_SCATTER_IMPL` unset.
- **Small-n is never a gate** (n≥73 zero-failure; mechanism depths 0.0/0.05/0.95/1.0 REQUIRED). One
  miss ⇒ extend to n≈130, never rerun-until-green. Trace-time env changes = a DIFFERENT program —
  never lead an attribution with armed probes (F3).
- **Relaunch after any pod crash** (leaked EngineCore / ~1800s PG timeout); fetch flight files FIRST.
  TPU serialized to the main session; agents JAX_PLATFORMS=cpu in worktrees.
- **After any setup.sh run, verify `crontab -l`** (the silent-abort bug that cost this VM its mirror is
  fixed, but the habit stays). Durable backups: `scripts/backup_bundle.sh` →
  `gs://driftbench-dsv4-uc/backups/glm-tpu/<ts>/` (us-central2 ONLY). **Never-committed work dies with
  the VM — commit+push at every milestone; the bucket mirror is not a backup** (it lagged to nonexistence
  once already).

## Target (unchanged)

`zai-org/GLM-5.2-FP8` (753B/40B, 78L, MLA + DSA index_topk=2048/32 heads/IndexShare freq 4, MTP layer
78, 1M ctx) on the 32-chip v4 pod `db-v4-64-od` ONLY. Weights at
`gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/`. Stage-2 sparse gate: passkey ≥95% @128K, n≥73
zero-failure + bounded divergence. Stage-3: MTP acceptance ~5 (FROZEN).

## The exact next task

**Watch the running gate to verdict** (persistent monitor on `~/glm-run/gate128k_outer.log`; per-depth
GCS checkpoints are automatic). On PASS: bank in RESEARCH_LOG + docs refresh + `backup_bundle.sh`, then
**256K** (fit/geometry probe — dcp=8 was deferred to this stage; novel geometry = ~40 min cold compile —
then needles, then the sparse-vs-dense throughput A/B at IDENTICAL dcp), then land `~/wt-sibling-alias`
+ draft the t2j upstream PR. The residual hunt starts FROM THE SPECIMEN (db run 193). The corpus-first
rule applies to every new domain.

## Owner-gated (draft, don't do)

Upstream PR submission (incl. the geometry-assert/holdouts/CP-feature series and the two silicon-bug
reports under `docs/upstream/` — the owner submits, AI-assist disclosed), the GPQA-198 @16K "go", any
new machine/VM/TPU (never — only same-region bucket + disk-attach to the 8 existing hosts), force-push,
external comms.
