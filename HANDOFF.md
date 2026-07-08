# HANDOFF — GLM-5.2 on TPU v4 (read this first, every new chat)

**Updated:** 2026-07-07 (evening) — **STAGE 1 IS DONE on the 32-chip pod** (GLM-5.2-FP8 serves,
GSM8K banked, the fatal core-halt class root-caused and fixed on hardware, 32.9 tok/s aggregate);
**GPQA-Diamond n=198 is IN FLIGHT** (run 48, `~/glm-run/gpqa198.log`); **Stage-2 (DSA sparse
decode) + DCP + dense-MTP M1 are CODE-COMPLETE on staging branches, CPU-validated, adversarially
reviewed through round 6 (round 7 in flight), NOT yet run on TPU.**
**The exact next work is scripted: `docs/11-pod-runbook.md` — the ordered post-GPQA pod sequence
with copy-paste commands, expected outcomes and abort criteria. Execute it top to bottom.**

**Read, in full, before doing anything:** this file → `CLAUDE.md` (rules) → `docs/11-pod-runbook.md`
(the next actions) → `docs/RESEARCH_LOG.md` (latest entries) → `PLAN.md` → the design docs as needed:
`docs/01` (DSA kernel), `docs/03` (throughput), `docs/04` (host path), `docs/05` (KV/DCP), `docs/06`
(pass-boundary crash forensics), `docs/07` (card protocol), `docs/08` (MTP), `docs/10` (observability),
`docs/reviews/*` (the adversarial-review ledger). DSV4 base: `~/moe-tpu/{CLAUDE,HANDOFF}.md`.

---

## State — what is DONE and hardware-validated

- **Stage-1 serving WORKS on the pod** (details: RESEARCH_LOG 06:40 + 15:05 entries): 753B FP8
  streamed GCS→HBM in ~10 min (runai + sharding-derived EP filter, 32/256 experts/host), FP8 kept
  resident (23.06/30.75 GiB/chip), checkpoint-exact block scales, dense MLA via mla.v2 +
  cross-shard all-gather, pure TP-32 mesh, KV 128 blocks × 512 = 65,536 tokens.
- **GSM8K numbers (all in `bench/results.db`, full per-item provenance):**
  - run 26 "smoke16" n=4 @ fork `10efa393`: acc 75.0 (1 truncation miss at cap 512) — the
    byte-identity reference for the staging switch (runbook §2).
  - run 43 "waveA2" n=16: **acc 93.75** (2 truncation misses at cap 1024).
  - run 47 n=32 @ `02e44b36`, bucket 32, max_seqs 16: **acc 87.5** (6/32 = truncation misses at
    cap 1024), 20,280 gen tokens in 616.7 s = **32.9 tok/s aggregate** (~35× the 0.92 tok/s
    single-stream start). The cap, not the model, drives the misses → runbook §1 reruns n=128 at
    `--max-new 2048`.
- **The fatal core-halt class is CLOSED — the OOB fix story:** 7 consecutive batched runs died
  with roaming single-host `Error Interrupt`/core-halts. First attributed to per-host compile
  skew (`JAX_SHARE_BINARY_BETWEEN_HOSTS=1` — CORRECTED in-log: Ray-dedup artifact + composition
  luck; waveB/waveB2 crashed WITH sharedbin). Real root cause: **OOB VMEM read in mla.v2
  `pack_new_kv` at bkv-block-boundary decode (kv_len % 512 == 0, odd-slot alignment)** — fork fix
  `63427f86`, merged `02e44b36`, **validated on hardware** by run 47 (the exact config that
  previously died 100% at the first 512-page crossing ran clean). Forensics: `docs/06`;
  round-7 review of the fix pending (`docs/reviews/round7-pg2-oob.md`).
- **GPQA-Diamond n=198 IN FLIGHT** (run 48: bucket 32, max_seqs 8, max-len 8192, max-new 4096,
  greedy, launched 19:40 UTC, ~4–5 h). `--batch-size 0` ⇒ items land in the DB only at the end.
  Runbook §0 = outcome triage (completed → log Δ vs card 91.2; crashed → `triage_crash.sh` +
  retry recipe).
- **Stage-2 + DCP + MTP are CODE-COMPLETE (CPU-only so far; TPU untouched by them):**
  - `GLM_DSA_MODE=off|xla_ref|pallas_decode` — 2a XLA-reference indexer + oracle (RoPE verdict:
    **interleaved**, E1/E2 on real weights), 2b Pallas lightning-indexer + exact hierarchical
    top-k, 2c gathered-segment sparse-MLA decode kernel, 2a.2 paged indexer k-cache (KVCacheSpec),
    integration `pallas_decode` (decode sparse, prefill/mixed falls back dense in-graph).
    118 tests on the integration branch; gate-off byte-identity hash-tested.
  - DCP (`GLM_MLA_DCP=1` + `decode_context_parallel_size`): per-shard kernel + LSE combine +
    strided positions — the 128K-passkey capacity unlock (docs/05; KV table in runbook §5).
  - Dense-MTP M1 (G1–G3): draft parity vs composed-HF reference **max rel Δ 3.1e-6, top-1 100%**;
    M2 (pod greedy-equivalence) is runbook §7 — **prepped zero-turnaround 2026-07-08**: bench
    `GLM_SPEC_K=k` engine knob (unset = byte-identical, unit-tested) + `bench/mtp_m2_check.py`
    (per-item identity gate + acceptance-stat scrape, stub-tested; exit 0/1/2), §7 carries the
    exact n=8-sequential command pair (mtp-g4 + OOB fix verified coexisting on `-next`).
- **Benchmark harness**: batched generation (`--batch-size`), `--protocol card|greedy` (card =
  temp 1.0/top_p 0.95/163,840 cap/byte-pinned Exact-Answer system prompt; per-request seeds
  OMITTED on this backend — the round-6 F1 fix; the loud banner at build is the fix working),
  pinned dataset revisions, GPQA content-hash shuffle, `glm_longctx.py` passkey ladder to 1M.
- **Observability toolkit** (docs/10): per-step **flight recorder** (`GLM_FLIGHT_RECORDER=1`,
  JSON lines to `/tmp/glm_flight_*.jsonl`, 15 µs/step, SIGKILL-durable), **`triage_crash.sh`**
  (log parse + 8-host flight fetch + step alignment; sync mode ⇒ halted step = last line),
  `GLM_LOG_STATS=1` (10 s tok/s), `DSV4_OBSERVE_COMPILES`, launcher flight-file pruning.
  Read docs/10's field notes before interpreting any crash.

## Fork branch map (`~/tpu-inference`, worktrees at `~/tpu-inference-<tag>`)

| branch | tip | what | on origin? |
|---|---|---|---|
| `glm-5.2-v4` | `02e44b36` | **mainline** — Stage-1 + obs + OOB fix; the pod runs this NOW | yes |
| `glm-5.2-v4-next` | `cda8a707` | **staging** — mainline (incl. OOB fix) + r5fix + 2a2 + 2int + dcp + r6fix + sparse-prefill + **mtp-g4 (merge `534cd74d7`)** + det (all gated off) | **origin @ `886eaceb4` already has mtp-g4 + OOB (enough for M2); push before use for det** |
| `glm-5.2-v4-2int` / `-2a2` / `-r5fix` / `-r6fix` / `-obs` / `-pg2` / `-dcp` | — | feature branches, all merged into `-next` | no |
| `glm-5.2-v4-mtp` | `6beacb5d` | dense-MTP M1 — frozen; **superseded for M2 by the mtp-g4 merge into `-next`** (never run M2 from here: lacks the OOB fix) | no |
| `glm-5.2-v4-sparse-prefill` | `53c5e5ee` | Stage-2 sparse-prefill pointer (work in flight, another session) | no |
| `pr-g1..5-*` | — | upstream PR series slices (`docs/02` + `docs/pr-descriptions/`; owner submits; G5 = `pr-g5-mla-pure-tp` @ `132a11f9`, cut 2026-07-08 after the pod validation, pushed) | no |

## Review-round ledger (adversarial review is part of the loop — every change-set gets one)

| round | scope | reports | outcome |
|---|---|---|---|
| 1 | Stage-1 diff (4 lenses) | `docs/reviews/stage1-{a,b,correctness,numerics}.md` | 3 HIGHs fixed (engine-boot kv_cache_spec, default-on indexer, CPU test) |
| 2 | PR #2324 TP-topology port | (no standalone files; fixes = fork `a429be54`) | gather-before-TuningKey, page-512 v4 gate, TPU_MIN_TOKEN_BUCKET plumbing |
| 3 | parallel-agent deliverables (longctx/extract/datasets/docs) | `round3-*.md` | 2 HIGHs (BPE length −17.2%, `[gMASK]` claim), revision pinning, extractor fixes |
| 4 | pod bring-up series (EP filter, launcher, scales/logits) | `round4-*.md` | hardening `761ea755` |
| 5 | Stage-2 kernels (indexer, sparse-MLA, dsa-path+S1) | `round5-agent*.md` | fixes on `-r5fix` (`c8a51543`); **their on-TPU checklists = runbook §3's expected-failure list** |
| 6 | dcp, paged-indexer, observability, mtp-card, pr-branches | `round6-*.md` | fixes on `-r6fix`/`-dcp`/`-2a2` + bench F1 seed fix; honest-nulls CORRECTION of the sharedbin claim |
| 7 | pg2 OOB fix + 2int integration | `round7-2int.md` LANDED (correctness green, 108/108; F1–F3 = MAJOR perf/HBM/aliasing blockers for ENABLING `pallas_decode` — folded into runbook §3, incl. its ordered on-TPU gate list; F4: MTP spec-decode forces dense fallback); `round7-pg2-oob.md` pending | in flight |

## Operating landmines (learned the hard way — expect these)

- **Worktree policy: agents NEVER edit the main checkout** `~/tpu-inference` — the live pod
  imports it on every host. All feature work in worktrees (`git worktree add`) on feature
  branches, merged via `-next`. Same for review agents: read-only on the main checkout.
- **The pkill self-match landmine:** un-escaped `pkill -f` patterns match the pkill/ssh command
  line itself — a launcher stop-phase pattern once killed its own ssh carrier. Every pattern is
  bracket-escaped (`'VLLM::[E]ngineCore'`, `'[R]ayWorkerWrapper'`). Keep it that way; the stop
  phase also SIGKILLs the cohabiting ASPt stack (see the launcher's SHARED-POD POLICY — coordinate).
- **TPU access is serialized to the main session.** Helper agents run `JAX_PLATFORMS=cpu` set
  BEFORE python starts (one reviewer grabbed the TPU by setting it after import).
- **`GLM_*` envs are trace-time + worker-side** → raylet-baked via the launcher (`EXTRA_ENVS`)
  AND exported on the driver. Only `TPU_DISABLE_DSA_INDEXER`/`DISABLE_WEIGHT_REQUANTIZATION`/
  `TPU_MIN_TOKEN_BUCKET` force-propagate. `GLM_DSA_MODE` now shapes the **KV-cache spec** —
  cross-host divergence = inconsistent cache topology, not just a wrong mode.
- **Relaunch after any pod crash** (leaked EngineCore / ~1800 s PG timeout) — but fetch flight
  files FIRST (`triage_crash.sh`); the launcher prunes them and a pod restart wipes /tmp.
- Flaky FP8-dequant-during-load crash (~60% on DSV4, environmental) → retry; 2 consecutive =
  full relaunch. `OMP_NUM_THREADS=1` everywhere. No pipeline parallelism. The parity harness is
  single-chip-only BY DESIGN. Buckets: real constraint is 32-way divisibility — bucket 32 is
  legal and validated (run 47).
- **Durable backups**: `scripts/backup_bundle.sh` → `gs://driftbench-dsv4-uc/backups/glm-tpu/<ts>/`
  (verified bundles, ALL fork branches, results.db snapshot, logs, docs). Run it after every
  milestone; us-central2 ONLY (§COST in CLAUDE.md).

## Target (unchanged)

`zai-org/GLM-5.2-FP8` (753B/40B, 78L, MLA + DSA index_topk=2048/32 heads/IndexShare freq 4,
MTP layer 78, 1M ctx) on the 32-chip v4 pod `db-v4-64-od` ONLY. Weights staged at
`gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` (150/150 files, 755.7 GB, verified).
Stage-1 gate: card-family scores within ~1–2 pts at ≤8K ctx. Stage-2 gate: passkey ≥95% to
≥128K + bounded divergence. Stage-3: MTP acceptance ~5.

## The exact next task

**Execute `docs/11-pod-runbook.md` in order:** (0) GPQA triage → (1) GSM8K n=128 max-new 2048 →
(2) `-next` staging switch + byte-identity smoke vs run 26 → (3) DSA compile probe
(`pallas_decode`, expected-Mosaic-error list enumerated) → (4) Gate D0 sparse==dense →
(5) DCP bring-up + passkey ladder 8K/32K/128K → (6) AIME-2026 card n=30 → (7) MTP M2 k=1→5.
Each step: env block, expected outcome, abort criteria, 3/3 rule. After each milestone:
RESEARCH_LOG + HANDOFF + commit/push + backup bundle.

## Owner-gated (draft, don't do)

Upstream PR submission (`docs/02` + `docs/pr-descriptions/` — the owner submits, AI-assist
disclosed; coordinate with PR #2324/yiqiliu2), any new machine/VM/TPU (never — only same-region
bucket + disk-attach to the 8 existing hosts), force-push, external comms.
