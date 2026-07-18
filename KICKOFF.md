# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves
correctly, (2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates
(passkey ≥95%@128K n≥73; throughput ≥256K). Self-correct; when unsure pick + log.

## STATE (root-cause hunt; RESEARCH_LOG 07-17 08:20 → 07-18 06:35)
- Stage 1 serves; kernels silicon-validated; ✅ DENSE 128K GATE CLOSED 77/77; 12.6× campaign; safety
  commit + ops kit landed; MTP FROZEN. Fork tip **c68794241**, synced 8×.
- **THE ENGINE LOTTERY = SILENT WEIGHT-LOAD CORRUPTION**, ~56-60% of engine inits, per-host,
  per-launch, contiguous-granule, NaN/Inf-or-FINITE garbage. Detection landed:
  GLM_PWAL_NAN_CHECK (indexer params, RAISES) + **GLM_LOAD_NAN_CHECK** (full-weight on-device scan
  at load_model tail, reject byte-dumps). ⚠ NaN-refusal is necessary NOT
  sufficient — finite garbage is invisible (the p5 miss).
- **ROOT-CAUSE CANDIDATE (fix in flight): the t2j ALIAS RACE** — utils.py t2j bitcast branch makes a
  ZERO-COPY numpy view of the torch storage; JAX H2D staging is immutable-until-transfer-completes;
  `_free_cpu_storage`/cleanup_sharding `resize_(0)` frees it mid-flight ⇒ freed/reused heap lands on
  device. UNIFIES DSV4's never-root-caused "flaky dequant
  crash" (~60%/build, retry-mitigated, unverified): unmapped ⇒ SIGSEGV; reused ⇒ silent garbage.
- REFUTED: per-host binaries (fingerprints identical), PWAL copies (code map), runtime clobber
  (timeline: born-NaN at write; "page-0" was the null block), scale-as-primary ([128,128]
  arithmetic), streamer concurrency (A/B: 3/8 vs 6/8 — lower WORSE), F8_E8M0 (GLM ckpt has none).
- **OWNER RULE: on any domain shift RE-READ CLAUDE.md/HANDOFF/RESEARCH_LOG/suggestions.md under the
  new lens BEFORE building instruments** — the corpus held this root cause while six instruments
  circled it. And dump1090: dissect ONE specimen vs reference before rate experiments.
- DOC DEBT: HANDOFF body + CLAUDE.md Progress tail stale — pay next docs pass.

## FRONTIER (in order)
1. **Land the t2j fix** (worktree glm-t2j-fix; its test must FAIL on pristine then PASS — the
   proof) → adversarial review → sync 8×.
2. **VALIDATION: ~10 init-only draws, both checks armed — corruption must collapse 56%→0.** Any residual ⇒
   dissection specimen decides (PWAL unarmed so LOAD dumps fire; 3-way diff corrupt/clean/GCS).
3. **RE-GATE: gate_sparse128k.sh** — update PIN to the fix tip + add GLM_LOAD_NAN_CHECK=1 to its
   RAYLET_ENVS (health probe + refusing loads + fixed loader = triple protection). n=77, mechanism
   depths first, miss-abort at 2, per-depth ckpts, ONE miss ⇒ extend n≈130.
4. 256K A/B at IDENTICAL dcp; fp8-KV after its dcp=1 needle. 5. GSM8K n≥200; GPQA@16K (owner-gated); MTP
   unfreezes; PR re-cut (+ upstream the t2j fix — bites every tpu-inference user).

## HARD RULES
COST: gs://driftbench-dsv4-uc only; NEVER create machines/TPUs; disk-attach pre-authorized (§COST)
if local weights become the fix. METHOD: observability-first; fix root cause; gated + CPU test +
adversarial review + pod validation; honest nulls. SMALL-n never a gate. COMMIT+PUSH every
milestone. Agents: worktrees, JAX_PLATFORMS=cpu, never git on worker checkouts. Owner submits PRs.

## READ FIRST
HANDOFF.md → RESEARCH_LOG 07-17 08:20 on → docs/11 §8. Verify 8× fork tip on all hosts.

## LANDMINES
Sync verify 8× ALWAYS (partial syncs ×2 — clear stale index.lock + re-pull). moe-tpu bucket mirror is
DEAD-STALE — corpus searches use the GitHub origin. RAY_DEDUP_LOGS=0 on forensic runs (dedup destroys
digit-bearing evidence). Armed PWAL raise PREEMPTS the LOAD census+dumps. Poisoned dump tars compress
~150:1 (instant triage). Firewall tag orphans on pod recreation. Disk: w-0 baseline incl. scratch.
Scatter bakes per-path (dense pageloop STAYS; DSA flat unset). GLM_* raylet-baked AND driver-exported.
setsid --wait. "PASS" greps match hlo_passes.cc. Init geometry must match the warm XLA cache.
