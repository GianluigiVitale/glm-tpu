# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). **OWNER RULE: NO SHORTCUTS — root cause properly fixed + validated before any
re-gate; no workaround gating.** Self-correct; when unsure pick + log.

## STATE (2026-07-20 17:00 UTC; RESEARCH_LOG 07-19 21:10 → 07-20 14:10)
- Fork tip **a10d2a426** synced 8× (t2j fix, NaN/checksum integrity stack, read probes, xla-attend
  gate, audit fixes). Dense 128K gate CLOSED 77/77. GATE3 (sparse) died 33/35 honestly.
- **THE RESIDUAL = per-engine SELECTION DEGRADATION** (2 mangled-digit specimens: pred '7657' vs gold
  '797567', '665060' vs '648060' — needle positions mostly-but-not-fully selected). ELIMINATED, each
  instrument-proven: weights (byte-checksum), caches (bit forensics; Guard-2 trips = page-reuse FALSE
  POSITIVES), write path (barrier A/B), Mosaic decode attend (xla-bypass A/B, wiring CI-proven),
  proc/mesh order, v2 gather transforms as sole cause (v1 arm sick too).
- **RUNNING: the precamp arm** — FULL campaign revert (BT_WIDTH=full, PREFILL_ATTN=masked, mbt 1024,
  v1 selection, no Guard-2) ×5 draws, 32K geometry (~1-1.5h/draw). CLEAN ⇒ bisect {owned-width,
  segment-prefill, chunk-2048} one at a time. SICK ⇒ fault PREDATES the campaign ⇒ arm
  **GLM_DSA_DUMP_TOPK** (exists, zero code) on fixed-seed sick+healthy draws and diff selections.
- LOAD CLASS (separate): streamer delivers corrupt bytes CPU-side (CPU-scan-attributed); ~20% of
  draws auto-refused (costs a retry). Fix = gcsfuse/local-disk (§COST pre-authorized) — land RIGHT
  BEFORE gate4, never mid-hunt (one variable at a time).
- Amplified repro: `hunt_residual.sh` + ARM_ENVS/ARM_TAG/PIN/GEO_MAX_LEN/GEO_BLOCKS/GEO_MBT/
  LADDER_LENGTHS overrides (ARM_ENVS = raylet-tail last-wins; verify via worker /proc environ).
- 128K-shape armed variants (interpret/read-barrier/xla-attend) die at engine-init compile — probe at
  32K geometry; the 128K mitigation-compile issue is post-fix work.
- Ready: stage256k.sh (update PIN+envs at launch), ~/wt-sibling-alias (4 commits, land post-gate),
  PR audits banked (re-cut actions listed; DCO = owner-side), GCS backups current.

## FRONTIER (in order)
1. Precamp verdict → branch per above (bisect vs selection-dump dissection).
2. ROOT-CAUSE the guilty component (jaxpr/HLO comparison, pageloop-report methodology) → proper fix →
   CPU-bitwise proof → adversarial review → metal validation on the 30-90min repro cycle.
3. Streamer fix (gcsfuse Plan A, disk fallback) + validate loads clean.
4. **GATE4** n=77 @128K: fixed config, full integrity stack, 32K health needle (5K proven blind),
   Guard-1 on / Guard-2 OFF (false-positive class). Miss-abort 2; per-depth ckpts.
5. 256K stage (stage256k.sh). 6. GSM8K n≥200, GPQA@16K (owner-gated), MTP unfreeze, PR re-cut.

## HARD RULES
COST: gs://driftbench-dsv4-uc only; NEVER create machines/TPUs; disk-attach pre-authorized. METHOD:
observability-first; corpus-first re-read on domain shift; ONE VARIABLE AT A TIME; wiring must be
falsifiable (spy/liveness tests, environ checks); honest nulls; small-n never a gate. COMMIT+PUSH
every step. Agents: worktrees, JAX_PLATFORMS=cpu. Serialize TPU. Owner submits PRs; no force-push.

## READ FIRST
HANDOFF.md → RESEARCH_LOG **2026-07-19 21:10 onward** → the running arm's outer log in ~/glm-run/.
Check pgrep -f "hunt_residual[.]sh" BEFORE any pod action.

## LANDMINES
THIS VM IS POD WORKER-0 (--worker=all git mutates the local checkout). pkill/pgrep -f SELF-MATCHES the
shell's eval line — bracket the pattern ("name[.]sh"). `ls -td` globs race outer-log FILES — use
explicit dirs. NEVER edit a script bash is executing. Dump step-files ACCUMULATE (~273MB/step @128K —
disk guard at <15G is armed). Don't sync workers mid-arm (breaks provenance). Load-refusals ≈20% of
draws — classifier handles them (LOAD_REFUSED ≠ the diff-pair sick). Armed-variant cold compiles can
exceed 1h — LADDER_TIMEOUT_S. setsid --wait; RAY_DEDUP_LOGS=0 on forensics; GLM_* raylet AND driver.
