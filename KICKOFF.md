# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). **OWNER RULE: NO SHORTCUTS — root cause properly fixed + validated before any
re-gate; no workaround gating.** Self-correct; when unsure pick + log.

## STATE (2026-07-24 13:15 UTC; RESEARCH_LOG 07-23 21:40 → 07-24 04:00)
- ⭐ **ENGINE LOTTERY SOLVED + VALIDATED; docs/17 COMPLETE (incl. Phase J) — the record.** Streamer
  delivers corrupt bytes ~0-3 tensors/host/launch in THREE flavors, all measured: **zero-fill**
  (AUTO-REPAIRED — GLM_WK_OOB_DIR PWAL-time bitwise repair from the gcsfuse mirror; 5/5 in gval),
  **NaN** + **finite-garbage** (REFUSED — NaN scans + GLM_STATE_HASH_REF golden manifest; garbage
  specimen layers.1 banked 07-24 02:50). Stack = repair → refuse → relaunch; ZERO unverified serves
  possible. Root-of-root (streamer/_free_cpu_storage ordering) = upstream-report material, NOT a blocker.
- **GATE4 v3 RUNNING** (relaunched 13:10 UTC, ~/glm-run/gate128k_20260724T131056Z, n=77, PIN
  dc0443a43, REF+OOB armed). Attempt-1 aborted INFRA (disk watchdog; 0 misses counted): 35G of
  closed-hunt scratchpad debris on w-0 — cleaned (hosts 53-77G free), lesson logged. Attempt-1
  depth 0.0 drew HEALTHY on try 3 (after 2 CORRECT refusals) — the stack works at gate geometry.
- Fork tip **dc0443a43** synced 8× (sync_workers.sh MACHINE-VERIFIES 8×HEAD==origin + no index.lock).
- Torchax: PWAL escapes need the PAIR `no_dispatch(), DisableTorchFunction()`; CPU tests must run
  under torchax.default_env() (docs/17 §6(g)).

## FRONTIER (in order)
1. **GATE4 v3 to verdict** (~15-20 h from 13:10): PASS 77/77 ⇒ bank + backup_bundle.sh. ONE miss ⇒
   extend n≈130; 2-miss abort ⇒ forensics FROM THE SPECIMEN (never rate experiments).
   Refused/repaired engines are NORMAL (HEALTH_RETRIES=8/depth). If a depth STARVES on retries ⇒
   land the banked MANIFEST-DRIVEN PWAL repair (verify the fused leaf vs /tmp/golden.json at PWAL,
   repair from mirror on ANY mismatch) + short revalidation arm, then re-gate.
2. **256K**: stage256k.sh (READY @ dc0443a43, REF+OOB armed) — dcp=8 bring-up → 32K sanity ×2 →
   256K mechanism smoke → sparse-vs-dense throughput A/B at IDENTICAL dcp=8.
3. **Benchmarks**: GSM8K n≥200 → AIME-2026 n=30 → GPQA-198@16K (owner-gated go).
4. **MTP M2 unfreeze** (after gates). 5. Land ~/wt-sibling-alias; PR series re-cut; the STREAMER
   upstream bug report. Post-gate cleanups: health-classifier VERIFIED count (require 8),
   line-168 noise, wk-oob NaN-half repair ext.

## HARD RULES
COST: gs://driftbench-dsv4-uc only; NEVER create machines/TPUs; disk-attach pre-authorized. METHOD:
observability-first; corpus-first re-read on domain shift; ONE VARIABLE AT A TIME; falsifiable wiring;
honest nulls; small-n never a gate (n≥73; mechanism depths 0.0/0.05/0.95/1.0 REQUIRED). COMMIT+PUSH
every step. Agents: worktrees, JAX_PLATFORMS=cpu. Serialize TPU. Owner submits PRs; no force-push.

## READ FIRST
HANDOFF.md (07-24 header) → docs/17 (§5 protections, §6 rules) → RESEARCH_LOG **07-23 21:40
onward** → the running gate's orchestrator.log. Check pgrep -f "gate_sparse128k[.]sh" BEFORE any
pod action.

## LANDMINES
THIS VM IS POD WORKER-0 (--worker=all git mutates the local checkout). pkill/pgrep -f SELF-MATCHES —
bracket the pattern ("name[.]sh"). NEVER edit a script bash is executing. `ls -td` races outer-log
FILES — use explicit run dirs. Dumps + SESSION SCRATCHPAD accumulate — purge closed-campaign scratch
at campaign close (35G of it caused the attempt-1 abort); disk watchdog floors at 15G. Don't sync
workers mid-arm. setsid --wait nohup </dev/null every driver; RAY_DEDUP_LOGS=0; GLM_* raylet AND
driver. gcsfuse mounts drop on relaunch — orchestrators re-ensure them; manual launches must too.
