# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). **OWNER RULE: NO SHORTCUTS — root cause properly fixed + validated before any
re-gate; no workaround gating.** Self-correct; when unsure pick + log.

## STATE (2026-07-24 01:30 UTC; RESEARCH_LOG 07-23 21:40 → 07-24 01:20)
- ⭐ **THE ENGINE LOTTERY IS SOLVED END-TO-END** (6-day hunt; full story docs/17 incl. Phase J).
  Streamer delivers corrupt-but-finite bytes ~0-3 tensors/host/launch; window narrowed
  **post-dequant/pre-t2j CPU storage** (dequant-time check saw GOOD values; suspect
  _free_cpu_storage resize_(0) ordering — upstream-report audit material). THE FIX, **VALIDATED on
  metal** (gval 4 draws, PIN dc0443a43): the SELF-HEALING LOAD — GLM_WK_OOB_DIR PWAL-time zero-fill
  repair from the gcsfuse mirror (5/5 strikes repaired bitwise, layers.10 ×4 + layers.1) →
  GLM_STATE_HASH_REF golden-manifest refusal (every serving engine VERIFIED 8/8; 12/12 needles) →
  NaN-scan refusal (1/1 fail-closed). Repair → refuse → relaunch. Zero unverified serves.
- **GATE4 v3 (sparse 128K, n=77) LAUNCHED 07-24 01:25** — gate_sparse128k.sh @ dc0443a43, REF+OOB
  armed, hardened (health probe, empty-depth retry, oob-mount preflight+remount, miss-abort at 2).
  Run dir ~/glm-run/gate128k_20260724T*. Refused/repaired engines are NORMAL — relaunch handles them.
- Fork tip **dc0443a43** synced 8× (sync_workers.sh now MACHINE-VERIFIES 8×HEAD==origin + no
  index.lock — a stale lock silently ate w6's reset 07-23; the fingerprint guard caught it in 75 s).
- Torchax lesson (docs/17 §6(g)): PWAL runs under BOTH torchax modes; escape = the PAIR
  `no_dispatch(), DisableTorchFunction()`; CPU tests must run under torchax.default_env().

## FRONTIER (in order)
1. **GATE4 v3 to verdict** (~14-20 h): PASS 77/77 ⇒ bank + backup_bundle.sh. ONE miss ⇒ extend
   n≈130; 2-miss abort ⇒ forensics FROM THE SPECIMEN (never rate experiments).
2. **256K**: stage256k.sh (READY @ dc0443a43, REF+OOB armed) — dcp=8 bring-up → 32K sanity ×2 →
   256K mechanism smoke → sparse-vs-dense throughput A/B at IDENTICAL dcp=8.
3. **Benchmarks**: GSM8K n≥200 → AIME-2026 n=30 → GPQA-198@16K (owner-gated go).
4. **MTP M2 unfreeze** (after gates). 5. Land ~/wt-sibling-alias; PR series re-cut; the STREAMER
   upstream bug report (deterministic zero-fill + the window + _free_cpu_storage audit). Banked
   optional: wk-oob guard NaN-half repair (converts NaN-refusals into serves).

## HARD RULES
COST: gs://driftbench-dsv4-uc only; NEVER create machines/TPUs; disk-attach pre-authorized. METHOD:
observability-first; corpus-first re-read on domain shift; ONE VARIABLE AT A TIME; falsifiable wiring;
honest nulls; small-n never a gate (n≥73; mechanism depths 0.0/0.05/0.95/1.0 REQUIRED). COMMIT+PUSH
every step. Agents: worktrees, JAX_PLATFORMS=cpu. Serialize TPU. Owner submits PRs; no force-push.

## READ FIRST
HANDOFF.md (07-24 header) → docs/17 (the post-mortem — §5 protections, §6 rules) → RESEARCH_LOG
**07-23 21:40 onward** → the running gate's orchestrator.log. Check
pgrep -f "gate_sparse128k[.]sh" BEFORE any pod action.

## LANDMINES
THIS VM IS POD WORKER-0 (--worker=all git mutates the local checkout). pkill/pgrep -f SELF-MATCHES —
bracket the pattern ("name[.]sh"). NEVER edit a script bash is executing. `ls -td` races outer-log
FILES — use explicit run dirs. Dump step-files ACCUMULATE (~273MB/step @128K; disk watchdog armed —
gate-class runs UNARMED except LAYERS=2 health dumps, purged per depth). Don't sync workers mid-arm.
setsid --wait nohup </dev/null every driver; RAY_DEDUP_LOGS=0; GLM_* raylet AND driver. Armed cold
compiles can exceed 1h. gcsfuse mounts drop on relaunch — orchestrators re-ensure them (preflight
does it; manual launches must too).
