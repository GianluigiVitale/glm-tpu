# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). **OWNER RULE: NO SHORTCUTS — root cause properly fixed + validated before any
re-gate; no workaround gating.** Self-correct; when unsure pick + log.

## STATE (2026-07-23 08:30 UTC; RESEARCH_LOG 07-22 21:15 → 07-23 08:10)
- ⭐ **ROOT CAUSE FOUND AND MEASURED (5-day hunt closed): STREAMER FINITE-CORRUPTION of loaded
  weights.** The runai streamer probabilistically delivers corrupt-but-FINITE bytes for ~a few random
  tensors per host per launch (deterministic wrong bytes when it strikes; e.g. layers.10 indexer
  wk_weights_proj sum 48387836 vs true 239851472, its adapted derivative ZEROED). NaN scans + the H2D
  checksum are blind to the finite flavor BY DESIGN. Victim = an indexer weight ⇒ that layer's
  selections degrade ⇒ the whole "state class" (per-instance score states, pos≥2048 gating, entry-layer
  variation, mangled digits, gate2/3 deaths). Severity = how many host replicas share the victim
  (1/8 ⇒ healthy-divergent; 8/8 ⇒ sick). The "load class" (NaN) and "state class" (finite) were ONE BUG.
- PROOF CHAIN (all banked): entry bracketed L15/L17 → GLM_STATE_HASH (8448b738c, fingerprints all
  19,640 leaves incl. derived) → statepair arm: sick engine differs on EXACTLY the 2 layer-10 leaves ×8
  hosts; healthy pair differs on 3 single-host leaves (benign carriers).
- Fork tip **8448b738c** synced 8×. In flight: GLM_STATE_HASH_REF manifest-refusal build (agent).
- Instruments proven en route: onehot scorer (harmless, keep), idx-no-donate (harmless), checksum/NaN
  stack, the 30-90min ladder repro (hunt_residual.sh + overrides).
## FRONTIER (in order)
1. Land GLM_STATE_HASH_REF (manifest refusal — the categorical finite-class detector): bootstrap the
   golden manifest from a verified engine (GLM_STATE_HASH_WRITE), cross-check vs the banked healthy
   sums, commit manifest to gs://driftbench-dsv4-uc/manifests/.
2. **THE STREAMER FIX**: gcsfuse Plan A (mount gs://driftbench-dsv4-uc/models on all 8 hosts, switch
   GLM_MODEL/load path; $0) — else the pre-authorized local-disk attach. Validate: N init draws, ALL
   manifest-clean (vs the ~2-3 corrupt-leaf/launch baseline).
3. **GATE4** n=77 @128K: fixed load path + manifest refusal + full stack + 32K health needle.
4. 256K (stage256k.sh — update PIN/envs). 5. Benchmarks (GSM8K n≥200, GPQA@16K owner-gated), MTP,
   sibling-alias landing, PR re-cut (incl. the STREAMER BUG REPORT upstream — deterministic-bytes
   repro makes it filable).
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
draws — classifier handles them (LOAD_REFUSED ≠ the diff-pair sick). Armed cold compiles can exceed
1h (LADDER_TIMEOUT_S). setsid --wait; RAY_DEDUP_LOGS=0 forensics; GLM_* raylet AND driver.
