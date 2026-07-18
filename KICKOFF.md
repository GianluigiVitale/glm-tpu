# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). Self-correct; when unsure pick + log.

## STATE (2026-07-18 19:30 UTC; RESEARCH_LOG 07-18 09:05→19:15)
- Fork tip **a225d16b4** synced 8×: t2j alias fix (lottery root cause; 3/3 adversarial reviews banked)
  + PWAL/LOAD NaN refusals + CPU stage-splitter + **GLM_LOAD_CHECKSUM** (categorical per-load byte-verify
  cpu-vs-device at the t2j boundary — catches FINITE corruption; validation 4/4 clean, live 8 hosts,
  verified=1882/host, 312 benign 0-d skips).
- **THE SPARSE 128K GATE IS RUNNING**: ~/glm-run/gate128k_20260718T110127Z, n=77, PIN a225d16b4,
  **33/33 through d=0.0/0.05/0.95** — the gate2 killer cell CLEARED (gate2's 0/11 = the lottery, not the
  kernel). ETA ~05-07 UTC 07-19. Per-depth GCS ckpts; miss-abort at 2; HEALTH_RETRIES=8.
  **CHECK ITS STATE FIRST (orchestrator.log + results.db) — NEVER launch pod work while it runs.**
- **RESIDUAL SPECIMEN (hypothesis REVISED)**: d=0.05 try-1 sick engine was byte-verified CLEAN on every
  surface (H2D checksums, PWAL/LOAD, cache dumps 0-NaN) yet FLUENT-FILLER missed a 5K needle ⇒ residual
  ≈14%/draw is NOT H2D weight corruption. Candidates: (a) CPU-side finite corruption pre-t2j (needs GCS
  reference-checksum manifest vs pre-t2j bytes), (b) engine-instance STATE (XLA program draw, device
  order, KV/selection). Specimen banked: db run 193 + specimen_d005_try1/. Health probe caught it in
  2 min (gate2 burned 10 h on the same class).
- Xprof 128K: NO single dominator (S2 not justified); chunk ≈9.35 s at tiny kv ⇒ per-chunk cost
  dominates prefill. Efficiency targets (post-gate): top_k 21.8%, gathers 16%, collectives ~24%.
- Ready to land post-gate: **~/wt-sibling-alias** (4 commits: weight_utils/gpt_oss/multimodal alias
  fixes + 0-d checksum coverage; 38 tests green). Backup bundle in GCS 07-18.

## FRONTIER (in order)
1. **Gate verdict**: PASS 77/77 ⇒ bank + DSA 128K gate CLOSED (update all docs). ONE miss ⇒ extend
   n≈130. 2-miss abort ⇒ forensics from the armed instruments + the specimen, never rate experiments.
2. **256K**: fit/geometry probe (dcp=8 deferred to this stage; novel geometry = cold compile ~40 min),
   correctness needles, then sparse-vs-dense throughput A/B at IDENTICAL dcp.
3. Land sibling-alias; draft the t2j-fix upstream PR (owner submits).
4. Residual hunt FROM THE SPECIMEN: (a) CPU reference checksums or (b) engine-state instruments.
5. Benchmarks: GSM8K n≥200, GPQA-198@16K (owner-gated), fp8-KV dcp=1 needle, MTP unfreeze, PR re-cut.

## HARD RULES
COST: gs://driftbench-dsv4-uc only; NEVER create machines/TPUs; disk-attach to the 8 hosts
pre-authorized. METHOD: observability-first; **corpus-first re-read on any domain shift**; fix root
cause; gated + CPU test + adversarial review + pod validation; honest nulls; small-n never a gate.
COMMIT+PUSH every milestone. Agents: worktrees, JAX_PLATFORMS=cpu, read-only on the fork. Serialize TPU
access. Owner submits PRs; no force-push (follow-up commits only).

## READ FIRST
HANDOFF.md → RESEARCH_LOG 07-18 09:05 on → the gate orchestrator.log. Verify 8× a225d16b4.

## LANDMINES
**THIS VM IS POD WORKER-0** — a `--worker=all` git command mutates the LOCAL checkout too. `ls -td
~/glm-run/gate128k_*` races gate128k_outer.log — use the explicit run-dir name. Sync verify 8× ALWAYS
(stale index.lock ⇒ rm + re-pull). RAY_DEDUP_LOGS=0 on forensics. Armed PWAL raise preempts LOAD
census. Init geometry must match the warm XLA cache (novel = ~40 min compile). setsid --wait; pkill
patterns paren-free. "PASS" greps match hlo_passes.cc. GLM_* raylet-baked AND driver-exported. Scatter
bakes per-path (dense pageloop STAYS; DSA flat unset). Disk: w-0 baseline includes scratch; archive+purge
per draw. Poisoned dump tars compress ~150:1. Firewall tag orphans on pod recreation (memory:
glm-pod-worker0-vm-identity).
