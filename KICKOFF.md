# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves
correctly, (2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates
(passkey ≥95%@128K n≥73; throughput ≥256K). Self-correct; when unsure pick + log.

## STATE (post lottery-hunt 07-17; RESEARCH_LOG 08:20→19:10 = the six-instrument arc)
- Stage 1 serves; kernels silicon-validated; ✅ DENSE 128K GATE CLOSED 77/77; 12.6× efficiency
  campaign (sparse prefill beats dense); MTP FROZEN.
- ✅ Safety commit LANDED (fork tip **34d2eef37**, synced 8×): F6 headsplit refusals,
  GLM_WRITE_PROBE scatter sentinel, combo suite (F8 closed), ops kit (disk watchdog, dump
  archiver, orchestrators with miss-abort + INFRA taint + engine health probe).
- ✅ **LOTTERY SOLVED-IN-CLASS.** Scrambled discriminator: 4/6 valid draws MISS (67%). Byte-diff →
  quiet-NaN in layer-1's indexer k-cache, whole-HOST granularity; binaries exonerated (per-host
  fingerprints identical); PWAL copies exonerated (code map: the k-write is shared straight-line
  code). Then **GLM_PWAL_NAN_CHECK caught it AT INIT: loaded wk NaN/Inf on 3 hosts — the per-host
  runai stream + fp8 dequant silently delivers corrupt tensors** (e4m3 decodes garbage bytes to
  NaN; the DSV4-era "flaky dequant crash", now silent). Currently it only LOGS (attribution=
  UPSTREAM) — engines still serve corrupt loads.
- OPEN: p5-class miss (clean engine, coherent filler, all 22 dumped slots clean — likely the same
  load bug in an unscanned tensor); page-0-clean reconciliation (full per-step specimen:
  gs://driftbench-dsv4-uc/dumps/probe_timeline_20260717T165934Z/probe3_MISS/).

## FRONTIER (in order)
1. **Fork: refuse corrupt loads** — PWAL check RAISES on UPSTREAM non-finite when armed, and WIDEN
   the init scan to ALL loaded weights (also answers the p5 blast radius). CPU tests + review +
   land + sync 8×.
2. **LOADER FIX HUNT** (root fix > detect-and-relaunch): reload-and-compare a flagged tensor;
   streamer integrity/retry (RUNAI_* knobs); fp8-dequant adapter race audit.
3. p5-class discriminator only if the widened scan doesn't explain it (clean-engine d=0.95
   repeats, armed topk scores).
4. **RE-GATE: gate_sparse128k.sh** (n=77, mechanism depths first; health probe + refusing loads =
   double protection; miss-abort at 2; per-depth GCS ckpts; ONE miss ⇒ extend n≈130, never
   rerun-until-green).
5. 256K A/B at IDENTICAL dcp; fp8-KV only after its dcp=1 needle. 6. GSM8K n≥200; GPQA@16K
   (owner-gated); then MTP unfreezes; PR re-cut.

## HARD RULES
- COST: gs://driftbench-dsv4-uc (us-central2) ONLY. NEVER create a machine/VM/TPU. ONE FP8 copy.
- METHOD: observability-FIRST (docs/suggestions.md — the six-instrument arc is the proof); FIX
  root cause; every change gated + CPU test + adversarial review + pod validation; honest nulls.
- SMALL-n IS NEVER A GATE (n≥73 zero-failure; depths 0.0/0.05 AND 0.95/1.0 REQUIRED).
- COMMIT+PUSH every milestone; agents in worktrees, JAX_PLATFORMS=cpu, NEVER git on worker
  checkouts (index.lock races engine init); TPU serialized to main. Owner submits PRs.

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md 2026-07-17 (08:20 on) → docs/11 §8.
Fork `glm-5.2-v4-next`; verify tip 34d2eef37 on all 8 hosts.

## LANDMINES
Sync verify 8× ALWAYS (partial syncs hit 4/8 TWICE — stale index.lock: clear + re-pull).
Firewall tag orphans on pod recreation (update allow-ray-pod-internal target-tags). DISK: w-0's
baseline includes scratch/session-logs — watchdog check before long runs; dumps to GCS or deleted.
Scatter impls per-path (dense pageloop STAYS raylet-baked; DSA flat unset). GLM_* raylet-baked
(EXTRA_ENVS) AND driver-exported; trace-time env = DIFFERENT program. setsid needs --wait.
"PASS" greps match hlo_passes.cc. Armed dumps: LAYERS=2-only ≈ 4.5MB/step/proc (fine); all-slot
≈ 29G/host transient. Cross-run selected-set-exact UNACHIEVABLE: tripwire + kth_band criterion.
