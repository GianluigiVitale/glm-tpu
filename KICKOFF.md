# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to TPU v4. DON'T STOP/ASK until (1) it serves
correctly, (2) HF-card benchmarks match within noise, (3) the **DSA sparse kernel** clears its gates
(passkey ≥95%@128K at n≥73; throughput ≥256K). Self-correct; when unsure pick + log.

## STATE (post VM-loss recovery 07-16; nothing committed was lost)
- Stage 1 serves; Stage-2 kernels silicon-validated; MTP code-complete, FROZEN.
- ✅ DENSE 128K GATE CLOSED 77/77 (run 124, bf16+GLM_DCP=4). Sparse rungs 1-6 closed; 128K mechanism
  smokes 4/4 EXACT ×2.
- ✅ EFFICIENCY CAMPAIGN 12.6× at 128K (sparse prefill ~220 tok/s BEATS dense): segment prefill +
  BT_WIDTH=owned + MERGE/OWNED_SEG/SEG_GATHER=v2 + chunk 2048. Pin a98c77c9 (fork tip, synced 8×).
- ❌ GATE2 DIED: 22/22 then d=0.95 **0/11, NEW signature** (fluent filler = needle NOT retrieved; run
  165). d=1.0 next engine 2/2 ⇒ engine-instance lottery ~1/7 — **CONFOUNDED: w-4 disk at 0 through that
  window**. UNRESOLVED.
- Known-broken on metal (refusals NOT in code yet): headsplit×segment; headsplit dcp=2/H_local=4.
  HEADSPLIT stays OFF (adds nothing under segment).
- 07-16: pod recreated (disks wiped) → reprovisioned 7/7 @ a98c77c9 (scripts/provision_worker_glm.sh);
  XLA caches COLD. LOST, rebuild: safety commit, write-probe guard, disk-watchdog, orchestrators,
  the 14-probe results.

## ACTIVE FRONTIER (in order)
1. **Ops-debt/safety commit** (CPU; every recent gate death traces here): disk-watchdog + dump quotas;
   F6 headsplit×segment trace-time refusal; docstring truth pass; combo-matrix tests; F4 RELAUNCH env
   fix; miss-abort watchdog; write-probe guard (spec: docs/upstream/pageloop report).
2. **THE 14-PROBE DISCRIMINATOR**: 14× single-needle 128K engines, FIXED seed, gate config (dcp=4 +
   segment + headsplit off + flat + chunk 2048 + owned/all-v2 + APC off), GLM_DCP_CACHE_DUMP armed
   HOST-side only (traced program UNCHANGED — F3); watcher classifies INFRA-FAIL vs verdict. Bad draws
   recur ⇒ real lottery ⇒ byte-diff bad-vs-good prefill (scrambler protocol) → bisect that side.
   14/14 good ⇒ it was disk pressure ⇒ ops fixes only.
3. Masked-backstop smoke on the owned/v2 program (F7) + one armed bitwise cell at T=2048 (F8), then
   **RE-GATE sparse 128K n=77** (7 depths×11, mechanism cells FIRST, per-depth GCS checkpoints, ~12.7h;
   ONE miss ⇒ extend n≈130, never rerun-until-green).
4. 256K A/B dense-vs-sparse at IDENTICAL dcp; fp8-KV only after its own dcp=1 needle (fp8 has NEVER made
   a validated metal token). 5. GSM8K n≥200; GPQA@16K (owner-gated). Then MTP unfreezes; PR re-cut.

## HARD RULES
- COST: gs://driftbench-dsv4-uc (us-central2) ONLY, never EU. NEVER create a machine/VM/TPU. ONE FP8 copy.
- METHOD: observability-FIRST (docs/suggestions.md): build the instrument that makes the failure VISIBLE
  before debugging; ad-hoc verify scripts are instruments too — overriding a tool's verdict needs equal
  rigor. FIX root cause. Every change gated (byte-identical off) + CPU test + adversarial review + pod
  validation. Honest nulls; the record over the narrative.
- SMALL-n IS NEVER A GATE: n≥73 zero-failure; depths 0.0/0.05 AND 0.95/1.0 REQUIRED.
- COMMIT+PUSH every milestone — uncommitted work DIES with the VM (proven 07-16); the bucket mirror is
  NOT a backup. Agents: worktrees, JAX_PLATFORMS=cpu; TPU serialized to main. HF_TOKEN in .env. Owner
  submits PRs.

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (07-12 20:30 on — campaign → gate2 → confound → recovery)
→ docs/11 §8. Fork `glm-5.2-v4-next`; verify tip a98c77c9 on all 8 hosts.

## LANDMINES
**WORKER STALENESS:** push origin explicitly + sync_workers.sh + 8× same hash + GLM_EXPECT_CODE_HASH
(the pin fights YOU too — re-pin collides with a queued engine). **DISK PRESSURE = first-class failure:**
df ALL 8 hosts before long runs; gate-class runs UNARMED (armed 128K ≈ 230GB). **Scatter impls are
per-path mirror images:** dense GLM_DCP_SCATTER_IMPL=pageloop STAYS raylet-baked (its code-default is
metal-BAD; the dense fallback fires INSIDE sparse serving at ctx≤topk); DSA default flat — leave unset.
GLM_* envs raylet-baked (EXTRA_ENVS) AND on driver; trace-time env = a DIFFERENT program. setsid in
FOREGROUND forks+returns instantly. "PASS" greps match hlo_passes.cc. Armed topk dumps land on ONE host
(w-2). Cross-run selected-set-exact UNACHIEVABLE (MoE ulp drift): criterion = tripwire silent +
untruncated region EQUAL + kth_band ≲ 2× drift p95. Scrambler-byte-diff = the de-lottery protocol for
stale-HBM bugs.
