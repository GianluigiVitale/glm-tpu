# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4. DON'T STOP/ASK until (1) it serves correctly,
(2) HF-card benchmarks within noise, (3) the **DSA sparse kernel** clears its gates (passkey ≥95%@128K
n≥73; throughput ≥256K). Self-correct; when unsure pick + log.

## STATE (VM lost+recovered 07-16; nothing committed lost)
- Stage 1 serves; Stage-2 kernels silicon-validated; MTP FROZEN. ✅ DENSE 128K GATE CLOSED 77/77 (run
  124); sparse rungs 1-6 closed; 128K mechanism smokes 4/4 EXACT ×2.
- ✅ EFFICIENCY CAMPAIGN 12.6× at 128K (sparse prefill beats dense): segment prefill +
  BT_WIDTH=owned + MERGE/OWNED_SEG/SEG_GATHER=v2 + chunk 2048. Pin a98c77c9.
- ❌ GATE2 DIED: 22/22 then d=0.95 **0/11 NEW signature** (fluent filler = needle NOT retrieved). d=1.0
  next engine 2/2 ⇒ engine lottery ~1/7 — **CONFOUNDED: w-4 disk at 0 in that window**. UNRESOLVED.
- Known-broken on metal (refusals NOT in code): headsplit×segment; headsplit dcp=2/H_local=4 →
  HEADSPLIT stays OFF (free under segment).
- 07-16: pod recreated → reprovisioned 7/7 @ a98c77c9 (scripts/provision_worker_glm.sh); XLA caches
  COLD. LOST: safety commit, write-probe guard, disk-watchdog, orchestrators, 14-probe results — rebuild.

## FRONTIER (in order)
1. **Ops-debt/safety commit** (CPU): disk-watchdog + dump quotas; F6 headsplit×segment trace-time
   refusal; docstring truth pass; combo-matrix tests; F4 RELAUNCH env fix; miss-abort watchdog;
   write-probe guard (spec in docs/upstream/).
2. **THE 14-PROBE DISCRIMINATOR**: 14× single-needle 128K engines, FIXED seed, gate config (dcp=4,
   segment, headsplit off, flat, chunk 2048, owned/all-v2, APC off), GLM_DCP_CACHE_DUMP armed HOST-side
   only (traced program UNCHANGED — F3); watcher classifies INFRA-FAIL vs verdict. Bad draws recur ⇒
   real lottery ⇒ byte-diff bad-vs-good prefill (scrambler protocol) →
   bisect. 14/14 good ⇒ disk pressure ⇒ ops fixes only.
3. Masked-backstop smoke (F7) + one armed bitwise cell at T=2048 (F8), then **RE-GATE sparse 128K
   n=77** (7 depths×11, mechanism cells FIRST, per-depth GCS checkpoints, ~12.7h; ONE miss ⇒ extend
   n≈130, never rerun-until-green).
4. 256K A/B dense-vs-sparse at IDENTICAL dcp; fp8-KV only after its dcp=1 needle (fp8 NEVER made a
   metal token). 5. GSM8K n≥200; GPQA@16K (owner-gated); MTP unfreezes; PR re-cut.

## HARD RULES
- COST: gs://driftbench-dsv4-uc (us-central2) ONLY, never EU. NEVER create a machine/VM/TPU. ONE FP8 copy.
- METHOD: observability-FIRST (docs/suggestions.md): build the instrument that makes failure VISIBLE
  before debugging; ad-hoc verify scripts are instruments too. FIX root cause. Every change gated
  (byte-identical off) + CPU test + adversarial review + pod validation. Honest nulls.
- SMALL-n IS NEVER A GATE: n≥73 zero-failure; depths 0.0/0.05 AND 0.95/1.0 REQUIRED.
- COMMIT+PUSH every milestone — uncommitted work DIES with the VM (proven 07-16); bucket mirror ≠
  backup. Agents: worktrees, JAX_PLATFORMS=cpu; TPU serialized to main. HF_TOKEN in .env. Owner
  submits PRs.

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (07-12 20:30 on) → docs/11 §8. Fork `glm-5.2-v4-next`;
verify tip a98c77c9 on all 8 hosts.

## LANDMINES
**WORKER STALENESS:** push origin explicitly + sync_workers.sh + 8× same hash + GLM_EXPECT_CODE_HASH
(re-pin fights queued engines). **DISK PRESSURE = first-class failure:** df ALL 8 hosts before
long runs; gate-class runs UNARMED (armed 128K ≈ 230GB). **Scatter impls are per-path mirror images:**
dense GLM_DCP_SCATTER_IMPL=pageloop STAYS raylet-baked (code-default is metal-BAD; the dense fallback
fires INSIDE sparse serving at ctx≤topk); DSA default flat — leave unset. GLM_* envs raylet-baked
(EXTRA_ENVS) AND on driver; trace-time env = a DIFFERENT program. setsid FOREGROUND forks+returns
instantly. "PASS" greps match hlo_passes.cc. Cross-run
selected-set-exact UNACHIEVABLE (MoE drift): criterion = tripwire silent + untruncated EQUAL +
kth_band ≲ 2× drift p95.
