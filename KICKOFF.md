# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to **TPU v4** the DeepSeek-V4-Flash way.
**DON'T STOP/ASK** until (1) it serves correctly, (2) HF-card benchmarks match within noise, (3) the
**DSA sparse-MLA kernel** clears its gates (passkey ≥95% to ≥128K; throughput ≥256K). Self-correct;
when unsure pick + log. Only a hard block pauses THAT thread.

## STATE — DONE (don't redo)
- Stage 1 serves (753B FP8, 32 chips); Stage 2 kernels silicon-validated; MTP code-complete, FROZEN.
- **✅ DENSE 128K GATE CLOSED: 77/77 (Wilson LB 95.3%), run 124, bf16+GLM_DCP=4** — banked + backed up.
- DCP granularity double-×dcp FIXED (1f700c507). bf16+DCP=4 fits 128K (fp8 optional, ~80–122K ceiling).
- **✅ SPARSE RUNG 1 (emit_lse metal unit) + RUNG 2 CLOSED (2026-07-10).** Rung 2's instrument caught a
  REAL silicon defect: pageloop's v4 lowering DROPS sublane row-stripes of the DSA indexer k-cache
  (never-written HBM ⇒ stale-read lottery: evt00-arange / ×¼-stripe / clean-by-luck). Root-caused by
  scrambler-byte-diff; CPU logic exonerated (sentinel suite).
  **Fix landed: GLM_DSA_DCP_SCATTER_IMPL, default flat** (4f7d9a001; dense path keeps its own
  GLM_DCP_SCATTER_IMPL=pageloop — MIRROR-IMAGE metal histories, never share the env).
- **Obs stack (f0c63c302): GLM_DSA_DUMP_TOPK now dumps `topk_scores`; dsa_topk_diff has the SCORE-BLIND
  TRIPWIRE (arange rows ⇒ DIFFER even if runs agree) + `kth_band`.** RUNG CRITERION (owner-ratified):
  selected-set-exact is UNACHIEVABLE cross-run (MoE ulp drift) — the standard is tripwire silent + prefill
  EQUAL + set-diff kth_band ≲ 2× that event's cross-run drift p95. NEVER claim selection health without score-armed dumps.

## ACTIVE FRONTIER (in order)
1. **Sync workers to the fix tip after the adversarial review verdict** (push origin + sync_workers +
   8×same-hash + GLM_EXPECT_CODE_HASH re-pin — pin/sync COLLIDE if an engine is mid-flight, re-pin first).
2. **Rungs 3–6 under the flat default** (docs/11 §8; these double as the fix's pod validation): 4–6K
   chunked prefill + step-HLO honesty → **32K selected-set dcp=2 vs dcp=1, SCORES ARMED both sides**
   (rung45 pageloop-era dumps are void) → 32K/64K sparse passkey smokes.
3. **THE 128K SPARSE GATE — n≥73 zero-failure** (77 = 7 depths×11 incl. 0.0/0.05/0.95/1.0; ~19h; smoke
   the 4 mechanism depths ×1 first; 77/77→LB 95.3%; ONE miss ⇒ extend to n≈130, never rerun-until-green).
4. **256K throughput A/B** (dsa-sparse vs dense, dcp≥4; bench/dsa_throughput.py).
5. **Benchmarks at scale:** GSM8K n≥200, GPQA-198@16K (owner-gated). Then MTP unfreezes.

## HARD RULES
- **COST:** bucket `gs://driftbench-dsv4-uc` (us-central2) ONLY, never EU. **NEVER create a
  machine/VM/TPU** — only the 32 v4 chips / 8 hosts. ONE FP8 copy.
- **METHOD:** FIX root cause, never reward-hack. Every change gated (byte-identical off) + CPU test +
  independent adversarial review + pod validation. Observability-FIRST. Honest nulls (log them).
- **SMALL-n IS NEVER A GATE:** ≥95% needs n≥73 zero-failure. Depths 0.0–0.05 AND 0.95–1.0 REQUIRED.
- Agents edit in scratchpad copies (JAX_PLATFORMS=cpu); TPU serialized to main thread. Commit+push often.
  HF_TOKEN in `.env`. Owner submits upstream PRs.

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (2026-07-10 16:10 onward — the stripe forensics arc) →
docs/11 §8. Fork `glm-5.2-v4-next`; verify tip with `git log --oneline -1` (fix tip 4f7d9a001+).

## LANDMINES
**WORKER STALENESS:** bare push doesn't update origin — push origin + sync_workers + verify 8×same hash
+ bake GLM_EXPECT_CODE_HASH. **The pin fights YOU too:** syncing workers while an engine expecting the
old hash is queued kills it — re-pin until the run finishes. **setsid in FOREGROUND forks+returns
instantly** (watch the real child pid). Armed topk dumps land on ONE host (w-2; JAX callback dedupe).
Cache dumps: raylet-baked envs only (EXTRA_ENVS), per-host /tmp. **"PASS" greps match hlo_passes.cc.** GLM_* envs raylet-baked AND on driver.
Scrambler-byte-diff = the de-lottery protocol for stale-HBM bugs.
