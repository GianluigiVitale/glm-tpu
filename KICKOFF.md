# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to **TPU v4** the DeepSeek-V4-Flash way.
**DON'T STOP/ASK** until (1) it serves correctly, (2) HF-card benchmarks match within noise, (3) the
**DSA sparse-MLA kernel** clears its gates (passkey ≥95% to ≥128K; throughput ≥256K). Self-correct;
when unsure pick + log. Only a hard block pauses THAT thread.

## STATE — DONE (don't redo)
- Stage 1: 753B FP8 serves on 32 v4 chips. Stage 2 kernels silicon-validated (selected-set-EXACT);
  sparse passkey 100% @8K/32K. Stage 3 MTP code-complete (CPU).
- **64K dense 6/6 @dcp=1/bf16** (run 100). dcp=1 128K wall = HBM FRAGMENTATION (contiguous-free
  lottery); ceiling ~88–90K; anti-frag = LIBTPU rerun=5 + rwb_fusion=false.
- **fp8-KV COMPILES on v4** (pack_new_kv i8 fixes) — but fp8 NEVER generated a metal token; a cheap
  fp8 needle @dcp=1 is required first.
- **✅ DCP FIXED (1f700c507): block-table granularity double-×dcp** (engine ids 512·dcp² tok vs TPU
  512·dcp/entry). The "packed-write metal bug" NEVER EXISTED. Post-fix: 900tok/4K/16K ✓, 128K smoke
  7/7 (first ever). **bf16+DCP=4 fits 128K without fp8** (~29.2/30.75G). Upstream #3129 fixed the same
  bug 3h before us (broken v0.20–v0.24) — convergent; do NOT file.
- **Sparse-DCP Stages A+B+C CODE-COMPLETE, CPU-CERTIFIED ONLY** (tip 6f8855c3f; Stage B reviewed
  SAFE-FOR-METAL-LADDER; Stage C review in flight; gate-off jaxpr == HEAD). Zero sparse-DCP tokens on
  metal. Obs kit repaired fail-loud (6f45e0944: partial-dump refusal, GLM_EXPECT_CODE_HASH,
  granularity assert).

## ACTIVE FRONTIER (in order)
1. **Finish the DENSE 128K gate n=77 — RUNNING** (run_id 124: 7 depths×11, bf16+GLM_DCP=4, ~19h;
   42/42 @05:53 07-10; watchdog armed). Record + backup when it closes.
2. **Sparse metal ladder — docs/11 §8** (after Stage-C verdict + worker sync): emit_lse unit @dcp=1 →
   sparse decode dcp=2 ≤2048tok vs dcp=1 (logprobs+stash) → chunked prefill 4–6K + step-HLO honesty
   (ONE candidate all-gather pair/full layer, NO whole-cache collectives) → 32K selected-set dump dcp=2
   vs dcp=1 → 32K/64K passkey → **128K SPARSE gate n≥73**. Watch: emit_lse Mosaic lowering,
   ~134MB/layer/chunk candidate all-gather @T=2048/dcp=4, per-shard flash transient.
3. **256K throughput A/B** (dsa-sparse vs dense, dcp≥4; bench/dsa_throughput.py).
4. **Benchmarks at scale**: GSM8K n≥200 (retire the n=32 smoke), GPQA-198 @16K (owner-gated).

## HARD RULES
- **COST:** bucket `gs://driftbench-dsv4-uc` (us-central2) ONLY, never EU. **NEVER create a
  machine/VM/TPU** — only the 32 v4 chips / 8 hosts (disk-attach OK). ONE FP8 copy.
- **METHOD:** FIX root cause, never reward-hack. Every change gated (byte-identical off) + CPU test +
  independent adversarial review + pod 3/3. Observability-FIRST. Honest nulls.
- **SMALL-n IS NEVER A GATE:** ≥95% needs **n≥73 zero-failure** (~200 to survive one miss). Mechanism
  depths 0.0–0.05 AND 0.95–1.0 are REQUIRED cells. SMOKE ≠ GATE — label them.
- **MTP stays FROZEN** until the 128K dense+sparse and 256K gates close.
- Agents edit in WORKTREES (JAX_PLATFORMS=cpu); TPU serialized to main thread. Commit+push often; no
  force-push; owner submits upstream PRs. HF_TOKEN in `.env` (never commit).

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (07-09 09:20 on) → docs/11 §8. Fork
`glm-5.2-v4-next` @6f8855c3f.

## LANDMINES
**WORKER STALENESS:** 8 per-host checkouts; bare `git push -q` didn't update origin — after ANY fork
commit: push origin + `TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash scripts/sync_workers.sh`, verify all
8 = SAME hash, bake GLM_EXPECT_CODE_HASH. **setsid nohup** every driver (else SIGTERM'd mid-serve).
**"PASS" greps match hlo_passes.cc** — read real needle lines. GLM_* envs raylet-baked via EXTRA_ENVS
AND on driver. GLM routes via VllmModelWrapper. Relaunch after any pod crash.
