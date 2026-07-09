# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to **TPU v4** the DeepSeek-V4-Flash way.
**DON'T STOP/ASK** until (1) it serves correctly, (2) HF-card benchmarks match within noise, (3) the **DSA
sparse-MLA kernel** clears its gates (passkey ≥95% to ≥128K; throughput ≥256K). Self-correct; when unsure
pick + log. Only a hard block (credential, owner-gated push) pauses THAT thread.

## STATE — what's DONE (don't redo)
- **Stage 1 DONE:** 753B FP8 serves on 32 v4 chips (runai stream, FP8-resident, EP filter, pure TP×EP).
  GSM8K **96.9% (31/32)** = a **SMOKE test (n=32)**, NOT "at scale" (Wilson ~84–99%; needs n≥200 for a scale claim).
- **Stage 2 kernels SILICON-VALIDATED (single v4 chip):** GATE 2a (Mosaic compile, 3 lowering fixes +
  w-tile fallback) + 2b (**selected-set-EXACT** vs HF oracle; sparse-MLA fp32 9.5e-7 / bf16 1.95e-3).
  **Sparse passkey 100% every depth @8K AND 32K** — at 32K DSA attends only the 2048 selected positions
  (16× sparsification: the SELECTION works, not just runs). Sparse decode + prefill + IndexShare built.
- **Stage 3 code-complete (CPU):** MTP M1 draft-parity + G4 index-share; dense-MTP knob GLM_SPEC_K.
- Fork **`glm-5.2-v4-next`** = all above + DCP + guards, gated off. PRs g1–g6 drafted.

**CORRECTED 2026-07-09 (8 on-pod fit-checks): 128K batch=1 correctness is NOT a cache-capacity wall — it's
HBM FRAGMENTATION at compile.** The old "MLA latent replicated → 11.9 GiB/chip → needs DCP/fp8" premise is
**FALSE for batch=1**: 128K latent ≈ **~150 MiB/chip**, a 128K prompt leaves **>1 GiB free**. Real failure:
an XLA compile-scratch of **0.625 MiB × num_gpu_blocks** (161M @ the 258-block 128K floor) finds no
**contiguous** slot (best ~148M). gmu is inert once `num_gpu_blocks_override` is set; the lever is
**`LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"`** (scheduler ON,
118→148M). **64K builds clean @ dcp=1** (buffer ~84M). 128K is **~13.5M short** → next lever: raise KV
**block_size** (512→1024 halves num_blocks → buffer ~80M). **DCP/fp8 are for the throughput/256K gate, NOT
batch-1 correctness.** Still-true (throughput-scoped): DCP multi-chunk packed-WRITE metal bug; fp8-KV SHELVED
(2× v4 Mosaic cmpi); stale-stripe persistence FIXED.

## ACTIVE FRONTIER — do next, in order
1. **Land 128K dense passkey @ batch=1/dcp=1** — defeat compile-fragmentation (best flags −13.5M): raise KV
   block_size (fewer blocks → smaller scratch) or tune scheduler-rerun. 64K builds; verify needle, push to 128K.
2. Re-enable **DSA sparse** on the working dcp=1 path for the **128K SPARSE** gate (kernel PASSED silicon @32K).
3. **256K throughput** A/B (dsa-sparse vs dense) — where DCP packed-write / fp8-KV are actually needed.

## HARD RULES
- **COST:** bucket = **`gs://driftbench-dsv4-uc` (us-central2) ONLY**, never EU `gs://driftbench-storage`.
  **NEVER create a new machine/VM/TPU** — only the 32 v4 chips / 8 hosts (disk-attach OK). ONE FP8 copy.
- **METHOD (DSV4's):** FIX root cause, never patch/reward-hack. Every change gated (byte-identical off) +
  CPU test + **independent adversarial review** + pod 3/3. **Observability-FIRST** (cache-dump, A/B/C,
  scatter-only, new-KV dump, standing guards, docs/10) — it localized every metal bug. Honest nulls.
- **WORKTREES only** — NEVER edit the main `~/tpu-inference` checkout (the live pod imports it). TPU
  serialized to the main thread (agents set JAX_PLATFORMS=cpu before python). pkill by exact PID.
- Commit+push often; no force-push; owner submits upstream PRs. HF_TOKEN in `.env` (never commit).

## READ FIRST (in full)
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (the DCP saga) → docs/11-pod-runbook.md (pod recipes) →
docs/reviews/. Fork on `glm-5.2-v4-next`. Launch: `bash scripts/launch_glm_32chip.sh` — EXTRA_ENVS bakes
GLM_* into the raylet env (WORKERS need them too — a footgun).

## LANDMINES
GLM routes via **VllmModelWrapper** (vLLM path), NOT get_flax_model (a mistargeted fix = NO-OP). DCP fails on
**multi-chunk prefill** only (single-chunk always correct). Relaunch after ANY pod crash.
