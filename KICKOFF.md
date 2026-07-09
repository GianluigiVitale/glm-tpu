# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to **TPU v4** the DeepSeek-V4-Flash way.
**DON'T STOP/ASK** until (1) it serves correctly, (2) HF-card benchmarks match within noise, (3) the **DSA
sparse-MLA kernel** clears its gates (passkey ≥95% to ≥128K; throughput ≥256K). Self-correct; when unsure
pick + log. Only a hard block pauses THAT thread.

## STATE — what's DONE (don't redo)
- **Stage 1 DONE:** 753B FP8 serves on 32 v4 chips (runai stream, FP8-resident, EP filter, pure TP×EP).
  GSM8K **96.9% (31/32)** = **SMOKE (n=32)**, NOT "at scale" (needs n≥200).
- **Stage 2 kernels SILICON-VALIDATED (single v4 chip):** GATE 2a (Mosaic compile, 3 fixes + w-tile fallback)
  + 2b (**selected-set-EXACT** vs HF oracle; fp32 9.5e-7 / bf16 1.95e-3). **Sparse passkey 100% @8K AND 32K**
  (16× sparsification — the SELECTION works). Sparse decode + prefill + IndexShare built.
- **Stage 3 code-complete (CPU):** MTP M1 draft-parity + G4 index-share; dense-MTP knob GLM_SPEC_K.
- Fork **`glm-5.2-v4-next`** = all above + DCP + guards, gated off. PRs g1–g6 drafted.

**CORRECTED 2026-07-09 (on-pod): 128K batch=1 correctness = HBM FRAGMENTATION at compile, NOT a
cache-capacity wall.** 128K latent ≈ ~150 MiB/chip; a 128K prompt leaves >1 GiB free. Failure: a **~161 MB
transient (=ONE bf16 MLA KV layer, 0.625 MiB×num_blocks) allocated FRESH each step** finds no **contiguous**
slot (largest-free lottery ~114–157M). block_size INERT; gmu/scheduler can't reach 161M. **64K passkey
VERIFIED 6/6=100% @dcp=1/bf16**; ~90K = reliable bf16 ceiling.

**ROOT CAUSE (source agents): a donation-chain gap.** The mla.v2 cache write is meant in-place (L1 step-fn
donates kv_caches; L3 kernel donates cache_kv; pallas MUST-aliases) but the MIDDLE jit(shard_map)
(`attention_interface.py:1146`) didn't donate its cache arg → XLA copies 161M/step. **Fix `GLM_MLA_ALIAS_KV=1`**
(gated, byte-identical off, reviewed): donate L2 index 4 + pin L1 out-sharding → in-place → copy vanishes →
128K builds. **Backup fp8-KV** (161→80M): the "write cmpi" was MISATTRIBUTED (KV quantize is XLA/cmpi-free;
only Q-activation-quant's output cast is in-Mosaic fp8 → `DISABLE_MLA_Q_ACTIVATION_QUANTIZATION=1`), never
rebuilt since the branchless read fix → likely already works. DCP/fp8 = throughput/256K, not batch-1.

## ACTIVE FRONTIER — do next, in order
1. **Land 128K dense passkey @dcp=1** via `GLM_MLA_ALIAS_KV` (L2 donation): confirm the 161M `copy(` vanishes
   in the step HLO + it builds → verify needle 3/3. Backup: fp8-KV (Q-bf16). 64K already 6/6.
2. Re-enable **DSA sparse** on the working dcp=1 path for the **128K SPARSE** gate (kernel PASSED silicon @32K).
3. **256K throughput** A/B (dsa-sparse vs dense) — where DCP packed-write / fp8-KV are actually needed.

## HARD RULES
- **COST:** bucket **`gs://driftbench-dsv4-uc` (us-central2) ONLY**, never EU. **NEVER create a machine/VM/TPU**
  — only the 32 v4 chips / 8 hosts (disk-attach OK). ONE FP8 copy.
- **METHOD:** FIX root cause, never reward-hack. Every change gated (byte-identical off) + CPU test +
  **independent adversarial review** + pod 3/3. **Observability-FIRST** — it localized every metal bug. Honest nulls.
- Agents edit in WORKTREES (JAX_PLATFORMS=cpu); TPU serialized to the main thread. Commit+push often; no
  force-push; owner submits upstream PRs. HF_TOKEN in `.env` (never commit).

## READ FIRST
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md → docs/11-pod-runbook.md → docs/reviews/. Fork on
`glm-5.2-v4-next`. Launch: `bash scripts/launch_glm_32chip.sh` — EXTRA_ENVS bakes GLM_* into the raylet
env (WORKERS need them — a footgun). setsid the driver (it gets SIGTERM'd mid-serve otherwise).

## LANDMINES
GLM routes via **VllmModelWrapper** (vLLM path), NOT get_flax_model (mistargeted fix = NO-OP). "PASS" in
logs may be a grep hit on `hlo_passes.cc` — read real needle lines. Relaunch after ANY pod crash.
