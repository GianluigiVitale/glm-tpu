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

**2026-07-09 (on-pod, all workers synced): 128K correctness is an HBM wall, root-caused.** **64K passkey
VERIFIED 6/6=100% @dcp=1/bf16.** bf16 128K fails on a 161M per-layer cache transient (an XLA layout-decline,
copy-elim proven a DEAD END). fp8-KV halves the KV and its kernel now COMPILES on v4 (see below) but
dcp=1 still can't reach 128K (overlays+KV+weights > 30.75). Net: **128K needs cache SHARDING (DCP)**.

## ACTIVE FRONTIER — 128K NEEDS fp8-KV + DCP=2 (crystallized)
fp8-KV kernel now COMPILES on v4 (pack_new_kv i8 select_n cmpi + mask i8 muli fixed; parity 388/388) +
serves ~80–96K @dcp=1. But **128K@dcp=1 is HBM-STRUCTURAL**: overlays 2.05G (unrolled 78-layer code, no
flag) + KV 6.14G + weights 22.76 > 30.75; below 128K the 2.05G overlays fragments. bf16 copy-elim is a
DEAD END (donation binds; 161M is an XLA layout-decline). So:
1. **Fix the DCP multi-chunk packed-WRITE bug** (2nd kv_packing=32 tile mis-commit, owner-scatter
   `attention_interface.py:951-1078`; wired obs: `GLM_DCP_SCATTER_IMPL`=flat/barrier/onehot +
   `GLM_DCP_DUMP_NEWKV` DIFFER→MATCH falsifier). Then **fp8-KV + GLM_DCP=2** halves per-chip KV (6.14→3.07G,
   +2.5G margin) → 128K fits → verify needle ≥95%. 64K bf16 already 6/6; fp8 validated to ~80K.
2. Re-enable **DSA sparse** on the fp8+DCP 128K path for the **128K SPARSE** gate (kernel PASSED silicon @32K).
3. **256K throughput** A/B (dsa-sparse vs dense) — also via fp8+DCP.

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
**Workers have PER-HOST checkouts** — after ANY fork commit: `git push origin glm-5.2-v4-next` +
`TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash scripts/sync_workers.sh` (all 8 = SAME hash) BEFORE the pod test,
else stale worker code runs SILENTLY (bit me all night). GLM routes via **VllmModelWrapper**, NOT
get_flax_model. "PASS" in logs may be a grep hit on `hlo_passes.cc` — read real needle lines. Relaunch after
any pod crash. `setsid` the driver.
