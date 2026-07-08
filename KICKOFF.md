# GLM-5.2 → TPU v4 PORT — /goal (RESUME, ≤4k)

SOLO, FULLY AUTONOMOUS. Finish porting **GLM-5.2-FP8** to **TPU v4** the DeepSeek-V4-Flash way.
**DO NOT STOP / ASK / CHECK-IN** until BOTH hold: (1) GLM-5.2-FP8 serves correctly on the pod, (2)
reproduced HF-card benchmarks match within noise — AND the headline **DSA sparse-MLA kernel** clears
its gates (passkey ≥95% to ≥128K; FLOP/throughput win at ≥256K). Self-correct through blockers; when
in doubt pick the option you'd recommend, log it, never ask. Only a real hard block (missing
credential, owner-gated push) pauses THAT thread — keep every other track moving.

## STATE — what's DONE (don't redo)
- **Stage 1 DONE:** 753B FP8 serves on 32 v4 chips (runai stream, FP8-resident + per-tile dequant, EP
  filter, pure TP×EP). GSM8K n=32 **96.9%**. Fixed: determinism, the mla.v2 pack_new_kv OOB core-halt,
  the large-bucket compile-OOM (F1). All provenance in bench/results.db.
- **Stage 2 kernels SILICON-VALIDATED:** single-chip GATE 2a (Mosaic compile, w-tile fallback) + 2b
  (selected-set-EXACT vs HF oracle; sparse-MLA 9.5e-7/1.95e-3). Sparse **passkey 100% every depth
  @8K & 32K**. Sparse decode + prefill + IndexShare built (GLM_DSA_MODE=pallas_decode).
- **Stage 3 code-complete (CPU):** MTP M1 draft-parity + G4 index-share; dense-MTP knob GLM_SPEC_K.
- Integrated: fork **`glm-5.2-v4-next`** (all above + DCP + guards). Harness `~/glm-tpu`. PR series g1–g6 drafted.

## ACTIVE FRONTIER — do next, in order
1. **DCP → 128K:** the DCP cross-step KV-cache persistence fix (pin VllmModelWrapper step-fn + MTP
   _propose cache `out_sharding` to `P(BATCH,CONTEXT)` under the gate) is MERGED + independently
   reviewed. VALIDATE on metal: 2-chunk-prefill cache-dump must flip **DIFFER→MATCH**
   (`GLM_DCP_CACHE_DUMP`), guards silent (`GLM_DCP_ASSERT_SHARDING`/`_CACHE_SANITY`), THEN **dcp=2
   (dcp=4 if OOM) 128K passkey ≥95%/depth**. 128K NEEDS DCP (1 seq @dcp=1 = 12.2 GiB > free HBM).
2. **256K throughput** A/B (dsa-sparse vs dense), dcp≥4, `bench/dsa_throughput.py` — 2nd empty gate.
3. **MTP M2** on-pod: greedy spec-decode == non-spec (`bench/mtp_m2_check.py`, GLM_SPEC_K).
4. **Full GPQA-198 @16K** (owner-gated "go") + AIME card protocol; retry truncated tail via `--ids`.

## HARD RULES
- **COST:** bucket = **`gs://driftbench-dsv4-uc` (us-central2) ONLY**, NEVER EU `gs://driftbench-storage`.
  **NEVER create a new machine/VM/TPU** — only the 32 v4 chips / 8 hosts (disk-attach OK). ONE FP8 copy.
- **METHOD (DSV4's):** FIX root cause, never patch/reward-hack. Every change gated (byte-identical off)
  + additive + CPU test + **independent adversarial review** + pod 3/3. **Observability-FIRST** (flight
  recorder, cache-dump, on-metal discriminator probes) before debugging blind. Honest nulls, signed Δ,
  every number in results.db (raw output verbatim).
- **AGENTS work in git WORKTREES** — NEVER edit the main `~/tpu-inference` checkout (the live pod
  imports it). TPU serialized to the main thread (agents set JAX_PLATFORMS=cpu before python). pkill by
  exact PID, never a pattern that matches your own command.
- Commit+push both repos often; no force-push; owner submits upstream PRs. HF_TOKEN in ~/glm-tpu/.env (never commit).

## READ FIRST (in full)
`~/glm-tpu`: HANDOFF.md → docs/RESEARCH_LOG.md (dated narrative — incl. the full DCP saga) →
docs/11-pod-runbook.md (exact pod recipes: kernel gates, DCP debug triad, passkey/throughput/MTP) →
docs/reviews/ (rounds 1–9). Fork checked out on `glm-5.2-v4-next`.
Launch: `bash scripts/launch_glm_32chip.sh` — EXTRA_ENVS bakes GLM_* into the raylet env (WORKERS need
them, not just the driver — a repeated footgun).

## LANDMINES
GLM routes via **VllmModelWrapper** (the vLLM path), NOT get_flax_model. fp8-KV retired (v4 Mosaic
arith.cmpi compile bug). DCP failed on **multi-chunk prefill** only (single-chunk was always correct).
Relaunch the cluster after ANY pod crash before re-running.
