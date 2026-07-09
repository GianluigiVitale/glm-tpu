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

**The 3 remaining blockers are ALL in the KV/CACHE layer — the DSA kernel PASSED.** (1) **DCP multi-chunk
packed-WRITE** corrupts the 2nd `kv_packing=32` tile on metal — kernel + scatter arithmetic CPU-exonerated;
only the metal write/new-KV path (CPU-blind) remains. (2) **fp8-KV**: TWO v4 Mosaic `arith.cmpi` legalize
blockers (read-dequant fixed; a 2nd cmpi in the write path) → **SHELVED** on v4 for now.
(3) **Stale-stripe persistence** — **FIXED** (VllmModelWrapper step-fn + MTP `_propose` out-sharding).

**Why the cache is the blocker:** MLA latent KV is **replicated per-chip** (P(BATCH), pure-TP) → one 128K
seq ≈ **11.9 GiB/chip** + 23 GiB/chip FP8 weights **> 30.75 budget** → 128K needs **DCP** or **fp8-KV**.
Passkey@128K (CORRECTNESS, batch=1) vs throughput@256K (batch/DCP/fp8) = **different gates**, decoupled.

## ACTIVE FRONTIER — do next, in order
1. **Empirically fit-check 128K passkey at batch=1 / bf16 / DCP-off** (the cache-replication question) — if
   it fits, correctness@128K decouples from DCP.
2. **Fix the DCP packed-write OR land the KV-capacity path** for the **128K SPARSE** gate; the on-pod
   scatter-only/cache-dump diff = the DIFFER→MATCH falsifier.
3. **256K throughput** A/B (dsa-sparse vs dense) via DCP/fp8 — the 2nd empty gate.

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
