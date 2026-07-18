# CLAUDE.md — glm-tpu (GLM-5.2 on TPU v4)

> Operating rules + context for porting **GLM-5.2** to **TPU v4**. Read this first every session.
> **START each new chat by reading, IN FULL:** `HANDOFF.md` (current state + exact next task), then this
> file (the rules), then `PLAN.md` (the phases), then `docs/00-feasibility-memo.md` (the config-verified
> feasibility study this whole plan is built on).
> **Update the Progress section (below) + `docs/RESEARCH_LOG.md` + `HANDOFF.md` as you go** — same discipline
> as the DeepSeek-V4-Flash port. Staleness is a bug.

This is the GLM-5.2 sibling of **`~/moe-tpu`** (the completed DeepSeek-V4-Flash-on-v4 port). **The DSV4 work
is the base, not a reference to admire from afar — you reuse its machinery.** Before building anything, read
the DSV4 material so you do not reinvent the wheel (§"What transfers", below, tells you exactly where to look).

---

## Operating mode — RUN AUTONOMOUSLY UNTIL TRULY BLOCKED (read first)

Long autonomous build. **Do not stop at phase boundaries or to ask for reassurance.** Pick the next step
(from `HANDOFF.md` → `PLAN.md`), implement it, validate it against a reference, **commit + push both repos**,
update the docs, continue. Self-correct; try multiple approaches when something fails.

**Only stop and surface to the owner when genuinely blocked:**
1. **A decision only the owner can make** — provisioning any new machine/VM/TPU (never allowed — §COST; only a
   same-region bucket + disk-attach to the existing 8 hosts), changing scope/target, anything outside the two
   repos / this VM / the 32-chip pod.
2. **Missing access/credentials** you cannot obtain (a gated download you lack a token for, a quota).
3. **An irreversible or outward-facing action** — pushing a PR to *upstream* `vllm-project/tpu-inference`,
   sending external comms, force-pushing, deleting/overwriting something you did not create. **Draft it, don't
   do it**; leave it for the owner.
4. **A hard technical wall after honest effort** — ≥3 genuinely different approaches tried + documented. (A
   normal bug is not a wall — keep debugging.)

Token/time cost is not a reason to stop.

---

## QUALITY BAR — FIX, don't PATCH; PR-grade; adversarially reviewed (read every session)

**No shortcuts. Fix the root cause properly, step by step, as HIGH-QUALITY, PR-compatible code — not reactive
patches.** This is the exact methodology that made the DSV4 port land:

- **Correctness before performance.** Validate every component **bit-faithfully against a reference** (the HF
  `transformers` GLM-5.2 forward and/or the SGLang/vLLM **GPU** implementation) at a tiny random-weight config
  first, then real dims, then end-to-end — *before* optimizing anything. Two controls (fp32 + bf16). A
  benchmark gap is a **bug to fix**, not "drift to report".
- **Every change: gated + additive + tested, in this order** — model-unchanged byte-identical when the gate is
  off; a CPU unit test; sub-cube (single-host 4-chip) zero-recompile validation; then pod-validated **3/3**
  runs (probabilistic multi-host halts mean one pass ≠ done — the DSV4 worker-3 race, `~/moe-tpu/docs/15`).
- **PR-compatible with `vllm-project/tpu-inference`** — the backend is maintained mostly by **Googlers**
  (jrplatin, kyuyeunk, gxd, QiliangCui). Code stays organized, clear, minimally-invasive, and follows their
  `AGENTS.md`: **no pure code-agent PRs** — a human understands and defends every line; **the owner submits**.
  Draft the PR series (mirror `~/moe-tpu/docs/13`); every PR body carries the AI-assisted-disclosure line.
- **Adversarial review is part of the loop.** After a change, spawn perspective-diverse reviewers (correctness
  / numerics / PR-cleanliness / overclaiming) to attack it against the reference before you trust it — exactly
  as the DSV4 port did. "Compiles ≠ works" — only *running* the parity catches decomposition bugs.
- **Observability-FIRST — build the instrument that makes the failure VISIBLE before you debug it** (the
  `docs/suggestions.md` lesson, now proven). When a bug reproduces only on metal, do not iterate blind: build
  the black box first (`docs/10`). The kit: the per-step **flight recorder**, the **DCP debug triad**
  (`GLM_DCP_CACHE_DUMP` + the two fail-loud guards `GLM_DCP_ASSERT_SHARDING` / `_CACHE_SANITY`), and on-metal
  discriminator probes. This pinned the fatal core-halt class and the DCP one-stripe-drop (cache-dump: exactly
  half the rows stale at dcp=2) — each **only after every CPU hypothesis was exhausted/falsified**. A
  metal-only bug is a signal to instrument, not to guess.
- **Enumerate before whacking moles.** For the DSA kernel + the serving-region programs, map the whole class
  first (the DSV4 `ExecutableCompileObserver` lesson, `~/moe-tpu/docs/15`) then fix systematically.

---

## COST DISCIPLINE — the bill is watched (READ; this shapes the whole plan)

The owner reviews GCP billing. Cloud Storage was the top line item (June 2026: **€134 Cloud Storage +
€47 NA↔EU data transfer**), driven almost entirely by **`gs://driftbench-storage` living in EUROPE-WEST4**
while the pod is in **us-central2**. Every storage decision here is made to minimize cost:

- **The pod is `db-v4-64-od` in `us-central2-b`.** Store GLM-5.2 weights **ONLY in a us-central2 bucket** —
  **`gs://driftbench-dsv4-uc` (US-CENTRAL2)**. **NEVER `gs://driftbench-storage` (EUROPE-WEST4)** — EU storage
  is dearer and streaming EU→us-central2 incurs cross-region egress. Same-region also gives the fast load
  (~12 GiB/s, ~3.5 min — the DSV4 number).
- **ONE copy of the model, kept FP8.** Use the **FP8-native** checkpoint (`zai-org/GLM-5.2-FP8`, ~744 GB) — do
  **not** make a bf16 conversion: 753B in bf16 is **~1.5 TB > the 1024 GB pod HBM** (>32 GiB/chip even sharded),
  so it will not fit. The FP8 weights must stay **resident in HBM** (~750 GB / ~23 GiB/chip fits); v4 has no FP8
  MXU, so dequant FP8→bf16 **per-tile INSIDE the Pallas MLA/GMM kernels** (memo §4d / PR #2324) — **NOT** the
  DSV4 load-time full-bf16 dequant (that OOMs at 753B; see the transfer table).
- **PREFER streaming — it is cost-optimal and DSV4 proved it works.** Stage **HF → a us-central2 GCS bucket
  ONCE** (streaming, zero local disk — the `scripts/stage_*_to_gcs.py` pattern in `~/moe-tpu/scripts`), then at
  serve time each host streams **GCS → HBM** in parallel via **`load_format="runai_streamer"`** with on-device
  FP8→bf16 dequant. DSV4 did this for a 295 GB checkpoint with no per-host local copy — try it first.
- **BUT a per-host local copy IS allowed if it is genuinely required.** If you determine (or discover it is
  mandatory) that each host needs the model on local disk — e.g., GLM's load path won't stream, or the streamer
  can't keep 8 hosts fed — you are **EXPLICITLY AUTHORIZED to create and attach a large (~1000 GB) persistent
  DISK to each of the 8 existing hosts** and copy the model onto it from the same-region bucket. **Say so
  explicitly, then do it** (it is storage attached to hosts we already have, not a new machine). Prefer the
  smallest disk that fits (the FP8 is ~744 GB) and delete the disks when done.
- **⛔ THE TWO HARD RULES (the only ones that matter):** (1) the bucket **MUST be same-region (us-central2)** as
  the pod; (2) **NEVER create or request any new compute MACHINE or any TPU** — you may use ONLY the existing
  32 v4 chips / 8 hosts. Attaching disks to those 8 hosts is fine; adding a host, a VM, or a TPU is NOT — the
  owner has nothing else and cannot spin up more.
- **ONE model copy** (FP8, no bf16 conversion — above). **Clean up as you go** — no redundant checkpoints, no
  stale XLA caches, no EU copies, no orphaned disks.

---

## Goal

Make **GLM-5.2** (FP8, `zai-org/GLM-5.2-FP8`) run **perfectly on TPU v4**, where "perfectly" is three things:

1. **Correct** — no shape/scale/dtype/routing bugs; the forward does what the GPU/HF reference does (dense-MLA
   first, then true DSA).
2. **Quality-faithful** — reproduce the **HF model-card benchmarks within noise**, with **full provenance**
   (§Benchmarks). TPU-vs-GPU bit-identity is impossible and is **not** the bar.
3. **Fast** — the **DSA lightning-indexer / top-k sparse-MLA kernel** is the headline. *No public JAX/Pallas DSA
   inference kernel exists on TPU* (verified against PR #2324, MaxText, and vLLM's DSA roadmap — see the memo).
   The DSV4 CSA/HCA compressed-attention Pallas kernel is the closest thing in the ecosystem and is the
   strategic asset to adapt. This is the net-new contribution the whole DSA family (DeepSeek-V3.2/V4, Kimi-K2.6,
   GLM-5.x) needs on TPU.

---

## The model — GLM-5.2 (config-verified; full detail in `docs/00-feasibility-memo.md`)

`architectures=[GlmMoeDsaForCausalLM]`, `model_type=glm_moe_dsa`, needs `transformers>=5.3.0`. Text-only.

- **753B total / ~40B active.** 78 layers; `first_k_dense_replace=3` (3 dense + 75 MoE). hidden 6144, MoE
  intermediate 2048, **256 routed + 1 shared experts, top-8**, `scoring_func=sigmoid`, `topk_method=noaux_tc`,
  `routed_scaling_factor=2.5`, `norm_topk_prob=true`. vocab 154880, `max_position_embeddings=1M`,
  `rope_theta=8e6`, `rope_interleave=true`, dtype bfloat16.
- **Attention (MLA):** `kv_lora_rank=512`, `q_lora_rank=2048`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`,
  `qk_head_dim=256`, `v_head_dim=256`, `head_dim=192`, `num_attention_heads=64`, `num_key_value_heads=64`.
- **DSA indexer:** `index_head_dim=128`, `index_n_heads=32`, **`index_topk=2048`**, `index_topk_freq=4`
  (**IndexShare** — one `full` indexer layer reused by the next three `shared` layers, per the `indexer_types`
  schedule), `index_skip_topk_offset=3`, **`indexer_rope_interleave=true`** (⚠ layout differs from
  DeepSeek-V3.2's non-interleaved indexer RoPE — verify carefully), `index_share_for_mtp_iteration=true`.
- **MTP:** `num_nextn_predict_layers=1` (extended to up to 5 draft tokens for speculative decoding).
- **Precision:** FP8-native checkpoint (also BF16, NVFP4). On v4 (no FP8 MXU) → dequant FP8→bf16 on-device.

**How GLM-5.2's DSA differs from DSV4-Flash's CSA** (this is the adaptation work, not a copy): `index_topk=2048`
(vs DSV4's 512), **32** indexer heads (vs 64), `index_head_dim=128`, **interleaved-RoPE** indexer layout,
**IndexShare** (4-layer indexer reuse incl. MTP), sigmoid noaux_tc top-8 gating, and the MLA head dims above.

---

## What transfers from the DSV4 port — and WHERE to look (do NOT reinvent)

The DSV4 port (`~/moe-tpu` + fork branch `dsv4-flash-v4`) already solved most of the systems problems. Read
these before writing GLM code:

| Need | Reuse verdict | Where (in `~/moe-tpu` unless noted) |
|---|---|---|
| vLLM/torchax arch registration | **Direct** — one-line registry add | `docs/05-existing-work-audit.md`; fork `models/vllm/vllm_model_wrapper.py`, `model_loader.py`. PR #2324 shows the exact `GlmMoeDsaForCausalLM` registry entry to port. |
| **CSA/HCA compressed-attention Pallas decode kernel** | **Modify — highest-value asset** | fork `kernels/mla/dsv4/kernel.py` + `custom_ops/deepseek_v4_attention.py`; `docs/09` (kernel plan), `docs/11` (mla.v2 map). Adapt to GLM DSA: topk=2048, 32 heads, interleaved-RoPE, IndexShare. |
| ATTN_HEAD head-sharding for HBM fit | **Direct→Modify** (re-verify GLM dims) | `docs/12-prc-dense-weight-sharding.md`; fork head-shard commits. 744 GB FP8 needs the same multi-host HBM mgmt. |
| FP8-on-v4 compute | **Modify — do NOT reuse the DSV4 load-time dequant** | DSV4's `REQUANTIZE_WEIGHT_DTYPE=bfloat16` dequants the whole model to **bf16 resident in HBM** — that fit only because DSV4 was 284B (bf16 909 GiB). 753B bf16 ≈ 1.5 TB **> 1024 GB HBM → OOM at load.** Keep FP8 **resident** and dequant FP8→bf16 **per-tile in the Pallas MLA/GMM kernels** (memo §4d, PR #2324; Mosaic rejects an FP8 RHS on v4). Watch the flaky dequant crash (DSV4 hit it too). |
| runai streaming load (GCS→HBM) | **Direct** | `~/moe-tpu/scripts/stage_base_to_gcs.py` (HF→GCS), `load_format="runai_streamer"`, the `F8_E8M0`/FP8 dtype-map fix. |
| per-rank KV-sizing + DPScheduler long-ctx fix | **Direct** | `docs/16` + the DPScheduler warning. GLM targets 1M context — KV capacity is the named bottleneck. |
| multi-host serving-race elimination | **Direct** (methodology) | `docs/15-worker3-multihost-race-blocker.md` + `ExecutableCompileObserver`. The model>1 hybrid mesh will re-expose it. |
| MoE (256 experts, top-8, noaux_tc) | **Direct** (GMM path) | fork `interface/moe.py`; note GLM is **sigmoid** gating + `routed_scaling_factor=2.5` (DSV4 was sqrtsoftplus ×1.5) — re-verify the scoring. |
| PR decomposition + CI-gate discipline | **Direct** (template) | `docs/13-pr-f-upstream-decomposition.md`, `docs/publication/PLAYBOOK_1`. |
| the whole methodology (read-first, adversarial reviewers, honest nulls) | **Direct** | `~/moe-tpu/CLAUDE.md` + `HANDOFF.md` — the operating discipline this file inherits. |

**Also relevant: tpu-inference PR #2324** (yiqiliu2, GLM-5.1-FP8 on v4-64, **DSA indexer DISABLED** via
`TPU_DISABLE_DSA_INDEXER`) — it de-risks *dense-MLA* GLM on v4 (port its registry + v4 MLA workarounds) but
runs DSA bypassed. It is unmerged/stalled (reviewer declined; missing `ready` CI label). Re-verify its state
before porting; coordinate with the maintainers so the DSA kernel is positioned as the missing piece.

---

## Phased plan (details + thresholds in `PLAN.md`)

- **Stage 1 (dense-MLA correctness) — days.** Register `GlmMoeDsaForCausalLM`; stage FP8 → us-central2 bucket;
  serve GLM-5.2-FP8 on the 32-chip v4 pod with **DSA bypassed (dense MLA)**; validate the forward vs the GPU/HF
  reference; reproduce a first benchmark (GPQA-Diamond / a generation-scored MMLU-family MC) within ~1–2 pts at
  ≤8K context. Threshold to proceed → Stage 2.
- **Stage 2 (the DSA kernel) — weeks, the critical path.** Adapt the DSV4 CSA compressed-decode kernel into a
  GLM DSA lightning-indexer (`index_n_heads=32`, `index_head_dim=128`, ReLU scoring, interleaved-RoPE) +
  top-k=2048 + sparse-MLA path, validated **selected-set-exact vs the reference** and bit-faithful to the XLA
  decode. Threshold → passkey ≥95% to ≥128K + long-context divergence from dense within tolerance.
- **Stage 3 (IndexShare + MTP) — throughput.** 4-layer indexer reuse (`index_topk_freq=4`, incl. MTP), then MTP
  speculative decode (up to 5 draft). Threshold → measurable FLOP/throughput gain at ≥256K + MTP acceptance ~5.

Keep every v4 gate (`tpu_generation()==4`) liftable to v6e/v7 (Ironwood = first native-FP8 TPU) so the DSA
kernel benefits the roadmap, not a v4-only dead end.

---

## Benchmarks + a provenance DB (proof of quality — non-negotiable)

To claim "it really works" we **reproduce the GLM-5.2 HF model-card benchmarks** and **prove every result is
traceable**. **Do NOT run anything that cannot be traced back to a stored, timestamped record.**

- **Get the benchmarks + test cases autonomously** (datasets from HF; the card's exact metrics/protocols). The
  card's set is agentic/reasoning-heavy — **prioritize the tractable ones first** (GPQA-Diamond, AIME 2026,
  HMMT, a generation-scored MMLU-family MC as the Stage-1 gate), then the heavy agentic ones (SWE-bench Pro, Terminal
  Bench, NL2Repo, Tool-Decathlon) which need full agentic harnesses. Card scores to match (verify at runtime):
  HLE 40.5 / AIME 2026 99.2 / GPQA-Diamond 91.2 / SWE-bench Pro 62.1 / Terminal Bench 2.1 ~81–82.7 / etc.
- **Provenance DB — `bench/results.db` (SQLite).** Store **literally everything** so any number is auditable:
  - `runs`: run_id, utc_timestamp, model (repo+revision/commit hash), harness git commit (glm-tpu), fork git
    commit (tpu-inference), env/mesh config (all `GLM_*`/vLLM flags, `attn_dp`, `num_gpu_blocks`, dtype), pod id.
  - `items`: run_id, benchmark, item_id, prompt/question (verbatim), gold answer, **model raw output
    (verbatim)**, extracted answer, correct (bool), score, n_prompt_tokens, n_gen_tokens, latency_ms, seed.
  - `summary`: run_id, benchmark, n, accuracy/metric, published_card_value, signed Δ, notes.
  - Never overwrite; append. A number with no `items` rows behind it is not a result.
- **Honesty:** signed Δ vs the card; instruct-vs-base and thinking-vs-non-thinking honestly labeled; report
  nulls first-class (the DSV4 discipline — HellaSwag −13.2 / triviaqa −6.3 were reported, not hidden).

---

## Two repos — read this first

- **`GianluigiVitale/tpu-inference`** (fork of `vllm-project/tpu-inference`): **all model/layer/kernel/quant code
  lives here**, on a **GLM working branch** (create off the DSV4 `dsv4-flash-v4` work — it carries the CSA
  kernel + head-shard + v4 enablement you are adapting). Editable-installed into `~/vllm-env`.
- **`GianluigiVitale/glm-tpu`** (this repo): the harness — configs, parity harnesses, bench + the provenance DB,
  scripts, docs. Set up via **`bash ~/setup.sh --folder=glm-tpu`** (clones + bucket restore + venv + 5-min
  sync cron; the `glm-tpu` case is already added to `~/setup.sh`).
- Commit + **push both repos often. No force-push.** `git push` is the only durable backup if the VM is
  preempted (the bucket mirror cron is `--delete` — do not rely on it as a source).

---

## TPU & framework rules (inherited from the DSV4 port)

- **The pod:** `db-v4-64-od`, `us-central2-b`, 8 hosts × 4 v4 chips = **32 chips / 1024 GB HBM**. This VM is
  **worker 0**. Launch across all hosts with `gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b
  --worker=all`.
- **Serialize TPU access** — never run concurrent TPU workflows on one host. Helpers/agents must not run TPU
  code directly.
- **Multi-host uses Ray** — run on `--worker=all`; `jax.distributed.initialize()` before device work; preserve
  the inline-import pattern in `model_loader.py` (avoids JAX-init failure under multi-host Ray).
- **`GLM_*` / `DSV4_*` env vars must be baked into the raylet env in the launcher** — vLLM's Ray executor
  carries only `VLLM_*` + a fixed allow-list to the workers (a DSV4 landmine).
- `OMP_NUM_THREADS=1` (a `copy_` OpenMP-after-fork segfault otherwise); `enable_dp_attention` additional_config
  (else `VllmConfig` rejects the MLA model); warm the XLA cache once, don't invalidate between runs.
- **No pipeline parallelism** — tensor/expert-parallel only. **After any pod crash, RELAUNCH** before re-running
  (leaked EngineCore / placement-group ~1800 s timeout).

---

## Research rigor / guardrails

- Correctness before performance; validate vs the GPU/HF reference (two controls). A benchmark gap is a bug.
- Honest nulls; no overclaiming; every number in the provenance DB. Past DSV4 sessions caught themselves
  overclaiming — keep that discipline.
- Commit + push both repos often; update `docs/RESEARCH_LOG.md` + this Progress section each session.
- **No external comms** — draft only; the owner sends. PRs to upstream `vllm-project` are **drafted; the owner
  submits**.
- **No new compute machines/VMs or TPUs** — only the existing 32 v4 chips / 8 hosts, plus (if needed) a
  same-region us-central2 bucket + disk-attach to those 8 hosts (§COST). No force-push.

---

## File structure — `glm-tpu` (this repo)

Model/kernel code lives in the fork (above). This repo holds everything around it:

```
glm-tpu/
├── CLAUDE.md                 # this file
├── HANDOFF.md                # current state + exact next task (read first)
├── PLAN.md                   # the phased plan + thresholds
├── README.md
├── configs/                  # glm52 config mirror + a tiny random-weight parity config
├── parity/                   # HF/GPU-reference vs TPU forward diffing (per-component)
├── bench/                    # benchmark harnesses + results.db (SQLite provenance)
├── scripts/                  # stage-to-GCS, launch_32chip, run scripts
└── docs/
    ├── 00-feasibility-memo.md   # the config-verified GO study — READ before coding
    └── RESEARCH_LOG.md          # dated entries, every session
```

---

## Progress

- [x] **2026-07-06 — Repo initialized** as the GLM-5.2 porting harness; `bench/` provenance DB + benchmark
  registry (GPQA-Diamond/MMLU-Pro/GSM8K/AIME-2026) + extractors/scorers pre-built and CPU-tested, datasets
  cached. FP8 target confirmed (`zai-org/GLM-5.2-FP8`, ~744 GB). Nothing ported yet at this point.
- [x] **2026-07-08 — Stage 1 DONE (dense-MLA) on the 32-chip pod.** 753B FP8 serves (runai GCS→HBM,
  FP8-resident + per-tile dequant, EP filter, pure TP-32). **GSM8K n=32 @ max-new 2048 = 96.9%** (31/32,
  1 truncation) — a **SMOKE test (n=32)**, NOT a scale claim (Wilson ~84–99%; needs n≥200 to claim scale),
  full provenance in `bench/results.db`. Fixed en route:
  determinism (canonical ascending-position gather), the mla.v2 `pack_new_kv` OOB core-halt (the fatal halt
  class — CLOSED, validated by the 8h52m zero-interrupt GPQA run), the large-bucket compile-OOM (F1).
- [x] **2026-07-08 — Stage 2 DSA kernels SILICON-VALIDATED** (single v4 chip): GATE 2a (Mosaic compile +
  documented w-tile fallback) + GATE 2b (**selected-set-EXACT vs the HF-math oracle**; sparse-MLA fp32
  9.5e-7 / bf16 1.95e-3). **Sparse passkey 100% in every (length,depth) cell @8K & 32K** (== dense).
  Sparse decode + sparse prefill + IndexShare all built (`GLM_DSA_MODE=pallas_decode`); D0 (sparse==dense)
  closed in the 753B engine.
- [x] **2026-07-08 — Stage 3 MTP code-complete (CPU).** M1 dense draft-parity vs composed-HF (max rel Δ
  3.1e-6, top-1 100%) + G4 IndexShare; dense-MTP engine knob `GLM_SPEC_K` (unset = byte-identical). M2
  (pod greedy-equivalence) prepped zero-turnaround.
- [x] **2026-07-09 (early) — SUPERSEDED by the 07-09/10 entries below** (the "packed-WRITE metal bug"
  turned out not to exist — the granularity root cause below explains everything). Kept for history:
  128K: persistence FIXED, packed-WRITE bug being bisected on metal. All 3 remaining
  blockers are in the KV/CACHE layer, NOT the DSA kernel (which PASSED).** (1) The stale-stripe **persistence**
  bug (one dcp stripe dropped across the scheduler-step boundary — cache-dump: exactly half the rows stale at
  dcp=2) is **FIXED** by pinning the **VllmModelWrapper step-fn + MTP `_propose` cache `out_sharding` to
  `P(BATCH,CONTEXT)`** under the gate — GLM's real vLLM path (an earlier flax-path fix was a no-op, caught by
  review; the A/B/C localizer then proved the carry is clean, `max|Δ|=0`). (2) The remaining DCP failure is
  localized to the **multi-chunk-prefill WRITE of the 2nd `kv_packing=32` tile** (logical page 2, positions
  1024–2047): the kernel and the owner-scatter arithmetic are CPU-exonerated (264 adversarial cases pass), so
  the bug lives on the metal-only write/new-KV path CPU cannot exercise — under observability-first on-metal
  bisection (cache-dump → A/B/C → correctness-diff → scatter-only → new-KV dump). (3) **fp8-KV SHELVED** — two
  independent v4 Mosaic `arith.cmpi` legalize blockers (read-dequant fixed branchless; a 2nd cmpi in the
  write/quantize path). 128K genuinely NEEDS DCP (shard the replicated cache) or fp8-KV (halve it) — **≥128K
  NOT yet demonstrated.**
- [x] **2026-07-08 — GPQA-Diamond n=198 (dense, DSA bypassed) = truncation-dominated.** Raw 52.5 is an
  artifact of the 4K gen cap (140/198 truncated mid-reasoning); **completed items 50/58 = 86.2%** (honest
  caveat: the completed subset skews toward easier/short-reasoning items). Rerun at 16K queued behind the
  DCP validation.
- [x] **2026-07-09 — 64K dense passkey VERIFIED 6/6 = 100% @ dcp=1/bf16** (run_id 100) after the 128K
  fit-check overturned the KICKOFF capacity premise: the dcp=1 wall is HBM **fragmentation** (compile
  scratch = 0.625 MiB × num_gpu_blocks vs a ~114–157M contiguous-free lottery), reliable bf16/dcp=1
  ceiling ≈ 88–90K. Best anti-frag lever: `LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5
  --xla_tpu_rwb_fusion=false"`. bf16 copy-elimination (GLM_MLA_ALIAS_KV L1+L2) = honest null, kept gated.
- [x] **2026-07-09 — fp8-KV kernel COMPILES on v4** — the pack_new_kv i8 `select_n` cmpi (`8a76ae5e3`),
  the i8 muli (`0f15e3ad0`) and the Q/output fp8 cast + q_scale over-scale (`65a6bb147`,
  `DISABLE_MLA_Q_ACTIVATION_QUANTIZATION`) all cleared — fp8-KV is no longer Mosaic-blocked. Ceiling
  @dcp=1 is ~80–96K (2.05G code overlays + 1.25G reserved eat the margin). **⚠ fp8 has NEVER generated a
  validated token on metal** — a cheap fp8 needle @dcp=1 is mandatory before any coupled fp8+DCP claim.
- [x] **2026-07-09 — ✅ THE DCP BUG: root cause FOUND + FIXED (`1f700c507`) — block-table granularity
  double-×dcp** (fork spec pre-multiply AND vLLM engine multiply → engine ids at 512·dcp² tokens vs TPU
  table at 512·dcp tokens/entry) — the multi-week "packed-write metal bug" NEVER EXISTED. Five
  independent confirmations incl. a live prediction; post-fix 900tok/4K/16K needles all correct (4K/16K
  were pred=None). En route: a CRITICAL process bug — **workers 1-7 ran stale 8802ebab all night**
  (per-host checkouts + bare `git push -q` not updating origin) → standing rule: push +
  `sync_workers.sh` + verify 8× same hash (+ `GLM_EXPECT_CODE_HASH`) before ANY pod test. Consequence:
  **bf16+DCP=4 fits 128K without fp8** (~29.2/30.75 GiB).
- [x] **2026-07-10 — Upstream verdict (#3129):** the granularity bug WAS upstream's (#2398), shipped
  broken in v0.20.0–v0.24.0, fixed on `main` by weiyu0824's PR #3129 **3 h before our fix** — treat
  `1f700c507` as convergent; memo `docs/upstream/dcp-block-granularity-report.md` (DO NOT file the
  corruption report). Still ours to upstream: the geometry assert, 2 residual granularity holdouts, and
  the real-CP owner-scatter+LSE attention (upstream dcp folds into head-TP — not true context
  parallelism).
- [~] **2026-07-10 — 128K smoked 7/7 (first 128K retrievals ever)** at 130,420 prompt tokens, bf16+DCP=4
  (runs 122/123, incl. mechanism depths 0.0/0.05/0.95/1.0); **THE DENSE GATE n=77 IS RUNNING** (run_id
  124, 7 depths × 11 trials, ~19 h, watchdog armed; 42/42 correct at 05:53 UTC). Owner rules adopted:
  small-n is NEVER a gate (≥95% needs n≥73 zero-failure), mechanism depths 0.0/0.05 + 0.95/1.0 required,
  **MTP FROZEN** until the headline gates close.
- [x] **2026-07-10 — Observability toolkit REPAIRED (`6f45e0944`+`165c76462`)** after a 25-agent audit
  found 2 BLOCKER + 6 MAJOR in the tools themselves (partial dumps could flip DIFFER→MATCH; bf16/fp8
  dumps crashed readers; fail-open guards; silent no-dump): diff refuses partial dumps, dtype-tagged
  dumps, `dcp_dump_check`, fail-loud guards, step-HLO honesty, per-worker code fingerprint +
  `GLM_EXPECT_CODE_HASH`, engine-vs-TPU granularity assert. Code audit of 8802ebab7..HEAD: no
  BLOCKER/MAJOR.
- [x] **2026-07-10 — Sparse-DCP stack (GLM_DSA_DCP) CODE-COMPLETE: Stages A+B+C** — distributed top-k
  (per-shard top-min(k,S_local) → all-gather → position-sort → top_k, elementwise-exact incl. tie order)
  + serving-path distributed sparse decode + distributed masked-prefill with LSE combine
  (`ed2a021be` / `4f390a61e`+`e7c239c91` / `6f8855c3f`). **CPU-CERTIFIED ONLY** (132/132 + 14/14 + 24/24;
  Stages B AND C adversarially reviewed SAFE-FOR-METAL-LADDER; gate-off jaxpr
  byte-identical to HEAD). Zero sparse-DCP tokens on metal yet — runbook §8 is the ladder.
- **Branch map:** `glm-5.2-v4` = pod mainline (Stage-1 + OOB fix); **`glm-5.2-v4-next` = integrated staging
  + what the pod runs** (tip `a98c77c9`: Stage-2 kernels + the granularity fix + fp8-KV v4 fixes + obs
  repairs + sparse-DCP A/B/C + scatter-flat + headsplit + segment S1 + W2.1 + gather-dominator + MTP-g4 +
  det, all gated off); upstream PR series `pr-g1..g6` drafted (G6 = the DSA kernels — the headline;
  cuts predate the efficiency campaign). Detail: `HANDOFF.md` + `docs/RESEARCH_LOG.md`.
- [x] **2026-07-11/13 — Sparse ladder rungs 1–6 CLOSED** (64K sparse 3/3 — first above 32K anywhere);
  128K mechanism smoke 4/4 EXACT twice; **efficiency campaign 12.6× end-to-end at 128K** (segment
  prefill + owned block-table width + gather-dominator v2s + chunk 2048; sparse prefill ~220 tok/s now
  beats dense); **gate2 (sparse 128K n=77) DIED at d=0.95 0/11** (fluent-filler signature) after 22/22 —
  engine-instance-lottery hypothesis CONFOUNDED by the w-4 disk-at-0 incident; the 14-probe fixed-seed
  discriminator is the next pod action. MTP still FROZEN.
- [x] **2026-07-16 — VM LOST + FULLY RECOVERED** (pod recreated, 8 disks wiped; nothing committed was
  lost — recovery audit in RESEARCH_LOG). Workers 1–7 re-provisioned 7/7 @ a98c77c9
  (`scripts/provision_worker_glm.sh`); results.db restored to the run-165 (gate2 0/11) checkpoint;
  setup.sh cron-abort bug fixed in the bucket. Lost forever: the safety/truth commit, write-probe guard,
  disk-watchdog (never committed — rebuild), d=1.0 discriminator db rows, XLA caches.
- [x] **2026-07-17 — Safety/ops-debt commit LANDED** (fork a98c77c9→845f4ffeb, 6-lens adversarial
  review, 2 BLOCKER + 8 MAJOR fixed pre-land): F6 headsplit refusals + docstring truth,
  GLM_WRITE_PROBE scatter sentinel (both widths), permanent combo suite (F8 closed), dense-impl
  refusal, in-repo ops kit (disk watchdog, dump archiver, orchestrators). The scrambled
  discriminator then PROVED the lottery (4/6 valid draws MISS) and byte-diff forensics localized it:
  per-HOST quiet-NaN poisoning of the layer-1 indexer k-cache; binaries exonerated by per-host
  fingerprints; caught AT INIT by GLM_PWAL_NAN_CHECK — **the LOADED weights arrive corrupt**.
- [x] **2026-07-18 — ROOT-CAUSE CANDIDATE + integrity layer.** Owner-enforced corpus-first re-read
  found it: the **t2j ALIAS RACE** (zero-copy numpy view of torch storage × `resize_(0)` eager free ×
  async H2D immutable-until-transfer staging) — unifies DSV4's never-root-caused "flaky dequant
  crash" (~60%/build there, ~56-60% of GLM inits here; crash = page unmapped, silent = page reused).
  Landed: GLM_LOAD_NAN_CHECK (full-weight on-device init scan + reject dumps; tip c68794241 synced
  8×). Refuted: PWAL copies, runtime clobber, scale-as-primary, streamer concurrency (A/B),
  F8_E8M0. Fix in flight (must-fail-first proof test); then validation draws (56%→0) → re-gate.
  **Standing owner rule: on any domain shift, RE-READ the core docs under the new lens BEFORE
  building instruments — the corpus is the first instrument.**

- [~] **2026-07-18 (later) — CHECKSUM ERA + THE GATE RUNS.** Adversarial reviews 3/3 on the integrity
  stack: t2j fix SAFE on the GLM path (sibling sites parked on ~/wt-sibling-alias); the stats review
  REFUTED "gate path open" (pooled-detector confound; NaN-only protections) ⇒ **GLM_LOAD_CHECKSUM**
  landed (a225d16b4): categorical cpu-vs-device byte-verify of every t2j-staged tensor, raises on
  mismatch, catches FINITE corruption; validation 4/4 clean. Xprof 128K: no ≥40% dominator (S2 not
  justified); per-chunk cost dominates prefill. **THE SPARSE 128K GATE LAUNCHED 11:05 UTC** — 33/33
  through d=0.0/0.05/0.95 (gate2's 0/11 death cell cleared 11/11 ⇒ gate2 died to the lottery, not the
  kernel). RESIDUAL SPECIMEN: one sick draw, byte-verified clean everywhere, fluent-filler miss ⇒
  residual is NOT H2D weight corruption (CPU-side-finite or engine-STATE; db run 193). Verdict ETA
  ~05-07 UTC 07-19; then 256K.

> Append dated entries each session. Keep `HANDOFF.md` in sync.
