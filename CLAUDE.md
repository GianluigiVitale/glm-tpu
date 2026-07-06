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
1. **A decision only the owner can make** — spending money beyond the explicitly-authorized staging VM (below),
   changing scope/target, anything outside the two repos / this VM / the 32-chip pod.
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
- **ONE copy of the model.** Use the **FP8-native** checkpoint (`zai-org/GLM-5.2-FP8`, ~744 GB) — do **not**
  make a bf16 conversion (that is ~1.5 TB and v4 dequants FP8→bf16 **on-device** at load anyway; the DSV4 530 GB
  bf16 bucket was pure waste — that lesson is why we keep only FP8).
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
| FP8→bf16 dequant-on-v4 load path | **Modify** | GLM is FP8-native (DSV4 instruct was FP4). Reuse the `REQUANTIZE_WEIGHT_DTYPE=bfloat16` on-device dequant; watch the flaky FP8-dequant-during-load crash (DSV4 hit it too). |
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
  reference; reproduce a first benchmark (GPQA-Diamond / an MMLU-family loglikelihood) within ~1–2 pts at
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
  HMMT, an MMLU-family loglikelihood as the Stage-1 gate), then the heavy agentic ones (SWE-bench Pro, Terminal
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
- **No paid cloud beyond the explicitly-authorized us-central2 staging VM** (and delete it after). No new TPUs.
  No force-push.

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

- [ ] 2026-07-06 — **Repo initialized as the GLM-5.2 porting starting point.** `glm-tpu` scaffolded (this
  CLAUDE.md + HANDOFF + PLAN + `KICKOFF.md` ≤4k /goal prompt + the feasibility memo in docs/00 + dir structure);
  `setup.sh --folder=glm-tpu` wired; created on GitHub (`GianluigiVitale/glm-tpu`, private). FP8 target confirmed
  (`zai-org/GLM-5.2-FP8`, ~744 GB). **The benchmark + provenance machinery is PRE-BUILT (CPU-tested):** `bench/`
  — the SQLite provenance DB (every question/timestamp/verbatim-reply/pass-fail + run provenance), the benchmark
  registry (GPQA-Diamond/MMLU-Pro/GSM8K/AIME-2026) + extractors/scorers, and a pluggable harness; datasets
  cached (MMLU-Pro/GSM8K/AIME-2026; GPQA gated → owner access-request). NOTHING ported yet — Stage 1 is the next
  chat's first task (see HANDOFF).

> Append dated entries each session. Keep `HANDOFF.md` in sync.
