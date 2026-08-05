# HANDOFF — GLM-5.2-FP8 on TPU v4

**Updated:** 2026-08-05 00:04 UTC. Read this file first, then `AGENTS.md`, `KICKOFF.md`,
`PLAN.md`, `docs/suggestions.md`, and the relevant recent entries in `docs/RESEARCH_LOG.md`.

## Project goal — do not narrow it

Finish `zai-org/GLM-5.2-FP8` on the existing 8-host/32-chip TPU-v4 pod so it is:

1. Correct: exact selected-set/tie-order behavior for DSA and protected state/cache integrity.
2. Quality-faithful: benchmark claims backed item-by-item by `bench/results.db`.
3. Fast: sparse DSA must beat dense at 256K. The immediate campaign gate is at least a 2x
   reduction from the original sparse decode step (<=200 ms/token, >=4.5 single-stream tok/s),
   with a >=3x stretch. The owner additionally wants the real hardware/software ceiling pursued;
   20–50 tok/s is a structural research target, not an achieved or guaranteed result.

No throughput-null endpoint. Fix measured stack bottlenecks. Every lever is default-off, exactness-
tested, single-variable metal-validated, traced, and smoke-gated before promotion.

## Hard operating boundaries

- Use only pod `db-v4-64-od`, zone `us-central2-b`: 8 hosts x 4 v4 chips = 32 chips/1 TiB
  distributed HBM. Never create a VM, host, machine, or TPU.
- Use only `gs://driftbench-dsv4-uc`; never use the Europe bucket.
- Serialize TPU work. Inspect all hosts before launch. Never edit a captured/running script.
- Raylet and driver must carry every `GLM_*` variable. Pin `GLM_EXPECT_CODE_HASH`; verify all eight
  serving checkouts clean and identical. Worker 0 is this VM.
- Preserve the load/state protection stack: OOB repair, whole-state NaN/checksum/hash manifest,
  write probes, exact results linkage, authenticated Ray ownership, post-stop census.
- Do not touch the owner's untracked `AGENTS.md`. No pipeline parallelism under the current project
  rules. The owner submits upstream PRs; no force-push or external communication.

## Authoritative achieved result

Original protected sparse baseline:

- Artifact `/home/gianl/glm-run/e0cap_sparse_20260803T083052607714673Z`
- 390.947743 ms/device token = **2.558 tok/s**

Current accepted corrected stack:

- Fork `979f818e020d3021b451d9ff30679a911b82f195`
- Exactness artifact `/home/gianl/glm-run/dcp_live_rows_exact_20260804T182105202837571Z`
  (DB 389 OFF / 390 ON): raw prefix `" 49"` identical; 129 events/4,081 rows per arm;
  selected set and tie order exact; remote archive verified.
- Health DB 391: protected 5K passkey predicted/gold `952687`, correct=True.
- E0 artifact `/home/gianl/glm-run/e0cap_sparse_20260804T205838387960340Z`, DB 392;
  remote `gs://driftbench-dsv4-uc/results/e0cap_sparse_20260804T205838387960340Z`.
- **287.666063 ms/device token = 3.476253 tok/s**; profiler-free wall median 3.3 tok/s.
- Improvement: -26.42% latency / +35.90% device tok/s versus the original baseline.
- Exact physical contract: named all-reduces 157; physical reductions 391; all-gathers 470.
- Selected-KV gather narrowed `[65536,640] -> [2048,640]`: 49.97 -> 1.77 ms/token.
  Sparse attend: 12.69 -> 0.48 ms/token.

Never quote `throughput_t1.json`'s profiler-contaminated aggregate as steady speed. The accepted
result is 3.476 device tok/s and about 3.3 wall tok/s. The prior 3.630 tok/s diagnostic is rejected:
its outside-shard slice added 78 reductions (391 -> 469).

The immediate >=4.5 tok/s gate and sparse-beats-dense-at-256K requirement are **not yet met**.

## Current trace diagnosis

Accepted E0 categories per token:

- collectives 161.35 ms
- sort/top-k 34.87 ms
- GMM 32.55 ms
- gather/scatter 17.98 ms
- compute 16.28 ms
- movement 15.68 ms
- sparse attention 0.48 ms

The dominant source is 75 MoE output combines at `live_rows_psum.py:85`, shape
`bf16[2,6144]`: **106.50 ms/token at only 0.02 GB/s**. This is launch/topology latency, not payload,
HBM capacity, useful math, or host idling. The current residual is repeatedly reconstructed and the
token executes 391 physical reductions across 78 sequential layers.

TPU v4 is not intrinsically limited to this speed. Published PaLM-540B reached 28.5 ms/token on 64
v4 chips using batch 64, 2K context, int8 weights, and a decode-specific 2D weight-stationary
layout. That is not comparable to this batch-1/32-chip/256K run. Official vLLM TPU currently marks
v4 experimental; its support matrix leaves multi-host TP/EP, CP/SP, MLA, and fused MoE unvalidated.
The current upstream GLM-5.2 performance sprint independently targets replacing MoE all-reduce with
reduce-scatter plus sequence parallelism. That corroborates this trace diagnosis and the all-gather/
feature-sharding direction, but it is not evidence that the local candidate is fast or exact.

## Sole active TPU workflow — do not launch another

Mandatory corrected-parent four-depth smoke:

- Run `/home/gianl/glm-run/lever_smoke128k_20260804T224754220401229Z`
- Launcher `/home/gianl/glm-run/corrected_parent_smoke_launcher_20260804T2247Z.log`
- Outer PID 3893688; driver 3896755; EngineCore 3896805
- Fork `979f818e0`; harness launch pin `52e0d00`; DCP4
- Gates: live-row psum=1, MoE fusion=1, DCP live attention=1, MoE all-gather=0
- Depths `0.0,0.05,0.95,1.0`, one trial each

As of 00:04 UTC: all eight hosts passed the 2,455-leaf state hash `371110325`, zero-nonfinite load
scan/checksum, and byte-exact dense-MLA/indexer-K/DSA-MLA donated-cache write probes. T32/T2048
compiled on all hosts. DB 393 is 3/4 correct: d=0.0 `705269`, d=0.05 `824794`, and d=0.95
`289958`, each predicted exactly and committed as an item row; d=1.0 is active on the same engine.
No traceback, OOM, compiler fatal, protection refusal, or disk alert. Preserve through 4/4,
immutable DB provenance, authenticated cleanup, `SUCCESS`, and allowed-bucket archive.

## Exact next sequence

1. Finish and archive the active corrected-parent 128K smoke. Do not promote on partial compute.
2. Correct the MoE all-gather branch onto parent `979f818e0`, commit/push a clean pin, then rerun
   focused CPU/Jaxpr/StableHLO tests.
3. Run protected same-pin OFF/ON exactness with only `GLM_MOE_DECODE_ALL_GATHER` varying.
4. If exact, run protected health and fresh 256K E0. Required HLO signature:
   reductions `391 -> 316`, all-gathers `470 -> 545`, named all-reduce remains 157. Accept only a
   real device+steady-wall gain with exact output/selection evidence.
5. If accepted, run the same four-depth 128K smoke before the next lever.
6. Validate MoE compute-row specialization (T32 -> T2, GMM m256 -> m16), then scorer-row narrowing
   and the lower-value DCP-LSE one-all-gather candidate, each through the same proof ladder.
7. For the higher ceiling, implement end-to-end 4x8 tensor/expert feature sharding: model-sharded
   residual and RMSNorm, subgroup attention projections, and expert-by-feature MoE. Then repair the
   sparse multi-token classification before enabling MTP/speculative decode. Never claim 20–50
   tok/s before protected local wall/device evidence.

## Prepared branches/worktrees

- Corrected accepted parent: `/home/gianl/tpu-inference-dcp-live-rows`, `979f818e0`, pushed/clean.
- MoE all-gather: `/home/gianl/tpu-inference-moe-live-allgather`, old pushed `5967dffa4` plus the
  corrected parent's staged `mla_attention.py`. Before metal: create a clean corrected-parent
  commit and rerun helper 8/8, fusion 7/7, env 17/17, exact-value stress, and StableHLO checks.
  Fresh CPU audit: the real 32-rank EXPERT topology is 100/100 exact for exactly representable
  values; StableHLO has one 32-way all-gather, one barrier, and the runtime-prefill psum branch.
  Generic bf16 values are not bitwise-identical because local reduction order differs from psum;
  do not overclaim CPU bitwise equivalence. Protected real-model token/selection exactness and the
  four-depth smoke are mandatory before acceptance.
- MoE compute rows: `/home/gianl/tpu-inference-moe-compute-live`, old `90431db22`; CPU 21/21.
  Transplant only after the all-gather lever is decided.
- Scorer live rows: `ebf12e8e4`; CPU DCP 32/32. Needs corrected-parent transplant later.
- DCP LSE all-gather: `422e9e31f`; CPU DCP 30/30 plus DCP2/4/8 stress. Lower priority.
- 2D f32 reduction prerequisite: `/home/gianl/tpu-inference-decode-2d-f32`, `dab2db7b3`, clean and
  pushed. It is only a numerical primitive; RMSNorm/attention/GMM end-to-end work remains.
- Harness repo main launch pin: `52e0d0094b937789026d7148bfac0cd65d525680`; expected status was
  only `?? AGENTS.md` before this HANDOFF update.

## Proof and observability rules

- `GLM_JAX_TRACE` plus `scripts/analysis/parse_xplane.py` is performance truth: require eight fresh
  XPlanes, 64 cores, exact selected decode windows, physical shapes/counts, source stacks, bytes,
  FLOPs, and fleet agreement.
- `GLM_LOG_STATS` after the trace is steady wall truth. Drop the mixed first window and compare to
  `1000/device_step_ms`.
- `GLM_DSA_DUMP_TOPK` plus `dsa_topk_diff.py` proves selected-set and tie-order identity. Pair any
  single-callback dump coverage with exact complete raw-token/DB equality.
- `GLM_DCP_ASSERT_SHARDING` and real `GLM_WRITE_PROBE` are decisive. The unchanged-row cache-sanity
  guard has a known replay/page-reuse false positive and is diagnostic only.
- Never accept profiler label changes alone. Count physical reductions and all-gathers inside the
  selected steps. Archive authenticated evidence before purging local traces.

## Already closed quality/correctness state

- Sparse 128K gate: 77/77, Wilson lower bound about 95.3%.
- 256K correctness: 4/4; engine-lottery/load-corruption path protected by OOB repair and manifest
  refusal. Sparse prefill beats dense after the prior 12.6x efficiency campaign.
- GSM8K 98.0%; AIME completed-correct 19/19; GPQA completed-item 83.9% with truncation caveat.
- MTP remains parked until the single-token sparse performance campaign and multi-token sparse
  classification are correct.

The attachment resume file is intentionally only a short pointer. Durable detailed state belongs
here and in `docs/RESEARCH_LOG.md`, not in an ever-growing chat attachment.
