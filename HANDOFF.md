# HANDOFF — GLM-5.2-FP8 on TPU v4

**Updated:** 2026-08-05 04:36 UTC. Read this file first, then `AGENTS.md`, `KICKOFF.md`,
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

Source audit confirms why the named axes are misleading here: the live mesh is
`data=1,attn_dp=1,attn_dp_expert=1,expert=1,model=4,dcp=8`, while both `MLP_TENSOR` and `EXPERT`
include `model × dcp`. Therefore these reductions are physically 32-way even though the source calls
them expert/tensor reductions. The higher-ceiling fix must separate feature/tensor width from the
eight-way context axis instead of merely renaming or shrinking the payload.

The on-device manifest accounts for about 721.2 GiB of state, including about 696.1 GiB of routed
expert weights/scales. Merely shrinking the tensor axes from `model × dcp` to `model` would replicate
roughly 25.1 GiB of non-routed state across the eight DCP ranks, adding about 5.5 GiB/chip and likely
exhausting the KV/overlay margin. The ceiling layout must genuinely tile weights in two dimensions
over the 4×8 mesh while separating feature/tensor work from context/expert communication.

TPU v4 is not intrinsically limited to this speed. Published PaLM-540B reached 28.5 ms/token on 64
v4 chips using batch 64, 2K context, int8 weights, and a decode-specific 2D weight-stationary
layout. That is not comparable to this batch-1/32-chip/256K run. Official vLLM TPU currently marks
v4 experimental; its support matrix leaves multi-host TP/EP, CP/SP, MLA, and fused MoE unvalidated.
The current upstream GLM-5.2 performance sprint independently targets replacing MoE all-reduce with
reduce-scatter plus sequence parallelism. That corroborates this trace diagnosis and the all-gather/
feature-sharding direction, but it is not evidence that the local candidate is fast or exact.

## Latest completed proof — pod released

Mandatory corrected-parent four-depth smoke completed and archived:

- Run `/home/gianl/glm-run/lever_smoke128k_20260804T224754220401229Z`
- Launcher `/home/gianl/glm-run/corrected_parent_smoke_launcher_20260804T2247Z.log`
- Fork `979f818e0`; harness launch pin `52e0d00`; DCP4
- Gates: live-row psum=1, MoE fusion=1, DCP live attention=1, MoE all-gather=0
- Depths `0.0,0.05,0.95,1.0`, one trial each

All eight hosts passed the 2,455-leaf state hash `371110325`, zero-nonfinite load scan/checksum,
and byte-exact dense-MLA/indexer-K/DSA-MLA donated-cache write probes. T32/T2048 compiled on all
hosts. DB 393 is 4/4 correct: d=0.0 `705269`, d=0.05 `824794`, d=0.95 `289958`, and d=1.0
`891482`; all four rows have 127,363 prompt tokens and 20 generated tokens. The DB snapshot passed
integrity check, authenticated cleanup ended with eight `CENSUS_OK` hosts, and local plus remote
`SUCCESS` are present at `gs://driftbench-dsv4-uc/results/lever_smoke128k_20260804T224754220401229Z`.
That proof closed with the pod at zero work before the exactness A/B below began.

## Latest exactness proof — pod released

Protected MoE decode all-gather exactness A/B:

- Run `/home/gianl/glm-run/moe_allgather_exact_20260805T002126894390635Z`
- Launcher `/home/gianl/glm-run/moe_allgather_exact_launcher_20260805T0021Z.log`
- Corrected pin `aa608543b`; identical harness launch pin `70f684a` in both arms
- Production-shape DCP8, 4,096-token prompt, two generated tokens; accepted live-row psum,
  MoE fusion, and DCP live-attention gates fixed on in both arms
- OFF completed as DB 394 with `GLM_MOE_DECODE_ALL_GATHER=0`: raw prefix `" 49"`, 129 selection
  dumps, final 2,455-leaf state manifest verified, real write probes passed, and authenticated
  cleanup ended with eight zero-work hosts.
- ON completed as DB 395 with `GLM_MOE_DECODE_ALL_GATHER=1`: raw prefix `" 49"` identical.
- The strict differ compared 129 aligned events over 4,081 live rows: zero diff events, tripwire
  rows, replication violations, or pad-row diffs. Selected set and tie order are elementwise exact.
- The OFF/ON step fingerprints are distinct, proving different executables ran. Both arms passed
  all state/write protections and eight-host authenticated cleanup. Local hashes verify, `SUCCESS`
  is present, and 547 objects/283.4 MiB are archived at
  `gs://driftbench-dsv4-uc/results/moe_allgather_exact_20260805T002126894390635Z`.

Exactness is accepted, but the performance trace below rejects this implementation.

## Latest health and recovered E0 — pod released; all-gather rejected

The protected health proof completed before E0:

- Run `/home/gianl/glm-run/resume_health_20260805T015811605234062Z`
- DB 396 predicted/gold `952687`, `correct=True`; all eight hosts passed the 2,455-leaf manifest,
  exact code/env census, real write probes, T32/T2048 compilation, and post-stop census.

The following E0 was interrupted by a disk alert during trace finalization, then recovered without
rerunning the model:

- Run `/home/gianl/glm-run/e0cap_sparse_20260805T024622713442900Z`; DB 397 has zero items/summary
  because the driver was stopped before output completion. This is diagnostic-only, not an accepted
  throughput run and has no profiler-free steady-wall result.
- All eight fresh XPlanes existed and were preserved at
  `gs://driftbench-dsv4-uc/results/e0cap_sparse_20260805T024622713442900Z/diagnostic_remote_xplanes/`.
  SHA-256 recovery, 8 files, 64 cores, and exactly 20 decode steps/core validate. The intended HLO
  signature is real: named all-reduce 157, physical reductions 316, all-gathers 545.
- Fleet device result is **290.762936 ms/token = 3.439228 tok/s**, versus the accepted parent's
  287.666063 ms = 3.476253 tok/s: **+1.08% latency / -1.07% rate**. Collectives regress
  161.35 -> 162.93 ms. The 75 replacement gathers alone cost 107.15 ms, versus 106.50 ms for the
  psums they replace. Each materializes `bf16[32,2,6144]` and still performs one sequential 32-way
  synchronization per MoE layer.
- Verdict: **MoE decode all-gather is rejected for performance.** It removes 75 reductions only by
  adding 75 equally expensive gathers; it does not remove synchronization depth. No fresh rerun or
  four-depth smoke is justified for a device regression.

Recovery cleanup is complete: the exact run-owned Ray cluster stopped, eight-host census is clean,
and worker 0 has 30+ GiB free. Three older local trace replicas (16.5 GiB) were purged only after
their exact GCS archives were verified. The current recovered traces and Ray session were purged
from all hosts only after the 5.71 GiB recovery archive completed.

The interruption exposed and fixed two harness defects in the working tree: watchdog descendants
now close flock FD 9 and cleanup kills the whole setsid group; disk free space is byte-exact and E0
requires 17 GiB preflight headroom while retaining the 15 GiB runtime floor. Static/syntax tests are
7/7 and the live disk check passes 8/8. Commit/push these harness changes with this handoff update.

No TPU workflow is active. Always rerun the strict eight-host census before launching another.

## Exact next sequence

1. On corrected branch `b3c25df47`, finish CPU/Jaxpr/HLO review of MoE compute-row specialization
   with all-gather fixed OFF. Expected pure-decode shapes are token rows `32 -> 2` and routed GMM
   rows `256 -> 16`, while the fallback/default-off programs remain unchanged.
2. Run protected same-pin OFF/ON exactness at production DCP8, varying only
   `GLM_MOE_DECODE_COMPUTE_LIVE_ROWS`; require exact raw tokens, selected set/tie order, state, and
   write probes.
3. If exact, run protected health and a fresh 256K E0. Accept only a real device plus profiler-free
   steady-wall gain. If it wins, run the four-depth 128K smoke before the next lever.
4. Then validate scorer-row narrowing and the lower-value DCP-LSE candidate through the same ladder.
5. For the higher ceiling, implement end-to-end 4x8 tensor/expert feature sharding: model-sharded
   residual and RMSNorm, subgroup attention projections, and expert-by-feature MoE. Then repair the
   sparse multi-token classification before enabling MTP/speculative decode. Never claim 20–50
   tok/s before protected local wall/device evidence.

## Prepared branches/worktrees

- Corrected accepted parent: `/home/gianl/tpu-inference-dcp-live-rows`, `979f818e0`, pushed/clean.
- MoE all-gather: corrected pin `aa608543b73921a330f48271d8263d4ec2ca14a4`, pushed and exact, but
  **performance-rejected** by the recovered fleet trace above. Keep the gate OFF for subsequent work.
- MoE compute rows: `/home/gianl/tpu-inference-moe-compute-live-corrected`, corrected pin
  `b3c25df47`, pushed atop `aa608543b`; all-gather remains default-off. Corrected-parent CPU tests
  pass 6/6 plus env 17/17. Protected HLO and metal exactness are next.
- Scorer live rows: `ebf12e8e4`; CPU DCP 32/32. Needs corrected-parent transplant later.
- DCP LSE all-gather: `422e9e31f`; CPU DCP 30/30 plus DCP2/4/8 stress. Lower priority.
- 2D f32 reduction prerequisite: `/home/gianl/tpu-inference-decode-2d-f32`, `dab2db7b3`, clean and
  pushed. It is only a numerical primitive; RMSNorm/attention/GMM end-to-end work remains.
- Harness repository durable `main` is `39c827c`; the recovery record, disk/watchdog fixes, and
  ownership tests are committed and pushed. The owner's untracked `AGENTS.md` remains untouched.

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

The attachment resume file is a stable pointer hard-capped below 4,000 characters. Never duplicate
volatile workflow/PID state there; current state belongs here and detailed evidence belongs in
`docs/RESEARCH_LOG.md`. This prevents stale attachment text from reviving an obsolete workflow after
context compaction.
