# GLM-5.2-FP8 on TPU v4: Low-Throughput Deep-Research Brief

**Status timestamp:** 2026-08-05 10:45 UTC

**Purpose:** self-contained technical brief for researching why single-stream GLM-5.2-FP8 decode is only about 3.3 tokens/s on a 32-chip TPU v4 pod, what has already been measured and tried, and what a credible fix must change.

**Research handoff rule:** treat the numerical claims and local artifacts below as the ground truth for this workload; use external results only to generate mechanisms or experiments, never as proof of local throughput.

**Local project:** `/home/gianl/glm-tpu`

**Inference fork:** `GianluigiVitale/tpu-inference`

This document is deliberately evidence-first. A hypothesis is labeled as such. A performance result is called accepted only when it has protected model correctness, state integrity, a fresh fleet-wide TPU trace, profiler-free steady-wall evidence, exact code/config provenance, and authenticated cleanup.

## 1. Executive summary

The system is not slow because the 753B model fails to fit, because TPU v4 is intrinsically incapable of serving a large model, or because sparse attention itself is expensive. The model fits across 32 distributed chips, but the current layout reconstructs activations through hundreds of sequential cross-chip collectives in every generated token.

The verified progression is:

| Stack | Device latency | Device rate | Profiler-free wall rate | Verdict |
|---|---:|---:|---:|---|
| Original protected sparse baseline | 390.947743 ms/token | 2.558 tok/s | about 2.5 tok/s | Baseline |
| Current accepted corrected stack | 287.666063 ms/token | 3.476253 tok/s | about 3.3 tok/s | Accepted |
| MoE compute-row candidate, repeat 2 | 280.646481 ms/token | 3.563202 tok/s | 3.394737 mean / 3.4 median | E0 valid; smoke pending |
| MoE all-gather replacement | 290.762936 ms/token | 3.439228 tok/s | not rerun after device regression | Rejected |

The accepted work improved device throughput by **35.90%**, but the result is still unusably slow and does not meet the project gate.

The current accepted trace spends **161.35 ms/token in collectives**. The largest single defect is **75 sequential MoE output combines**, each reducing only `bf16[2,6144]` across the physical 32-chip group. Together they cost **106.50 ms/token**, or about **1.42 ms per tiny reduction**, at an effective payload rate of only about **0.02 GB/s**. This is synchronization/launch/topology latency, not bandwidth-limited useful work.

The immediate target is at least:

- `<=200 ms/token` and `>=4.5` single-stream tok/s;
- at least a 2x reduction from the original sparse step, with a 3x stretch;
- sparse must beat dense at 256K;
- exact DSA selected-set and tie order, correct output, and protected state/cache integrity.

The accepted 287.666 ms step must lose at least **65.44 ms** to reach 4.5 tok/s (`222.22 ms/token`) and **87.67 ms** to reach the stricter 200 ms/token condition. Small payload reductions alone cannot reach the desired ceiling. The high-ceiling fix must remove or subgroup the repeated global synchronization and carry a compatible sharded residual layout through the full transformer.

A new MoE compute-row candidate has now passed semantic exactness, protected health, and two repeat 256K performance measurements. It narrows routed GMM rows from `256 -> 16` globally (`32 -> 2` routed rows per host-visible local formulation) during pure decode, but it does not remove the 75 global combines. The provenance-valid repeat reached **280.646481 ms/device token = 3.563202 device tok/s**, with **3.394737 tok/s clean wall mean** and 3.4 median. This is a real but small 2.44% latency / 2.11% wall improvement over the accepted stack. It passed the pre-registered E0 repeatability rule and therefore requires the four-depth smoke before promotion; it is not yet the accepted parent and does not materially change the structural diagnosis.

## 2. Exact workload and hardware being measured

### Model

`zai-org/GLM-5.2-FP8`:

- 753B total parameters, about 40B active parameters/token;
- 78 transformer layers;
- first 3 layers dense, remaining 75 layers MoE;
- hidden width 6144;
- 256 routed experts plus one shared expert;
- top-8 routed experts/token;
- MoE intermediate width 2048;
- sigmoid/noaux_tc routing, routed scaling factor 2.5;
- native FP8 checkpoint kept resident as FP8; TPU v4 dequantizes tiles for bf16 compute;
- DSA top-k is 2048, with 32 indexer heads and IndexShare over groups of four layers;
- maximum model context is 1M tokens.

The on-device protected state manifest accounts for about **721.2 GiB**, including about **696.1 GiB of routed expert weights and scales**.

### TPU pod

- Pod: `db-v4-64-od`, zone `us-central2-b`;
- 8 TPU VM hosts;
- 4 TPU v4 chips/host;
- 32 chips total, 64 TensorCores;
- 32 GiB HBM/chip, about 1 TiB aggregate distributed HBM;
- physical TPU topology: 2x4x4;
- approved storage: `gs://driftbench-dsv4-uc` in the same region.

The 1 TiB is **distributed memory**, not one coherent memory device. Each chip owns local weight/cache shards. Tensor/expert-parallel partial results cross the ICI interconnect. Adding chips increases capacity, but it does not guarantee lower batch-1 latency; a layout with a blocking collective in every sequential layer can become slower as the communication group grows.

### Performance workload

- single stream / one sequence;
- 256K context (`262144` prompt target);
- one generated token per decode step;
- DCP=8 for the KV cache/context dimension;
- TP world size=32;
- model mesh at the accepted run:
  `data=1, attn_dp=1, attn_dp_expert=1, expert=1, model=4, dcp=8`;
- runner compile token bucket is 32 even though only one request is live;
- the protected E0 measurement selects exactly 20 decode steps on every one of 64 cores across eight fresh XPlanes;
- separate post-trace logging provides profiler-free steady-wall decode speed.

This is not comparable to aggregate server throughput with tens or hundreds of concurrent requests.

## 3. Measurement and correctness methodology

The following tools exist because ordinary logs and profiler labels were insufficient or misleading:

1. **Fleet-wide device trace:** `GLM_JAX_TRACE` captures fresh XPlane data inside the real Ray workers. `scripts/analysis/parse_xplane.py` selects the exact decode window and attributes TPU operations by physical shape, source stack, bytes, FLOPs, and category.
2. **Physical collective count:** the parser counts HLO-category reductions and gathers in every selected core-step. This caught cases where a profiler name suggested fewer all-reduces while physical reductions had increased.
3. **Steady wall measurement:** `GLM_LOG_STATS` is sampled after tracing. The mixed first window is dropped. `throughput_t1.json` from a profiler-contaminated interval is never used as answer speed.
4. **Exact DSA semantics:** `GLM_DSA_DUMP_TOPK` plus `dsa_topk_diff.py` compares aligned events, selected sets, tie order, replication rules, and padded rows.
5. **State/load integrity:** a 2,455-leaf state manifest, whole-state non-finite scans, checkpoint-grounded out-of-band repair, checksums/hashes, and refusal-to-serve behavior protect against the previously discovered streaming-load corruption class.
6. **Cache/write integrity:** real donated-cache write probes execute the production scatter paths before a run is accepted.
7. **Code and environment integrity:** every worker records a code fingerprint; `GLM_EXPECT_CODE_HASH` refuses a stale worker. Every `GLM_*` variable must be present in both raylets and the driver.
8. **Ownership/cleanup:** Ray processes are positively tied to the run before cleanup; all eight hosts must finish with a zero-work census.
9. **Provenance:** benchmark rows and summaries are append-only in `bench/results.db`; protected artifacts are archived in the allowed bucket before local trace removal.

An optimization is promoted only through:

`default-off implementation -> CPU/Jaxpr/HLO evidence -> protected OFF/ON exactness -> health -> fresh 256K trace + steady wall -> four-depth 128K smoke`.

## 4. What the original protected baseline showed

Artifact:

`/home/gianl/glm-run/e0cap_sparse_20260803T083052607714673Z`

Fork pin: `94b746433`
Result: **390.947743 ms/token = 2.558 tok/s**

Major original costs per token:

| Category/operation | Approximate cost |
|---|---:|
| All collectives | 184.08 ms |
| All-reduce family | 131.47 ms over 232 named launches |
| Selected-KV gathers | about 66 ms |
| MoE GMM | 46.46 ms |
| Sort/top-k | 35.60 ms |
| Sparse attention kernel | 12.65 ms |

Two important conclusions followed:

1. Sparse attention was not the dominant cost. The system spent much more time selecting and materializing the top-2048 KV set than attending to it.
2. The decode executable carried a 32-row token bucket through operations even with one live request. This exposed a safe opportunity to narrow pure-decode work while preserving a full-shape fallback for prefill/mixed steps.

The original profiler showed 232 named all-reduces, but subsequent hardened analysis established that named labels are not a trustworthy physical launch count. The relevant accepted physical contract is described below.

## 5. Accepted improvements and their evidence

### 5.1 Pure-decode live-row hidden reductions

Candidate commit: `606f19ac8`
Gate: `GLM_DECODE_LIVE_ROWS_PSUM`

The candidate reduced pure-decode hidden reduction payloads from a 32-row bucket to the statically safe live prefix, then restored the dead suffix. Protected exactness produced identical raw output (`" 49"`) and exact DSA selected sets/tie order.

Device latency improved from 390.95 to **372.77 ms** (2.683 tok/s), but the standalone implementation was rejected because it broke a prior tuple fusion:

- named all-reduce count appeared to fall `232 -> 157`;
- physical HLO reductions actually increased `391 -> 466`;
- 75 fused pairs had become 75 extra routed launches.

This was a crucial observability finding: the payload optimization was useful, but the implementation increased synchronization depth.

### 5.2 Shared+routed MoE psum fusion

Candidate commit: `83915fe74`
Gate: `GLM_MOE_PSUM_FUSION`

The shared expert and routed expert local partials are packed into one reduction, then split before their historical scaling/addition points. Protected exactness covered 4,081 live selection rows with zero set, tie-order, tripwire, replication, or padding differences. This restored the lost 75-way tuple fusion and returned the physical reduction contract from 466 to 391.

### 5.3 DCP live attention rows inside the shard map

Accepted fork pin: `979f818e020d3021b451d9ff30679a911b82f195`
Gate: `GLM_DSA_DCP_DECODE_LIVE_ROWS`

This change narrows the expensive pure-decode DCP selected-KV gather/attention work to the live request prefix inside the shard map, while preserving full-shape owner scatter/cache writes and a runtime full fallback.

Measured effect:

- selected-KV gather shape: `[65536,640] -> [2048,640]`;
- dominant selected-KV gather: **49.97 -> 1.77 ms/token**;
- sparse attention: **12.69 -> 0.48 ms/token**.

Protected exactness:

- `/home/gianl/glm-run/dcp_live_rows_exact_20260804T182105202837571Z`;
- DB 389 OFF / 390 ON;
- raw prefix `" 49"` identical;
- 129 aligned events / 4,081 rows;
- exact selected set and tie order;
- state/cache protections passed.

Protected health:

- DB 391;
- 5K passkey predicted/gold `952687`, `correct=True`.

Accepted E0:

- `/home/gianl/glm-run/e0cap_sparse_20260804T205838387960340Z`;
- DB 392;
- **287.666063 ms/token = 3.476253 device tok/s**;
- about **3.3 steady-wall tok/s**;
- 26.42% lower latency / 35.90% higher device rate than the original baseline;
- physical contract: 157 named all-reduces, 391 physical reductions, 470 all-gathers.

The corrected stack then passed the mandatory four-depth 128K smoke:

- `/home/gianl/glm-run/lever_smoke128k_20260804T224754220401229Z`;
- DB 393;
- depths `0.0, 0.05, 0.95, 1.0`: 4/4 correct;
- all state/write protections passed;
- clean eight-host cleanup and durable archive.

### 5.4 Rejected outside-shard live-row formulation

A diagnostic formulation reached about **3.630 device tok/s**, but it inserted 78 additional reductions: physical reductions changed from 391 to 469. It was rejected despite the attractive aggregate number because the layout/launch regression was structurally unsafe and did not satisfy the protected contract.

## 6. Current accepted bottleneck breakdown

Source artifact:

`/home/gianl/glm-run/e0cap_sparse_20260804T205838387960340Z/analysis_t1.md`

Accepted device step: **287.666063 ms/token**.

| Category | Cost/token | Interpretation |
|---|---:|---|
| Collectives | **161.35 ms** | Dominant; repeated sequential synchronization |
| Sort/top-k | **34.87 ms** | Mostly 21 full IndexShare scorer/selection layers |
| GMM | **32.55 ms** | Routed MoE compute still uses padded routed rows |
| Gather/scatter | **17.98 ms** | Attention gather largely fixed; residual movement remains |
| Compute | **16.28 ms** | Other useful arithmetic |
| Movement | **15.68 ms** | Copies/reshards/layout work |
| Sparse attention | **0.48 ms** | No longer meaningful as a bottleneck |

The categories overlap only according to the parser's TPU event attribution rules; they are intended as an operation-level budget, not a host wall-clock summation model.

### 6.1 Dominant 75 MoE combines

- Source: `tpu_inference/layers/common/live_rows_psum.py:85` in the accepted pin;
- 75 invocations/token, one per routed MoE layer;
- physical payload per invocation: `bf16[2,6144]`;
- total: **106.495016 ms/token**;
- mean: about **1.420 ms/invocation**;
- effective payload throughput: about **0.02 GB/s**.

The payload is only about 24 KiB before collective protocol overhead. A 1.42 ms cost for such a small payload is direct evidence of a latency/synchronization floor. Reducing the payload further cannot remove the sequential dependency between transformer layers.

### 6.2 Sort/top-k details

The current trace attributes approximately:

- dominant DSA `top_k`: **23.03 ms/token**, 21 invocations, representative input `[32,33280]`;
- dominant DSA position sort: **9.24 ms/token**, 21 invocations, representative `[32,16384]`;
- total sort/top-k category: **34.87 ms/token**.

Only 21 layers run the full scorer because IndexShare reuses each full layer's selected indices in the following three shared layers. The full scorer still operates on the 32-row compile bucket in the accepted stack.

### 6.3 GMM details

The two routed GMM signatures total **32.55 ms/token** across the 75 MoE layers:

- first routed GMM / gated activation: about 22.02 ms;
- second routed GMM: about 10.53 ms;
- accepted trace uses routed row dimension `m=256` (32 token rows x top-8);
- the E0-valid compute-row candidate lowers both signatures to `m=16` for one live sequence, but total GMM time only falls to **31.67 ms/token**.

The measured 16x row reduction saves only about 0.88 ms of GMM time. Minimum tiles, weight reads, and launch overhead dominate these tiny decode GMMs, so row count and latency do not scale proportionally.

### 6.4 DCP LSE combine

The 78 attention-layer DCP LSE combines account for only about:

- `pmax`: 2.83 ms/token;
- tuple/all-reduce work: 2.32 ms/token;
- total: approximately **5.15 ms/token**.

A one-gather LSE candidate exists, but this trace corrected its priority downward. Even a perfect elimination cannot close the main gap.

## 7. Why the current mapping creates 32-way reductions

The live mesh is:

`data=1, attn_dp=1, attn_dp_expert=1, expert=1, model=4, dcp=8`.

In `ShardingAxisNameBase`, the relevant logical axes include both `model` and `dcp`:

- `MLP_TENSOR = ('attn_dp', 'attn_dp_expert', 'expert', 'model', 'dcp')`;
- `EXPERT = ('attn_dp', 'attn_dp_expert', 'expert', 'model', 'dcp')`.

All leading axes except model and dcp have size one, so both names physically mean `model(4) x dcp(8) = 32` ranks.

For GMM expert parallelism, current MoE weights are sharded only on the leading expert dimension with:

`P(ShardingAxisName.EXPERT)`.

Consequences:

- the 256 experts are distributed over all 32 chips;
- each chip owns eight complete experts (each host loads 32/256 experts across its four chips);
- each chip computes local expert contributions;
- the full hidden output is reconstructed with a global psum over the same physical 32-rank group;
- this occurs in each of 75 sequential MoE layers.

The source's logical word `EXPERT` obscures the fact that the collective is physically global across both the model and DCP axes.

## 8. Optimization attempts and verdicts

### Attempt A: live-row reduction payload

**Idea:** reduce only the live token prefix instead of all 32 padded rows.
**Result:** real 4.9% speedup, but initially split 75 fused tuple reductions.
**Verdict:** payload mechanism retained only after MoE psum fusion restored the physical launch contract.

### Attempt B: fuse shared+routed MoE reductions

**Idea:** pack two same-group partials into one psum.
**Result:** exact, removes the accidental 75-launch regression.
**Verdict:** accepted as part of the corrected stack.

### Attempt C: DCP live attention rows

**Idea:** keep full-shape cache writes, but gather and attend only the live request rows.
**Result:** selected-KV gather 49.97 -> 1.77 ms; sparse attention 12.69 -> 0.48 ms; full stack reaches 3.476 device tok/s.
**Verdict:** accepted and four-depth smoke-gated.

### Attempt D: replace MoE psum with all-gather

Candidate pin: `aa608543b73921a330f48271d8263d4ec2ca14a4`
Gate: `GLM_MOE_DECODE_ALL_GATHER`

**Idea:** because each expert contribution belongs to one rank, gather local outputs and select/sum locally instead of reducing.

Protected exactness passed:

- `/home/gianl/glm-run/moe_allgather_exact_20260805T002126894390635Z`;
- DB 394 OFF / 395 ON;
- identical raw `" 49"`;
- zero selected-set/tie-order differences over 129 events and 4,081 rows.

Protected health passed as DB 396.

Recovered E0 trace:

- `/home/gianl/glm-run/e0cap_sparse_20260805T024622713442900Z`;
- 8 fresh XPlanes, 64 cores, exactly 20 selected steps/core;
- physical reductions `391 -> 316`;
- all-gathers `470 -> 545`;
- replacement shape `bf16[32,2,6144]`;
- 75 gathers cost **107.145102 ms**, versus **106.495016 ms** for the psums;
- full device step regressed to **290.762936 ms / 3.439228 tok/s**.

**Why it failed:** it removed 75 reductions only by adding 75 equally sequential 32-way gathers, with a larger materialized result. It changed the collective type, not the synchronization depth.

**Verdict:** rejected for performance. No smoke rerun was justified after a protected device regression.

### Attempt E: pure-decode MoE compute rows (E0 valid; smoke pending)

Candidate pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`
Gate: `GLM_MOE_DECODE_COMPUTE_LIVE_ROWS`

**Idea:** specialize pure decode so routed GMM compute sees two token rows instead of 32 and 16 routed rows instead of 256; restore the dead output suffix. Keep all-gather off and retain full fallback for non-decode.

CPU/Jaxpr/StableHLO evidence:

- focused forced-four-CPU tests: 13/13;
- environment tests: 17/17;
- gate-off Jaxpr matches the legacy entry point after wrapper-name normalization;
- gate-on StableHLO contains both narrow and full branches plus padding restoration.

Protected production exactness:

- `/home/gianl/glm-run/moe_compute_rows_exact_20260805T044718436789636Z`;
- DB 398 OFF / 399 ON;
- identical raw prefix `" 49"`;
- 129 aligned events / 4,081 live rows;
- zero selected-set, tie-order, tripwire, replication, or pad-row differences;
- all eight hosts logged routed MoE rows `32 -> 2`;
- executable fingerprints and production HLO instruction counts changed, proving the candidate executed;
- state manifest and real write probes passed.

Protected health:

- `/home/gianl/glm-run/resume_health_20260805T062919033726354Z`;
- DB 400;
- passkey predicted/gold `952687`, `correct=True`;
- all eight state manifests and write probes passed;
- T32/T2048 compiled;
- authenticated cleanup passed.

Protected E0:

- `/home/gianl/glm-run/e0cap_sparse_20260805T071818146401337Z`;
- 256K, DCP8, one sequence, 256 measured tokens;
- trace 20 steps/core;
- retry 1: 281.777277 ms/device token = 3.548902 device tok/s; clean wall mean 3.401754 and median 3.4 tok/s;
- retry 1 is performance evidence but not promotable evidence because DB 401 recorded a live harness HEAD changed by a documentation commit instead of the captured launch pin;
- provenance-valid retry 2, DB 402: **280.646481 ms/device token = 3.563202 device tok/s**; clean wall mean **3.394737** and median 3.4 tok/s;
- retry 2 versus the accepted stack: 2.44% lower device latency, 2.50% higher device rate, and 2.11% higher clean wall mean;
- the two candidate device latencies differ by only 0.40%; both independently exceeded the pre-registered 1.5% device and wall thresholds;
- both traces contain 8 fresh XPlanes, 64 cores, exactly 20 steps/core, 78 DSA calls/step, 391 physical reductions, 470 all-gathers, and both routed GMM `m=16` signatures at 75 calls/step;
- retry 2 categories: collectives 158.68 ms, sort/top-k 34.60 ms, GMM 31.67 ms, gather/scatter 14.07 ms, compute 17.41 ms, movement 16.34 ms;
- the 75 MoE combines still cost 104.01 ms/token. GMM improved by only about 0.88 ms versus the accepted trace; much of the net 7.02 ms gain came from secondary gather/collective changes, which shows that smaller routed row counts do not make these tiny GMMs proportionally faster;
- exact DB linkage, state manifest, compile buckets, archive, and authenticated eight-host cleanup passed. The durable archive contains 83 objects / about 14.0 GB at `gs://driftbench-dsv4-uc/results/e0cap_sparse_20260805T071818146401337Z`.

This candidate attacks the 32.55 ms GMM category, not the 106.50 ms MoE collective category. Even eliminating all GMM time would leave about 255 ms/token, so it cannot independently reach the project gate. The protected result confirms that warning: it is worth retaining if the smoke passes, but it is not the throughput fix.

### Prepared but not yet performance-adjudicated candidates

1. **Scorer live rows**, original commit `ebf12e8e4`: narrow the row-independent DCP scorer and exact distributed top-k from 32 rows to the live prefix, then reconstruct exact sentinel padding. It targets much of the 34.87 ms sort/top-k category. It still requires transplant onto the current corrected parent, protected exactness, health, E0, and smoke if it wins.
2. **DCP LSE one-gather**, commit `422e9e31f`: replace pmax plus tuple psums with one gathered partial formulation. CPU coverage exists, but the accepted trace bounds the target at only about 5.15 ms/token, so it is lower priority.
3. **2D f32 reduction prerequisite**, commit `dab2db7b3`: preserves f32 partials needed by a future 2D reduction layout. It is a numerical primitive, not an end-to-end speedup.
4. **True 2D Tensor Parallel reference branch**, upstream/fork reference `fce8d6c41` (`patemotter/2dtp-merge`): demonstrates expert-by-feature sharding for the JAX DeepSeek path, including `edf`/`efd` reciprocal 2D specs. It does not directly support GLM's current vLLM/torchax GMM_EP path or the DCP8 cache layout; adaptation is substantial.

## 9. Why obvious axis changes are not valid fixes

### “Use only model=4 for tensor parallelism”

This would replicate non-routed model state across the eight DCP ranks. About 25.1 GiB of non-routed state is currently distributed over 32 chips. Replicating it over DCP adds roughly **5.5 GiB/chip**, likely exceeding the available KV/cache/overlay margin.

### “Use only dcp=8 for expert parallelism”

With no feature sharding, each chip would need roughly four times the routed expert state it holds today. The routed state is about 696.1 GiB, so an 8-way-only expert partition is far beyond 32 GiB/chip.

### “Replace all-reduce with all-gather”

Already tested. One blocking 32-way synchronization per MoE layer remained, and latency regressed.

### “The payload is tiny, so the collective should be free”

The trace proves the opposite. Tiny payloads expose fixed launch, rendezvous, topology, and sequential dependency costs. Bandwidth is not the limiting term.

### “All 1 TiB should behave like shared RAM”

TPU pod HBM is distributed. Every sharded matrix and activation has a placement contract. A global tensor value may require communication to reconstruct; capacity and coherence are separate problems.

## 10. Quantitative ceiling analysis of incremental fixes

Starting from 287.666 ms/token:

| Hypothetical perfect removal | Remaining latency | Implied rate | Meaning |
|---|---:|---:|---|
| Remove all 32.55 ms GMM | 255.12 ms | 3.92 tok/s | Compute-row alone cannot meet gate |
| Remove all 34.87 ms sort/top-k | 252.80 ms | 3.96 tok/s | Scorer alone cannot meet gate |
| Remove both GMM and sort/top-k | 220.25 ms | 4.54 tok/s | Barely reaches 4.5, still misses 200 ms |
| Remove both plus all 5.15 ms DCP LSE | 215.10 ms | 4.65 tok/s | Still misses 200 ms |
| Remove the 106.50 ms MoE combines | 181.17 ms | 5.52 tok/s | Structural target is large enough |

These are impossible upper bounds, not forecasts. They show why the campaign needs both immediate row/selection fixes and a structural collective-layout redesign.

## 11. The credible structural direction: genuine 4x8 expert-by-feature sharding

The central design requirement is to use both physical mesh dimensions for **weight capacity** while avoiding a full 32-rank reconstruction in every layer.

A credible design must include all of the following, not just a new `PartitionSpec`:

1. **Expert identity sharding** on one subgroup axis, so different ranks own different experts.
2. **Feature/tensor sharding** of each expert matrix on the other subgroup axis, so expert weights still fit without replication.
3. **A compatible activation/residual layout** carried between layers. If every layer immediately all-gathers a full residual, the design merely moves the same synchronization elsewhere.
4. **2D MoE GMM math**, including correct layouts for the up/gate and down matrices and FP8 block scales.
5. **Subgroup reduce-scatter/all-gather or collective-matmul**, so output reconstruction occurs in smaller groups or stays sharded.
6. **Attention and RMSNorm support** for the same residual layout. MoE-only 2D sharding cannot remain isolated if its output must be globally replicated before the next attention layer.
7. **DCP cache compatibility.** The current `dcp=8` axis also shards 256K KV context. Overloading it for feature or expert work changes cache and attention placement, so the layout must distinguish context ownership from MoE/tensor communication.
8. **HBM accounting** for weights, FP8 scales, KV cache, compiler overlays, and temporary collective/GMM buffers.
9. **Exact numerical contract**, especially if the reduction association changes. DSA selected sets/tie order and final tokens must survive protected metal A/B.

The reference `fce8d6c41` shows a useful mathematical pattern in the JAX DeepSeek path:

- activation width sharded on one model axis;
- MoE up weights `edf` sharded on expert/feature axes;
- MoE down weights `efd` use the reciprocal feature layout;
- unpermute derives local width instead of assuming full hidden width.

However, the production GLM path uses vLLM/torchax `GMM_EP`, where weights currently have only leading-expert sharding and `expert_parallel_gmm` globally psums the result. The reference branch therefore supplies a foundation and vocabulary, not a drop-in patch.

## 12. Research questions that should be answered next

### Collective floor and topology

1. What is the protected latency of a dependent chain of 75 `bf16[2,6144]` psums on this exact mesh for:
   - physical 32-way `model x dcp`;
   - 8-way `dcp` subgroups;
   - 4-way `model` subgroups?
2. Does the 1.42 ms/model-layer cost reproduce without model compute, or is it amplified by HBM/ICI contention from surrounding GMM/selection traffic?
3. What group-to-physical-topology mapping does `mesh_utils.create_device_mesh((1,1,1,1,4,8), ...)` choose on the 2x4x4 pod?
4. Are current small collectives using an unfavorable algorithm or cross-host path that can be altered with supported XLA collective options?
5. Can multiple logically adjacent collectives be fused without increasing physical reductions or changing exact reduction association?

A no-model multihost microbenchmark should enforce a genuinely dependent chain, use optimization barriers/nonlinear rank-dependent state so the compiler cannot fold it, inspect HLO to require exactly 75 collectives, and report distributions after warmup. It must run only after the model workflow releases the pod.

### 2D weight and activation layout

1. Which axis should own expert identity and which should own expert feature slices on the physical 4x8 mesh?
2. Can the MoE down projection produce a hidden-width shard that remains sharded through residual/RMSNorm/attention, avoiding a per-layer full all-gather?
3. What is the minimum communication schedule for exact top-8 EP routing plus two expert GEMMs under that layout?
4. Can `GMM_EP` be extended to accept 2D-sharded weights, or should a new backend compose GMM_TP within expert subgroups?
5. Can TPU collective-matmul V2 or all-gather-matmul/reduce-scatter-matmul kernels overlap communication with expert compute?
6. How must the runai streaming loader and EP host filter change when a host owns partial features for a wider expert set rather than complete experts?
7. How are FP8 block scales partitioned so dequantization remains local and exact?
8. What residual layout minimizes communication across the entire attention+MoE block, rather than optimizing MoE in isolation?

### Exact selection machinery

1. How much of the measured 23.03 ms top-k and 9.24 ms sort survives after scorer-row narrowing from 32 rows to the one live row?
2. If still material, can the DCP-local `lax.top_k` over `[live_rows,33280]` be replaced by an exact blocked or two-pass threshold selection while preserving lowest-position tie semantics?
3. Can IndexShare-dependent position sort/page lookup be hoisted once per 4-layer group without adding resharding?
4. Can the selected-KV gather remain fused into the Pallas attention kernel at all production geometries, now that live-row narrowing has already removed most of its cost?

### Compiler scheduling

1. Are `--xla_tpu_enable_while_loop_double_buffering` and async collective-fusion options already enabled by default in the current libtpu build?
2. Do they change physical HLO counts, memory pressure, selected-set bits, or only scheduling?
3. Can communication overlap exist in batch-1 decode when the next layer depends on the current collective result, or is only within-layer overlap feasible?
4. Can attention and FFN execute in parallel, as in published large-model layouts, to remove one reduction dependency per layer?

### MTP/speculative decode

MTP can improve effective answer throughput only after single-token DSA and multi-token sparse classification are correct. It does not make the base 287 ms token cheap. Research should separate:

- base decoder latency;
- draft latency;
- acceptance length;
- effective accepted tokens per target step;
- quality/correctness under DSA IndexShare reuse.

The 20–50 tok/s class may be more plausible as verified **effective** throughput with a strong MTP acceptance length than as current batch-1 base decode. It must never be claimed without local protected wall/device evidence.

## 13. Proposed experimental sequence

1. **Run the mandatory four-depth 128K compute-row smoke.** The repeatable 256K E0 cleared the pre-registered 1.5% device and wall thresholds. Promotion still requires all four mechanism depths, exact outputs, state/write protections, provenance, archive, and zero-work cleanup.
2. **If the smoke passes, promote compute rows as the new parent.** If it fails, keep the 287.666063 ms corrected stack as accepted and diagnose the correctness failure before any performance stacking.
3. **Transplant and protect scorer-row narrowing** on the resulting accepted parent. This is the largest remaining immediate non-structural category.
4. **Run the no-model subgroup collective microbenchmark** after the pod is clean. Use it to decide whether 8-way or 4-way subgrouping has enough latency leverage to justify the full 2D layout.
5. **Prototype 2D MoE weight specs and local math on CPU/Jaxpr/StableHLO**, including FP8 scales and loader/filter ownership. Do not stream the 753B model until shapes, local ownership, HLO collectives, and memory arithmetic are explicit.
6. **Build the smallest protected metal discriminator** that proves 2D expert×feature GMM and sharded residual behavior before an end-to-end 256K run.
7. **Extend the layout through RMSNorm/attention/residual**, then measure a full token. Promote only if physical synchronization depth and wall latency actually fall.
8. **Revisit exact top-k, compiler flags, and DCP LSE** based on the new trace budget, not their old projected value.
9. **Enable MTP only after base sparse decode and multi-token classification are correct.** Measure acceptance and effective answer speed separately.

## 14. External comparisons and why they do not prove local speed

### PaLM-540B on TPU v4

Pope et al., *Efficiently Scaling Transformer Inference*, report PaLM-540B at about 28.5 ms/decode step on 64 TPU v4 chips with int8 weights, batch 64, 2K context, and a decode-specific 2D weight-stationary layout. The paper supports three conclusions:

- TPU v4 is not intrinsically limited to 2–3 tok/s for large models;
- 2D partitioning becomes important at large chip counts;
- batch and carefully chosen layouts materially raise utilization and allow communication overlap.

It does **not** predict batch-1 GLM-5.2 at 256K on 32 chips.

Primary source: <https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf>

### Reported GLM-5.2 “~60 tok/s”

The disclosed vLLM benchmark used 128 concurrent prompts, 8,192 input tokens, and exactly one output token per prompt. Its 60 tok/s is aggregate output throughput. With one output token per request, steady TPOT/ITL is not measured. It is not a single-stream answer-speed comparison.

Primary source: <https://github.com/vllm-project/vllm/pull/46635>

### Current TPU software support

The vLLM TPU documentation describes v4 as experimental, and current support tables do not validate the decisive local combination of multi-host TP/EP, context/sequence parallelism, MLA, and fused MoE. The active GLM-5.2 optimization work independently targets MoE reduce-scatter plus sequence parallelism, which corroborates the local structural direction without proving a local gain.

Sources:

- <https://docs.cloud.google.com/tpu/docs/v4>
- <https://docs.vllm.ai/projects/tpu/en/stable/recommended_models_features/>
- <https://github.com/vllm-project/vllm/issues/46654>

## 15. Reproducible local evidence map

### Accepted baseline and result

- Original E0: `/home/gianl/glm-run/e0cap_sparse_20260803T083052607714673Z`
- Current accepted exactness: `/home/gianl/glm-run/dcp_live_rows_exact_20260804T182105202837571Z`
- Current accepted E0: `/home/gianl/glm-run/e0cap_sparse_20260804T205838387960340Z`
- Detailed accepted trace: `/home/gianl/glm-run/e0cap_sparse_20260804T205838387960340Z/analysis_t1.md`
- Corrected-parent smoke: `/home/gianl/glm-run/lever_smoke128k_20260804T224754220401229Z`

### Rejected all-gather

- Exactness: `/home/gianl/glm-run/moe_allgather_exact_20260805T002126894390635Z`
- Health: `/home/gianl/glm-run/resume_health_20260805T015811605234062Z`
- Recovered E0: `/home/gianl/glm-run/e0cap_sparse_20260805T024622713442900Z`
- Decisive local analysis: `ADJUDICATION.md` in that E0 directory

### E0-valid compute-row candidate

- Exactness: `/home/gianl/glm-run/moe_compute_rows_exact_20260805T044718436789636Z`
- Health: `/home/gianl/glm-run/resume_health_20260805T062919033726354Z`
- Repeatable protected E0: `/home/gianl/glm-run/e0cap_sparse_20260805T071818146401337Z`
- Valid trace summary: `/home/gianl/glm-run/e0cap_sparse_20260805T071818146401337Z/analysis_t2.md`
- Clean wall extraction: `/home/gianl/glm-run/e0cap_sparse_20260805T071818146401337Z/steady_decode_t2.json`
- Exact DB/env linkage: `/home/gianl/glm-run/e0cap_sparse_20260805T071818146401337Z/run_link_t2.json`
- Inference worktree: `/home/gianl/tpu-inference-moe-compute-live-corrected`
- Candidate pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`

### Analysis and provenance tools

- XPlane parser: `/home/gianl/glm-tpu/scripts/analysis/parse_xplane.py`
- Steady decode extractor: `/home/gianl/glm-tpu/scripts/analysis/extract_steady_decode.py`
- Provenance database: `/home/gianl/glm-tpu/bench/results.db`
- Detailed campaign journal: `/home/gianl/glm-tpu/docs/RESEARCH_LOG.md`
- Current durable handoff: `/home/gianl/glm-tpu/HANDOFF.md`

## 16. Definition of success

The throughput problem is not solved by a faster synthetic kernel, a profiler label change, aggregate multi-request throughput, or a candidate that is numerically close on CPU.

Success requires, on the existing 8-host/32-chip pod:

- `zai-org/GLM-5.2-FP8` serves correctly at 256K;
- exact DSA selected set and tie order under the protected comparison;
- correct raw tokens and quality-faithful benchmark behavior;
- state/load/cache protections pass;
- sparse beats dense at identical 256K geometry;
- device step `<=200 ms/token` and at least 4.5 single-stream tok/s, with profiler-free steady wall corroboration;
- a four-depth 128K smoke after every promoted lever;
- physical HLO counts and shapes prove the intended mechanism;
- all evidence is linked to `bench/results.db` and archived in the approved same-region bucket;
- any 20–50 tok/s claim is made only from protected local single-stream evidence, with base versus effective/MTP throughput clearly separated.

Until those conditions hold, the honest status is: **correct and substantially improved, with a small repeatable compute-row gain awaiting smoke, but still far too slow; the remaining dominant problem is repeated cross-chip synchronization caused by the current weight/activation layout.**
