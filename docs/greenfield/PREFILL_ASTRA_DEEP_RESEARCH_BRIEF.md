# Research brief for Astra 6 Pro — fast GLM MoE prefill on TPU v4

Snapshot: 2026-09-08, after DB590; committed implementation baseline
`6e3b11704bd1d6d40f3b7620cf2ae88dcfbb4ebd`, branch `rewrite/topology-first-decode`.
Prepared for an independent web deep-research session. This document is self-contained:
local paths below are evidence pointers, not files the web researcher is assumed to access.
Research advice does not change execution authority, numerical contracts or acceptance targets.

## Copyable assignment

Investigate how to turn our working but slow native-JAX GLM-5.2-FP8 implementation into an
efficient single-request, long-context inference engine on an EXISTING 32-chip TPU v4 slice.
Prioritize the fastest practical path to good prefill and time to first token, preserving
correctness and our working checkpoint. Read this entire brief before recommending changes.

Find and inspect official implementations, papers, actual benchmark commands/results,
maintainer discussions and relevant compiler issues. Do not produce only a literature survey.
Give an implementable design, a ranked experiment sequence, and explicit reasons to reject
attractive but incompatible approaches. Challenge our design and verification workflow as
well as our kernels. Do not assume our current B128 window is the right final architecture.

The owner's motivating question is: other MoE implementations reportedly achieve roughly
10,000 input tokens/s; what specifically do they do, and which techniques can work HERE?
That number is a research lead, NOT a verified comparable result or an adopted promise.
Do not dismiss it merely because another model/hardware differs; quantify the differences.
Equally, do not pass off output throughput, training throughput, prefix-cache hits or aggregate
multi-request throughput as single-prompt prefill. If no comparable result exists, say so.

Return a decision-oriented report answering the questions below, not requests for another
unbounded research round. For local facts unavailable to you, identify the smallest exact
measurement we should collect; do not invent traces, source behavior or benchmark results.

## TL;DR: what is solved and what is actually blocking completion

- Gate D IS CLOSED for the historical short-context WS32 decoder, under the reviewed numerical
  contract. Gate G chose WS32. We are not still trying to make the original decoder work.
- The serious mistake was treating memory-chunked SERIAL teacher forcing as adequate production
  prefill. A measured 127,363-token prompt took 16,425.515 seconds, about 4.56 hours. This was
  device work, not upload time. Repeating this at long context is now prohibited.
- Genuine token-batched, layer-major prefill now works through all 78 layers at 2K. Its small
  B17/B11 correctness configuration took 102.203 seconds for 2,034 prompt tokens, about
  19.9 input tokens/s by that request-prefill accounting. It is not a fast final engine.
- One real supplied-route MoE baseline improved 3.329x when processing the SAME 128 rows as
  one B128 call instead of eight B16 calls. Concentrated routing improved only 1.280x.
  These are scoped phase measurements, not full-model speedups or natural routing statistics.
- A B128 complete-layer window is implemented/default-off, with four <=32-row attention/DSA
  tiles followed by a larger grouped MLP. Its actual TPU graphs have been acquired/admitted
  locally, but its complete-layer numerical and timing experiment has NOT run.
- The next engineering work is that bounded numerical/timing experiment, then larger-window
  performance decisions, the changed path's own competitive 8K proof, and efficient 128K/256K
  proofs. A numerical worker is currently uncommitted/tested WIP, not deployed evidence.
- Missing: representative prefill phase attribution, natural expert occupancy, defensible
  final prefill/TTFT targets, long-capacity memory for the new graph, and delivered TTFT.
  Research should shorten this path, not restart solved exactness investigations.

## 1. Fixed hardware, model and operating constraints

| Item | Local project configuration |
|---|---|
| Hardware | 8 hosts x 4 TPU v4 chips = 32 chips; physical 2x4x4 topology |
| Device memory | Nominal 32 GiB/chip; observed allocator limit 33,014,398,976 bytes/chip |
| Model | `zai-org/GLM-5.2-FP8`; local checkpoint/config is authoritative |
| Parameters/layers | 753B total, approximately 40B active/token; 78 layers, first 3 dense, remaining 75 MoE |
| MoE | Hidden 6144; 256 routed experts; top-8; one shared expert; routed intermediate 2048 |
| Dense MLP | Intermediate 12288 |
| Attention | 64 heads; q rank 2048; KV rank 512; q head 256 = nonrotary 192 + rotary 64; value head 256 |
| DSA | 32 index heads x 128; top-2048 positions; 21 full indexers; IndexShare groups of four |
| Active layout | WS32_2D: expert8 x feature4; residual remains sharded; repeated collectives confined to those subgroups |
| Runtime | Native JAX 0.10.1 / libtpu 0.0.41 in current protected runs; Python 3.12 |
| Workload | ONE request; many known prompt positions processed together; batch-one autoregressive decode |
| Long workloads | Four passkey depths at 127,363 prompt tokens; full E0 prompt 262,144 tokens, allocated capacity 262,656 |

Full-indexer producer layers are 0,1,2,6,10,...,74. Layer3 is an IndexShare/MoE layer;
layer6 is the first full-indexer PLUS MoE combination, hence our next discriminator.

Do not confuse TPU v4 with DeepSeek V4, or chip count with TensorCore count. DeepSeek was an
earlier project lesson, not permission to replace GLM architecture or import DeepSeek-specific
semantics. Mechanisms from other models are useful only with an explicit compatibility argument.

Infrastructure is fixed: NEVER stop/delete/resize/recreate/manage the TPU, VM, node or queued
resource. No new compute, additional accelerator pool or disaggregated second pod. The only
bucket is `gs://driftbench-dsv4-uc` in US-CENTRAL2. Live storage must stay below
2,500,000,000,000 bytes; latest recorded census was approximately 2.002 TB, not a new live audit.
Soft delete is off. No full-size backup/repack/BF16 checkpoint copy. Before any >100 GB artifact,
justify necessity, temporary peak, retained bytes and the artifact it replaces.

The retained final WS32 pack is already available under
`/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z`.
Use its existing shards/scales. Do not recommend a new model download as the first experiment.
Legacy `tpu-inference` execution cannot be imported into this engine; isolated upstream kernel
ideas/code may be evaluated with compatibility and licensing review.

## 2. Evidence ledger — do not conflate these measurement scopes

| Evidence | Established result | Does NOT establish |
|---|---|---|
| DB567, historical WS32 8K | 20/20 compared raw tokens; reviewed DSA adjudication; decode p50 130.369 ms, 7.67 tok/s; peak 26.376 GB/chip; protected trace/archive/cleanup | Efficient prefill, full per-layer tensor comparison of that entire run, 256K completion |
| DB573–575 | Three serial-prefill 128K passkey depths sealed; DB574 prefill 16,425.515 s, decode 143.496 ms | New batched-path long-context correctness; all four depths; acceptable TTFT |
| DB572 | An 8K prompt at capacity 262,656; peak 29,655,086,080 B/chip, headroom 3,359,312,896 B | Full 256K prefill, new batched scratch/headroom |
| DB576–579 | Raw FP8 projection K1536/N2048, synthetic rows 8/32/128/256; p50 0.232–0.248 ms/batch | Full MoE reuse, causal layer throughput, general bitwise equivalence |
| DB583/584/587 | Supplied-route real MoE and real layer0/layer3 bounded numerical admissions; selected real weights, synthetic state and captured references | Full long-context throughput or natural workload routing |
| DB588 | Complete batched 2K B17/B11; 20/20 tokens, 14x21 DSA observations, state/cache; request-prefill 102.203 s; decode 129.171 ms; peak 26.397 GB/chip | B128 model result, competitive 8K cutoff proof, prefill-specific trace attribution, delivered TTFT |
| DB589 | Equal128 supplied-route real-MoE phase baseline, detailed below; numerical checks and 8-host cleanup | Full-layer/model speedup, sustained throughput, XPlane attribution or actual model router distribution |
| DB590 | All four actual layer6 graph pairs/allocations captured on 8 hosts; 118 s worker/collector; zero model/WK executable calls | Numerical execution, runtime scratch peak, full-layer speedup or permission to claim completion |

DB589 used 10 warmups and 50 samples. Each sample sums completed dispatch-to-completion
intervals; fleet health votes between calls are excluded. This is explicitly not request wall.

| Supplied routing | Eight B16 calls p50/p99, ms per 128 rows | One B128 call p50/p99 | p50 ratio |
|---|---:|---:|---:|
| Distributed | 40.046 / 57.523 | 12.028 / 12.158 | 3.329x |
| All routes on one expert owner | 37.474 / 52.846 | 29.283 / 40.809 | 1.280x |

Distributed active grouped tiles per projection fell 868 -> 345; useful lane fraction 0.371.
Concentrated tiles stayed 128 -> 128, all 1,024 routes on one owner, useful lane fraction 1.0.
Thus perfect tile occupancy does not imply balanced devices or best latency.

DB590 B128 vs B32 complete layer6 compiled allocations, bytes per chip:

| Graph | Arguments | Outputs | Temporary | Code | Aliases | Static collectives/call |
|---|---:|---:|---:|---:|---:|---:|
| B128 window | 331224064 | 4204032 | 203686912 | 49431552 | 0 | 28 |
| B32 control | 330626048 | 1746432 | 96066560 | 17125888 | 0 | 13 |

There are also separate BF16 WK decode and FP32 WK promotion executables. All four resident
code allocations total 67,150,848 bytes. Selected layer6 weights are 35 leaves,
326,079,840 bytes/chip. Compile residency is not numerical memory: each dispatch must budget
all live arrays, retained control/candidate outputs, scratch, code and a fixed 1 GiB reserve.
Lower collective counts alone do not prove lower latency or unchanged payload cost.

## 3. Current implementation and non-negotiable semantics

The old reference processes each prompt token through every layer, then advances the token.
The new path embeds a block of KNOWN tokens and processes those rows layer-major. Only the
last live prompt row needs the final logits/head. This is not batch-32 padded decode.

The current larger-window design runs four <=32-row causal attention/indexer prefixes inside
each layer, carrying that layer's KV and both index-cache states, then runs one B128 MLP/router
suffix on the concatenated real rows. These are device-program operations, not Python stage
dispatches. An outer prompt-block loop remains. The B128 flag defaults off.

The <=32 limit is an implementation/admission boundary, NOT a claimed hardware optimum.
The existing M64 index-repair primitive cannot simply accept an arbitrarily larger row shape.
Larger MLP windows can be separated from bounded attention/repair tiles, but unrolling many
tiles can increase executable size, memory and compilation cost.

Routed projections consume raw FP8 bytes/scales; grouped work sorts token/route pairs by
expert, schedules active tiles and restores original route-slot order before combination.
The existing row-tile quantum is eight; inspect actual group scheduling and VMEM use rather
than assuming a standard large dense GEMM. Full BF16 expert-table expansion is forbidden.

Exact grouped-kernel interface for upstream compatibility research: E4M3FN weight values
stored as U8 bytes, with FP32 scales per128x128 block. Per chip, each of gate/up has weight
shape `[32,2048,1536]` (expert,output,input) and scales `[32,16,12]`; down has weights
`[32,1536,2048]` and scales `[32,12,16]`. Thus gate/up shard the contraction dimension,
whereas down shards the output dimension. Activations are BF16. Gate/up partial outputs
and their feature4 reduction are FP32, then rounded to BF16 before SwiGLU. Down output and
route-weight products are BF16; the admitted path restores route-slot order, accumulates
those weighted values in FP32 and performs the expert8 FP32 combine before BF16 output.
These are not per-channel/MX quantization or an unspecified native-FP8 GEMM interface.

Crucial fusion constraint: in our feature-sharded layout, up/gate projections produce partial
results that must be combined over feature4 BEFORE SwiGLU. Applying activation independently
to feature partials and then summing is mathematically wrong. A generic fused EP GMM1+activation
kernel is not automatically a valid replacement. Fusing loads/projections while retaining this
boundary is a candidate; prove any different layout/communication plan explicitly.

Numerical/cache contracts:

- Exact ordered top-k/ties against the executing engine's own scores; selected state must be
  the state IndexShare/attention consume. Router route identities/order are exact in the
  bounded candidate/control protocol. No token dropping or expert-capacity overflow.
- Internal tensors use existing fixed bounded contracts; exact raw continuation tokens are
  required at short context. We do NOT demand universal legacy bitwise intermediate identity.
- Cross-engine DSA differences near cutoff need the reviewed independent-reference
  adjudication, not a post-failure tolerance increase. A new first-divergent event needs its
  own evidence. At 2K most histories do not exceed top-2048: this cannot replace competitive 8K.
- Prompt attention reads UNREPAIRED index keys throughout the entire prefill. M64-repaired
  keys are written to separate storage and become visible only when prefill finishes.
- Each layer owns its own KV. IndexShare can reuse selected positions, NOT another layer's KV.
- Exact absolute positions, causal visibility, reordered page tables, tails/padding, untouched
  cache bytes and atomic all-owner failure/rollback. Invalid padded rows must not poison live rows.
- Main-attention rotary uses the validated host-generated BF16 table with its adopted rounding
  boundary; indexer rotary stays on device. These choices have separate evidence. Do not
  reopen the rejected host-indexer-table proposal as an assumed correctness fix.

For long context, passkey extraction must be correct at all four depths; E0 has no external
correctness oracle, so record that limitation. Both still enforce own-score DSA and state/cache.

## 4. Highest-priority research questions

### A. Establish what fast official prefill actually does

1. Find primary-source TPU v4 MoE inference results and code first; then other TPU generations
   where they provide useful mechanisms. Search MaxText, JetStream, TPU-inference, Tokamax,
   MegaBlocks/grouped matmul implementations and official JAX/Pallas examples.
2. For each impressive throughput result, extract model total/active parameters, attention
   type, quantization, exact chip generation/count, prompt/output lengths, request concurrency,
   prefix-cache policy, warm/cold status, timing boundaries, software pin and actual command.
   Distinguish runnable configuration, measured result and marketing claim.
3. Is there verified ~10K input tokens/s for a single long prompt on 32 v4 chips or a usefully
   comparable slice? If not, what is the closest measured evidence and what prevents comparison?
4. Read the hot implementation, not only its README: token grouping, projection layouts,
   activation/reduction boundaries, attention kernels, tile sizes, pipelining and dispatch.
   Which mechanisms are essential, and which advertised features only improve concurrency?
5. For each reusable kernel, identify exact file/function/commit, license, supported dtype and
   quantization-scale layout, minimum hardware/runtime and shape constraints. Distinguish
   tested v4 support from a TPU-generic name. Modern SparseCore/MXU features may be incompatible.

### B. Choose a genuinely useful MoE grouping size and kernel

The arithmetic mean over all experts is B*8/256 = B/32 routed rows/expert:

| Prompt grouping B | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| Mean routed rows/expert | 4 | 16 | 32 | 64 | 128 |

This identity does not assume uniform routing, but the distribution and active-expert mean do.
For the uniform independent-routing SCENARIO, B128 touches approximately 251.6/256 experts.
Across 75 MoE layers, ideal routed weight traffic is roughly 5.57 GB/token at B128 versus
1.42 GB/token at B512, assuming each active expert's tiles load once per group. These are
ideal scenario bytes, not measured traffic; scales, reloads and other work are excluded.

1. Is B128 predictably too small for efficient v4 MoE? Recommend an initial bounded sweep
   among 512/1024/2048/4096 or justified alternatives, with realistic expert8/feature4 shapes.
2. Should we adapt the existing grouped raw-FP8 kernel, adopt a Tokamax/MegaBlocks primitive,
   or redesign its schedule? Compare actual benefits, engineering effort and numerical risk.
3. How should M/N/K tiles, expert-boundary tiles, double buffering, scale decoding and weight
   reuse be arranged on v4 VMEM/MXUs? Account for tails, active/empty experts and owner skew.
4. Can separate gate/up weights be loaded together or interleaved on the fly without a new
   checkpoint? Can projections share input reads while feature reduction remains before activation?
5. Why is the supplied one-owner case 29.283 ms versus distributed 12.028 ms at B128 despite
   better tile occupancy? State hypotheses and the smallest traces that distinguish them.
6. Which natural route statistics are sufficient: rows/expert, active tiles, owner imbalance,
   padding, maximum group, repeated weight loads? Specify exactly what to capture and how to
   derive it without replacing real routing with random/perfectly balanced routing.
7. Can grouping several attention tiles into a larger MLP window preserve a manageable HBM/
   VMEM/code-size budget? When should tile loops replace Python-unrolled graph construction?

### C. Make DSA and attention scale without dense long-context scratch

1. Design a tiled causal DSA scorer plus exact lowest-position top-2048 merge. Skip future-only
   tiles physically, not merely score full capacity and mask afterwards. Compare merge/sort
   algorithms, communication payloads and crossover sizes on TPU v4.
2. How to reuse index-key tiles across many queries without materializing
   [prompt_rows,index_heads,full_context]? Give explicit shapes, byte budgets and loop order.
3. DSA scorer work is quadratic in prompt length even though final attention selects only 2048
   positions. At what grouping/prefix length does it dominate? Is exact selection/sort or score
   arithmetic more likely limiting, and which bounded benchmark distinguishes them?
4. Can selected-KV gathering and latent sparse attention be fused/streamed with stable online
   softmax on v4? Handle causal partial blocks, arbitrary selected positions, page ownership,
   duplicates/sentinels, and the exact attention null/sink behavior where applicable.
5. Should full-indexer and IndexShare layers use different row/key tile sizes or prefix buckets?
   Can address metadata be reused over four layers while KV values remain layer-local?
6. How to preserve dual index caches/M64 repair without serializing the entire prompt? Is the
   current repair boundary intrinsically needed by the numerical contract or just the present
   kernel interface? Separate a safe implementation improvement from a numerical redesign.

### D. Remove communication and layout overhead without breaking the working plan

1. Within expert8 x feature4, map each MoE/attention reduction and its true dependency.
   Which communication can overlap matmul/DMA, and which lies before a required nonlinearity?
2. Are tuple packing, reduce-scatter, token dispatch, reciprocal feature layouts or fused
   collective/GMM kernels worthwhile at our row counts and physical 2x4x4 topology?
3. Give a before/after physical group, shape, byte and synchronization-count inventory for each
   recommendation. No repeated full32 hidden reconstruction, host-staged transport or all-chip
   gather disguised as an innocent reshard.
4. Is a different PREFILL partition compatible with unchanged resident checkpoint ownership and
   existing decode cache layout? Quantify transition/compile/memory costs. Historical PP8/PP16
   decode adjudication does not automatically rank batched prefill, but do not reopen a full
   architecture campaign without a concrete prospect of beating the existing path.

### E. Budget memory, lifetimes and compiler size honestly

1. Propose a complete per-chip liveness schedule at capacity262656: resident FP8/scales, KV,
   repaired/unrepaired index state, old/proposed rollback state, multirow activations, grouped
   route buffers, sort/gather scratch, code, communication buffers and reserve.
2. Which apparently copied cache arrays are real allocations in optimized HLO? Where can
   donation/in-place updates be used safely while preserving rollback and observer lifetimes?
   Distinguish logical tree aliases, physical buffer aliases and references retained by Python.
3. What is the smallest full-context SHAPE test plus selected-real-layer execution that can
   detect an OOM before loading/running the full 753B model? Do not treat single-layer HBM as
   full-model proof or compile-only counters as execution peak.
4. How many row/prefix/tail executable variants are worthwhile? Provide an executable-size and
   compile-time budget and a cache key covering source, topology, JAX/libtpu and static config.
5. Can persistent compiled-cache reuse shorten repeated load/compile cycles safely? What
   provenance checks must survive? No installing/upgrading the live environment speculatively.

### F. Build a credible roofline and end-to-end target, not a convenient milestone

Our reproducible APPROXIMATE useful arithmetic model yields:

| Prompt length | Linear PFLOP | Selected-attention PFLOP | Causal DSA PFLOP | Sum PFLOP |
|---|---:|---:|---:|---:|
| 127363 | 9.992445 | 2.810623 | 1.395305 | 14.198373 |
| 131072 | 10.283440 | 2.893135 | 1.477755 | 14.654331 |
| 262144 | 20.566880 | 5.809040 | 5.910997 | 32.286918 |

These counts exclude routing, norm/activation/softmax, top-k/sorting, repair, padding,
communication, formatting and some exact-path operations. They are not exhaustive physical
FLOPs or a latency prediction. Google specifies 275 BF16/int8 TFLOP/s and 1200 GB/s HBM per v4
chip; nominal fleet compute is 8.8 PFLOP/s. Raw FP8 storage does not by itself imply native
FP8 execution at that rate. Validate the model and effective dtype/kernel throughput.

1. Derive compute, HBM, ICI, vector/scalar/sort and dispatch budgets for the actual local shapes.
   Explain whether phases overlap; do not simply divide total FLOPs by peak and promise TTFT.
2. At 10K input tokens/s, full128K/full256K would take about13.1/26.2 seconds just for prefill.
   Is this defensible here? Give optimistic/credible/pessimistic ranges with explicit achieved-
   efficiency assumptions, not an unsupported claim that it is either easy or impossible.
3. Which few measurements would most reduce uncertainty before setting immutable final targets?
   Historical ~600 s legacy128K/item and1337.7 s legacy256K prefill are differently scoped;
   do not silently adopt them, or an interim200–400 s band, as a final quality bar.
4. Register distinct resident-weight no-prefix-hit prefill and warm delivered-TTFT definitions;
   include transfer/cache initialization/first token delivery. Keep cold load/compile, observer/
   trace/sealing, base decode and speculative throughput separate. Discuss sustained versus
   individually synchronized measurements and fleet stragglers.

### G. Find real bottlenecks quickly; preserve correctness without proof bureaucracy

1. Specify the smallest prefill XPlane capture that distinguishes MXU underfill, HBM/formatting,
   route imbalance, DSA sort, selected-KV gather, ICI stalls, executable launch and host stalls.
   Name actual available v4/JAX/XProf counters and their limitations; do not invent counters.
2. How to correlate exact optimized HLO/kernel identities, physical groups, active rows and
   owner maps with trace time? Explain overlap/wait accounting so category sums are not mistaken
   for critical-path wall time. Decode traces cannot answer prefill attribution.
3. Suggest sufficient tests for each optimization: independent math, identical-input replay,
   row-level bounds, exact route/set/ties, cache interventions, real one-layer test, own8K and
   long quality. Identify tests that provide no new information and should not be repeated.
4. Compiler fusion changed observable scalar-reference boundaries in earlier experiments.
   How should completed BF16 interfaces and FP32 route sums be preserved/tested without
   endlessly trying to emulate an accidental compiler reduction tree?
5. Can we reduce structural admission to reviewed fixed graph identity plus targeted semantic
   checks, avoiding a new general formal proof project? Identify any genuinely missing safety
   obligation that numerical tests cannot cover, rather than asking for blanket hardening.
6. Give a failure-localization decision tree: wrong routes, wrong selected set, tensor-only
   drift, OOM, hang, or wall regression. Each branch needs an inexpensive first check, original
   evidence to save and a stopping/escalation rule. Prefer minutes, not a long-model rerun.

### H. Finish the project without accumulating costs or introducing a new platform

1. What is the shortest dependency-ordered implementation sequence after the current layer6
   discriminator? Separate research that blocks the next step from useful optional follow-up.
2. How to preserve one minimal runtime/checkpoint setup, compact reproducibility metadata and
   irreplaceable evidence while avoiding full tensor dumps, duplicate HLO and extra weight packs?
3. Can small bounded asynchronous verified loading improve cold startup without a second full
   buffer? This is secondary: DB574 rank0 load139.3 s/compile~699.1 s did not cause its16,425 s
   prefill. Do not prioritize startup at the expense of the dominant prompt-processing problem.
4. What regressions should disqualify a prefill win: worse batch-one decode, unmeasured memory,
   larger irreversible storage, weaker exactness or worse delivered TTFT? Define the decision.
5. Speculation/continuous batching/multi-request throughput are not fixes for one prompt's
   prefill. Defer them unless a directly reusable primitive helps the present work.

## 5. Starting primary sources — inspect current code and record your own pins

These are starting leads from our read-only inspection, not an exhaustive survey or proof of
v4 compatibility. Verify status and dates yourself; cite exact source lines/functions/results.

- [Google TPU v4 architecture](https://docs.cloud.google.com/tpu/docs/v4): hardware limits and
  topology, not an achieved model throughput result.
- [Official Qwen3.5-397B prefill benchmark configuration](https://github.com/vllm-project/tpu-inference/blob/871abba1d70db1c9c7474bd80bd199546bf78365/.buildkite/benchmark/cases/daily/Qwen3.5_397B_prefill.json):
  the GBS1 case uses v7x-8,8192 input/1 output, max-concurrency1, max-num-seqs1,
  max-num-batched-tokens16384, FP8 KV, EP and prefix caching OFF. This is a CONFIGURATION,
  not a measured10K result, and it is not v4. It establishes that official single-request
  large prefill is a relevant path, not that all impressive figures require many requests.
- [Official MoE grouped execution](https://github.com/vllm-project/tpu-inference/blob/871abba1d70db1c9c7474bd80bd199546bf78365/tpu_inference/layers/common/fused_moe_gmm.py):
  uses Tokamax GMM, combined up/gate plus activation, then down projection and grouped output
  movement. Check the surrounding sharding before transferring this fusion to ours.
- [Experimental fused MoE](https://github.com/vllm-project/tpu-inference/blob/871abba1d70db1c9c7474bd80bd199546bf78365/tpu_inference/kernels/experimental/fused_moe/README.md):
  full GMM/activation/output ICI A2A fusion; a separate AG+GMM example discusses overlap.
  Its README lists further upstream-gather fusion and SparseCore offload as future work;
  do not describe those as already implemented performance results. Retune for device.
- [Tokamax](https://github.com/openxla/tokamax/tree/c4d7d68a92a40fbc4b70e5ca42242bba709b014e):
  current grouped-kernel lead. The older TPU-inference local gmm_v2 file labels itself stale.
- [MaxText inference microbenchmark](https://github.com/AI-Hypercomputer/maxtext/blob/c55443590e2925b5da66f82ea7bc9a32a8c662e3/src/maxtext/inference/inference_microbenchmark.py):
  compiles whole prefill lengths, warms, repeats and synchronizes; inspect timing scope.
  Its separate decode throughput metric is not prefill or delivered serving TTFT.
- [MaxText MoE](https://github.com/AI-Hypercomputer/maxtext/blob/c55443590e2925b5da66f82ea7bc9a32a8c662e3/src/maxtext/layers/moe.py):
  token/expert grouping reference. Training defaults must not be passed off as inference tuning.
- [JAX TPU matmul tutorial](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html),
  [sparse kernels](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html),
  [distributed Pallas](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html):
  use for implementation mechanisms, with version/hardware checks.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102):
  TPU v4 inference utilization/partitioning reference; not a GLM MoE benchmark.

## 6. Lessons that should prevent another week of avoidable work

These are local observations, not claims about an upstream hardware defect:

1. A memory-safe serial scan is not efficient prefill. Inspect real loop bodies and weight reuse.
2. In an early MoE trial, an extra BF16 local route-sum round was exposed by saved original
   arrays and CPU replay. An explicit FP32 route sum then passed unchanged bounded contracts.
3. A scalar reference's output changed when its observable boundaries changed. Exact prefix
   reproduction and completed-input suffix replay resolved the comparison. A BF16 cast in HLO
   alone did not prove a compiler cast-omission bug. Do not restart blind barrier/precision trials.
4. Exact legacy DSA set identity across different valid arithmetic was an inappropriate universal
   requirement. Independent reference adjudication closed the historical issue; own-score exact
   selection and exact short raw tokens remain mandatory. Do not relax real numerical defects.
5. NaNs can be masked into finite outputs. Health must cover live operands and cache writers;
   empty experts are valid and are not the same as invalid metadata.
6. Cheap composed tests caught manifest digest-kind mismatch, a plain shard_map/.lower API
   mismatch, block-table rank errors and JSON tuple/integer-key roundtrip defects. Test the actual
   producer -> serialization -> collector boundary, not only compatible in-memory mocks.
7. Recursive whole-HLO analyses and repeated scans did not scale. One indexed graph with
   memoized/iterative traversal and replay once per unique graph removed avoidable host cost.
8. Save raw arrays/graphs and fsynced memory/phase records before fallible checks. DB588 was
   recovery-sealed after a report serialization fix WITHOUT rerunning model computation.
9. Local errors before a collective can strand other hosts. Matched fleet votes, ownership-aware
   monitoring and recoverable evidence matter; a free lease alone does not prove an idle fleet.
10. Storage and controller disk are separate constraints. The latest handoff records only ~2.35 GB
    local free space versus the full-model4 GiB evidence floor. Resolve actual disk dependencies
    before a full run, without weakening the floor or deleting irreplaceable evidence. Research
    cannot determine what is currently open on this machine.

## 7. Required response format and decision quality

Start with a one-page executive answer: what is fundamentally wrong, what already works, what
should change first, and whether10K is supported, plausible under assumptions, or unsupported.
Do not report a fabricated project completion percentage or calendar ETA.

Then provide:

1. A source-backed benchmark comparison table with the dimensions in A, including unavailable
   fields and non-comparable cases. Link actual primary results and pinned implementation code.
2. A ranked recommendation table: priority, local evidence/hypothesis, exact algorithm/API,
   v4/layout compatibility, expected benefit RANGE and assumptions, numerical/HBM/storage risk,
   smallest decisive test, success/refusal condition and explicit adopt/defer/reject decision.
3. A concrete proposed prefill schedule with tensor ownership, tile loops, reduction/activation
   boundaries, dual-cache lifetime and the decode handoff. Include shapes/bytes, not only names.
4. A reproducible roofline/phase budget, uncertainty ranges and a short list of missing local
   measurements. Explain what each measurement would change in your recommendation.
5. At most FIVE first experiments, ordered by expected information gained per unit of cost.
   Give inputs, reference, instrumentation, cold versus execution budget, memory cap and stopping
   rule. Do not demand a new full-model benchmark for a question a real layer can settle.
6. A review of our proposed B128 step: continue as the immediate discriminator, alter it now,
   or replace it. Give concrete reasons and the smallest required delta; do not dismiss already
   acquired evidence or assume an unrun candidate succeeded.
7. A “do not do” list: attractive optimizations that are incompatible, redundant, unproved,
   dominated by another bottleneck, or expensive relative to the evidence they provide.
8. Separate VERIFIED source facts, LOCAL supplied evidence, DERIVED estimates and HYPOTHESES.
   Cite claims near the relevant text. If evidence conflicts, explain it rather than selecting
   the optimistic number. End with the implementer's next three concrete actions.

We will independently review the report and test its recommendations. Research cannot guarantee
that a recommendation fixes all blockers; useful work narrows uncertainty and identifies decisive tests.

## 8. Optional attachments / local evidence navigation

This brief is enough to start web research. If the researcher can accept additional files, the
highest-value small attachments are the cost model and layer-window design, not terabytes of dumps.

- [Current goal](../../goal.md) and [binding specification](../glm-tpu-revolution.md), especially
  amendments21–24. Earlier sections include superseded plans; later amendments govern.
- [Efficiency audit](ENGINE_EFFICIENCY_AUDIT.md): cumulative findings; read its later updates
  before treating early “pending” notes as current blockers.
- [Cost model](PREFILL_COST_MODEL.md): exact arithmetic expressions and scoped baseline protocols.
- [Layer-window design](PREFILL_LAYER_WINDOW.md): current interfaces and fixed layer6 discriminator.
- [Memory admission](PREFILL_MEMORY_ADMISSION.md), [decoder assembly](PREFILL_DECODER_ASSEMBLY.md).
- [Observability playbook](GATE_D_OBSERVABILITY_PLAYBOOK.md), [lessons](GATE_D_LESSONS.md),
  [evidence map](EVIDENCE_MAP.md), and the latest [handoff](../../HANDOFF.md) entries.
- Compact receipts under `docs/artifacts/`:
  `prefill-batched-own2k-sealed-20260908.json`,
  `prefill-moe-equal128-baseline-20260908.json`,
  `prefill-window-layer6-four-graph-acquisition-20260908.json`.

Code for a later targeted review (web research does not require local filesystem access):

- `glm_tpu/greenfield/runtime/ws32_batched_prefill.py` and `ws32_decoder.py`.
- `glm_tpu/greenfield/kernels/ws32_prefill_{window,layer,moe,linear,dsa,attention}.py`.
- `glm_tpu/greenfield/kernels/prefill_{dsa,routes,cache}.py`.
- `glm_tpu/greenfield/kernels/pallas/{prefill_grouped_fp8,fp8_matmul}.py`.
- `glm_tpu/greenfield/validation/ws32_prefill_memory.py`.
- `scripts/greenfield/prefill_window_{protocol,admission,acquisition}.py`.
- `scripts/greenfield/prefill_window_worker.py`: WIP at this snapshot, not deployed or sealed.

No new TPU run, checkpoint, benchmark promotion or acceptance-policy change was performed to
prepare this document. It describes the current frontier, not a pause instruction for engineering.
