# GLM-5.2-FP8 TPU v4 Greenfield Rewrite Specification

**Document purpose:** Codex-ready engineering specification for a clean rewrite of the GLM-5.2-FP8 inference engine on the existing 8-host / 32-chip TPU v4 slice.

**Target repository:** `/home/gianl/glm-tpu`  
**Legacy inference fork used only as an oracle:** `GianluigiVitale/tpu-inference`  
**Recommended branch:** `rewrite/topology-first-decode`  
**Recommended worktree:** `/home/gianl/glm-tpu-topology-rewrite`  
**Status date:** 2026-08-05  
**Primary workload:** one live sequence, 256K context, one generated token per base decode step  
**Primary objective:** minimum single-stream latency, not aggregate multi-request throughput

---

## 0. Instructions to Codex

Implement this specification as a new, default-off engine. Do not incrementally refactor the accepted engine into the new design.

Before changing code:

1. Read this document completely.
2. Read `/home/gianl/glm-tpu/HANDOFF.md`.
3. Read `/home/gianl/glm-tpu/docs/RESEARCH_LOG.md`.
4. Inspect the existing correctness, trace, provenance, cleanup, state-manifest, DSA-diff, and cache-write tools.
5. Create a dedicated branch and worktree.
6. Record the exact starting commit of every repository involved.
7. Do not delete, rewrite, or weaken any existing protected evidence.
8. Do not run a full 753B checkpoint experiment until the phase gates in this document authorize it.
9. Do not claim a performance win from CPU tests, HLO inspection, a synthetic kernel, named profiler labels, aggregate throughput, or profiler-contaminated wall measurements.
10. Keep every new optimization default-off until it has passed the required correctness and performance gates.

The old engine remains the production correctness oracle. The new engine must not import its model-execution path. Reuse only data definitions, checkpoint knowledge, validation tools, benchmark logic, and other clearly isolated utilities.

---

# 1. Executive decision

## 1.1 Is the previously proposed all-32-chip 2D/3D architecture certainly the best possible design?

No.

It is a credible high-performance design and must remain a serious comparator, but it is not proven to be the minimum-latency design for this exact workload.

The local evidence shows that the current failure is caused by excessive tensor/expert parallelism on the critical path: 75 MoE layers each perform a blocking reconstruction over the full physical 32-chip group. The current stack uses all 32 chips in every layer primarily to distribute weight capacity. That simultaneously creates very small per-chip decode operations and a global synchronization barrier after nearly every MoE layer.

For batch-one inference, the stronger starting hypothesis is:

> Use model depth to distribute weight capacity with topology-aligned pipeline stages, and use only the smallest local parallel group that gives sufficient HBM and useful per-layer compute.

This changes the communication problem from approximately 75 blocking 32-chip combines per token to:

- 75 local 2-way or 4-way combines, depending on the selected plan;
- a small number of point-to-point activation transfers between pipeline stages;
- no full 32-chip activation reconstruction inside a transformer layer.

The recommended first end-to-end architecture is therefore:

> **Eight layer-pipeline stages, one TPU VM host per stage, four topology-local chips per stage, with all repeated layer collectives confined to that four-chip group.**

This plan is named:

```text
PP8_LP4
```

where:

- `PP8` = eight pipeline stages;
- `LP4` = four-chip local parallel group per stage.

A mandatory second plan is:

```text
PP16_LP2
```

where each host contains two two-chip stages. It trades eight additional stage transfers for lower two-way collective latency.

A mandatory high-complexity comparator is:

```text
WS32_2D
```

where all 32 chips participate in every layer using genuine two-dimensional weight-stationary sharding and a persistently sharded residual.

The rewrite is successful only after the same protected workload compares these plans. The architecture must make plan selection explicit rather than embedding one unchangeable logical mesh.

## 1.2 Why PP8_LP4 is the recommended first implementation

The protected state is about 721.2 GiB. The average state allocation is therefore approximately:

```text
721.2 GiB / 32 chips = 22.54 GiB/chip
```

Pipeline partitioning does not increase this average:

```text
PP8_LP4:
  721.2 GiB / 8 stages = 90.15 GiB/stage
  90.15 GiB / 4 chips = 22.54 GiB/chip

PP16_LP2:
  721.2 GiB / 16 stages = 45.08 GiB/stage
  45.08 GiB / 2 chips = 22.54 GiB/chip
```

Actual assignments must be balanced by bytes, FP8 scale ownership, KV allocation, temporary buffers, and compiler overlays—not merely by layer count.

PP8_LP4 is preferred as the first working system because:

- it maps naturally to the existing eight TPU VM hosts;
- all repeated layer collectives can remain host-local;
- only seven inter-stage boundaries exist;
- each stage has four-chip aggregate HBM bandwidth;
- it avoids the current 32-rank per-layer synchronization;
- it is simpler to bring up than a complete all-32-chip 2D MoE implementation;
- it creates larger local decode operations than the current 32-way sharding, potentially improving tiny-GMM utilization;
- it leaves a clear path to PP16_LP2 and speculative pipeline execution.

## 1.3 What “best possible” means in this project

No architecture is declared best from design arguments alone.

The best plan is the plan that minimizes protected profiler-free single-stream wall latency while satisfying:

- exact DSA selected sets and tie order;
- correct raw output;
- state/load/cache integrity;
- full 256K serving;
- acceptable HBM margin;
- fresh fleet-wide physical traces;
- exact code/config provenance;
- authenticated cleanup.

The plan search is part of the architecture, not an optional later optimization.

---

# 2. Ground truth from the accepted workload

Treat the following local facts as authoritative for this project:

- Model: `zai-org/GLM-5.2-FP8`.
- Total parameters: 753B.
- Active parameters per token: about 40B.
- Layers: 78.
- First three layers dense.
- Remaining 75 layers MoE.
- Hidden width: 6144.
- Routed experts: 256.
- Shared experts: one.
- Routed top-k: eight.
- MoE intermediate width: 2048.
- DSA top-k: 2048.
- DSA indexer heads: 32.
- IndexShare: selected indices reused over groups of four layers.
- Protected on-device state: about 721.2 GiB.
- Routed expert weights and scales: about 696.1 GiB.
- Hardware: 32 TPU v4 chips, eight hosts, four chips per host.
- Physical topology: 2×4×4.
- HBM: 32 GiB/chip.
- Current accepted latency: 287.666063 ms/token.
- Current accepted device rate: 3.476253 tok/s.
- Current accepted wall rate: about 3.3 tok/s.
- Accepted collective time: 161.35 ms/token.
- Dominant 75 MoE combines: 106.495016 ms/token.
- Mean dominant combine latency: about 1.420 ms.
- Dominant payload: `bf16[2,6144]`.
- Current physical group: all 32 ranks represented by logical `model × dcp`.
- GMM category: 32.55 ms/token.
- Sort/top-k category: 34.87 ms/token.
- Sparse attention kernel: 0.48 ms/token.
- Replacing the MoE reduction with an all-gather did not help.
- Reducing routed decode rows from 256 to 16 produced only a small gain.
- The main problem is synchronization depth and layout, not payload bandwidth.

Never replace these local measurements with numbers from external papers.

---

# 3. Primary architecture: PP8_LP4

## 3.1 Physical placement

Discover and record the real device metadata at runtime:

- JAX device id;
- process index;
- host id;
- physical TPU coordinates when exposed;
- core/chip relationship;
- local device ordering;
- neighbor relationships inferred from coordinates;
- ICI-visible topology.

Do not assume that JAX device ordering matches the desired topology.

The first plan groups devices by TPU VM host:

```text
pipeline stage 0 -> host 0 -> 4 local chips
pipeline stage 1 -> host 1 -> 4 local chips
...
pipeline stage 7 -> host 7 -> 4 local chips
```

Within each stage, select the local mesh shape only after inspecting physical coordinates. Candidate local shapes include:

```text
LP4_1D: 4
LP4_2D: 2 × 2
```

The initial reference path may use one-dimensional local parallelism. A two-dimensional local layout is an optimization candidate.

## 3.2 Layer assignment

Assign contiguous transformer blocks to pipeline stages.

The partitioner must optimize a measured or estimated objective containing at least:

- persistent weight bytes;
- FP8 scale bytes;
- KV-cache bytes at 256K;
- temporary activation bytes;
- compiler overlay allowance;
- measured per-layer decode cost;
- DSA full-scorer cost;
- routed GMM cost;
- attention cost;
- stage-boundary transfer cost;
- IndexShare grouping constraints.

Do not partition by equal layer count.

Prefer boundaries that keep an IndexShare group on one stage. When that is impossible, explicitly transfer the compact selected-index state and measure the cost. A stage boundary must never trigger a selected-KV materialization or full hidden reconstruction beyond what the stage transfer already requires.

Generate a machine-readable stage manifest:

```json
{
  "plan": "PP8_LP4",
  "stages": [
    {
      "stage_id": 0,
      "process_index": 0,
      "device_ids": [0, 1, 2, 3],
      "layer_start": 0,
      "layer_end_exclusive": 10,
      "persistent_weight_bytes": 0,
      "fp8_scale_bytes": 0,
      "kv_bytes_at_256k": 0,
      "reserved_overlay_bytes": 0
    }
  ]
}
```

The final values must come from the actual packed checkpoint and cache implementation.

## 3.3 Stage-local model parallelism

The first correct implementation should use a simple, explicit local layout.

### Attention and dense matrices

Use a local tensor-parallel layout across the four stage chips.

The initial implementation may keep the residual replicated within the stage if that produces the simplest exact path. However:

- replication must never extend beyond the local four-chip stage group;
- all physical collectives must name only the local group;
- the HLO contract must reject any 8-, 16-, or 32-chip layer collective;
- stage transfer must not accidentally invoke a global reshard.

After correctness, implement a persistent stage-local hidden-sharded variant and compare it against local replication.

### Routed experts

Initial expert layout:

```text
256 routed experts / 4 stage chips = 64 complete experts per chip
```

Each chip stores complete dimensions for its local expert identities for the layers assigned to the stage. For one token:

1. Compute exact top-eight routing.
2. Broadcast or otherwise make the small live residual available to the local chips.
3. Each chip executes only selected experts it owns.
4. Fuse shared-expert and routed-expert local partials when exact association permits.
5. Combine the output over exactly four local chips.
6. Continue to the next layer with no global collective.

This turns the dominant full-pod combine into a host-local combine.

The implementation must handle the worst case where all eight selected experts reside on one chip. Do not assume even routing for batch one.

### Shared expert

Evaluate three exact implementations:

1. Feature-shard the shared expert across the local group.
2. Assign it to one local chip and combine its result with routed output.
3. Replicate only if measured HBM permits and performance justifies it.

Do not silently duplicate weights.

## 3.4 Stage-to-stage activation transport

The transferred object should be the minimum state needed by the next stage:

- residual activation;
- position/token metadata;
- compact DSA/IndexShare metadata only when it crosses a boundary;
- no weights;
- no KV pages;
- no full batch bucket.

The live residual payload is small:

```text
bf16[1,6144] ≈ 12 KiB
```

Potential transport mechanisms, in order of implementation:

1. A global JAX program using explicit `collective_permute`/`ppermute` across stage neighbors.
2. Explicit JAX resharding only if HLO proves it lowers to the intended point-to-point pattern.
3. Distributed Pallas asynchronous remote copies after the reference path is correct.
4. A persistent multi-host runtime only if the first two approaches cannot avoid host-dispatch latency.

Do not use Ray object transfer, host DRAM staging, GCS, Python RPC, or ordinary host networking on the decode critical path.

The token selected by the last stage must be returned to stage zero through a tiny explicit path. This return must not cause all model state to synchronize.

## 3.5 Pipeline execution inside one token

Base decode remains autoregressive:

```text
stage 0 -> stage 1 -> ... -> stage 7 -> logits/sample -> token return
```

Pipeline “bubbles” are not the primary base-latency concern because transformer layers are already sequential. The relevant cost is the sum of stage compute plus seven transfers.

The pipeline runtime must nevertheless avoid host-side dispatch between stages. A phase-zero skeleton must prove that a token-sized activation can traverse all stages in one compiled device program or an equivalently low-overhead persistent runtime.

This is a critical gate. Do not load the model before it passes.

---

# 4. Mandatory architecture challengers

## 4.1 PP16_LP2

Topology:

```text
16 pipeline stages × 2 adjacent chips/stage
```

Goals:

- replace four-way local collectives with two-way collectives;
- increase local per-chip weight and compute work;
- keep average persistent state near 22.54 GiB/chip;
- measure the cost of 15 stage boundaries.

Prefer mapping each pair to directly adjacent chips along the physical dimension of length two when the discovered coordinates permit it.

The plan must use the same model semantics, checkpoint format abstraction, correctness suite, and benchmark harness as PP8_LP4.

## 4.2 WS32_2D

This is the all-32-chip two-dimensional weight-stationary comparator.

Requirements:

- weight capacity distributed across all 32 chips;
- expert identity sharded on one physical subgroup;
- expert feature dimensions sharded on another subgroup;
- residual kept sharded through the entire transformer block;
- reciprocal layouts for MoE up/gate and down projections;
- no full residual all-gather between layers;
- only topology-aligned subgroup collectives;
- explicit FP8 scale sharding;
- attention, RMSNorm, residual, logits, and KV layout compatible with the same hidden shard;
- context ownership separated from expert ownership;
- no logical axis that silently expands to the full 32-chip group unless that operation is intentionally global.

This plan is more complex but may achieve lower base latency by using all 32 chips’ HBM bandwidth concurrently. It must be evaluated, not assumed superior.

## 4.3 PP32_LP1 research plan

A one-chip-per-stage plan eliminates intra-stage collectives but is not a primary target because:

- 78 layers cannot be evenly packed onto 32 single-chip stages;
- three-layer stages may exceed HBM after scales, KV, and overlays;
- one chip supplies only one chip of HBM bandwidth at a time;
- active FP8 weight traffic alone creates a rough lower-bound concern for 30–70 tok/s.

Implement only after exact byte packing shows it is feasible and after PP8/PP16 evidence suggests it is useful.

## 4.4 Legacy comparator

Keep one plan descriptor representing the accepted legacy geometry:

```text
LEGACY_TP32_DCP8
```

It is never the default for the new engine. It exists to ensure benchmark and trace tooling can compare the rewrite against the accepted parent under identical workload definitions.

---

# 5. Engine design

## 5.1 Native JAX execution

Implement the core decoder directly in JAX.

Use:

- `jax.jit`;
- `jax.shard_map`;
- explicit `Mesh`;
- explicit `PartitionSpec`;
- explicit collectives;
- explicit output sharding;
- HLO/StableHLO inspection.

Avoid relying on automatic partitioning for the repeated critical path until the resulting physical operations are proven.

Automatic sharding may be used inside isolated local kernels only when:

- input/output sharding is explicit;
- the generated HLO is inspected;
- no global collective is inserted;
- exactness and performance are measured.

## 5.2 Separate executables

Compile separate programs for:

```text
prefill_chunk
decode_batch1
decode_batchN
verify_k_tokens
sample_logits
```

`decode_batch1` must use a true static live row count of one.

It must not carry a batch-32 compile bucket, dead rows, dead routing assignments, dead sort rows, or dead selected-KV rows.

Prefill and mixed scheduling must not share the batch-one executable.

## 5.3 Core state types

Create explicit typed structures similar to:

```python
@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int
    hidden_size: int
    num_routed_experts: int
    num_shared_experts: int
    top_k: int
    moe_intermediate_size: int
    dsa_top_k: int
    index_share_group_size: int

@dataclass(frozen=True)
class ExecutionPlan:
    name: str
    pipeline_stages: int
    local_parallel_size: int
    stage_assignments: tuple["StageAssignment", ...]
    local_mesh_shape: tuple[int, ...]
    residual_layout: str
    expert_layout: str
    kv_layout: str
    transport: str

@dataclass
class DecodeState:
    residual: jax.Array
    kv_cache: "KVCacheState"
    dsa_state: "DSAState"
    position: jax.Array
    request_state: "RequestState"
```

Do not represent sharding only in comments. Add runtime assertions and trace-time checks.

## 5.4 No hidden global tensors

Inside the transformer, a global logical shape is permitted only if its sharding is explicit and no physical global materialization occurs.

Add an HLO linter that rejects:

- repeated full-pod `all-reduce`;
- repeated full-pod `all-gather`;
- repeated full-pod `reduce-scatter`;
- resharding that reconstructs `bf16[1,6144]` or larger over all 32 chips after each layer;
- implicit collectives over combined expert/context axes;
- decode tensors with 32 token rows.

The linter must report:

- source stack;
- HLO operation;
- physical replica groups;
- operand/result shapes;
- count per decode step;
- whether the operation is inside the repeated layer region.

## 5.5 Numerical policy

Maintain a documented numerical contract for every kernel:

- checkpoint storage dtype;
- FP8 scale dtype;
- dequantization dtype;
- accumulation dtype;
- reduction dtype;
- activation dtype;
- output dtype;
- reduction association;
- routing tie semantics;
- sampling semantics.

Exact DSA selected sets and tie order are mandatory.

For model arithmetic, define three comparison levels:

1. Bitwise equality where historically required and feasible.
2. Bounded tensor error for internal layer equivalence.
3. Exact raw generated token and benchmark correctness.

Any changed reduction association must be explicitly reviewed and protected on real hardware.

---

# 6. Checkpoint and memory architecture

## 6.1 Offline final-layout packing

Create an offline packer that writes weights in the exact final device ownership layout for each execution plan.

Do not load the original checkpoint and perform large runtime reshards.

Packed artifacts must include:

- plan id;
- model revision;
- source tensor name;
- source shape/dtype;
- destination stage;
- destination device-coordinate group;
- destination local shard;
- FP8 scale ownership;
- checksum;
- total bytes;
- padding bytes;
- format version.

Example output structure:

```text
packed/
  manifest.json
  PP8_LP4/
    stage_00/
      device_slot_00/
      device_slot_01/
      device_slot_02/
      device_slot_03/
    ...
  PP16_LP2/
  WS32_2D/
```

Do not create every packed layout immediately. Implement PP8_LP4 first, but make the manifest versioned and plan-aware.

## 6.2 Streaming loader

The loader must:

- stream directly into final local shards;
- validate every leaf against the manifest;
- validate shape, dtype, byte count, scale ownership, and checksum;
- detect non-finite values;
- refuse to serve on incomplete state;
- produce the same protected state-manifest evidence as the accepted engine;
- avoid transient double-buffering of the full local state;
- record peak host and device memory.

## 6.3 HBM budget

For every stage/device, account for:

- persistent model weights;
- FP8 scales;
- KV cache at target context;
- DSA/index state;
- executable/program memory;
- compiler overlays;
- temporary matmul buffers;
- collective buffers;
- transport buffers;
- safety margin.

Promotion requires a measured peak-HBM report for all 32 chips.

Do not use aggregate 1 TiB as a capacity argument.

The gate is:

```text
No chip may depend on unmeasured headroom.
```

Set a concrete safety margin after the first compiled one-layer and short-context runs. Until then, report both measured peak and free bytes rather than inventing a fixed limit.

---

# 7. Kernel implementation order

## 7.1 Reference kernels first

Implement exact, readable JAX reference paths for:

1. RMSNorm.
2. Linear projections.
3. Rotary/position logic.
4. DSA scorer.
5. Exact distributed top-k.
6. Selected-position representation.
7. KV page lookup.
8. Sparse attention.
9. Router.
10. Routed expert computation.
11. Shared expert.
12. Residual addition.
13. Final norm.
14. Vocabulary/logits.
15. Sampling.

Reference kernels must expose explicit local shapes and collectives.

## 7.2 Pallas only after layout correctness

Replace critical paths with Pallas in this order:

1. FP8 up/gate expert matmul.
2. Activation and scale handling.
3. FP8 down expert matmul.
4. Fused routed/shared output combine.
5. DSA scorer.
6. Exact top-k and position ordering.
7. Selected-KV gather fused with sparse attention.
8. Stage-to-stage asynchronous remote copy.
9. Stage-local RMSNorm/linear fusion where useful.
10. Verification of multiple speculative tokens.

Each Pallas kernel must have:

- an interpreted reference test where supported;
- CPU/JAX reference comparison;
- TPU correctness test;
- odd/tail shape tests;
- exact dtype contract;
- HLO/kernel count assertions;
- a microbenchmark;
- a fallback path.

Do not tune tile sizes before proving the intended communication layout.

---

# 8. DSA and 256K context design

## 8.1 Batch-one shapes

All batch-one DSA operations must use one live row.

Forbidden in `decode_batch1`:

```text
[32, ...] scorer inputs
[32, ...] position sorts
[32, ...] selected-position arrays
[32, ...] gathered KV
```

## 8.2 Exact top-k

Preserve:

- exact selected set;
- exact lowest-position tie order;
- exact padding/sentinel behavior;
- exact IndexShare replication rules;
- exact event alignment used by the existing comparator.

Start with the existing exact algorithm translated into the new local group.

Only attempt a blocked/two-pass selector after the one-row rewrite has been measured.

## 8.3 KV ownership

KV cache belongs to the stage that owns the corresponding attention layer.

Within a stage, shard context pages over the local chips.

Do not include the pipeline-stage axis in an attention-layer collective group.

The attention path should:

1. score or obtain selected positions;
2. identify local KV owners;
3. fetch only selected blocks;
4. combine exact attention statistics over the local group;
5. leave the residual in the stage-local layout.

The long-term target is a fused selected-KV gather plus sparse-attention kernel.

## 8.4 IndexShare-aware partitioning

Keep four-layer IndexShare groups on one stage where the memory/compute balancer permits.

If a group crosses a stage boundary:

- transfer only compact selected indices and required metadata;
- preserve exact event identity;
- add an explicit trace category;
- compare against a plan where the group remains intact.

---

# 9. Runtime and orchestration

## 9.1 Pipeline skeleton gate

Before model code, implement a synthetic pipeline executable with the same eight-host grouping.

The payload must traverse:

```text
stage 0 -> stage 1 -> ... -> stage 7 -> stage 0
```

Requirements:

- no host DRAM staging;
- no Python dispatch per stage;
- no Ray data movement;
- no all-32-chip collective;
- explicit point-to-point groups in HLO;
- deterministic values;
- 1,000+ warmed iterations;
- p50, p90, p95, p99 latency;
- fresh multi-host trace;
- exact operation count.

Test payloads:

```text
bf16[1,6144]
bf16[2,6144]
bf16[1,2048]
small routing metadata
```

If a single compiled global JAX program cannot predicate stage-local work without executing inactive-stage compute, stop and solve that runtime problem before continuing.

Candidate approaches:

- `shard_map` plus explicit stage axis and `ppermute`;
- uniform padded stage slots plus verified device-dependent control flow;
- distributed Pallas remote-copy skeleton;
- persistent stage executors with device-resident queues, only if required.

## 9.2 Decode control loop

Keep token loop control device-resident where practical.

Avoid per-token reconstruction of Python objects, Ray messages, or host synchronization.

Host logging must be asynchronous and outside the measured critical path.

Sampling may occur on the final stage. Return only the selected token and minimal state needed for the next step.

## 9.3 Failure behavior

Any stage failure must:

- stop the run;
- prevent serving;
- preserve logs and manifests;
- avoid partial cleanup that destroys evidence;
- identify owned processes before termination;
- finish with the existing authenticated zero-work census.

Add failure-injection tests for:

- missing checkpoint shard;
- checksum mismatch;
- non-finite shard;
- wrong device group;
- stale code hash;
- dropped stage transport;
- cache-write failure;
- inconsistent DSA state;
- insufficient HBM.

---

# 10. Repository layout

Create the new engine under a clearly isolated tree. Suggested layout:

```text
glm_tpu/
  greenfield/
    __init__.py
    config.py
    errors.py
    types.py

    topology/
      discover.py
      coordinates.py
      groups.py
      plans.py
      validate.py

    partitioning/
      cost_model.py
      layer_assigner.py
      memory_model.py
      manifest.py

    checkpoint/
      format.py
      pack.py
      load.py
      checksums.py
      state_manifest.py

    model/
      geometry.py
      state.py
      layer.py
      decoder.py
      logits.py

    sharding/
      specs.py
      residual.py
      experts.py
      attention.py
      kv.py
      hlo_contract.py

    kernels/
      reference/
        rmsnorm.py
        linear.py
        router.py
        moe.py
        dsa.py
        attention.py
        sampling.py
      pallas/
        fp8_gmm.py
        moe_fused.py
        dsa_scorer.py
        exact_topk.py
        sparse_attention.py
        transport.py

    runtime/
      compile.py
      pipeline.py
      transport.py
      stage.py
      prefill.py
      decode.py
      verify.py
      sampling.py

    validation/
      tensor_diff.py
      dsa_diff.py
      token_diff.py
      cache_probe.py
      health.py
      provenance.py

    benchmarking/
      collective_chain.py
      transport_chain.py
      one_layer.py
      short_context.py
      long_context.py
      steady_wall.py
      trace_contract.py

scripts/
  greenfield/
    inspect_topology.py
    build_plan.py
    pack_checkpoint.py
    inspect_packed_checkpoint.py
    microbench_collectives.py
    microbench_pipeline_transport.py
    run_one_layer_equivalence.py
    run_short_decode.py
    run_256k_decode.py
    compare_plans.py
    inspect_hlo_contract.py
    promote_candidate.py

tests/
  greenfield/
    unit/
    hlo/
    checkpoint/
    topology/
    kernels/
    runtime/
    equivalence/
    integration/
    failure_injection/

docs/
  greenfield/
    ARCHITECTURE.md
    EXECUTION_PLANS.md
    NUMERICAL_CONTRACT.md
    CHECKPOINT_FORMAT.md
    TEST_MATRIX.md
    PERFORMANCE_LOG.md
```

Adapt names to the repository’s existing Python packaging conventions, but preserve isolation and responsibility boundaries.

---

# 11. Test strategy

## 11.1 Test levels

### L0: Static and unit tests

Run without TPU:

- type checking;
- linting;
- geometry validation;
- partition plan validation;
- byte accounting;
- checkpoint manifest parsing;
- stage assignment determinism;
- topology grouping from synthetic coordinates;
- routing ownership;
- IndexShare boundary rules;
- numerical reference functions;
- failure cases.

### L1: Forced multi-device CPU tests

Use forced CPU devices to validate:

- `shard_map` semantics;
- collective groups;
- stage routing;
- expert ownership;
- residual transport;
- exact DSA ordering;
- full and narrow branches;
- plan interchangeability.

CPU tests prove semantics, not TPU performance.

### L2: StableHLO/HLO contract tests

For every critical executable, assert:

- expected collective type;
- expected replica groups;
- expected physical count;
- expected operand/result shapes;
- absence of full-pod repeated collectives;
- one live decode row;
- no accidental all-gather of full residual;
- no dead batch bucket;
- no extra stage transfer;
- candidate code fingerprint differs when intended.

### L3: Single-host TPU tests

Use one four-chip host for:

- local TP/EP correctness;
- local collective chain measurements;
- one real layer;
- FP8 packing and load;
- Pallas kernels;
- cache writes;
- DSA exactness;
- memory profiling.

### L4: Multi-host synthetic tests

Use all eight hosts without the full model:

- topology discovery;
- 75-operation dependent collective chains;
- PP8 transport chain;
- PP16 transport chain;
- global-vs-local collective comparison;
- runtime control-flow proof;
- cleanup/provenance.

### L5: One-layer protected metal A/B

Compare one real GLM layer against captured legacy inputs.

Cases:

- dense layer;
- full DSA scorer layer;
- IndexShare reuse layer;
- MoE layer with distributed selected experts;
- adversarial expert concentration;
- router ties;
- zero scales;
- extreme finite scales;
- cache boundary positions.

### L6: Full decoder at short context

Run the complete 78-layer model at 2K or 8K.

This is the first end-to-end architectural discriminator.

Require:

- correct raw tokens;
- exact DSA selected sets and tie order;
- state and cache protection;
- fresh trace;
- steady wall;
- HLO contract;
- memory report;
- clean fleet.

Do not proceed to 256K if the short-context engine still performs full-pod repeated synchronization or has unacceptable latency.

### L7: 128K four-depth smoke

Use the existing protected depths:

```text
0.0
0.05
0.95
1.0
```

Require four of four correct, full protections, archive, provenance, and cleanup.

### L8: 256K E0

Use the exact protected workload and measurement methodology from the accepted stack.

No performance result is accepted without:

- fresh eight-host XPlanes;
- 64 cores represented;
- exact selected decode window;
- profiler-free steady wall;
- exact code/config linkage;
- state/cache protections;
- durable archive;
- authenticated zero-work cleanup.

## 11.2 Suggested commands

Codex should implement commands equivalent to:

```bash
pytest -q tests/greenfield/unit
pytest -q tests/greenfield/topology
pytest -q tests/greenfield/checkpoint
pytest -q tests/greenfield/kernels
pytest -q tests/greenfield/hlo
```

Forced CPU:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=32 \
pytest -q tests/greenfield/runtime tests/greenfield/equivalence
```

Topology inspection:

```bash
python scripts/greenfield/inspect_topology.py \
  --output /home/gianl/glm-run/topology_<timestamp>/topology.json
```

Plan generation:

```bash
python scripts/greenfield/build_plan.py \
  --plan PP8_LP4 \
  --context-length 262144 \
  --output plans/pp8_lp4.json
```

Collective benchmark:

```bash
python scripts/greenfield/microbench_collectives.py \
  --groups 2,4,8,32 \
  --chain-length 75 \
  --shape 2,6144 \
  --dtype bf16 \
  --dependent \
  --verify-hlo
```

Pipeline transport:

```bash
python scripts/greenfield/microbench_pipeline_transport.py \
  --plan PP8_LP4 \
  --iterations 2000 \
  --warmup 200 \
  --shape 1,6144 \
  --dtype bf16 \
  --verify-hlo
```

Checkpoint pack:

```bash
python scripts/greenfield/pack_checkpoint.py \
  --source zai-org/GLM-5.2-FP8 \
  --plan plans/pp8_lp4.json \
  --destination gs://driftbench-dsv4-uc/checkpoints/glm52/pp8_lp4/<version>
```

One-layer equivalence:

```bash
python scripts/greenfield/run_one_layer_equivalence.py \
  --plan PP8_LP4 \
  --captured-input <artifact> \
  --legacy-output <artifact> \
  --layer <layer_id>
```

Short decode:

```bash
python scripts/greenfield/run_short_decode.py \
  --plan PP8_LP4 \
  --context-length 8192 \
  --measured-tokens 256 \
  --trace-steps 20
```

Plan comparison:

```bash
python scripts/greenfield/compare_plans.py \
  --plans PP8_LP4,PP16_LP2,WS32_2D \
  --workload protected_e0
```

Use the repository’s existing run ownership, DB, archive, and cleanup wrappers rather than bypassing them.

---

# 12. Performance experiment matrix

## 12.1 Collective floor

Measure a genuinely dependent chain of 75 operations for:

- 2-way local group;
- 4-way local group;
- 8-way group;
- 32-way group.

Operations:

- all-reduce;
- reduce-scatter;
- all-gather;
- collective-permute;
- all-to-all where supported;
- fused tuple reduction;
- FP8/bf16/f32 variants where numerically relevant.

Shapes:

```text
bf16[1,6144]
bf16[2,6144]
bf16[1,2048]
f32[1,6144]
routing metadata payloads
```

Prevent compiler removal using:

- rank-dependent initial state;
- nonlinear dependency between iterations;
- optimization barriers where supported;
- output checksum;
- HLO count assertion of exactly 75 operations.

Report full distributions, not one mean.

## 12.2 Plan-level measurements

For each plan, record:

- total device ms/token;
- steady-wall tok/s;
- stage compute times;
- local collective time;
- inter-stage transport time;
- attention time;
- DSA scorer/top-k time;
- GMM time;
- movement/reshard time;
- idle/inactive device behavior;
- HBM peak per chip;
- physical collective counts/groups;
- compile time;
- executable size.

## 12.3 Decision criteria

Choose PP16_LP2 over PP8_LP4 only if its lower local-collective cost exceeds:

- eight additional stage boundaries;
- reduced aggregate active HBM bandwidth;
- any worse stage imbalance;
- any runtime-control overhead.

Choose WS32_2D only if it beats the best pipeline plan on protected wall latency and does not reintroduce global reconstruction.

---

# 13. Acceptance gates

## Gate A: topology and runtime skeleton

Pass when:

- physical topology is recorded;
- stage groups are explicit;
- PP8 and PP16 transport chains work;
- no host staging occurs;
- HLO matches intended point-to-point groups;
- inactive stages do not execute model-equivalent compute;
- cleanup and provenance work.

## Gate B: checkpoint packing

Pass when:

- every leaf maps to an explicit owner;
- byte totals reconcile;
- checksums round-trip;
- FP8 scales remain local to weight shards;
- load is direct to final layout;
- missing/corrupt shards refuse service.

## Gate C: one-layer exactness

Pass when representative dense, DSA, IndexShare, and MoE layers match the protected oracle under the documented numerical contract.

## Gate D: full short-context model

Pass when:

- the complete decoder generates correct tokens;
- DSA exactness passes;
- state/cache protections pass;
- no repeated 32-chip layer collective exists;
- fresh trace and steady wall are valid;
- memory headroom is measured.

## Gate E: useful architectural improvement

Minimum promotion gate:

```text
<= 200 ms/token
>= 4.5 profiler-free single-stream tok/s
```

This only proves the rewrite is useful.

## Gate F: strong base decoder

Target:

```text
<= 125 ms/token
>= 8 tok/s base
```

Stretch:

```text
<= 100 ms/token
>= 10 tok/s base
```

## Gate G: maximum-latency plan adjudication

Compare PP8_LP4, PP16_LP2, and WS32_2D under identical protected conditions.

Promote the fastest correct plan, even if it contradicts the initial recommendation.

## Gate H: effective interactive throughput

After the base decoder is stable, implement multi-token verification/speculation.

Report separately:

- base target-model ms/step;
- draft/MTP latency;
- proposed tokens;
- accepted tokens;
- mean accepted length;
- effective accepted tok/s;
- exactness/quality.

Do not label effective throughput as base decode throughput.

---

# 14. Speculative and pipeline-aware decode

A pipeline design leaves most stages idle during ordinary one-token traversal. After base correctness and latency are solved, use that structure to improve effective throughput.

Implement:

```python
verify_k_tokens(state, proposed_tokens)
```

The target model should verify multiple proposed tokens in one traversal where model semantics permit.

Research tracks:

1. Native GLM MTP heads if available and correct.
2. External draft model with exact speculative sampling.
3. Pipeline-aware speculation that keeps multiple stage groups busy.
4. Tree verification only after linear verification is correct.

Protected metrics must distinguish:

```text
target steps/second
accepted tokens/target step
effective accepted tokens/second
```

The 20–50 tok/s class is most credible as effective throughput after the base engine is substantially faster.

---

# 15. Coding rules

1. Python type annotations are required for new public APIs.
2. Configuration must be immutable after compilation begins.
3. Every execution plan must be serializable and hashable.
4. Every run must record the plan hash.
5. No implicit default mesh axes.
6. No combined logical axis names that obscure physical groups.
7. No silent fallback to global collectives.
8. No runtime checkpoint repartition of the full model.
9. No dead batch rows in batch-one decode.
10. No performance-specific code path without an exact fallback.
11. Every feature flag defaults off until promotion.
12. Every benchmark artifact is append-only.
13. Every TPU run uses authenticated ownership and cleanup.
14. Never modify historical result rows.
15. Fail loudly on stale code hash, wrong plan hash, or missing environment propagation.

Suggested environment variables:

```text
GLM_ENGINE=greenfield
GLM_EXECUTION_PLAN=PP8_LP4
GLM_EXPECT_CODE_HASH=<hash>
GLM_EXPECT_PLAN_HASH=<hash>
GLM_GREENFIELD_TRACE=1
GLM_GREENFIELD_LOG_STATS=1
GLM_GREENFIELD_DUMP_TOPK=1
GLM_GREENFIELD_HLO_CONTRACT=1
```

Do not overload old flags with changed semantics.

---

# 16. Commit plan

Use small, reviewable commits.

Suggested sequence:

1. `greenfield: add immutable model geometry and execution plan types`
2. `greenfield: discover and validate TPU physical topology`
3. `greenfield: add plan-aware memory and layer partitioner`
4. `greenfield: add HLO collective contract parser`
5. `greenfield: add dependent collective-chain benchmark`
6. `greenfield: add PP8 pipeline transport skeleton`
7. `greenfield: add PP16 pipeline transport skeleton`
8. `greenfield: add plan-aware checkpoint manifest and packer`
9. `greenfield: add direct final-layout streaming loader`
10. `greenfield: add reference dense and normalization kernels`
11. `greenfield: add exact router and reference MoE`
12. `greenfield: add exact DSA scorer/top-k path`
13. `greenfield: add stage-local KV and sparse attention`
14. `greenfield: add one-layer legacy equivalence harness`
15. `greenfield: compile complete short-context decoder`
16. `greenfield: add protected trace and wall benchmark integration`
17. `greenfield: add PP16 plan implementation`
18. `greenfield: add WS32 2D layout prototype`
19. `greenfield: adjudicate protected plans`
20. `greenfield: add multi-token verification path`

Each commit must keep relevant tests green.

---

# 17. Stop conditions

Stop and diagnose rather than continuing if any of the following occurs:

- a repeated layer collective includes all 32 chips;
- pipeline transfer stages through host memory;
- inactive pipeline stages execute substantial model compute;
- a new path requires the batch-32 decode bucket;
- DSA selected sets or tie order change;
- state manifest differs without an explained layout-only transformation;
- checkpoint packing cannot reconcile bytes;
- HBM safety margin is unknown;
- physical collective count differs from the declared contract;
- device latency improves but clean wall latency regresses;
- a profiler label improves while HLO physical synchronization worsens;
- the short-context model is still dominated by communication;
- the runtime cannot execute a stage chain without per-stage Python dispatch.

---

# 18. Definition of done

The rewrite branch is complete only when:

- GLM-5.2-FP8 serves at 256K on the existing 32-chip TPU v4 slice;
- the new engine is independent of the legacy model-execution path;
- a plan-aware final-layout checkpoint exists;
- exact DSA selected sets and tie order pass;
- raw tokens and quality tests pass;
- state/load/cache protections pass;
- repeated transformer-layer collectives are topology-local;
- no full-pod hidden reconstruction occurs inside the transformer;
- PP8_LP4 and PP16_LP2 have protected measurements;
- WS32_2D has either a protected measurement or a documented, evidence-backed rejection gate;
- device and profiler-free wall measurements agree;
- the four-depth 128K smoke passes;
- 256K E0 passes;
- every artifact is linked to `bench/results.db`;
- evidence is archived to the approved bucket;
- all eight hosts finish with authenticated zero-work cleanup;
- base and effective/speculative throughput are reported separately.

The final architecture is the fastest plan that satisfies all of these conditions.

---

# 19. External design references

These references provide mechanisms and context. They are not proof of local throughput.

1. Google Cloud, **TPU v4 architecture and topology**  
   https://docs.cloud.google.com/tpu/docs/v4

2. JAX documentation, **Manual parallelism with `shard_map`**  
   https://docs.jax.dev/en/latest/notebooks/shard_map.html

3. JAX documentation, **Explicit sharding and manual per-device programming**  
   https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html

4. JAX documentation, **Writing TPU kernels with Pallas**  
   https://docs.jax.dev/en/latest/pallas/tpu/details.html

5. JAX documentation, **Distributed computing in Pallas for TPUs**  
   https://docs.jax.dev/en/latest/pallas/tpu/distributed.html

6. Pope et al., **Efficiently Scaling Transformer Inference**  
   https://arxiv.org/abs/2211.05102

7. Jouppi et al., **TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings**  
   https://arxiv.org/abs/2304.01433

8. Barham et al., **Pathways: Asynchronous Distributed Dataflow for ML**  
   https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/

9. Leviathan et al., **Fast Inference from Transformers via Speculative Decoding**  
   https://research.google/pubs/fast-inference-from-transformers-via-speculative-execution/

10. Google AI Hypercomputer, **MaxText**  
    https://github.com/AI-Hypercomputer/maxtext

---

# 20. Final directive to Codex

Do not begin by porting the whole model.

Implement and prove these three things first:

```text
1. Explicit physical topology and local groups
2. A device-resident PP8 stage-transfer skeleton
3. A single exact MoE layer whose combine never leaves its four-chip stage
```

Then prove a complete short-context decoder.

Only after that should the branch load the complete 753B checkpoint at 256K.

The core optimization invariant is:

```text
Use pipeline depth to distribute weight capacity.
Keep repeated communication inside the smallest useful topology-local group.
Move only the live residual between stages.
Never reconstruct the hidden state across all 32 chips inside a transformer layer.
```

The initial recommendation is PP8_LP4, but evidence—not this document—must choose the final plan.

