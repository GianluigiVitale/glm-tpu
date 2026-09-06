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
- PP8_LP4 has a protected measurement; PP16_LP2 has a protected measurement or a documented,
  evidence-backed rejection gate (§22.3);
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


---

# 21. Amendment 2026-09-05 — Gate D correctness contract

This amendment is binding. Where it conflicts with the exactness wording of §1.3, §5.5, §8.2,
§11.1 (L6), §13 Gate D, §17 and §18, this section governs. Nothing else changes: plans,
topology-local collectives, default-off optimizations, protected measurement, provenance, cleanup and
all performance gates are unchanged. No historical evidence is reinterpreted or weakened, and this
section grants no install or execution authority.

## 21.1 Why

- The protected WS32 8K numerical runs `greenfield_ws32_short_decoder_8k_numerical_20260826T213125786075567Z`
  and `greenfield_ws32_short_decoder_8k_numerical_20260827T011711674195301Z` produced exact 20/20 raw
  tokens, valid state/cache, bitwise-exact event-0 DSA and measured HBM, and were refused only because
  DSA event 1 (layer 1) replaced seven of 2,048 selected positions relative to the legacy engine. No
  aligned score statistics were recorded for those runs.
- Gate C DB421 recorded that an independent raw PyTorch CPU scorer and the greenfield TPU FP32 scorer,
  both correct under the bounded contract (score max/mean error `0.003605/0.000965`), disagree on two
  of 2,048 cutoff members. Bit-identical selected sets across two correct implementations are
  therefore not a correctness property; they depend on rounding history. The legacy TPU engine was not
  a party to that comparison, so the legacy's own error against an FP32 reference has never been
  measured and must be measured under §21.2.
- The seven-swap event-1 signature is invariant across association changes: PP8 runs of 08-08, 08-09
  and 09-02 with different norm schedules record identical seven-position sets, the WS32 plan with a
  different sharding also refuses at event 1 with seven replaced positions (its records list counts,
  not positions), and one PP8 run whose layer-1 RMS input was certified legacy-exact still had six
  event-1 mismatches. This is evidence against attributing the swaps to reduction-association
  rounding alone and for a deterministic layer-1 arithmetic difference of unknown sign. §21.2 item 4
  exists to adjudicate exactly that; the swaps are not pre-declared benign or boundary-explained.
- Bit-exact reproduction of the legacy prefill's 32-way partial-sum and reduction-tree order for every
  projection in every layer is what "exact selected sets versus legacy" implies for an engine that
  intentionally changes association. From 2026-08-16 to 2026-09-04 it consumed more than ten protected
  tags with no decoder result. §5.5 already defines bounded internal error (level 2) and exact tokens
  (level 3); this amendment applies those levels to Gate D.

## 21.2 Gate D correctness contract

1. **Raw tokens (level 3, exact).** The protected N-token greedy continuation must equal the sealed
   legacy oracle tokens. Any divergence fails Gate D. The oracle's top-1/top-2 logit margin at the
   divergent step is recorded as diagnostic only.
2. **Within-engine DSA exactness (level 1, unchanged).** Distributed selection and lowest-position tie
   order must have zero mismatches against a canonical top-k of the engine's own executing score row,
   and that exact state must be what IndexShare and attention consume.
3. **Cross-oracle DSA agreement (level 2, adjudicated).** *(Item 3 and item 4 revised in place on
   2026-09-05 after the first offline adjudication; see §21.4.)* For every observed event, with the engine set
   `E`, the oracle set `O`, engine and oracle scores `s_e`, `s_o` over the same decode position and
   the same key positions, and cutoff scores `c_e`, `c_o` (the 2,048th score under lowest-position
   tie order):
   - `R` is an independent high-precision (FP64 CPU) reference of the event's score row computed
     from the sealed prompt/generated tokens and the checkpoint weights by code that shares nothing
     with either engine: a reference forward of every layer up to the event's producer, including
     the indexer inputs (normalized hidden, q-a, query, head weights, keys). It is validated against
     the legacy intermediate captures before use (it must reproduce the legacy row within the
     legacy's own rounding) and its implementation, dtype and source hash are recorded before the
     adjudication. A reference built from legacy *intermediate* captures measures the legacy scorer
     only and is a diagnostic, not `R`: it cannot separate legitimate input rounding from a defect
     (see §21.4).
   - Aligned positions `A = E ∩ O`. `eps_event = max_{p∈A} |s_o(p) − R(p)|`, the oracle's own
     demonstrated error against the math, which the engine cannot inflate. In addition the engine's
     error against the same `R` must satisfy `max_{p∈A} |s_e(p) − R(p)| ≤ κ · eps_event` and
     `std_{p∈A}(s_e − R) ≤ κ · std_{p∈A}(s_o − R)` with the pre-registered factor `κ = 2`: a BF16/FP8
     engine cannot be closer to FP64 math than its own rounding, so the cap is relative to the legacy's
     measured error against the same reference, never an absolute scorer-only constant.
   - *(Form revised 2026-09-05: the oracle archives scores only for its selected positions, so the
     band is defined on the reference row, which exists for every position.)* With `c_R` the
     reference cutoff (2,048th value of `R` under lowest-position ties), every position in the
     symmetric difference `E Δ O` must satisfy `|R(p) − c_R| ≤ eps_event`, and `|E Δ O|` must not
     exceed the number of positions with `|R(p) − c_R| ≤ eps_event` (the reference ambiguity band
     cannot explain more swaps than it holds).
   - Ties at either cutoff are resolved by lowest position on both sides before the comparison.
   - Any violation is a hard failure. `eps_event` and the cap are recorded per run, never hand-chosen.
   - *Scope (revised 2026-09-05):* items 3–4 adjudicate every event whose upstream selected state is
     identical to the oracle's, up to and including the first event where `E ≠ O`. Once a
     boundary-explained divergence exists, later events attend over legitimately different sets and
     cannot be compared to the legacy set-by-set; they are recorded (agreement statistics, cutoffs,
     within-engine exactness) but not adjudicated against the oracle. For every recorded event the
     run must publish `|E Δ O|` and the minimum reference-cutoff distance of the differing positions
     where a reference row exists; any later event with `|E Δ O| > 1024` raises a diagnostic alarm
     that requires an entry in `GATE_D_LESSONS.md` before the result is used for promotion (not a
     refusal). End-to-end correctness beyond the first divergent event rests on item 1 (exact raw
     tokens for all protected steps) and item 5.
4. **Systematic bias (hard).** Over `A`, `m_e = mean(s_e − R)`, `m_o = mean(s_o − R)`, `s` the
   sample standard deviation of `s_e − R`, `n = |A|`. Require `|m_e| ≤ κ · |m_o| + 3·s/√n` with the
   same pre-registered `κ = 2`. A larger bias is a hard failure that localizes a real arithmetic
   defect; it is never tolerated. `κ` was fixed on 2026-09-05 before any layer-1 result against the
   full-forward `R` existed; it may not be raised after a failed adjudication, and any change requires
   a new reviewed amendment. *Amendment 2026-09-05 22:20Z:* `s` is the POPULATION standard
   deviation (numpy `ddof=0`), which is what the sealed Gate D adjudication and §21.5 computed; the
   original wording said "sample". `ddof=0` yields a smaller `s` and therefore a strictly tighter
   bound, so no earlier result is weakened, and the difference at `n ≈ 2000` is 0.02%. The same `κ`
   scales the caps in item 3 and the `|m_o|` term here: a rerun at a smaller `κ` tightens all three
   tests together. `scripts/greenfield/adjudicate_ws32_first_divergent_event.py` implements this and
   reproduces the sealed numbers bit-for-bit. The FP64 reference row `R` must be a
   `docs/artifacts/gate-d-*.npy` artifact whose working-tree CONTENT is identical to the blob
   committed at `HEAD` (a tracked path can be overwritten in place, which is not the same thing); it
   must be the row this tool registers for the exact event `(context, decode position, producer
   layer, norm-eps convention)` being adjudicated, and that registration must name the reviewed
   validation record that checked `R` against the legacy intermediate captures per item 3 above. Its
   digest is declared on the command line, its convention is declared and carried into the record
   (§21.5 records two conventions whose rows are indistinguishable by shape), and the reference
   implementation's committed tree hash is recorded. A new event therefore requires a new REVIEWED
   registration, which is the pre-registration step. Every earlier adjudication attempt on the same
   event is disclosed in the record's `prior_attempts`; `basis` names only what the record stands on,
   and a refused analysis may never appear there. Disclosure is best-effort by construction: attempt
   files are untracked outputs, so an operator who moves one out of `docs/artifacts` leaves no trace.
   What is enforced is that the key is always present, so "no earlier attempt" is asserted rather
   than omitted, and that every declared attempt is SHA-bound. The record must name, in `analysis`,
   the PASS §21.2 analysis it stands on, and that analysis must adjudicate the same run, step and
   event and declare the same reference row. The later-event alarm threshold is fixed at 1024 and a
   record that carries any other value is refused: it is not an operator choice. The observer archive
   is bound to the archive the declared source run itself published under its own tag, not to a
   directory name. Every one of these rules is enforced by the LOADER the sealer calls, not by the
   offline tool: a rule only the producer consults can be widened in a working tree, used once and
   reverted without leaving a trace in the record.

   *Pre-registration is proven, not assumed.* The sealer requires the record, the analysis it stands
   on AND the FP64 reference row to be committed in the run's OWN pin (`--code-hash`),
   byte-identical to the blob on disk. It then RE-DERIVES items 3-4 from the run's own observations,
   the sealed oracle and that row, and refuses unless the re-derivation passes and reproduces the
   pre-registered divergence: a verdict the sealer merely reads is a claim by whoever wrote the
   file, so the record's job is to fix the row and the expected divergence in advance, never to
   supply the answer. The run executed at that commit, so an artifact present in it with these bytes existed
   before the run produced the data it judges; a record written to fit an observed divergence cannot
   satisfy this, and checking `HEAD` would not do it because HEAD moves after the run. The sealed
   Gate D record already satisfies it: its blob at pin `4286509` is the blob on disk.

   *Grandfathering, stated rather than hidden:* the one record that closed Gate D,
   `docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json`, SHA-256
   `4da05468120e3c2e9b82d03931018e0d14eebc5fc28e339381658a04457cd26b`, predates this amendment and
   carries neither `reference_row` nor `analysis`. It is exempt BY DIGEST, so the exemption covers
   exactly that file and cannot be transferred: `expected_sha256` is verified first, and the
   exemption is then keyed on that verified digest. It exempts the record from CARRYING the
   bindings, never from the re-derivation: the loader resolves its reference row from the single
   `.npy` in its own basis, which is unambiguous and stays correct when §21.5's second norm-eps
   convention row is registered for the same event. Verified: re-deriving event 1 from the Gate D
   run's own arrays passes and reproduces the sealed numbers.

   *What the re-derivation does and does not establish.* It removes the operator's verdict from the
   chain: the six checks are computed from the run's own observations, the sealed oracle and the
   pre-registered row. It does not validate the ROW against the checkpoint or the tokens. An
   operator who commits a fitted row together with its `REFERENCE_ROWS` entry into the run's own pin
   would pass every mechanical check, because a row chosen as `R ≈ s_e` drives the engine deltas to
   zero and inflates `eps_event`. The soundness of items 3-4 therefore still rests on a human having
   reviewed that registry entry and the record that validated the row against the legacy captures —
   which is what the registry exists to force into review, and why a new event requires a new
   reviewed entry rather than a new file. Because the registry and the §21.2 arithmetic are read
   from this repository's source at seal time, an adjudicated seal additionally refuses to run from
   a modified enforcement surface (the sealer, `glm_tpu/greenfield/{validation,benchmarking,sharding,
   runtime,kernels/reference}`, `glm_tpu/greenfield/types.py`, the model config and
   `docs/artifacts/`): an edit that widens what is accepted must
   be committed, and therefore reviewable, rather than made and reverted around a seal. The refusal
   also covers `--assume-unchanged` and `--skip-worktree`, which hide an edit from `git status`
   without committing anything, and the sealer puts its own repository first on `sys.path` and
   refuses to run if the §21.2 modules resolve outside it. This check runs BEFORE the record's
   schema, basis and source-run checks. The run's pin must also be contained in the
   published reviewed branch, and the enforcement surface must be the surface committed at that pin
   (or at a declared recovery pin), so a scratch branch carrying a widened registry cannot seal even
   with a clean tree. Workers 1-7 fetch the pin from `origin` and check it out, so a pin that is not
   published cannot have run at all; the controller refreshes its remote-tracking ref before sealing.
   *What this is not:* a remote-tracking ref is an ordinary local ref and an operator with shell
   access can write one, so this raises the bar from "edit a file" to "forge a published branch or
   collude with the reviewer". It is not a cryptographic guarantee and must not be described as one. The digest comparison is made after the schema, basis, oracle
   and source-run checks, all of which the record satisfies, so any future tightening of those must
   be checked against it explicitly. The exempt record provably carries neither binding, because the
   exemption is keyed on its content digest. History is not weakened and no new record may use it.
5. **Internal tensors (level 2).** Layer outputs, residuals and caches are compared under the bounded
   contracts in `docs/greenfield/NUMERICAL_CONTRACT.md`; cache/state structure (positions, tails,
   validity, pages, manifests) remains exact.
6. **Unchanged:** no repeated 32-chip layer collective, fresh eight-host trace, profiler-free steady
   wall, measured HBM headroom, provenance, DB linkage, archive and authenticated 8/8 zero-work cleanup.

## 21.3 Consequences

- Bit-exact reproduction of legacy prefill or decode intermediate arithmetic is not a Gate D
  requirement and not a prerequisite for a protected 8K run.
- The exact-M2048 StrategyND association fingerprint and any legacy reduction-tree emulation are
  optional research, default deferred. Their code, evidence and the V11 validator correction remain
  committed history.
- The standing prohibition "no complete WS32 rerun without a bounded layer-1 state-boundary proof"
  is satisfied only by a passing offline §21.2 items 3–4 adjudication of the archived
  `…20260827T011711674195301Z` event-1 arrays. A failing adjudication keeps the prohibition and
  redirects work to localizing the layer-1 defect by chunk/row probes, not 8K runs.
- Feasibility must be confirmed before promising zero TPU time: the engine event-1 score row (WS32
  NPZ `2be686ff…eeb1`), the oracle event-1 score row, the sealed legacy layer-1 cache
  (`d9058cc6…`, 8,155 rows) and the FP32 query at position 8155 (oracle `79b813da…`) must all be
  archived and hash-verified. If any is missing, one bounded capture run (not a decoder run) is the
  next step.
- §17 "DSA selected sets or tie order change" is read as "fail §21.2 items 2–4". §1.3, §11.1 L6 and
  §18 "exact DSA selected sets and tie order" are read as "§21.2 items 2–4 pass".

## 21.4 First offline adjudication — 2026-09-05 (diagnostic, CPU only)

Record: `docs/artifacts/gate-d-event1-offline-adjudication-20260905.json`. Inputs: WS32
`…20260827T011711674195301Z` rank-0 NPZ (`2be686ff…eeb1`), the sealed 8K DSA oracle whose manifest
self-hash `f8154c5f…` the run recorded, the legacy layer-1 internals at position 8155 and the legacy
layer-1 BF16 prompt index cache (8,155 rows; file `afe683d8…`, key tensor `8d656d71…`). Findings:

- Scorer identification: with the FP32 query and the FP32 current key both rounded to BF16 (as the
  cache keys already are), an FP64 reference from the legacy inputs reproduces the oracle's event-1
  top-2,048 set with zero swaps and the exact cutoff `80.49651`; aligned oracle−reference error is
  max `1.9e-5`, mean `−1.3e-6`, std `5.7e-6`, consistent with FP32 accumulation. The legacy scorer
  is therefore exact given its inputs; a scorer-only ambiguity band holds one position.
- The engine's aligned scores are the oracle's shifted by mean `−0.118`, std `0.022`
  (fit `s_e = 0.999656·s_o − 0.0897`). All fourteen swapped positions lie within `0.036` of a
  cutoff and far outside any scorer-only band. The engine scorer is exact at event 0, so the
  deviation originates entirely in the engine's layer-1 inputs (layer-0 output → layer-1 norm →
  q-a/query/head weights/keys), a ≈0.15% effect.
- Convention validation of the reference forward against the exact legacy event-0 set from
  embeddings only: interleaved indexer RoPE (config `indexer_rope_interleave`) gives 4 boundary swaps
  with oracle−reference aligned error mean `+0.0119`, std `0.0053`, max `0.030`; the half-split
  layout gives 296 swaps and is rejected. The legacy's layer-0 deviation from the reference is
  `+0.012` under the q-a/kv-a eps `1e-5` convention and `−0.006` under `1e-6` (§21.5); this is the
  scale item 3's relative cap must be measured against.
- Not decided: whether the engine's layer-1 input deviation is legitimate rounding of a differently
  associated layer 0 or a defect. A legacy-input reference cannot decide it. The decisive next step is
  the independent FP64 reference forward of §21.2 item 3 (layer 0 plus layer-1 indexer inputs from the
  archived token ids `d860b7f4…` — never re-tokenized — and the checkpoint weights), validated against
  (a) the exact legacy event-0 set at 8155, (b) the legacy layer-0 index-cache dumps, and (c) the
  legacy layer-1 internals at 8155 and layer-1 cache, after which the legacy's and the engine's
  layer-1 deviations and event-1 rows are both measured against it. No TPU time is required.

## 21.5 Math-reference adjudication of event 1 — 2026-09-05 (diagnostic, CPU only)

Record: `docs/artifacts/gate-d-event1-math-reference-adjudication-20260905.json`; reference code
`scripts/greenfield/reference_cpu/`; reference row `docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy`.
The reference follows the HF GlmMoeDsa definition with two vLLM/legacy conventions where HF differs
(interleaved indexer RoPE; config `rms_norm_eps = 1e-5` for the q-a/kv-a norms, HF default `1e-6`).
Both engines under test share those conventions. Independently reviewed (Fable, verdict
`m2048-v11-fable-ref-verdict.txt`): code and numbers reproduce; structure validated to the noise
floor (query rebuilt from the legacy's own q-a state through the reference `wq_b`+RoPE has slope
`0.999999`, head weights `1.000000`, keys `1.000021`).

- Validation against the legacy layer-1 captures: relative RMS deviation 0.04–0.38% (BF16 noise
  floor). Least-squares slopes legacy~reference are `1.000366` normalized hidden, `1.000549` q-a,
  `1.000695` query, `1.000339` head weights, `1.000021` keys: a systematic +0.04–0.07% legacy excess
  downstream of the two RMSNorms whose mechanism is not identified (the legacy's captured q-a state
  fits `eps ≈ 1e-6` better than `1e-5`; the residual stream mean-square is ≈2e-5, so eps is a
  first-order term in this model).
- Event 1, `eps = 1e-5` convention: legacy row mean `+0.1118`, std `0.0310`, max `0.2270`; engine
  row mean `−0.0062`, std `0.0310`, max `0.1135`. `eps_event = 0.227`; reference cutoff
  `80.38919`, band 451; reference set differs from the oracle by 10 and from the engine by 13
  positions; all fourteen swapped positions lie inside the band.
- Event 1, `eps = 1e-6` convention (reviewer rerun): reference row shifts `+0.0505`; legacy
  `+0.0612 / 0.0311 / 0.1773`; engine `−0.0568 / 0.0311 / 0.1561`. Layer-0 legacy bias vs the
  reference is `+0.012` under `1e-5` and `−0.006` under `1e-6`.
- Items 3 and 4 pass for the engine at event 1 under both conventions, at `κ = 2` and at `κ = 1`.
  No convention was found under which the engine fails or under which the legacy's absolute bias
  falls below the engine's. The sign and size of the two biases are convention-dependent, so no
  claim is made that either engine is "more accurate"; the claim is only that the engine's event-1
  deviation from the model is no larger than the legacy's, and the seven swaps are boundary noise
  within the legacy's own error against the model. The layer-1 "defect" hypothesis is not supported.
- Decomposition (diagnostic): legacy inputs through an exact scorer `+0.189`, BF16 scorer rounding
  `−0.077`, net `+0.112`; the engine's input contribution (`≈ +0.055`) is inferred from a
  bitwise-identical scorer, not measured.
- Consequence, effective only once this record is reviewed and persisted: the standing prohibition
  on a complete WS32 rerun is replaced by authorization for one protected 8K run whose observer
  implements §21.2 (refuse only on token mismatch, within-engine inexactness, cache/state structure,
  locality, or a first-divergent-event divergence not equal to the pre-registered adjudicated one;
  record later events with the alarm rule) with full trace/wall/HBM/DB/archive/cleanup protections.
  Its success closes Gate D.

## 21.6 Gate D closed — 2026-09-05 (protected 8K WS32 run sealed under §21.2)

- Run `greenfield_ws32_short_decoder_8k_numerical_20260905T085534575653049Z` (code pin `4286509…`, recovery/sealing pin `579b13f…`, WS32_2D on 8 hosts × 4
  chips, exact-DSA path, StrategyND dense overlay ON (overlay manifest `a8dc8791…4b6a`, manifest file
  `c17194b6…5c8c`, overlay SUCCESS `166566b9…32a6`), streaming tmpfs checkpoint transport, adjudication
  record `4da05468…cd26b`) is sealed: remote `SUCCESS` (`e40760f8…f662`, generation 1788607616743487),
  summary `683fe2e1…08fa`, source ledger `c533af64…c0c8` (128 generation/CRC/SHA-bound fleet objects),
  protected DB run 567 (`greenfield_78layer_8k_ws32`, item
  `gate_d_s21_exact_tokens_adjudicated_dsa_state_cache`, record `2c9b9d93…2295`), censuses pre,
  recovery-pre and post 8/8 zero-work.
- Contract items: (1) raw tokens exact over the sealed oracle prefix, 20 of 29 generated (`909682cb…8173`); (2) within-engine DSA order/ties
  exact at all 14 observed steps; (3–4) event 0 exact, event 1 equal to the pre-registered adjudicated
  divergence (§21.5), later events recorded with the alarm acknowledged against the committed profile
  and a lessons entry (maximum 1948/2048 at step 2, event 17, layer 62); (5) satisfied BY INHERITANCE
  from the Gate C bounded one-layer contracts (exact dense, full-DSA, IndexShare, adversarial MoE
  equivalence against captured legacy inputs): no per-layer tensor of this 78-layer run was compared
  against a reference, which the classification states as `DEEP_LAYER_TENSORS_NOT_BOUNDED_IN_THIS_RUN`;
  (6) cache/state structure exact, no repeated 32-chip layer collective, fresh 8-file/64-core XPlane at
  2 traced steps, profiler-free p50 `130.369 ms/token` (p99 `132.414`, 7.67 tok/s), peak HBM
  `26,375,554,560` bytes/chip (headroom 6.64 GB). Gate E met; Gate F not.
- Sealing history (`recovery_failures/`, five directories): the original 09:37Z seal refused by design
  on the §21.2 later-event alarm (acknowledgement machinery then added under review); recovery attempt 1
  (09:55Z) refused on a stale source ledger bound to the earlier recovery pin; recovery attempt 2
  (10:35Z) materialized, then refused in the XPlane aggregation because the module regex named the
  non-exact decode body; recovery attempt 3 (11:14Z) sealed. A latent DB-rollback identity defect was
  found by tests during review, never by a refusal. None of these changed the run's evidence; fixes are
  in `HANDOFF.md` and `GATE_D_LESSONS.md`.
- Classification: `RAW_TOKENS_EXACT;DSA_WITHIN_ENGINE_EXACT;STATE_CACHE_EXACT_STRUCTURE;DSA_EVENT0_EXACT;
  DSA_EVENT1_ADJUDICATED_S21_2;LATER_EVENTS_RECORDED_NOT_ADJUDICATED;DEEP_LAYER_TENSORS_NOT_BOUNDED_IN_THIS_RUN;
  LATER_EVENT_ALARM_ACKNOWLEDGED_WITH_LESSONS_ENTRY;PROTECTED_WALL_TRACE_HBM`. Gate D is closed under
  this contract with item 5 by inheritance; Gates G (plan adjudication), long contexts and §18 follow.

# 22. Gate G adjudication — 2026-09-05 (offline, from protected evidence)

## 22.1 What Gate G asks and what exists

Section 13 asks for PP8_LP4, PP16_LP2 and WS32_2D compared under identical protected conditions and
the fastest correct plan promoted. Protected complete-decoder evidence exists for two of the three:

| Plan | Context | DB run | Tag | Correctness | Profiler-free p50 | Peak HBM/chip |
|---|---|---|---|---|---|---|
| PP8_LP4 | 2K | 563 | `greenfield_short_decoder_compile_pp8_2k_…_20260827T194450473110366Z` (sealed by recovery `…_pp8_2k_gate_d_recovery_20260827T203000000000000Z`, `success_sha256` `95a18e10…e8ff`) | exact tokens; 14 steps × 21 DSA events exact | `245.639880 ms` (4.071 tok/s) | 26,303,084,032 B |
| WS32_2D | 2K | 553 | `greenfield_ws32_short_decoder_2k_numerical_20260816T040707909547486Z` (`success_sha256` `a943497d…d6a9`; SUCCESS file SHA `c16a491d…6ad1f`) | exact tokens; exact DSA/state/cache | `122.630667 ms` (8.155 tok/s) | 24,789,135,872 B |
| WS32_2D | 8K | 567 | `greenfield_ws32_short_decoder_8k_numerical_20260905T085534575653049Z` (`success_sha256` `e40760f8…f662`) | §21.6 | `130.368724 ms` (7.671 tok/s) | 26,375,554,560 B |
| PP8_LP4 | 8K | — | 2026-09-02 attempts: two executions refused at DSA event 1 with a seven-swap divergence of the same size as WS32's (not adjudicated under §21), one refused from event 0, the others failed before execution (HLO/prefill contracts, sync faults); no wall recorded | — | — | — |
| PP16_LP2 | any | — | no 78-layer decoder was ever built; only stage-zero Gate D diagnostics (compensated, forced-round, projection-contraction), all rejected; one-layer derivatives DB558 | — | — | — |

The 2K rows are an identical-condition protected comparison: both runs execute the same sealed 2K
prompt against the same token oracle (`f580c149…efe19`) and DSA oracle (`71224832…4f57`), on the same
pod, with fresh eight-host XPlanes, host-side profiler-free wall (2 warm-up + 10 timed samples, fleet p50;
PP8's complete-step wall includes the token return), and 8/8 cleanup, and both are
exact; both prompts are 2,034 tokens. WS32_2D is `2.003×` faster than PP8_LP4 (`245.640 / 122.631`).
Residual differences, disclosed: WS32 2K allocated `context_capacity 8192` (conservative for WS32);
WS32 2K ran without the exact-DSA and dense-overlay variants that WS32 8K used; DB563 is the 2026-08-27
PP8 configuration and the later 09-02 PP8 variants never recorded a wall (kernel gains could only touch
the ≈40 ms of non-permute busy time, so no PP8 variant can close a 2× gap). At 8K only WS32_2D has a
protected measurement; it is within 6.3% of WS32's own 2K figure across those variants.

## 22.2 Why PP8 is slow, from its own protected trace

DB563's 64-core XPlane attributes `206.353 ms` of the `245.640 ms` step to 17 pipeline
collective-permutes (eight split-residual transfers, eight `s32[1,2053]` IndexShare/metadata transfers,
one token return); local all-reduces total `0.464 ms`. The protected boundary benchmarks (DB449, DB564)
measure a single stage transfer at `0.28–0.32 ms` and the production two-transfer boundary at p50
`≈0.41 ms`, so the 206 ms is not wire time: it is each core waiting while the other seven stages run. With one live sequence a pipeline executes its stages
strictly in series, so the step time is the sum of the per-stage times plus boundary transfers, and each
stage's ~10 layers run on only 4 of the 32 chips. WS32_2D keeps all 32 chips busy on every layer with
topology-local collectives, which is where the 2× comes from.

## 22.3 PP16_LP2: documented, evidence-backed rejection for the latency objective

PP16_LP2 halves the chips per stage (2) and doubles the stage count (16). Under the same one-sequence
serial-stage law: (a) the summed stage time cannot fall below PP8's, because the same 78 layers are
executed by 2 chips at a time instead of 4 — a bandwidth- or compute-bound stage on half the chips takes
at least as long, and in the memory-bound decode regime about twice as long; (b) boundary transfers
double (about 33 permutes; at the measured `≈0.3–0.4 ms` each this is `10–14 ms`, small but additive).
The per-stage slowdown is measured, not modeled: the protected real layer-3 rows give PP16_LP2 (2 chips)
`3.835–3.873 ms` per layer (DB558, DB562) against PP8_LP4 (4 chips) `2.165–2.197 ms` (DB467/471/473/482),
a `1.75×` ratio for the same MoE layer (caveats: layer 3 is one MoE layer, the three dense layers differ,
and DB482 carries `reconstruct_down_fp32=True` while the PP16 rows do not — a small extra cost on the PP8 side).
Bounds: with zero per-stage slowdown `≥ 245.6 ms` (no better than PP8, `≥ 1.88×` slower than WS32 8K);
with the measured layer ratio `78 × 3.84 ms ≈ 300 ms` of stage compute alone, before transfers and the
non-layer work, i.e. `≥ 2.3×` slower than WS32 8K. The latency leg is therefore anchored on protected
measurements and is independent of memory. PP16's only structural advantage is per-chip memory (2× the
weight capacity per stage pair). The memory leg is provisional: WS32_2D holds the full 753B FP8 model
with `6,638,844,416 B` per-chip headroom at 8K (DB567); its decode caches total `3.45 GB` per chip at
256K (KV `78 × 512 pages × 64 rows × 640 × bf16 = 3.27 GB`, IndexShare `21 full layers × 512 × 64 × 128 ×
bf16 = 0.18 GB`), a `3.34 GB` growth from 8K, which the headroom covers; prefill/activation buffers at
256K are not yet measured and are what L7/L8 must confirm. Building a PP16 78-layer decoder to measure a
plan whose latency the measurements already bound below PP8's would consume weeks of TPU-protected work
for a result that cannot change the promotion. PP16_LP2 is therefore
REJECTED for the single-stream latency objective with this evidence. This is the same form of closure
that §18 already allows for WS32_2D ("a documented, evidence-backed rejection gate"); §18 is amended below
to allow it for PP16_LP2 symmetrically. The rejection is reopened only if a protected WS32_2D
long-context gate fails for a reason that PP16 would cure (memory), which §22.3's numbers make unlikely.

## 22.4 Promotion and consequences

- WS32_2D is the promoted plan: the fastest plan with protected exact evidence at both 2K and 8K.
- §18 line "PP8_LP4 and PP16_LP2 have protected measurements" becomes "PP8_LP4 has a protected
  measurement; PP16_LP2 has a protected measurement or a documented, evidence-backed rejection gate
  (§22.3)". PLAN item 6 is closed accordingly.
- PP8_LP4 keeps its protected 2K measurement (DB563) and its Gate D role as the first exact 78-layer
  path; no further PP8 8K launches are authorized (the §21 contract would adjudicate its event-1 swaps
  exactly as WS32's, but a protected PP8 8K wall cannot change the promotion).
- Gate E (`≤ 200 ms`, `≥ 4.5 tok/s`) is met by WS32_2D at 2K and 8K only; PP8_LP4 at 2K (`245.6 ms`,
  `4.071 tok/s`) meets neither E nor F. Gate F (`≤ 125 ms`, `≥ 8 tok/s`) is met by WS32 at 2K (`122.63`,
  `8.155`) and not at 8K (`130.37`, `7.671`). Optimization toward F at 8K stays default-off and behind the
  long-context gates.
- Next: L7 (128K four-depth smoke) and L8 (256K E0) on WS32_2D, then §18.

# 23. Long-context gates L7/L8 on WS32_2D — design (2026-09-05, reviewed: v3 APPROVE, rotary addendum v3.3 APPROVE-WITH-P2 folded)

## 23.1 Workloads (reproduced offline, provenance-bound; no new legacy capture)
- L7: the four legacy protected passkey prompts of DB run 403 (depths 0.0/0.05/0.95/1.0, seeds
  16780345/16781195/16796495/16797345, 127,363 prompt ids each incl. prefix [154822,154824], gold
  705269/824794/289958/891482, legacy 4/4 correct, legacy latency ≈600 s/item). Rebuilt with
  `bench/glm_longctx.build_trial(tok, 128000, depth, seed)` + GLM tokenizer; verified 2026-09-05: keys ==
  gold, id counts == DB, and the rebuilt text SHA-256 == the sha256 embedded in the DB's stored
  head+tail prompt field for all four (byte-identical prompts). Sealed by a new CPU-only variant of
  `capture_short_context_oracle` that rebuilds the prompt from the seed (the stored field is truncated at
  65,536 chars), checks the embedded sha256, prepends the two special ids and records that the legacy
  tokenized server-side; manifest/SUCCESS format unchanged; legacy `raw_output` (20 tokens) kept as a
  diagnostic reference, not as the pass criterion.
- L8: the legacy E0 prompt of DB run 402 (`dsa_throughput.build_prompt_ids/v1`: prefix + RandomState(
  34353209).randint(256, 154820, 262142) → 262,144 ids; stored sha256 of the int32 stream; measure
  window 256 tokens; legacy 1.263 tok/s = 791.83 ms/step, t_prefill 1337.7 s). Sealed the same way.

## 23.2 Prefill: chunked exact teacher-forcing (required by memory), sealed semantics preserved
The exact-DSA prefill scans one decode step per prompt token and stacks the per-token full-indexer
normalized inputs `[L, 21, 1536]` bf16 per chip for ONE post-scan M64 index repair: 0.53 GB at 8K,
8.2 GB at 128K, 16.9 GB at 256K — it cannot fit at long context (8K headroom 6.64 GB). The sealed
semantics are: every prompt step scores the UNREPAIRED on-device index rows of all earlier positions;
the repaired rows are visible only to decode. A chunked program must keep exactly that, so:
- The chunk program takes a static chunk of C prompt tokens, scans them against the carried state
  (whose index cache stays unrepaired, as today), and writes the repaired rows for exactly those C
  positions into a SEPARATE repaired-index buffer (same shape as the index cache, zeros initially,
  0.18 GB per chip at 256K) using the repair kernel's `position_offset` (an int32 scalar, traced) and
  static `valid_rows`. The carried state is unchanged in kind; nothing scanned ever sees a repaired row.
- After the last chunk the runner installs the repaired buffer as the state's index cache and frees
  the unrepaired one — bit-for-bit what the monolithic program produced (it overwrote exactly the
  prompt rows of an otherwise-zero cache). Numerics are therefore independent of C by construction.
- Two programs, always both: `prefill_chunk` (length C) for the first ⌊(L−1)/C⌋ chunks and
  `prefill_tail` (length L − ⌊(L−1)/C⌋·C ∈ [1, C]) for the last one, so the graph set is fixed for
  every workload and no padding token is ever teacher-forced (8K: 3×2048 + 2011; 128K: 62×2048 + 387;
  256K: 127×2048 + 2048).
- Kernel API change (reviewed, tested): `repair_stage_local_prompt_index_cache` accepts a traced
  int32 `position_offset` (static `valid_rows` unchanged); the linter's `kind="prefill"` expectations
  (21 repair gathers, split-RMSNorm counts) apply to both programs and are verified on the tail graph.
- Host loop: after each chunk the runner logs wall, projects total prefill time and aborts fail-closed
  if the projection exceeds the worker budget minus margin; prefill wall is recorded separately.
- Graph schema: `BASE_GRAPHS` becomes (`prefill_chunk`, `prefill_tail`, `observer`, `decode`,
  `cache_probe`); wrapper upload list, acquisition pins (`--expected-prefill-chunk-*`, `--expected-
  prefill-tail-*`), sealer expectations and evidence materializer updated together; 2k/8k acquisitions
  re-pinned (the sealed DB553/DB567 records are untouched history).
- `prefill_chunk` in the §5.2 sense (multi-row) remains the first post-DoD item; per-layer fixed
  cost dominates the 130 ms step (45 ms collectives, 44 ms weight-format custom calls, 11 ms
  gathers), so a multi-row body would amortize nearly linearly — that is what makes 256K
  *interactive*; it is not required by L7/L8 and would delay the only gates that can still fail.

## 23.3 Proving the chunked prefill before using it (identity, not just contract pass)
- Step A0 is superseded by §23.8: the indexer rotary stays on device (legacy-faithful, tombstoned
  host table); the main-attention long-position effect is measured by a zero-cost diagnostic inside
  Step B's first worker with a pre-registered per-(pair, band) decision rule, and only a failing rule
  triggers the main BF16 table path (B′) described there.
- Step B (8K identity): two protected 8K numerical runs at capacity 8192 with the chunked prefill at
  C = 2048 and C = 512 (acquisition first for each). Each must pass the unchanged §21 sealer AND
  reproduce DB567's bit-level witnesses: `numerical_array_manifest_sha256` `057af89f…caac` (all 14×21
  DSA observations), cache probe `kv_rows_sha256` `67d03f75…` / `index_rows_sha256` `a8724ce5…`, token
  sha `909682cb…8173`. Identity at two chunk sizes proves C-independence and equivalence to the sealed
  program; the chunked prefill is then accepted by a §23 equivalence record, not a new gate.
- Step C (capacity pre-measurement): two protected runs with the 8K prompt at capacities 131,072
  (256 pages) and 262,656 (513 pages) measure per-step wall and memory at long capacity (local top-k
  and page-table gathers scale with capacity); tokens and the DB567 witnesses must stay identical
  (capacity must not change numerics). Wrapper gets a capacity override; classification
  `CAPACITY_MEASUREMENT` so these are never read as Gate D records. Prefill hours, worker timeouts
  and the projection margin for L7/L8 are derived from these measurements.

### 23.3.1 Equivalence record — chunked exact prefill (2026-09-05)

Two protected 8K runs with the chunked prefill reproduced every bit-level witness of the sealed Gate D
run DB567 (`…8k_numerical_20260905T085534575653049Z`): numerical array manifest `057af89f…caac` (all
14 × 21 DSA observations, scores and sets), cache probe `kv_rows` `67d03f75…` / `index_rows` `a8724ce5…`,
raw tokens `909682cb…8173`, all `dsa_steps` comparisons and the final state (position 8183, context 8184):

| run | pin | chunk plan | prefill wall | fleet p50 | sealed as |
|---|---|---|---|---|---|
| `…8k_numerical_20260905T163437339905065Z` | `172ae03` | 3 × 2048 + 2011 | 1,004 s | 129.73 ms | DB 568, `success_sha256` `066de64b…` |
| `…8k_numerical_c512_20260905T182725949766820Z` | `00c6eee` | 15 × 512 + 475 | 988 s | 130.05 ms | DB 569, `success_sha256` `1d68c6be…` |

Both were sealed by recovery at pin `b51c626` (alarm acknowledgement bound to the lessons entries naming
each tag) with the §21 classification of DB567.

Identity at two chunk sizes proves C-independence; the chunked program is the sealed program in pieces
and is accepted on this record, not by a new gate. The five chunk-independent graphs of both acquisitions
are byte-identical to DB567's in StableHLO (and, in the C = 512 acquisition, in optimized HLO too);
the recompiled optimized texts of the first differ only by instruction names, channel/stack-frame ids and
metadata (normalized instruction multisets identical). Teacher-forced prefill cost at 8K capacity:
≈121–123 ms per prompt token. Each chunk length is a distinct graph pair and needs its own acquisition.

## 23.4 Capacities and memory
- 128K: prompt 127,363 + observer 14 + warmup 2 + iterations 10 + trace 2 + 1 = 127,392 ≤ 131,072 ✓.
- 256K: prompt 262,144 + observer 14 + warmup 2 + iterations 256 + trace 2 + 1 = 262,419 → capacity
  262,656 (513 pages) (v1's 512 pages was wrong and would have refused after the acquisition).
- Per-chip decode caches (bf16): 8K 0.108 GB, 128K 1.72 GB, 513 pages 3.46 GB; 8K peak 26.38 GB of
  33.0. Chunked prefill temp at C = 2048 ≈ 2.0 GB regardless of L (2.37 − 0.53 + 0.13). Decode at 513
  pages ≈ 29.1 GB of arguments plus temps → roughly 1.5 GB margin; the acquisition gates on compiled
  memory before any numerical run.

## 23.5 Correctness contracts (no legacy DSA oracle exists at these lengths; none is captured)
- L7 (`--long-context passkey`): pass iff `extract_passkey(detok(first 20 greedy tokens)) == gold` for
  each prompt, four of four runs; plus within-engine exact DSA order/ties at every observed step,
  cache/state structure, locality/HLO contracts, fresh trace, profiler-free wall, HBM, DB, archive,
  8/8 cleanup. The 20 tokens are compared with the legacy `raw_output` ids and recorded exact/inexact
  as a diagnostic; nothing may be labelled "raw tokens exact". Classification
  `PASSKEY_EXACT;DSA_WITHIN_ENGINE_EXACT;NO_CROSS_ORACLE`; item id `l7_128k_passkey_d<depth>`; DB rows
  link legacy run 403.
- L8 (`--long-context e0`): 256-step profiler-free decode window after the 262,144-token prefill,
  preceded by the harness's 18 observer/warm-up/trace steps (legacy warm-up was 0; both recorded), with
  fresh 8-file/64-core XPlane, p50/p99, HBM, generated ids recorded; no correctness oracle (legacy E0
  had none) → `NO_CORRECTNESS_ORACLE`; within-engine DSA exactness and state/cache contracts still
  enforced. Legacy vs WS32 prefill wall and ms/step recorded side by side. Item id `l8_256k_e0`.
- Sealer mirrors both modes; the §21 token/DSA-oracle contract remains the only path for 2k/8k.

## 23.6 Multi-hour protected runs (reviewer Q3)
- Worker processes detached (`setsid nohup … timeout --kill-after`), runner process name unchanged so
  the census pattern `[r]un_short_decoder_ws32[.]py` still sees them; per-rank completion marker files
  `WS32_LONG_DONE tag rank status` written only after the EXIT-trap upload completes; the orchestrator
  polls markers over fresh ssh sessions.
- Attach/resume mode keyed by TAG: re-acquires the controller lease and re-polls markers, so an
  orchestrator crash cannot orphan a run; on orchestrator failure workers are never blindly killed and
  never abandoned — either attach, or `pkill` by tag followed by an 8/8 census proof.
- The failure-exit census reporting BUSY while own workers run is expected and recorded as such.
- Worker timeout = measured projection × 1.5 (set per context from step C), hard bound via `timeout`.

## 23.7 Sequence and honesty
A (CPU, reviewed): dual-buffer chunked prefill + kernel offset API + graph schema + runner loop/
projection + sealer/wrapper contexts 128k/256k + capacity override + detached launch/attach + oracle
builders + tests (CPU: chunk/tail program builders, schema, oracle builders; equality of chunked vs
monolithic repair on a small synthetic cache). A0: rotary micro-check. B: two 8K identity runs
(C = 2048, 512) against DB567 witnesses. C: two capacity runs. D: 128K acquisition, four L7 runs in order 1.0, 0.0, 0.05, 0.95 (first failure stops early).
E: 513-page acquisition, one L8 run. F: §18 — "serves at 256K" is qualified in §18 itself as
steady-state decode at 256K with prefill measured in hours and `prefill_chunk` (§5.2) an explicit
open item; base vs effective throughput reported separately (Gate H untouched).

## 23.8 Rotary at long positions: measure first, legacy-faithful by default (addendum v3.3)

Facts established in review (correcting an earlier draft that proposed a DSA host table): the legacy indexer computes rotary ON DEVICE
(`tpu_inference/.../glm_dsa_indexer.py:1078-1086`, `jnp.cos/sin(positions_f32 * inv_freq)`), the same
formulation as WS32's `rotary.rotary_cos_sin` at its indexer sites; the FP32 DSA host table is a
refuted, tombstoned variant (`docs/artifacts/gate-d-dsa-rope-table-8k-refusal-adjudication.json`,
classification `HOST_ROTARY_TABLE_REFUTED_AS_LEGACY_FAITHFULNESS_FIX`: without the table event 0 was
bit-exact vs the legacy oracle, with it 2,046/2,048 scores moved). The legacy 128K passkey (DB403) and
256K E0 (DB402) both ran with on-device indexer rotary. Therefore:

1. Indexer rotary stays on device (legacy-faithful). No DSA table. The tombstone stands.
2. Main attention: legacy uses a host FP32→BF16 cos/sin table (`patch_rotary_cos_sin_cache_numpy`)
   whose torch source multiplies BF16 tensors; the FP32-products-plus-one-final-round form is the
   DB531-evidenced XLA lowering (DB531 reproduced the legacy 64-wide suffix bitwise with
   `apply_rotary_fp32_final_round`). WS32 (`ws32_layer.py:842-866`) computes cos/sin on device in the
   query dtype (BF16) and rotates in BF16 — the DB530 mismatch class. This is a legacy-faithfulness gap
   that DB553/DB567 tolerated (exact tokens at 2K/8K under §21). Whether it matters at 128K/256K is
   UNMEASURED: the only datum is 9.9e-3/4.0e-3 at 4,962 rad, measured on the indexer FP32 path at
   position 8155 (not the main BF16 path); an FP32 angle at 2.6e5 rad has ulp ≈1.6e-2
   on the highest-frequency pairs for host tables too, so "accuracy vs FP64" is not the criterion —
   fidelity to the accepted legacy form is, and its cost is a new §21 adjudication (B′) plus re-based
   witnesses for Step C. Not paid until shown necessary.
3. Zero-cost diagnostic as a declared side program in a protected worker (no extra lease;
   capacity-independent — a 262,657-row positions array). Amendment 2026-09-05 18:50Z: the design placed
   it in Step B's first worker, but both Step B runs (C = 2048 at `172ae03`, C = 512 at `00c6eee`)
   executed before the diagnostic existed; it therefore first runs in Step C's first capacity run, and if
   it fails that capacity run is repeated with the main table (the accepted fallback). The diagnostic's
   verifier pins the protocol (κ, bands, θ, rotary dim, the full window, 4 × 32 cells) and refuses any
   record not computed on the TPU backend. It compiles and executes one extra small TPU program OUTSIDE
   the timed window and the traced steps (before the model programs compile), is declared in the run
   record with its SHA-pinned script and artifact path/SHA, and leaves the census/HLO/trace contracts
   untouched (the sealer sees a declared, pinned side program). Content: jitted `rotary_cos_sin` at
   positions 0..262,656 for the main form (BF16 out, `apply_rotary` in BF16 on a fixed unit vector) and
   the indexer form (F32 out), compared with FP64 rows AND with the legacy main form (host FP32→BF16 table
   + FP32 products + final round) on the same vectors; recorded as an artifact with max/percentile error
   per rotary pair index (32 pairs; pair 0 has exact FP32 integer angles, pair 1 carries ≈8e-3 rad of
   FP32 angle ulp at 2.6e5 rad) and per position band (≤8,192 / ≤32,768 / ≤131,072 / ≤262,656), with the
   count of bf16-cast components that differ, using the same measurement form on both sides (rotated
   unit vector, BF16 output). Decision rule, pre-registered per (pair, band) cell: if in every cell the
   on-device main form differs from the legacy form by no more than κ=2 times the legacy form's own
   deviation from FP64 in that cell, L7/L8 run with the current path; otherwise the main BF16
   table is implemented (default-off flag, `apply_rotary_fp32_final_round` at both `:842` query and
   `:855-866` current-key sites, table SHA keyed by capacity, new acquisitions/pins), adjudicated at 8K as
   B′ with in-run first-divergent-event adjudication against the archived FP64 reference rows (reviewed
   sealer change; two-run fallback otherwise), and Step C witnesses re-based on B′.
4. Step B/C keep v3's contract (tables OFF, identity to DB567 witnesses). Site inventory for the
   record: on-device rotary at `ws32_layer.py:425, :511 (via dsa_index_keys_from_projection), :685,
   :734, :842` and `prefill_index.py:160`.

## 23.10 Evidence layout v2 — one compressed HLO object per graph (2026-09-05)

A protected run uploaded its HLO text once per rank: eight byte-identical copies of every graph and
form, ≈3.4 GB of the ≈5.8 GB a numerical prefix costs. The sealer already treated them as one object
(it requires every rank's recorded SHA-256 pair to agree and re-hashes each file), so seven copies were
redundancy, not information.

Layout `hlo_single_gzip_v2`: each graph/form is uploaded once, gzip-compressed, under the rank-agnostic
name `hlo/<graph>.<form>.gz`; every rank still records both SHA-256s of the *inflated* text. The
materializer reads the layout from the runner records (all ranks must agree; an absent field means the
original `hlo_per_rank_v1`, so sealed prefixes stay re-materializable), requires the remote HLO object
set to equal the layout's, binds each compressed object by size, CRC32C and SHA-256, inflates it once,
requires the inflated SHA to equal the SHA every rank recorded, and hard-links the per-rank names so
the sealer's per-rank replay is unchanged. The source ledger records the compressed object and its
`inflated_sha256`, and the ledger hash is bound into SUCCESS. Because this controller is pod worker 0,
the inflated file is a hard link to the worker's own text when the SHA matches, so the layout costs no
extra controller disk. The sealer's required layout is a CLI pin, not a constant, so a v1 prefix can
still be re-validated. Uploads are redundant: every rank writes the same object with `--no-clobber`,
and a precondition failure is tolerated only when the remote object *inflates* to this rank's own text.

Saving: ≈3.0 GB per acquisition and per numerical run. With it, the remaining long-context programme
costs ≈21 GB rather than ≈54 GB.

## 23.9 Rotary measurement and the main-attention host table — 2026-09-05 (decision)

The §23.8 diagnostic ran on the TPU inside the 131,072-capacity acquisition
(`greenfield_ws32_short_decoder_8k_acquire_cap131072_20260905T200533413431414Z`, pin `00c6eee`,
backend `tpu`, `TPU v4`, jax 0.10.1, libtpu 0.0.41) and returned `FAIL`: 45 of 128 pre-registered
(pair, band) cells lie outside κ=2. Artifact:
`docs/artifacts/gate-l-ws32-rotary-long-position-diagnostic-20260905.json`.

Worst cell per position band, all at rotary pair 1 (maximum absolute error of the effective
`cos`/`sin` against FP64):

| positions | on-device main | legacy host table | ratio | on-device indexer |
|---|---|---|---|---|
| ≤ 8,192 | 0.0115 | 0.0022 | 5.3× | 0.0111 |
| ≤ 32,768 | 0.0444 | 0.0031 | 14.1× | 0.0444 |
| ≤ 131,072 | 0.1779 | 0.0071 | 24.2× | 0.1778 |
| ≤ 262,656 | 0.3545 | 0.0134 | 26.6× | 0.3543 |

Three readings, and the decision each supports.

1. **The on-device error is a range-reduction effect, not a dtype effect.** The device indexer (FP32
   output) and the device main path (BF16 output) carry the same error, so it originates in
   `jnp.cos/jnp.sin` of a large argument, exactly as the PP8 V2/V3 replays found at position 8155.
2. **The indexer stays on device.** The legacy indexer evaluates the same on-device form, and the
   protected refusal record `gate-d-dsa-rope-table-8k-refusal-adjudication.json` showed event 0 is
   bit-exact against the legacy oracle only without a host DSA table. Faithfulness is the criterion,
   and the on-device form is the faithful one; the tombstone stands.
3. **The main attention adopts the host table (A′).** The legacy main path consumes a host FP32→BF16
   `cos|sin` table, applies FP32 products and one final BF16 round (DB531 reproduced the legacy
   64-wide suffix bitwise). WS32 evaluates it on device and deviates from the legacy form by up to
   26.6× the legacy's own deviation from FP64 at 256K positions, on essentially every BF16 component
   (261,839 of 262,144 differ in the worst band). That is a divergence from the reference
   implementation that grows with context, exactly where L7/L8 must claim correctness.

Cost and alternative, recorded for the audit. A′ is a default-off flag: one replicated BF16
`[capacity, 64]` table (33.6 MB per chip at 262,656), a per-step row gather, and
`apply_rotary_fp32_final_round` at the two main-attention sites; it changes numerics, so it must be
re-adjudicated at 8K under §21 (B′) and every graph must be re-acquired. The alternative — run L7 with
the on-device form and fix only on failure — was rejected: a passkey failure costs ≈5 h of pod time per
prompt and would not identify the rotary as the cause, and a pass would leave a known 27× divergence in
the record at the moment §18 is claimed.

Honest calibration: the pre-registered rule fails even in the ≤8,192 band, where the sealed Gate D runs
(DB 567/568/569) produced exact 20/20 tokens and a bit-exact event 0 with the on-device form. The rule
is therefore a *faithfulness* criterion, stricter than token exactness at 8K; it is not weakened
retroactively, and A′ is adopted for faithfulness rather than because Gate D was wrong.

Consequences: the 131,072 acquisition's HLO pins are superseded for the table-on path (its lasting
value is this diagnostic and its compiled-memory report — 27.46 GB of arguments plus 1.23 GB of temp
per chip for the chunked prefill at 131,072, so 128K fits with ≈4.3 GB of margin). Sequence becomes
A′ (code, reviewed) → 8K acquisition with the table on → B′ 8K numerical adjudication under §21 →
Step C acquisitions and capacity runs with the table on → L7 → L8. The diagnostic itself stays
default-off in later runs: it measures the device form, which the main path no longer uses, and this
artifact is its record.

