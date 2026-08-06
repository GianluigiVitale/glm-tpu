# Greenfield performance and mechanism log

No greenfield full-decoder model-performance measurement exists yet. The real-layer result below is
checkpoint-backed model compute; all other results are protected synthetic TPU mechanisms. None
reports token speed.

## 2026-08-05 — protected Gate C correctness, not performance

DB 421 / `greenfield_gate_c_pp8_20260805T224645828157364Z` executes real dense, full-DSA, and
IndexShare layers on physical PP8 stage 0 at code `dc20b3f`. It passes direct final-owner loading,
bounded raw-oracle tensor comparisons, exact selection/tie order for actual TPU scores, exact
8,192-byte IndexShare state reuse, cache integrity, local HLO, fresh XPlane, HBM, DB/archive, and
clean-fleet checks. Dense/DSA/IndexShare optimized HLO contains `0AG/1AR`, `3AG/0AR`, and
`2AG/3AR`, respectively, over only the four-chip stage. Peak HBM for the bounded proof is
0.281 GB/chip.

The raw PyTorch CPU and TPU FP32 scorers have bounded numerical drift, causing two different
members at the 2,048-of-2,304 cutoff. This is an explicit diagnostic: the TPU distributed top-k is
elementwise exact against the canonical ordering of its actual score row, but raw cross-backend
position identity is not claimed. The run intentionally collected no profiler-free latency
distribution and sets `performance_claim=false`; its XPlane is evidence of physical execution,
not a token-speed measurement.

## 2026-08-05 — complete checkpoint load integrity, not performance

DB 420 / `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z` proves that the complete
PP8 base-decoder final layout loads directly on all 32 chips. It validates 750,122,559,744 payload
bytes / 122,640 final shards with device round trips and no host FP8 dequantization, global concat,
or runtime checkpoint reshard. Maximum weights-only peak HBM is 24.841 GB/chip, leaving at least
8.173 GB of the runtime-reported 33.014 GB. Local checksums, DB integrity, all three 8/8 censuses,
and all 47 byte-identical remote archive files pass.

Load duration is checkpoint initialization and is not decode latency. This run did not compile or
execute a decoder, measure KV/DSA/executable/overlay HBM, or produce a token. It therefore provides
no tok/s estimate and cannot be used to claim that 2.7 tok/s has improved.

## 2026-08-05 — protected exact PP8 real sparse layer

DB 417 / `greenfield_real_layer_pp8_20260805T165737737514245Z` executes real GLM layer-3 FP8 MoE
weights on one isolated physical four-chip PP8 stage at code `db19893aa...fc78`. The direct loader
verifies the bounded pack and oracle identities, sends only final-owner shards, performs 24 device
dequantizations, and performs zero host FP8 dequantizations or global tensor concatenations.

Profiler-free wall after 200 warmups, 1,000 samples/case:

| case | p50 | p90 | p95 | p99 | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| normal, routes span all 4 chips | 0.696215 | 0.716489 | 0.722115 | 0.742864 | 0.697678 | 1.527490 ms |
| all 8 routes on one chip | 1.134090 | 1.155466 | 1.163958 | 1.195100 | 1.177258 | 40.141347 ms |

The retained concentrated maximum is one host-wall outlier; p50–p99 remain tight. Exact route IDs
pass. Both cases have output max/p99 error `0.03125/0.01171875`; means are `0.002329/0.002352`, well
inside the documented BF16 reduction-association contract.

Optimized HLO contains exactly one `bf16[2,1,6144]` all-reduce over `{{0,1,2,3}}`, no other
collective, and no `[32,6144]` tensor. Measured peak HBM is `5.640 GB/chip` and post-timing live HBM
is `4.860 GB/chip` versus `33.014 GB` available. The separate fresh 20-step XPlane, collected only
after wall timing, reports one physical `psum`/step, `0.579 ms` mean device step, and `0.325 ms` in
the collective. All evidence hashes, DB 417 integrity, approved archive, and 8/8 cleanup pass.

This establishes that real topology-local sparse-layer compute is sub-millisecond in the normal
case and that the legacy `2.7–3.3 tok/s` result is not an inherent per-layer TPU floor. It does not
predict or claim 78-layer latency, 256K attention latency, serving wall rate, or token throughput.

## 2026-08-05 — protected exact PP16 challenger and layer adjudication

DB 418 / `greenfield_real_layer_pp16_20260805T172807177182695Z` executes the same real layer,
source revision, input, router cases, independent oracle, warmup/sample counts, wall methodology,
and protection contract on captured two-chip stage 10. Its final layout owns 128 complete routed
experts and shared-intermediate width 1024 on each chip.

| plan/case | p50 | p90 | p95 | p99 | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| PP8 normal | 0.696215 | 0.716489 | 0.722115 | 0.742864 | 0.697678 | 1.527490 ms |
| PP16 normal | 0.900610 | 0.919235 | 0.925838 | 0.949264 | 0.902438 | 1.275810 ms |
| PP8 concentrated | 1.134090 | 1.155466 | 1.163958 | 1.195100 | 1.177258 | 40.141347 ms |
| PP16 concentrated | 1.075795 | 1.093593 | 1.101284 | 1.156754 | 1.084054 | 7.424879 ms |

PP16 lowers the trace-observed local combine from `0.325282` to `0.260664 ms/step`, but
doubles owner-local weights and normal-route compute. Consequently PP16 normal p50 is
`0.204395 ms` (`29.36%`) slower, while its concentrated adversary is `0.058295 ms`
(`5.14%`) faster. PP8 also has eight fewer pipeline boundaries. This one-layer evidence
provisionally keeps PP8 as the leading base plan; it does not replace complete-decoder protected
adjudication.

PP16 route ids are exact. Normal output max/p99/mean error is
`0.015625/0.0078125/0.002121`; concentrated is `0.03125/0.0078125/0.002140`.
Optimized HLO has exactly one `bf16[2,1,6144]` all-reduce over `{{0,1}}`. Post-timing
HBM is `9.711 GB/chip` with `11.276 GB` measured peak. The fresh four-core XPlane has
20 steps/core and one physical all-reduce on all 80 core-steps. DB/archive/hashes and 8/8 cleanup
pass. These remain per-layer measurements, not tokens/second.

## 2026-08-05 — protected PP8/PP16 device-resident transport

DB 415 / `greenfield_transport_20260805T142953361259007Z` measures one complete closed stage ring
per invocation at code `577aa4bf706976f3d25f552526d0b6eaa5d4920e`. PP8 uses four physical
eight-stage lanes; PP16 uses two physical sixteen-stage lanes. Each case has 200 warmups, 2,000
samples, deterministic checksums, exact physical pair/shape/count HLO, eight-host agreement,
approved archive, and clean pre/post census.

Fleet-maximum-host profiler-free results:

| plan | payload | control p50 | transport p50 | net p50 | p90 | p99 |
|---|---|---:|---:|---:|---:|---:|
| PP8 | `bf16[1,6144]` | 0.300895 | 0.331145 | 0.030250 ms | 0.359551 | 0.521339 ms |
| PP8 | `bf16[2,6144]` | 0.302515 | 0.330270 | 0.027755 ms | 0.357699 | 0.427286 ms |
| PP8 | `bf16[1,2048]` | 0.303800 | 0.327900 | 0.024100 ms | 0.353567 | 0.426943 ms |
| PP8 | `int32[1,2048]` | 0.305175 | 0.329265 | 0.024091 ms | 0.356155 | 0.469542 ms |
| PP16 | `bf16[1,6144]` | 0.331230 | 0.407385 | 0.076155 ms | 0.436538 | 0.566279 ms |
| PP16 | `bf16[2,6144]` | 0.324830 | 0.401925 | 0.077095 ms | 0.432748 | 0.541938 ms |
| PP16 | `bf16[1,2048]` | 0.327400 | 0.399780 | 0.072381 ms | 0.429788 | 0.575288 ms |
| PP16 | `int32[1,2048]` | 0.327875 | 0.399570 | 0.071695 ms | 0.428250 | 0.470942 ms |

Each optimized program contains exactly 8 PP8 or 16 PP16 `collective-permute` operations over the
captured topology-neighbor pairs and exact payload dtype/shape. There is no other collective,
full-pod synchronization, host transfer, Ray/Python stage dispatch, or model-equivalent compute.

DB 416 / `greenfield_transport_trace_20260805T143832942547470Z` is the separate trace proof at
`8aee3351a61f89141762dda237f582b8deed3a1c`. Both plans have eight fresh XPlanes, 64 TPU cores,
and 20 selected invocations/core. PP8 has exactly 8 physical permute starts and dones per step;
PP16 has exactly 16; forbidden collective count is zero. Trace-contaminated device/cycle values are
not substituted for DB 415 profiler-free latency.

Diagnostic `...T143608529930365Z` was rejected before DB insertion because the parser tree included
both worker-0's original file and its downloaded canonical copy. The fixed proof isolated canonical
fleet files and re-ran at a new exact pin.

Transport is therefore not a plausible large bottleneck: even the full PP16 live-residual ring adds
less than `0.08 ms` over matched control. The remaining ceiling depends on stage-local model
compute/layout and elimination of legacy per-layer global arrival barriers.

## 2026-08-05 — protected dependent collective floor

All accepted runs used 75 genuinely dependent operations, 200 warmups, 1,000 measured samples,
rank-dependent nonlinear feedback, barriers, exact optimized-HLO groups/counts/pairs/shapes,
bitwise first/last checksums, eight-host agreement, append-only DB linkage, approved-bucket archive,
and clean pre/post census. Values are fleet-maximum-host latency for the whole 75-operation chain.

Dominant `bf16[2,6144]` results:

| operation | g2 p50 | g4 p50 | g8 p50 | g32 p50 | DB/run |
|---|---:|---:|---:|---:|---|
| control | 0.454 ms | 0.449 ms | 0.456 ms | 0.459 ms | 406 / `...T133905344573798Z` |
| all-reduce | 0.836 ms | 1.083 ms | 1.471 ms | 3.941 ms | 406 |
| all-gather | 0.798 ms | 1.012 ms | 1.475 ms | 4.503 ms | 407 / `...T134254891049866Z` |
| collective-permute | 0.651 ms | 0.653 ms | 0.653 ms | 0.791 ms | 410 / `...T135644529157649Z` |
| all-to-all | 0.828 ms | 0.913 ms | 0.972 ms | 1.558 ms | 410 |
| fused tuple all-reduce | 1.123 ms | 1.331 ms | 1.756 ms | 4.202 ms | 410 |

DB 408 and 409 are protected single-case validation runs for asynchronous collective-permute HLO
and fused tuple all-reduce HLO. They are superseded for latency by DB 410 but remain valid mechanism
evidence.

The supported six-operation matrices also completed at code
`fcd8426735119fee34ab8adc9e8c14b762adc2f8`:

- DB 411 / `greenfield_collectives_20260805T135850389312854Z`: `bf16[1,6144]`, 24 cases.
- DB 412 / `greenfield_collectives_20260805T140125151247631Z`: `bf16[1,2048]`, 24 cases.

The final required payloads completed at `b12af9633c8b14648db8d2a2ccd9a3c577a04817`:

- DB 413 / `greenfield_collectives_20260805T141114991474088Z`: `f32[1,6144]`, 24 cases.
- DB 414 / `greenfield_collectives_20260805T141333601664455Z`: `int32[1,2048]` routing metadata,
  20 cases (tuple reduction is intentionally undefined for integer metadata).

Representative fleet-max p50s for 75 operations:

| payload | operation | g2 | g4 | g8 | g32 |
|---|---|---:|---:|---:|---:|
| `bf16[1,6144]` | all-reduce | 0.893 | 1.116 | 1.515 | 3.997 ms |
| `bf16[1,6144]` | collective-permute | 0.705 | 0.704 | 0.721 | 0.844 ms |
| `bf16[1,2048]` | all-reduce | 0.713 | 0.841 | 1.081 | 3.948 ms |
| `bf16[1,2048]` | collective-permute | 0.637 | 0.650 | 0.653 | 0.782 ms |
| `f32[1,6144]` | all-reduce | 0.819 | 1.078 | 1.479 | 3.920 ms |
| `f32[1,6144]` | collective-permute | 0.634 | 0.646 | 0.650 | 0.779 ms |
| `int32[1,2048]` | all-reduce | 0.696 | 0.840 | 1.060 | 3.931 ms |
| `int32[1,2048]` | collective-permute | 0.617 | 0.641 | 0.640 | 0.768 ms |

Full p50/p90/p95/p99 distributions and all 1,000 samples per case are retained in each artifact.
FP8 is not a numerically relevant live-residual, metadata, reduction, or stage-transfer dtype in
the declared engine contract; it is checkpoint weight storage with bf16/f32 dequantized arithmetic.
No synthetic FP8 transport number is substituted for that contract.

### Reduce-scatter support boundary

Required small decode reduce-scatter has no accepted timing. TPU-v4 optimized XLA rewrote the 75
requested reduce-scatters into 75 all-reduces even with
`xla_tpu_decompose_every_reduce_scatters_hlos=false` and non-equivalent result segments. Protected
diagnostics `...T134618415607642Z`, `...T135045281384327Z`, and `...T135140118884391Z` failed closed
before timing, archived the diagnostic HLO, and ended clean. Reporting those as reduce-scatter
latency would be false.

### Architectural conclusion

The legacy trace attributes `106.495 ms/token` to 75 full-pod `bf16[2,6144]` MoE combines, or
`1.420 ms` per layer. The exact dependent TPU chain needs only `3.941 ms` total at g32 p50; after
subtracting control, about `46.4 us` per raw all-reduce remains. Seventy-five full-ring nearest-
neighbor permutes need `0.791 ms` total.

Therefore small-payload ICI bandwidth is not the legacy 106 ms floor. The dominant loss must be
arrival skew, layout/reshard work, barrier waiting, and surrounding legacy decomposition. A primitive
swap alone can save only a few raw milliseconds. The topology-first stage-local layout and device-
resident PP8/PP16 transport remain the required structural experiment.

## 2026-08-06 — rejected complete PP8 reference-body diagnostic

`greenfield_short_decoder_compile_pp8_20260806T004625161993456Z` loaded the complete runtime
checkpoint and compiled the real 78-layer 2K decoder body. This is a rejected diagnostic, not a
latency or token-rate result.

- Runtime load: about `104.713 GB/host` from the exact 32-file derivative.
- Compile: `380.003 s`; optimized HLO about `192,401` instructions.
- Backend expansion: `2,707,043` program bundles and `580` overlays.
- Memory: `1.17 GB` program plus `23.41 GB` arguments; observed about `25.46 GiB/chip`.
- Physical collectives: `219 AG / 294 AR / 16 CP`, all topology-local. The 294 AR instructions
  carry 312 logical results: arities `277x1 / 16x2 / 1x3` and result components
  `81 bf16[1,6144] / 75 bf16[2,1,6144] / 78 f32[256] / 78 u32[1,1,128]`.

Fleet device sequencing proved execution was active, not deadlocked: stages advanced one by one at
100% duty while later stages waited at the pipeline permute. Extrapolated one-body latency was
roughly 25–30 minutes, so the harness's 13 invocations could not complete. The cause is executable
explosion from whole-matrix U8 lookup/dequantization plus the Python-unrolled conditional expert
path. The run was stopped, diagnostic HLO/driver/runtime evidence was archived, and all eight hosts
ended `CENSUS_OK`. No tok/s value can be derived from it.

The corrective path begins with a compact Pallas kernel that retains raw U8 in HBM, dequantizes
only one 128x128 tile in VMEM, and performs BF16 MXU work with FP32 accumulation. Full-model reruns
remain prohibited until production-shaped kernel compile/correctness/HLO/microbenchmark evidence
shows the reference graph explosion is removed.

## 2026-08-06 — protected Pallas FP8 up-projection kernel

DB 422 / `greenfield_fp8_matmul_20260806T015232890679994Z` passed at `df44475` on TPU v4 for the
production expert-up shape `M=8, K=6144, N=2048`. Raw U8 checkpoint codes are same-width bitcast to
E4M3FN. The Pallas body DMAs and dequantizes only one 128x128 weight tile in VMEM, applies the exact
FP32 block scale, converts the tile to BF16, and accumulates the MXU result in FP32.

The optimized HLO has exactly one TPU custom call, an FP8 `2048x6144` operand, a bounded
`f32[48,128]` aligned scale table, no complete BF16/F32 weight matrix, and 69,632 bytes of scoped
VMEM. Compile time is `0.538 s`. The protected output is elementwise exact against complete JAX
dequantization plus FP32-accumulating dot (`max/p99/mean abs = 0`). After 200 warmups, 1,000
profiler-free samples are:

| p50 | p90 | p95 | p99 | mean |
|---:|---:|---:|---:|---:|
| 0.520605 | 0.530900 | 0.534043 | 0.544112 | 0.521162 ms |

Peak process HBM is `315,956,736` bytes. DB integrity, evidence hashes, approved archive/SUCCESS,
and eight-host pre/post cleanup pass. This is standalone blocking host-wall latency for one kernel,
not a complete expert, layer, decoder, or token-rate result. The next gate fuses up/gate, activation,
down, and local combine before replacing the exact fallback in the real-layer harness.

## 2026-08-06 — protected paired FP8 gate/up kernel

DB 423 / `greenfield_fp8_up_gate_20260806T020551714072561Z` passed at `7654338`. One Pallas custom
call consumes the production-width BF16 M8/K6144 input, two raw E4M3FN N2048/K6144 matrices, and
two bounded `f32[48,128]` scale tables. It returns distinct BF16 gate/up results with FP32
accumulators. Both are elementwise exact against complete dequantization plus FP32 dot, and the HLO
contains no full BF16/F32 weight overlay. Compile time is `0.540 s`; scoped VMEM is 184,320 bytes;
peak process HBM is 695,120,896 bytes.

| p50 | p90 | p95 | p99 | mean |
|---:|---:|---:|---:|---:|
| 0.815435 | 0.827151 | 0.833372 | 0.851290 | 0.816744 ms |

These are 1,000 profiler-free blocking samples after 200 warmups. DB integrity, hashes, approved
archive/SUCCESS, and eight-host pre/post census pass. This is a same-weight M8 mechanism test, not
a selected-expert GMM, layer, decoder, or token-rate result. The next kernel must allow the eight
routes to address different locally owned expert matrices without a decoded-weight overlay.

## 2026-08-06 — selected-expert gate/up correctness; current layout rejected

DB 424--427 prove the harder batch-one mechanism in which all eight routes address distinct raw-FP8
expert matrices. Route order, local ownership, exact-zero non-owner behavior, bounded comparison,
no decoded full-matrix overlay, DB/archive integrity, and 8/8 cleanup pass. These are not promoted
performance results because every implementation is far above the sub-millisecond one-layer budget:

| DB / implementation | p50 | p90 | p99 | compile | peak HBM |
|---|---:|---:|---:|---:|---:|
| 424 / K128 flat | 18.744257 | 18.760322 | 18.787954 ms | 0.826 s | 2.470 GB |
| 425 / K512 reordered | 18.008643 | 18.023037 | 18.053317 ms | 0.818 s | 2.433 GB |
| 426 / K512 triple-buffer pipeline | 20.305469 | 20.323746 | 20.349554 ms | 1.102 s | 2.433 GB |
| 427 / vector scales + pipeline | 23.196868 | 23.211888 | 23.234651 ms | 2.050 s | 2.435 GB |

All accepted comparisons have max absolute BF16 error `0.0078125`; up is elementwise exact in DB
427. Three vector-scale diagnostics failed comparison before timing and have no DB claim. They
found a physical BlockSpec indexing bug, which DB 427 fixes, but the corrected layout is slower.

The next controlled experiment removes a confounder rather than tuning this result: DB 427 accepts
the two full 64-expert raw tables as `[G,N,K]` and transposes them to the Pallas `[G,K,N]` access
order inside the timed JIT. The greenfield checkpoint/kernel contract must persist raw expert
weights as `[G,K,N]` and retain compact per-block scales. Until that protected test is fast,
selected-expert activation/down integration is blocked and no decoder or tok/s claim exists.

Follow-up diagnostics `...T024137692707617Z` at `5b77934` and `...T024407985780210Z` at `911ca88`
supplied `[G,K,N]` directly and failed an over-strict one-total-call HLO gate before correctness or
timing. Both have no DB/performance claim and ended 8/8 clean. The second run preserved full HLO and
corrected the earlier interpretation: its two auxiliary calls are bounded
`AssumeGatherIndicesInBound` markers for compact scale gathers, not weight transformations. The
main Pallas operand is raw `u8[64,6144,2048]`; tile-local U8-to-F8 bitcast is inside the kernel. The
revised contract permits exactly those two metadata markers and rejects any other auxiliary call or
full F8 table view.

DB 428 / `greenfield_fp8_selected_up_gate_20260806T024602582280149Z` then passed at `e5a70be` with
the final raw `[G,K,N]` layout. All eight routes select distinct local matrices. Gate/up comparisons
pass with combined max/p99/mean BF16 absolute error `0.0078125/0/5.77e-7`; up is elementwise exact.

| p50 | p90 | p95 | p99 | mean | compile |
|---:|---:|---:|---:|---:|---:|
| 4.492525 | 4.504463 | 4.509004 | 4.517450 | 4.494040 ms | 0.905 s |

This is `5.16x` faster than DB 427's `23.196868 ms` p50 and proves final persistent access order was
a major cost even though the auxiliary-call attribution was corrected. HLO contains one selected
Pallas kernel over two `u8[64,6144,2048]` tables, exactly two bounded scale-gather markers, no other
custom call/full F8 or decoded overlay, and 946,176 scoped VMEM bytes. Peak HBM is 2.395 GB.
DB/archive/hashes, remote `SUCCESS`, and 8/8 cleanup pass.

This remains a performance rejection: `4.49 ms` is gate/up alone for the adversarial eight-distinct-
experts-on-one-chip case, not an expert, layer, decoder, or token rate. The next protected split must
execute only the normally owned route count (approximately two on a PP8 chip) while retaining an
explicit concentrated-eight ceiling.

## 2026-08-05 — protected topology/local-group proof

Artifact `greenfield_topology_20260805T125842425591441Z`, DB 405, proved runtime physical inventory
and PP8/PP16 group manifests on all eight hosts. Observed topology is `2x4x4`; TPU-VM suffix order is
not JAX process order. The accepted PP8 process ring is `[0,2,4,6,7,5,3,1]`; PP16 uses 16 adjacent
two-chip stages. This is topology evidence only, not transport or model performance.
