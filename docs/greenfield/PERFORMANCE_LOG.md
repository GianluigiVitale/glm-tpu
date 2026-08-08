# Greenfield performance and mechanism log

Protected PP8 2K full-decoder evidence exists at DB484: p50 `244.091151 ms` and
`4.096830 tok/s`, with exact tokens/DSA and complete protections. It passes 2K Gate-D correctness
but not Gate E. No accepted 8K, 128K, 256K, PP16 full-decoder, or WS32 full-decoder performance
measurement exists yet. Bounded layer/kernel/diagnostic results below are not token-speed proof.

## 2026-08-06 — complete feature-runtime checkpoint, not performance

`greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z` is the complete executable PP8
checkpoint selected by DB 441. At pack code `d9a883b`, 32 files contain 834,177,357,824 payload
bytes and exactly 26,068,042,432 runtime weight bytes/chip. Manifest `54e2f89b...d9917`, layout
`ba21c4ec...c9e`, plan `f46f91c3...826a`, schedule `b407fcf5...1773`, source transformation,
file hashes, remote generation/CRC32C, mounted verification, approved archive, and 8/8 cleanup pass.

Commits `74a2952`, `33aa420`, and `aff0f42` bind the corresponding default-off feature-MoE backend
to the complete decoder and fail before execution unless optimized HLO contains 75 each of the exact
three raw-U8 production kernels with no decoded full-expert overlay. Focused coverage is 14/14. No
TPU decoder-body execution, complete token, latency, or tok/s claim is attached to this artifact or
these commits; the protected 78-layer body compile/load is the next discriminator.

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

DB 429 / `greenfield_fp8_selected_up_gate_normal_two_20260806T025246823530722Z` and DB 430 /
`greenfield_fp8_selected_up_gate_concentrated_eight_20260806T025341496172586Z` passed together at
`b900cea`. Stable device compaction maps original routes into owned-first slots, uses the dynamic
owned count as the Pallas pipeline bound, masks unwritten slots, and restores exact top-8 order.
There is no host/Python routing.

| case | local routes | p50 | p90 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|---:|
| normal interleaved | 2 | 1.341385 | 1.354894 | 1.360536 | 1.371749 | 1.342174 ms |
| concentrated | 8 | 4.496970 | 4.509263 | 4.514649 | 4.524864 | 4.498625 ms |

Normal comparison max/p99/mean error is `0.00012207/0/5.76e-9`; concentrated retains
`0.0078125/0/5.77e-7`. Non-owner outputs are exact zeros. Each HLO contains one selected raw-U8
Pallas kernel, four bounded gather-index markers, no unexpected call/full F8 or decoded overlay.
Peak HBM is 2.307/2.395 GB. Both DB/archive/hash/remote-SUCCESS and clean-fleet gates pass.

Compaction is a `3.35x` normal-path improvement over the all-eight DB 428 ceiling and adds only
about `0.004 ms` to the concentrated result. The `1.34 ms` normal gate/up projection is still not a
full expert or layer and remains above the desired promotion budget; it has no decoder/tok-s claim.

DB 431 / `greenfield_fp8_selected_up_gate_normal_two_20260806T030747934509941Z` and DB 432 /
`greenfield_fp8_selected_up_gate_concentrated_eight_20260806T030852365380579Z` tested a single
persistent `[G,K,gate_then_up]` raw table and one combined Pallas stream at `ea8a61a`:

| case | p50 | p90 | p95 | p99 | mean | peak allocation |
|---|---:|---:|---:|---:|---:|---:|
| normal interleaved | 1.327685 | 1.340283 | 1.344851 | 1.355045 | 1.329922 ms | 3.919 GB |
| concentrated | 4.509895 | 4.523469 | 4.527514 | 4.537501 | 4.511209 ms | 4.007 GB |

Both protected comparisons reproduce DB 429/430's error bounds. HLO has one raw
`u8[64,6144,4096]` Pallas call, one bounded scale-gather marker, and one exactly constrained
`s32[8,2]` compact-route restore marker feeding only the final `bf16[8,4096]` order-restoring
gather; no decoded overlay or unexpected call exists. The normal p50 gain is only `1.02%`, the
concentrated ceiling regresses `0.29%`, and peak allocation increases by about 1.6 GB. The result is
therefore an honest performance null, not a promotion. The split-stream DB 429/430 layout remains
the accepted baseline. DB/archive/hash/remote-SUCCESS and eight-host pre/post cleanup all pass.

The next compact-scale diagnostic `greenfield_fp8_selected_up_gate_normal_two_20260806T031705611686395Z`
at `364867e` failed closed before correctness/timing: Mosaic could not prove that the dynamic
K-block offset into `f32[64,16,48]` was aligned to TPU's 128-element HBM tile. It has no DB or
performance claim, preserved its compiler error, and ended 8/8 clean.

DB 433 / `greenfield_fp8_selected_up_gate_normal_two_20260806T031953587365200Z` at `8be32e0`
tested the corrected final `[G,Nblock,128]` scale layout. The kernel loads an aligned two-output-
block scale tile and masks/reduces the current four K blocks in VMEM. The comparison reproduces the
normal baseline max/p99/mean error `0.00012207/0/5.76e-9`; HLO has one raw-U8 Pallas call with two
`f32[64,16,128]` scale operands, two bounded gather markers, and no unexpected call or decoded
overlay. The protected distribution is:

| p50 | p90 | p95 | p99 | mean | compile | scoped VMEM |
|---:|---:|---:|---:|---:|---:|---:|
| 3.416799 | 3.429730 | 3.433913 | 3.443074 | 3.418136 ms | 1.986 s | 6,596,608 B |

This is a `2.55x` regression versus DB 429's `1.341385 ms`; measured peak allocation remains about
2.308 GB. The mask/reduction expansion is much more expensive than the existing bounded selected-
scale staging, so the candidate is rejected without a concentrated run. DB/archive/hashes/remote
SUCCESS and 8/8 cleanup pass, and the DB 429/430 kernel/layout was restored.

DB 434 / `greenfield_fp8_selected_up_gate_normal_two_20260806T032417543569735Z` and DB 435 /
`greenfield_fp8_selected_up_gate_concentrated_eight_20260806T032510518408956Z` at `a21ad09` tested
whether declaring the independent compact-route grid axis `parallel` changes TPU scheduling:

| case | p50 | p90 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| normal interleaved | 1.344635 | 1.358001 | 1.362222 | 1.370948 | 1.345833 ms |
| concentrated | 4.497115 | 4.509551 | 4.514593 | 4.527513 | 4.498973 ms |

Both preserve the exact DB 429/430 numerical and HLO contracts and pass DB/archive/hash/remote-
SUCCESS/8-host cleanup gates. Relative to `1.341385/4.496970 ms`, the annotation is a `0.24%`
normal regression and effectively zero (`0.003%`) concentrated change. It is rejected and restored.
DB 429/430 now form the selected gate/up basis for activation/down fusion; this does not promote a
layer, decoder, or token-speed result.

## 2026-08-06 — selected FP8 SwiGLU/down kernel

Protected DB 436 / `greenfield_fp8_selected_swiglu_down_normal_two_20260806T034440557202346Z` and
DB 437 / `greenfield_fp8_selected_swiglu_down_concentrated_eight_20260806T034542641482354Z` passed
at `f496eb1`. The kernel accepts distinct BF16 gate/up route rows, forms exact BF16 SwiGLU inside
Pallas, selects each owned route's final-layout raw `u8[64,2048,6144]` down matrix, dequantizes only
VMEM tiles, accumulates in FP32, and restores original top-8 order with exact-zero nonowners.

| case | local routes | p50 | p90 | p95 | p99 | mean | max / p99 error |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal interleaved | 2 | 0.773045 | 0.781022 | 0.786141 | 0.794347 | 0.773929 ms | 0.015625 / 0.0078125 |
| concentrated | 8 | 2.421714 | 2.432291 | 2.435744 | 2.445903 | 2.423081 ms | 0.03125 / 0.03125 |

Both use the identical optimized-HLO SHA `abf6e710...aca7c`: one raw-U8 Pallas call, two exact
BF16 compaction scatters, one bounded selected-scale gather, one bounded final-order gather, no
unexpected auxiliary call, and no full F8/BF16/F32 weight overlay. Scoped VMEM is 491,520 bytes;
compile is 0.819/0.806 seconds and peak allocation is 1.489/1.502 GB. DB integrity, full evidence
hashes, approved archive/SUCCESS, and eight-host pre/post census pass.

Diagnostic `...T034144197515706Z` at `22e46ab` compiled the same one-kernel mechanism but failed
closed before comparison/timing because two exact gate/up compaction-scatter index markers were
misclassified as final-output restoration. The corrected contract pins each marker to its exact
consumer rather than permitting a generic custom call. The diagnostic has no DB/performance claim
and ended 8/8 clean.

This completes the standalone activation/down proof only. It avoids writing the activated
intermediate between SwiGLU and down, but separate gate/up outputs still cross an HBM/launch
boundary. Adding DB 429 and DB 436 p50 gives an unmeasured `2.114 ms` normal two-call estimate, not
a layer or token-rate result. Next evidence must compose route weighting, shared expert, the exact
four-chip local combine, and the protected real layer oracle; fusion must then remove the remaining
boundary if the measured layer misses budget.

## 2026-08-06 — exact final-layout Pallas real MoE layer

Protected DB 438 / `greenfield_real_layer_pp8_pallas_20260806T050514347248323Z` passed at
`5fed847`. The direct raw loader binds derivative manifest `3da63bd9...e427`, source manifest
`68ef8201...f938`, and independent oracle `c63ffa19...ebff`; it performs 56 final-owner transfers
with zero dequantization, host concat, or runtime weight transpose. A composition error found by
the preceding failed correctness run is now fail-closed: standalone shared kernels use one
128-wide contraction tile per checkpoint scale block, while selected routed kernels retain their
explicitly multi-scale 512-wide tile.

| case | p50 | p90 | p95 | p99 | mean | max / p99 / mean output error |
|---|---:|---:|---:|---:|---:|---:|
| normal all-slot routes | 3.164060 | 3.185520 | 3.193292 | 3.229554 | 3.166073 ms | 0.03125 / 0.01171875 / 0.002388 |
| concentrated slot-2 routes | 7.171980 | 7.194523 | 7.204370 | 7.263152 | 7.173990 ms | 0.03125 / 0.01171875 / 0.002444 |

All route indices are exact and route-weight max error is below `9e-8`. HLO SHA
`0c8878cc...b20a` contains exactly four raw-U8 Pallas calls, seven bounded gather markers, two
bounded scatter-index markers, three non-collective local-layout `ConcatBitcast` calls, and one
`bf16[2,1,6144]` all-reduce over local ranks `{{0,1,2,3}}`; no decoded weight overlay or other
collective exists. Compile is `1.674 s`; peak HBM is 2,431,646,720 of 33,014,413,312 bytes/chip.
The fresh 20-step XPlane, 1,000 synchronized samples after 200 warmups, DB integrity, approved
archive/remote `SUCCESS`, hashes, and 8/8 cleanup pass.

This is the first protected exact no-overlay real layer, not a decoder or tok/s result. It is
performance-rejected: normal latency is `4.54x` DB 417's correctness-only decoded-overlay layer
and four sequential Pallas launches/boundaries remain. Next evidence must fuse the selected
gate/up/SwiGLU/down boundary, then the shared boundary if required, and repeat DB 438's complete
real-layer gate.

## 2026-08-06 — fused selected routed real MoE layer

Protected DB 439 / `greenfield_real_layer_pp8_pallas_20260806T052141734174170Z` passed at
`fb04875`. One raw-U8 Pallas call now performs selected gate/up, exact BF16 SwiGLU, and down; the
two gate/up route tables remain VMEM scratch and never cross an HBM/launch boundary.

| case | p50 | p90 | p95 | p99 | mean | change vs DB 438 |
|---|---:|---:|---:|---:|---:|---:|
| normal all-slot routes | 3.121940 | 3.143600 | 3.150629 | 3.178332 | 3.123250 ms | -1.33% |
| concentrated slot-2 routes | 7.063344 | 7.085381 | 7.094795 | 7.131541 | 7.066779 ms | -1.51% |

Routes remain elementwise exact and output max/p99/mean error remains at most
`0.03125/0.01171875/0.002444`. HLO SHA `918bbabd...826f` has three raw-U8 Pallas calls, five
bounded gather markers, no bitpacked gather/scatter helper, three exact local `ConcatBitcast`
layouts, one `bf16[2,1,6144]` all-reduce over `{{0,1,2,3}}`, and no decoded overlay. Compile is
`1.534 s`; generated code falls from 1,812,992 to 1,544,704 bytes and measured peak HBM is
2,431,378,432 bytes/chip. Fresh XPlane step/busy/physical-psum time is
`5.047/4.690/2.788 ms`; psum remains `59.4%` of busy time. DB/archive/hashes/remote `SUCCESS` and
8/8 cleanup pass.

The boundary elimination is valid but not the dominant bottleneck. Fuse the smaller shared
gate/up/SwiGLU/down boundary once; if its wall effect is similarly marginal, prioritize the
measured route-imbalance/collective-arrival skew rather than further launch-only polishing. This
remains a layer result, not decoder latency or tok/s.

## 2026-08-06 — fully fused shared boundary is rejected

Protected DB 440 / `greenfield_real_layer_pp8_pallas_20260806T052955364574577Z` at `cfd5bab`
replaced the two shared projection calls with one raw-U8 Pallas call that keeps two `bf16[8,512]`
gate/up tiles in VMEM across exact SwiGLU and down. The HLO `4a0807b1...dc54` has only two Pallas
calls total, five bounded gathers, three exact local `ConcatBitcast` layouts, one local all-reduce,
and no overlay. Correctness, 2,430,962,688-byte peak HBM, fresh XPlane, DB/archive/hashes/remote
`SUCCESS`, and 8/8 cleanup pass.

| case | DB 439 p50 | DB 440 p50 | change |
|---|---:|---:|---:|
| normal all-slot routes | 3.121940 | 3.169569 ms | +1.53% |
| concentrated slot-2 routes | 7.063344 | 7.122444 ms | +0.84% |

The XPlane explains the regression: physical psum is effectively unchanged
`2.787760 -> 2.786832 ms`, while custom-call busy time increases `1.723666 -> 1.862916 ms`.
The candidate is rejected. Its tested kernel remains default-off, and the active stage plus HLO
guard are restored to DB 439. The launch-boundary search is closed; next work must target measured
route imbalance and collective arrival skew. This is still not a decoder or tok/s result.

## 2026-08-06 — feature sharding removes route-arrival skew

The append-only derivative `greenfield_one_layer_pallas_feature_pack_20260806T054520020812918Z`
replaces 64 complete routed experts/chip with all 256 expert identities and a 512-wide intermediate
slice/chip. Gate/up and reciprocal down ownership remain final-layout raw U8; persistent payload is
unchanged at 9,716,380,672 bytes. Manifest `a8b91435...5cc6`, layout `e613d9ef...c431`, exact source
transformation, remote `SUCCESS`, and direct loading with no runtime dequant/concat/transpose pass.

Protected DB 441 / `greenfield_real_layer_pp8_pallas_feature_20260806T055854589778101Z` at
`65ded2c` passed both oracle cases:

| case | DB 439 p50 | DB 441 p50 | p90 | p95 | p99 | change |
|---|---:|---:|---:|---:|---:|---:|
| normal all-slot routes | 3.121940 | 2.308015 | 2.336313 | 2.348303 | 2.380885 ms | -26.07% |
| concentrated slot-2 routes | 7.063344 | 2.318155 | 2.348175 | 2.361629 | 2.393219 ms | -67.18% |

Routes are elementwise exact. Output max/p99/mean error is at most
`0.03125/0.01171875/0.002507`, and route-weight max error remains below `9e-8`. Optimized HLO SHA
`3bbd527f...383f` has the exact `u8[256,6144,512]` gate/up and `u8[256,512,6144]` down operands,
three Pallas calls, five bounded gathers, three exact local layout calls, one
`bf16[2,1,6144]` all-reduce over `{{0,1,2,3}}`, and no decoded overlay or other collective.
Compile is `1.560 s`; generated code is 1,616,896 bytes and peak HBM is 2,430,860,800 bytes/chip.

The fresh XPlane identifies the mechanism: selected-kernel time is essentially unchanged
`1.581196 -> 1.576724 ms`, while the physical psum falls `2.787760 -> 0.028511 ms` and total busy
time falls `4.690090 -> 1.901923 ms`. The prior 2.79 ms “collective” was almost entirely
route-dependent arrival waiting, not a 12 KiB four-chip reduction floor. DB integrity, evidence
hashes, approved archive/remote `SUCCESS`, and authenticated 8/8 pre/post cleanup pass. The feature
layout is selected for PP8 full-runtime integration. This is one real layer, not token throughput.

## 2026-08-05 — protected topology/local-group proof

Artifact `greenfield_topology_20260805T125842425591441Z`, DB 405, proved runtime physical inventory
and PP8/PP16 group manifests on all eight hosts. Observed topology is `2x4x4`; TPU-VM suffix order is
not JAX process order. The accepted PP8 process ring is `[0,2,4,6,7,5,3,1]`; PP16 uses 16 adjacent
two-chip stages. This is topology evidence only, not transport or model performance.

## 2026-08-06 — rejected complete feature-body diagnostic

`greenfield_short_decoder_compile_pp8_pallas_feature_20260806T084346269707216Z` at `a8194cd`
completed the real 78-layer/2K feature body on all eight hosts. Fleet-max profiler-free p50/p99 over
ten synchronized samples is `58,804.040455/58,804.322717 ms`; no token or tok/s claim is permitted.
Load/compile maxima are `278.584/186.853 s`, and measured peak HBM is
`26,144,010,752/33,014,398,976` bytes/chip.

HLO `64df6dc7...2ea0` contains 79,861 instructions, 1,195,999 bundles, 389 overlays,
`219AG/294AR/16CP`, and 75 each of the three raw-U8 feature-MoE kernels. It contains no decoded
expert overlay. Every gather/reduce is limited to one of the eight exact four-chip groups, so the
58.8-second wall is not a full-pod collective regression. The remaining layer graph still performs
reference whole-matrix FP8 dequantization/projections for attention, DSA, sparse attention, and
dense MLP. A fresh two-step fleet XPlane is the next discriminator before replacing those paths in
the specification's Pallas order.

All decoder metadata and direct-load invariants pass. Final DB/archive `SUCCESS` did not: the shell
finalizer required a device-dequantization field that this loader version did not emit. The HLO,
eight records, diagnostic archive, and clean 8/8 failure census are preserved, but the run remains
evidence-rejected and has no DB row.

## 2026-08-06 — protected complete feature-body attribution

DB 442 / `greenfield_short_decoder_compile_pp8_pallas_feature_trace2_20260806T092025101122999Z`
at `0cd5209` successfully sealed the corrected loader metric and a profiler-after-wall fleet trace.
The one profiler-free body sample is `58,804.002894 ms`; this is body-only diagnostic wall, not a
token or tok/s result. Compile max is `169.123 s` and peak HBM is `26,144,010,752` bytes/chip.

Eight XPlanes cover 64 cores and two selected steps/core. Mean device step/busy time is
`56,722.255839/54,643.549475 ms`. The profiler attributes `47,294.061096 ms` to 16 compact
stage-permute start/done regions and `7,318.308852 ms` to gather/scatter. The former is serial
pipeline waiting: dequant gather signatures alone total `7,317.697973 ms` per average core, and
eight PP8 stages imply `58,541.584 ms`, within 0.45% of profiler-free wall. Caller attribution is:
attention output `3,353.665`, shared q_a `1,616.121`, q_b `1,077.740`, kv_b `477.370`, kv_a
`413.463`, dense `283.205`, DSA wq_b `69.072`, and DSA wk `27.063 ms/core`. Feature-MoE costs
`14.784 ms/core`. Therefore whole-matrix reference FP8 dequantization is the immediate 2K floor;
the permutes and local ICI are not optimization targets.

HLO `7ef2b071...f59a` passes exact local collectives and feature-kernel counts. Summary/XPlane
summary SHAs are `0361d44e...64e1` / `91a424fc...d17`. Approved archive/remote `SUCCESS`, DB
linkage, and 8/8 cleanup pass. The result remains performance-rejected and does not complete Gate D.

## 2026-08-06 — production-shape Pallas DSA scorer passes protected metal

DB 443 / `greenfield_dsa_score_20260806T095455945075126Z` at `d068a9f` measures the exact one-row
256K/LP4 scorer shape (`32` heads, `128` dimensions, `65,536` local keys). After 200 warmups, 1,000
profiler-free samples are:

| mean | p50 | p90 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 0.327558 | 0.326595 | 0.335482 | 0.341040 | 0.350320 ms |

The TPU/reference FP32 score max/mean/p99 error is
`2.861e-6/2.417e-7/1.386e-6`, and the exact 2,048-position output/order has zero mismatches. HLO
`5e2b7185...b295` has one named Pallas call and no per-head HBM overlay, dead-row shape,
collective, or unexpected custom call. Compile is `0.394 s`; peak HBM is `20,491,776` bytes.
Runner/summary SHAs are `992bc991...fe12` / `37789e92...0af`; DB/archive/remote `SUCCESS` and 8/8
cleanup pass. This latency is standalone scorer wall, not a layer or token result.

## 2026-08-06 — exact TensorCore DSA top-k passes production-shape protected metal

DB 445 / `greenfield_dsa_topk_20260806T102752126905724Z` at `3870c2f` measures the exact bitonic
TPU-v4 path. The local runtime shape is `f32[1,65,536]` plus arbitrary global `s32[65,536]`
positions to 2,048 ordered pairs; the global shape merges a deliberately permuted
`f32/s32[4,1,2,048]` union. Both TPU JAX and independent host lexicographic oracles match scores,
positions, valid counts, sentinels, and lowest-position high-score ties elementwise.

| phase | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| local 65,536→2,048 | 1.364911 | 1.364405 | 1.376845 | 1.379481 | 1.388090 ms |
| merge 4×2,048→2,048 | 0.338555 | 0.337671 | 0.349403 | 0.354289 | 0.362475 ms |

Local/merge compile is `8.381/6.176 s`; peak allocation is `17,794,560` bytes. Optimized HLO
`e6b8e209...e7e90e` / `d990a754...0b674` contains exactly six/two named Pallas calls, with no XLA
sort/top-k, collective, unexpected call, or dead row. Runner/summary SHAs are
`8d41a09c...bbc7` / `a0bd5514...5924`; DB/archive/remote `SUCCESS` and 8/8 cleanup pass.

DB 444 at `bacfbdf` first proved the exact reduction structure but is performance-rejected:
local/merge p50 was `59.979532/4.495320 ms`. Bitonic is `43.96x/13.31x` faster. Three subsequent
compile diagnostics failed closed on TPU-v4 layout/scalar/compiler limitations before timing and
ended clean. This closes standalone Section 7.2 item 6, not integration, layer wall, or tok/s.

## 2026-08-07 — second real-prompt Gate D attempt reaches DSA observation

Rejected diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T001553313414003Z`
at `34b7611` passed the corrected production/observer/prefill HLO gates, loaded the real model,
executed the 2,034-token device prefill, and reached position 2,034 of the separate observer replay.
The old comparator then rejected event 0 offset 39 (`970` legacy versus `1670` greenfield) and
reported 40,075 total-order position mismatches across 21 DSA events. Count, producer, stage-lane,
padded-slot, and next-position contracts passed. Execution stopped before the production warmup,
timing, or trace window, so this run has no DB row and no answer-rate claim. Diagnostics are
preserved and authenticated failure-exit census is 8/8 clean.

The mismatch exposed an over-strict harness assumption: independent TPU programs are not required
to have identical total FP32 score order. Commit `b406e3a` replaces that gate with the documented
contract—exact top-k set/count/tails and lowest-position ties against the executing device scores,
while retaining position-aligned legacy scores/order as diagnostics. The
same audit changed q_a/kv_a LoRA RMSNorm from `1e-5` to `1e-6`; the later pinned-config/source
audit below proves that historical classification was wrong and restores `1e-5` as the required
contract. The observer now stores and hashes its raw score/position tensors and remains a
separate callback-free, no-donation executable. Production outputs/HLO are observer-off. Local
verification passes 59 relevant tests plus syntax/static checks. A new protected run is required;
`b406e3a` itself is not performance evidence.

## 2026-08-07 — executing-device DSA contract passes first real step

Rejected diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T011313172353535Z`
at `9281341` passed real load, production/observer/prefill compile, the 2,034-token device prefill,
and all executing-device DSA checks at position 2,034: 21/21 exact causal sets/counts/tails,
canonical score order/lowest-position ties, producer IDs, four-lane replication, padding, and next
position. It then failed the newly added same-input Gate C score tolerance applied to the
independent full-network legacy path (`max/p99/mean 69.366638/28.665792/3.300944`). It stopped
before token replay/timing, has no DB row or answer-rate claim, and ended 8/8 clean.

The retrieved raw observation (989,858 bytes, SHA `3cfaa3b3...a53803`) shows layer 0 is close
(`0.028/0.025/0.010`, correlation ~1.0), then score error grows monotonically with depth while each
greenfield event's best correlation remains its same legacy layer. Thus the failed tolerance was
methodologically invalid: Gate C bounds compare two programs on the same captured hidden input;
later full-decoder hidden inputs already differ after topology/reduction reassociation. Commit
`715870e` keeps these score statistics diagnostic, gates the exact first token before replay, and
continues to require executing-device exactness and all subsequent raw tokens. It also makes the
physical JAX-process-0 worker upload all 14 raw observer artifacts for final SHA/content validation.
Local verification passes 59 relevant tests plus static checks. Another protected run is required.

## 2026-08-07 — full DSA replay passes; first raw-token divergence localized

Rejected diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T014648797562615Z`
at `0a42c93` passed the real 2,034-token prefill first token (`220`) and all 294 observer DSA events
(14 steps x 21 producers): exact causal set/count/tails, executing-score order/lowest-position
ties, producer IDs, replication, padding, and next position. The observer token sequence matched
the independent oracle through recurrent offset 9, then offset 10 expected `16345` and produced
`12877`. Observed sequence was
`[104550,101294,16,13,3155,537,10662,432,13,576,12877,374,6303,13]`.
All 14 raw observations are preserved; failure-exit census is 8/8 clean.

The run stopped before production warmup/timing/trace, so it has no DB row, answer-rate, or Gate D
claim. Commit `ffa3db7` adds an observer-only canonical global top-16 logit record with exact IDs,
FP32-cast score bits, expected-token rank/margins, fleet lane health, and per-tensor hashes. It adds
no collective and allows only the two declared token-exchange result shapes to differ from
production; every non-token collective remains an exact isolation gate. Full local verification is
343 passed / 1 skipped plus Python/Bash/ShellCheck/diff checks. Protected metal evidence is next.

## 2026-08-07 — exact fused residual association is integrated, awaiting protected metal

Source inspection of the accepted fused norm established a concrete semantic mismatch: it forms
`hidden + residual` in FP32, normalizes that unrounded value, and separately carries a BF16-rounded
sum. The earlier greenfield decoder instead rounded the sum to BF16 before normalization at both
norms in every layer. A deterministic BF16 fixture proves the two formulas differ; this is a direct
arithmetic counterexample and a plausible cumulative explanation for the position-2,044 token
inversion, not yet protected-model proof.

The default-off split-state path now preserves both live components through dense, DSA,
IndexShare, MoE, final norm, the eight PP8 stage transfers, and teacher-forced prefill. A forced
32-device CPU run passes two recurrent steps and prefill with unchanged collective counts, exactly
eight residual transfers of local shape `[2,1,H]`, local groups only, and byte-identical StableHLO
for the disabled default. The protected runner records the flag, HLO dtype/shape/count, 24,576-byte
stage payload, 12,288-byte incremental buffer, fleet agreement, HBM, and DB provenance. CPU tests
and static checks pass; no TPU run, token correction, timing result, Gate D, or throughput claim is
made here.

## 2026-08-07 — first split-state protected compile exposes one exact HLO-contract delta

Rejected pre-execution diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260807T142647912804579Z`
at `4c2cc0b` passed fresh eight-host census and exact-pin sync, loaded the real runtime checkpoint,
and compiled the production executable identically on all hosts. The HLO gate rejected one logical
shape delta before model execution: split-state stage-0 embedding lowered to one
`bf16[1,1,6144]` local all-reduce instead of the default path's `bf16[1,6144]`. All remaining result
shapes matched, including 75 `bf16[2,1,6144]` MoE reductions; counts were `219AG/372AR/17CP`; and
the residual contract found exactly eight `bf16[2,1,6144]` permutes with no forbidden full-pod
tensor. Source metadata binds the singleton reduction to `cond/branch_1_fun/psum` in token
embedding.

All eight host logs are byte-identical (`c6387cb0...aaba`) and cleanup ended with eight unique
`CENSUS_OK` markers. No decoder, observer, prefill, token, timing, trace, DB, `SUCCESS`, Gate D, or
throughput result exists. The contract now pins the singleton embedding form only when split state
and complete-token execution are both enabled. Revalidation passes the archived split HLO
(`418c75be...0023`) with zero violations and also passes the prior default TPU HLO unchanged.

## 2026-08-07 — split-state protected draw corrects tokens but outer inventory rejects evidence

Rejected post-execution diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260807T145659123253046Z`
at `e5df9f4` passes the inner production/observer/prefill HLO contracts, all 294 DSA events, and the
sealed token sequences. In particular, position 2,044 now emits expected token `16345` rather than
the prior `12877`; production matches 13/13 available prefix tokens and the isolated observer
matches 14/14 recurrent tokens. All eight fresh XPlanes exist and the failure census is 8/8 clean.

The outer validator then rejected the already-correct inner feature contract because its expected
dictionary omitted the 75 `greenfield_fp32_to_bf16_r8_h6144` conversion boundaries required by
the enabled FP32 down-reconstruction path. The actual and inner-expected dictionaries agree
exactly and have no forbidden overlay. A conditional outer expectation plus static regression fixes
that validator defect; 27 focused tests pass, and a read-only execution of the patched validator
over the archived draw passes every condition before DB mutation.

The rejected draw measured fleet-max p50/p99 complete-step wall
`244.285871/244.535177 ms` (`4.093565` implied tok/s), peak HBM `26,245,004,800` bytes/chip, and
`6,769,394,176` bytes minimum measured margin. HLO is `bc23eca0...515a` with
`219AG/372AR/17CP` and eight local `bf16[2,1,6144]` transfers. Because there is no DB row,
summary, sealed archive, or local/remote `SUCCESS`, none of these timings is an accepted Gate D/E
claim. One identical protected retry from the corrected clean pin is required.

## 2026-08-07 — first accepted complete PP8 decoder passes protected 2K Gate D

DB 484 / item 1768 / tag
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260807T153043648919419Z`
at `095d7a1` passes exact sealed tokens and all 294 DSA events. The former position-2,044 mismatch
is fixed (`16345`), and the production and observer sequences have zero mismatches. Direct load,
state/cache, HLO, metadata, observer isolation, fleet agreement, fresh traces, DB snapshot,
approved archive, and authenticated pre/post cleanup all pass.

Fleet-max profiler-free complete-step p50/p99 is `244.091151/244.247375 ms`, yielding
`4.096830` single-stream tok/s. Peak HBM is `26,245,004,800` bytes/chip with
`6,769,394,176` measured bytes remaining. Eight XPlanes cover 64 cores and two steps/core. HLO
`bc23eca0...515a` has exact `219AG/372AR/17CP`, eight `bf16[2,1,6144]` stage transfers, only local
four-chip repeated groups, and no forbidden reconstruction/overlay/dead row.

This closes protected 2K Gate D correctness, not Gate E: latency exceeds 200 ms and rate is below
4.5 tok/s. Summary/XPlane/DB snapshot SHAs are `1ba77357...b60`, `0d8ac98d...afa0`, and
`5789f5e8...26b8`; remote `SUCCESS` binds DB 484 and all sealed checksums pass. Next is the
required 8K decoder proof, then evidence-driven PP8 optimization before 128K and 256K.

## 2026-08-07 — DB 488 rejects raw-FP8 fused-QKV projection association

Bounded diagnostic DB 488 / tag
`greenfield_layer0_dsa_association_20260807T205946537394555Z` at `05d9915` passed its exact
artifact, raw-FP8/FP32-scale, HLO, DB/archive, local/remote `SUCCESS`, and authenticated 8/8 clean
census contracts. The TP32-local pack HLO keeps the physical `bf16[32,32,82]` result live and has
no collective or callback; the pack and global-state HLO SHAs are `b6134565...668e` and
`0cfebff1...6d98`.

Neither physical projection association restores the sealed 8K DSA order. The TP32-local path has
1,703 order mismatches, one selected-set swap, and max/mean absolute score error
`0.0340824/0.0259581`; the global path has 1,650 order mismatches, one set swap, and mean error
`0.0261870`. The associated Pallas result preserves the set but misses 1,523 order slots with mean
error `0.0221107`. This is correctness-diagnostic evidence only: it changes no decoder and carries
no latency, Gate-D, or throughput claim.

The remaining high-value discriminator is upstream of the scorer: the real fused projection
physically shards q-a as 32 x 64 and the logical 2,048-wide RMSNorm therefore has a distributed
reduction association. The next bounded proof must adapt the already pinned legacy sharding/norm
semantics and verify exact HLO before any full-model retry.

## 2026-08-07 — DB 489 rejects distributed q-a RMSNorm association

Bounded diagnostic DB 489 / tag
`greenfield_layer0_dsa_association_20260807T220615983460791Z` at `54edbf7` passes its exact
32-chip projection/norm HLO, q-residual artifact, DB/archive, local/remote `SUCCESS`, and three
authenticated 8/8 clean censuses. The diagnostic HLO contains exactly one global `f32[32]`
all-reduce and one `bf16[32,64,32]` all-gather; HLO and artifact-manifest SHAs are
`314956e1...0202` and `046b4f0e...50e2`.

It does not restore sealed order. XLA has an exact selected set but 1,501 order mismatches;
one-row Pallas has an exact set but 1,249 mismatches. Their score max/mean errors are
`0.0304594/0.0228811` and `0.0268021/0.0191863`, respectively. This run is diagnostic-only and
contains no decoder latency or token-rate result.

The unmodified variants match the reconstructed FP32/divsqrt baseline state elementwise; this is
not a captured internal-state comparison because only sealed selected positions/scores exist. The
next concrete discriminator is therefore the accepted XLA DCP scorer's exact three-page local
geometry, replacing the diagnostic's mismatched nested 84-page reconstruction before another
upstream q-a hypothesis is attempted.

## Exact local DCP scorer checkpoint — implementation only

The next bounded probe now compiles the accepted three-page local XLA scorer geometry and stitches
the eight DCP stripes outside the executable. It reuses the sealed DB489 q-a artifact, so it does
not repeat the diagnostic full-pod projection/norm. Focused coverage is 36/36 and the complete
CPU-only suite is 416 passed / 1 skipped; the exact 20,188-byte CPU HLO contract passes. These are
static/correctness readiness facts only. No TPU result, decoder latency, token rate, Gate-D result,
or performance comparison exists for this checkpoint.

## DB 490 — exact local DCP scorer rejected; no performance claim

DB 490 / `greenfield_layer0_dsa_association_20260807T224711202903102Z` at `01ba8cd` passes exact
HLO, DB/archive/SUCCESS, evidence hashes, and 8/8 cleanup. HLO `e4e4d6cd...65b4e` reproduces the
accepted three-page local scorer shapes and has no collective/callback. Its stitched score row is
elementwise identical to the prior pagewise reconstruction (zero delta), so scorer geometry is
rejected as the missing 8K association. Baseline/distributed states remain exact-set but have
1,640/1,501 order mismatches. This diagnostic has no timing or throughput result.

The pinned model config SHA `22e49334...65ff` declares `rms_norm_eps=1e-5`, and accepted vLLM uses
it for q_a/kv_a RMSNorm. Earlier greenfield work incorrectly changed those norms to `1e-6`; DB489
used the same wrong value. The next bounded run therefore repeats only the existing distributed
q-a phase at model epsilon `1e-5`, with config/numerical provenance, then applies the exact local
scorer. Production and its performance status remain unchanged pending exact set/order proof.

Diagnostic readiness is CPU-only: 33/33 focused and 417 passed / 1 skipped full greenfield tests,
plus Bash, ShellCheck, Python, JSON and diff checks. It is not TPU or performance evidence.

## DB 491 — model-epsilon DSA diagnostic; no performance claim

DB 491 / `greenfield_layer0_dsa_association_20260807T231449677046310Z` at `ea879a2` passes the
bounded HLO, model-config provenance, DB/archive/SUCCESS, checksum, and three authenticated 8/8
cleanup contracts. Its exact local DCP XLA result preserves the 2,048-position set with zero swaps
but has 1,408 order mismatches. Max/mean/signed/p99 score error is
`0.00506306/0.00134283/+0.000276074/0.00403500`. This is about a 17x mean-error improvement over
DB489's wrong `1e-6` q-a epsilon, but it does not satisfy the exact order gate. One-row Pallas has
1,161 order mismatches and mean error `0.00436344`.

The run is explicitly diagnostic-only: it has no profiler-free timing, decoder execution, Gate-D
result, or token-rate claim. Production remains at the last accepted DB484 status pending the
remaining numerical-association proof.

## BF16-origin wk no-op — protected refusal, no DB or performance claim

Protected bounded launch
`greenfield_layer0_dsa_association_20260807T235427432046987Z` at `948f981` reused the sealed DB491
q-a artifact and entered only the one-host score phase. Its novelty guard stopped before scorer
execution because the BF16-origin-`wk` candidate and direct-FP32-origin baseline produced
elementwise-identical prompt keys. Thus the upstream adapted-weight dtype difference is erased by
the complete projection/key-LayerNorm/RoPE/BF16-cache boundary and cannot change selected order.

The run intentionally has no runner summary, DB row, final `SUCCESS`, decoder execution, latency,
or throughput result. Partial evidence is retained under the approved diagnostic prefix. The
failure-exit census has eight unique `CENSUS_OK` hosts and SHA
`892250e022e819539c51ee39a2c281b4a32cb16cf13dd787932c459db6ab099d`. This closes the candidate;
it must not be repeated or used to authorize a Gate-D retry.

## DSA internal observer trace failure — no performance result

Protected attempt `greenfield_legacy_layer0_dsa_internals_20260808T005359078558816Z` failed during
warmup tracing on an illegal torchax-tracer-to-NumPy conversion. No request executed and no timing,
token-rate, DB, comparison, or final `SUCCESS` artifact exists. Cleanup ended with eight unique
`CENSUS_OK` hosts; 11 failure diagnostics / 2,123,660 bytes are verified in the approved bucket.
Observer fix `83ff4a357` is test evidence only and does not change any latency or ETA claim.

## DB499 exact DSA query association — correctness only

Protected DB499 / `greenfield_layer0_dsa_query_association_20260808T034636385240375Z` at
`b41c3ab` proves the accepted layer-0 query bitwise from the raw final-owner FP8 state when the
complete local FP32 `wq_b` shard exists before the M=1 dot. The PP8 candidate has shape
`f32[1024,2048]` (8 MiB), HLO SHA `ea5e5c56...6c89`, zero mismatches and no global weight
reconstruction. DB495--DB498 Pallas/streamed alternatives retain 4,096 down to 1,208 mismatches
and are rejected. DB/archive/remote SUCCESS and 8/8 cleanup pass. These bounded runs deliberately
record `performance_claim=false`; no latency, Gate-D, or token-rate result follows. The next
performance-admissible evidence is the corrected protected 8K complete decoder.

## Corrected 8K DSA-observer refusal — no performance result

Protected attempt
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260808T041407656729112Z`
at `f129e63` stopped before its measured window because the first decode-step DSA event stream was
not exact. Token `101252` is correct and event 0 retains the accepted set, but event 1 has seven
selected-set swaps and an aligned common-score mean/signed delta of
`0.18290268/-0.18290268`. The run contains no profiler-free latency distribution, tok/s, DB row,
final `SUCCESS`, or Gate-D result. Its eight logs agree byte-for-byte and cleanup is 8/8 clean.
The last accepted PP8 decoder performance therefore remains DB484 at `244.091151 ms` p50 and
`4.096830 tok/s`; Gate E is still not passed.

## All-event 8K observer refusal — no performance result

Protected diagnostic
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_dsa_internal_trace2_20260808T093721804151742Z`
at `380659a` repeated the sealed `f129e63` DSA payload bitwise and stopped before timing. It proves
the first production divergence is layer-0 q-a state (494/2,048 BF16 mismatches, max `0.015625`),
not the later DB499 `wq_b` projection. The layer-1 comparison is downstream and diagnostic only.
There is no latency distribution, tok/s, DB row, `SUCCESS`, or Gate-D result. The parent artifacts
and comparison are hash-sealed in the approved bucket and cleanup is 8/8 clean. DB484 remains the
only accepted PP8 decoder performance result; Gate E remains open.

## DB501 bounded q-a rejection — no performance result

DB501 at `b4488076` ran one local four-chip host for an arithmetic-only matrix and found no exact
one-row N82 candidate. Its 31-second elapsed value includes compilation and orchestration and is
not decoder latency. There is no token generation, timed window, XPlane, wall tok/s, Gate-D, or
Gate-E result. DB484 remains the accepted PP8 performance point at `244.091151 ms` p50 and
`4.096830 tok/s`.

## DB502 exact q-a convolution — no performance result

DB502 at `c230c11` proves a one-row N82 convolution matches the accepted layer-0 q-a state bitwise.
Its 21-second elapsed value includes compilation and protected orchestration on one local host. It
does not generate tokens and has no timed decoder window, XPlane, wall tok/s, Gate-D, or Gate-E
claim. DB484 remains the accepted PP8 performance point at `244.091151 ms` p50 and
`4.096830 tok/s`.
