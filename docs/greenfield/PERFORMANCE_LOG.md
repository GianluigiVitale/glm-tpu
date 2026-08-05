# Greenfield performance and mechanism log

No greenfield model-performance measurement exists yet. Results below are protected synthetic TPU
mechanism measurements; they do not report token speed.

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

## 2026-08-05 — protected topology/local-group proof

Artifact `greenfield_topology_20260805T125842425591441Z`, DB 405, proved runtime physical inventory
and PP8/PP16 group manifests on all eight hosts. Observed topology is `2x4x4`; TPU-VM suffix order is
not JAX process order. The accepted PP8 process ring is `[0,2,4,6,7,5,3,1]`; PP16 uses 16 adjacent
two-chip stages. This is topology evidence only, not transport or model performance.
