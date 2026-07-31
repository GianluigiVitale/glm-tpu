# E0 sparse-vs-dense decode differential (device-side, per step)

| arm | trace | attention | wall ms/step | device step ms | step-cycle ms | busy ms/step | idle ms/step |
|---|---|---|---:|---:|---:|---:|---:|
| sparse | `/home/gianl/glm-run/e0cap_sparse_20260730T235238Z/trace` | DSA top-2048 sparse MLA, dcp=8 | ~397-440 | 389.4 | 402.5 | 387.1 | 15.5 |
| dense | `/home/gianl/glm-run/e0cap_dense_20260731T043422Z/trace` | dense MLA over full 262K, dcp=8 | 260.8 | 269.3 | 281.9 | 267.1 | 14.9 |

Trace-vs-trace (apples-to-apples, both profiled): device busy delta **120.0 ms/step**; device step-duration delta 120.1 ms; step-cycle delta 120.6 ms. The nominal ~142 ms wall delta (403 vs 260.8) mixes a profiled sparse number with an un-profiled dense number; ~120 ms of it is real sparse-specific device time, the residual ~20 ms is measurement basis (profiling overhead + host-side wall tail), not a device category.

## The differential table (per-core means, ms/step; % = share of that arm's step-cycle)

| category | sparse ms/step | sparse % | dense ms/step | dense % | delta (sp-de) |
|---|---:|---:|---:|---:|---:|
| gather/scatter/dyn-slice | 66.11 | 16.4% | 28.98 | 10.3% | +37.13 |
| sort/top-k | 35.60 | 8.8% | 0.91 | 0.3% | +34.69 |
| compute (fusion/dot/conv) | 22.91 | 5.7% | 9.30 | 3.3% | +13.61 |
| pallas: dsa_sparse_decode (sparse MLA attend) | 12.65 | 3.1% | 0.00 | 0.0% | +12.65 |
| collectives | 183.65 | 45.6% | 171.02 | 60.7% | +12.63 |
| pallas: gmm grouped matmul (MoE experts) | 46.40 | 11.5% | 37.14 | 13.2% | +9.26 |
| data movement (copy/transpose/reshape) | 15.12 | 3.8% | 7.67 | 2.7% | +7.45 |
| control-flow self-time (conditional/while) | 3.45 | 0.9% | 0.88 | 0.3% | +2.58 |
| async start/done markers | 1.16 | 0.3% | 2.35 | 0.8% | -1.19 |
| custom-call: other (fp8 weight formatting etc.) | 0.02 | 0.0% | 8.81 | 3.1% | -8.79 |
| **device busy total** | **387.1** | 96.2% | **267.1** | 94.7% | **+120.0** |
| device idle (inter-step gap) | 15.5 | 3.8% | 14.9 | 5.3% | +0.6 |

## All-reduce / collective invocation counts per step (w0 core 0; SPMD-identical on all 64 cores)

| collective | sparse n/step | sparse ms/step | dense n/step | dense ms/step | count verdict |
|---|---:|---:|---:|---:|---|
| all-reduce | 232 | 128.58 | 232 | 126.16 | IDENTICAL 232 - common-mode floor |
| all-gather | 472 | 43.29 | 394 | 21.99 | sparse +78 (1/layer: dcp top-k index exchange) |
| all_gather | 42 | 2.76 | 156 | 14.10 | dense +114 (2/layer dcp q/KV exchange) |
| all-to-all | 312 | 4.11 | 312 | 4.24 | IDENTICAL 312 - MoE, common-mode |
| psum | 81 | 4.21 | 81 | 4.21 | IDENTICAL 81 - common-mode |
| pmax | 78 | 2.40 | 78 | 2.54 | IDENTICAL 78 - common-mode |
| **total** | **1217** | **185.4** | **1253** | **173.2** | sparse runs 36 FEWER collective invocations |

**The all-reduce floor is common-mode**: both arms run exactly 232 all-reduces/step (~0.54-0.55 ms each; largest instances 1.8-3.1 ms). In fact the sparse arm runs **fewer** total collective invocations than dense (1217 vs 1253): there are NO extra sparse selection collectives to delete. Composition differs per layer — sparse: 3x `all-gather` at `custom_ops/mla_attention.py:2590` + 1x :2134 + 1x :2525 (the distributed top-k index/score exchange) ; dense: 2x `all_gather` at `attention_interface.py:914/918` + 2x `all-gather` at :922/926 + 1x at `flash_attn_mla.py:179` (the dcp q/KV exchange + LSE combine). Sparse's exchanges carry bigger payloads (top-2048 indices+scores per rank), so its all-gather family costs +6.4 ms/step MORE on fewer calls; its 232 all-reduces also run +6.2 ms/step slower than the identical dense set (HBM/ICI contention from 5.5x-vs-2.0x concurrent DMA streams).

## Where the ~120 ms/step sparse-specific device delta lands

| group | delta ms/step | of the delta | nature |
|---|---:|---:|---|
| selection gathers, net (sparse selected-KV/index gathers `gather_custom_fusion` +60.1, minus the dense KV write/update machinery sparse avoids -23.0) | +37.1 | 31% | SPARSE-SPECIFIC |
| selection sort/top-k (`indexer_kernel.py:613` top-k 63/step + `mla_attention.py:1080` position-sort 78/step) | +34.7 | 29% | SPARSE-SPECIFIC |
| selection glue (extra fusions, copies, broadcast-selects, control flow) | +22.4 | 19% | SPARSE-SPECIFIC |
| attend swap: `dsa_sparse_decode` 12.65 replaces dense `MLA-d` custom-calls 8.8 | +3.9 | 3% | SPARSE-SPECIFIC (net) |
| collectives: SAME/fewer invocations, slower + bigger payloads (+6.2 on the identical 232 all-reduces; +6.4 net all-gather family) | +12.6 | 11% | CONTENTION + payload, not extra calls |
| MoE gmm: identical kernels, identical 150 calls/step, ~25% slower under sparse | +9.3 | 8% | CONTENTION (HBM DMA 5.5x vs 2.0x) |
| **total** | **+120.0** | 100% | |

Common-mode floor (present nearly identically in BOTH arms, ms/step): all-reduce ~125-131 (232/step), all-to-all 6.3-6.4 (312/step), psum+pmax ~6.9 (159/step), MoE gmm 37-46 (150/step), MoE router top-k (`fused_moe_gmm.py:545`, 225/step, <1 ms), sampling/inter-step gap ~14-15. Common-mode total ~190-205 ms/step — i.e. even deleting ALL sparse-specific cost only reaches the dense arm's ~270 ms/step, which is itself 64% collectives.

## Strongest conclusions for optimization targeting (exact semantics only)

1. **The TP-32 all-reduce floor (~125 ms/step, 232/step, bit-identical count in both arms) is the single biggest lever and it pays in BOTH arms.** Nothing sparse-specific touches it: it is per-layer single-token activation reduction latency (~0.54 ms each, largest 1.8-3.1 ms). Exact-semantics attacks: fuse consecutive reductions per layer, reduce-scatter+all-gather restructuring, ICI-topology-aware scheduling. Halving it buys ~63 ms/step in sparse AND dense.
2. **The sparse-specific budget is ~120 ms/step and 78% of it is the selection machinery, not collectives.** Net selection gathers +37 (gross +60 at `sparse_mla_kernel.py:635` take_along_axis), sort/top-k +35, glue +22 = ~94 ms/step, vs the attend swap net +3.9. There are NO extra sparse collective invocations to remove (sparse runs 36 fewer than dense). Exact-semantics attacks: fold the selected-KV gather into the `dsa_sparse_decode` Pallas kernel (index-driven DMA instead of materializing gathered KV), and replace the full `sort` position-restore at `mla_attention.py:1080` + XLA `top_k` with a partial/fused Pallas top-2048 (the values are exact top-k either way).
3. **~15 ms/step of the sparse delta is pure contention, recoverable for free by fixing #2.** The identical 232 all-reduces run +6.2 ms/step slower and the identical 150 gmm calls +9.3 ms/step slower under sparse; async-DMA concurrency is 5.5 streams/host (sparse) vs 2.0 (dense). Shrinking the gather traffic returns the collective and MoE kernels to their dense-arm speeds — so the effective yield of the selection-machinery fix is ~94+15 ≈ 110 of the 120 ms/step.

Method note: aggregation identical to the sparse report (`parse_xplane.py` nested self-time sweep, per-core means over 64 planes, 15 steps; dense agg in `dense_all_hosts_agg.json`). Per-invocation counts from w0 core 0; counts verified SPMD-identical across cores.

