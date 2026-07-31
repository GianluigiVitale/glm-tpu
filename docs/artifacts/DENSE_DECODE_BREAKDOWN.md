# E0 dense decode — device-side XPlane breakdown

Trace: `/home/gianl/glm-run/e0cap_dense_20260731T043422Z/trace` — 8 hosts, 64 TPU-v4 TensorCore planes (8 `/device:TPU:N` planes per host), 15 decode steps (`jit_step_fun_impl`), single sequence, 262144-token context, **DENSE MLA** (no DSA selection), context-parallel dcp=8 (per-rank ~32K contiguous KV), TP-32. Measured un-profiled wall: 260.8 ms/step.

Method: identical to the sparse breakdown — nested-aware self-time sweep over the `XLA Ops` line of every device plane; step boundaries from `XLA Modules`; per-core means over all 64 cores (ops are SPMD-identical).

## Window / step accounting

- Device op window: **4228.6 ms** -> 281.9 ms per step-cycle (incl. inter-step host gaps)
- Mean `jit_step_fun_impl` device duration: **269.35 ms** (min 262.3, max 275.8 across all cores/steps)
- Mean inter-step gap: **13.4 ms** per step (sparse arm: 13.9 ms — same host-side sampling gap)
- Device busy: mean 4005.8 ms of 4228.6 ms window -> **idle 5.27%** (range 5.24-5.32% across cores)
- Un-profiled wall was 260.8 ms/step; profiled device step is 269.3 ms and step-cycle 281.9 ms — ~3-8% profiling overhead, consistent with the sparse arm (wall 397-440 vs device 389.4).

## (a) Per-category self-time (per core)

| category | total ms (15 steps) | ms/step | % of busy | % of step-cycle |
|---|---:|---:|---:|---:|
| collectives | 2565.2 | 171.02 | 64.0% | 60.7% |
| pallas: gmm grouped matmul (MoE experts) | 557.1 | 37.14 | 13.9% | 13.2% |
| gather/scatter/dyn-slice | 434.7 | 28.98 | 10.9% | 10.3% |
| compute (fusion/dot/conv) | 139.5 | 9.30 | 3.5% | 3.3% |
| custom-call: other (fp8 weight formatting etc.) | 132.2 | 8.81 | 3.3% | 3.1% |
| data movement (copy/transpose/reshape) | 115.0 | 7.67 | 2.9% | 2.7% |
| async start/done markers | 35.2 | 2.35 | 0.9% | 0.8% |
| sort/top-k | 13.7 | 0.91 | 0.3% | 0.3% |
| control-flow self-time (conditional/while) | 13.1 | 0.88 | 0.3% | 0.3% |
| *device idle (gaps, mostly between steps)* | 222.8 | 14.85 | - | 5.3% |

Note the category `custom-call: other` is NOT fp8-formatting here: it is the **dense MLA decode kernel itself** (`MLA-d-bq_1-bkvp_1-p_512-bsz_1`, `kernels/mla/v2/kernel.py:2936`, 78/step = one per layer, 7.50 ms/step) plus its two `xpose_pipeline` transposes (~1.0 ms/step).

## (b) Top 25 ops by self-time (per core)

| op (base name) | self ms | ms/step | total ms | count | category |
|---|---:|---:|---:|---:|---|
| `all-reduce` | 1872.1 | 124.81 | 1872.1 | 3480 | collectives |
| `gmm_v2-g_8-m_256-k_6144-act_silu-n_4096-tm_64-tk_3072-tn_128` | 377.4 | 25.16 | 377.4 | 1125 | pallas: gmm grouped matmul (MoE experts) |
| `all-gather` | 297.9 | 19.86 | 297.9 | 5910 | collectives |
| `scatter_custom_fusion` | 265.9 | 17.72 | 265.9 | 44475 | gather/scatter/dyn-slice |
| `all_gather` | 195.1 | 13.01 | 195.1 | 2340 | collectives |
| `gmm_v2-g_8-m_256-k_2048-act_None-n_6144-tm_64-tk_2048-tn_512` | 179.7 | 11.98 | 179.7 | 1125 | pallas: gmm grouped matmul (MoE experts) |
| `MLA-d-bq_1-bkvp_1-p_512-bsz_1` | 112.5 | 7.50 | 112.5 | 1170 | custom-call: other (fp8 weight formatting etc.) |
| `all-to-all` | 96.3 | 6.42 | 96.3 | 4680 | collectives |
| `fusion` | 88.6 | 5.90 | 88.6 | 39795 | compute (fusion/dot/conv) |
| `gather_custom_fusion` | 69.6 | 4.64 | 69.6 | 5805 | gather/scatter/dyn-slice |
| `psum` | 63.0 | 4.20 | 63.0 | 1215 | collectives |
| `broadcast_in_dim` | 50.9 | 3.39 | 50.9 | 49170 | data movement (copy/transpose/reshape) |
| `constant_dynamic-slice_fusion` | 47.9 | 3.19 | 47.9 | 23400 | gather/scatter/dyn-slice |
| `pmax` | 40.8 | 2.72 | 40.8 | 1170 | collectives |
| `bitcast_dynamic-update-slice_fusion` | 35.6 | 2.37 | 35.6 | 22230 | gather/scatter/dyn-slice |
| `slice-done` | 29.5 | 1.97 | 29.5 | 32400 | async start/done markers |
| `copy` | 24.4 | 1.62 | 24.4 | 16440 | data movement (copy/transpose/reshape) |
| `reshape` | 16.0 | 1.07 | 16.0 | 11580 | data movement (copy/transpose/reshape) |
| `while` | 13.1 | 0.88 | 436.1 | 1185 | control-flow self-time (conditional/while) |
| `dynamic_slice` | 12.8 | 0.85 | 12.8 | 22245 | gather/scatter/dyn-slice |
| `slice` | 10.0 | 0.67 | 10.0 | 8145 | data movement (copy/transpose/reshape) |
| `xpose_pipeline_shape_64x32x512_xpose_1x0x2_n_tile_64_m_tile_16_pa_0_pi` | 8.0 | 0.54 | 8.0 | 1170 | custom-call: other (fp8 weight formatting etc.) |
| `broadcast` | 7.9 | 0.53 | 7.9 | 22305 | data movement (copy/transpose/reshape) |
| `sort` | 7.8 | 0.52 | 7.8 | 2265 | sort/top-k |
| `xpose_pipeline_shape_32x64x512_xpose_1x0x2_n_tile_32_m_tile_16_pa_0_pi` | 7.6 | 0.50 | 7.6 | 1170 | custom-call: other (fp8 weight formatting etc.) |

Source attribution (event metadata `source` stat, w0 core 0):

- `MLA-d-bq_1-bkvp_1-p_512-bsz_1` — `tpu_inference/kernels/mla/v2/kernel.py:2936` (the paged dense-MLA decode Pallas kernel; 78/step, 7.50 ms/step, i.e. 0.096 ms/layer to attend 262K tokens under dcp=8).
- `scatter_custom_fusion` (17.7 ms/step, 2965/step) + `constant_dynamic-slice_fusion` + `bitcast_dynamic-update-slice_fusion` — `tpu_inference/layers/common/attention_interface.py:1148-1247`: the dense KV-cache write/update machinery (dense pays this instead of the sparse arm's selection gathers).
- `all-gather`/`all_gather` — `attention_interface.py:914/918/922/926` + `layers/vllm/backends/flash_attn_mla.py:179`: the dcp=8 context-parallel q/KV exchange + LSE combine, ~5 per layer.
- `gather_custom_fusion` here is almost all MoE (`layers/common/fused_moe_gmm.py`), not attention.

## (c) Per-host step durations (straggler check)

| host | mean | std | min | max |
|---|---:|---:|---:|---:|
| w0 | 268.24 | 2.10 | 264.51 | 271.93 |
| w1 | 269.76 | 2.83 | 265.69 | 275.79 |
| w2 | 270.01 | 2.89 | 263.78 | 275.79 |
| w3 | 269.25 | 2.81 | 263.52 | 275.79 |
| w4 | 271.09 | 2.82 | 265.27 | 275.79 |
| w5 | 268.95 | 3.31 | 262.69 | 275.79 |
| w6 | 268.55 | 3.42 | 262.99 | 275.79 |
| w7 | 268.94 | 2.93 | 263.01 | 275.79 |

Per-step spread across hosts (max-min of host means, ms): 7.8, 6.2, 5.1, 4.8, 4.7, 4.9, 5.5, 3.6, 4.6, 4.3, 3.0, 3.2, 4.0, 3.1, 3.6

Step durations, host w0 core 0 (ms): 268.2, 272.2, 267.1, 268.7, 266.0, 270.8, 269.9, 265.8, 266.6, 268.3, 271.7, 269.2, 270.2, 264.9, 268.8

## Busy vs idle per core (summary)

All 64 cores: busy 4003.8-4007.0 ms of 4228.6 ms window; idle 5.24-5.32%. No stragglers.

## Async DMA line (overlaps compute; not on the critical-path tables)

| async op | total ms (all hosts) | count |
|---|---:|---:|
| slice | 38039.6 | 259200 |
| async-copy | 28229.8 | 150840 |

Average concurrent async-DMA streams per host: ~2.0 (sparse arm: ~5.5 — the sparse selection gathers keep 2.75x more DMA in flight).

## Collectives per-invocation detail (w0 core 0)

| collective | invocations/step | mean self per invocation | ms/step |
|---|---:|---:|---:|
| all-reduce | 232 | 0.544 ms | 126.2 |
| all-gather | 394 | 0.056 ms | 22.0 |
| all_gather | 156 | 0.090 ms | 14.1 |
| all-to-all | 312 | 0.014 ms | 4.2 |
| psum | 81 | 0.052 ms | 4.2 |
| pmax | 78 | 0.033 ms | 2.5 |

**232 all-reduces/step — the same count as the sparse arm** (~0.54 ms each): the TP-32 single-token all-reduce floor is common-mode, not sparse overhead. Dense collectives total 171.0 ms/step = **63.5% of the dense step's busy time** — dense decode at 262K/dcp=8 is even more collective-dominated than sparse decode.

## Strongest observations (dense arm)

1. **Dense decode is a collective-latency machine.** 171 ms/step (64% of busy) is collectives; all-reduce alone is 124.8 ms/step over the identical 232 invocations the sparse arm runs. The attention itself (`MLA-d` kernel) is 7.5 ms/step — 2.7% of the step.
2. **Attending 262K densely costs less device time than selecting 2048 sparsely.** Dense MLA attend+KV machinery ≈ 7.5 + 23.3 (scatter/dyn-slice writes) + 33.1 (dcp all-gathers) ms/step, while the sparse arm's selection stack (top-k/sort/gathers/glue) costs ~95 ms/step on top of its own attend. At this context length the v4 MLA kernel under dcp=8 reads per-rank 32K contiguous KV at streaming bandwidth.
3. **No stragglers, same idle floor.** Idle is 5.3% (~15 ms/step inter-step host gap, same absolute gap as sparse); per-host means agree within ~2 ms.

