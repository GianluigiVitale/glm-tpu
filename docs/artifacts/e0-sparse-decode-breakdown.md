# E0 sparse decode — device-side XPlane breakdown

Trace: `/home/gianl/glm-run/e0cap_sparse_20260730T235238Z/trace` — 8 hosts, 64 TPU-v4 TensorCore planes (8 `/device:TPU:N` planes per host), 15 decode steps (`jit_step_fun_impl`), single sequence, 262144-token context, sparse DSA attention, dcp=8, TP-32.

Method: nested-aware self-time sweep over the `XLA Ops` line of every device plane (control-flow container ops — `conditional`/`while` — contain child ops on the same line; naive sums double-count by ~50%). Step boundaries from the `XLA Modules` line. All ms values are per-core means over all 64 cores unless noted; ops are SPMD-identical across cores.

## Window / step accounting

- Device op window (first to last op): **6037.9 ms** -> 402.5 ms per step-cycle (includes inter-step host gaps)
- Mean `jit_step_fun_impl` device duration: **389.42 ms** (min 382.6, max 397.7 across all cores/steps)
- Mean inter-step gap (device idle between modules, incl. tiny sampling programs): **13.9 ms** per step
- Device busy: mean 5805.9 ms of 6037.9 ms window -> **idle 3.84%** (range 3.82-3.92% across cores)
- Profiled wall step time was ~397-440 ms; device step-cycle here is ~403 ms mean.

## (a) Per-category self-time (per core)

Denominators: % busy = share of device busy time (5806 ms); % step-cycle = share of 402.5 ms x 15 window.

| category | total ms (15 steps) | ms/step | % of busy | % of step-cycle |
|---|---:|---:|---:|---:|
| collectives | 2754.7 | 183.65 | 47.4% | 45.6% |
| gather/scatter/dyn-slice | 991.6 | 66.11 | 17.1% | 16.4% |
| pallas: gmm grouped matmul (MoE experts) | 696.0 | 46.40 | 12.0% | 11.5% |
| sort/top-k | 534.0 | 35.60 | 9.2% | 8.8% |
| compute (fusion/dot/conv) | 343.7 | 22.91 | 5.9% | 5.7% |
| data movement (copy/transpose/reshape) | 226.7 | 15.12 | 3.9% | 3.8% |
| pallas: dsa_sparse_decode (sparse MLA attend) | 189.8 | 12.65 | 3.3% | 3.1% |
| control-flow self-time (conditional/while) | 51.8 | 3.45 | 0.9% | 0.9% |
| async start/done markers | 17.3 | 1.16 | 0.3% | 0.3% |
| custom-call: other (fp8 weight formatting etc.) | 0.3 | 0.02 | 0.0% | 0.0% |
| *device idle (gaps, mostly between steps)* | 231.9 | 15.46 | - | 3.8% |

## (b) Top 25 ops by self-time (per core)

| op (base name) | self ms | ms/step | total ms | count | category |
|---|---:|---:|---:|---:|---|
| `all-reduce` | 1964.7 | 130.98 | 1964.7 | 3480 | collectives |
| `gather_custom_fusion` | 970.7 | 64.71 | 970.7 | 28560 | gather/scatter/dyn-slice |
| `all-gather` | 547.7 | 36.51 | 547.7 | 7080 | collectives |
| `gmm_v2-g_8-m_256-k_6144-act_silu-n_4096-tm_64-tk_3072-tn_128` | 469.8 | 31.32 | 469.8 | 1125 | pallas: gmm grouped matmul (MoE experts) |
| `top_k` | 352.1 | 23.48 | 352.1 | 4320 | sort/top-k |
| `gmm_v2-g_8-m_256-k_2048-act_None-n_6144-tm_64-tk_2048-tn_512` | 226.2 | 15.08 | 226.2 | 1125 | pallas: gmm grouped matmul (MoE experts) |
| `dsa_sparse_decode` | 189.8 | 12.65 | 189.8 | 1170 | pallas: dsa_sparse_decode (sparse MLA attend) |
| `sort` | 181.9 | 12.13 | 181.9 | 3735 | sort/top-k |
| `fusion` | 172.9 | 11.53 | 172.9 | 84885 | compute (fusion/dot/conv) |
| `copy` | 167.9 | 11.19 | 167.9 | 18345 | data movement (copy/transpose/reshape) |
| `broadcast_select_fusion` | 99.1 | 6.61 | 99.1 | 1185 | compute (fusion/dot/conv) |
| `all-to-all` | 94.8 | 6.32 | 94.8 | 4680 | collectives |
| `psum` | 63.7 | 4.25 | 63.7 | 1215 | collectives |
| `while` | 44.6 | 2.97 | 319.9 | 330 | control-flow self-time (conditional/while) |
| `pmax` | 42.5 | 2.83 | 42.5 | 1170 | collectives |
| `all_gather` | 41.3 | 2.75 | 41.3 | 630 | collectives |
| `slice` | 18.6 | 1.24 | 18.6 | 10065 | data movement (copy/transpose/reshape) |
| `broadcast_in_dim` | 18.3 | 1.22 | 18.3 | 5865 | data movement (copy/transpose/reshape) |
| `reshape` | 17.0 | 1.13 | 17.0 | 10425 | data movement (copy/transpose/reshape) |
| `dynamic_slice` | 11.9 | 0.80 | 11.9 | 20475 | gather/scatter/dyn-slice |
| `slice-done` | 10.9 | 0.72 | 10.9 | 35640 | async start/done markers |
| `scatter_custom_fusion` | 8.5 | 0.57 | 8.5 | 1485 | gather/scatter/dyn-slice |
| `pad_clamp_fusion` | 8.5 | 0.56 | 8.5 | 23955 | compute (fusion/dot/conv) |
| `conditional` | 7.2 | 0.48 | 2667.2 | 1485 | control-flow self-time (conditional/while) |
| `iota_add_fusion` | 7.1 | 0.48 | 7.1 | 20475 | compute (fusion/dot/conv) |

## (c) Per-host step durations (straggler check)

Mean `jit_step_fun_impl` duration per host (mean of its 8 cores), ms:

| host | mean | std | min | max |
|---|---:|---:|---:|---:|
| w0 | 388.29 | 2.80 | 384.11 | 394.30 |
| w1 | 390.85 | 3.07 | 385.45 | 395.61 |
| w2 | 388.10 | 3.21 | 383.13 | 395.29 |
| w3 | 389.41 | 3.15 | 384.51 | 395.28 |
| w4 | 389.82 | 3.04 | 384.68 | 395.28 |
| w5 | 388.23 | 3.20 | 383.66 | 395.28 |
| w6 | 391.61 | 2.95 | 387.20 | 397.34 |
| w7 | 389.06 | 3.38 | 384.75 | 395.28 |

Per-step spread across hosts (max-min of host means, ms): 9.8, 3.0, 4.2, 8.8, 4.6, 4.7, 4.4, 4.6, 4.1, 3.3, 5.1, 4.9, 4.2, 4.2, 4.0

Step durations, host w0 core 0 (ms): 385.8, 388.9, 386.8, 391.6, 386.7, 387.8, 388.0, 386.9, 384.3, 385.9, 386.6, 394.5, 391.1, 392.7, 390.3

## (d) Busy vs idle per core

| worker | plane | busy ms | window ms | idle % |
|---|---|---:|---:|---:|
| w0 | /device:TPU:0 | 5804.3 | 6037.8 | 3.87% |
| w0 | /device:TPU:1 | 5804.5 | 6037.8 | 3.86% |
| w0 | /device:TPU:2 | 5804.0 | 6037.8 | 3.87% |
| w0 | /device:TPU:3 | 5804.2 | 6037.8 | 3.87% |
| w0 | /device:TPU:4 | 5805.7 | 6037.7 | 3.84% |
| w0 | /device:TPU:5 | 5805.9 | 6037.7 | 3.84% |
| w0 | /device:TPU:6 | 5805.4 | 6037.8 | 3.85% |
| w0 | /device:TPU:7 | 5805.6 | 6037.8 | 3.85% |
| w1 | /device:TPU:0 | 5806.8 | 6037.9 | 3.83% |
| w1 | /device:TPU:1 | 5807.0 | 6037.9 | 3.82% |
| w1 | /device:TPU:2 | 5807.0 | 6037.9 | 3.82% |
| w1 | /device:TPU:3 | 5807.2 | 6037.9 | 3.82% |
| w1 | /device:TPU:4 | 5806.4 | 6037.9 | 3.83% |
| w1 | /device:TPU:5 | 5806.6 | 6037.9 | 3.83% |
| w1 | /device:TPU:6 | 5806.5 | 6037.9 | 3.83% |
| w1 | /device:TPU:7 | 5806.7 | 6037.9 | 3.83% |
| w2 | /device:TPU:0 | 5805.6 | 6037.9 | 3.85% |
| w2 | /device:TPU:1 | 5805.8 | 6037.9 | 3.84% |
| w2 | /device:TPU:2 | 5804.6 | 6037.9 | 3.86% |
| w2 | /device:TPU:3 | 5804.8 | 6037.9 | 3.86% |
| w2 | /device:TPU:4 | 5801.4 | 6037.9 | 3.92% |
| w2 | /device:TPU:5 | 5801.6 | 6037.9 | 3.91% |
| w2 | /device:TPU:6 | 5805.0 | 6037.9 | 3.86% |
| w2 | /device:TPU:7 | 5805.2 | 6037.9 | 3.85% |
| w3 | /device:TPU:0 | 5806.8 | 6037.9 | 3.83% |
| w3 | /device:TPU:1 | 5807.0 | 6037.9 | 3.82% |
| w3 | /device:TPU:2 | 5806.9 | 6037.9 | 3.83% |
| w3 | /device:TPU:3 | 5807.1 | 6037.9 | 3.82% |
| w3 | /device:TPU:4 | 5805.8 | 6037.9 | 3.84% |
| w3 | /device:TPU:5 | 5806.0 | 6037.9 | 3.84% |
| w3 | /device:TPU:6 | 5805.9 | 6037.9 | 3.84% |
| w3 | /device:TPU:7 | 5806.1 | 6037.9 | 3.84% |
| w4 | /device:TPU:0 | 5807.3 | 6037.9 | 3.82% |
| w4 | /device:TPU:1 | 5807.5 | 6037.9 | 3.82% |
| w4 | /device:TPU:2 | 5807.1 | 6037.9 | 3.82% |
| w4 | /device:TPU:3 | 5807.3 | 6037.9 | 3.82% |
| w4 | /device:TPU:4 | 5807.2 | 6037.9 | 3.82% |
| w4 | /device:TPU:5 | 5807.4 | 6037.9 | 3.82% |
| w4 | /device:TPU:6 | 5807.1 | 6037.9 | 3.82% |
| w4 | /device:TPU:7 | 5807.3 | 6037.9 | 3.82% |
| w5 | /device:TPU:0 | 5804.2 | 6037.9 | 3.87% |
| w5 | /device:TPU:1 | 5804.4 | 6037.9 | 3.87% |
| w5 | /device:TPU:2 | 5803.9 | 6037.9 | 3.88% |
| w5 | /device:TPU:3 | 5804.1 | 6037.9 | 3.87% |
| w5 | /device:TPU:4 | 5804.0 | 6037.9 | 3.87% |
| w5 | /device:TPU:5 | 5804.2 | 6037.9 | 3.87% |
| w5 | /device:TPU:6 | 5804.6 | 6037.9 | 3.86% |
| w5 | /device:TPU:7 | 5804.8 | 6037.9 | 3.86% |
| w6 | /device:TPU:0 | 5807.1 | 6037.9 | 3.82% |
| w6 | /device:TPU:1 | 5807.3 | 6037.9 | 3.82% |
| w6 | /device:TPU:2 | 5806.6 | 6037.9 | 3.83% |
| w6 | /device:TPU:3 | 5806.8 | 6037.9 | 3.83% |
| w6 | /device:TPU:4 | 5807.1 | 6037.9 | 3.82% |
| w6 | /device:TPU:5 | 5807.3 | 6037.9 | 3.82% |
| w6 | /device:TPU:6 | 5806.6 | 6037.9 | 3.83% |
| w6 | /device:TPU:7 | 5806.8 | 6037.9 | 3.83% |
| w7 | /device:TPU:0 | 5805.4 | 6037.9 | 3.85% |
| w7 | /device:TPU:1 | 5805.6 | 6037.9 | 3.85% |
| w7 | /device:TPU:2 | 5806.8 | 6037.9 | 3.83% |
| w7 | /device:TPU:3 | 5807.0 | 6037.9 | 3.82% |
| w7 | /device:TPU:4 | 5806.6 | 6037.9 | 3.83% |
| w7 | /device:TPU:5 | 5806.8 | 6037.9 | 3.83% |
| w7 | /device:TPU:6 | 5806.8 | 6037.9 | 3.83% |
| w7 | /device:TPU:7 | 5807.0 | 6037.9 | 3.82% |

## Async DMA line (`Async XLA Ops`, only present on plane TPU:0 of each host)

These events overlap compute (DMA engines), so they are NOT part of the critical-path table above. Totals summed over the hosts that carry them:

| async op | total ms (all hosts) | count |
|---|---:|---:|
| slice | 136755.7 | 285120 |
| async-copy | 130906.3 | 329400 |

Average concurrent async-DMA streams per host during the window: ~5.5 (i.e. DMA engines busy ~5.5x window; these overlap TensorCore compute).

## Collectives per-invocation detail (w0 core 0)

| collective | invocations/step | mean self per invocation | ms/step |
|---|---:|---:|---:|
| all-reduce | 232 | 0.565 ms | 131.0 |
| all-gather | 472 | 0.077 ms | 36.5 |
| all-to-all | 312 | 0.020 ms | 6.3 |
| psum | 81 | 0.052 ms | 4.2 |
| pmax | 78 | 0.036 ms | 2.8 |
| all_gather | 42 | 0.066 ms | 2.8 |

232 all-reduces/step at ~0.55 ms each: batch=1 single-token activations over TP-32 spanning hosts -> pure ICI latency, not bandwidth. The largest individual all-reduce instances run 2.2-3.1 ms per invocation.

## Decode vs prefill-derived guesses

Prior guesses were extrapolated from PREFILL traces. Measured decode (% of device busy; % of step-cycle in parens):

| bucket | prefill-derived guess | decode measured | verdict |
|---|---:|---:|---|
| top-k/sort | 21.8% | 9.2% (8.8%) | overestimated ~2.4x — top-k does not scale with batch the way prefill implied |
| gathers/scatters | 16% | 17.1% (16.4%) | matches |
| collectives | ~24% | 47.4% (45.6%) | ~2x underestimated — decode is collective-latency dominated |

Additional decode-only findings not visible in the prefill extrapolation:

- `dsa_sparse_decode` (the actual sparse MLA attend over the selected 2048 tokens) is only 12.7 ms/step (3.3% of busy). The *selection machinery* around it (top-k + sort + index/KV gathers + psum/pmax combines) costs ~100 ms/step — roughly 8x the attend itself.
- MoE expert compute (`gmm_v2` grouped-matmul Pallas kernels, fp8 weights) is 46.4 ms/step (12.0% of busy).
- ~32,300 device ops execute per step per core (484,680 over 15 steps); mean op duration ~12 us — the graph is fragmented into many tiny latency-bound ops.

## Validation against xprof's C++ analyzer

`hlo_stats` run via the xprof C API on w0 agrees with this parser to <1%: total device self-time 5804.9 ms/core (C++) vs 5804.3-5805.9 ms (this parser); `gather_custom_fusion` 971.2 vs 970.7 ms; `top_k` 352.1 vs 352.1 ms; `dsa_sparse_decode` 189.8 vs 189.8 ms. Differences in the fleet table come from averaging over all 64 cores vs w0 only. Source attribution from hlo_stats: the dominant gathers originate at `tpu_inference/kernels/dsa/sparse_mla_kernel.py:635` (take_along_axis KV gather) inside the sparse-MLA path of the deepseek_v2-family model.

## Strongest observations

1. **Decode is collective-latency bound, not compute or idle bound.** Collectives consume 46% of the step-cycle (184 ms/step), 2x the prefill-derived guess; all-reduce alone is 131 ms/step across 232 invocations (~0.55 ms each) — single-token payloads on TP-32, i.e. a pure latency floor. Fusing/batching reductions (or reduce-scatter+all-gather restructuring) is the biggest lever.
2. **The DSA selection machinery dwarfs the sparse attend.** top-k+sort (36 ms/step) plus gathers (66 ms/step) vs 12.7 ms/step for `dsa_sparse_decode` itself. The prefill-derived 21.8% top-k share halves at decode (8.8%), but selection+gather together still cost ~25% of the step.
3. **No stragglers, minimal idle.** All 64 cores are 96.2% busy; idle is almost entirely the ~14 ms inter-step host gap (sampling programs). Per-host step means agree within 3.5 ms, per-step cross-host spread <10 ms. Device step-cycle is ~403 ms, matching the low end of the 397-440 ms wall measurement; the 440 ms tail is host-side, not device-side.

## Parsing notes / limitations

- Step boundaries ARE identifiable (15 `jit_step_fun_impl` module events per core); per-category tables aggregate the whole window and divide by 15 — per-step category attribution (event-midpoint binning) is also stored in `all_hosts_agg.json`.
- Self-time requires nesting reconstruction on the `XLA Ops` line (conditional/while contain children); naive duration sums overcount by ~50%.
- Cross-host clocks are not aligned in these XSpaces (device line timestamp_ns=0), so straggler analysis uses per-host step *durations*, not absolute start skew.
- Each host exposes 8 `/device:TPU:N` planes (64 total) with `has_megacore=1`; numbers are per TensorCore plane.
- The `Async XLA Ops` DMA line exists only on plane TPU:0 of each host and overlaps compute; it is excluded from the critical-path tables.
- `%time stalled by DMA` from hlo_stats reports 0 for the big gathers; their measured HBM BW (~19 GiB/s effective on the biggest gather) indicates scattered-row access, not sequential streaming.

