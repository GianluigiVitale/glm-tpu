# Prefill/decode speed: what the Kaggle TPU reference engines do differently, and the low-hanging fruit for WS32

Date: 2026-09-19. Branch `perf/reference-lowhanging-fruit-20260919` from main
`493b67de`. Scope: a source-level comparison of the two recent engines in
[ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (folders
`glm53-flash` and `qwen38-27b`, cloned at commit `1aa1f08`, 2026-09-15) against
this repository's WS32 decode and layer-major prefill, a quantified ranking of
the structural differences, and an opt-in implementation of the two decode
items that could be proven on the CPU mesh. Sections 1-4 are the
source analysis; section 7 holds the TPU v4 measurements taken the same day on
the pod with synthetic weights (`tools/perf_tpu_microbench.py`), which
correct several of the estimates in section 4 and are the numbers to plan by.
Nothing here is a protected/sealed result. The frozen `MODEL_SOURCE` pin
`edecdd94` is untouched: all new code lives in `glm_tpu/perf/`, `tests/perf/`
and `tools/perf_*.py`.

## 1. Baselines being compared

| Engine | Hardware | Model | Prefill | Decode, one stream | Source |
|---|---|---|---:|---:|---|
| WS32 (this repo) | 32x TPU v4, 8 hosts | GLM-5.2-FP8, 753B/40B active, 78 layers, DSA top-2048 | 62.8 tok/s at 2K (DB610); 45.5 at 128K (DB619); 32.2 at 256K (DB620) | 7.66 tok/s greedy (DB610, 130.6 ms p50); 6.57 tok/s sampled (DB621, 148.0 ms p50) | `docs/artifacts/prefill-*-sealed-*.json`, `docs/release/user-response-db621-sealed-20260914.json` |
| glm53-flash | 8x TPU v5e (Kaggle) | GLM-5.3-Flash, 320B/18B active, 45 layers (34 KDA linear + 11 MLA/DSA with k-pool), 3-bit experts + int8 | ~1,500-1,800 tok/s | ~64 tok/s (~90 aggregate at 3 streams) | `glm53-flash/README.md` |
| qwen38-27b | 8x TPU v5e (Kaggle) | Qwen3.8-27B bf16 dense hybrid (48 DeltaNet + 16 attention) | ~10,300 tok/s | ~130 tok/s with MTP speculative decoding (78 without) | `qwen38-27b/README.md` (vllm-tpu 0.28.0 + one patch) |

Hardware scale: 32 v4 chips have roughly 5.6x the bf16 MXU rate and 5.9x the
aggregate HBM bandwidth of 8 v5e chips; GLM-5.2 moves 2.2x the active
parameters of GLM-5.3-Flash per token and stores them at 8 bits instead of
3, over 78 layers instead of 45 with a 2,048-key exact indexer instead of a
pooled one. A like-for-like structural ceiling is therefore not "the same
numbers"; it is closer to a few thousand prompt tokens/s and several tens of
generated tokens/s, and the current engine is one to two orders of magnitude
below that on both axes. The gap is fixed per-operation overhead, not
arithmetic or bandwidth (section 3).

## 2. What the reference engines do that WS32 does not

Ranked by how directly each maps onto a WS32 cost that is already measured.

1. **One kernel for all routed experts of a step, expert id from scalar
   prefetch** (`glm53/pallas_moe.py::moe_matvec`, `resident.py`). The grid
   is the route slots; each step DMAs exactly the chosen expert; unowned
   work does not exist. WS32 runs eight `lax.cond` blocks per sparse layer,
   each with three 192-step Pallas grids and its own feature-4 `psum`
   (`kernels/ws32.py::ws32_moe_pallas_from_routes_mapped`).
2. **Candidate top-k sampling, no vocabulary sort** (`engine.py::device_sample`:
   "XLA sorts are slow on the TPU; the sort variant cost ~2 ms per token").
   WS32's `nucleus_sample` stable-sorts the 154,880-entry row with two keys and
   runs two full-length cumsums per token.
3. **Sequence-sharded attention with partial softmax + reduce-scatter of
   outputs** (`model.py::_attend`, `seq_shard > 1`): queries are all-gathered
   (small), every chip attends to its own keys for all heads, outputs and
   log-sum-exp are `psum_scatter`ed over heads. WS32 does the opposite:
   each chip gathers its owned selected KV rows into a `[rows, 2048, 640]`
   BF16 tensor padded with zeros and `psum`s the whole tensor over expert-8
   (`ws32_layer.py`/`ws32_prefill_attention.py`, scope
   `selected_cache_expert_exchange`): 2.6 MB per layer per decode step, 84 MB
   per 32-row prefill tile per layer.
4. **Two-stage exact top-k across chips with a cut check and fallback**
   (`model.py::indexer_select`): local top-`k_local` (128) per chip,
   all-gather, global top-K, redo with the full K only if a chip's list was
   cut. WS32 selects the full top-2048 on every chip from a
   position-argsorted row, all-gathers 8x2048 candidates and argsorts/top-ks
   the union again (`reference/dsa.py::local_topk_candidates`,
   `_merge_topk_candidates_scored`); in prefill the same chain runs once per
   512-key tile (`prefill_dsa.py::causal_dsa_local_candidates`).
5. **Nothing leaves the device between steps**: token ids, positions, PRNG
   key and sampler state stay resident (`DeviceSampler`, "per-step scalar
   transfers cost ~2 ms"). WS32 per token: `block_until_ready`, four separate
   device-to-host reads (`np.asarray` of token, health, position, length),
   three eight-host `process_allgather` votes and one `device_put` of the
   SHA-256 uniform (`runtime/ws32_request_session.py`,
   `scripts/greenfield/run_short_decoder_ws32.py::_batched_fleet_all`).
6. **Prefill in 1,024-token pieces with donated cache buffers**; expert work
   as one grouped GEMM over route-sorted rows (`moe_apply_grouped`, chunk
   256). WS32 prefill blocks are 128 rows, attention/DSA run in four rolled
   32-row tiles per layer (`ws32_prefill_window.py`), the grouped FP8 GEMM
   uses 8-row tiles or 32-row expert panels, and every block pays two host
   votes and six device-to-host reads.
7. **Fewer tiny launches**: Sinkhorn written as one elementwise fusion ("~40
   launches per layer per token = 10 ms/token on v5e"), `one_hot` instead of
   gather for route weights ("XLA gather ~2.5 ms per call"). The same
   sensitivity applies to v4 here: every Pallas grid step costs about 0.35 us
   of fixed overhead, so a 192-step one-row projection is ~70 us of overhead
   for 3 MB of weights that stream in 2.5 us.
8. **Speculative decoding with the model's own MTP head** (qwen38-27b, +34%
   decode with lossless verification, vllm-tpu patch). GLM-5.2 ships
   `num_nextn_predict_layers = 1`; the WS32 runtime never uses it and the
   release explicitly excludes speculative decoding.
9. **Persistent compile cache and warm serving**: startup 16-22 min on Kaggle
   with a shipped XLA cache versus the WS32 controller's ~38 min cold load and
   compile per invocation (DB621). Orthogonal to tok/s but the same lever.
10. **Continuous batching / prefix caching** (glm53 scheduler): not a WS32 goal
    (one live row), listed for completeness.

## 3. Where the WS32 decode step spends its time (evidence, not a profile)

Recorded facts that bound the decomposition of the 130.6 ms greedy step:

* The protected WS32 real-layer runs (`docs/greenfield/WS32_2D_PROTOTYPE.md`)
  timed ONE sparse layer's MoE body at 1.257 ms p50 (normal routing) and
  2.304 ms (all eight routes on one owner) in August. Measured again on
  2026-09-19 inside one program (section 7, chained slope, no host
  dispatch): **0.444 ms normal, 0.652 ms two-per-owner, 1.898 ms
  concentrated**. Seventy-five sparse layers are therefore **~33 ms of the
  130.6 ms step**, not the ~94 ms a naive reading of the old receipt gives.
* DB621's sampled step (148.0 ms) versus DB610's greedy step (130.6 ms) is
  NOT the sampler: measured, the full-vocabulary nucleus head costs 0.98 ms
  per call against 0.58 ms for the greedy head (section 7). The 17 ms gap is
  the capacity step (166,912 versus 8,192 slots; `GATE_D_LESSONS.md` records
  +10% per capacity step) plus the sampled request loop's host work.
* Capacity 8,192 -> 131,072 costs +10% on both prefill and decode
  (`GATE_D_LESSONS.md`, 2026-09-06), and 2K vs 128K decode differ by only
  14 ms: the step is dominated by fixed per-operation costs, not by context.
* Structural census of the compiled/traced programs
  (`tools/perf_op_census.py`, receipt
  [`op-census-cpu32-20260919.json`](op-census-cpu32-20260919.json)); one
  traced layer of each kind, extrapolated to GLM's 3 dense/full, 18
  sparse/full and 57 sparse/shared layers:

| Per decode step (78 layers, excl. embedding/head) | frozen | challenger (this branch) |
|---|---:|---:|
| `psum` (feature-4 / expert-8 all-reduces) | 1,341 | 741 |
| `all_gather` | 213 | 213 |
| Pallas launches, static | 2,622 (1,800 under route `cond`s) | 897 |
| Pallas launches, executed per chip at normal routing (1 owned route/layer) | ~1,047 | 897 |
| Pallas grid steps for MoE projections per sparse layer (128x128 tiles) | 6 x 192 = 1,152 | 4 x 12 = 48 with 512x512 tiles |
| `lax.cond` | 5,943 | 2,193 |
| `sort`/`argsort` | 240 static; XLA CSE keeps ~half (compiled HLO) | 240 |

The 8-layer fixture's compiled CPU executable confirms the direction:
all-reduce 99 -> 64/66, and it shows that the per-layer page-table metadata
sorts (`stage_local._require_decode_metadata`) are de-duplicated by XLA's
CSE (32-33 traced sorts -> 15-16 compiled), so they are not a decode cost
worth chasing.

Roofline for context: at normal routing one chip reads ~2.7 GB of weights
per token (routed + shared experts, attention projections, dense layers),
which is 2.2 ms at 1.2 TB/s; arithmetic is negligible. What is NOT
negligible on v4 is decoding those FP8 bytes: measured ~35-40 us per 3 MB
(section 6.3), i.e. ~40 ms per token, the single largest identifiable cost. With ~950 small
collectives per step at ~20-30 us each the collective floor of the current
layer structure is ~20-30 ms. A step of 25-40 ms (25-40 tok/s) is therefore
the realistic target of the items below; ~50+ tok/s needs fewer collectives
per layer (items D5/D6) or multi-token steps (item D7).

## 4. Ranked low-hanging fruit

Estimates are derived from the measurements in section 3 and the census;
they are expectations to be tested by one TPU acquisition each, not results.

### Decode

| # | Change | Reference analogue | Evidence for the size of the win | Status |
|---|---|---|---|---|
| D1 | Route-grouped MoE: one Pallas grid over the OWNED route slots (traced grid size, scalar-prefetched expert ids), 256/512 tiles, ONE stacked feature-4 `psum` for all routed+shared gate/up partials, ONE expert-8 `psum`. 27 launches/10 collectives/8 conds per layer -> 4/2/0. | glm53 `moe_matvec` | **Measured** per layer (section 7): frozen 0.444 / 0.652 / 1.898 ms (normal / two-per-owner / concentrated) -> grouped-256 0.356 / 0.500 / 1.366 ms, i.e. **-20% / -23% / -28%**; about **-6.6 ms per 75-layer step** at normal routing, more when routes concentrate | **Implemented and measured**: `glm_tpu/perf/fp8_routed_experts.py`; bitwise equal to the frozen body on the 32-device CPU mesh and on the pod (all eight ranks, synthetic weights) |
| D2 | Candidate-set nucleus sampler: per-shard top-k, all-gather candidates, frozen rule on the ordered candidates, on-device sufficiency check with fallback to the frozen full sort. | glm53 `device_sample` | **Measured** head cost per call: greedy 0.58 ms, frozen nucleus 0.98 ms, candidates (k=256) 0.69 ms -> **-0.3 ms per token**. Real but small; the 17 ms DB621/DB610 gap was misattributed (section 3) | **Implemented and measured**: `glm_tpu/perf/ws32_sampling_candidates.py`; 0 token differences in 1,200+ randomized draws incl. forced fallbacks |
| D3 | One-row FP8 projections are **decode-bound on v4**, not launch-bound: the same `[1,1536] x [2048,1536]` projection costs 10 us with a resident BF16 table and 43-60 us with every FP8-decoding variant tried (frozen Pallas 128-tiles 60, Pallas 512-tiles 49, XLA fused decode+dot 48, integer bit-trick decode 43). TPU v4 has no FP8 datapath; decoding 3 MB of e4m3 on the vector units costs ~35-40 us. | glm53 keeps its non-expert matrices int8/bf16, not FP8 | **Measured** (section 6.3). The frozen step decodes ~2.7 GB of FP8 per chip per token -> roughly **40 ms of the 121 ms step is FP8 decode** | Root cause established; the fix is D8 below |
| D8 | **Pre-decoded BF16 residency for every non-routed weight** (**implemented and measured: 120.3 -> 72.1 ms with D1**) (attention q_a/kv_a/q_b/kv_b/o, DSA wq_b/wk, shared experts, dense layers). `bf16(f32(bits) * scale)` is exactly what the frozen kernel computes per element, so a resident BF16 copy gives bitwise-identical products: an EXACT transformation with no new numerics. Cost: +1.05 GB/chip (attention+DSA) + 0.7 GB (shared) + 0.02 GB (dense) = **+1.8 GB/chip**, inside the recorded headroom (6.6 GB at capacity 8,192, 4.5 GB at 166,912, 3.1 GB at 256K, the last one tight). Routed experts (22 GB/chip) stay FP8. | glm53 int8 non-expert weights, bf16 activations | Expected from section 6.3: ~13 ms (attention) + ~9 ms (shared) + <1 ms (DSA/dense) = **~22 ms of the 121 ms step (18%)**; routed experts keep ~9 ms of decode unless a packed decode kernel is written | **Implemented**: `glm_tpu/perf/bf16_resident.py` (exact per-table decode, mirrors of the attention/DSA/dense/shared bodies); CPU test: same tokens and DSA selections, KV within one BF16 ulp on <1% of elements; pod: 72.1 ms/step |
| D9 | **Routed-expert decode is at its software floor** (6.3): decode-only 43 us per 3 MB, packed integer decode 2.5x slower, bit-trick 10% faster. The busiest owner's routes (2-3 x 9 MB) set every sparse layer's critical path (~18 ms of straggler wait + 12 ms of kernels per step). What remains: (a) multi-row steps, i.e. MTP speculative decoding (D7): one decode serves every row in the step; (b) INT8 routed experts on v4's native int8 MXU (no decode; a re-quantization with activation scaling, NOT exact, needs its own quality validation); (c) nothing exact and cheap. | glm53's 3-bit planar experts are decoded in-kernel on v5e, which has a different VPU/MXU balance | measured ceiling: even a free decode would save at most ~25 ms; a 2x cheaper one is not available in software | Measured and closed as a software item; D7/INT8 are the follow-ups |
| D10 | DSA cache gathers and selection: `gather_custom_fusion` 355 calls / 5.7 ms per step (the full index-cache `jnp.take` on 21 layers and the aligned selected-KV gather on 78) plus 5.4 ms of `top_k`/`sort`; score directly against the cache pages (no gather), two-stage top-k with cut check (D6). | glm53 scores pooled keys in place, two-stage select | **~11 ms per step measured** in the trace; realistic **-5 to -7 ms** | Documented |
| D4 | Host loop per token: one fused device-to-host read of `(token, health, position, length)`; precompute all `max_new_tokens` SHA-256 uniforms once and index them on device (removes the per-token `device_put`); one fleet vote per token instead of three (fold the elapsed-time vote into the acceptance vote; keep the delivery-failure vote lazy). | glm53 `DeviceSampler`, resident token ids | Each `process_allgather` is a device collective + host sync over 8 hosts; 3 votes + 5 transfers are plausibly **5-15 ms** of the 148 ms sampled step (DB621: 147.975 ms p50 wall vs the device step) | Documented; touches the frozen request session (`SESSION_SHA`), so it needs its own reviewed change |
| D5 | Selected-KV exchange as LSE merge instead of a 2.6 MB zero-padded `psum`: all-gather the 8-head absorbed queries over expert-8 (64 KB), attend locally to the owned selected rows for all 64 heads, `psum_scatter` outputs (128 KB) and merge with `combine_stage_local_attention`. Both primitives exist: `pallas/sparse_attention.py::stage_local_sparse_mla_pallas` returns `(output, lse)`, `reference/attention.py::combine_stage_local_attention` merges them. | glm53 `_attend` with `seq_shard` | 78 x 2.6 MB all-reduce per step; ~100 us each -> **-5 to -8 ms** decode; dominant for prefill (P1) | Documented |
| D6 | DSA selection: two-stage exact top-k with cut check (local top-256, gather, global top-2048, fallback), or the repo's default-off bitonic `pallas/topk.py` kernels, in place of argsort + top-2048 + gathered argsort + top-k per full-indexer layer. | glm53 `indexer_select` | 21 layers x (top-k over 20,864 + merge over 16,384) sorts; legacy category "sort/top-k 34.87 ms/token" (`docs/glm-tpu-revolution.md` section 2); expected **-10 to -25 ms** | Documented; exactness needs the fallback path |
| D7 | Speculative decoding with GLM-5.2's shipped MTP head (`num_nextn_predict_layers = 1`), lossless verification. | qwen38-27b (+34%, 12/12 greedy prompts identical) | Multiplies whatever step time remains by the acceptance rate; the only route past the collective floor | Out of the release scope; needs cache-rollback semantics like the vllm-tpu patch |

### Prefill

| # | Change | Reference analogue | Evidence for the size of the win | Status |
|---|---|---|---|---|
| P1 | LSE-merge attention (D5) in the prefill tiles: the `[32, 2048, 640]` BF16 selected-KV `psum` is 84 MB per 32-row tile per layer, 312 per 128-row block = 26 GB of all-reduce traffic per block. | glm53 `_attend` | A ring all-reduce of 84 MB over 8 chips is ~1.5-3 ms -> **0.5-0.9 s of the 2.0 s per-block time at 2K (25-45%)**; replaced by ~6 MB of query gather + output scatter | Documented; same primitives as D5 |
| P2 | DSA candidate chain: score all local keys once per tile and take ONE `top_k` (scores `[32, 16384]` f32 are 2 MB, no memory reason for 512-key tiles), then the two-stage merge with cut check. At 128K the chain is 33 sorts per tile-layer x 21 layers x 4 tiles x 995 blocks. | glm53 `indexer_select` | The repo's own planning integral attributes **1,772 s of the 2,802 s 128K prefill** (6,895 of 8,152 s at 256K) to DSA selection/merge (`PREFILL_PERFORMANCE_TARGETS.md`) | Documented; the biggest lever at >=128K |
| P3 | Wider attention tiles (64-128 rows) once P1 removes the 84 MB exchange; fewer rolled `scan` iterations and collectives per layer (4 -> 1-2). | glm53 `q_block` 128 | 4 x ~10 collectives x 78 layers = ~3,100 collectives per block | Follows P1 |
| P4 | Larger MXU row tiles in the grouped FP8 GEMM (8-row tiles / M32 panels -> 128-256 rows) and 512-wide K/N tiles. | glm53 `moe_apply_grouped` chunk 256, Megablox `tm` | Weight re-streaming per active tile and grid overhead; secondary at 2K, matters as blocks widen | Documented |
| P5 | Fewer host boundaries: 1,024-token pieces, one fused device-to-host frontier read, one vote per piece. | glm53 prefill pieces | 2 votes + 6 reads per 128-row block; ~20 ms/block of 2,000 ms today, but dominant once P1/P2 land | Documented |

What is **not** low-hanging: exact FP32 DSA scoring in decode
(`precision="highest"`, 21 layers, tiny at one row), the FP8 weight format
(4x the bytes of the 3-bit reference experts), and the eight-host consensus
protocol itself. The exactness contract also means every item above needs a
compile acquisition, HLO/memory admission and a real-layer timing before
promotion, exactly as `WS32_2D_PROTOTYPE.md` did for the Pallas MoE body.

## 5. What this branch adds

Everything is opt-in and outside the frozen source pin; the supported path is
untouched and `python tools/check_release.py` still passes.

| Path | Role |
|---|---|
| `glm_tpu/perf/fp8_routed_experts.py` | `fp8_routed_projection` (route-grouped, scalar-prefetch, multi-block tiles; unowned slots are exact zeros with a pinned DMA block) and `ws32_moe_grouped_routes_mapped` (drop-in for `ws32_moe_pallas_from_routes_mapped`) |
| `glm_tpu/perf/ws32_sampling_candidates.py` | `ws32_nucleus_sample_candidates_mapped` and its split head; exact fallback via replicated `lax.cond` |
| `glm_tpu/perf/ws32_decoder_challenger.py` | `build_ws32_challenger_decoder_program`: the frozen sampled-decoder argument order with the two swaps (`Ws32PerfOptions`) |
| `tests/perf/` | Bitwise equality against the frozen kernel, frozen MoE body and frozen 8-layer decode step on 32 CPU devices; sampler equality over randomized draws; fail-closed paths |
| `tools/perf_op_census.py` | Structural census (jaxpr and compiled CPU HLO) of frozen vs challenger; writes the receipt above |
| `tools/perf_tpu_microbench.py` | Eight-process TPU v4 microbenchmark on the real mesh with synthetic FP8 weights at GLM shapes: MoE layer body (frozen vs grouped, three route patterns), sampler heads, one-row projection variants, full 78-layer greedy step with a profiler trace; `--summarize` merges the eight rank receipts |

Run the checks (about 10 minutes on this host):

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/perf
JAX_PLATFORMS=cpu python tools/perf_op_census.py --output /tmp/census.json
```

Run the pod microbenchmark (eight processes; sync the worktree to
`~/glm-tpu-perf-ref` on workers 1-7 first, then on every host):

```bash
JAX_PLATFORMS=tpu PYTHONPATH=. python tools/perf_tpu_microbench.py \
  --coordinator 192.168.0.37:8476 --output ~/glm-run/<tag> --which moe,sampler,tiles
python tools/perf_tpu_microbench.py --summarize ~/glm-run/<tag> --summary-output docs/perf/<receipt>.json
```

Numerical boundary of the implemented items, stated plainly:

* D1 is arithmetic-identical per element to the frozen body (same block
  decode, same 8-row MXU tile, same FP32 K-order accumulation, same BF16
  roundings and route-sum shape). The stacked feature `psum` reduces the same
  operands in one collective; on CPU the results are bitwise equal, on TPU
  the all-reduce reduction order of a stacked payload is expected but not
  proven identical, which is what the existing real-layer max-abs-error 0.03125
  contract would decide.
* D2 changes only the summation order of the softmax normalizer (psum of
  per-shard sums versus a descending serial sum), which can flip the nucleus
  boundary only when `previous_mass` lands within one FP32 ulp of `top_p`.
  All other steps (ordering, tie rule, inclusive crossing token, draw
  scaling) are the frozen rule on the candidate prefix; the fallback keeps
  the result defined whenever the nucleus is not provably inside the
  candidates.

Integration path on hardware (one acquisition, following the repo's usual
gates): compile `build_ws32_challenger_decoder_program(mesh, config,
options=Ws32PerfOptions(), sampling=NucleusConfig())` next to the frozen
sampled decoder, run the HLO/memory admission on its optimized HLO (all
collectives are still feature-4/expert-8 groups; CPU HLO confirms), time the
real sparse layer with `ws32_moe_grouped_routes_mapped` against the recorded
1.257/2.304 ms, then a short 2K numerical run against DB610's tokens.

## 6. TPU v4 measurements (2026-09-19, synthetic weights, all 32 chips)

Method: `tools/perf_tpu_microbench.py`, eight `jax.distributed` processes on
the physical `expert=(x,y) x feature=z` mesh, random FP8 payloads and scales
at GLM-5.2's exact shapes (timing is value-independent), 60 timed calls after
warm-up, per-call wall with `block_until_ready`. "Slope" runs the body
`n=1` and `n=17` times inside one program and reports the per-body
difference, which removes the ~0.35 ms host-dispatch floor of a one-body
program; the `n=1` figure is what a one-dispatch-per-token request loop pays.
Fleet spread across the eight ranks was below 1% on every metric. Receipts:
[phase 1](tpu-microbench-phase1-20260919T102046Z.json) (frozen kernel with
all-slot grid), [phase 2](tpu-microbench-phase2-20260919T103021Z.json)
(owned-slot dynamic grid, the committed kernel).

### 6.1 Sparse-layer MoE body (one row, 32 local experts, GLM shapes)

| Body | normal (1 route per owner) | two per owner | concentrated (8 on one owner) |
|---|---:|---:|---:|
| frozen `ws32_moe_pallas_from_routes_mapped` | 0.444 ms | 0.652 ms | 1.898 ms |
| grouped, 128x128 tiles (all-slot grid, phase 1) | 0.819 ms | 0.946 ms | 1.706 ms |
| grouped, 128x128 tiles (owned-slot grid) | 0.455 ms | 0.649 ms | 1.814 ms |
| grouped, 256x256 tiles (owned-slot grid) | **0.356 ms** | **0.500 ms** | **1.366 ms** |
| grouped, 512x512 tiles (owned-slot grid) | 0.364 ms | 0.510 ms | 1.391 ms |

Reading: iterating unowned slots at 128x128 tiles costs more than the frozen
conditionals save (phase 1); the committed kernel therefore sizes its grid by
the owned-slot count. The win is 20-28% of the layer body, i.e. about 6.6 ms
of a 130 ms step at normal routing. Outputs equal the frozen body bit for bit
on every rank (local-shard comparison in the receipts).

### 6.2 Output head (hidden 6,144, vocabulary 154,880)

| Head | per call |
|---|---:|
| frozen greedy | 0.58 ms |
| frozen nucleus (full stable two-key sort + cumsums) | 0.98 ms |
| candidate nucleus, k=64 / 256 / 1024 per shard | 0.71 / 0.69 / 0.73 ms |

### 6.3 One-row FP8 projection `[1,1536] x [2048,1536]` (slope per call)

| Variant | per call |
|---|---:|
| frozen `fp8_block_matmul_f32`, 128x128 grid (192 steps) | 59.6 us |
| `fp8_routed_projection`, one slot, 128x128 | 64-75 us |
| same, 256x256 | 50-54 us |
| same, 512x512 / 1024x512 | 47-49 us |
| same, 2048x512 | 90 us |
| XLA fusion: `bitcast->f32 * scale -> bf16` of the selected table, then `dot` | 48 us |
| same, with `dynamic_index_in_dim` selecting one of 32 experts | 48 us |
| XLA fusion with an integer bit-trick e4m3->bf16 decode (exact for all 256 codes) | 43 us |
| **resident BF16 table, plain `dot` (no decode)** | **10 us** |
| Pallas 512x512, frozen decode + dot (bitwise equal to the frozen kernel) | 46 us |
| Pallas 512x512, **decode only** (no MXU work) | **43 us** |
| Pallas 512x512, packed 4-bytes-per-lane integer decode + dot | 115 us |
| Pallas 512x512, resident BF16 tile + dot | 9 us |

Reading ([phase 7 receipt](tpu-microbench-phase7-decode-20260919T120021Z.json)):
tile size removes only the per-grid-step part; every FP8 variant, Pallas or
XLA, lands at 43-60 us because TPU v4 decodes e4m3 on the vector units. The
no-decode reference at 10 us is the HBM-bound cost of the projection. A one-row
FP8 projection therefore pays ~35-40 us of decode per 3 MB, and the frozen
step decodes ~2.7 GB per chip per token: about 40 ms of the 121 ms step.
Decode alone is 43 of the 46 us; the MXU work is ~3 us. A packed 32-bit-lane
integer decode is 2.5x SLOWER (v4's vector unit does not favour integer bit
manipulation) and the XLA bit-trick decode only 10% faster, so software FP8
decode on v4 is near its floor at ~14 ns per 1,024 elements. Pre-decoding to
BF16 (D8) removes the cost wherever the extra bytes fit; for the 22 GB of
routed experts that cannot fit, the remaining options are multi-row steps
(speculative/MTP decoding amortises one decode over several tokens) or an
INT8 expert format on v4's native int8 MXU path (not exact).

### 6.4 Complete 78-layer greedy decode step (capacity 8,192, synthetic weights)

| Program ([phase 5](tpu-microbench-phase5-step-20260919T112532Z.json), [phase 6](tpu-microbench-phase6-step-20260919T114126Z.json) receipts) | p50 | p99 | tok/s at p50 |
|---|---:|---:|---:|
| frozen greedy decoder (`build_ws32_decoder_program`) | 120.3-120.5 ms | 125.1 ms | 8.30 |
| challenger: grouped MoE (owned-slot grid, 512x512 tiles), everything else frozen | 106.9 ms | 110.9 ms | 9.36 |
| challenger: grouped MoE, 256x256 tiles | 105.7 ms | 126.3 ms | 9.46 |
| **challenger: grouped MoE + BF16-resident non-routed weights (D8)** | **72.1 ms** | 76.0 ms | **13.87** |

Per-step device-time categories from the per-host profiler traces
([trace categories receipt](tpu-trace-categories-20260919T114126Z.json),
`scripts/analysis/parse_xplane.py`, rank 0's four chips, two traced steps):

| Category (ms per step) | frozen | grouped MoE | grouped + BF16-resident |
|---|---:|---:|---:|
| device step | 118.6 | 103.9 | **70.3** |
| Pallas custom calls (FP8 decode + matmul) | 47.0 | 43.5 | 12.0 (routed experts only) |
| collectives (`psum` 546 calls, all-reduce, all-gather) | 45.4 | 35.9 | 35.4 |
| gather / scatter / dynamic-slice (cache reads) | 7.7 | 6.7 | 6.4 |
| sort / top-k (DSA selection, 21 layers) | 5.3 | 5.4 | 5.4 |
| fusions / dots | 4.1 | 4.0 | 6.6 (the BF16 dots) |
| data movement | 4.0 | 3.9 | 1.8 |

Reading: D8 removes 31 ms of FP8 decode from the step and the grouped MoE
removes ~10 ms of collectives/conditionals. After both, **collectives are 52%
of the device step**. Split by call site
([collectives receipt](tpu-trace-collectives-20260919T114126Z.json)):

| Collective, per step | frozen | grouped + BF16 |
|---|---:|---:|
| routed-expert `psum` over expert-8, `bf16[1,1536]`, 75 calls | 28.0 ms (374 us each) | 17.9 ms (238 us each) |
| selected-KV `psum` over expert-8, `bf16[1,4,512,640]` (2.6 MB), 78 calls | 7.0 ms | 6.6 ms |
| o-projection `psum` over expert-8, `bf16[1,1536]`, 78 calls | 2.1 ms (27 us each) | 2.1 ms |
| router logits all-reduce, 75 calls | 1.9 ms | 2.0 ms |
| RMSNorm square-sum `psum` `f32[1,1]`, 157 calls | 2.4 ms | 1.8 ms |
| routed gate/up feature `psum` (8 conditional calls -> 1 stacked `[9,2,2048]`) | 2.0 ms | 1.0 ms |
| everything else (q_a/kv_a feature psums, DSA gathers) | 2.4 ms | 2.9 ms |

A 3 KB expert-8 all-reduce costs ~27 us here (o-projection), so the 238 us
average of the routed-expert `psum` is mostly **waiting for the busiest expert
owner**: with one row and eight routes over eight owners, one owner usually
holds two or three routes and decodes 9 MB of FP8 per route (~120 us each)
while the others idle in the collective. The routed-expert path is therefore
still ~30 ms of the 70 ms step (12 ms of kernels on the average chip plus the
straggler wait), and cheaper FP8 decode for the routed experts is the next
lever (D9). Tokens differ from the frozen program only through BF16
accumulation-order effects on synthetic random weights (first-token trails in
the receipt); the CPU test on the fixture shows identical tokens and DSA
selections.

The frozen figure agrees with DB610's 130.6 ms once that run's per-token host
loop (three eight-host votes, four device-to-host reads, one device_put) is
taken into account, so synthetic weights reproduce the real step cost. All
eight ranks agree within 0.2 ms. The grouped MoE alone removes **13.6 ms
(-11.3%)** from the step, twice the isolated-layer estimate (75 x 0.08 ms):
inside the fused step the removed conditionals and collectives also stop
serialising neighbouring work.

## 6.5 D5/D10 and prefill follow-up (2026-09-19)

A fixture audit found that the original synthetic generator seeded partially
replicated tables differently on unsharded axes. The corrected
`partition_axes_only_v2` generator has a CPU32 replication proof. Older receipts
remain archived exploratory timings; use the
[corrected eight-rank receipt](tpu-microbench-replica-correct-20260919T143305Z.json)
and [follow-up report](D5_D10_PROGRESS_20260919.md) for the current evidence.

With correct replicas, D1+D8+D5+D10 runs in **64.30–64.43 ms/token
(15.52–15.55 tok/s)** on synthetic weights, versus 72.55–72.90 ms for D1+D8
and 121.20–121.54 ms frozen in the same acquisition. D5's gain is much smaller
than the original estimate; D10 supplies most of the improvement. The 30-sample
p99 is 84.9–85.6 ms, versus 79.7–80.2 ms for D1+D8. D10 isolated TPU
scores/selections match bitwise at 8K/128K; D5 remains a numerical boundary
and trained-weight validation is still required.

At M32/full-128K, P1 attention measures **3.92 -> 1.64 ms**. P2 DSA measures
**23.24 -> 2.04 ms (about 11.4x)** against DB610's admitted settings
(`paired_position_sort=True`, `sorted_local_merge=True`, key tile 512), with
bitwise-equal results. The default tiled configuration is much slower and is
not the production comparator. These are primitive timings, not full-model
prompt throughput. P1/P2 are now integrated in an opt-in prefill builder with
CPU state/health proofs; D8 prefill residency and complete-model TPU admission
remain open. All eight hosts passed the final authenticated idle check.

## 7. Reference material

* Clone used for the analysis: `/home/gianl/reference-repos/kaggle-tpu-lab`
  (sparse checkout of `glm53-flash/` and `qwen38-27b/`, MIT, commit
  `1aa1f08`, 2026-09-15). Not vendored into this repository.
* Frozen bodies compared: `glm_tpu/greenfield/kernels/ws32.py`,
  `ws32_layer.py`, `ws32_sampling.py`, `ws32_prefill_*.py`, `prefill_dsa.py`,
  `pallas/fp8_matmul.py`, `runtime/ws32_decoder.py`,
  `runtime/ws32_request_session.py`, `scripts/greenfield/ws32_native_benchmark_runtime.py`.
* Owner-recorded targets: 10,000 prompt tokens/s remains the requested final
  objective and 500 tok/s the non-final milestone
  (`docs/greenfield/PREFILL_PERFORMANCE_TARGETS.md`); nothing here changes
  that registration or claims progress against it.
