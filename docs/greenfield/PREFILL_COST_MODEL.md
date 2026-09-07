# Prefill cost model and baseline admission

2026-09-07. §24 planning evidence, NOT a latency prediction, performance result or registered
acceptance threshold. Independent Astra arithmetic proposal, reproduced by the main agent from
`configs/glm-5.2-fp8-config.json`. Code pin `6644dea8` for the existing source examined here.
No new full-model run is needed to calculate these budgets.

## Useful arithmetic, not physical instructions

One multiply-add counts as two operations. Geometry: hidden6144, layers78 (dense3/MoE75),
intermediate12288/2048, routed256/top8 plus shared1, attention64 heads, q-rank2048,
kv-rank512, q-nope192 + q-rope64 = q-head256, value-head256, index heads32×128,
full indexers21, selected top2048. Do not confuse config `head_dim=192` with q-head256.

Reproducible expressions (standard Python, no model or JAX needed):

```python
attention_projection = 2 * 78 * (
    6144 * 2048 + 6144 * (512 + 64) + 2048 * 64 * 256
    + 64 * 256 * 6144 + 64 * 192 * 512 + 64 * 512 * 256
)
mlp = 2 * (75 * 3 * 6144 * 2048 * 9 + 3 * 3 * 6144 * 12288)
index_projection = 2 * 21 * (2048 * 32 * 128 + 6144 * 128 + 6144 * 32)

def cost(tokens):
    m = min(tokens, 2048)
    selected_pairs = m * (m + 1) // 2 + max(0, tokens - 2048) * 2048
    linear = tokens * (attention_projection + mlp + index_projection)
    sparse = 2 * 78 * 64 * (512 + 64 + 512) * selected_pairs
    dsa = 2 * 21 * 32 * 128 * tokens * (tokens + 1) // 2
    return tuple(value / 1e15 for value in (linear, sparse, dsa, linear + sparse + dsa))
```

Attention projection terms are q-a, kv-a, q-b, output projection, absorbed query and value
reconstruction. Sparse attention counts QK over latent512+rope64 and PV over latent512 for
causal `min(position+1,2048)` positions. DSA counts causal valid key pairs only; this assumes
future invalid positions are skipped, unlike the source's current capacity-wide scoring.

| Prompt tokens | Linear PFLOP | Sparse PFLOP | DSA PFLOP | Total PFLOP |
|---|---:|---:|---:|---:|
| 127363 (actual L7) | 9.992445 | 2.810623 | 1.395305 | 14.198373 |
| 131072 (full128K planning) | 10.283440 | 2.893135 | 1.477755 | 14.654331 |
| 262144 (full256K E0) | 20.566880 | 5.809040 | 5.910997 | 32.286918 |

Excluded: router projection, activation/norm/softmax, head-weight aggregation, selection/sorting,
repair, prompt heads, exact-path aliases, padding, communication and memory operations. The
index term is a semantic approximation, not every physical exact-path projection. These are
NOT exhaustive FLOP counts or latency lower bounds. Useful arithmetic grows approximately2.20×
between full128K and256K, not a universal2× or4× rule for execution latency.

## MoE reuse determines the useful grouping window

One routed expert's three FP8 matrices contain `3*6144*2048 = 37,748,736` bytes. Across75
layers and256 experts this is724,775,731,200 bytes, excluding scales and other weights.
FP32 scales per128×128 matrix block add approximately0.0244% for these divisible shapes.

For B rows, assume each independently chooses8 distinct experts uniformly. This is an explicit
SCENARIO, not observed GLM routing. Expected active experts = `256*(1-(31/32)**B)`.
Ideal routed bytes/token = `75*3*6144*2048*active_experts/B`, assuming each active expert's
tiles load once per group. Repeated tile loads, padding and other weights increase traffic.

| B rows | Active experts | Mean rows/active expert | Ideal routed GB/token (fleet) |
|---|---:|---:|---:|
| 8 | 57.4 | 1.11 | 20.32 |
| 32 | 163.3 | 1.57 | 14.45 |
| 128 | 251.6 | 4.07 | 5.57 |
| 256 | 255.9 | 8.00 | 2.83 |
| 512 | 256.0 | 16.00 | 1.42 |

Therefore8–32 rows is a cheap correctness discriminator, not proof of sufficient MoE reuse.
Separate a larger routing/layer block from smaller attention row tiles when memory requires it.
Skew may increase reuse but overload an owner; scratch/capacity tests must cover it without
dropping routes. Execution sorted by expert must restore original token/route-slot order before
the BF16 per-token sum. Associating that sum in expert order is not automatically equivalent.

The existing raw-FP8 kernel's row_tile8 is a padding quantum, not an independent grid axis:
its whole padded row block occupies VMEM. Larger rows increase scratch, not just useful work.
This is consistent with [JAX's TPU matmul tiling/pipelining guidance](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html),
but the local kernel and its measured allocations decide feasibility.

## Decision: bounded baseline campaign before target registration

Astra proposed a planning band of200–400s for full128K prefill and450–900s for full256K.
Main decision: these are hypotheses to test against a phase budget, NOT chosen acceptance
thresholds. They imply approximately36–73 useful TFLOP/s fleet-wide after all non-arithmetic
time is included; no existing measurement establishes that rate. Do not select the easiest
end of a band because a candidate fails. An inability to support the band is an implementation
gap to report, not permission to silently weaken the target.

Before candidate performance trials, freeze specific targets from the following minimal
measurements. Existing baseline measurement is allowed to establish those targets; it is
not itself an optimized-candidate promotion or authorization for an hours-long reference run.

1. Existing raw-FP8 primitive at real local shapes, rows8/32/128/256 (then512 only if memory
   permits), warmed wall and peak scratch. Separate useful rows from padded rows. The new
   K130/N135 CPU tests do not answer real-shape TPU throughput or association.
2. DSA score/top-k and selected-KV attention across valid prefixes and row counts, separating
   useful causal work from allocated-capacity work. No dense rows×heads×full-context temporary.
3. Representative feature4/expert8 collective bytes/time for multirow payloads. Do not scale
   the44.075ms decode collective category by row count and call it a prefill measurement.
4. Routing occupancy from existing captures if available; otherwise retain uniform and skew
   scenarios explicitly. A repeated one-row route copied B times is not representative routing.
5. Input transfer, cache initialization and first-token delivery allowance, including worst
   capacity262656 and both unrepaired/repaired index caches. Bind compiled and measured memory.

Reuse the existing protected wrappers, current final-layout weights and evidence stack. The
old `microbench_fp8_matmul.py` accepts only a fixed shape per kernel, so its `--rows` argument
does NOT yet authorize a generic row sweep. Extend the smallest relevant harness under review;
do not execute unsupported flags or bypass its shape/HLO/provenance checks. Small synthetic
baseline probes may inform the cost model but cannot prove real-layer or full-model performance.

Warm TTFT threshold = prefill threshold plus a MEASURED fixed allowance for input transfer,
cache initialization and actual first-token delivery. Resident weights/executables, no prefix
cache hit; disclose tokenization/transport boundaries. Cold load/compile is separate. Sum phase
budgets conservatively with stated overhead/straggler allowances before launching a candidate.

## Current admission result

`tests/greenfield/kernels/test_prefill_multirow_fp8.py` reuses the unchanged primitive:
8/17/32 distinct rows, nonzero and zero output tails, K130/N135 boundaries, zero row/scales,
FP32 equality versus individual calls, BF16 rounding and row permutations. Together with
registry checks: **10 CPU tests passed in9.38s**. Independent review accepted the initial test;
its nonzero-tail coverage suggestion was implemented before persistence. This supplies CPU
semantic evidence only. No TPU baseline, new prefill execution or speedup has been measured.

## Bounded harness protocol — 2026-09-07

Opt-in `GLM_GREENFIELD_FP8_MATMUL_KERNEL=ws32_prefill_baseline` with
`GLM_GREENFIELD_FP8_BASELINE_ROWS=8` (then32/128/256 after admission) uses the existing
`run_fp8_matmul_microbench.sh`. Exact local shape K1536 N2048, raw-FP8 F32 partial, one chip
on the existing four-chip host; synthetic distinct input rows, not captured model activations.
No subgroup reduction or expert routing in this baseline.200 warmups/1000 timed samples,
600s runner deadline plus30s kill grace; stop and diagnose the first failed shape.
No512-row trial is admitted yet. Four serialized invocations are simpler than a new sweep
controller; no numerical full-model load or checkpoint artifact is needed.

Existing HLO/no-full-weight-overlay and tolerance-based reference checks remain. Record HLO,
compiler allocation estimate, device memory stats, useful/padded rows, JAX/libtpu versions,
compile wall and warmed host-dispatch-through-completion distribution. The intersample scalar
transfer is outside timing; this is not sustained throughput. Compiler/device HBM accounting
includes reference work and does not measure VMEM peak or bitwise batched/serial TPU identity.
DB identity names the baseline and exact M/K/N. Supplemental authenticated root checks cover
the libtpu lock, actual accelerator holders and starting greenfield runners; existing fleet
census/leases remain. Post-census precedes successful DB finalization; archive readback binds
every object's generation/size/CRC/SHA and terminal identities. This does not replace protected
real-layer/full-model traces or other §24 budget components.
