# Grouped MoE prefill — bounded implementation design

2026-09-07. Main-agent implementation plus independent Astra scheduling/code review.
CPU-admitted, unwired; F32 grouped projection TPU-admitted DB580, not full MoE/production.
§24 and existing contracts govern.

## Decision and sources

DB576–579 show the existing standalone raw-FP8 projection can consume8–256 rows at the
real local K1536 N2048 shape. That is not the routing distribution: most experts receive
far fewer rows. Do not compute a full B-row matrix for every one of256 experts.

Reuse the installed JAX0.10.1 Megablox metadata/scheduling mechanism, not stock `gmm`:
`jax/experimental/pallas/ops/tpu/megablox/gmm.py` defines `make_group_metadata` (line79),
group-dependent BlockSpecs and dynamic active-row-tile grid (lines384,476–548).
Installed file SHA256 `9fe9f9807d73d9c63c4f2749e6f6eeb61645fee2793c9dbdd3f15f2e992cd209`.
`common.py` SHA256 `fa1927e6a951c4daa2f67a228f5488d05025de5a1dbc7c0345802de436bfcc3c`;
stock gmm accepts BF16/FP32 weights, not checkpoint U8 payloads. This is an implementation
API whose version/source must stay bound, not an assumed stable public contract.

Retain `kernels/pallas/fp8_matmul.py` tile arithmetic: U8→E4M3FN bitcast, FP32 scale product
rounded BF16, contraction-order FP32 accumulation, explicit F32 gate/up and BF16 down outputs.
Weights remain `[32,N,K]` per owner; include expert identity in block/scale indexing.
No whole-table BF16 expansion, full-pod hidden gather, legacy execution import or repack.

## Schedule

1. Flatten B×8 route records, retaining `token*8+slot`. Sort by expert then original flat ID;
   build256 int32 counts and inverse permutation. Invalid IDs/duplicates must fail admission,
   never clip silently. Respect the existing router's top8/tie/weight semantics.
2. Gather sorted local hidden features into `[8B,1536]`. Hidden is already replicated over
   expert8; no new all-owner activation dispatch is necessary.
3. Each owner schedules its32 contiguous experts with global counts, `start_group=owner*32`,
   `num_nonzero_groups=32`, `visit_empty_groups=False`. Use dynamic active row tiles rather
   than full-B compute for every expert. Initial row tile8/16, then measure32.
4. Grouped raw-FP8 gate/up produce F32 partials. Keep the existing stacked feature4 reduction
   and BF16 activation boundaries. All feature peers must share routing metadata/order.
5. Grouped down outputs BF16. Apply original route weight rounded BF16 and round the product.
   Restore `[B,8,1536]` in original route-slot order with unowned routes explicitly zero.
6. Use the existing `jnp.sum(..., dtype=bf16)` route expression/association, F32 expert8 sum,
   BF16 rounding and routed scale/shared-expert combination. A sequential replacement sum
   is not automatically the same arithmetic. Compare actual outputs under the contract.

## Memory and correctness traps

- Group boundaries can share an aligned row tile. Megablox revisits it with masked stores;
  group-row and contraction grid dimensions are `arbitrary`, not parallel.
- Start output from defined zeros or prove every read/write; unowned rows and masked values
  cannot contain uninitialized data when entering a collective.
- Zero-route owner: explicit zero-output branch; do not execute an unsupported zero grid.
- One expert receives at most B unique-token routes, but an owner may receive all8B. Never
  allocate per-owner capacity from average routing or drop overflow.
- At B512, one F32 `[4096,2048]` buffer is32MiB; gate/up64MiB before stack/reduce temporaries,
  plus gathered rows/down/permutation/activation buffers. Account for overlapping live ranges
  and long caches, compiler overlays and actual measured peaks. VMEM must remain tile-sized.
- Keep collectives outside grouped Pallas kernels. Different expert owners may execute
  different active-tile counts, while each feature group must reach identical collectives.

## Smallest decisive sequence

CPU metadata round-trip/count/coverage tests first: zero owner, all routes on one owner,
one expert receiving B rows, empty groups between live groups, shared boundary row tiles and
nontrivial slot permutations. Validate every record survives exactly once.

Then small interpreted grouped raw-FP8 projection, per-route gate/up/down comparisons and
forced32 CPU collective tests. Real-shape one-layer TPU comes before a short full decoder.
Both CPU and real TPU must test skew, tails, zero scales and exact restoration; no whole-model
performance claim from metadata tests or a single projection. In parallel, bounded existing
DSA/attention/collective baseline measurements complete the §24 target-registration budget.

## CPU implementation admission

Implemented `kernels/prefill_routes.py`, `kernels/pallas/prefill_grouped_fp8.py`, and
`kernels/ws32_prefill_moe.py`. No decoder imports them. Routing sort/inverse/counts retain
every route; grouped calls use dynamic active tile count, initialized output alias, masked
boundary revisits and group-indexed U8 weight/FP32 scale tiles. Invalid counts/offsets skip
the grouped call and return zeros/false health. Invalid/duplicate routes and bad route weights
propagate false health from MoE. Every future serving caller must collect all-chip health;
these APIs do not claim standalone exception-based runtime enforcement.

Initial routing/primitive suite17 tests passed8.01s. Review found int32 overflow in the offset
upper-bound addition; changed to `offset <= global_groups-local_groups` with INT_MAX tests.
Expanded primitive suite6 tests passed6.46s, including two output blocks and K1152 crossing
the ninth scale block. Forced32 CPU MoE test passed6.37s:17 rows,64 experts/top8,
hidden128/intermediate32/block32, distributed routes and all136 routes on one owner. Outputs
are bit-identical to the old one-row Pallas MoE path, including zero hidden/scales/route weight.
Small geometry is CPU semantic admission only; production128×128 blocks and real weights
still require TPU proof.

CPU HLO has exactly one feature4 tuple all-reduce and one expert8 all-reduce: XLA combined
the two independent feature sums. Initial test expected three source-level calls and failed
only at that count, after numerical identity passed; it now verifies the actual optimized
count and exact membership. No collective exemption or production linter changed. Independent
Astra review PASS for CPU-stage persistence after the offset fix; not TPU execution approval.

## First TPU arithmetic admission — DB580

`greenfield_fp8_ws32_grouped_admission_20260907T194202909841356Z`, pin
`5d6e99d8225fe82e5dab59293175e5d29f34a82b`: synthetic local G32/N2048/K1536,
136 sorted route rows, F32 output, row tile8. Distributed136 singleton experts,
all136 routes on one owner (eight experts ×17 rows) and empty-owner cases all have
zero bit mismatches against the existing per-row TPU FP8 projection. No tolerance widening.
One grouped Pallas call, raw U8 table, no full decoded table or collectives. Auxiliary
calls are one compact integer gather annotation and two u32[256] allocations.
Compile1.448s; runner8s, entire guarded workflow34s. Compiler arguments101180928B,
outputs1115136B, temp1097216B; process peak107012096B including reference, not VMEM peak.
Both pre/post censuses8/8, independent root accelerator-holder checks, wrapper exit0,
generation-verified archive and DB580 with NULL latency. No timed samples/performance claim.
Exact record hashes and archive terminal in HANDOFF. Next: BF16 down and mapped real-MoE
TPU admission, causal DSA/attention/IndexShare layers and phase budgets, then short decoder.
This does not prove end-to-end speedup or finish the prefill pivot.
