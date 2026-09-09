# Rolled prefill-window integration

2026-09-09; starting pin4bc20c0d. CPU-stage implementation, default-off. No
new TPU, whole-model numerical, memory, performance or TTFT admission.

## Decision and evidence

DB597 provides the current full-model2K control:65.668s/30.974prompttok/s,
129.798msdecode. DB599 selects an exact local sorted-pair DSA merge; DB600
selects expert-relative M32/N256 panels, suffix10.506→6.306ms, but only7.685%
less wide partial phase-sum wall. Neither component result measures their
combined complete-model effect. Preserve both original receipts and failures.

The existing `ws32_prefill_window.py` uses a Python loop to trace up to four
copies of its B32 attention/DSA prefix. Blindly expanding it to larger windows
would multiply code as well as row storage. The separately compiled completed
prefix/suffix from DB600 is a useful numerical reference, not a reason to put
thousands of synchronous host dispatches into production prefill.

Main-agent and independent Astra decision: retain the current heterogeneous
78-layer order, roll only each layer's bounded prefix loop, then execute one
wide panel MLP. No new checkpoint or second decoder implementation.

## Implementation

`ws32_prefill_layer_window_mapped(..., rolled_prefix=True)` executes
`lax.scan(unroll=1)` over causal tiles. B128 uses four B32 iterations. Static
tails above32 pad to a multiple of32 and trim row results; smaller static
tails retain their narrow row shape. Offsets are bounded before addition,
counts preserve the original span-validity refusal, and empty trailing tiles
cannot write cache or advance the committed frontier.

Only three arrays form the scan carry: this layer's proposed KV, unrepaired
index and repaired index caches. Seven row outputs are stacked: normalized
MLP input, carried residual, selected positions/counts/scores, health and
normalized input. No stack of historical cache generations is returned.
The single wide MLP consumes flattened BF16 normalized rows, and the ordinary
layer result is assembled from its own outputs and final proposed caches;
there is no dependency on an alternative reference suffix.

The existing runtime threads `rolled_prefix`, `expert_panels` and
`sorted_local_merge`, all static booleans and defaultFalse. These new runtime
options require explicit `mlp_window=True`. Existing paired position sorting,
M64 repair and raw checkpoint views are reused. Production decode and the
protected launcher/admission remain unchanged.

IndexShare selections are row-specific; each layer always uses its own KV.
Prompt attention reads only unrepaired keys. The existing all-owner atomic
block commit still owns all frontiers and repair promotion. A late-row or
single-owner failure rolls back the whole proposed block, not just its tile.

## Numerical and memory limits

A BF16-typed scan output is NOT proof of the separately completed executable
boundary used by DB600. Neither nested JIT nor a barrier would establish that.
This is a new compiled realization requiring its own original-fixture and
short-model evidence. Retained optimized while loops and actual buffer/dtype
boundaries must be inspected; do not require byte identity of incidental
compiler helpers or infer numerical equality from dtype alone.

Stacked row outputs intentionally scale with window size; proposed caches and
required original rollback state remain live. A while loop does not prove
aliasing or a specific HBM reduction. Actual compiler allocations, all-live
buffer accounting and32-chip peak memory remain required before promotion.
No full BF16 expert-weight expansion or donated rollback source is introduced.

## Focused admission and next action

CPU checks use actual eight-layer kernels, producer/share boundaries2→3/6→7,
two windows, tails, poison padding, repaired-history independence, final repair
promotion and late single-owner rollback. Static33 near capacity is compared
against explicit B32 followed by B32 with one live row and31 invalid padded rows,
matching the executing scan's scorer shape. Actual production78-layer B128/B33 abstract
interfaces are checked without allocating weights. These prove CPU composition,
not TPU arithmetic, utilization or request latency.

The first combined tests exposed an unsupported SMALL TEST FIXTURE: hidden512
implies local down-projectionN128, but panels requireN256. The optional panel
fixture uses hidden1024/intermediate256, preserving historical default fixtures;
kernel geometry checks were not weakened.

Both focused actual CPU32 tests PASS310.65s: static33 near capacity and the
eight-layer/two-B128-window composition with tails, causal/repair independence
and atomic rollback. The latter inspects exactly eight outer rolled loops, not
nested Pallas loops. Historical unrolled execution and three option-refusal
tests also passed; production78-layer rolled abstract shapes passed separately.
Reuse registry4tests PASS1.93s. No repeat of these baselines is needed.

Preserved diagnostic limitation: a literal staticM1 final reference call changed
the scorer's batch shape and differed in25/128 selected scores by at most
2.3841858e-7 on CPU. That failed test does not establish a candidate defect or
M1 bit identity. Correcting the reference to the actual B32 one-live-row shape
made the strict full-state comparison pass; no tolerance changed. The hardware
original-boundary and whole-model numerical gates remain necessary.

Next: one retained-fixture integration discriminator using existing selected
weights/DB600 originals and current protection tools. Bind new raw graphs and
actual rolled-loop/memory evidence; preserve existing bounded arithmetic,
own-score DSA/tie, state and cache rules. Then protected own2K request wall and
numerical proof; own8K follows success. No repeated component baseline, scalar
tap campaign, B32 full-model intermediate or automatic hours-long run.

Independent Astra final review: noP0-P2; the shape-matched tail test is valid
with the stated M1 limitation. Hardware comparison reuses the immutable DB600
control assembly after authenticating its original receipt. Use the unchanged
`prefill_window_protocol.compare_case`: exact ordered DSA positions/counts and
route IDs, existing per-row/aggregate output/residual/route-weight/written-cache
bounds, exact untouched cache bytes and health/finiteness/causal score ordering.
This comparator is NOT score-bit identity or independent full-score-row top-k.
Save all candidate fields from its own single-path result and independently
replay the comparison at collection. Any mismatch stops promotion; no adjusted
tolerance or new baseline. Passing permits own2K, not a full-model DSA claim.

Further larger windows and long-prefix DSA work remain necessary; B128 and
this design do not promise500 or10K prompttok/s. Final targets are unchanged.
