# Completed B32 prefixes → B128 MLP: new candidate contract

2026-09-08. Default-off design; not hardware admission or performance proof.
This is a NEW execution realization, not a repair or promotion of the failed
fused B128 window. Historical failures, DB588 and DB592 stay unchanged.

## Why this next, and what the evidence does not say

DB592 captured the actual layer6 B128/fourB32 boundary in 218 seconds of worker/
collection time, with two WK and five model calls, no timing samples and 8/8
normal/root cleanup. Instrumentation changed the candidate's original signature
on all32 owners. Control DSA/routes/output reproduced, but its residual did not.
Consequently these captures cannot establish the original failure's cause.

The offline all-owner replay establishes a useful boundary for a new design:

- Both routers select canonical top8 from their own captured biased scores.
  Their FP64 projections on their own completed BF16 inputs differ from captured
  logits by at most 9.57e-8/1.17e-7. The captured input differences are upstream:
  880 BF16 elements across four unique feature shards, in rows2/74/94/101.
- Both routes disagree with full-FP64 sigmoid/bias selection at the same three
  ordered rows (4/25/64), including one set difference at row4. Exact FP64 route
  equality is NOT a new acceptance rule. Projection error and finite nonlinear
  arithmetic must not be confused with a defective own-score selector.
- At row2 the 508 causal keys are identical, while query/head operands differ.
  Each selected score row is complete and correctly sorted. Rounding the captured
  query to BF16 before an FP64 scorer predicts observed scores to 2.47e-8/7.21e-8;
  this is a labelled diagnostic model, not proof of a TPU internal boundary.

See `../artifacts/prefill-window-captured-input-fp64-v2-20260908.json` and the
analysis script recorded there. It starts from captured inputs, NOT §21's full
token/checkpoint forward reference R. No original-cause or numerical promotion.

## Candidate

1. Compile ONE ordinary B32 prefix executable from the existing
   `ws32_prefill_transformer_layer_mapped(prefix_only=True)` path. It includes
   attention, DSA/IndexShare, the three cache proposals and post-attention norm;
   no routed/shared expert computation. No additional diagnostic taps.
2. Complete four calls to this same executable, with offsets and live counts
   from the existing window protocol. Each later call consumes the preceding
   proposal KV, unrepaired-index and repaired-index buffers. Keep the original
   committed state intact until the existing all-owner acceptance frontier.
3. Concatenate the four completed BF16 MLP inputs on device, preserving feature
   ownership, and call ONE existing B128 router/grouped-MLP suffix. It consumes
   actual prefix values, never a host reconstruction or supplied route override.
4. Carry original prefix residual/metadata/health into the assembled result;
   combine prefix, suffix and finite-output health. No precision, scale,
   tolerance, checkpoint format or speculative change.

The prefix's compiled context is independent of suffix row count. Completion is
an actual executable boundary, not a BF16 cast or optimization-barrier assumption.
This does NOT guarantee equality with the old fused B32 or B128 realization.
Those outputs remain separate scientific comparisons, never overwritten.

## Smallest control and its limits

Call the B32 suffix four times on the SAME completed prefix outputs, in original
row order. Compare its output, ordered routes, route weights and health against
the B128 suffix under existing bounds, including each row and padded tails.
Shared prefix cache/DSA equality is BY CONSTRUCTION; it is not an independent
attention/DSA/complete-layer validation. Explicitly report that limitation.

First CPU proof: real API composition, reversed32-owner placement, causal carry,
partial final tile, poisoned inactive rows, incoming/suffix failures and rollback
ownership. Reuse existing fixtures, subset loader, compiler journals, fleet
guards, original-array publication and DB accounting. No second protection stack.

Then acquire the new prefix/suffix graphs and actual simultaneous memory. The
old DB590/591 hashes cannot authorize them. One selected-layer protocol can run
two completed-WK calls, four prefixes, one wide suffix and four narrow suffixes;
do not add more observation variants. Keep this a suffix-on-completed-input
admission, not a disguised full-layer or whole-model PASS. Protected full-layer
obligations and own short-decoder/8K numerical proof remain before promotion.

## Efficiency decision, fixed before measurement

The discriminator must include prefix calls, device assembly, synchronization
and the suffix in profiler-free candidate wall, with all32-owner memory budgets.
Compare equivalent 128-row work. Report shared-prefix cost separately but never
subtract it out of candidate end-to-end wall. Prefix outputs must remain device
resident. No host activation/weight transfer or repeated weight loading.

Five candidate executable calls per layer/window can add dispatch overhead.
The useful saving is four narrow MLP calls replaced by one wide call; whether
that exceeds the completion/assembly cost must be measured. DB589's isolated
3.329x/1.280x does not predict this full-layer result. Do not adopt the new path
if total wall regresses or new cache/code/scratch headroom is unknown.

If this standalone realization is admitted, integration must preserve real
boundaries efficiently (a device-call mechanism or a measured bounded schedule),
not quietly re-inline everything into the failed fused graph. Layer-major
multirow prefill remains the production objective, not per-token decoder scans.
Own short decoder → competitive8K → registered TTFT scaling → efficient L7/L8
remain required. No further research report is a prerequisite.

## Review

Existing independent Astra reviewer and main analysis agree on this candidate
and its limitations after DB592's offline results. No P0–P2 in the offline
analysis. Reviewer withdrew an incorrect resident-code total after checking the
actual four compiled analyses (67,827,200 B). Design agreement is not approval
to launch: CPU composition, fixed new graphs/memory, current-diff review,
persistence and fresh protected preflights remain required.

## CPU implementation status — 16:21Z

`scripts/greenfield/prefill_completed_window.py` now builds actual prefix/suffix
programs and device-only assembly. The existing layer builder adds one default-off
completed10-output prefix mode; default full-layer/capture behavior remains.
No worker wiring or hardware authorization. CPU32 reversed physical placement
test executes four causal prefixes, one wide/four narrow suffixes, compares the
existing fused B32 realization on CPU, and covers33live rows, poisoned padding,
unchanged original caches, prior proposal carry and health failures.

Reviewer P2 fixed: scalar-int32 validation for offset/live counts before arithmetic;
float/bool/vector/Python-int refusals and negative/oversized count health failures.
Combined15tests (composition, offline arithmetic, reuse) PASS33.14s; production
6144-wide prefix/B32/B128 suffix abstract schema PASS2.98s without model payload.
The earlier30.24s composition pass overlaps; do not add it to the count.
Independent reviewer clears CPU commit/push/mirror, not TPU execution.

## Compile-only integration — 2026-09-08

Distinct kernel `ws32_prefill_completed_window_acquisition`, protocol
`ws32-prefill-layer6-completed-prefix-window128-compile-only-v1`. Existing campaign
now prepares five programs: wk_decode, wk_promote, prefix, candidate, control.
All prompt/cache/WK inputs are abstract; no executable dispatch. Actual selected
weights still authenticate35 leaves/326,079,840B per chip. Compiler journal,
raw graphs, allocations and allfive resident snapshot precede independent
generation-bound eight-host/32-owner collection. This is compilation evidence,
not runtime scratch, numerical equivalence or speed. Shared-prefix control remains
not independent full-layer validation. All numerical/capture mixed modes refuse.

38 CPU tests PASS7.83s, one unchanged composition test deselected; includes real
production6144 schema, compiler/journal, eight-rank collector, extracted shell DB,
zero-dispatch sentinels and close-failure voting. Earlier33 overlap. Reviewer P2
old four-graph DB item corrected to a distinct five-graph item and tested.
Conditional independent review: one compile-only acquisition after persistence,
mirror and fresh protected preflights. Exact graph/memory admission follows.

Local controller recovery: seven DB575 materialized trace copies (rank1–7),
2,101,827,129B, removed with root no-holder and sealed-generation/SHA/CRC checks.
Cloud originals retained; receipt `../artifacts/db575-local-trace-eviction-20260908.json`.
Free space497,627,136→2,599,481,344B. No bucket/weight deletion or TPU action.

## Actual acquisition sealed — DB593, 16:47:58Z

Run `greenfield_fp8_ws32_prefill_completed_window_acquisition_l6_20260908T164531308007030Z`,
pin `2bb606cd0106ed8302d894767c323f8c39dc7a7f`,98s worker/collector, zero WK/model
calls, normal/root8/8clean. Same-region archive161,837,560B/280objects. Exact
receipt `../artifacts/prefill-completed-window-five-graph-acquisition-20260908.json`
SHA11164d7feef5d20040809c1fce67bd4af488106afca47c657d8c67912989da2c.

Prefix/B128suffix/B32suffix compiler scratch91,736,576/31,054,336/11,256,320B;
outputs1,738,240/402,432/107,520B. Allfive code totals25,306,624B. Noaliases.
Prefix9physical subgroupcollectives; eachsuffix5. Existing actual-SSA FP32
route-sum checker passes bothsuffixes. Allgraph structural nohosttransport/
no fullfloatingweight expansion/localgroups checks pass. These observations
are not a registered exact profile or measured numerical HBM.

Independent evidence review: no material error;32owner validation inherited
from sealedcollector. Next register these fixed original graphs and simultaneous
five-executable/live-prefix-output budget, then separately reviewed numerical
protocol. No reacquisition or expanded symbolic proof. B128 suffix agreement
on sharedcompletedprefixes still cannot certify independent full-layer DSA.
