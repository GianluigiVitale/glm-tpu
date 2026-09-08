# Throughput priority: observe device costs, then optimize the dominant ones

CURRENT21:06Z: DB596 seals paired-sort gain, prefix3.88x/partialwide3.18x,
DB594 byte reproduction32owners,8host trace/cleanup. Receipt in EVIDENCE_MAP.
Next existing B17/B11 full-engine paired flag and distinct own2K admission,
not another isolated phase baseline or B128 integration prerequisite. Preserve
oldprofile; CPU-preregister graphs and inspect actual HLO/HBM in one numerical
run. All historical "next baseline" instructions below are superseded by this.
Full-model prefill remains DB588~19.9tok/s;10K target NOT achieved.

2026-09-08, owner directive: extract maximum feasible throughput from all32 TPUv4
chips in prefill AND batch-one decode. A first correct implementation is not the
goal. This does not relax correctness, authorize infrastructure changes or claim
that100% of every hardware resource can be saturated simultaneously. Independent
Astra review and main-agent source/evidence inspection agree on the next test.

## What20 prompt tokens/s actually measures

DB588 used2034prompt tokens in120 full-stack B17/B11 calls at capacity8192.
Fleet-max request-prefill102.202932s is19.902tok/s. This is a measured slow
implementation, not a hardware ceiling and not the new B128 path's speed.
Rank0 `batched_prefill_complete.rank0.json`, execution fields:

| Component | Seconds |
|---|---:|
| Sum of120 compiled block dispatch-through-completion intervals |99.716047|
| Input placement/completion |0.080847|
| Cache initialization |0.225099|
| One-time all-live memory admission |1.327408|
| Remaining host/control/accounting, by subtraction |0.853311|
| Total request-prefill |102.202712|

Block calls account for97.5669% of rank0 wall. They include dispatch/completion,
so they are NOT device-only attribution. Removing every other measured cost
would save only2.4331%. Do not prioritize removing safety checks or blaming GCS.
The receipt is `../artifacts/prefill-batched-own2k-sealed-20260908.json`; originals
are in `/home/gianl/glm-run/greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1_20260908T090441883274696Z/`.

Decode remains129.171ms/token,7.7417walltok/s inDB588; it is separate from prompt
throughput. DB589 supplied-route MoE B128 versus8xB16 measured3.329x distributed
and1.280x concentrated. Those are not full-model speedups. DB594 now passes three
shared-completed-prefix/B128-versus4B32 suffix numerical cases on32owners;
it has no performance samples and is not independent full-layer DSA admission.

## Observability in use and the gap

`../suggestions.md` was reread in full38lines: make execution observable before
guessing fixes. Existing tools are not missing wholesale:

- Original StableHLO/optimized HLO, physical4/8collectives, no podwide repeated
  hidden reconstruction, raw compiler allocations and source hashes.
- Per-chip live-buffer budgets/post-call peaks and32physical owner mapping.
- Fsynced compiler/phase journals, completed operand/output arrays, exact route/
  own-score DSA and cache replay, all-host failure votes and authenticated cleanup.
- `watch_ws32_run.py`, `WS32_ORPHAN_RECOVERY.md`, DB/archive/generation checks.
- `run_short_decoder_ws32.py` already captures eight-host decode XPlanes.
- `scripts/analysis/parse_xplane.py::aggregate_fleet` already categorizes physical
  device events. Its decode-default module selection must not be used for prefill.

The material gap is a PREFILL trace explaining MXU/VPU activity, memory movement,
collectives, sorts/gathers and idle time on the critical path. Decode traces and
HLO labels cannot supply that attribution. Do not claim measured utilization,
bandwidth or percent peak where trace/counter support is absent.

## Next decisive baseline (bounded, not another numerical archaeology campaign)

1. Reuse DB594's actual completed-prefix/suffix/helper programs and selected
   layer6 weights. No new checkpoint, no full-model load for this phase question.
   Preserve the three-case numerical result; no new capture outputs or precision
   variants that change its realization.
2. Add a distinct default-off bounded phase-baseline mode to the existing runner.
   Preplace inputs; reuse completed WK and actual admitted programs. Compare the
   same128rows: fourB32prefix calls plus B128suffix versus the SAME prefix sequence
   plus fourB32suffixes. Include required device preparation, assembly, dispatch
   and completion in total wall. Exclude reference comparisons/archiving from
   timed samples, disclose their separate costs, and retain exact health checks.
   Reset to identical starting caches/offsets for repeated samples. The current
   comparison-only `assemble` helper consumes BOTH suffix results: do not count
   that combined workflow as B128 latency. Use independently assembled execution
   paths, or label phase-summed estimates explicitly until real path wall exists.
3. Capture a few SEPARATE traced samples with `jax.profiler.trace` on all8hosts,
   completing every named phase inside trace. Reuse fleet aggregation with exact
   executed prefill module names. Require8XPlanes/64cores and explicitly report
   unclassified time. No profiler-contaminated throughput samples.
4. Report prefix, suffix, assembly and total wall distributions, actual route
   occupancy/active expert visits, per-chip memory and stragglers. Synthetic
   history/one layer is phase orientation, not full-model or long-context proof.
5. Use observed dominant phases to choose the next change. Register full128K/
   256K prefill and delivered-TTFT targets from phase budgets BEFORE optimized
   candidate promotion; baseline gathering is not permission to fit targets.
   No thousands-tok/s promise without a supported hardware/resource budget.
6. The first short-decoder performance run must trace representative nonfinal
   prefill blocks AND the final tail separately from profiler-free request wall.
   Finish own8K and efficient four-depth128K/256K protections on the changed path.

CPU-test trace configuration/module selection and actual publication/collector
composition before deployment. Review only the new diff/evidence. Both leases,
fresh8host normal/root census and storage floors remain mandatory. Trace API
failures must preserve originals and propagate across hosts; never strand peers.

Storage correction: DB594's archive ledger covers1,708,766,880B, exceeding the
pre-run estimate of<1GB. Its complete live results prefix has518objects totaling
2,802,533,016B, including worker publication plus collected fleet copies. The
summary alone is171,923,721B; nested per-row numerical statistics and duplicated
worker records are not compact. Preserve this sealed evidence, but do not repeat
that layout for timed samples: retain original bounded arrays once, bind compact
aggregate/worst-row reports to them, and use generation-qualified references
instead of extra payload copies. Re-measure disk/storage budget before launch.

## Ranked optimization candidates, pending device attribution

1. **Insufficient MoE weight reuse.** B17 offers few rows per expert; scaling the
   routing window and using expert-relative panels are the main candidates.
   Test B128 then larger B512/1024 only with measured scratch and non-dropping
   skew handling. Increasing a tile constant alone is not a scheduling design.
2. **DSA and sparse attention work.** Attribute scoring, repeated selection,
   gathers and sparse attention separately. Exact vectorized merges, causal
   prefix skipping and fused gathers are separate candidates; preserve ties,
   causal cache ownership and unrepaired/repaired lifecycle. Reuse research1–3
   adjudications, not a fourth general research request.
3. **Cache/movement overhead.** Inspect actual copy/DMA cost and aliases before
   changing old/proposed/repaired buffer lifetimes. Never donate the only valid
   committed state or infer dominance from a source `.at[].set` expression.

For decode, use its existing critical-path trace to rank communication, FP8
formatting, selection/gathers and memory traffic; repeat protected wall only for
an actual candidate. Batch-one cannot borrow multi-request aggregate throughput
as a win. Compute-bound prefill and bandwidth/dependency-bound decode need
different kernels/schedules. Speculation remains separate effective throughput.

Stop expanding formal checker proofs once the existing bounded contracts admit
the real graph; spend the next hardware budget on measured bottlenecks. A profile
must lead to a ranked change, not become another open-ended tooling project.

## Phase sampler staged — 2026-09-08 18:50Z (CPU only)

`scripts/greenfield/prefill_phase_baseline.py` reuses the numerical worker's
extracted `execute_window`; no duplicate kernel or altered compiler stack.
Three warmups, ten untraced samples and two separately traced traversals reset
to the same initial caches. Each validates the exact19-call order. Partial
phase sums include preparation but exclude independent final assembly; the
existing comparison assembly consumes both paths and is reported separately.

BudgetedCalls still times dispatch through all-leaf completion, excluding
checks/capture/votes and separately reporting excluded costs. Full call-budget
witnesses stream once to bounded gzip32MiB/host,287calls including optional2WK;
JSON keeps compact offsets/SHA pointers. This does not yet bound the entire
future run's arrays/XPlanes/publication. Trace entry/exit and summary failures
vote; review caught and fixed an unvoted summary exception. Tests cover local/
peer summary refusal and nonfinite clock, trace cleanup and no successor.

Both actual suffix widths are named jit_suffix: trace reports their combined
10executions per two traversals; wall samples keep widths separate. Parser
categories are fleet means, not critical-path sums. Between-module cycle/idle
fields include intervening useful graphs and host checks, not hardware-idle
or MXU utilization. Preserve raw unclassified events and qualify FLOP/byte
estimates rather than assuming they are measured hardware counters.

27phase+4reuse tests PASS3.15s; original three-case worker/consumer replay
PASS88.47s. Existing independent Astra: no remaining P0-P2, CPU persistence
only. Next: DB594 original-output authentication, WK/worker continuation,
compact evidence consumer/generation-qualified collector/DB/trace publication,
complete storage preflight and current-diff launch review. No TPU run or new
speed claim. Do not repeat cleared numerical or graph-acquisition experiments.

## Original authentication and worker continuation — 2026-09-08 19:08Z

`prefill_phase_originals.py` derives an805220B capsule from32 SHA-bound DB594
files (runner, competitive original NPZ and two WK NPZs per host), retaining
their generation/size/CRC/SHA provenance. Every32physical slot has exact
component shape/storage-dtype/byte hashes and35selected-weight hashes. Capsule
SHA706813a99acb3095b4782e68ebfbd94b8a342d4f9d2bddf6b65982a0fe258ffa.
No new math reference or numerical verdict; these are sealed executing outputs.

OriginalVerifier retains one first-traversal capture; all repeats must match
the same DB594 bytes. First or later mismatches are retained before refusal.
Existing execute_numerical now has explicit phase_baseline=False opt-in:
distinct protocol/journal, CompactPhaseCalls selected before the two existingWK
calls, exact WK reproduction, then only the fixed competitive adapter. Partial
execution totals include the retained current entry if its archival fails.
No model/kernel/compiler changes. The acquisition and launcher do NOT yet
select this mode, so this is not hardware launch authority.

Stream reader validates one bounded gzip member at a time and its compact
pointer; refuses gaps, overlap, extra members, unindexed suffix, truncation,
SHA/field mismatch and oversize inflation. Semantic call-budget/journal replay
must still be connected to the collector; do not trust these byte checks as
memory admission. Tests69PASS7.50s, including all32 local archived outputs,
typed runtime binding, repeat/mismatch retention, stream mutations and actual
WK/phase routing with fixture devices; unchanged original three-case worker
consumer regressionPASS88.32s. Independent Astra P2 partial-counter finding
corrected and reviewed, CPU persistence only.

Next minimal integration: preserve existing five-model compile frames, select
the phase journal plus four helper compiles, consume streamed witnesses with
the existing budget/owner logic, fixed first-capture/trace publication and
generation-qualified collector. Exactly135model+150helper+2WK calls. Test actual
composition before launch. Whole-layer latency remainsNULL: phase sums are
not independent end-to-end performance. Restore current controller headroom
before launch if necessary; no weakened floor or new checkpoint.

Local disk recovery: root aggregate runner.json and results_ckpt.db copies of
DB594 evicted306885280B after exact remote generation/size/CRC/SHA and root
no-open-holder checks. All fleet originals, summary/terminal/ledger and primary
DB remain. Exact restoration recipe in
`../artifacts/db594-local-summary-cache-eviction-20260908.json`.

## Compact consumer — 2026-09-08 19:25Z (CPU only)

`prefill_phase_evidence.py` now reuses the ordinary consumer's graph/journal
and per-call memory checks. It exhausts287 gzip witnesses; lifetime peaks
remain continuous across all15traversals, including WK and final assembly.
Exact phase names/order and separate warmup/wall/trace inventories bind the
recomputed sums. First-capture input/component arrays and WK are independently
replayed against the capsule. Later repeat equality is the pinned worker's
checked execution record, NOT15independently replayed NPZs.

14CPU testsPASS13.34s use actual DB594 ninegraph pairs, first originals/WK and
real CompactPhaseCalls/PhaseJournal/run_samples with fixture traversal/math/
counters. Mutations cover missing final assembly, partial witness, wrong sums,
bool wall/visits, trace-vs-wall mixup, owner/scope/count changes, cross-sample
peak regression and failed/reordered/extra journal events. Existing independent
Astra finds noP0-P2; CPU persistence only. No TPU execution or new speed result.
Regressions66PASS92.55s (original actual3case worker/consumer plus prior phase
tests), ordinary window consumer16PASS60.61s. No repeated model acquisition.

Next connect the explicit mode through the ORIGINAL acquisition frames and
existing probe/campaign, publish bounded first-capture/witness/XPlane files,
join eight owners' records and distinct phase-estimate DB accounting. Actual
launch-chain composition and storage/preflight/review remain before hardware.
Do not repeat this local graph/numerical replay as a new TPU acquisition.

## Launch integration — 2026-09-08 19:49Z

Original compile frames/probe/campaign now select the explicit phase mode.
Actual ninecompile/WK/sampler and eight-rank original-array/stream publication/
collector/DB composition pass;89focused regressions154.22s,2publication cap
cases1.39s (one overlaps),4reuse1.86s. Reviewer caught helper PhaseJournal
compatibility and cumulative failed-trace upload bounds; corrected before TPU.
One closed original XPlane perhost, no payload copy at finalization, exact
hostnames/64cores/module counts. Null DB latency/correctness/score; max-host
unprofiled partial phase sums are explicitly not an independently assembled
full-layer path or full-model throughput. Trace category/cycle caveats stay.

One bounded phase profile next after reviewed commit/push/mirror and fresh
leases/8host normal-root census. Existing selected layer6 only, no checkpoint
copy. Trace128MiB/host, artifacts256MiB/host, receipt fleet2GiB maximum;
actual download bytes+256MiB free required before materialization. Full archive
including duplicate publications/summaries/DB budget<5GiB. Bucket fresh
19:45Z2,007,587,552,184B inUS-CENTRAL2, softdelete0. Local headroom1.152GB after
unused pytest fixtures removed; actual prelaunch recheck remains necessary.
10K prefilltok/s is the owner's requested target, not measured feasibility or
a one-hour guarantee. This wiring produces no new speed result by itself.

## Measured decision — 2026-09-08 20:29Z

DB595 SEALED: eight-host actual prefix trace locates84.23% of meanbusy in
gather/scatter, dominated by argsort-result permutation of candidate scores
17.051ms and positions4.447ms perB32. SparseMLA itself0.062ms. No claim that
84% of all modelwall or chip utilization is explained. Profiler-free fourB32
prefixes129.546ms per128rows; wide suffix10.460ms, narrow24.515ms. No end-to-end
speed promotion; independent assembly is absent from partial sums.

Next implementation replaces only those two permutation gathers with paired
stable sort (positions sole key, scores payload). It is now default-off and
CPU-exact through the real completed-prefix builder;33 tests pass, independent
review clear. Do ONE combined changed-prefix compilation/admission, DB594
reproduction, wall+trace run. Refuse before timing if reproduction changes;
inspect actual memory/physical groups. Do not run another baseline, separate
acquisition or progressively smaller numerical taps. Then integrate measured
winner into full prefill and run its own short decoder before8K/longcontext.
