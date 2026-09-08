# Throughput priority: observe device costs, then optimize the dominant ones

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
