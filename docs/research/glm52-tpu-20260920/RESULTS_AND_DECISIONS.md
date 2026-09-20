# Frozen GLM-5.2 TPU performance research — 2026-09-20

The owner stopped this campaign on September 20. This is a research record,
not a plan to resume it. Pending experiments are cancelled or untested, not
negative measurements. The final answer comparison was stopped before any
answer case completed. No more TPU experiments are required for this freeze.

The retained ordinary research path delivers approximately **14.3 wall output
tokens/s**. Its qualified short 2,034-token request measured **138.85 prompt
tokens/s and 14.04 wall decode tokens/s**. MTP did not produce a qualified
improvement: long outputs diverged from ordinary, and its verifier dominated
runtime. The hardware's ultimate throughput ceiling was not established.

## Record and recovery

- [Exact measured comparison rows](MEASUREMENTS.md), including failed answers.
- [Artifact register](EXPERIMENT_REGISTER.md): original paths, schemas, hashes
  and recovery links for the removed receipt files.
- [Historical notebooks](history/README.md): the full development chronology,
  numerical contracts, reference reviews, unsuccessful approaches and source pins.
- [Stop, archive and publication record](FREEZE_OPERATIONS.md).

The pre-cleanup research commit is `9dedce4b`, preserved on the private branch
`preserve/glm52-tpu-research-freeze-20260920`. Original files can be recovered
with `git show 9dedce4b:<original-path>`. The experimental implementation stays
on the perf branch. Frozen `MODEL_SOURCE=edecdd94` is unchanged. Publishing this
record does not deploy that implementation into the release engine.

## Hardware, model and measurement boundaries

The target was GLM-5.2-FP8, 78 target layers and its native layer-78 MTP module,
on the existing `db-v4-64-od` pod: eight hosts, four v4 chips per host, 32 chips
total, `us-central2-b`. WS32 uses feature-four/expert-eight ownership. Experiments
were serialized under workload/pod leases; CPU pytest never used the pod.

We compared real geometry with synthetic weights for kernel diagnosis, then
checked admitted candidates with real weights and the preserved DB610 trail.
Synthetic health flags, short token parity, cache tolerances and completed-answer
correctness are separate evidence. None substitutes for the others. Model-only
decode rates exclude work included by delivered wall rates and must not be mixed.

The historical release DB610 baseline was 62.761 prompt tokens/s and 7.660 wall
decode tokens/s. Its 128K/256K campaigns remain protected historical evidence;
they were not restarted. Research rates exclude cold loading/compilation and
network transport. Long delivered rates include required host votes and rank0
token-file write/flush. The initial token produced by prefill is excluded from
the timed decode-token numerator in the preserved suite.

## What worked and remains useful

| Mechanism | What changed | Evidence and retained boundary |
|---|---|---|
| D1 grouped routed experts | Group selected rows by local expert and reuse expert work; explicitly write zero for empty route slots | An empty-owner defect was reproduced and fixed on TPU. Only results after the fix qualify. Grouping survives in the ordinary research path. |
| D8 resident BF16 non-routed tables | Decode attention, DSA, dense/shared tables once; leave routed expert weights FP8 | Removes repeated v4 software FP8 conversion, at additional residency cost (early estimate about 1.8 GB/chip). CPU numerical boundaries and trained short token checks are documented. It is not universal bitwise equivalence. |
| D10 DSA | Score physical cache pages without gathering full key vectors; use a checked two-stage shortlist | Reduces gathers and sorting. D1/D8/D10 matches DB610's 29 tokens on every host after the routed fix. |
| D4 packed host loop | Reduce host tree materialization while preserving health votes and delivery | Paired real-weight wall rate 13.32 to 14.04 tok/s, +5.44%; final state/residual bitwise equal in that trial. It used an in-memory delivery sink, five excluded warm steps and 23 timed steps. |
| D8/P1/P2 prefill | Reuse resident BF16 tables, local attention reduction and tile selection/merge work | Qualified short request-loop prefill 138.85 tok/s; clean no-D5 run 138.95 tok/s. These are about 2.2x the historical DB610 prefill rate, not long-context or broad quality proof. |
| Native MTP infrastructure | Acquired, packed and loaded native weights; implemented draft generation, acceptance, commit/refresh and rollback guards | Real MTP runs completed. Infrastructure working is distinct from speculative target correctness and speed. No independent trained-native reference parity was established. |

The clean no-D5 real-weight run measured 14.55–14.69 model decode tok/s,
66.98–67.70 ms p50, with 29/29 DB610 agreement. Those model-only rates exclude
host checks and delivery. The representative ordinary suite instead measured
14.27–14.44 delivered wall tok/s; its six-case aggregate is **14.3175 tok/s**.

## Why early impressive numbers are not the frozen result

The first synthetic generator gave unequal values to nominal replicas by folding
both mesh axes into partially sharded weights. It was corrected to fold only
partitioned axes. The old receipts are exploratory timing evidence, not valid
replicated-model exactness proofs.

A separate routed-kernel bug left empty-owner output slots unwritten. A real
model trial produced zeros after its first token despite metadata health passing.
Correcting that write removed the explosion but still gave only 17/29 DB610
matches. Full ablations then isolated the remaining discrepancy to decode D5:
D1/D8 and D1/D8/D10 each passed all 29 tokens; D1/D8/D5 reproduced 17/29.

Consequently the often-quoted 72.1 ms, 64.3 ms and 15.57–15.60 tok/s synthetic
figures are not qualified serving baselines. A finite final cache or healthy
metadata is insufficient to certify equivalent numerical execution.

## Rejected or inconclusive numerical and kernel candidates

**Decode D5, local LSE merge.** Queries are gathered, each owner attends its
selected keys, then partial outputs are merged. It reduced communication in
microbenchmarks, but local normalization and BF16 rounding introduced a different
numerical path. CPU fixture bounds did not prevent a different trained greedy
token. Decode D5 stays disabled; this does not reject the separately qualified
prefill composition.

**Fused feature reductions.** Q-a/KV-a/WK/head partials were concatenated for a
single feature collective while preserving intended split/rounding boundaries.
CPU checks were bitwise equal, including real projection geometry. The synthetic
TPU full-step candidate produced zero tokens and was slower (roughly 122 ms
versus 64 ms in the then-used, subsequently bug-affected comparator). It remains
disabled. Those rates are not an equivalent-output performance comparison.

**D9 routed FP8 conversion.** TPU v4 software conversion was already near its
measured floor: roughly 43 of 46 microseconds per 3 MB in the diagnostic. Packed
conversion was about 2.5x slower. The routed path remains conversion-bound in
that analysis. INT8 weights would change numerical scope and were not admitted;
this campaign did not establish an INT8 full-model speedup.

**Global-max attention from Kaggle.** This adaptation gathers queries, uses a
global score maximum, reduces FP32 numerators/denominators, and normalizes at
the end. It still rounds exponent weights to the key dtype before their value
product; it is not wholly FP32 attention. On all 32 chips, the synthetic primitive
was 1.20–1.22x faster for three queries and 2.35–2.38x for 32-query tiles; one query
was slightly slower. Empty owners, causal/page boundaries and small numerical
fixtures were checked.

The trained verifier replay rejected it: code R2/R3 differed at token index 5
on every host and committed caches differed. R3 target-only windows took
144–149 ms versus 200–202 ms for three ordinary steps, excluding draft,
acceptance, refresh/commit and delivery. No serving speedup follows from this.
The separate trained prefill trial passed DB610 29/29 at **135.13 prompt tok/s**;
recent ordinary prefill was **140.59 tok/s** in a separate run. That observation
is about 3.9% lower, not an alternating paired estimate. Longer-context pairs
were cancelled before measurement.

**Public fused EP MoE.** The source was adapted to EP32 because the original
feature-four/expert-eight placement needs a feature reduction before activation.
Two/three target rows needed padding to 32; layout conversion and scale expansion
were startup work, while input/output redistribution would be timed. Default
public scratch exceeded v4 VMEM. M64/single-buffer declared scratch was estimated
at 10,234,880 bytes. An actual-loop CPU simulation found and corrected a
single-buffer refill hazard; full CPU DMA interpretation could not complete.

The bounded real-geometry TPU test failed candidate-R2 compilation on **all eight
hosts**, requiring **35.39 MiB VMEM versus 16 MiB available**. Compiler layout
padding, temporaries and spills exceeded the declared-scratch estimate; reported
spill slots alone were 4.99 MiB. No candidate numerical or latency trial ran,
including the planned prefill cases. This adaptation is rejected at memory
admission, not measured as slow and not proof against fused EP on other hardware.
No further kernel redesign was attempted after that decision.

## MTP and speculative decoding: measured outcome

MTP is the drafter inside speculative decoding. R2 verifies a pending token and
one proposal; R3 verifies the pending token and two proposals. They are not two
independent accelerators that can simply be stacked. A wrong proposal should be
rejected without changing the target's ordinary greedy result.

| Preserved request | Ordinary wall tok/s | R2 | R3 |
|---|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 |
| Structured | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 |
| Entire fixed suite, tokens/summed wall seconds | 14.3175 | 13.5546 | 14.6093 |

The aggregate R2 difference is -5.33%; R3 is +2.04%. Neither reaches 25%, and
neither speculative mode is qualified because outputs diverge. These historical
runs used component profiling and ordinary-first order. The proposed new
profiling-off/alternating comparison never completed and must not be implied.

Long-output mismatches appeared at code index 5, prose index 6 and structured
indices 816/823 for R2/R3. DB610 passing did not establish long-output parity.
Prose completed but had technical errors; the code case used all 7,168 tokens
without finishing; structured values were correct but unwanted Markdown fences
failed the requested standalone format. Token equality and task correctness are
different gates. Peak observed suite HBM was 28,228,678,144 bytes/chip at capacity
8,192, a run-wide high-water mark rather than an independent per-mode peak.

In the first structured R3 case, verification used about 114.13–114.19 seconds
of 128.689 seconds decode wall. It emitted about 2.924 tokens/round out of a
maximum three, while draft plus refresh cost about four seconds. High acceptance
therefore did not imply a large speedup: verification itself remained expensive.
The measurements are limits of these implementations, not a hardware MTP ceiling.

## Correctness isolation and unresolved arithmetic

The first same-prefix replay reset ordinary and R1/R2/R3 to identical committed
state and compared every commit count. All ranks reproduced the code R2/R3
mismatch at index 5 and prose R3 at index 6. Every zero-prefix commit was bitwise
equal. Across 1,344 full-versus-written-span checks, numerical cache differences
were confined to accepted spans. Divergence existed in target evaluation before
acceptance; there was no evidence that accepting a bad draft alone caused it.

Layer/head instrumentation located first-layer R2/R3 differences when it preserved
the original result. R1 instrumentation changed hidden outputs itself despite
token agreement, so its apparent first differing layer was not a reliable cause.
This is why the trace cannot claim a complete arithmetic root-cause diagnosis.

Four/five-row CPU variants failed the unchanged numerical envelope. One R4
batched residual error was 0.1640625 versus the 0.0625 absolute-error limit, with
committed KV also outside bounds. Unrolling KV preparation alone or switching
only attention order did not fix the full gate. Fully unrolled attention passed
stronger R1–R4 CPU bitwise state checks while retaining pooled expert work.

On trained inputs, unrolled R3 matched predictions in four short reset windows
and took 142–147 ms versus 197–205 ms for ordinary three-step windows. R2 still
diverged; nonzero commits still differed in cache/metadata. This was not enough
to qualify long speculative execution. Wider MTP was not admitted or pursued.

## What the public implementations actually supplied

The Kaggle source pin was `1aa1f083ae253470d9f355fe9c3eb12003300e91`, covering
both Qwen and GLM model folders. Original `vllm-project/tpu-inference` was pinned
at `9cab26a702c448c40710f504360d0d9f78e227a7`. The historical reviews retain exact
files/functions, PR states at review time, downloaded-source hashes and licensing
boundaries. No claim is made about their later upstream state.

- Multi-query verification/page crossings (#3332): relevant boundary tests and
  window sizing were considered; this runtime has no upstream RPA classifier to
  patch directly. Lowering did not show a prefill-sized query interface.
- Proposal compilation/device rejection (#2533/#2535/#2610): draft construction
  was already compiled for exact small row counts. Profiling-off and device
  acceptance were implemented and CPU-tested; their short TPU comparison was
  prepared but cancelled before launch. No speed claim is available.
- Occupancy/group indexing (#3219/#3476): division-based row indexing and M8
  active-group bounds were already represented and tested. No new gain is
  attributed to mechanisms already in the baseline.
- v4/token alignment (#2324/#2248): BF16 weight conversion, FP32 scores and
  restoration of original token slots were audited. Open patches were reference
  material, not a validated replacement engine or a known missing parity fix.
- Selected last-query IndexShare, one post-norm and compact shard argmax were
  already represented. Our own sparse primitives were not independent validation.
- Qwen GDN rollback was useful as an invariant reference, but GLM5.2 has no GDN.
  DeepSeek-v4 sparse MLA has different cache packing, geometry and features; it
  was not a drop-in replacement for the GLM BF16 640-wide cache.

## What was stopped, and the frozen conclusion

The final ordinary/global-max-prefill answer suite launched as
`perf_real_ordinary_suite_20260920T152417Z`, immutable source `38a34b4d`.
It used the original prose/code/structured cases plus separate scheduling and
7,671-token prompt controls, with two alternating repeats. The owner stopped it
before any answer case completed. All eight authenticated workers were stopped,
and the controller confirmed idle hosts at **15:43:39 UTC**. Its interrupted
preparation is not a completed answer, throughput comparison or failed model test.

The short host-blocking/host-unprofiled/device-unprofiled DB610 control matrix
passed 127 CPU integration checks; release checks passed 524 tests with one
skip and two warnings. Its controller was never launched. The near-8K paired
prefill result, completed scheduling answer, profiling-off speculative speed and
independent trained-native parity therefore remain **unmeasured**, not disproven.

Freeze the ordinary research result and retain the rejected implementations in
Git for reproducibility. Do not enable MTP, decode D5, fused input reductions or
the failed fused-EP adaptation in serving. Do not call a component gain a model
gain, or use incomplete answers to claim correctness. The owner ended the campaign;
closing its history does not claim that every earlier planned test was completed.
