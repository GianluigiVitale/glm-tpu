# Frozen prefill — actual layer0 norm boundary

2026-09-09. Completion-only diagnostic, not optimization or a new baseline.
Authority: goal.md / §25. DB603 remains the accepted engine. DB605 seals the
original dense01 reproduction; it does not fix the 8K generated-token11 mismatch.

## What the evidence establishes

DB605 reproduces both DB604 full layer0/1 cache branches on all32owners. In
the live rows, layer0 returned residual and input-normalized state match between
schedules; layer0 output differs1200 unique BF16words, first at position44.
Layer1 input-normalized differs202words. Neither observation identifies a defect.

The existing fused RMSNorm adds BF16 attention update and combined residual in
FP32. It uses that unrounded sum for normalization and returns a separately
BF16-rounded residual. Equal returned residual does not imply identical norm
inputs. Saved HLO also shows a BF16[4,32,1536] scan-output stack feeding both
M128 dense projections (one via a completed copy). It does not establish actual
normalizedMLP equality or explain row44 in wide versus row12 in narrow.

## Implemented CPU-only mechanism

`scripts/greenfield/ws32_dense_norm_boundary.py` isolates the new diagnostic.
The original dense builder and all production kernels remain byte-identical.
Only the fixed four-tile scheduling adapter is reproduced; all model arithmetic
calls existing kernels. No legacy import, host callback, new checkpoint, precision
change, row-size sweep, sampling or TPU initialization.

`build_capture_program` uses the original prompt/embedding/two layer trees/WKs,
keeps layer1 unchanged, and records layer0's executing hooks as scan auxiliary
outputs. Carries remain the original three caches. The suffix remains B128.
Owner axes [8,4] preserve physical observations, including nonreplicated partials.

| Field | Local shape/call | dtype |
|---|---|---|
| post_norm update, residual, normalized, carried | [128,1536] | BF16 |
| post_norm summed | [128,1536] | FP32 |
| local_square_sum, square_sum, inverse | [128,1] | FP32 |
| boundary normalized_mlp | [128,1536] | BF16 |
| boundary live | [128] | bool |
| post_norm weight, once/call | [1536] | BF16 |

Fixed device packet:2,757,248B/chip. Save only live numerical rows on host,
weight once/call and the FULL128-row live mask. Five calls add22,096,384rawB/rank;
original model capsules81,933,312B become104,029,696B before suffix captures.
This is payload accounting,
not actual HBM admission. Keep packet capsules separate from original NPZs:
combining fields would exceed the existing reader's128-entry inventory guard.

`build_completed_dense_suffix` reuses the original dense MLP with completed
BF16 normalized inputs and physicalB128. Caller must first reproduce each
schedule's original output from its own captured inputs. Only then compare
identical captured inputs at the two fixed row placements. A separate executable
may perturb results; it cannot be assumed equivalent from HLO or CPU success.

## Exact relevance requirement

`ws32_dense_norm_originals.py` binds the reviewed DB605 original receipt, source
ledger, original runner and allfive NPZs by bytes/SHA/CRC. It reuses the bounded
NPY reader and compares encoded arrays without dtype coercion or tolerance.
Existing retained preflight now authenticates the original remote generations,
as detailed below; outer norm campaign routing remains disabled.

Require every RETAINED field on both layers and allfive calls. Endpoints retain
all12 fields/full cache pages; intermediate narrow32/64/96 retain9 row fields
and full128-row health, not caches. Never claim comparison of unsaved caches or
run extra hardware just to manufacture them. The existing64 DB604 endpoint
comparisons remain mandatory. If instrumentation changes any retained output,
preserve the packet and refuse attribution to the original realization.

## Decision from actual packet

1. Different update/residual/FP32sum: the difference reaches the norm input.
2. Equal sum but different square sums/inverse: normalization reduction boundary.
3. Equal norm operands/statistics but different normalized result: output boundary.
4. Equal actual masked normalizedMLP but different dense output: suffix realization
   or row placement requires the completed-input discriminator.

None alone establishes the8K token11 cause. Only a demonstrated correction followed
by this frozen path's own protected8K test can close that obligation.

## Verification and exact next integration

- Actual forcedCPU32 two-layer/five-call capture versus original builder:
  all outputs byte-equal, packet ownership/FP32sum/rounding/masks, poisoned padding,
  cache carry, invalid spans, completed suffix equality and M32 refusal PASS46.18s.
- Production abstract shapes/no payload reads/no device allocation and TPU-target
  raw lowering PASS48.78s. New capture raw663350B SHA
  `65379b5f5a87c2aef25687ce846e149749857a831924123a228ff267976f617c`;
  suffix raw28917B SHA
  `f9c1a7fa602a2c255fa85c0c5264b75e1241b0ab75aa3e434a7e27c432e7f73f`.
  These are not actual optimized TPU graphs or measured allocations.
- Original all8host/32owner/5capsule replay plus strict mutations:11PASS4.74s.
  First test caught an incorrect old launch_process_id field; actual DB605 uses
  launch_rank. Corrected locally without a TPU run or altered evidence.
- Independent final reviewer: noP0-P2;10negative tests1.41s. Supports CPU
  persistence only, never actual hardware admission or causal attribution.

Next adapt existing protected dense worker and original-array collector with a
distinct diagnostic identity, bounded packet publication, actual new graph/memory
admission and retained-byte reproduction before attribution. Reuse completed WK,
selected55-leaf loader, both leases, voted calls and 8host cleanup. Do not launch
the original nine-call campaign unchanged or run a new full-model/8K test yet.
No launcher integration, actual packet, cause, numerical promotion or speed claim
exists from this CPU-only implementation.

## Same-job continuation — implemented, hardware launch still disabled

`ws32_dense_norm_protocol.py` fixes exactly18 executable calls: four original WK
calls, five capture calls, five own-input completed suffixes, four cross-placement
suffixes. No new workload variant or performance sampler. `ws32_dense_norm_worker.py`
uses existing BudgetedCalls, all-live memory census, original12-field capture,
owner-local packet reader and atomic NPZ/JSON preservation.

The existing `ws32_dense_frontier_execution.execute` accepts a distinct fixed norm
protocol with authenticated originals, selects the new four abstract jobs and
NormJournal, then performs the same compile/WK/finalization lifecycle. Its original
nine-call mode remains the default. This is an internal continuation, not a CLI
or permission to run through the old campaign.

1. Preserve allfive original outputs and norm packets. Fleet-vote exact retained
   DB605 fields and all DB604 endpoint-cache comparisons BEFORE any suffix call.
2. Replay each captured normalizedMLP from its OWN packet at original placement,
   physicalB128 and original live count. Save output/live rows and full128 health.
   Fleet-vote exact output/health reproduction BEFORE any cross-placement call.
3. Move each wide packet's32-row segment at0/32/64/96 to the first32 rows of a
   physicalB128 suffix; zero padding. Compare live outputs with the original
   wide suffix's corresponding segment. Different results remain diagnostic,
   not permission to modify precision or a proven8K token11 cause.

`build_owner_packet_suffix` preserves explicit [8,4] owner axes for BOTH operands
and outputs. No expert replication is presumed and no hidden state is gathered
to the host. The adapter adds no communication; the original dense feature4
gate/up and expert8 down reductions remain unchanged. Distinct expert inputs
must be replayed together, not as independent replicated-input calls.
The earlier global-feature suffix is a CPU comparison helper, not
the deployed interface. Invalid start/count metadata yields false health.
Allfive device packets remain live and must enter actual all-live HBM admission.

`ws32_dense_norm_prepare.py` adapts the original selected55-leaf metadata-only
preparation. Both WK raw graphs and the capture graph remain unchanged. The new
owner-explicit suffix raw is33495B SHA
`b895cadecf1571b351e41bebdd440bfe14f5ead52cf69562aac92a2fb1defb23`.
Production abstract/no-payload/no-device-allocation preparation and allfour raw
lowerings pass45.71s. Raw HLO is not optimized TPU HLO or measured memory.

Original payload accounting per host:81,933,312B prior model originals +
22,096,384B norm packets +4,723,200B nine suffix outputs/health =108,752,896B.
Nineteen1MiB capsule allowances produce128,675,840B, below the original128MiB
model cap134,217,728B. Keep the separate96MiB WK and31MiB auxiliary caps;
reconcile actual original/reference/controller copies under the existing256MiB
worker and6GiB whole-archive budgets before launch. No checkpoint copy.

Remaining integration: independently bind actual observed norm weight; extend the
independent original-array collector for these packets and suffixes; acquire and
validate actual four-program HLO/memory with the CPU-tested inspector below;
route the distinct entry/campaign/publication/accounting path. Preserve allfour
graphs before any inspection refusal. Outer inclusive deadline and fresh controller
disk floor remain mandatory. Do not launch the old9-call campaign or retry8K.

Final CPU source review finds noP0-P2; independent worker/readers/compiled lifecycle
and old-mode suite73PASS14.31s. ActualCPU32 extended owner/placement/padding test
plus4reuse tests5PASS49.81s. Its first nonreplicated-input oracle was wrong:
evaluating each expert independently discards the original expert8 down sum.
Corrected direct mapped-MLP reference keeps every distinct expert input together;
no model arithmetic change. The test proves the adapter preserves ownership and
the existing collective semantics, not independent-expert computation or TPU
output identity. All completed originals precede failure votes and cannot be
overwritten via existing NPZ/pending/JSON or dangling pending symlink.

## Generation-bound preflight and runtime — CPU integrated

`ws32_dense_norm_originals.materialize` reuses exact-generation cloud reads and
the original bounded NPZ reader. Fixed immutable DB605 receipt authenticates each
rank's ledger, runner and five NPZs (seven objects). Region, size, CRC and SHA
must match. Existing files are verified, never replaced; existing or dangling
`.partial` paths refuse before download. Combined DB604+DB605 reference bytes
are checked against128MiB before remaining payload, with1GiB disk reserve.
All eight actual combined sizes, bytes in launch-rank order:
`30328911,30329491,30321026,51623966,51616199,51624029,30329178,51624346`.

Existing `ws32_dense_frontier_preflight` admits only the distinct fixed norm tag
and binds hostname, prompt/RoPE, checkpoint/source identity and owner metadata.
Existing `ws32_dense_frontier_runtime` reauthenticates these originals before
selected loading, binds live physical owners, then compares the complete loaded
55-leaf owner records to DB605 before RoPE placement/execution. This binds the
loaded norm weights; the collector must still verify the observed packet weight.
Norm preparation selects four jobs and supplies originals to the18-call executor.
The original nine-call mode remains unchanged; outer campaign is not yet wired.

58CPU tests56.82s cover actual archive/reference bytes, all eight owner joins,
fake-cloud exact generation and corruption/budget/disk refusals, before-JAX
preflight and fixture-runtime continuation plus historical regressions.
An initial runtime test caught integer dictionary keys changing type at JSON
comparison; use sorted complete owner-record lists, never weaken comparisons.
Independent current-delta review noP0-P2; focused11PASS3.54s after the partial
symlink correction. CPU persistence only: no actual packet, cause/fix,8K pass,
optimized norm TPU HLO, live HBM or speed result follows from this integration.

## Fixed compiler inspector — source-derived, actual TPU graphs still missing

`ws32_dense_norm_admission.py` reuses original dense capture loop/collective/
kernel/helper checks with the capture's own raw pin and unchanged WK checks.
The isolated suffix requires exactly three originalM128/K1536/N1536 raw calls,
the feature4 FP32→BF16 stacked gate/up sum and expert8 FP32→BF16 down sum.
No scalar votes, other collective or floating full-weight expansion is admitted.
Existing kernel interface/alias/liveness checker receives only an optional layer
resolver; suffix requires its exact unscoped ENTRY name and returns layer-1.
All historical callers keep the original strict scoped-layer behavior.

Helpers reuse original closed-copy completion on at most three U8[1536,1536]
weight copies and at most one s32[128] row-take annotation. No scratch allocation.
These are bounded source predictions, not claims about actual optimized helper
compatibility. Unexpected realization refuses only AFTER allfour original graphs
and compiler memory are preserved. Original selected-layer memory maxima and
separate actual all-live reserve/peak checks remain; CPU is not HBM proof.

Tests: new+original dense/actual retained helpers/reuse74PASS5.67s; earlier
new+historical kernel43PASS132.37s includes both protected full-model original
graph replays, with overlap. Independent review19PASS1.21s/noP0-P2; live allowed
annotation positive and live bad-helper mutations added afterwards to address
its test-coverage note. CPU TPU-target suffix raw33495B/b895cade... remains
byte-identical; raw debug metadata contains the registered kernel and reduction
scopes. This is not actual TPU optimized-HLO admission. Collector/outer wiring
and actual18-call packet/reproduction evidence remain prerequisites to attribution.

## Independent original-array collector — CPU integrated

`ws32_dense_norm_evidence.py` reuses the original five-capsule schema/cache reader
and authenticates all19 model NPZs/sidecars against exact headers, dtypes, counts,
bytes and digests. Every retained DB605 field and DB604 endpoint cache must
reproduce. Captured post-norm weights match the selected checkpoint leaf; each
own-input suffix must match its captured layer0 output and full128-row health.
Only then are four cross-placement outputs and nine norm/boundary field byte
differences reported, without claiming causality. Full live masks and health are
independently replayed; trimmed numerical padding remains worker-check-only.

The existing fleet consumer selects the fixed18-call journal/fourgraph profile,
retained-generation identity and all32-owner checks. Complete cache records join
by(branch,slot), cross records by(start,slot); runtime owner ordering cannot cause
a false refusal. Duplicates, invalid keys and changed values still fail.

36CPU tests39.86s: norm original replay/fleet and original transport/recovery/
campaign/reuse regressions. Corrected norm+oldfleet suite8PASS35.46s separately.
Independent owner-key tests4PASS1.43s, noP0-P2. Real DB605 originals and physical
identities are used, but norm packets/math/compiled execution and outer fleet
compute are fixtures: not actual TPU compatibility or an8K fix. Existing outer
entry/campaign/transport/accounting remains the next integration, followed by
actual fourgraph/HBM and same-job reproduction. No full-model retry yet.

## Existing protected campaign route — integrated, hardware pending

Kernel `ws32_dense_norm_boundary` selects tag stem `ws32_dense_norm_d01` in
`run_fp8_matmul_microbench.sh`, with zero samples and the original both-lease,
pre/post normal/root census,600s worker,120s upload and780s SSH protections.
The controller must have at least6GiB free before deployment. Existing probe and
entry select the fourgraph norm inspector and18-call terminal vote, never the
old nine-call run. All eight retained preflights finish before any TPU init.

The same exact-generation transport publishes19model+4WK NPZs/sidecars,
fourgraph pairs and reproduction/own_reproduction/cross_comparison JSON. Every
known pending NPZ is retained under its original model/WK cap; incomplete sets
cannot collect. Limits remain128MiB model,96MiB WK,31MiB auxiliary,256MiB/rank
and6GiB entire archive, including worker/controller copies. No checkpoint copy.
Controller collector and accounting reauthenticate fixed DB604/DB605 roots;
item `dense01_norm_db605_own_cross_eighteen_calls_v1` has NULL correctness,
score and latency. Recovery replays already-collected originals without TPU.

76CPU regressions66.61s include actual campaign-to-SQLite with fixture SSH/cloud/
compute. Independent entry/transport13PASS8.49s and outer review noP0-P2.
Two-mode recovery-to-archive2PASS3.29s, explicit census/cloud/math fixtures.
Actual optimized-HLO compatibility, live HBM and all capture/suffix relevance
checks are NOT established by CPU integration. One protected diagnostic follows
review, persistence and fresh preflights. No full8K retry or numerical fix yet.

## Actual fourgraph refusal and correction — 2026-09-09

Protected173035/cff9444f preserved allfour graphs, then refused before all WK/
modelcalls onall8. `../artifacts/prefill-dense-norm-compiler-refusal-20260909.json`
binds exact generations/CRC/SHA, memory and8clean. No DB/numerical result.

Capture optimized SHA226a23450d34472bb3da4f25e1862a22a18a22626705106cef538a9201ff624e,
2954163B: three f32[4,32,1] and one f32[4,32,1536] AllocateBuffers.
Norm-only callback reuses original helpers/merge scratch and prefix counter:
loop slots13/14/17/18 get exactlyfour own-slice writes; old contents only serve
their own DUS base, never update/model operands; completed leaves reach only
packet outputs27/28/31/32. Singleton forwarding uses FP32 ADD with negative-zero
initializer. This does not replace numerical packet relevance checks.

Suffix optimized SHA429b2c97f5d1a0ce4aea692a1ba5328206c568e15602c90b8b82c174a98600e8,
77418B: one s32[1024] annotation, not predicted128. Bind pad128+896, clamp0..127,
slice[0:128], identity index forwarding and original bf16[128,1536] gather.
Only three existing closed U8[1536,1536] copies otherwise. Both WKs unchanged.
Historical inspector defaults still reject new stacks; no production changes.

Allfour actual graph/raw/memory/JSON replays pass;61CPUtests20.45s include partial
writes, displaced counter, old-stack reads, root/output swaps/escapes, invalid
reduce/padding/clamp/slices/gather dimensions and old dense regressions.
Independent46PASS20.03s/noP0-P2. Then persistence/freshguards and ONE corrected
18call run; no acquisition-only job or full8K retry before demonstratedfix.
