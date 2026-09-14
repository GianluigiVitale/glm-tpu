# B128 layer6 refusal — original evidence and smallest next discriminator

Status: 2026-09-08. Numerical admission FAILED, not a speed result. Historical
Gate D/DB567 and batched2K/DB588 remain sealed. Larger-window prefill, its own
competitive8K and efficient L7/L8 remain open. This note grants no TPU launch.

## Exact failed run

- Tag: `greenfield_fp8_ws32_prefill_layer_window_numerical_l6_20260908T132240925858058Z`.
- Executed pin: `0f994e373dade76ede59da179da6856d17234e5d`.
- All4 actual graph profiles passed; completed WK decode/promote, one candidate
  B128 and four B32 controls. Seven executable calls total; five model calls.
- Failed at `boundary/comparison`: `window/control ordered DSA selection differs`.
  No competitive or tail case ran. No timing/performance campaign ran; no DB row
  or SUCCESS exists for this failed numerical trial.
- Wrapper terminated exit1 at13:25:07Z. Its normal census and root device/libtpu
  census each confirm eight distinct clean hosts. No model retry was launched.
- Originals: `gs://driftbench-dsv4-uc/results/<tag>/workers/rank0/` through
  `rank7/`,16 files per rank plus each `worker_receipts.json`.

The read-only `scripts/greenfield/analyze_prefill_window_failure.py` downloads
the exact recorded generations, verifies sizes/CRC32C/SHA256, original fixture,
four graph identities across hosts, selected checkpoint tensor hashes, eight
unique processes and actual32-device/slot mapping. It replays original arrays
without changing the admission comparator. No checkpoint payload is copied.

Receipt: `docs/artifacts/prefill-window-boundary-refusal-v2-20260908.json`.
The earlier `prefill-window-boundary-refusal-20260908.json` remains disclosed:
it had not revalidated physical device-to-slot assignment. Independent review
caught that omission; v2 supplies the missing mapping proof and reproduces the
same numerical findings. Neither receipt promotes the failed trial.

## What the arrays establish

All32 owners agree on the replicated observations; same-feature expert replicas
are byte-identical. Four feature shards remain distinct where appropriate.

| Observation | Original B128 versus fourB32 |
|---|---|
| DSA order changes | rows2,52,69,82,101,121 |
| DSA selected-set changes | none; this boundary has only506–633 causal keys, below2048 |
| Own selected-score order/ties/count/padding | pass on both paths; not a competitive full-score-row proof |
| Ordered router changes | 15rows |
| Router selected-set changes | rows2,10,32,62,113 |
| Pre-attention normalized input | bit-identical |
| KV | written differences at slots0–3; within unchanged bounds |
| Unrepaired index | written differences at slots0–7; within unchanged bounds |
| Repaired index / untouched cache bytes | bit-identical on all32 owners |
| Output/residual | unchanged aggregate and every-row bounds pass |
| Maximum output/residual absolute difference | 0.001956939697265625 / 0.000244140625 |
| Route weights, raw slotwise | fail; maximum0.009262464940547943 |
| Route weights aligned by expert ID on123 matching-set rows | bounds pass, maximum0.00008444488048553467 |

The last comparison is diagnostic ONLY: it excludes the five changed-set rows
and cannot turn the exact ordered-router failure into a pass. Set changes:
row2 expert144 versus208; row10 206 versus200; row32 39 versus234;
row62 220 versus239; row113 250 versus200 (candidate versus control).

## What is not established

This is NOT proved to be harmless rounding, a faulty router, a faulty selector,
or hardware bypass of a BF16 conversion. Matching pre-attention normalization
is not matching router input. Matching selected sets below2048 is weak evidence:
all valid keys fit. Source-audit correction: attention canonicalizes positions
privately, so raw DSA permutations alone do NOT change its traversal order.
Most changed-route rows do not coincide with changed DSA-order rows; that alone
does not locate the cause because attention/query and normalization arithmetic
can also change without a selected-set change.

Maximum written KV difference is3.725290298461914e-9; unrepaired index difference
is0.001953125. These changes are upstream of routing and MUST be included in
localization. The initial rank0-only summary had incorrectly generalized exact
caches to all32 owners; independent review of v2 caught that interpretation error.
The fixed graph/source identities and bounded cache checks rule out some wiring
failures, not floating-point arithmetic differences inside the prefix.
Do not retry unchanged, insert speculative barriers, emulate a legacy reduction
tree, widen tolerances, or replace the reference on the strength of these arrays.

Additional rank4 original-generation replay locates its sole changed KV element
at prompt-relative row38, and its two changed index elements at rows42/98 (slots0/4).
This is later than the first DSA order change at row2. In a correct causal path,
those later cache differences cannot explain that first changed selection.
The earliest DSA discriminator must therefore include query/head/scorer operands,
not only the downstream router; no specific arithmetic mechanism is proved yet.

## Smallest next discriminator

Use the same retained layer6 subset and boundary fixture only. Reuse existing
layer/DSA/router primitives and protected selected-layer worker/publication;
do not build another engine or repeat full-model/cleared layer0/3 tests.

1. Capture candidate and control's actual DSA query/head-weight/current-key
   operands at the first differing selection, plus attention update, combined residual
   BEFORE postnorm, completed BF16 normalized MLP input, router partial/global
   logits, correction bias, biased scores, ordered routes and route weights.
   Preserve original selection/cache/output observations needed to compare to
   this failure. The original `normalized` field is pre-attention, not MLP input.
2. Observation changes may perturb compilation. First require reproduction of
   the original failure's route IDs and DSA order/scores, and quantify whether
   original output/cache/residual fingerprints reproduce. Do not attribute the
   original failure from a diagnostic whose relevant boundary has changed.
   Preserve such a perturbation as a separate result; no automatic retry series.
3. On completed captured BF16 router inputs, replay the existing router at B128
   and fourB32. Compare against FP64 CPU dot/selection using the same captured
   input and checkpoint-bound weights/bias. Use each path's own input when they
   differ; a reference made from one path's inputs cannot adjudicate the other.
4. Decision: identical input with divergent routing localizes router lowering.
   Different input with own-input-correct routing redirects to attention/postnorm.
   Wrong own-score selection localizes selector/metadata, not a tolerance issue.
   DSA row2 needs its own-input score replay with identical causal keys, since
   later changed cache rows cannot explain it. Expand attention projection
   captures only if the first differing boundary justifies them.

This is a bounded **diagnostic**, not a replacement numerical-admission protocol.
The owner-facing external research request is
[the preserved historical research brief](https://github.com/GianluigiVitale/glm-tpu/blob/b667f00f1ae48c8ff37e92500550c1395d74c66d/docs/greenfield/PREFILL_NUMERICAL_BOUNDARY_RESEARCH_BRIEF.md).
It asks for targeted compiler/numerical localization advice; local work does not
wait for that report and its proposals do not automatically change this protocol.
Its new graphs, output schema, memory budget, provenance and original-signature
reproduction need focused CPU composition and independent review before launch.
No new model/weights, full decoder, competitive/tail rerun or performance sweep
is needed to answer this first question. Stop expanding the diagnostic once
the first differing completed boundary is identified; fix or adjudicate that
specific difference under the existing contract, then resume larger-window work.

## CPU capture implementation — 2026-09-08, not deployed

Actual kernels now accept an optional trace-time `_observe` sink (defaultNone).
The sink is constructed INSIDE `build_layer_programs`' mapped candidate and its
arrays leave through explicit device outputs; no host callback or per-stage
Python dispatch is introduced. Fixed tile0/32/64/96 namespaces prevent repeated
tracing from appending/overwriting captures. All observations are outside JAX
control-flow bodies, so no loop-local tracers escape. Original12 outputs remain
available alongside the new capture dictionary; ordinary callers retain12 only.

Captures: actual pre-selector DSA query/head/currentkeys/logicalpage keys and
positions/causal lengths/live masks; actual postnorm twoBF16inputs, unrounded
FP32sum, local/global square sums, inverse, weight and output; actual attention
update/combined residual/completed normalizedMLP; actual router input/clean/live,
checkpoint router weight, localpartial/local/global logits/bias and executing
sigmoid/biased scores/routes/weights. No second prefix implementation or copied
arithmetic supplies these fields. Unmasked selector weights stay distinct from
the normal result's padded-zero route weights.

`scripts/greenfield/prefill_window_boundary.py` reuses those builders for B128
and B32, plus the actual prefill router for same-completed-input replay. The old
layer3 router diagnostic only admits<=32 and remains unchanged. Its host reader
reads addressable shards only, verifies explicit expert/feature singleton indices
against authenticated device→mesh slots and rejects missing/duplicate/misassigned
owners. All12-field reproduction reports bind dtypes/shapes/bytes/SHA and flag
signature changes separately; neither result can promote numerical admission.

CPU32 actual full-indexer+MoE fixture passed capture/plain equality, fourcontrol
prefix comparisons, reversed physical placement, row/owner/causal metadata,
actual FP32norm inputs/sums, completed-input B128/B32 router replay, retracing
and poisoned partial tails. This is CPU mechanism evidence, not TPU identity.
Production shape tracing uses abstract checkpoint leaves:96 B128 fields,
33 B32 fields, extra output payload7,552,896/1,962,720 bytes per chip respectively.
These sums are NOT compiler temp, total HBM, wire traffic or archive-size bounds.

NEXT: integrate a boundary-only diagnostic into the existing selected-layer
worker/publication path. Preserve original failed-run generation pins, and make
original-signature reproduction precede attribution/replay conclusions. Budget
all resident programs, original/control/capture buffers and each active scratch;
publish partials before fallible checks. New graph outputs AND changed source
locations require their own reviewed graph identity/admission, not reuse of
DB590's raw hashes or broader coordinate masking. No TPU workflow, worker flag,
new admission, changed tolerance, full-model rerun or speed claim in this commit.

## 14:33Z — boundary compile-only integration, CPU evidence

Distinct `ws32_prefill_window_boundary_acquisition` kernel/tag uses the existing
protected layer6 worker, collector and wrapper. It compiles actual captured B128
and B32 plus unchanged-computation WK decode/promote from abstract inputs, keeps
all4 executables resident through the snapshot, and never dispatches them. Mixed
capture/numerical mode refuses before compilation. Old DB590 remains historical;
new model/source/output graph hashes are mandatory.

Actual `compiled.out_info` records original12 and96/33 capture shapes/dtypes and
per-field owner shardings. Schema bytes are fsynced BEFORE validation; malformed
shape/sharding/tree observations retain their error and still collect control.
This is compiler-reported metadata, not independent output-value proof. Replay
binds raw graphs, compiler allocations, schema/error/journal, selected32 owners
and all8 hosts. DB correctness/score/latency remain NULL. Fresh compilation does
not establish original-signature reproduction or numerical HBM safety.

38 CPU acquisition/boundary tests pass57.70s, including the actual production
abstract preparation, installed-JAX compiled output API on a cheap shape-only
graph, complete worker/journal/JSON/collector/DB flow, and failure preservation.
Independent Astra review: schema-preservation P2 corrected, no remaining P0-P2;
compile-only deployment conditional on historical regressions/persistence and
fresh protected preflights. No new TPU run at this entry.

NEXT acquire these4 actual graphs, then budget/register the instrumented
numerical diagnostic and capture only the original boundary case. No full-model
trial, precision change, relaxed comparator or claimed cause.

## 14:42:09Z — actual capture graphs acquired, DB591

Run `greenfield_fp8_ws32_prefill_window_boundary_acquisition_l6_20260908T143917951158196Z`
at5e7143759b6563237751b55c89007bd506d59bb1 sealed after119sworker/collector.
All8 workers compiled4 programs and executed0; normal/root cleanup8/8. Actual
capture schemas96/33 agree across the fleet with no schema/inspection errors.
Candidate/control:28/13 local collectives, no host transport/full weight expansion;
temp201,563,136/95,899,136B; output11,830,272/3,758,592B;
code49,949,184/17,284,608B. Compile-residency peak326,458,368B is NOT numerical
peak. Original graph hashes, SUCCESS/summary generations and cleanup digests:
`../artifacts/prefill-window-boundary-four-graph-acquisition-20260908.json`.

Do not reacquire unchanged graphs. Next numerical admission must use these
actual raw identities, simultaneous program/buffer/scratch budget, selected
checkpoint ownership, original failure generations, and original-signature
reproduction before interpretation. Existing one-layer protection/publication
path remains the implementation base. First-case only; no competitive/tail or
performance sweep yet. Own8K/efficientL7/L8 remain open.

Owner researchv3 was read in full while this acquisition ran; see
PREFILL_RESEARCH_V3_ADJUDICATION.md. Important new corrections: actual attention
canonicalizes selected positions, both scorer paths are B32, and installed
top_k already guarantees lower-index ties. Avoid replaying a raw permutation
as if it necessarily changes production attention order, or adding redundant
tie repair. The current captures include both original outputs and operands;
row2 selected scores cover its entire live history, so another scorer tap is
not automatically needed. Research does not replace numerical evidence.

## 14:57Z — fixed DB591 graph and simultaneous-memory admission

`scripts/greenfield/prefill_window_boundary_admission.py` registers all four
DB591 graphs, not DB590's uninstrumented model. Exact StableHLO and optimized
bytes except the same seven outer host coordinates; model metadata, stack
topology and opaque bodies remain bound. Original compiler output schema
digests bind all12 original and96/33 capture fields and shardings. Existing
physical paired collective schedule is unchanged; both actual FP32 expert
combine proofs pass. No new symbolic arithmetic project or reacquisition.

All four executable code allocations total67,827,200B. Per-dispatch budget
reuses all-live buffer accounting, adds active output/scratch plus all resident
code, retains the1GiB reserve and subtracts no aliases. Captures and earlier
cache generations must remain counted. This is CPU admission, not a measured
numerical peak or launch permission.

50CPU tests pass9.85s (46new plus4registry), none skipped, including all four
actual original graph replays, strict serialized report/schema/type mutations,
host-only coordinate shifts and all-live memory refusal. Independent existing
Astra reviewer finds no substantiveP0-P2, approves CPU persistence. First test
run caught a512-byte typo in the test's expected code sum; corrected against
the fixed allocations, not by changing the budget or acquired evidence.

NEXT: wire a distinct boundary-only numerical mode through existing worker/
collector/protected wrapper. Keep original compile call stack. Bind original
failed generations/32owners, reuse completedWK and BudgetedCalls with this
new memory profile, preserve actual12outputs AND captured operands before any
fallible publication/comparison. Exactly2WK+1candidate+4causallycarried controls;
no numerical verdict from the old failed comparator. Compare original signatures
first, report perturbation explicitly. Same-input replay is a separate graph
requiring its own admission if later evidence calls for it. No launch yet.

## 15:09Z — boundary-only capture worker staged, CPU only

`prefill_window_boundary_worker.py` now provides the distinct diagnostic journal,
fixed original-v2 receipt binding and boundary-case executor. It verifies the
synthetic fixture against archived hashes, binds original fingerprint slots to
the physical mesh (not device-id==slot), and retains original generation/CRC/SHA
sources in the run record. Fingerprint comparison checks all12 fixed-shape/dtype
outputs separately from the five signature fields. Perturbation is an explicit
diagnostic outcome, never a numerical PASS or permission to attribute the old
failure to a new boundary. Attribution must inspect relevant full-output matches.

The existing numerical continuation gains an explicit defaultFalse diagnostic
mode and reuses completedWK, per-call census/votes and journal finalization.
BudgetedCalls receives the DB591 budget function, leaving the old default intact.
Only1candidate+4controls execute after2WK. Each control carries actual original
KV/unrepaired/repaired outputs; captured extras cannot replace cache state.
Every original12 and captured operand is saved before health/schema checks;
individual control originals remain alongside their concatenated comparison.
BF16 captures use rawuint16 storage with dtype/shape/SHA manifest, no host fetch
of global arrays. No same-input router program or timing sweep is dispatched.

Independent Astra review found a missing finiteness check on new observations:
healthy final outputs do not imply finite norm/router intermediates. Fixed after
persistence for all BF16/F32 capture fields of this fully live128/32 fixture;
original score padding is separate and legitimately contains negative infinity.
Injected NaN/Inf with healthy originals refuses aftercontrol0, retains bytes,
and dispatches no successor. Review clears CPU persistence after tests.

Evidence:30tests77.50s (newcapture+historicalworker), plus4new early continuation
scope/profile/protocol/mesh refusal tests1.21s;34distinct cases, none skipped.
Archived original rank0's4owners×2paths reproduce all12 fingerprints against
generation-bound runner/NPZ hashes; every-field mutations fail the right report.
All32 original fingerprints remain bound by the reviewed v2 receipt; this step
does not claim a new fleet download. Earlier28tests are superseded, not additive.

NEXT: independent NPZ/manifest/7call/journal replay in the existing collector,
then explicit compiler/worker/controller/wrapper mode wiring and composed tests.
Preserve original acquire_programs compile frames so only the already-admitted
seven host coordinates move. Mixed compile-only/numerical scopes must refuse.
Review current integration, persist/mirror and fresh protected preflights before
one bounded run. Last live model execution remains the13:25Z refusal; DB591 was
compile-only. Own8K/efficientL7/L8/TTFT stayopen; no new numerical result.

## 2026-09-08 — protected boundary diagnostic integration

Distinct kernel `ws32_prefill_window_boundary_diagnostic` now connects the existing
acquisition/probe/campaign/wrapper. Compilation retains the original outer call
frames and DB591 graph/schema/allocation admission; no model source changes.
Exactly two WK calls, one B128 candidate and four causally carried B32 controls;
the timeout remains600s. No same-input router graph or competitive/tail case runs.

New `prefill_window_boundary_evidence.py` independently replays all original
NPZ fields,96/33 captures, finite operands, exact four-control reconstruction,
original-v2 fingerprint binding and reproduction/perturbation classification.
It reuses graph/journal/WK replay and owner-bound all-live budgets for seven calls.
The existing collector verifies eight hosts,32 selected owners and remote
generation/CRC/SHA identities. Database correctness/score/latency are NULL.
Diagnostic completion is not numerical admission and never proves a cause.

Actual eight-rank producer/collector plus extracted shell/DB simulation passes
148.71s;45 capture/acquisition tests pass43.38s. Missing-capture injection refuses
with NumPy KeyError (the initial test expected ValueError, corrected in the test
only).75 acquisition/admission/registry tests passed12.30s before the last three
scope tests; counts overlap and must not be added. Four updated registry tests
pass1.89s. Final campaign regression:13PASS616.74s, one already-passed diagnostic
collector case deselected. Covers old eight-rank collector/DB and all five
finalization failures in both modes. No skipped evidence cases.

Independent existing Astra reviewer found no new P0-P2 and conditionally permits
one protected capture after pending tests, clean persistence/mirror, both leases,
fresh storage and normal/root eight-host census. Main audit agrees. Preserve
refusal/perturbation bytes; no automatic retry. Reproduction must precede original
cause attribution, not evidence retention. This entry does not claim execution.

Storage at15:29:58Z: US-CENTRAL2, soft-delete0,56,712 live objects totaling
2,002,638,392,467B; below2.5e12. No checkpoint payload or infrastructure change.

## 15:43:34Z — first capture refused before dispatch; host-container defect

Tag `greenfield_fp8_ws32_prefill_window_boundary_diagnostic_l6_20260908T154130256870977Z`,
pin062256cfee2c850c1efee93e916b7b5becfe8be1, terminalexit1. Allfourrawgraphs
passedadmission; `bind_compiled` then refused originalmesh. Actual runtime
`Ws32PhysicalMesh.device_ids` is tuple-of-tuples; originalJSON is list-of-lists.
Python container inequality rejected identical8x4IDs/order. PublishedJSONmesh
matches the original exactly. No WK/model executable dispatched and no capture
NPZ exists. This is a harnessfailure, not a changed numerical signature.

All96originalworkerobjects and8receiptledgers re-read by exactgeneration/CRC/SHA;
allfourrawadmissions replayed onall8, selected32ownerledger agrees. Receipt
`../artifacts/prefill-boundary-runtime-mesh-refusal-20260908.json`,SHA
dfbdb0b521ced9f1b0a34fcf85f97cd6f4b10566b3f44f05c682a511775f95ed.
Normal/rootzero-workcensus8/8; no automaticretry or recoverednumericseal.

Narrowfix validates strict8x4integers then normalizes list/tuplecontainers only,
never IDs/order/types. Existingvotedbind_runtime nowbindsoriginalsbeforeweight
load; postcompiledefensivebinding remains. ActualWs32PhysicalMesh tuplefixture
and extracted realworkerbinding exercise runtime→JSON boundary, reordered IDs,
bool/float/shape/flat and localmappingrefusals. Original tests had used onlyJSON
and therefore missed this boundary; their priorpasses remainlimited evidence.
Independent Astra findsnoP0-P2; correctedcapture conditional on targetedtests,
cleanpersistence/mirror/bothleases/freshpreflights. No model/HLO/precisionchange.

## DB592 — capture completed, original signature perturbed; offline decision

Corrected pin988818433e002049e21034a114efae9ed2e9aa81 executed tag
`greenfield_fp8_ws32_prefill_window_boundary_diagnostic_l6_20260908T155153878825613Z`.
Terminal15:56:48Z exit0, DB592;218s worker/collector, exactly2WK+5model calls,
0timing samples. Normal/root postcensuses8/8. SUCCESS SHA
`5e8f72b449d5da5742b304f61e8213cad78d8ff6af8325abcb98b5449fdeaafd`,
generation1788883008844143. Diagnostic DB correct/score/latency remain NULL.
176-object archive ledger totals1,001,989,228B; no checkpoint copy.

Every physical owner reports INSTRUMENTATION_PERTURBED_ORIGINAL_SIGNATURE.
Candidate changed six original fields: output/positions/residual/routes/
route_weights/scores. Control changed residual only. No original-cause attribution
or promotion is possible from this capture, even though current routes agree.

Offline all32-owner replay validates original NPZ schemas/hashes, fingerprints,
causal carry, captured checkpoint router weight/bias and physical replicas.
Fresh remote compact terminal/ledger bind local payload hashes; this is NOT
another remote download of all payloads. Receipt:
`../artifacts/prefill-window-captured-input-fp64-v2-20260908.json`, SHA
`b1c9d8fa5e9789dc4a64b05c8d65174ddb11865fb16cbc9219edd7e4d900f471`.
The initial analysis receipt is retained and disclosed; v2 adds separate sigmoid,
DB NULL, cleanup and derived resident-code checks without changing the results.

Router FP64 dot error on own completed BF16 input: max9.57e-8/1.17e-7.
Canonical own-score routes exact both. Full-FP64 sigmoid/bias changes ordered
rows4/25/64 and setrow4 on both paths; no new FP64-route-equality requirement.
Captured sigmoid differs from FP64 of captured logits by~1.2e-6. Completed MLP
inputs differ880 BF16 elements in rows2/74/94/101 across four unique features.
DSArow2 has identical508causal keys, different query4090elements/head18elements;
both captured own-score orders are exact. A hypothetical BF16-query FP64 scorer
fits within2.47e-8/7.21e-8, but does not establish an internal hardware boundary.
This localizes the CAPTURED difference upstream of selection, not the original
candidate's cause and not §21's token/checkpoint math reference.

Measured post-call allocator peaks429,626,368–429,692,928B across32devices,
plus67,827,200B resident executable budget counted separately. Selected-layer
diagnostic only, not full-model physical HBM proof. Stop adding observation
variants. New design/CPU composition:
`PREFILL_COMPLETED_WINDOW_CANDIDATE.md`. Four completed B32 prefixes followed
by one B128 MLP retain broad reuse; its four narrow suffix controls share the
same completed prefix by construction. Own graph/memory, unchanged numerical
checks, full-layer/short decoder and inclusive wall evidence remain necessary.
