# Bounded full-history frontier — staged, not hardware-admitted

CURRENT2026-09-11: saved seven-graph admission is complete; execution/guarded
entry/materializer capture and independent graph/journal/call replay now have
CPU tests and independent noP0-P2 review. Receipt
`../artifacts/prefill-history-execution-local-20260911.json` binds exact scope.
Nine programs/331calls/2087journalstages;147MiB materializer cap counts112MiB WK,
2MiB exact JSON and33MiB one refusal (supersedes earlier112,198,400B planning).
Next fleet/boundary/materializer collector and nested capped transport/outer
integration, localspace, then ONE protected fullhistory diagnostic. No new
numerical execution, runtimeHBM, original-event reproduction or token11fix.
Earlier "Next" sections are historical where this current checkpoint supersedes.

Purpose: localize the corrected batched path's own8K token failure without
another full78-layer retry. Original evidence and limitations are in
[PREFILL_CANONICAL_8K_FAILURE.md](PREFILL_CANONICAL_8K_FAILURE.md).
No model arithmetic, checkpoint, numerical bound or performance target changes.

## Implemented core

`scripts/greenfield/ws32_history_frontier.py` calls the existing embedding and
actual layer-window kernels for layers0..6. It returns completed update and
carried residual SEPARATELY, each layer's normalized input, ordered routes,
weights, owner health and producer0/1/2/6 selected positions/counts/scores.
It carries seven KV owners and four unrepaired/repaired index owners; shared
layers3/4/5 neither acquire their own index slot nor reuse another layer's KV.
Unhealthy or invalid-capacity proposals retain prior caches and return false.
This stateless core does not enforce monotonic host offsets or complete prompt
coverage; the eventual guarded executor must enforce both and chain health.

Four explicit programs retain physicalB128/B114. The candidate sets
canonical_dense=True only for dense0..2. The control sets it FALSE, matching
the retained original live32 run, not a newly corrected control. No head,
generation, full-state load, CLI, deployment or production default is added.

`ws32_history_frontier_prepare.py` reuses the existing current-source guard,
authenticated metadata/schema and selected-layer name inventory. It returns
only ShapeDtypeStruct inputs; it reads no weight payload and places no arrays.
It retains original78-layer geometry while selecting exactly201 raw leaves:
embedding plus seven COMPLETE layer trees. Existing selected loader will verify
those actual bytes; metadata alone does not establish payload integrity.

Persistent raw operand accounting from all32 manifest owners:

| Operand | Bytes/chip |
|---|---:|
| Selected raw weights | 1,424,692,176 |
| Three cache families, ONE branch | 11,272,192 |
| Four completed FP32 WK matrices | 12,582,912 |

Weights/WK/RoPE are shared across branches; caches are independent. Selected
weight reads total45,590,149,632B over32chips, from existing retained owner files,
NOT a new persistent artifact or checkpoint copy. These are NOT peak HBM or a
complete experiment budget: exact-decode operands/overlay, programs, scratch,
simultaneous captures, host originals and archive still need explicit accounting.

## Required continuation (status 2026-09-10: items 1–3 staged on CPU; 4–5 pending)

1. Build the bounded first-decode observer from the existing
   `ws32_transformer_layer_mapped`, exact-DSA materializer and verified original
   StrategyND overlay for dense0..2. Raw-prefill weights/WK alone are NOT the
   original observer. Do not fabricate head weights or sample with a reduced head.
2. Preserve original8155 IDs and both schedules: candidate63×128+91live tail;
   control254×32+27live tail. Both tails physical114. Interleave one wide block
   and corresponding narrow blocks, retaining separate branch caches and only
   bounded first-difference operands/locations plus compact comparison records.
3. Continue BOTH complete histories even after finding a stream difference.
   Install repaired keys only after full history; observe original token220 at
   position8155/contextlength8156. Require EACH branch's saved step0 event0..3
   positions/counts/scores to reproduce before any original-cause attribution.
   CPU equality or a first128-only match cannot establish that reproduction.
4. Reuse existing BudgetedCalls, compiler journal, actual HLO/memory guards,
   protected fleet/collector/DB/archive/cleanup with a DISTINCT bounded protocol.
   Neither the old two-layer profile nor these functions authorize execution.
   Bind exact original inputs, source pins and all32 physical owners; register
   selected/overlay bytes, simultaneous HBM/host peak and output/archive caps.
5. Inspect saved actual graphs before numerical dispatch; preserve originals
   before refusals. No new generic symbolic/precision proof or blind MoE fix.

## Local evidence and review

Initial actualCPU32 B128+114tail cache/lifecycle/padding/refusal comparison:
1PASS161.16s. Production metadata-only four-program abstract schema test:
1PASS50.95s; explicit payload-open and device_put traps remain enabled.
Neither executes TPU or proves original numerical reproduction.

Independent gpt-6-astra core/preparation review found no implementation P0–P2,
but identified missing coverage of exported observations and False/live32.
The expanded test compares all exported boundaries/producers to instrumented
ORIGINAL eight-layer runtime returns, for both branches: **2PASS318.12s**.
Candidate128+19 and control32+19 use physical128/114, including poison padding,
cache/repair lifecycle and incoming-unhealthy/out-of-capacity rollback. The
reviewer cleared this coverage/source delta for persistence after that pass;
no P0–P2 remains. Reuse-registry tests4PASS1.95s; git diff --check passes.
Observer and guarded integration above remain pending. No TPU was launched.

## Exact first-decode observer — staged, CPU parity proven (2026-09-10)

`scripts/greenfield/ws32_history_observer.py` runs the ORIGINAL decoder's embedding and
`ws32_transformer_layer_mapped` for layers0..6 only, with exact-DSA operands and the StrategyND
dense overlay, on a branch's completed REPAIRED caches, observing token220 at position8155 /
context8156. It returns the six boundary fields per layer plus the four producer0/1/2/6 DSA
observations; no head, sampled token, next state or serving output exists. The witness identity
and incoming health are folded into the program's `accepted` scalar, and every output is
replaced by a sentinel (positions −1, counts0, scores −inf, floats NaN, route ids −1, health
False) unless the whole fleet accepted the step, because the kernels still compute under a false
health flag. `ws32_history_observer_prepare.py` binds the original overlay by manifest/SUCCESS
digests, reads no payload, places no arrays, and budgets FOUR promoted query owners per slot
(promotion names one decoded array four times; XLA materializes four distinct buffers).

Loop parity against `_ws32_decode_impl` is proven on CPU32 (`tests/greenfield/runtime/
test_ws32_history_observer.py`): the observer reproduces the production `observe` step byte-for-byte
across all six boundary fields of layers0..6 and all four DSA observation arrays on an eight-layer
fixture at capacity8192 with exact DSA, plain dense weights and interpreted kernels, using RANDOM
populated caches so the four observations are pairwise distinct (under zero caches producers0/1/6
are byte-identical and a permutation would not be caught). Plain dense is admitted ONLY under
interpretation: the StrategyND kernel accepts production shapes alone, and a compilable plain-dense
observer would observe a model the failed run never ran. The StrategyND default additionally pins
page512/segment512/epsilon1e-5/width640. What remains hardware-only: StrategyND numerics, populated
8155-row reads, actual HBM, and the step0 event reproduction itself.

Independent Fable5.1 review (round1) returned BLOCK on four P2s, all fixed as described: advisory
identity gate → masked outputs; unpinned numerical geometry → pinned; zero-cache test could not
discriminate producers → populated caches; alias budgeting fiction → four owners. It verified loop
parity item by item, `bind_selected_views`, the overlay/exact bindings and the absence of any
token/state/head output.

## Two-branch host driver — staged, CPU exercised (2026-09-10)

`scripts/greenfield/ws32_history_protocol.py` fixes the identity: protocol
`ws32-history-frontier-l06-two-branch-first-decode-v1`, layers0..6, producers0/1/2/6, 201 selected
leaves / 1,424,692,176B per chip, 12 overlay tensors / 65,691,648B per chip, the pinned 8155-token
prompt, witness220/8155/8156, and `plan()` — 319 interleaved steps in 64 groups reproducing BOTH
production schedules exactly (candidate 63×128 + 91 in physical114; control 254×32 + 27 in
physical114), each control block inside its candidate block. The retained step0 originals are keyed
by LAUNCH rank and read from the two committed receipts (`prefill-canonical8k-token-refusal-…` and
`prefill-frozen-live32-diagnostic-…`), themselves bound by SHA-256; all eight ranks' json/npz pins
parse, and the local live32 rank0 NPZ matches its pinned digest.

`scripts/greenfield/ws32_history_worker.py::execute_history` drives both branches under the
budgeted-call contract: independent zero caches (alias-checked), device-side health chaining per
branch, row-aligned EXACT comparison of every layer boundary field per interleave group with the
first differing (layer, field, position) recorded, exact operands retained only for the first
differing group under the 128MiB rank budget, per-owner cache digests, then the observer on each
branch's completed repaired history and byte reproduction against that branch's retained original.
The report is persisted BEFORE every refusal; unhealthy steps are filed under `unhealthy_*` names,
never as a reproduction witness. `tests/greenfield/validation/test_ws32_history_worker.py` (CPU32,
short plan 128+19) asserts: identical branches → every group equal, caches equal, observations
equal, reproduction refused against placeholders yet everything persisted; a candidate-only token
perturbation → first difference at layer0/position40 in group0, operands retained once, layer0 KV
digests differ, control reproduces while the candidate cannot; an unhealthy incoming block → refusal
after retaining its rows, one call only; wrong plan / foreign prompt digest / incomplete originals →
refused before any dispatch.

## Next

CURRENT2026-09-11: allseven saved DB611 graphs now pass distinct original-bound
admission, including complete scoped helper/scratch checks. Selected preflight,
nine-program runtime/memory adapter and append-once call evidence CPU/review ready.
Receipt `../artifacts/prefill-history-admission-runtime-local-20260911.json` binds
14source/test files, separate terminal batches and independent noP0-P2 review.
No payload load, new TPU call, original-event reproduction or8Kfix in this batch.

Next implement materializer capture, execution/entry/collector and enforce the
331call schedule. Derive whole publication/failure/localspace budget with graph,
WK, exact-materializer, history and call originals; proposed512MiB/rank is planning
only. Actual nine-program simultaneousHBM and both original-event reproduction
remain mandatory. Do not reuse compiler-only5.07GB floor or launch another
acquisition. Earlier next-actions below remain implementation history only.

DB611 is now SEALED (2026-09-11 18:44Z), superseding the acquisition-next steps below.
Run greenfield_fp8_ws32_history_frontier_compile_20260911T183647058550669Z atb9e402a9:
allseven actualgraphs,383sworker/collector,zero payload/WK/modelcalls,32owners and8clean.
Complete regionalprefix1,130,365,640B;185controller ledger entries/136worker originals
verified. Reuse these exact originals for bounded admission, not another acquisition.
Actual code363,145,216Btotal/maxscratch243,835,904B are not measuredruntimeHBM.
Physical4/8groups/fixed-four counters pass; helper/profile/runtime/entry/collector remain.
Numerical protocol331calls =8WK+2exact+319blocks+2observers. Fourproducer WKoriginal
budget112,198,400B cannot inherit dense96MiB; reconcile with128MiB historycapsules,
ninegraphs/logs before numerical launch. Preserve both source-event reproduction gates.

Current2026-09-11: independent GPT review has cleared the observer/driver plus
new owner-local capture for persistence. ActualCPU32 parity2PASS291.90s;
production metadata/protocol/reuse10PASS48.05s; final owner/failure60PASS1.72s.
Receipt `../artifacts/prefill-history-driver-local-20260911.json` binds exact
source/tests and discloses the corrected mock-observer fixture failure.
Local capture uses addressable shard data with recorded global column slices;
actual controller owns feature1 only. Signed-zero/replica comparisons are byte
exact. Completed dispatch outputs and two conflicting replica copies survive
refusal, fresh roots/originals are required, and the production prompt SHA is
the default mandatory binding. No actual TPU or original-event reproduction.

1. Compile-only acquisition of the SEVEN new graphs (four frontier programs, observer, exact
   decode/promote) through the existing weight-free compiler campaign as a fourth compile mode,
   with raw StableHLO pins preregistered by offline TPU-target lowering AFTER this batch freezes
   the scripts (raw graphs carry source locations). WK decode/promote reuse their registered pins.
2. Admission for those originals (raw pins, memory caps, forbidden host transport/expansion,
   physical-group locality), then runtime binding (selected 7-layer load, overlay load, exact
   materialization), preflight (retained originals by generation), entry and collector, mirroring
   the dense frontier modules.
3. Controller headroom: `docs/artifacts/db602-db609-local-compile-copy-eviction-review-20260910.json`
   verifies 98 collected rank1..7 compiler-original copies (3,411,885,438B) against their rank
   receipts and re-read cloud generations. Reviewed tool at9227d65d passed the v2 dry-run;
   apply receipt records98files/3,411,885,438B removed, free5,542,117,376B.
   All14 rank0 files retained. Recheck5.07GB floor before launch. Cloud originals
   remain; no checkpoint or TPU deletion. Restore via exact generation URIs in manifest.
4. Before the compiler integration, register per-graph materializer output caps:
   four independent query buffers per producer exceed the generic96MiB cap.
   Abstract byte accounting is not measured HBM; actual allocations still govern admission.

## Seven-graph compiler continuation — 2026-09-11, CPU integration

`ws32_history_compile.py` adapts the existing metadata-only compiler rather than
adding a launcher. Fourth fixed mode: `ws32_history_frontier_compile`, seven jobs,
17 original files per host,30 journal records and12 voted phases. All eight hosts
must verify raw checkpoint AND original StrategyND overlay metadata before runtime.
The seven raw identities must be registered before that preflight; all actual
compiler originals are preserved before any allocation-cap refusal. No executable
is invoked. Original-file collector, physical32owner validation, same-region
publication and NULL correctness/score/latency DB accounting are reused unchanged.

Independent abstract accounting, confirmed by production-shape CPU evaluation:

| Graph | Input bytes/chip | Output bytes/chip | Output cap |
|---|---:|---:|---:|
| candidate/control B128 | 1,449,596,441 | 27,978,625 | 96 MiB |
| candidate/control B114 | 1,449,596,385 | 26,151,359 | 96 MiB |
| exact_decode | 21,156,992 | 106,741,760 | 128 MiB |
| exact_promote | 106,741,760 | 213,696,512 | 256 MiB |
| observer | 1,694,113,757 | 130,536 | 96 MiB |

Counts use declared output shard specs because `eval_shape` strips output sharding;
all four promoted query buffers per producer count separately. Common ceilings
remain2GiB arguments/1GiB temporary/128MiB code/zero aliases. These are conservative
compiler refusal caps, NOT measured runtime HBM. WK decode/promote are unchanged
companions, not part of this seven-graph acquisition.

The fixed256MiB/rank publication cap covers at most2GiB fleet originals. The
wrapper also archives8 collected copies, rank0's local copy and one candidate
optimized HLO plus SQLite snapshot188,583,936B and small metadata. A conservative
normal-prefix allowance is5GiB:18×256MiB+DB =5,020,422,144B before metadata.
A late publication failure republishes the controller tree under diagnostic/,
so allow8GiB failure-aware:28×256MiB+2DB =7,893,360,640B before metadata.
These are planning allowances, not enforced global caps.
Local10×256MiB+DB+1GiBreserve =3,946,680,320B
before metadata; keep5.07GB controller floor. Actual optimized sizes remain
unmeasured. Worker900s/SSH1080s are bounds, not forecasts. No new checkpoint.

CPU integration evidence:87 worker/CLI/fleet/publication/SQLite tests5.41s on the
first run, plus14 adjacent budget-workflow regressions90.43s. Final adapter guards
42PASS1.42s; actual production metadata/IO/allseven registered RAW1PASS110.82s,
with payload/device_put/TPU compile/dispatch traps. Allseven raw fullbytes matched
across fresh programs/caller frames, total10,330,420B/rank. Exact table at the end
of ws32_history_compile.py moves no builder source locations. Initial31PASS139.89s
is superseded by the expanded43 cases, not43 additional distinct cases.
Receipt `../artifacts/prefill-history-compiler-route-local-20260911.json` binds
source/tests, pins, review and limits. Final independent GPT review cleared P0-P2
for persistence/one guarded compile-only acquisition. Actual compilation, structural/runtime
admission and the original-event reproduction are still pending.
