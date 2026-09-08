# Layer-major prefill decoder assembly

2026-09-08. §24 default-off implementation; no new TPU/full-model/performance result.
Starting pin `a6a3bba9e23622969fd6ded488f542904b273a91`. Reuses real-layer
admissions DB583/584/587; original failed reference realizations remain failed.

## Execution and state

`glm_tpu/greenfield/runtime/ws32_batched_prefill.py` is a separate program from the
promoted one-row decoder. One invocation embeds B live prompt rows together, visits
each transformer layer once over those rows, and appends its proposed cache writes.
Python enumerates layers during tracing only; there is no host layer dispatch or
scan of the complete decoder per token. Existing per-layer tiled/row-local kernel
loops retain their meanings; this claim does not say all inner work is parallel.

The initial B1..32 interface matches admitted layer kernels. It is a correctness
integration window, NOT a claim that B32 is sufficient for interactive throughput.
The routing occupancy model in `PREFILL_COST_MODEL.md` still motivates larger
routing/linear windows with smaller attention tiles before performance promotion.
No new long-context performance run is authorized by this CPU assembly.

The immutable configuration explicitly requires raw-layout weights, page512 and
the accepted host main-attention rotary table. Exact-convolution aliases and
StrategyND dense overlay configs are refused, not silently ignored. A future
runner must bind a raw view of the same retained weight arrays and separately
bind the promoted decoder's exact/overlay view; it must not create a new pack.
Repair receives the existing completed, checkpoint-authenticated FP32 wk tensors
for each full-indexer owner. The kernel does not rematerialize wk per block.

`Ws32BatchedPrefillState` carries existing decoder state plus the separate repaired
index buffer, immutable request prompt length, and a finished bit. Fresh allocation
creates zero caches/frontier0/context-length1. A nonzero resume must authenticate
the whole state, including page tables and both index buffers; merely asserting
a nonzero frontier does not prove a populated prefix. There is no resume loader
or persistent-state authenticity claim in this module.

Key invariants:

- Append offset is taken only from carried decoder.position. Context length must
  equal offset+1; count is positive, at most B, and cannot exceed prompt length or
  consume reserved decode capacity. Only live count advances the frontier.
- Explicit full-indexer slot map is0,1,2,6,..., not layer%4. KV indexes all layers;
  index/repair buffers index only full producers. Every shared layer uses its own
  KV and the current producer's compact B×topK selected state.
- Hidden update and carried residual remain separate through both norms. Repair
  is computed inside each producer from that producer's own normalized inputs.
- All prompt chunks score unrepaired keys. Repaired keys become active only in
  the final successful commit. Finished states refuse further prefill calls.
- Only the final block runs the one-row final norm/head on its last LIVE row.
  Intermediate chunks return token−1. No per-prompt-row vocabulary projection.
- Health accumulates every layer/live row and the head, then two explicit scalar
  subgroup reductions (feature4, expert8) produce all-owner agreement once per
  block. Head control flow depends on replicated schedule metadata, not possibly
  different owner health. No repeated32-chip model collective is introduced.
- A failed proposal returns unchanged KV/index/repair/frontier/page tables and
  latches false health everywhere. No cache donation is used yet. Input/output
  aliasing and actual compiled peak HBM remain mandatory hardware admission work.
- `finish_ws32_batched_prefill` requires complete healthy state before releasing
  its one-row decoder state and first greedy token. It synchronizes at the final
  serving boundary, not between layers or prompt tokens.

## Smallest CPU checks

The current test uses actual eight-layer weight pytrees and actual layer kernels
on forced32 CPU, with reduced hidden/expert geometry and distinct layer weights.
It independently wires completed layer calls and compares the composition outputs.
This is a state/ownership composition oracle, not independent model arithmetic.
Repair wk values and prefix activations are synthetic; this cannot certify real
checkpoint resume, real-model numerical exactness, or TPU rounding behavior.

Coverage: layer2→3 dense-to-MoE/IndexShare, layer6→7 full-DSA+MoE producer/share,
two chunks offset505/B17 then11-live tail, nonidentity page table, producer
replacement, separate KV/indices, last-live sampling, padded invalid token IDs and
NaN rotary values, premature/repeated finalization, invalid counts/frontiers/page
aliases, repaired-history independence, final-block single-owner health failure,
and an actual raw one-row decoder step using the final repaired state/token.

Independent reviewer identified invalid initial fixture topK16/segment8 against
the production decoder's segment128 constraint. Corrected fixture topK128 and
segment128; no production guard changed. Tests and exact outcome: latest HANDOFF.

## Next protected evidence

Trace the production78-layer input/weight/state schema without allocating weights.
Wire the existing protected short-decoder campaign to this separate graph pair,
using narrow static tails and the existing completed repair weights/host RoPE.
Inspect actual acquired HLO groups, useful rows, cache aliases and per-chip memory
before numerical execution. Resolve major avoidable allocations before proceeding.
The short decoder must earn its OWN §21 raw-token, cutoff-active own-score DSA,
first-divergent-event adjudication and cache/state protection; DB587 cannot stand
in for that proof. Preregister quantitative prefill/TTFT criteria before candidate
performance trials. Efficient four-depth L7 and full L8 remain open.

## Protected acquisition wiring — 2026-09-08

Starting pin61860d94. `GLM_GREENFIELD_WS32_PREFILL_MODE=layer_major_raw_v1`
selects separate short-context ACQUISITION ONLY, with exact decode/host main
RoPE required, B17 default/B1..32 permitted, `_bp1` tag suffix. Numerical runs,
inherited serial adjudication and all long-context runs are refused for this
mode until its actual compiler/allocation profiles and short proof exist.
New serial long-context launches are also refused (§24); recovery of existing
serial evidence is preserved. Serial mode remains the historical default.

Worker binds raw-prefill views BEFORE discarding loaded arrays. Promoted decode
keeps its original view. Both share original objects; no full checkpoint or
weight copy. Completed producer wk arrays are reused. The two compiled programs
use six inputs (IDs, scalar live count, typed state, raw weights, completed wk,
host RoPE), with no old serial donation indices. Acquisition state placeholders
share existing initial cache buffers. Both actual prefill graphs and the five
existing materializer/decoder/observer/cache graphs are preserved before final
structural authorization. No model-prefill graph executes in this acquisition.

`benchmarking/ws32_batched_prefill.py` currently collects exact payload families,
shapes, helper signatures, live layers and non-ADD reducers from the actual graph.
It ALWAYS reports UNREGISTERED and passedFalse: it is NOT the completed linter.
Matching hashes/labels/local groups cannot approve it. Expected refusal writes
the complete per-host runner envelope with HLO_REFUSED before raising, so the
existing upload trap retains HLO, memory/inventory/provenance and logs remotely.
No HLO_ACQUIRED/SUCCESS/DB performance row is manufactured from that refusal.

Independent review's source-derived inventory for later registration:

- Common attention/norm branch78×; full DSA21×; dense3×; MoE75×; one embedding
  and one conditional final one-row head. Do not multiply whole layer0/3 profiles:
  eighteen full-DSA+MoE layers combine independent branches.
- Raw FP8 semantic callsites588, grouped225, structured156, sparse78:1047 total.
  Acquisition must determine actual textual multiplicity/CSE/shared callees.
- Count tuple reduction operand leaves, not guessed psum counts. Exact physical
  feature4/expert8 groups/global IDs and ADD reducers for all model sums.
- Two S32 scalar MIN health reductions require feature→expert→actual commit
  predicate lineage; names alone do not authorize MIN anywhere else.
- Adapt actual FP32 route-sum SSA/fusion proof to every75 MoE layer and actualB;
  handle tuple leaf forwarding without treating an unrelated attention sum as
  route evidence. No BF16 intermediate/original_type correction is allowed there.
- Acquire exact compiler-helper operands/counts, bias lowering, conditional-head
  realization and cache aliases. Do not reuse single-layer capacity1024/B17
  helpers blindly at8192 or narrow tail. Full cache arrays legitimately exceed
  the old layer-only element ceiling; distinguish them from weight expansion by
  shape AND lineage. Intentional21 repair hidden gathers need their exact consumer.

The host execution adapter is tested but not yet enabled in the numerical worker.
It requires two all-host AND decisions per block: structural health, then budget/
logging continuation. Every rank participates even on final block; a peer-only
refusal prevents the next dispatch. Logging errors vote false before raising.
Actual runtime wiring must supply genuine multihost consensus (existing protected
layer worker uses process_allgather of a compact integer), not identity/host-local
bool. These checks are outside layer execution and inside reported request wall.
This accounting ends at token readiness, explicitly NOT actual-delivery TTFT.

Next: review the concrete acquisition cost/preflight and compile actual main/tail
graphs once; use its preserved per-host evidence to implement the strict linter
and allocation admission. Then wire the numerical adapter/sealer and candidate's
own §21 observation/adjudication. No representative layer arithmetic rerun.

Acquisition launch capsule: `configs/greenfield-ws32-batched-acquisition.json`.
Existing retained checkpoint/overlay metadata hashes rechecked locally. Cold
planning estimate15–25min is not a guarantee; the reviewed worker hard ceiling
is45min (serial default remains4h). Upload/cleanup are outside that worker timer.
The wrapper's batched census now supplements its ordinary process/container
check with `fp8_baseline_guard` root device/inode/libtpu/PID checks, even if the
ordinary census fails. Root output is embedded in the archived ordinary census
file, including expected-refusal exit. Caller holds user rsync lease; wrapper
holds workload lease. No model numerical execution and no automatic retry.

### First acquisition outcome (03:25Z)

Pin58c747f9 compiled main B17, then failed in host liveness recursion before tail;
original graphs/logs preserved, all8 normal/rootclean. Local explicit-stack fix
replays actual main; independent health/group inspection agrees. This is NOT
complete acquisition or model numerical evidence. Receipt:
`../artifacts/prefill-batched-acquisition-recursion-failure-20260908.json`.
Next add per-compile partial runner/memory journaling before inspection, then
review acquisition of missing tail/remaining graphs and allocation evidence.
No input/output aliases are declared in main; unique retained raw/overlay/exact
buffers plus outputs/temps/executables must be budgeted without summing duplicate
argument trees. Measure both-prefill-executable residency and later actual
numerical peak; compile-only cannot supply the latter. No automatic rerun.

### Per-compile preservation and next acquisition

`scripts/greenfield/ws32_acquisition_journal.py` is now wired to every compile
in batched acquisition only. Separate append-only/fsynced
`acquisition_journal.rankN.jsonl` records source/checkpoint/topology/host/slots,
load snapshots and local device IDs, then lower/compile start, compiled memory
and device stats, raw HLO hashes, inspection report or exception. Tail's compiled
snapshot occurs while both prefill executables reside. It does not measure a
numerical-prefill peak. Final complete runner validation is unchanged.
The existing exit uploader preserves journals under
`diagnostic_local/<tag>/acquisition_journal.rankN.jsonl`. They require the existing
allow-failure-diagnostics path for recovery/materialization, never silently become
primary runner records, and cannot authorize acquisition or execution.

42CPU tests1.72s pass, including actual7graph writers, injected parser failure,
mode/partial refusal, all-callsite order and actual shell uploader failure.
Independent current-diff review: no material findings. One acquisition is
conditionally approved after clean commit/push/mirror and fresh root/normal fleet
preflights under both leases. Same retained weights/overlay,6GBreserve and45min
worker ceiling. Original launch→mainHLO took~13min; complete graph duration still
unmeasured, planning25–40min only, upload/cleanup separately observed.
Require all8 complete refused envelopes,7graph pairs, per-compile journals and
authenticated cleanup. No model numerical execution, sealing or automatic retry.

### Complete acquired graph set — 04:23Z2026-09-08

Run133fe71f `...acquire_c17_hrope_bp1_20260908T034734466845101Z` completed
all7 graph inspections with expected HLO_REFUSED,44 original archived objects,
all8 complete envelopes/journals and normal/root8/8clean. Receipt and every
graph hash/allocation: `../artifacts/prefill-batched-seven-graph-acquisition-20260908.json`.
All ranks agree. Main B17/temp855351808B and tail B11/temp802251264B each output
113267200B and alias0, atcapacity8192; neither model graph executed. Actual
main/tail route proofs need no tuple forwarding; index computations/roots once,
then parameterize B and require all75 scoped FP32 route reductions. Health
proof must follow subgroupMIN→predicate→commit/rollback/token sentinel, not labels.
Next strict profiles and unique-buffer allocation budget, numerical wiring and
own §21 short proof. This evidence does not require another acquisition launch.

The indexed75-layer route-sum proof is implemented and replayed on both captured
graphs.61focused tests including those original HLOs pass88.91s; independent CPU
review noP0-P2. It deliberately leaves full-profile registration false. Next
health/commit/rollback, exact compiler helpers/payloads and cache/repair/memory
ownership checks; original single-layer guards and model arithmetic stay unchanged.

### Atomic commit boundary proof — 2026-09-08

New `benchmarking/ws32_batched_commit_hlo.py` consumes the existing indexed module
and entry-live closure. MainB17/tailB11 original captured graphs pass offline.
Exactly two live scalar S32 MIN reducers, feature4 then expert8, form the actual
boolean commit decision. Every mutable ENTRY state leaf is the correct slot of
that one conditional. Refusal preserves the original KV/index/repair/selection/
frontier/finished leaves and returns false health; page table and prompt length
are original input identities outside the conditional. The index-cache rollback
uses actual ordered axis0 slices0:6/6:12/12:18/18:21 plus ConcatBitcast, not a
shape-only assertion. Array-layout bitcasts refuse, even at unchanged dimensions.

Final is the acquired safe-end schedule equal to original prompt length AND
original finished=false. The committed frontier/context are that end/end+1.
The final active index is the same SSA value as the proposed repaired output;
head selection uses the same final predicate and its health feeds the owner vote.
The returned token is head output0 only for consensus AND final, otherwise -1.

Scope limits are emitted in the result. A nonfinal value unresolved as identical
to repaired is NOT proof of unrepaired provenance or numerical inequality.
All-layer health contributions, proposed cache writes/ownership, true-head
sampling arithmetic and physical aliasing/memory still need their separate
full-profile checks. The diagnostic remains UNREGISTERED; these offline proofs
cannot enable numerical execution. No additional acquisition/model load.

### Resident memory adapter — 2026-09-08

`PREFILL_MEMORY_ADMISSION.md` records the new per-local-buffer census, conservative
compile/counter budget and exact limits. The host adapter takes one all-live
snapshot before its first dispatch, requires fleet agreement, and records this
cost inside request wall. Both real compiled analyses and any declared additional
resident executables are budgeted; records rederive rather than trust pass flags.
Tests cover peer-only failure, missing counters, shared/nonargument allocations
and release of old cache generations. Actual TPU census/peak, preregistered reserve
and numerical worker/sealer wiring remain open; full HLO remains UNREGISTERED.
Release worker compile-state placeholders before allocating fresh numerical state.

### Exact physical collective inventory — 2026-09-08

`benchmarking/ws32_batched_collective_hlo.py` checks the actual B17/B11 profiles:
787 operations,159 gathers and628 reductions (400 single-input,207 two-input,
21 four-input);898 reduction operand leaves and159 gather output leaves.
Per-layer source schedule is78 attention,21 fullDSA,3 dense and75MoE, plus one
embedding, final-only head and two scalar health votes outside the layer scopes.
Counts include exact feature4/expert8 group multiplicity/global IDs, entry liveness,
gather axes, ordered operand/result pairs and actual scalar ADD/MIN reducers.
The F32 accumulation→mixed F32/BF16 tuple results are retained. Router-bias and
head candidate gathers lower to exact zero-insert sums, not an optional opcode.

The last producer,layer74, has a distinct four-leaf tuple order in BOTH captures:
128/4/2048/576 with F32/F32/BF16/BF16 outputs. Other full producers use
576/128/2048/4 with BF16/F32/BF16/F32. This acquired exception is explicit;
the first offline replay exposed the overgeneralized layer0 template before TPU.
No tuple-reordering or broad shape exception was introduced.

Metadata only assigns per-layer inventory buckets. It does not prove operand
ownership, correct repair consumers, all-layer health or proposed-cache writes.
The existing independent atomic proof binds the ONLY two actual MINs to commit.
Full profile remains UNREGISTERED and numerical worker/sealer disabled. Further
work uses the existing captured originals; no new acquisition or cleared-layer run.

### Compiler helpers and scratch containment — 2026-09-08

`benchmarking/ws32_batched_helper_hlo.py` checks exact B17/B11 helper families,
single-input index annotations, absent/false side effects and live instructions.
The450 U32[256] scratch allocations form225 local searchsorted loops,3perMoE
layer. Each distinct pair appears only in its own eight-leaf initializer at
slots1/2; the initializer has only the expected live while user, exact state
shapes, body/condition bindings and layer scope. This proves containment, not
the numerical correctness of the search algorithm or initialization-before-read.

Every ConcatBitcast has four immediate completed async slices of the SAME SSA
source, exact full disjoint coverage and exclusive start→done→concat uses. Axis0
quarters cover U8 weights, F32 WK, BF16[1024,640] and B17's S32[278528]. Index
cache spans are0:6/6:12/12:18/18:21; KV spans0:20/20:40/40:60/60:78. RoPE
BF16[8192,64] splits axis1 into16-wide quarters, not axis0.

Operand order is NOT assumed logical: current KV operands put60:78 before40:60,
and weight/WK permutations vary between main/tail. The existing decoder's
`_exact_wk_feature_slice_instructions` already checks this compiler mechanism
using slice placement attributes (including historical sealedDB567 evidence).
The new check adapts that mechanism and explicitly does NOT broaden
`PrefillIdentity._reconstruct` or prove arbitrary concat as numerical identity.
Correct weight/cache/WK consumer ownership must still be bound separately.

User indexes cover only watched slice handles or allocation-bearing computations;
there is no full-module scan for each of hundreds of helpers. Pallas calls are
explicitly out of scope, not admitted by the helper result. Full-profile remains
UNREGISTERED and protected numerical execution/sealing remain disabled.

### Pallas interfaces and source schedule — 2026-09-08

`benchmarking/ws32_batched_kernel_hlo.py` binds1047 actual live calls per graph:
588raw FP8,225grouped routed projections,156structured kv-b and78sparse attention.
The per-layer multiset records exact names, paired operand/result shapes/dtypes
and aliases. Producer instructions must supply the recorded shapes. Grouped
compiled calls include a dynamic-grid scalar, so output alias is operand8, not
source-level7; scheduling metadata arrays have rows+255 entries. Side effects
must be absent/false. Short graphs refuse floating tensors at full local expert
table size; this size rule is explicitly not a long-capacity allocation policy.

This shares the existing parsed computation index and live closure. It does NOT
inspect opaque kernel arithmetic, prove the correct same-shaped checkpoint leaf
or route schedule values. The full HLO/code pins, admitted layer evidence and own
short numerical proof are still required. Full profile remains UNREGISTERED.
Next: repaired and unrepaired cache ownership, all-layer health, actual memory
admission and numerical worker/sealer wiring. No unchanged graph acquisition.

### Repair/cache ownership frontier — original B11 graph, 2026-09-08

Independent reviewer traced the concrete layer0 repaired path; main-agent raw
instruction inspection confirms the projection, selected mixed tuple leaf and
writer linkage below. This is a research map, NOT an implemented passing proof.
Use the existing133fe71f acquisition receipt to bind the exact graph bytes; names
below are lookup anchors for that original tail, not a general name allowlist.

`fusion.10982` projected-key leaf1 → `multiply_add_fusion.245` F32[64,128].
Its nonrotary half is `slice.66920` BF16[64,64] (cast folded into slice), forwarded
by `copy-done.1976`; rotary half arrives via `copy.16569`. They feed operands5/6
of `is-finite_reduce_fusion.1065`, whose metadata names layer2. Only ROOT leaf7
is this layer0's repaired BF16[11,128] keys: disjoint -inf-padded halves combine
with maximum and final conversion. `get-tuple-element.88808` selects leaf7.

That result enters `tuple.28446` slot3 → `conditional.940` true branch
`region_9478.9495_spmd` → nested update-returning `scatter.5732` (own old rows,
indices and owner/live-selected update). The conditional's result is passed to
`constant_dynamic-update-slice_fusion.247`; its ROOT `scatter.6000` is actually
opcode dynamic-update-slice with four zero indices, inserting outer slot0.
An instruction called scatter is not necessarily a scatter opcode.

Reviewer traced outer repaired slots0..19 through fusions247..228; slot20 is
inserted inside the accepted commit branch from conditional960. Its original
base is repaired ENTRY leaf11. The other same-shaped chain (fusions344/342/...)
is UNREPAIRED. Verify both main/tail with actual operand/tuple bindings, not scope
metadata, equal dimensions, or the fact that one projection appears somewhere
among all operands of a mixed fusion.

Implementation obligations: selected-leaf fusion forwarding; exact disjoint
half assembly; update-returning scatter reducer/dimensions, old-cache slice and
owner/live index selection; unique outer slots0..20 with preserved remaining
axes and correct original base; bind final tree to accepted repaired output.
The unimplemented mask/index arithmetic and rotary correctness checks are not
established by this research chain. No new capture or unchanged TPU run needed.

### Both index-cache storage stacks checked — 2026-09-08

`benchmarking/ws32_batched_cache_hlo.py` now traces from accepted repaired output
and nonfinal active-index output, not a scope search. Both originals pass exact
21-slot chains: indices20..0, zero other axes, original bases ENTRY11/3. Each
conditional preserves its own old slot or passes that same slot through an exact
scalar replacement scatter. Earlier producer writes are accepted only when their
outer slot is strictly earlier/disjoint; they cannot change the selected slice.

The row-store uses explicit contiguous BF16 SLOT[1,16,64,128]↔FLAT[1024,128]
bitcasts with matching row-major layouts and T(8,128)(2,1). These are checked
locally, not registered as arbitrary array identity. `PrefillIdentity.resolve`
can optionally STOP at an array shape boundary; its default refusal is unchanged.
Unrepaired conditionals return two views, with storage in leaf1, so both branch
resolution and previous-write shape checks preserve the selected tuple path.
Wrong leaf/slot, original base, displaced axis, dropped prior update, wrong old
slice, transposed physical layout and nonreplacement reducer mutations refuse.

This checks storage ownership, not key values, masks, row-address arithmetic or
health. Full profile remains UNREGISTERED. The source model and worker are unchanged.

### Admission sequencing decision — independent review, 2026-09-08

The next bounded2K discriminator does NOT require symbolic proof of every rotary,
mask and index arithmetic instruction. That would duplicate empirical arithmetic
admission and defer the next decisive hardware result. Required remaining HLO
work: own completedWK and normalized gather into each repaired key (leaf-sensitive
dependency, not arithmetic equivalence), all-layer/repair/write health into the
actual commit consensus, and composition of all narrow guards on the SAME pinned
graph. Retain targeted CPU interventions for nonidentity pages, boundary/tails,
invalid metadata and owner-local failure; compare actual short-model outputs
under unchanged§21, exact routes/tokens and dual-cache/state rules.

Before dispatch still require registered reserve, actual fleet-wide memory census,
audited executable/state lifetime, bounded failure publication and cleanup; measure
execution peak rather than promote compilation counters. This decision changes
the verification method, not thresholds, storage invariants or launch authority.

### Repair-input and writer provenance — 2026-09-08

`ws32_batched_repair_hlo.py` binds each actual repaired writer to its own21-slot
projection and completedWK ENTRY2324+slot. Actual multiplication is inspected:
F32[64,128] convolution, bf_oi->bf, default/highest operand precision, explicit
same-index tiled input wrappers. WK Concat uses checked same-source complete
slice scaffolding, not assumed concatenation identity. Source normalized rows
come through the row-repeat gather from the same-layer feature4 gather.

Review found and fixed an important scope gap: the feature gather's label and
shape do not prove its input belongs to the producer. The check now binds that
input to the identical SSA value feeding the actual same-layer raw index-key
FP8 Pallas call, before exact zero padding (B17→24 adds7 rows;B11→16 adds5).
The mutation replaces the input underneath the unchanged gather label and must
refuse. Changing the gather reference or label alone would not cover this bug.

`ws32_hlo_leaf_dependencies.py` memoizes each computation/selected-leaf dependency
summary with symbolic parameter paths and substitutes actual caller operands.
Traversal uses an explicit work stack; a3000-node test checks depth safety.
Mixed output tuples cannot attribute another leaf's projection to this writer.
The actual scatter's live update uses the bound key; that key's projection
terminals must be exactly own leaves0/1, never another producer. Constants,
including -inf, do not have graph operands even if a lexer accepts the spelling
as a name. Unknown control/custom operations refuse; one exact gather annotation
forwards dependency only. Completed copies remain distinct from unfinished data.

This is provenance, not arithmetic equivalence: multiplying a dependent value
by zero still has syntactic ancestry. In particular it is NOT the next health
proof. Health must establish that failure prevents commit: AND unions required
obligations; OR/unknown choices cannot inherit an obligation from just one input.
The exact inactive-row exemption (~live OR health) needs its real mask bound.
Actual memory census/reserve and own§21 numerical evidence remain mandatory.
