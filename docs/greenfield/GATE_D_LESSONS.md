# Gate D Lessons and Non-Repeat Rules

This is the compact operational memory for the GLM-5.2 greenfield short-context gate. The
append-only evidence remains in `docs/RESEARCH_LOG.md`; this file records the reusable rules.

## Current boundary

- DB550 proves all 32 real layer-0 dense down partials match DB548 bitwise: `0 / 196,608`
  mismatches and raw SHA `9d9f65dd...16e35`.
- Therefore checkpoint packing, FP8 decode/scales, SwiGLU, all 16 contractions, scheduled fusion
  geometry and rank-local outputs are closed. Do not reopen or rerun those hypotheses.
- The protected 32-chip replay proves the physical M32 StrategyND row zero exactly matches the
  DB533 software tree: `0 / 6,144` mismatches, common raw SHA `efde8532...b4fc`, hidden-2795 bits
  `47808`. Standalone physical association is closed too.
- The first global-collective/RMS replay reproduced DB548's one-ULP control, but its reconstructed
  residual history was hybrid. It validates replay mechanics, not an accepted boundary.
- DB551 proved that observing dense plus residual perturbs downstream DSA. The residual-only retry
  proved a stronger rule: even a non-returning single-tensor debug consumer can change fusion or
  scheduling. It produced the correct token but DSA diverged from event 1, so it was refused and
  rolled back. Do not repeat legacy boundary captures.
- The refused residual row was nevertheless byte-identical to DB548's already sealed direct
  post-attention residual, SHA `a105fdbd...99f8e`. This equality does not promote the observer or
  turn the source bundle into an accepted oracle; it only proves another hour-scale capture cannot
  supply new bytes.
- The protected direct-residual replay then exactly reproduced rejected DB549: 1,073/6,144
  mismatches and SHA `229dc8ac...812f`. Equal external BF16 bytes do not preserve the compiler
  context of a fused producer/consumer boundary. Direct residual substitution and every standalone
  StrategyND/RMS variant are now closed.
- The first one-graph discriminator kept pre-dense RMS, real final-layout contraction, the physical
  32-chip psum and layer-1 RMS together, yet exactly reproduced DB548's sole hidden-2795 mismatch.
  That closes graph externalization alone. Its preserved HLO then exposed a concrete untested
  boundary: accepted schedules only the scalar RMS reduction, while the control schedules a tuple
  containing the full residual sum. The protected scalar-only challenger was still nonexact and
  exactly reproduced rejected DB549: 1,073/6,144 mismatches, SHA `229dc8ac...812f`. RMS schedule
  and layer-1 recomputation are therefore closed too. The protected attention-before-dense
  ordinal challenger produced the identical rejected row, so collective site/ordinal context is
  closed as an isolated arm. The protected scalar-only pre-dense challenger then reproduced the
  same rejected DB549 row despite exact gate-fusion recomputation, so pre-dense RMS scheduling is
  closed too.
- The next boundary is the composition that none of those isolated arms tested. Accepted HLO
  carries the M32 embedding StrategyND result, its validity predicate and the attention StrategyND
  result directly into both downstream fused norms. The greenfield discriminator supplies exact
  separate M1 BF16 rows and pads them inside the fusion. The residual source is independently
  exact: its SHA `02d045b9...1a3` equals checkpoint embedding row 220 byte for byte. Test the full
  accepted source context once; do not reopen the individual ordinal or RMS theories.
- That composed discriminator's first protected attempt compiled in about 25 seconds and stopped
  before arithmetic on an over-strict HLO guard. Its exact scheduled TPU graph was recovered and is
  now the regression oracle: optimized-HLO SHA `081d1b1f...163f8`. The validator passes that graph
  locally in about 1.3 seconds and rejects mutations of every newly admitted source/layout edge.
  Never repeat a protected compile merely to iterate a validator against this same graph.
- The corrected protected run is nonexact at 1,031/6,144 values, observed SHA
  `3f633b26...28af`. It improves the frozen 1,073-value miss but does not justify integration. This
  activates the closure rule below: manual arms are finished. Automatically diff the live accepted
  and candidate SSA graphs before encoding any further challenger.
- The automatic diff report (SHA `aadf8589...b81b`) proves both RMS arithmetic trees and gate/down
  schedules already match. It also exposes a physical pre-dense predicate-boundary difference:
  accepted carries `S(3)` while the candidate does not. Its first actionable delta is not a new
  formula: accepted consumes native embedding lookup and row-parallel attention-projection
  producers, whereas the candidate rebuilds their final BF16 rows from external M1 inputs. Reuse
  the existing greenfield producers in one graph; do not simulate their outputs with another
  pad/select/collective sequence. Preserve the accepted predicate boundary. The accepted layer-1
  result also consumes all sources in one fusion, while the candidate materializes the carried
  embedding+attention sum in an extra fusion. Preserve producer/consumer fusion ownership whenever
  bitwise exactness depends on scheduled association.
- The native-source successor must be tested in two distinct steps. Forced-32 CPU `eval_shape`
  proves its 13-input global sharding/shape contract in about 1.5 seconds, but CPU cannot lower the
  real non-interpret Pallas kernels. The first protected attempt is therefore compile acquisition
  only: atomically preserve StableHLO/optimized HLO before an intentionally absent exact pin forces
  refusal. Build the exact value-flow/layout validator from that one preserved graph and only then
  authorize a separate numerical execution. Never loosen a TPU contract speculatively or rerun a
  compile just to rediscover an already captured lowering.
- That acquisition is complete. Its exact StableHLO/optimized-HLO SHAs are
  `0884c34e...83d66` / `4b13a9f1...af63`; both pre/failure censuses are 8/8 clean and no
  arithmetic or terminal result exists. The preserved graph is now the sole compiler oracle for
  this candidate. Iterate its validator locally, review it once as an immutable batch, then spend
  one separate seconds-scale run only on the 6,144-value numerical verdict. Do not launch another
  compile-only discovery for the same code/compiler pin.
- That numerical run is complete and exactly reproduces the prior candidate: 1,031/6,144
  mismatches with observed SHA `3f633b26...28af`. Native embedding and attention producers did not
  change the row, so do not repeat or cosmetically rearrange that source substitution. The decisive
  scheduled difference is downstream: accepted keeps the entire layer-1 output calculation on
  M32 inside one fusion, while the diagnostic materializes its carried sum and performs the final
  arithmetic after M1 slicing. Preserve M32 through the layer-1 norm/weight calculation and slice
  only the completed output. This is an observed HLO boundary, not a new numerical theory.
- Make row selection an executor concern for this discriminator. Returning only M1 from the
  compiled graph lets XLA legally pull the slice before arithmetic; returning the completed M32
  value keeps the disputed physical extent live, after which the host comparator may select row
  zero. Prove the global output shape locally, but acquire and pin the real TPU fusion before any
  numerical execution.
- That full-M32 acquisition is complete. Its acquisition optimized graph SHA was
  `e1260889...e3ee`; after reviewed terminal lines shifted source locations, the exact numerical
  code pin produced `5bb78e31...2046`. The only diff is the stack-frame table line numbers; the
  executable graph is byte-identical. Its live root is `u16[32,6144]` and consumes all six accepted source roles in
  one fusion, so the disputed early M1 materialization is absent. The compile acquisition stopped
  before arithmetic on its deliberately empty pin and ended 8/8 clean. Never rerun it merely to
  inspect the same graph. Pin and mutation-test this preserved lowering locally, then run one
  numerical replay.
- A full raw optimized-HLO digest includes non-executable source-location tables. Compile acquisition
  and numerical execution should use the same runner source layout when possible. If reviewed
  terminal-only lines shift those locations, require an exact whole-file diff proving that only the
  stack-frame line table changed, then pin the numerical code's raw digest; never broadly normalize
  or ignore metadata drift without that proof.
- When the compiled extent is itself the hypothesis, preserve the complete output artifact rather
  than hashing and discarding it. Terminal proof must recompute the full array hash, bind every
  local replica and deterministic repeat, prove the separately compared row is exactly row zero of
  that array, and carry the mode/full hash into `SUCCESS`. Otherwise an exact row could be sealed
  without proving that the intended M32 graph actually supplied it.
- The protected full-M32 discriminator is exact: `0 / 6,144`, common row SHA
  `9936ee1e...d3039`, full-array SHA `2a4fe2dd...42dc`, with exact HLO/source/artifact/archive and
  8/8 cleanup proof. This closes the layer-0 numerical hypothesis search. Do not add another
  bounded arithmetic arm or revisit the 1,031/1,073-mismatch variants. The only authorized next
  use is production integration of the same boundary followed by the complete protected 8K Gate-D
  confirmation.
- The exact discriminator's M32 tensor has one live row and 31 NaN sentinel rows. It is diagnostic
  evidence only and must never enter `decode_batch1`. Production must keep one logical row, prove
  no batch-32 dead rows or repeated full-pod hidden reconstruction, and reproduce row zero's
  weighted-output arithmetic with a true-M1 physical kernel/boundary. Do not relabel sentinel rows
  as virtual-rank state or weaken the dead-row linter to admit them.
- The true-M1 Pallas lowering has now been acquired once. Its exact StableHLO/optimized-HLO hashes
  are `14c6c757...7702` / `1fa0957a...5813`; the Pallas call itself accepts and returns only
  one-row tensors. Freeze this lowering and mutate/replay it locally. Do not spend another compile
  acquisition on the same code/compiler pin, and do not confuse its upstream M32 diagnostic
  producers with M32 Pallas I/O. One bounded numerical execution decides the kernel; only an exact
  row authorizes decoder integration.
- That protected execution rejected the Pallas formula at `2,104 / 6,144` mismatches, observed SHA
  `28b7db46...2c20`. Offline replay of the readable FP32-add/RMS, BF16-round and weight-multiply
  contract reproduces the TPU Pallas row exactly. Freeze this result: do not tune or rerun the same
  formula, and do not mistake hidden-2795 agreement for row exactness.
- The exact M32 and rejected non-Pallas M1 HLOs share the same M32 RMS reduction/inverse. Their
  remaining discriminator is the final weighted-output geometry: exact M32 uses
  `T(8,128)(2,1)` with megacore split 0, while M1 uses `T(2,128)(2,1)` with split 1. Test this
  directly with one model-free multi-arm replay that holds inputs/scalar fixed; never reload model
  weights to test a final-fusion layout.
- `with_layout_constraint` currently serializes only major/minor order and explicitly drops tile
  metadata. When the hypothesis is a concrete TPU tile, require it through
  `Format(Layout(major_to_minor=..., tiling=...), sharding)` on the compiled result and pin the
  resulting entry/result layout in HLO. A source-level `Layout(..., tiling=...)` call is not proof.
- Do not freeze an offline scalar unless the offline computation first reproduces the accepted TPU
  control. The readable NumPy inverse reproduces the rejected Pallas row, not the accepted output.
  For a final-layout discriminator, compute one shared scheduled M32 inverse inside the graph and
  feed that exact SSA value to every output-layout arm; otherwise scalar arithmetic and physical
  output geometry change simultaneously and the experiment cannot classify either mechanism.
- A concrete `Format(Layout(...))` that lowers on forced CPU is still only a requested output
  contract, not proof that TPU v4 accepts that tile for the logical shape. On the protected
  `uint16[1,6144]` graph, TPU XLA selected `T(2,128)(2,1)` and refused the requested M32
  `T(8,128)(2,1)` override before emitting executable HLO. Treat this as an evidence-backed shape/
  tile legality rejection: never retry direct M1/M32 result coercion on this compiler pin. If the
  physical-lane hypothesis remains necessary, keep public I/O true M1 and express the alternate
  geometry only inside an explicitly bounded Pallas scratch/kernel.
- A Mosaic/Pallas call in a replicated multi-device `jax.jit` is still subject to automatic
  partitioning and TPU lowering refuses it before HLO persistence. Enclose the whole function in an
  explicit `jax.shard_map` with exact replicated/local specs before `jax.jit`, as the proven
  integrated path does. A single-device Pallas interpreter/JAXPR test does not exercise this
  boundary; every standalone multi-device Pallas probe needs a forced-device regression that
  compiles the mapped graph and proves `sdy.manual_computation` before metal.
- The corrected explicit-map acquisition is now frozen at StableHLO/optimized-HLO SHAs
  `0fda9f03...9c28` / `0107fe68...12b`. It proves the output-only Pallas call has true-M1 public
  I/O, no collective, one shared M32-derived scalar input and one live ENTRY result. Iterate only
  against these preserved graphs; never compile this code pin again merely to inspect its lowering.
- Multi-host replicated `jax.device_put(host_array, replicated_sharding)` performs a host-value
  equality check for replicated inputs; identical NaN sentinels fail because NaN is not equal to
  itself. When exact NaN bits are part of a diagnostic input, place the same contiguous buffer on
  each addressable local device and assemble it with `jax.make_array_from_single_device_arrays`.
  Test equality by raw bits, and do not replace diagnostic sentinels merely to placate the check.
- An internal physical tile cannot recover fusion association that was already destroyed at the
  kernel boundary. The output-only M8 Pallas kernel reproduced the previously rejected Pallas row
  exactly, while its external-row M32/ordinary-M1 controls reproduced the known 1,073-mismatch
  row. Freeze both. A successor is justified only when it moves the exact native producer roles
  proven by the accepted output fusion—not merely the same final BF16 row bits—inside the boundary.
- The source-fused successor has now been compiled exactly once. Its StableHLO/optimized-HLO SHAs
  are `aa087f36...11583` / `c6e6cc38...9558f`; it has true-M1 public I/O and one live Pallas call
  consuming the exact dense, attention, embedding, validity, M32-derived inverse and norm-weight
  roles. Freeze this lowering and iterate only on the recovered files. A compile-acquisition
  failure archive is not a terminal result: say explicitly that `diagnostic_hlo/` exists while
  tensor/comparison/DB/ledger/`SUCCESS` do not. Recover and authenticate those remote compiler
  files rather than rerunning the pod. After an immutable review, one numerical 6,144-value verdict
  decides this final observed boundary; exactness authorizes one full 8K confirmation but does not
  itself close Gate D.
- That numerical verdict is nonexact at 2,135/6,144. Source fusion changes only 106 values relative
  to the rejected output-only Pallas row. Both rows are wrong at 2,078 positions: 2,055 retain the
  same wrong bits and 23 change to different wrong bits. Freeze both arms: native-source ownership
  is not the dominant missing mechanism. A further physical-tile probe is justified only if every
  `8x128` lane holds live feature data and the external semantic result is still one row; never
  spend another run on NaN/dead sibling lanes or a renamed readable formula.

## Evidence ladder

1. Localize the first failing tensor boundary; do not optimize from the final token alone.
2. Capture the smallest real accepted tensor immediately before that boundary.
3. Recompute all candidate arithmetic offline from sealed bits.
4. Replay only the disputed operation on TPU with the real tensor and exact scheduled HLO.
5. Integrate only a bitwise-exact result, then run one complete protected 8K confirmation.

When a replay with byte-identical external tensors changes the result, the externalization boundary
itself is the experiment. Move the minimum producer and consumer into one graph; do not recapture
the same bytes or invent another scalar/association theory. If the integrated result still differs,
compare its exact scheduled HLO against the accepted HLO and test one observed structural delta at
a time; never infer a new arithmetic theory from source code alone.

A full checkpoint/8K run is forbidden while a smaller capture or replay can decide the same
hypothesis. Compile once and replay sealed values. Keep negative results: they permanently remove
branches from the search tree.

## Source-coherence rules

- Shape, position, model id and tensor SHA are not enough to combine evidence from separate runs.
  Every replay input and target must carry one coherent event identity: run/tag, code pin, prompt,
  layer, position, semantic role and capture point.
- If sources intentionally differ, label the result a hybrid control and forbid an accepted-oracle
  conclusion. Reproducing that control validates replay mechanics only.
- Build and validate the source-coherence manifest before launching TPU work. Do not discover after
  execution that the target and an input came from different arithmetic histories.

## Observer-effect rules

- Returning or materializing an internal tensor can change fusion and reduction association.
  DB541 is invalid for exactly this reason.
- “Non-returning” is necessary but not sufficient: adding a debug consumer can itself force
  materialization. Capture only the smallest missing value and never observe an already-closed
  intermediate in the same hook.
- When even that smallest consumer perturbs the oracle, stop capturing. Reuse already sealed equal
  bytes for diagnostic replay, label the provenance limitation explicitly, and require a later
  uninstrumented end-to-end confirmation.
- An oracle observation is accepted only if the observer run itself retains exact raw output and
  exact DSA events. Prefer existing consumed inputs or pre-hooks that return nothing.
- Never treat a numerically plausible observed tensor as an oracle after the observer perturbs the
  production result.

## HLO proof rules

- Operation names, counts, shapes, labels and metadata substrings are insufficient.
- Bind exact SSA value flow from pinned inputs through dtype/layout transforms and arithmetic to
  the live ENTRY result. Reject dead correct decoys, rogue same-shape arithmetic and cross-wiring.
- A caller name and callee name do not prove the callee arithmetic. Bind the complete selected
  fusion body, reducer parameters/opcode and live ROOT operands; mutation-test arithmetic bypasses
  inside both the reduction and post-reduction/rsqrt bodies.
- When StableHLO uses quoted generic operation names, count anchored assignments after stripping
  comments but retaining real quoted op names. A sanitizer that erases every string also erases the
  collective cardinality it was supposed to prove.
- Parse real attributes outside quoted metadata and comments. Pin physical layouts, replica groups,
  reducer parameters/opcode, scheduled backend geometry and synchronous collective form.
- Every accepted HLO form needs adversarial mutation tests and a SHA-pinned preserved-real replay.
- When one exact diagnostic executable is intentionally compiler-pin-specific, a full optimized-HLO
  digest is an acceptable strongest outer gate: it rejects changes to any source edge, fusion body,
  physical layout or backend configuration. Still emit a structural summary of the inputs, live
  collectives, kernels and root so the accepted artifact remains auditable; never substitute the
  digest for the separate protected numerical verdict.
- Write raw StableHLO and optimized HLO atomically before applying the semantic validator. Mark the
  prevalidation record non-valid and publish no result until every proof passes. A validator
  refusal after compilation must preserve the graph needed to correct the proof; never spend
  another protected attempt rediscovering an HLO that was already available in memory.
- When a validator rejects a new compiler form, recover or dump that exact scheduled graph first.
  Add only the exact value-flow-equivalent lowering observed there, then require mutations of each
  newly admitted edge to refuse. Do not weaken the proof from an exception string or source-level
  expectation.
- Value-flow equality does not prove physical execution equality. For an admitted TPU lowering,
  pin the physical layouts at every source parameter, slice/reshape boundary, arithmetic result and
  live root whose reinterpretation can change association or row ownership; test parser-valid
  layout-only mutations as well as opcode/source mutations.
- Causal ordering is part of the graph contract even when a dependency is numerically inert. If a
  sealed-input reconstruction replaces an upstream producer, retain the accepted finite/validity
  guard or equivalent exact SSA dependency before the next collective; otherwise XLA may coalesce
  or reorder independent reductions and invalidate the discriminator.
- In a multi-host compile-acquisition run, process 0 writing HLO is not enough. Synchronize every
  JAX process after the atomic graph/prevalidation writes and before an intentionally failing
  validator, so peer teardown cannot destroy the only compiler evidence.

## Numerical rules

- One BF16 ULP is material: DB548's single hidden-index mismatch changed layer-1 DSA selections.
- Preserve BF16 rounding points and physical reduction association; algebraic equivalence is not
  exact execution equivalence.
- Capture pre-reduction partials before changing contractions. Exact partials plus a wrong combined
  row localize the fault to association; exact combination moves investigation downstream.
- Do not infer a global RMS correction scalar from NumPy arithmetic that does not first reproduce
  the sealed TPU control row bitwise. Scheduled BF16 conversion/fusion can skip representable output
  codes at individual coordinates; a scalar sweep over the wrong arithmetic model is not evidence.

## Review and run discipline

- Work in a coherent bulk, run mutation/static tests, then request one review of an immutable staged
  diff SHA. Apply findings in one correction batch and ask the same reviewer only for correction
  closure. Do not repeatedly audit unchanged code.
- Record every accepted/rejected hypothesis, artifact SHA and exact next action in the handoff and
  research log before compaction.
- Protected runs require clean code pins, exact source hashes, physical topology/host bindings,
  pre/post eight-host zero-work census, CRC-verified complete remote object equality and remote
  `SUCCESS` last. Diagnostics never become performance claims.
- Every capture mode that can commit a provenance row must arm the same exact prefix-aware rollback
  before model execution. A mode is incomplete if a later DSA/sealing refusal can leave an
  unauthenticated provisional row.
- Treat a CLI output path as part of its API: this benchmark derives sibling `hlo/` and `replay/`
  directories from the output parent. A wrapper path refactor must assert all derived artifact paths,
  not only the JSON destination. A completed device call without terminal artifact collection remains
  diagnostic, even when all host records agree numerically.

## Iteration budget and closure rule

- A new Gate-D hypothesis gets one coherent implementation batch, forced-device HLO generation,
  adversarial local tests, terminal-wrapper execution and one immutable review before metal. Do not
  discover CLI/schema/archive defects through sequential TPU launches.
- Timebox the local batch. If the exact observed HLO delta cannot be encoded and locally refused
  within one work block, stop adding arms and build a structural diff tool for the two preserved
  graphs instead.
- After the composed accepted-source discriminator, a nonexact result ends manual one-variable
  guessing. Preserve both scheduled graphs and automatically diff live SSA source, dtype, layout,
  fusion ownership and backend geometry from pinned inputs to the first unequal result.
- Gate D closes only with the complete protected decoder. A bitwise-exact bounded row authorizes
  that confirmation; it does not itself close the gate.
- When a real kernel cannot lower on CPU, keep the local/metal boundary narrow: abstractly trace
  the full graph and test all source/checkpoint/wrapper schemas locally, then spend one protected
  compile solely to acquire the TPU-specific lowering. The acquisition must be structurally
  incapable of arithmetic publication.

## Decision after the model-free replays

- Hardware row zero equals DB533 software. Freeze dense arithmetic and standalone association.
- The global StrategyND/RMS program is mechanically valid and completes in tens of seconds. Both
  hybrid and direct-residual forms are nonexact; the direct form exactly reproduces DB549 and is
  permanently rejected.
- The one-graph contraction→StrategyND→RMS control reproduces DB548 and is closed. Its accepted
  scalar-only RMS schedule/recompute challenger exactly reproduces DB549 and is closed as well.
  The ordinal discriminator reconstructed the sealed attention row through a first
  value-preserving StrategyND reduction, then ran dense second; its protected output was unchanged
  from rejected DB549. Freeze that arm. The pre-dense scalar/gate-fusion discriminator also
  reproduced DB549 and is frozen. The remaining bounded discriminator combines the already proven
  pieces in the exact accepted source context: embedding M32 collective and validity select,
  attention M32 collective, pre-dense scalar/gate fusion, dense psum and layer-1 scalar/recompute.
  It loads only layer-0 weights and remains no-DB/default-off until all 6,144 values are exact.
- The first scalar-only launch reached the intended schedule but failed only in proof. Recovered TPU
  HLO shows the final fusion forms the M1 F32 sum from two independently sliced BF16 row-zero
  sources. This exact lowering is now the sole admitted correction; it does not reopen RMS
  arithmetic or authorize a decoder run until the complete protected row is exact.
- Integrate only a structural boundary that makes the entire 6,144-value accepted row exact. Never
  patch one coordinate or accept a scalar selected from an inexact arithmetic surrogate.

Do not return to hour-scale hypothesis runs or already exact contractions.
