# Gate D Lessons and Non-Repeat Rules

This is the compact operational memory for the GLM-5.2 greenfield short-context gate. The
append-only evidence remains in `docs/RESEARCH_LOG.md`; this file records the reusable rules.

## Current boundary

- The PP16 final-layout y-x-z run closes all upstream layer-0 arithmetic in the live two-chip graph:
  dense update and carried residual are bitwise exact, while only the final normalized row repeats
  DB549 at `1,073/6,144` mismatches and SHA `229dc8ac...812f`. The one-logical-row output-ownership
  acquisition at `d1354ea` changed StableHLO but optimized to the same live boundary: tuple-owned
  carried state plus scalar, then detached rsqrt and normalized-output fusions. It is a zero-sample,
  terminal `REJECTED` acquisition with optimized/StableHLO SHAs `8d4f1cdf...e9036` /
  `0db3e6e...79123f`; never execute or repeat it.
- Offline replay disproves a variance/scalar-tree successor. FP32 inverse `178.59491`
  (`0x4332984c`) plus one final BF16 round reproduces rejected TPU SHA `229dc8ac...812f` exactly;
  an intermediate BF16 round reproduces frozen Pallas SHA `28b7db46...2c20`. Accepted SHA
  `9936ee1e...d3039` admits no global scalar under either tested one-row formula. Freeze scalar
  association, output ownership, layout coercion, true-M1 Pallas, feature-tiled and native-XLA
  bitwise-equivalence routes. Gate D remains open and may resume only through a genuinely new legal
  one-row topology-local architecture candidate: internal rows may use the documented bounded
  comparison, but raw tokens, DSA selected sets/tie order, integrity, HLO/locality, HBM, trace and
  profiler-free wall remain exact. Do not rerun unchanged PP8/WS32 8K paths with sealed DSA drift.
- Treat empty optional evidence directories as absent, not as recursive-copy operands. Archive a
  file tree with `gcloud storage rsync` (without delete) before building the exact remote ledger;
  keep DB publication provisional until terminal `SUCCESS`, and recover a fully validated model
  run from its exact snapshot rather than rerunning it for an outer archive bug.
- A small byte count is not a compact transport if it consumes a separate cross-stage launch.
  DB563's live `s32[1,2053]` IndexShare/control vector is only 8 KiB, yet its eight separate
  ppermutes help make 17 pipeline launches cost 206.353 ms of a 245.640-ms step. Before any 8K
  rerun, test exact bit-packing with the split residual in the smallest transport chain and require
  one launch per stage plus exact unpacked bytes. Never infer the win from payload arithmetic.
- A multi-arm discriminator must preserve the evidence from a rejected compiler hypothesis and
  continue independent arms. Treat only the explicitly adjudicated physical property as a
  recordable rejection (here, absence/misbinding of the 16-KiB query fusion); wrong StableHLO,
  inputs, arithmetic, collectives or provenance remain fatal. This prevents the first experimental
  layout from consuming a protected attempt before the stronger arm executes, without turning a
  graph-validation failure into numerical evidence.
- Treat a new topology as a numerical-state-machine port, not a fresh formula
  implementation. Before its first complete-model run, inventory every exactness correction already
  proven by the accepted challenger and make each one a named HLO assertion. WS32 copied the DSA
  formulas but omitted four protected PP8 mechanisms: the physical-M64 prompt-cache repair, the
  completed BF16 q boundary plus grouped FP32 query-owner association, the normalized-BF16 barrier
  plus completed FP32 `wk` owner with divide/sqrt key norm, and DEFAULT scorer precision. The
  omission survived small-context set checks and cost a full 8K run. Architecture-specific ownership
  may change how a mechanism is realized, so compare bounded candidate layouts on real TPU inputs;
  never mechanically copy a tuple width or silently discard the requirement.
- A DSA selected-set pass is not a ranking discriminator when the live context is no wider than
  `top_k`. The corrected WS32 2K run had 2,034 prompt positions for `top_k=2,048`, so every live
  position was selected even though its position-aligned legacy score diagnostic was materially
  different. The first protected WS32 8K run is the first real ranking test: raw tokens are exact
  20/20, but layer 0 has the same selected membership with mean/max score delta
  `0.00445265/0.0122719`, and layer 1 onward swaps membership. Every future DSA gate must use a
  context strictly wider than `top_k` or directly prove the complete score/ranking boundary; a
  shorter-context set check is only a state/tail/coverage check.
- A new execution architecture must port the numerical state contract, not merely layer formulas
  and weights. WS32 initially carried one already-rounded BF16 residual, although the authoritative
  decoder contract carries `(hidden_update, residual)`, normalizes their unrounded FP32 sum, and
  independently BF16-rounds the carried sum at both boundaries of all 78 layers and final norm.
  That omission survived one-layer tolerance tests, then accumulated until protected 2K token 10
  flipped. Every future architecture needs an adversarial bitwise split-boundary test and a
  complete-HLO count before full-model execution.
- Compile acquisition and numerical execution are separate. One protected acquisition may preserve
  every graph pair; validator development must replay those local graphs and never spend another
  TPU compile on the same lowering. A numerically failed run can still close load/topology/HBM/
  wall mechanisms, but its wall rate is diagnostic and cannot be published as performance.
- When raw tokens first diverge, bind the exact token index to a sealed top-logit record before
  changing the sampler. At WS32 2K index 10, expected token 576 was rank 1 and observed EOS rank 3
  with margin 1.25, proving accumulated body arithmetic rather than a tie, EOS fallback or token
  alignment error.
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
- The authorized all-live successor now maps semantic `[1,6144]` feature order bijectively onto
  `[8,768]`, executes six `[8,128]` programs with every lane live, and reshapes back to one row.
  Treat this feature axis as a representation only, never as eight batch rows. Its protected mode
  is disjoint and HLO-unpinned: acquire the TPU lowering once, require the observed schedule and
  exact live value flow before arithmetic, and reject the route if it does not lower to the
  intended all-live tile. Do not fall back to the frozen dead-lane kernel.
- The protected all-live execution is complete and nonexact at 2,135/6,144. Its output is
  byte-for-byte identical to the rejected six-source dead-lane arm (zero differing values, common
  raw SHA `04adc5dc...950f`, common NPY SHA `e932a86a...c80a4`). Therefore dead lanes and feature
  placement are not the missing mechanism. This closes the complete true-M1 Pallas equivalence
  route. Do not add another scalar, layout, source-fusion, scratch or feature-tiling arm.
- A new Gate-D successor must change the execution architecture while preserving the already
  proven checkpoint/contraction/attention facts. It must use the mature PP8 production loader and
  one-row/local-group contracts; it may not import legacy execution, admit M32 sentinel rows, or
  reconstruct the hidden state over the full pod.
- Permit exactly one final native-XLA mechanism discriminator before that production return: a
  single semantic row may be represented bijectively as `[8,768]` so XLA, rather than a custom
  Pallas formula, owns the full add/RMS/weight fusion. Compile once, prove the preserved HLO
  offline, and execute at most one 6,144-value verdict. This is a full-pod one-layer oracle and
  never a production architecture. If nonexact, stop bitwise internal-arithmetic iteration and use
  the specification's bounded internal tensor comparison plus exact raw-token Gate-D contract on
  the real PP8 decoder; do not invent a successor discriminator.
- The one acquisition proves this representation is not retained: `[8,768]` exists in StableHLO
  but is absent from optimized HLO, whose live result is the already rejected M1
  `T(2,128)(2,1)`/megacore-split-1 fusion. Reject on schedule and do not spend a numerical run.
  Both custom-Pallas and native-XLA bitwise equivalence branches are closed. Resume only the real
  PP8 short decoder and evaluate internal tensors under the documented bounded-error level while
  keeping DSA ordering, raw tokens, state/cache integrity and physical topology contracts exact.

## Architecture-pivot rules

- When every bounded implementation of one physical arithmetic boundary converges on a frozen
  nonexact result, close the mechanism family. Do not convert review findings into an endless
  supply of new numerical hypotheses. Proof hardening protects a chosen experiment; it does not
  justify rerunning the same rejected architecture.
- Gate D is an end-to-end exact-token/DSA/integrity/HLO/HBM/wall gate. A bitwise internal tensor is
  useful localization evidence, not the gate itself. Conversely, bounded internal error is allowed
  only when the complete token and DSA contracts remain exact.
- Before writing a successor, inventory existing branches, packed artifacts, kernels, topology
  tools and negative evidence. Record the source pins and exact adaptation boundary in the reuse
  registry so later compactions cannot restart from a blank design.
- Capacity arithmetic must precede a full-model load, but it must not be mislabeled as Gate B or
  HBM proof. Separate source-byte coverage, final-owner byte intervals/checksums, direct loading,
  KV/DSA state, compiler overlays, temporary buffers and measured per-chip peak HBM.
- Iterate TPU-specific proof offline from one preserved compiler graph. The next WS32 experiment is
  a bounded real layer whose final-owner shards, packed lineage, subgroup reductions and live root
  are locally fail-closed before metal. Do not use the complete 8K decoder to discover a packer,
  HLO-validator, wrapper or archive defect.
- Keep checkpoint and TPU discriminators separate. The reviewed real WS32 layer pack completed and
  sealed in 70 seconds without initializing JAX/TPU, so packing, direct-load, byte provenance and
  archive failures can never consume a full decoder run. Only its immutable manifest may enter the
  later one-layer HLO/oracle experiment.
- For WS32, the invariant is one logical row and a persistent local hidden shard. Feature-4 and
  expert-8 reductions are allowed; a repeated 32-chip group, physical `[32,6144]` activation, or
  hidden all-gather is immediate rejection.
- Never infer JAX process ownership from the TPU-VM worker suffix. On this sealed fleet the launch
  workers `0..7` map to JAX processes `[1,6,0,7,2,4,3,5]`; authenticate the complete captured
  launch-host/JAX-process/device permutation before direct loading. Likewise,
  `NamedSharding.addressable_devices` is an unordered set: feed
  `make_array_from_single_device_arrays` only in the insertion order returned by
  `addressable_devices_indices_map(global_shape)`, and reconstruct values in a content-level test.

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
- A compiler refusal does not require another metal run when the failure trap has preserved the
  exact graph. Recover the approved `diagnostic_hlo/` objects, compare local and remote CRC32C, pin
  both complete digests, and mutation-test the observed lowering offline. Rediscovering identical
  HLO wastes the protected iteration budget.
- Replacing dead batch lanes with a feature tiling is safe only when the graph proves a bijection:
  exact row-zero sources, shape-preserving `[1,6144] -> [8,768]`, six live `[8,128]` programs,
  ordered source roles and a layout-only return to semantic M1. Shape equality alone does not prove
  feature order or physical layout.
- A scheduled collective result dtype is not necessarily its accumulator dtype. TPU XLA can fuse a
  following BF16 conversion into an all-reduce result while retaining an F32 operand and exact F32
  reducer. Prove and publish operand, reducer and scheduled-result dtypes separately; never derive
  the reduction arithmetic from the result prefix alone.
- TPU v4 may expose `local_hardware_id=None`. Bind runtime devices to the sealed fleet-observed
  local-device order and require any non-null runtime id to agree; do not invent an id or reject a
  valid runtime solely because this optional field is absent.
- A sealed manifest and its payload need not share a directory. Treat the manifest root and exact
  payload subdirectory as separate authenticated inputs, and test the same layout the protected
  mount exposes before launch.

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
- `bash -n` validates only the outer wrapper, not shell programs stored in strings for remote
  execution. Extract every generated worker command and run `bash -n` on the evaluated value; also
  compile every embedded Python heredoc locally. A stray command separator after a heredoc is a
  deterministic launch failure even when the wrapper itself parses.
- A distributed checkpoint has two independent terminal prefixes: payload/manifest and run
  evidence. Recompute exact slot ownership from the pinned physical topology, rederive byte/tensor
  cardinalities, verify remote size plus CRC32C and exact object equality, and publish `SUCCESS`
  last in each prefix. Never include `SUCCESS` in a recursive nonterminal upload.
- An oracle may keep payload under `oracle/` and terminal `SUCCESS` at the artifact root. Carry
  both paths explicitly, hard-pin the terminal bytes and bind its manifest identity; never infer
  the commit-marker location from the payload directory.
- Full-model allocator policy is part of HBM provenance. Compute persistent state plus cache before
  launch, pin the chosen XLA memory fraction in every worker environment, and seal that exact value;
  never depend on an ambient shell variable or the default 75-percent reservation.
- A GCSFuse checkpoint mount may be deliberately read-only. Finalize a manifest in a writable run
  directory against exact read-only payload symlinks, remove those links before failure archiving,
  and upload the manifest through a create-only storage API. Never infer writability from a mounted
  path merely because new remote objects become visible there.
- Worker evidence uploaded before orchestration is already part of the remote append-only prefix.
  A later create-only bulk uploader must skip those objects, then authenticate one exact combined
  local/remote object set and CRC ledger. Re-uploading downloaded worker records is a deterministic
  precondition failure, not additional protection.
- A manifest is not a protected checkpoint commit marker by itself. Direct load requires an exact,
  self-hashed `SUCCESS` bound to manifest bytes, code, mesh, topology, inventory and cardinalities.
  Immediately before publishing it, revalidate all payload sizes, CRC32Cs and immutable generations
  against both the preflight ledger and manifest; a prior check does not close the replacement race.
- Give each remote-vacancy proof a distinct local evidence filename. Prefixes commonly share the
  same final tag component, so deriving both filenames with `basename` silently overwrites one.
- A worker's `passed` field, tensor SHA or error summary is not terminal numerical evidence. Preserve
  the exact raw BF16 bits (or an equivalently content-addressed tensor), reconstruct every physical
  shard in the orchestrator, recompute hashes/errors against the independently sealed oracle and
  prove exact all-device ownership before publication.
- Remote `SUCCESS` and its DB row are one terminal unit. On failed publication, delete the DB row
  only after authenticated generation-matched `SUCCESS` deletion or proven absence; otherwise
  retain the linkage and fail loudly for repair.
- Compile acquisition and numerical execution are separate trust states. Persist every complete
  StableHLO/optimized-HLO pair first; acquisition permits only the explicit vacant-pin failures,
  while numerical execution requires all acquired digests and an independent structural replay
  before the first call to each model program. A self-reported HLO result is not terminal proof.
- `jax.named_scope` provenance is representation-specific. The current Shardy StableHLO printer
  may emit no `loc` records while optimized HLO retains exact `op_name` paths. Bind the required
  scope to a parsed live instruction and an exact `/`-delimited path component; never accept a raw
  substring, dead instruction, metadata/source-file decoy or prefix collision.
- A multi-graph compile-only acquisition must atomically preserve every independent graph before
  returning its aggregate structural verdict. Do not stop after the first graph and pay another
  full checkpoint load merely to discover the next compiler form. This does not authorize device
  execution: numerical mode still validates each exact pinned graph before its first call.
- Index immutable compiler structures once. Rescanning a 173,829-instruction module for each of
  1,226 reducers made one offline replay take 163.6 seconds; a computation index produced the same
  report in 25.9 seconds. Proof code is part of iteration latency and must not hide avoidable
  quadratic work.
- A prompt scan lowers through `while`. Cross-computation liveness must include both its exact
  condition and body roots; otherwise live prompt collectives are invisible or a dead decoy can be
  misclassified. Unit-test this with a dead collective inside the body before TPU acquisition.
- HLO call-graph proof must cover the real compiler vocabulary, not only fusion and collectives.
  The complete decoder uses live `to_apply` callees on reduce, sort, reduce-window and scatter;
  replay the preserved full graph and fail closed on every unobserved callee attribute.
- TPU tuple all-reduce can share one ordinary two-parameter scalar reducer across every operand.
  Prove equal arity, compatible geometry and that exact reducer; do not invent a tuple-root fixture
  that the protected compiler never emits.
- Short-context DSA owners may hold fewer rows than the global top-k (for example 256 local rows at
  2K versus top-k 2,048). Pad each owner score row to top-k before `lax.top_k`, preserve `-1/-inf`
  tails, then merge by global position. A short prompt does not invalidate the fixed top-k graph.
- Branching a multi-gigabyte cache to separate observer and performance programs can exceed HBM.
  Keep recurrent state device-resident and donated; use the proof-only observer for the sealed DSA
  prefix, then continue with the normal executable for profiler-free timing. Pin both HLOs.
- Terminal fleet validation must compare replicated token, DSA, state and cache evidence across all
  eight hosts, replay every saved HLO, parse all eight XPlanes, and calculate distributed wall from
  the per-iteration maximum—not accept eight unrelated local `passed` booleans.
- Before a complete-model TPU compile, run a forced-32 abstract trace built from the real inventory,
  exact 2,310-leaf final-layout plan and all decoder/observer/prefill/cache-probe entry points. This
  catches Python, shape, dtype and sharding composition errors in roughly two minutes without model
  payload I/O. It is a preflight, not TPU HLO, numerical or performance evidence.
- Do not make every launch rank rehash every checkpoint owner. Each rank hashes only its four exact
  final-owner files; the terminal fleet validator proves disjoint slots and complete 0--31 coverage.
  This preserves local-read integrity without multiplying a 786-GB checkpoint scan eightfold.
- A terminal validator must recompute token, DSA and cache verdicts from sealed raw arrays. A JSON
  `passed` field is not a numerical artifact, even when eight hosts repeat it.
- Architecture ports must start from the authoritative numerical state machine, not from the
  nearest single-tensor API. WS32 initially collapsed the required `(update, carried_residual)`
  pair and normalized an already-rounded BF16 sum twice per layer. That known PP8 failure mode
  survived unit arithmetic tests but flipped a protected token. Assert every required rounding
  boundary and its count in the complete HLO before the first numerical run.
- Evidence fanout needs a disk budget just as model execution needs an HBM budget. Sixty-four
  per-rank HLO paths can contain only eight unique byte streams; downloading every path separately
  consumed about 2.49 GB, then eight unique XPlanes consumed another 2.23 GB and filled the
  orchestrator disk after a valid run. Verify each remote generation/CRC/SHA, then hard-link equal
  local content. Preflight the bytes that truly need downloading plus a fixed sealing reserve.
- A worker success marker must mean both execution and all required evidence uploads succeeded.
  Best-effort uploads belong only in the failure trap; suppressing upload errors on the success
  path can turn eight green markers into an incomplete remote fleet.
- The provenance DB is shared state, not a worktree-local scratch file. Every protected wrapper
  must use the canonical `/home/gianl/glm-tpu/bench/results.db`, open it read-only before launch,
  require the exact schema and absence of the run tag, and never silently initialize an empty
  ignored `results.db` in a feature worktree.
- A completed TPU result may be sealed without rerunning model work only when recovery is itself a
  protected workflow: preserve and content-address the original failure diagnostics, bind every
  preexisting remote object's immutable generation/CRC/SHA, rerun the unchanged numerical sealer,
  obtain a fresh eight-host census while holding the fleet lease, record both execution and
  recovery code hashes, then publish DB, exact remote ledger and `SUCCESS` last. Recovery must not
  delete inconvenient evidence or reinterpret a failed numerical result.
- Compile acquisition and numerical execution have different exact evidence schemas. Acquisition
  emits runner JSON/log plus HLO and deliberately has no tensor NPZ or XPlane; numerical adds both.
  Derive expected host-record suffixes once and reuse that helper in remote-set validation and
  download loops. Pin the full cardinalities in tests (`80` acquire, `96` numerical for WS32), and
  exercise a complete acquisition materialization with no NPZ so a test cannot encode the bug.
- A protected compile is recoverable evidence. If all workers and HLO uploads completed but later
  sealing failed, generation-pin and replay those graphs under a recovery code hash; never rerun
  the compiler merely to satisfy an orchestrator schema defect. Capture materializer/sealer stderr
  in archived logs, publish the recovery census, and make partially uploaded acquisition
  diagnostics authenticated and retryable without deleting source HLOs.
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

## Upstream-obsolescence guardrail

- Before any multi-day implementation or protected long-context promotion, snapshot the current
  model revision and the exact relevant upstream commits/PR states, then mechanically compare the
  live source with the historical hypothesis. Reproduce and minimize a current failure before
  reusing an old patch. Preserve parity tests, benchmarks and failure analysis; port the idea, not
  the stale diff. Generate summaries from canonical raw artifacts instead of copying headline
  values between documents.
- Make the first discriminator the cheapest boundary that can falsify the hypothesis. For DSA this
  includes `2047/2048/2049` and page-boundary cases: at `<=2048`, selected-set equality cannot prove
  sparse ranking or consumption. Promote `2K -> 4K -> 8K -> 32K -> 128K -> 256K`, recording the
  exact last pass and first failure rather than spending a full-pod run to discover a small-shape
  contract error.
- The 2026-08-27 source check confirms that `tpu-inference` PR #2324 is still open, correctness-first
  and explicitly below performance expectations. Its loader, FP8-v4 and cross-shard correctness
  ideas are research inputs; its Ray/Python execution, PP limitations and full-token all-gathers do
  not satisfy this engine's topology/locality contract. The two August research handoffs therefore
  do not override `goal.md` or `docs/glm-tpu-revolution.md`: legacy `tpu-inference` remains an oracle
  only, and the isolated native-JAX topology-first engine remains the production architecture.

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
- The output-only and six-source dead-lane M8 Pallas arms are frozen negative evidence at
  2,104/6,144 and 2,135/6,144 mismatches. Their near-identity rules out more scalar/layout tuning.
  The all-live feature-tiled M8 graph is also frozen: its protected output is byte-identical to the
  six-source arm and remains nonexact at 2,135/6,144. No true-M1 Pallas discriminator remains
  active. Select an architecture-level successor from existing production code and evidence.
- The WS32 successor confirms that the old latency was an implementation boundary, not TPU-v4 ICI:
  replacing whole-matrix dequantization with owner-selected raw-FP8 tile-local Pallas changes the
  exact real-layer p50 from `516.5821635` / `1160.885836` ms to `1.256975` / `2.303685` ms while
  retaining `0.03125` maximum error, 15/15 live inputs and only feature-4/expert-8 reductions. Carry
  this body into the complete decoder; do not optimize the rejected readable body or reopen PP8
  scalar arithmetic. A fast exact layer selects integration but never substitutes for the complete
  protected decoder required to close Gate D.

Do not return to hour-scale hypothesis runs or already exact contractions.
