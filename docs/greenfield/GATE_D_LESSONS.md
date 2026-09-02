# Gate D Lessons and Non-Repeat Rules

This is the compact operational memory for the GLM-5.2 greenfield short-context gate. The
append-only evidence remains in `docs/RESEARCH_LOG.md`; this file records the reusable rules.

## TPU optimization may erase identity-barrier names

- A named `lax.optimization_barrier` is a valid source/StableHLO seal, but its opcode and exact
  `op_name` suffix are not a durable optimized-TPU contract. The sealed PP16 acquisition retained
  all four intended values and exact roots while erasing every exact barrier marker; only one
  enclosing scope survived on a folded reshape.
- Do not fall back to surviving scope substrings, root shapes alone or an ad hoc ancestry walker.
  Preserve the exact StableHLO contract and authenticate the complete optimized executable with
  the existing debug-provenance-stripping canonicalizer. Pin SHA, byte count, canonicalizer
  version and stripped-reference count explicitly. The outer wrapper must hash the actual raw HLO
  files, recanonicalize the actual optimized HLO, byte-compare that result with the archived
  canonical file, and only then cross-check runner/contract records; self-reported hashes are not
  an independent boundary.
- Treat a validator refusal after compile as diagnostic only. Archive the HLO before failure,
  authenticate the exact remote set, prove cleanup, and do not infer numerical or Gate-D progress.

## Current boundary

- The corrected 15-root acquisition remains valid HLO/HBM/locality evidence, but its separately
  reviewed DB518 numerical successor already executed exactly once at commit `dafe2ee`, tag
  `greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z`. It makes the
  position-113 key/normalization and all 8,155 layer-0 prompt-cache rows bitwise exact, yet every
  ordinary model output remains byte-identical to the sealed rejection. The first retained
  mismatch is one layer-1 BF16 normalized value per owner at hidden index 2795; q-a, query,
  head-weight and event-1 DSA drift follow. DB518 is therefore closed as a causal Gate-D fix. Do
  not run its numerical wrapper again and do not launch an unchanged complete 8K decoder.
- The accepted DB485 M32, rejected DB518 feature-2 and sealed WS32 feature-4 HLOs prove matching
  logical BF16 state association but distinct RMS-reduction/weighted-output geometry. They do not
  reveal the missing unrounded physical state or correction materialization. The accepted hidden
  callback is permanently rejected because it changed the executable class and reproduced DB551.
  Gate D remains open. Admit only a genuinely new legal one-row, topology-local mechanism, first
  falsified with the smallest sealed offline token/DSA authority available; exact full-8K tokens
  and every DSA selected set/tie order remain the closure authority.

## Historical and frozen boundaries

- The compile-only predecessor at pushed pin
  `79a1590812bd805187737cc85d2068d8486519a6`, tag
  `greenfield_pp16_feature2_prefill_acquire_20260829T023702949220285Z`, is superseded historical
  evidence. Main arithmetic was not invoked. StableHLO pins 64 N6144 attention producers, 64
  N6144 dense-down convolutions and eight
  half reducers; optimized HLO additionally proves all eight causal producer-frontier/owner-half
  lineages. One-row/half-sharded roots, two adjacent partitions and forbidden-transport contracts
  pass. Load/peak use is 1,203,933,696 / 1,249,780,224 bytes per device with 31,747,741,184 bytes
  in the post-compile largest free block; all three materializers have zero collectives. The exact
  20-object `US-CENTRAL2` archive and pre/sync/post 8/8 checks are independently byte-verified.
  This predecessor is compile-only evidence, not numerical, Gate-D or performance proof. Its
  missing normalized-hidden, q-a-state, DSA-query and head-weight comparisons were added and the
  changed 15-root graph was acquired at `a2ea1e9` above. Do not repeat its obsolete next step.
- The half-width producer variant executed at code pin `363a52b7c8a4201ffbbb899352b7159c26a95b80`
  is numerically rejected and frozen. Its abstract graph SHA `ab5be45a...cb2d` is shared with the
  later full-width-rounded-then-slice variant and is not a sufficient freeze identity by itself.
  The rejected variant's one protected invocation passed load/HLO/locality/HBM but its first
  carried boundary missed `968/6,144` BF16
  values (`3f6c86ed...53d2` versus `35a601b7...044c`), followed by `1,852/2,048` event-1 position
  and `2,048/2,048` score mismatches. Exact valid count, contract bit and one current key do not
  rescue it or prove the other 8,155 keys/query. Do not repeat this half-width variant or extend it
  to a full 8K run.
  The distinct terminal rejection archive is
  `greenfield_pp16_feature2_numerical_recovery_20260829T004819684644332Z`; its source remains
  diagnostic-only and its 30.424-second wall is not performance evidence.
- Real DB550 leaves prove the existing half reducer/add semantics are exact on accepted inputs:
  forced-two-CPU y→x→z replay
  matches dense/carried at `0/6,144`, SHAs `efde8532...b4fc` / `35a601b7...044c`, with one
  `[4,1,3072]` ppermute and no full hidden. The final carried capture cannot isolate which of the
  two half-width producers caused drift. The only admitted successor changes both unproved
  producer geometries: run each already-proven DB539 attention O-projection and DB550 dense-down
  full-width virtual-rank contraction, round BF16, immediately slice each leaf into ordered halves,
  then use the existing LP2 half reducer and persistent feature state. This successor is locally
  implemented behind one default-false library flag; the protected wrapper requires explicit value
  `1` and refuses the frozen value `0`. JAXpr independently pins its 64 full-width attention calls.
  Variant-aware StableHLO/optimized-HLO contracts require exact live attention/dense geometry,
  row-zero/unit-stride/ordered-complementary half slices, four attention plus four dense half
  reducers, one-row/half-sharded roots and no full-hidden stack. Reducer proof must be causal, not
  independent counts: trace each permute input through nested fusion bindings to only its matching
  producer frontier and exact two-stack/y-tree/owner-select path. Do not discard half identity at
  that frontier: trace each lower/upper slice one-to-one into its half stack, require the exact
  compiler LP2 owner chain (`u32 partition-id -> AND 1 -> s32 convert -> EQ s32 0`) by direct
  operands, not ancestor counts, and pin `owns_half0 ? upper : lower` for the peer payload.
  Hostile hybrid, dead-decoy, wrong-row, duplicate/swap-half, cross-scope reducer, stack-leaf and
  select-branch, shifted-zero predicate and metadata-decoy direction/mask swaps are mandatory;
  executable attributes must never be inferred from provenance text. The corrected adjacent suite
  passes 116 tests. The compile-only acquisition above closes this graph's real-state HLO/HBM
  prerequisite with the exact boundary comparisons and hostile tests present. Numerical execution
  remains unauthorized until the separate strict numerical wrapper is pinned to the `a2ea1e9`
  acquisition, adversarially reviewed, committed, pushed and byte-mirrored. Scalar,
  normalization, consumer-fusion, output-ownership, half-width-contraction fitting and unchanged
  decoder retries remain frozen. Gate D stays open.
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
- Every tested offline CPU consumer-fusion formulation is rejected before TPU use. The accepted hidden row passed
  through the production one-row N82 qkv-a helper reproduces accepted q-a bitwise, while the
  rejected one-round row misses `999/2,048`. Direct double-round, FP32-input/folded-weight and
  post-scale forms all miss `617--1,081` q-a values. Searching `+/-4,096` FP32 inverse ULPs finds
  no exact global scalar, no exact repeating group through 128 lanes, and no exact contiguous block
  of 32 or more features. Sparse exact subsets at `4/256` and `71/512` modulo groups are not a
  physical law; freeze per-lane/tile fitting as oracle-fitting policy. CPU does not prove TPU query
  association or exclude every physical fusion, but compiler-layout hope alone cannot authorize an
  acquisition for these formulas. Dense and carried rows remain bitwise exact; bounded comparison
  applies only to normalized/q-a/internal rows. Admit only a genuinely new plan-level physical
  state/ownership mechanism whose authority is exact full-8K tokens and every DSA set/tie order.
- A zero WS32 `score_diagnostic` does not describe a mismatched event. The validator appends score
  arrays only after count and selected-set equality; in the rejected 8K run its 2,048 aligned
  values belong to exact event 0, while event 1 is excluded. Never infer that common positions in
  a seven-swap event have exact scores from that field.
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
  This historical resume instruction is superseded by the sealed cutoff-active 8K event-1
  membership failure: do not rerun an unchanged PP8 decoder.

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
- An offline JAX test is offline only when its process explicitly sets `JAX_PLATFORMS=cpu` before
  import. Never rely on the test name, `XLA_FLAGS`, or the caller's environment: on a TPU worker,
  an omitted platform pin can open `/dev/accel*` during collection or lowering. Such a process is
  unauthorized outside the fleet lease and its output is inadmissible; terminate the exact PID,
  verify no accelerator handles remain, and rerun with the CPU platform forced.

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

- Cloud metadata field names are versioned interfaces. Terminal verification must accept exactly
  one unambiguous CRC value from `crc32c_hash` and/or `crc32c`, refuse absence or disagreement, and
  always bind it to a generation-qualified content read. A receipt-schema failure after arithmetic
  must be recovered without changing the original terminal or rerunning the model.
- Before another graph variant, reconstruct compact state against its complete accepted oracle.
  The PP16 full/half-width variants were identical across 11 outputs, while CPU reconstruction
  localized the upstream cache to 71 one-coordinate drifts and the first to position 113 / hidden
  35. Observe that first boundary and require every existing output unchanged; do not use a later
  normalized-row mismatch as the next broad search surface.
- A first-time observer lowering has no pre-existing canonical TPU HLO identity. Keep its initial
  metal action compile-only, archive the actual StableHLO/optimized/canonical graph, and inspect and
  pin that graph before permitting a call. Source names and root geometry are acquisition evidence,
  not a substitute for the later unchanged-output numerical guard.
- An optional observer must not alter the prerequisites of its default-off path. Hash and record
  observer-only oracles only when the observer is enabled, and regression-test malformed feature
  flags before any Git, GCS or TPU action. Compile every production Python heredoc rather than
  assuming outer `bash -n` covers embedded programs.
- Raw JAXpr text can encode backend-only `AbstractMesh` rendering even when the complete semantic
  graph is identical. Never weaken this to a set of unexplained hashes. Admit only exact known
  runtime fragments and counts, pin each complete raw identity, canonicalize only those fragments,
  and pin the complete canonical JAXpr. Mixed CPU/TPU meshes, unknown device kinds, missing
  fragments and every other textual difference must refuse.
- A complete canonical-HLO pin is the admission boundary, but observer semantics should also be
  explained and machine-checked. Follow ENTRY roots through while GTEs, body/condition references,
  same-index scan handoffs, current/prior selects and the predicate/count recurrence. Preserve
  first-scan initialization explicitly; a four-scan observer can otherwise expose allocated decoys
  while still having the right root shapes and names.
- Ancestry is not branch identity. For fused observer selects, resolve callee parameters to caller
  operands and bind predicate, true-current and false-prior separately; seeing all three somewhere
  in a value's ancestry admits `current=prior` substitutions. Likewise, require observation count
  as the exact two-operand `prior + convert(predicate)` recurrence, not merely an `add` whose graph
  happens to contain both values.
- Dependency sets also discard executable order and logical polarity. Preserve and pin ordered
  callee operations, operand order and layout attributes such as complementary padding ranges;
  require an exact predicate or explicitly allowed shape-only wrapper. A graph depending on
  `not(predicate)` has the same ancestry but the opposite observation semantics. Apply the same
  rule to the predicate converted into an observation counter.
- A pinned SSA name is not a pinned value: an attacker can retain the name and shape while changing
  its defining opcode to copy the prior branch. Bind the executable definition and complete
  transitive causal ancestry of every expected caller source, including referenced computations,
  as part of the ordered branch certificate.
- A pre-run oracle hash does not protect a post-run classifier from local source replacement.
  Re-hash every classification input immediately before it is consumed, retain the original
  source-identity record, and parse from that exact in-memory byte buffer. Hashing a path and then
  reopening it leaves a smaller but real time-of-check/time-of-use gap.
- `bash -n` does not compile quoted Python heredocs. Extract and compile every production heredoc
  in a permanent test; a wrapper with six embedded programs is only syntactically covered when all
  six are checked.
- Compare a newly observed boundary against both the accepted oracle and the sealed prior
  candidate. At p113, the active PP16 key differed from accepted by one BF16 value but was exactly
  the old greenfield row; this selects the already-proven DB518 correction instead of another
  arithmetic search.
- A correct post-scan repair cannot fix a value already consumed inside a causal scan. Prompt keys
  for the current layer must have the accepted physical association before that layer's attention
  scores them; repairing only the next layer's cache after the chunk is too late.
- A chunk-local exact reconstruction must consume the authenticated runtime position tensor, not a
  Python-derived equivalent range. Pad both values and positions through the same live-row index so
  the compiled RoPE/key association remains bound to the scan input even when the sealed values are
  currently equal.
- After first-time compilation, replace acquisition hints with both immutable whole-graph identity
  and a causal consumption certificate before numerical execution. For chunked prompt keys this
  means following the exact initializer field through the scan body, live-row slice, cache-write
  conditional, scorer consumer and carried roots; shape, source names and root positions alone do
  not prove that the corrected key is used. Also bind persistent state independently: require the
  first scan's exact zero cache and every later cache initializer from the same predecessor while,
  or a locally correct chunk can silently discard all keys written by earlier chunks.
- The 2026-08-29 layer-1 two-operand callback repeated DB551 exactly: 557,434 position and 573,438
  score mismatches, first at event 1, despite exact token and event 0. “Non-returning” is not
  “non-consuming”; the callback creates a distinct value consumer. The two callback runs share the
  same seven ordered executable, executable-including-data and host-transfer fingerprint triples,
  while accepted DB485 is disjoint in all three dimensions. This is consistent with
  materialization/fusion/scheduling change, but no sealed HLO localizes the responsible optimized
  operation and the source pins contain other diagnostic support deltas. Do not promote the
  consistency statement into a causal compiler claim. The
  observer pin, registration, launch mode and accepted sealer are permanently tombstoned. Never
  repeat this layer-1 fused-boundary callback capture class.
- Certificate `docs/artifacts/callback-executable-class-certificate.json`, SHA
  `6e58bca9...6e48`, byte-authenticates DB485's manifest/SUCCESS/full code pin/run/item identity and
  all three logs, every anchored code marker, run date/PID/tag windows, token-bucket order and
  509/507/45-object archive inventories. No inventory has a compiler-artifact-named object; payloads
  were not exhaustively scanned for embedded compiler IR. The original no-TPU optimized-HLO diff
  therefore has no sealed recovery path without one new compile. A CPU JAXpr/StableHLO is not a
  substitute. No documented TPU value tracepoint currently promises unchanged executable identity;
  stop hidden-value capture work.
- If separately reviewed, admit at most one accepted-tree compile-only dump with no decode and a
  hard DB485 seven-triple equality gate covering executable, executable-including-data and
  host-transfer identities. A mismatch seals an inadmissible diagnostic and stops; a match permits
  mechanism inspection only. It cannot close Gate D or promise recovery of the hidden FP32 value.
  DB518 already differs in downstream q-a/query/head/event-1 DSA, so do not weaken its boundary
  policy to justify a known-risk full 8K run. A numerical successor still requires a genuinely new
  bounded plan-level physical state/ownership mechanism.
- “Seven scheduled modules” is not a bucket certificate. Bind every artifact independently to its
  logical bucket using an already-proven physical signature. For DB485 that means exactly 156
  source-backed `bf16[M,6144]` row-parallel reductions, 32 partitions, an exact M32--M2048 map and
  strictly increasing compiler module ids. The superseded binary-share assumption expected one
  owner plus seven nonowners; direct evidence requires seven host-local replicas on every exact
  worker suffix 0--7, a common raw size/SHA identity, and worker 0 as canonical only after all eight
  audits agree. Compare every decompressed payload with its recorded raw size and SHA before raw
  deletion.
- A compile-only diagnostic still needs production-grade lifecycle rules. The driver must own a
  bounded killable session; Ray ownership remains armed until zero-work census passes; post-launch
  cloud operations are bounded; source mounts are explicitly read-only. A remote prefix is not
  owned until vacancy succeeds. Upload generation-zero nonterminal objects, replay every object and
  the exact name set, then publish `SUCCESS` alone and last. Never add diagnostics after terminal
  publication begins.
- A Git archive is not automatically the installed runtime identity. vLLM's generated
  `_version.py` is untracked at the accepted pin; omitting it reconstructs a `dev` runtime and can
  guarantee failure only after the expensive model initialization. Inject and byte-pin the exact
  generated version file, import that reconstructed path on all eight hosts before model work and
  seal both the semantic version and file hash. Do not weaken the DB485 version check.
- Under `set -euo pipefail`, never probe an HLO root with `find | wc` before classifying the root.
  The superseded binary-share design treated absent/empty roots as nonowners; direct evidence now
  requires a nonempty root on all eight hosts. A missing, empty, non-directory or unreadable root is
  a failure. Exercise the actual compactor command for every branch before TPU work.
- Shell `-e` does not detect a dangling symlink. Every append-only local/fleet vacancy check must
  require both `! -e` and `! -L`; otherwise SCP can follow a stale transport link outside the
  tag-owned namespace while cleanup removes only the link. Likewise, evidence walkers must lstat
  the complete tree and reject all symlinks and special files instead of following or ignoring them.
- `SUCCESS` is a proof contract, not merely a terminal filename. Before generation-zero upload,
  reconstruct its exact canonical bytes from the sealed manifest, exact remote prefix and immutable
  ledger; require every provenance field, `db_run_id=None`, and all numerical/performance/Gate-D
  claims false. Missing, extra, reordered or altered fields and a prefix folder-marker object fail.
- A multihost JAX debug dump is not necessarily a binary-share one-owner artifact. The 2026-08-30
  accepted DB485 acquisition emitted seven qualifying scheduled HLOs on every one of eight hosts;
  the old one-owner/seven-nonowner validator correctly refused publication but discarded the remote
  compact payloads during authenticated cleanup. Receipts prove replicated dump materialization,
  not independent compilation or byte equality. Future admission must collect all eight fixed-schema
  audits first, compare exact bucket-sorted raw size/SHA identities independent of compiler filenames,
  replay the canonical worker-0 payload and attempt bounded preservation of all divergent compressed
  sets. The normal matching path may clean up after eight audits agree and worker 0's copied payload
  validates. Once failure preservation begins, delete remote dumps only after eight copied payloads
  validate; otherwise retain every exact tag dump wherever present, require eight retention receipts
  and report preservation incomplete rather than claiming cleanup.
  Never burn another full compile merely because a post-compile evidence assumption was wrong.
- The corrected DB485 retry sealed seven identical scheduled HLOs across all eight hosts without a
  request. Its M32 module proves the first dense output is BF16 and layer-1 RMS consumes
  `FP32(BF16 dense + BF16(BF16 attention + BF16 residual))`. DB518's FP32 carried operand descends
  through an exact copy chain from a BF16-corrected embedding-owner sum with literal BF16-zero
  fallback selections. This establishes a logical association match, not physical BF16
  materialization across the DB518 FP32 tuple/copy boundary. The missing-unrounded-state cause
  remains unresolved; do not reject or select it from HLO correction metadata alone.
- Another inspected accepted/current distinction is RMS-reduction plus weighted-output
  geometry: accepted squares `32x6144` locally and uses `T(8,128)(2,1)`, window `2x48`, split 0;
  DB518 squares `1x3072`, scalar-reduces over feature-2, weights using `T(2,128)(2,1)`, window
  `1x12`, split 1 and gathers afterward. WS32 squares `1x1536` and scalar-reduces over feature-4.
  This agrees with the historical
  full-M32-versus-one-row localization. Direct M1 tile coercion, true-M1 Pallas, source-fused,
  scalar, output-ownership and layout routes are already rejected. Moving the gather before the
  weight is another output-ownership/layout formulation, not a new architecture. Require a
  genuinely untested legal one-row topology-local mechanism before any TPU work.
- Do not treat PP16 alone as an independent causal comparator. The sealed WS32 8K HLO
  `8f964f9e...0ce6` binds an update-plus-embedding BF16 round at the pre-dense boundary and exact
  BF16 StrategyND row0-tree operands plus the normalized BF16 round and BF16 weight multiplication
  at the next-layer-RMS boundary on `1x1536` feature shards. It does not prove the complete dense
  value path between those boundaries. Its
  feature-square reduction and normalized-feature gather use exact four-device groups, window
  `1x6` and split 1. This independently places WS32 in the same one-row weighted-output physical
  family rather than supporting a hidden FP32-state theory. A useful certificate must reject
  swapped sources, wrong owner indices, broken zero/select/copy/inverse/weight/dense paths,
  missing rounds, wrong groups/layouts and duplicate target calls;
  a matching substring or operation count is not a value-flow proof.
- The accepted next-layer qkv-a consumer is now part of the fail-closed causal certificate, not an
  open fusion hypothesis. `fusion.9360` passes the weighted RMS result as explicit
  `bf16[32,6144]` to SHA-pinned `fused_computation.16511`; a SHA-pinned BF16 bitcast is its only
  input transformation before the BF16-input/BF16-weight, FP32-accumulating N82 convolution. No
  pre-round FP32 RMS operand is exposed across that HLO boundary. This proves semantic dataflow,
  not physical BF16 materialization. Direct/non-rooted RMS-to-qkv fusion is duplicate-closed and
  must not receive another compile or TPU run.

## LP2 K-half N82 reduction passes only the offline admission gate

The full-K N82 path used by DB518 is not the same accumulation mechanism as contracting one
hidden-feature half per LP2 rank and reducing a compact FP32 N82 partial. The diagnostic retains
DB518's gathered/replicated row and full packed weights, keeps one token row, selects `K=3072` plus
24 exact scale blocks on each rank and performs exactly one local `{0,1}` FP32
`[32,1,82]` reduction before the existing BF16 q-a boundary. Its forced-two-CPU StableHLO contains
no gather, host callback, nonlocal group or dead token rows.

Real SHA-authenticated p8155 replay proves the full-K control and FP32-partial arm both map accepted
hidden to accepted q-a at `0/2,048`; BF16-rounded partials miss `656/2,048` and are rejected. On
DB518's current hidden row, CPU full-K and FP32-partial are identical, miss accepted q-a at 46
values and differ from the captured TPU q-a at one value. Therefore CPU does not decide the only
surviving arm's TPU physical association and supplies no Gate-D evidence.

This candidate has a narrow coherent-state exception to the general mixed-cache prohibition.
DB518 builds the complete layer-1 prompt index cache from normalized hidden plus `wk` before
layer-1 qkv-a; the candidate changes only that later projection. Current key and head weights depend
on the unchanged normalized hidden, not qkv-a, but are computed later inside DSA. The saved cache is
post-event and must have its current slot authenticated/removed before replay.

The coherent forced-two-CPU replay closes the arm before metal. Its DB518 captured-q control cannot
reproduce the TPU event (1,723 ordered-position mismatches), and the intended split-K association
still misses accepted event 1 at 1,892 positions/all scores/18 set members. This also confirms that
CPU event arithmetic is not a TPU oracle. Capsule
`pp16-feature2-qkv-khalf-event1-cpu-rejection.json` SHA `b7f5e432...f9ec` is classified
`CPU_EVENT1_ADMISSION_REJECTED;NO_TPU_SUCCESSOR`. Do not build or launch a TPU discriminator for
this arm. Its DB518 history is candidate-coherent and source-authenticated, but accepted layer-1
cache exactness is unknown; the accepted-row arm is sensitivity only. Return to the upstream
normalized-state cause.

## Admit mechanism families offline before compilation

- A new mechanism name is not a new mechanism. Compute its canonical association/consumer/
  reduction/representation/transport fingerprint and map it to the sealed
  formula/layout, Pallas, ownership/fusion, callback, downstream split-K, mixed-authority,
  tolerance or unchanged-plan families before spending JAX/TPU time.
- Structural legality and causal coherence are separate gates. Require exactly one logical row,
  local groups of at most four, no full-pod hidden reconstruction and no host effect; then require
  one SHA-bound candidate history. Reuse the stdlib observability auditor to validate the exact
  typed arrays, slices and raw SHAs for RMS input, normalized row, cache history, query, head
  weights, current key and event-1 scorer state; bind common code/plan/executable/coherence identity
  plus layout/owner evidence. Watchpoint names over an opaque blob are a permanent hostile test.
- `expose` or `resolve` must act at the first missing watchpoint. A downstream DSA kernel, renamed
  layout, or borrowed cache does not move the frontier. An offline pass still sets
  `tpu_successor_authorized=false` and needs a separate reviewed compilation/execution batch.
- `NO_ADMISSIBLE_MECHANISM` is scoped to the catalogued candidates. Never convert it into a general
  impossibility claim. Source plus sealed accepted-HLO audit closes only direct unrounded-shadow
  substitution: it removes the accepted BF16 recurrent-state round. An unused no-dependency value
  is non-causal, and origin-only use is the existing transient fused sum. Rounded/widened,
  compensated or auxiliary device-consumer forms remain unadjudicated separate fingerprints; do
  not collapse them into the rejected direct form or advance them without causal HLO and the full
  coherent-state contract. Artifact `plan-local-persistent-fp32-shadow-source-rejection.json`
  machine-binds this limited supersession. No form is currently authorized for JAX/TPU.
- Snapshot current upstream refs before designing a mechanism. As of the sealed 2026-08-30 audit,
  experimental GLM5 branches add downstream DSA/indexer/cache/attention work but no new layer-1
  fused-add/RMS physical ownership mechanism.
- Open every input parent and report-output parent component through dirfds with
  `O_DIRECTORY|O_NOFOLLOW`. Final-component checks alone do not stop an intermediate symlink from
  redirecting append-only evidence.

## Normalize shadow proposals before compilation

- Rounded-then-widened primary state is the accepted BF16 recurrence expressed with another dtype
  conversion; it does not create a new retained value. Historical split-RMS/barrier evidence already
  rejects that physical family. Do not compile a renamed version.
- A compensation that restores discarded FP32 low bits is direct unrounded substitution and changes
  model semantics. A compensation cancelled without a later device consumer is non-causal. If the
  cancellation stays live only through a device consumer, it becomes the distinct unresolved
  `compensated_auxiliary_dependency` declaration within the broader auxiliary-device search
  umbrella. Its equivalence to an explicit tuple auxiliary remains unproved.
- A BF16 primary recurrence plus an FP32 auxiliary device dependency is structurally distinct but
  not evidence-complete. Before compile-only review it needs a source-semantics binding, a causal
  StableHLO contract, an offline candidate-coherent eight-watchpoint capsule and separate admission.
  The compile-only acquisition must then preserve optimized-TPU-HLO value flow before any numerical-
  run review. StableHLO tuple membership alone is insufficient: output-ownership work changed
  StableHLO and optimized back to the same rejected live boundary.
- Run `scripts/greenfield/adjudicate_shadow_variants.py` first. Its append-only stdlib report
  authenticates evidence, one-row/local/no-host rules and declared normal forms. The declarations
  are not source/AST bindings. The current result leaves two separate unresolved variants under one
  search umbrella and authorizes no JAX, compilation or TPU work.

## Audit capsule constructability before implementing a survivor

- A numerically useful old artifact is not automatically a new candidate capsule. DB518 directly
  preserves normalized/query/head/event-1 state and a post-event cache, but not p8155
  `rms_input_fp32` or `current_key`; all present bytes remain bound to DB518's code and optimized
  HLO. Relabeling them for an auxiliary-shadow implementation would be mixed authority.
- The rejected callback's FP32 sum is not missing data that can be promoted later. Its consuming
  observer changed the executable class and reproduced the frozen DSA mismatch, so those bytes are
  perturbation evidence only.
- Admission ordering must be representable in the schema. The current observability v1 accepts
  optimized-HLO or generic executable identities, rejects explicit `stablehlo`, requires a non-null
  executable-identity SHA and has no explicit source/AST authority; admission v1 binds the same
  tuple. Do not overload a generic fingerprint to hide that omission. V1 cannot faithfully enforce
  source/AST + StableHLO + offline capsule *before* compile-only acquisition.
- Preserve v1 for its historical claims. Build an append-only precompile v2 with explicit source
  and StableHLO authority, hostile identity tests, narrow offline claims and a hard false TPU flag.
  Run `scripts/greenfield/audit_capsule_constructability.py` first; its current report is
  `docs/artifacts/gate-d-capsule-constructability.json` SHA
  `77fa743228fb322a1ca0ac1190f73f27a59e12cd56b4a6637e5980c38bb1a11f`.

## Precompile admission must authenticate the candidate, not merely describe it

- Preserve executable-bound admission v1. A precompile path is a separate schema because its
  authority is committed source/blob/AST plus plan and StableHLO, not a not-yet-existing executable
  identity.
- Bind the v2 implementation itself, inherit the exact v1 closure set, and match the sealed
  survivor catalogue and normal forms. Otherwise a new schema can silently erase old negative
  evidence or admit a renamed family.
- A StableHLO operation name or tuple membership is not causality. Require an SSA chain from the
  auxiliary source to a live returned auxiliary, require the accepted BF16 primary result to remain
  independently live, and require its complete backward slice to remain identical to the accepted
  primary. The source may legitimately be shared by the unchanged primary and the auxiliary.
- Do not parse StableHLO with regular expressions. Comments, string attributes, constants and
  opaque calls can forge textual edges. Use a pinned MLIR parser, bind the exact accepted primary
  backward slice, bind the candidate-specific auxiliary slice, and require every candidate
  collective component to fit one sealed physically local plan group.
- A Git commit string is not source authority when `PATH`, `GIT_DIR`, object directories or replace
  objects remain ambient. Pin the Git executable bytes, scrub `GIT_*`, disable replacement, retain
  one repository dirfd and bind each tree object id to the exact parsed blob bytes.
- Source-semantics and StableHLO certificates are authenticated review inputs, not self-proving
  facts. Their exact source/plan/artifact identities must be reviewed before they can support an
  offline admission.
- An abstract AST with unresolved helper names is a declarative DSL, not executable source
  authority. Bind every concrete operator and the real integration caller before admission.
- A hash-checked copied parser tree is still mutable during import, and sealed parser files alone
  are insufficient if CPython or an early stdlib import remains same-UID writable. Bind and
  provision the entire runtime tree root-owned/read-only; launch `-I -S` from a root-owned CWD with
  no inherited import environment. Copy every parser blob into a Linux memfd, apply
  write/grow/shrink/further-seal prohibitions, independently reread the sealed bytes, load sources
  from exact fds and require native mappings to match the sealed device/inode identities. Parse
  maps with `split(maxsplit=5)` and validate all remaining mapped files by identity, ownership and
  content rather than by a `jaxlib` pathname substring. Keep ephemeral inode comparisons internal
  and normalize their public receipt, or identical authority runs become nondeterministic.
- Root-owned is not enough if provisioning preserves source modes/xattrs or executes a mutable repo
  pathname under sudo. Include canonical safe modes in the complete tree hash; strip and reject
  setuid/setgid/sticky bits and every xattr; refuse root validator execution. Treat provisioning as
  trusted manual administration outside parser authority: install reviewed bytes once as a fixed
  root-owned non-writable tool and invoke it only through exact `/usr/bin/python3 -I -S` or an
  equivalent absolute-path isolated shebang. That pre-start command is the security boundary; an
  in-process executable/flags check only detects accidental misinvocation after startup. Fail
  admission unless the tool matches bound source, and
  nofollow-validate exact root:root staging parents. Publish with atomic `RENAME_NOREPLACE`; a
  preceding `exists()` check cannot protect dangling symlinks or a privileged race. Include the runtime root's own
  mode in the tree hash. Perform the final Python-module/maps scan after parsing, dialect registration,
  verification and every causal/collective walk so lazy dependencies cannot escape the receipt.
- A coherent capsule must carry all eight watchpoints under one code/source/plan/StableHLO/coherence
  authority, with role-specific arrays, raw selected-slice SHAs, layout and physical-owner evidence.
  The two BF16 RMS operands are separate evidence; independently derive their FP32 sum and require
  exact bytes at `layer1.rms_input_fp32` rather than trusting a captured or self-declared sum.
  A coherence id supplied by the capsule author proves nothing; require a pinned replayable producer
  or candidate-execution receipt. Relabeled DB518 arrays and callback-observer bytes remain invalid.
- Plan authority must reproduce the complete sealed physical group order, topology hash, exact
  group size and one shared local owner group for every watchpoint. Layout is not a free label:
  PP16 and PP8 accept only `lp2.local` and `lp4.local`. Recompute the plan hash from those fields and
  reject a renamed layout even when the attacker also rebinds every outer SHA.
- Bound total NPZ decompression before reading members, reject non-stored/deflated compression,
  traverse every parent with no-follow directory descriptors and make append-only publication safe
  against concurrent replacement.
- At the historical source-only milestone, v2 bound concrete tuple-candidate source/callsite and exact PP16 plan authority
  plus causal StableHLO. Its null capsule bindings are deliberate tombstone state after coherent
  event replay rejected the tuple mechanism, not a missing successor task. The compensated variant now binds
  distinct concrete committed source at `e16d74f`; at that milestone it still lacked plan, causal
  StableHLO, pinned producer and coherent capsule authority. It
  always leaves compilation/TPU authorization false. This is an honest sequencing result, not
  Gate-D progress through execution.
- Never validate historical source authority against the moving worktree. Read the exact Git blobs
  from the authority's pinned commit; otherwise an unrelated later source addition can make a
  correct sealed manifest appear stale or tempt a reviewer to weaken historical evidence.
- A scalar inversion must target the actual retained program edge. Here qkv-a consumes
  `BF16(BF16(sum*inverse)*weight)`, so an interval for only `BF16(sum*inverse)` omits a sealed
  per-element operand and cannot adjudicate the boundary. Keep the useful BF16-input derivation;
  do not revive tombstoned global-scalar or variance-tree searches from an incomplete inversion.
- A physical plan can be shared while its watchpoint meaning cannot. The compensated candidate
  legitimately reuses the sealed PP16 group order, but its plan document must additionally bind
  candidate id, mechanism fingerprint, verified source authority/code pin and the exact mapping
  `restored_input_rms_fp32 -> layer1.rms_input_fp32.value`. Include the independently enforced
  binary32 sum derivation and operand order in the content-derived plan hash. A copied tuple plan
  or fully rehashed source/frontier/derivation mutation must fail closed. This removes only plan
  authority; it does not prove cancellation, StableHLO causality or coherent numerical output.
- A lowering environment may be shared while producer identity cannot. Give each candidate its own
  fixed installed/source path, module symbol, named auxiliary result and dependency-manifest digest;
  pass candidate id into the immutable parser rather than inferring it from a generic tuple label.
  Preserve the historical tuple contract exactly. Also validate receipt, SUCCESS and certificate
  schema versions as strict integers: JSON floats and bools compare equal to integers in Python.
- Preflight the literal parent CLI interpreter before spending a reviewed parser start. Package
  import requirements such as `enum.StrEnum` are launcher requirements even when the eventual
  child parser has its own sealed runtime. Record a pre-argparse failure as a burned attempt; fix
  only the launcher under a fresh tag after persistence and review.
- A synthetic causal-slice fixture is not authority for the committed lowering. Reconstruct its
  canonical hash to diagnose drift, then align the fixture to exact real operations—including
  optimization barriers—and require the immutable parser to reproduce the new expectation. Emit
  compact expected/observed fields instead of one aggregate `validator result drifted` message so
  the first failed bounded run identifies the precise stale invariant.
- A canonical, self-hashed local terminal is not status authority after its trusted publisher exits
  when the run directory remains same-UID mutable: both terminal and receipt can be replaced with a
  self-consistent opposite result. Cross the status and remote terminal identity directly from the
  immutable publisher process after final upload/replay (or retain an authenticated descriptor),
  then treat reopened local names as supplementary consistency checks only. Attack-test local
  substitution after publisher return; it may cause fail-closed denial but must never flip the
  remote accepted/rejected result.
- XLA optimized-HLO text embeds the calling Python file path and call-site line numbers in its
  FileNames/FileLocations tables. A numerical driver that is a different installed file from the
  compile-only acquisition driver can never pass a raw byte comparison against the accepted HLO,
  even when StableHLO and the graph body are byte-identical (v1 numerical tag burned 2026-09-01).
  Either acquire the HLO from the exact file that will execute it, or bind an exact reviewed
  source-location bridge (path + call-site lines, single-occurrence substitutions, derived hash) and
  compare against the derived bytes. Pin the call-site line constants with a test against the driver
  source, because formatters shift them.
- TPU on-device `jnp.cos`/`jnp.sin` of large rotary angles (position × inverse frequency up to
  ~8155 rad at position 8155, theta 8e6) deviate from the true values by up to ~1e-2, while a host
  f32 table (vLLM/legacy) is accurate to the f32 argument-rounding level (~7e-5). The DSA indexer
  key/query rotary evaluated on device therefore diverges from the accepted keys by ~2e-3 per
  element while projection (f32-accurate) and key LayerNorm (bit-exact) are innocent. Evaluate
  rotary tables on the host with the accepted f32 formula and gather rows by position on device;
  never trust on-device transcendental range reduction for exactness.
- A rejected bounded discriminator is still decisive evidence when its outputs are archived: the
  V2 rejection was fully adjudicated on CPU from `outputs.npz` alone (implied cos/sin from
  pre-/post-rotation pairs) without another TPU run.
- Injecting archived TPU keys into the capsule's CPU stage-local DSA replay is a cheap, exact
  selection witness: the V3 host-row key reproduced the accepted layer-1/position-8155 event
  bit-for-bit (positions and scores) while the V2 on-device-rotary key perturbed every score by up
  to 4.2e-3 without flipping that event. A single-key witness cannot reproduce the protected 8K
  failure (seven swapped positions at the same event) because there every cached prompt key and the
  query carried their own position-dependent rotary error; only the full decoder with host rows for
  keys, queries and prefill can answer that.
- Correction to the rotary lesson above: accuracy against the F64 truth is not the acceptance
  criterion; identity with the legacy oracle is. The legacy `tpu-inference` GLM DSA indexer evaluates
  `jnp.cos`/`jnp.sin` on device (`glm_dsa_indexer.py:rope_cos_sin`), so the greenfield on-device
  rotary already matched it bit-for-bit at layer 0 (2,048/2,048 scores, zero delta) and the host
  FP32 table moved every layer-0 score by up to 2.6e-3 while leaving the layer-1 divergence untouched.
  Before proposing a numerical "fix", read the oracle engine's source for the exact formulation and
  check the discriminator against the oracle's own outputs at the *first exact* event, not against an
  F64 reference; a bounded proof of faithfulness to the truth can be a proof of unfaithfulness to the
  oracle.
- Compare the two protected runs (with and without a change) event by event before believing a
  bounded discriminator: a change that leaves the first failing event's error statistics unchanged
  (six swaps, mean |Δscore| 0.0148 in both) did not touch the cause, whatever it did elsewhere.
- Before hunting a row cause, invert the accepted row for its admissible scalar window (Compass v2
  T2). At layer 1/position 8155 the accepted and greenfield normalized rows are exact functions of
  the *same* FP32 input and differ only in the FP32 `rsqrt(mean+eps)` scale by 1–4 ulps; a
  one-element BF16 flip is the expected signature of that, and weeks of partial/contraction/tree
  challengers were chasing a value that was already right.
- Replay the carry exactly as the compiled program does, not as the source reads: on TPU the
  residual stream is carried in FP32 (no BF16 rounding between attention and MLP) and
  `normalized.astype(bf16) * weight` is emitted with a single rounding. A CPU replay that rounds
  where the source says to round is 1,000–2,000 elements off and proves nothing.
- The layer-1 one-ULP miss was the variance-reduce schedule, not the data: on TPU the same FP32 carry
  reduced as `f32[32,6144]{T(8,128)} -> f32[32]` (the accepted program's shape) reproduces the
  accepted row bit-for-bit, while the single-row `f32[1,1,6144] -> f32[]` reduce lands 1–4 ulps low.
  Match the accepted program's reduce *shape and layout*, not just its arithmetic.
- When cloning a hardened chain, every constant that encodes identity (environment maps, install
  paths, version suffixes, expected dependency records) must be covered by a cross-file consistency
  test or a test against real run records; two approved starts were lost to literals no test read.
- A shape/count HLO contract can be satisfied by the schedule it is meant to exclude: the DB548
  control arm reduces the same `f32[32,6144]{T(8,128)} -> f32[32]` as the accepted arm; the only
  emitted difference is whether the reduce fusion squares a materialized carry or an `add` fused
  into it. Bind lineage (what is squared, where it was materialized), not shapes.
- Turning a flag on for "the norm" is not coverage: the decode step reaches RMS norms through five
  kernels; three were still on the default schedule after the first integration. Enumerate every
  `rsqrt` in the compiled step and classify each one (the only legitimate non-RMS `rsqrt` is the
  DSA key LayerNorm), instead of trusting the call sites you edited.
- Printed StableHLO restarts `%N` numbering inside nested regions; a flat name→definition map
  silently drops every copy after the first (5 of 10 `rsqrt` vanished). Scope SSA names per region.
- Every SHA cascade must be re-run after *any* edit to a pinned file, including a two-line constant
  fix; the chain tests exist to catch this and did.
- An exemption in a fail-closed contract is itself a contract: "looks like a LayerNorm" (subtract +
  epsilon) exempted anything centred; bind every fact of the exempted op (width from the plan,
  axis, scale per opcode, combiner, rows) or the exemption becomes the hole.
- `git checkout --ours -- <path>` only acts on conflicted paths; a cleanly merged file silently takes
  the merge result. To keep one branch's file, `git checkout <commit> -- <path>` and verify the
  bytes before committing.
- Untracked files and inherited environments are code paths: a protected launch must fail on
  untracked files on every worker, list all three remote namespaces, and run committed bytes from
  a fresh detached worktree of the pin under `env -i`.
- Every change to a protected launch boundary must be dry-run end to end on CPU before it meets a
  tag: replay the exact worker prologue (worktree creation, clean check, `env -i` interpreter start,
  the script's own self-checks) as a test. String-matching the runner text proved nothing and cost a
  Sol-approved tag.
- A fail-closed contract refusal on TPU is evidence, not a loss: the archived TPU HLO of a refused run
  diagnosed two faults offline in under an hour (an unscheduled norm hidden in a fused kernel, and
  older one-live-row contracts refusing a deliberate 32-row operand) with zero further TPU time.
- When a flag changes a tensor shape on purpose, every older shape-only contract that encoded the
  opposite intent must be re-bound by lineage, not loosened by shape; and the flag's coverage must be
  counted on the real program (313 `rsqrt`), not on the call sites you edited.
- Helpers that scan all instructions per lookup are fine on a probe module and take >10 min on a
  232k-instruction decoder; index once per module.
- XLA reorders `pad` and `convert` freely; a dtype-and-shape allowance derived from the source order
  (`convert -> pad`) misses the emitted order (`pad -> convert`). Bind allowances to the SSA slice and
  the rows, not to the dtype the source happened to pad in.
