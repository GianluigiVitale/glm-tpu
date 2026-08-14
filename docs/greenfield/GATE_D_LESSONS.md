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
  closed as well. The remaining observed HLO delta is pre-dense RMS scheduling: accepted retains
  only the scalar reduction and recomputes the rounded residual/normalized value in the gate
  fusion; the control retains a scalar-plus-full-residual tuple. Test that boundary only.

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
- Parse real attributes outside quoted metadata and comments. Pin physical layouts, replica groups,
  reducer parameters/opcode, scheduled backend geometry and synchronous collective form.
- Every accepted HLO form needs adversarial mutation tests and a SHA-pinned preserved-real replay.
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

## Decision after the model-free replays

- Hardware row zero equals DB533 software. Freeze dense arithmetic and standalone association.
- The global StrategyND/RMS program is mechanically valid and completes in tens of seconds. Both
  hybrid and direct-residual forms are nonexact; the direct form exactly reproduces DB549 and is
  permanently rejected.
- The one-graph contraction→StrategyND→RMS control reproduces DB548 and is closed. Its accepted
  scalar-only RMS schedule/recompute challenger exactly reproduces DB549 and is closed as well.
  The ordinal discriminator reconstructed the sealed attention row through a first
  value-preserving StrategyND reduction, then ran dense second; its protected output was unchanged
  from rejected DB549. Freeze that arm. The remaining bounded discriminator restores accepted
  scalar-only pre-dense RMS plus gate-fusion recomputation while retaining the already frozen
  layer-1 split. It loads only layer-0 weights and remains no-DB/default-off until all 6,144 values
  are exact.
- The first scalar-only launch reached the intended schedule but failed only in proof. Recovered TPU
  HLO shows the final fusion forms the M1 F32 sum from two independently sliced BF16 row-zero
  sources. This exact lowering is now the sole admitted correction; it does not reopen RMS
  arithmetic or authorize a decoder run until the complete protected row is exact.
- Integrate only a structural boundary that makes the entire 6,144-value accepted row exact. Never
  patch one coordinate or accept a scalar selected from an inexact arithmetic surrogate.

Do not return to hour-scale hypothesis runs or already exact contractions.
