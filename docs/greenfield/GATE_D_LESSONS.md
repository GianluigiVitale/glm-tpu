# Gate D Lessons and Non-Repeat Rules

This is the compact operational memory for the GLM-5.2 greenfield short-context gate. The
append-only evidence remains in `docs/RESEARCH_LOG.md`; this file records the reusable rules.

## Current boundary

- DB550 proves all 32 real layer-0 dense down partials match DB548 bitwise: `0 / 196,608`
  mismatches and raw SHA `9d9f65dd...16e35`.
- Therefore checkpoint packing, FP8 decode/scales, SwiGLU, all 16 contractions, scheduled fusion
  geometry and rank-local outputs are closed. Do not reopen or rerun those hypotheses.
- The one remaining layer-0 discriminator is physical M32 StrategyND association versus the
  carried-residual/layer-1 RMSNorm boundary. Gate D is not closed until the complete 8K decoder has
  exact tokens and DSA plus its trace, wall, memory and integrity evidence.

## Evidence ladder

1. Localize the first failing tensor boundary; do not optimize from the final token alone.
2. Capture the smallest real accepted tensor immediately before that boundary.
3. Recompute all candidate arithmetic offline from sealed bits.
4. Replay only the disputed operation on TPU with the real tensor and exact scheduled HLO.
5. Integrate only a bitwise-exact result, then run one complete protected 8K confirmation.

A full checkpoint/8K run is forbidden while a smaller capture or replay can decide the same
hypothesis. Compile once and replay sealed values. Keep negative results: they permanently remove
branches from the search tree.

## Observer-effect rules

- Returning or materializing an internal tensor can change fusion and reduction association.
  DB541 is invalid for exactly this reason.
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

## Numerical rules

- One BF16 ULP is material: DB548's single hidden-index mismatch changed layer-1 DSA selections.
- Preserve BF16 rounding points and physical reduction association; algebraic equivalence is not
  exact execution equivalence.
- Capture pre-reduction partials before changing contractions. Exact partials plus a wrong combined
  row localize the fault to association; exact combination moves investigation downstream.

## Review and run discipline

- Work in a coherent bulk, run mutation/static tests, then request one review of an immutable staged
  diff SHA. Apply findings in one correction batch and ask the same reviewer only for correction
  closure. Do not repeatedly audit unchanged code.
- Record every accepted/rejected hypothesis, artifact SHA and exact next action in the handoff and
  research log before compaction.
- Protected runs require clean code pins, exact source hashes, physical topology/host bindings,
  pre/post eight-host zero-work census, CRC-verified complete remote object equality and remote
  `SUCCESS` last. Diagnostics never become performance claims.
- Treat a CLI output path as part of its API: this benchmark derives sibling `hlo/` and `replay/`
  directories from the output parent. A wrapper path refactor must assert all derived artifact paths,
  not only the JSON destination. A completed device call without terminal artifact collection remains
  diagnostic, even when all host records agree numerically.

## Decision after the model-free replay

- Hardware row zero differs from DB533 software: change only the bounded four-chip combine
  association, prove it with the same real replay, then integrate.
- Hardware row zero equals DB533 software: freeze dense arithmetic and inspect only carried residual
  and layer-1 RMSNorm provenance/arithmetic.

Do not return to hour-scale hypothesis runs or already exact contractions in either branch.
