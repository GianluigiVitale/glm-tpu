# Prefill research v3 — local adjudication

2026-09-08. Owner report `research-prefill-v3.md` read in full: 1,222 logical
lines, 1,221 newline characters, 59,847 bytes, no final newline. Original SHA256:
`5ad5a7b7e66e2a88f2c3f007f0b7ac3b4ab93050c23642a6beca9ab6132a70a9`.
Preserve its bytes. Its embedded filecite/cite handles are not independently
resolvable citations here; local sources and the explicit primary URLs below
support the decisions. Independent existing gpt-6-astra reviewer checked the
bounded numerical/merge/panel proposals; conclusions below incorporate that
review and the main agent's separate source inspection.

## Outcome

Useful refinement of research2, not a demonstrated root cause or a performance
result. Keep actual row-2 DSA operands and completed MLP/router inputs first.
Expert-relative panels, exact merge alternatives and rolled prefix scan remain
separate default-off experiments after the current numerical discriminator.
Do not start a full INT8 migration, widen numerical thresholds or repeat a
full-model test to answer a one-layer question.

## Corrections that avoid unnecessary experiments

1. **No `is_stable` argument does not mean unstable top-k.** Installed JAX0.10.1
   `_src/lax/lax.py:3524` has signature `(operand,k,*,axis=-1)` and explicitly
   promises that equal values select lower indices first. Current local DSA
   helpers already sort by absolute position before `lax.top_k`. Thus they use
   this guarantee to implement the required ties. Do not add threshold repair
   or a second K-sort merely because the newer keyword is absent. A faster
   alternative still needs identical position/validity/sentinel semantics.
2. **Raw DSA order is not attention traversal order.** Actual
   `ws32_prefill_attention.py` calls `gather_stage_local_selected_kv_aligned`,
   which calls `canonicalize_selected_positions` in `reference/attention.py`.
   Attention gets ascending-position private copies; DSA state remains score
   ordered. With identical valid sets/counts, raw DSA permutations alone do not
   imply different attention accumulation order. Report Case D should first
   compare actual canonical selected-cache operands, not launch a permutation
   replay that bypasses this production canonicalization.
3. **Both current DSA scorer calls have at most32 query rows.** B128 contains
   four B32 attention prefixes, not one B128 scorer. A standalone B128 scorer
   changes geometry and is optional research, not reproduction. Prioritize
   each path's captured B32 inputs and compiled context.
4. **Equal scores alone do not convict selection.** Positions, causal lengths,
   validity/counts and candidate ownership must also be equal before report
   Case C can identify a selector defect.
5. **An eighth/ninth margin is not a complete route-set certificate.** With
   heterogeneous errors another unselected expert can overtake a selected
   expert. Cover every selected/unselected competitor using per-expert bounds
   (or a valid uniform bound); internal selected-pair margins also matter for
   ordered top8. Include finite-precision sigmoid and bias-add errors, not
   just exact-math sigmoid's 1/4 Lipschitz bound. These are diagnostics, not a
   replacement for the existing failed comparator.
6. **Linear merge complexity is not TPU latency.** The shown top-K prefix loop
   needs K comparisons, whereas4095 is a full two-list merge bound. A dependent
   scalar loop can be slower than vectorized bitonic/partial-selection lowering.
   Keep the current compiled helper and research2's bitonic merge as controls.
7. **4MiB materialization is not measured wire traffic.** The butterfly's three
   K-list exchanges are a source-level payload model; physical routing, tuple
   lowering, subgroup placement and dependency latency still need measurement.
   Do not call it a measured2.67x bandwidth or throughput improvement.

## Useful additions beyond research2

- Expert-relative workspace starts padded to supported alignment address the
  earlier arbitrary-start BlockSpec/store gap. Preserve route-slot identity,
  safe tail reads, nonoverlapping stores and zero contribution from padding.
  An M32/N256/K128 panel is a bounded challenger, not a config-only change:
  the current checkpoint has128x128 scales, so N256 needs two correctly mapped
  scale rows without pretending the checkpoint block geometry changed.
- Compare current M8, current global M32 and expert-relative M32 separately.
  Deterministic expert-visit/block-byte counts distinguish panel scheduling
  from simply increasing row tiles. Report logical bytes separately from
  physical HBM transactions, and owner imbalance separately from row occupancy.
- Gate/up shared-X scheduling may save activation reads; it does not eliminate
  either weight matrix and must retain feature4 FP32 reduction BEFORE BF16
  SwiGLU. No value/association identity is implied by unchanged interfaces.
- Double-buffered logical panel budgets are planning inputs only. Actual VMEM,
  compiler scratch/spills, live cache buffers and HBM must be established.
- Roll only the bounded per-layer32-row prefix, not the full decoder per token.
  Carry latest proposals and preserve external rollback state. A single WhileOp
  does not establish physical buffer aliasing or zero copies. Do not stack
  generations or donate the only committed cache to save presumed memory.
- Defer INT8: W8A16 is still usually dequantize-to-BF16 with this activation
  interface; W8A8 introduces activation quantization. Both retain one-byte
  weights and cannot by themselves cure repeated expert visits. No additional
  model download or checkpoint is authorized by this report.

## Current discriminator and what has already happened

While this report was read, the existing protected compile-only acquisition
finished as DB591 at pin `5e7143759b6563237751b55c89007bd506d59bb1`,119s
worker/collector, normal and root8/8clean, terminal14:42:09Z. Four actual graphs
were acquired; candidate/control expose original12 plus96/33 capture fields,
while the other two decode/promote WK. No model/WK calls.
The report's tiny row-2-only capsule is attractive, but replacing these freshly
acquired captures would require another graph acquisition and omit some original
failure-signature rows. The current bounded selected-layer capture is already
small relative to its weights; keep it unless measured memory or perturbation
requires a narrower instrument. Capturing both paths is necessary to compare
their operands; a B128-only capsule cannot supply missing B32 operands.

Next: register these actual graphs and simultaneous numerical memory budget;
run only the original boundary case, preserving all original outputs BEFORE
fallible validation. Require original signature reproduction before attributing
the failure. Row2 has fewer than2048 live keys: original selected scores already
cover every live key and can be position-aligned, so do not add a new full-score
tap unless that evidence is insufficient. Then own-input FP64/same-input replay
decides which boundary to fix. No blind full-layer rerun or generic precision
sweep. Efficient own8K, four-depth L7, L8 and final TTFT remain open.

## Primary references checked

- [Pinned JAX0.10.1 top-k source](https://github.com/jax-ml/jax/blob/jax-v0.10.1/jax/_src/lax/lax.py):
  checked against the installed source, including lower-index ties.
- [JAX top-k documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html):
  current API differs; do not import new keywords into0.10.1.
- [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html):
  hardware orientation, not a compiled allocation or proof of kernel support.
- [Google TPUv4 specifications](https://docs.cloud.google.com/tpu/docs/v4):
  same advertised BF16/INT8 peak; no automatic2x compute claim.

Third-party integer-checkpoint availability is not needed for the current
decision and was not independently revalidated; do not treat report links as
authorization to download those weights.
