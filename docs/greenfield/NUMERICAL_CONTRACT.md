# Greenfield numerical contract

This document records the arithmetic boundary required before any optimized
kernel may replace the native-JAX reference. The legacy repository is an
artifact-producing oracle only; it is not imported by the greenfield engine.

## GLM-5.2 sparse MoE

- Storage: routed/shared expert weights are FP8 E4M3 with FP32 inverse scales
  over checkpoint-oriented `[out, in]` 128x128 blocks. Router weights are
  BF16; correction bias is FP32.
- Load arithmetic: multiply FP8 values by their FP32 block scales and cast the
  resulting weights to BF16 on device. TPU v4 has no FP8 MXU path.
- Router: cast the BF16 input and router weight to FP32, compute FP32 logits,
  apply FP32 sigmoid, and add correction bias for selection only.
- Ties: `jax.lax.top_k` order, which selects the lowest expert ids for equal
  scores. The exact selected ids are part of the protected result.
- Selected weights: gather from unbiased sigmoid scores, normalize in FP32,
  then cast to BF16 at expert combine. No approximate top-k is permitted.
- Routed expert: BF16 gate/up/down dots with SwiGLU and a BF16 top-8 reduction.
  TPU MXU accumulation uses floating-point accumulators and the observable dot
  result boundary is BF16.
- Shared expert: the same BF16 SwiGLU contract with intermediate size 2048.
- PP8 ownership: experts 0–63, 64–127, 128–191, and 192–255 are complete on
  the four respective stage chips. The shared intermediate dimension is
  split 512 ways per chip.
- Combine: local routed and shared partials are stacked as independent value
  domains, reduced once over exactly the four-chip stage, split, then routed
  output is multiplied by BF16 2.5 and added to the shared output. The live
  decode shape is exactly `[1,6144]`.

Bitwise equality is required for routing ids and deterministic reference
fixtures. Real checkpoint layer output uses a recorded bounded-error contract
against the captured legacy oracle because the topology rewrite changes the
replica reduction association. Raw-token exactness remains mandatory at the
full-decoder gates.
