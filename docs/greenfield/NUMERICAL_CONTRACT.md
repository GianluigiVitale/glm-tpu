# Greenfield numerical contract

This document records the arithmetic boundary required before any optimized
kernel may replace the native-JAX reference. The legacy repository is an
artifact-producing oracle only; it is not imported by the greenfield engine.

## Core decoder primitives

- RMSNorm converts activations to FP32 for square/mean/rsqrt, rounds the
  normalized value back to the activation dtype, then multiplies the norm
  weight in that dtype. Transformer/final norm use epsilon `1e-5`; attention
  LoRA norms use the model's `1e-6` default.
- Linear weights retain checkpoint orientation `[out_features, in_features]`.
  Dequantized BF16 activations/weights produce an explicit BF16 result; FP32
  router/indexer projections request FP32 explicitly. Bias and leading-shape
  broadcasting are validated rather than inferred.
- Dense layers use BF16 gate/up/down projections and SwiGLU, followed by an
  exact-shape, exact-dtype residual add. Vocabulary logits are not silently
  upcast.
- Main MLA and indexer RoPE use dimension 64 and theta `8,000,000`, with
  sine/cosine angles derived in FP32. The accepted interleaved layout pairs
  `(0,1), (2,3), ...` and re-interleaves the result. Half-split pairing remains
  an explicit reference option; it is never selected implicitly.

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
- PP16 ownership: experts 0–127 and 128–255 are complete on the two respective
  stage chips. The shared intermediate dimension is split 1024 ways per chip.
- Combine: local routed and shared partials are stacked as independent value
  domains, reduced once over exactly the plan's two- or four-chip stage, split, then routed
  output is multiplied by BF16 2.5 and added to the shared output. The live
  decode shape is exactly `[1,6144]`.

Bitwise equality is required for routing ids and deterministic reference
fixtures. Real checkpoint layer output uses a recorded bounded-error contract
against the captured legacy oracle because the topology rewrite changes the
replica reduction association. Raw-token exactness remains mandatory at the
full-decoder gates.

The real layer-3 oracle is a standalone raw-checkpoint PyTorch CPU transcription, not a greenfield
JAX fallback. It stores the deterministic BF16 input, FP32 router logits/weights, exact route ids,
per-expert BF16 outputs, routed/shared decompositions, and final BF16 output for normal distributed
routing and an all-eight-on-one-chip adversary. Its source revision and accepted legacy/vLLM source
file hashes are part of the manifest. Real TPU layer output uses bounded tensor comparison because
the topology rewrite intentionally changes reduction association; route ids remain exact.

Protected PP8 DB 417 passes this contract. Normal and single-chip-concentrated route ids are exact;
both final-output comparisons have max absolute error `0.03125`, p99 `0.01171875`, and mean below
`0.00236`. The same oracle and tolerances bind the mandatory PP16 challenger.

Protected PP16 DB 418 also passes. Routes are exact; normal output max/p99/mean error is
`0.015625/0.0078125/0.002121`, and concentrated output is
`0.03125/0.0078125/0.002140`. Its physical combine is exactly one BF16 two-rank reduction.
