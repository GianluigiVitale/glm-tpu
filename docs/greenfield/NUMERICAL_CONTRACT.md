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

## DSA scorer and selected positions

- The indexer query is `[rows,32,128]`; `decode_batch1` fixes `rows=1`. Its
  key cache is `[context,128]`. Query/key projections, biased key LayerNorm,
  per-head dots, ReLU, signed head weighting, and the final head sum are FP32
  with JAX matmul precision pinned to `highest`.
- The first 64 query/key dimensions use accepted interleaved RoPE. Per-head
  dots are scaled by `128**-0.5`, ReLU occurs before signed head weighting,
  and head weights are scaled by `32**-0.5`.
- Selection is exact `lax.top_k`, never approximate. Equal finite scores are
  ordered by lowest global position. Distributed selection keeps a full
  `min(2048, local_context)` candidate width per local context owner, restores
  ascending global-position order before the final exact merge, and is thus
  invariant to collective concatenation order.
- Exactness is defined against the FP32 score row produced by the executing device program: the
  distributed merge must equal a canonical global top-k of that same row, including lowest-global-
  position ties. Independent cross-backend score tensors use bounded comparison because CPU
  PyTorch and TPU XLA need not share dot/reduction association. Raw CPU positions are retained as
  a diagnostic and may never replace, seed, or relax runtime selection.
- Compact selected state is `positions int32[rows,2048]` plus
  `valid_counts int32[rows]`; invalid tail slots are exactly `-1`. IndexShare
  reuses the score-ordered positions unchanged. Only the positions array is
  transferred across a stage boundary: valid counts are derived from the
  exact `-1` suffix, producer layer is compile-time schedule metadata, and
  event position already travels with the residual. No KV row is transferred.

## Stage-local KV and sparse MLA

- The packed latent-cache row is BF16 width 640: normalized latent width 512,
  rotated key width 64, then 64 inert padding lanes. A logical 512-token page
  is striped in-page over only the plan's two- or four-chip local stage group.
- DSA/IndexShare state remains descending-score ordered. Attention makes a
  private ascending-global-position copy, with `-1` tail preserved, so its
  reduction order is a deterministic function of the selected set. Duplicate,
  stale, non-causal, out-of-page, or malformed-tail state sets a device health
  predicate false; it never causes an out-of-range cache read.
- Each local owner gathers and attends only its disjoint selected subset.
  Queries are absorbed latent `[rows,64,512]` plus rotated `[rows,64,64]`;
  scores and softmax are FP32 and scale by `256**-0.5`. Max-shifted,
  unnormalized softmax weights round to the cache dtype before the latent PV
  dot; normalization by the FP32 sum occurs afterward, matching the TPU path.
- Owner partials emit normalized latent plus FP32 log-sum-exp. The exact
  stage-local merge weights each partial by its LSE; an empty owner contributes
  zero and an all-empty row returns zero. No stage/pod axis participates.

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

Protected Gate C DB 421 passes the device-score contract for real dense/full-DSA/IndexShare TPU
layers. Distributed selection and tie order have zero mismatches against a canonical top-k of the
actual TPU score row, and the exact resulting state is consumed by IndexShare. The independent
PyTorch CPU row has bounded score error but swaps two of 2,048 members at the cutoff (2,046 set
overlap), so raw cross-backend position identity is explicitly false. No tolerance is applied to
the runtime selection assertion, and the CPU positions are not used as runtime state.
