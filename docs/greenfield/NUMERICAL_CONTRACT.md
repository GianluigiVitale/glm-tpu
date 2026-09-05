# Greenfield numerical contract

This document records the arithmetic boundary required before any optimized
kernel may replace the native-JAX reference. The legacy repository is an
artifact-producing oracle only; it is not imported by the greenfield engine.

## Core decoder primitives

- RMSNorm converts activations to FP32 for square/mean/rsqrt, rounds the
  normalized value back to the activation dtype, then multiplies the norm
  weight in that dtype. Transformer/final norm and q_a/kv_a attention LoRA
  RMSNorm use the pinned model config's epsilon `1e-5`. The indexer key
  affine LayerNorm is a distinct operation and retains its source epsilon
  `1e-6`.
- The DSA `wq_b` exact path materializes only the complete topology-local FP32
  owner shard before its true-row projection. DB522 requires the fused q-a BF16
  result to complete behind an optimization barrier. Protected DB525 then
  requires four top-level aliases of that same owner, four N1024 reductions
  retained behind one grouped barrier, and only the first result live. TPU must
  emit one tuple-valued fusion with `megacore_allreduce_bytes=16384`; ordinary
  N1024, N128, precision and GSPMD alternatives are rejected by DB521--524.
  PP8 never reconstructs the global `wq_b` matrix or carries another owner's
  state. This correction remains default-off until the actual fused-q-a,
  separate local materializer and helper composition is protected-exact.
- The sealed all-event observer proves production layer-0 normalized hidden is
  bitwise exact but its q-a state differs in 494/2,048 BF16 values (max
  `0.015625`). The accepted source uses 32 shard-major fused q-a/kv-a dots with
  physical width `N=82` (`64 q-a + 18 kv-a`) before distributed RMSNorm.
  Greenfield may virtualize that association inside one local stage only with
  a true `[1,6144]` row, no full-pod collective, and no `[32,...]` token bucket.
  The 32-way diagnostic collective itself remains forbidden in production.
  Projection and norm association stay default-off until a bounded TPU probe
  matches the accepted BF16 q-a state elementwise.
- DB501 rejects ordinary M1 N82 dot association: all tested projection mappings
  and FP32 norm reductions produce the same 376-mismatch BF16 result. Its TPU
  HLO lowers M1 dot to multiply/reduce, whereas the exact accepted DB491 local
  body is a zero-spatial `convolution` with `bf_io->bf` labels. A direct
  one-row convolution is therefore required; restoring an external
  `[32,6144]` decode bucket remains forbidden.
- Protected DB502 proves the direct one-row N82 convolution bitwise for the
  accepted layer-0 q-a state under all four tested norm associations. Production
  may use this mechanism only behind a default-off backend, with already-packed
  shard-major `[32,6144,82]` FP8 weights and `[32,48,82]` FP32 scales. It must
  reuse the fused 576-wide kv-a companion, retain the physical
  `f32[1,82] convolution`, and never perform q-a/kv-a packing per decode step.
  This bounded result does not prove later layers, the decoder, or performance.
- Production pin `0082bac` implements that contract without importing the
  diagnostic execution path. It accepts only final-layout U8/FP32 packed
  tensors, performs no runtime q-a/kv-a concatenation, and supplies the fused
  kv-a result directly to IndexShare. Separate and fused projection state are
  mutually exclusive. This remains unpromoted until a protected production
  helper/layer proof and the complete packed derivative pass.
- Protected DB503 executes that integrated production selector on TPU and is
  bitwise exact for both normalized q-a (0/2,048 mismatches) and the fused
  576-wide kv-a companion. Its optimized HLO has exactly one physical
  `f32[1,82] convolution ... bf_io->bf`, one live row, and no collective,
  callback, dead row, or forbidden separate state. This closes production
  helper arithmetic/HLO only; the full packed derivative, all 78 layers, raw
  tokens, DSA, HBM, and decoder performance remain separate gates.
- The accepted fused residual-add/RMSNorm boundary first adds the two BF16
  inputs in FP32. Its normalization consumes that unrounded FP32 sum, while
  the independently carried residual is the same sum rounded to BF16. A
  rounded-first BF16 add followed by ordinary RMSNorm is not equivalent. The
  default-off split-state decoder therefore preserves raw hidden update plus
  carried residual through every layer and applies the same fused operation at
  final norm. Its PP8 stage payload is exactly `bf16[2,1,6144]` (24,576 bytes),
  adding 12,288 bytes to the historical one-component transport; no additional
  row, collective, cache, weight, or metadata payload is allowed.
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
- Protected DB518 proves the prompt index-key producer requires each logical
  M2048 chunk to execute as 32 M64 partitions. Each partition consumes a BF16
  `[64,6144]` normalized lhs and FP32 `[128,6144]` adapted `wk`, requests mixed
  `[DEFAULT,HIGHEST]` dot precision, and performs the biased FP32 key LayerNorm
  before leaving the map. Projection, pre-RoPE, post-RoPE, and the complete
  8,155-row BF16 cache are elementwise exact under this association.
- The default-off PP8 prefill adaptation records the exact BF16 normalized
  projection input already consumed by every full indexer on its owning stage;
  it must not re-normalize the rounded split-residual boundary. The post-scan
  repair overwrites only that stage's LP4-owned cache rows. At 8K its fixed
  history budget is 501,043,200 bytes/device. Admission requires 84 physical
  M64 calls, no grouped `[32,64]` key-norm reduction, no collective, host
  callback, full-pod history, or non-positive measured HBM headroom. Recurrent
  `decode_batch1` remains the unchanged true one-row executable; DB518 is
  arithmetic evidence, not an integrated Gate-D result.

## DSA scorer and selected positions

- The indexer query is `[rows,32,128]`; `decode_batch1` fixes `rows=1`. Its
  key cache is `[context,128]`. Query/key projections, biased key LayerNorm,
  per-head dots, ReLU, signed head weighting, and the final head sum are FP32
  with JAX matmul precision pinned to `highest`.
- For the default-off exact PP8 reference path, fused q-a must complete at a BF16 barrier before
  query projection. Each physical owner materializes its local `wq_b` shard to FP32 once, and the
  same local owner value is supplied through four aliases so TPU retains one tuple-valued four-dot
  N1024 reduction fusion; only the first result stays live. DB526 proves the complete production
  composition matches accepted q-a and query bitwise with no collective/global owner state. The
  feature remains disabled unless DB525 mechanism and DB526 composition artifacts are pinned.
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
  position ties. Independent cross-backend score tensors on the same captured hidden input use
  bounded comparison because CPU PyTorch and TPU XLA need not share dot/reduction association. Raw
  CPU positions are retained as a diagnostic and may never replace, seed, or relax runtime
  selection.
- The full-decoder observer therefore records positions and the bit-exact FP32 scores emitted by
  the same executing `lax.top_k`. It gates canonical executing-score order/ties and exact
  set/count/tails. Full-network topology reassociation changes the hidden input presented to later
  layers, so position-aligned legacy score errors and total rank order are recorded diagnostics,
  not Gate C same-input comparisons and not gates. Exact raw tokens remain mandatory. The observer
  is a separate no-donation executable; the production executable remains observer-off.
- When raw tokens diverge, that separate observer may export a compact canonical global top-16
  logit record: BF16 vocabulary logits are cast to FP32 only for bit-exact storage, scores descend,
  equal scores use lowest token ID, and the emitted winner must equal the recurrent output. This is
  diagnostic evidence only: expected-token rank/margins never relax exact raw-token equality. The
  wider score/id payload must reuse the two existing local exchanges, add no collective, and leave
  all non-token collective shapes/counts identical to production.
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
- The DB537-selected PP8 alternative is separate and default-off. Every lane canonicalizes the
  same ascending 2,048-position selection, writes only its owned BF16 cache rows into those exact
  slots and writes BF16 zero elsewhere. One four-lane stage-local BF16 sum reconstructs the
  selected `[1,2048,640]` segment exactly because every element has one nonzero owner. It never
  gathers the paged cache or crosses the LP4 group.
- Each lane retains only its local 16 query heads and runs the pre-gathered sparse-MLA recurrence
  with a 512-row segment block. Scores, online max/sum and normalization remain FP32; unnormalized
  probabilities round to BF16 before the latent PV, and the final `[1,16,512]` latent rounds to
  BF16 once. DB537 proves this H16/B512 arithmetic bitwise exact against the accepted 64-head
  attended latent. B128 is explicitly nonexact and may not substitute.
- In the DB537 path the old query all-gather and output/LSE/validity all-gathers are absent. The
  selected-cache exchange is exactly one BF16 LP4 sum per layer; no validity collective is needed
  because all lanes validate identical global selection/page metadata before ownership masking.
  Every exact-name B512 call must be inside the attention scope, its cache operand must depend on
  exactly one scoped sum through only cache-preserving bitcast/copy/reshape transforms, and all
  exchange/call links must be bijective. Any bypass, escaped group, old attention-exchange scope,
  wrong B512 kernel count/shape or dead decode row is a hard HLO refusal.
- DB539 fixes the subsequent attention-output projection association. Each LP4 owner computes
  eight independently BF16-rounded K512 contraction partials from its local 4,096-wide output
  projection input. One four-owner LP4 all-gather forms 32 virtual partials locally, then the
  DB533 physical-row-zero `y -> x -> z` StrategyND tree reduces them. This applies to attention
  projection only; dense and MoE combines retain their established local contracts.
- The complete 78-layer StrategyND attention path must contain exactly 624 exact-name K512 Pallas
  calls and 78 four-rank local all-gathers. Each gather must consume exactly eight calls, every
  call must feed exactly one gather, no unapproved leaf or arithmetic may enter the gather operand,
  and every gather must remain live at the decoder root. TPU's folded `bf16[32,1,6144]` value is
  explicitly classified as 32 virtual contraction partials resident within LP4, never as a
  physical 32-chip residual reconstruction. The feature is default-off and requires the protected
  split-residual, selected-cache B512 path.

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

For DB539's StrategyND attention-output path, optimized HLO is necessary but insufficient because
backend fusion erases slice order and most individual BF16 additions. The paired StableHLO contract
requires shard `i` to use input/weight columns `[512i:512(i+1)]` and scale columns
`[4i:4(i+1)]`, all eight partials to share one layer source triple, the exact DB533 physical row
permutation, and the complete barrier-rounded `y -> x -> z` tree. The exact tree must be the
gather's sole consumer and reach a function or manual-computation return. Swapping rows, cross-
wiring layers, reassociating an add, or returning a bypass is a hard refusal. The lhs and scale pad
calls must also resolve to exact private helpers: scalar i32 zero converts to BF16/FP32, then pads
`[1,512] -> [8,512]` with high `[7,0]` or `[4,48] -> [8,128]` with high `[4,80]`, both with zero
low/interior values and direct return lineage. An opaque, unknown or differently placed pad refuses.

## Gate D cross-oracle correctness (spec §21, 2026-09-05)

- Level 3 (exact) applies to raw greedy tokens against the sealed legacy oracle and to every
  structural cache/state fact. Level 1 (bit-exact) is required only within the engine: distributed
  selection and lowest-position tie order versus a canonical top-k of the engine's own executing
  score row. Level 1 is **not** required against the legacy engine's intermediate arithmetic.
- Reference `R`: an independent FP32 CPU score row computed from the sealed legacy inputs of the
  event by code sharing nothing with the engine scorer; implementation, dtype and hash recorded first.
- Cross-oracle selected sets are exact or boundary-explained with `eps_event = max |s_o − R|` over
  the aligned positions `A = E ∩ O` (the oracle's own error; the engine cannot inflate it), an
  absolute pre-registered cap on `max |s_e − R|` (DB421 `0.003605` unless a layer-specific bounded
  oracle value exists), every position of `E Δ O` within `eps_event` of both cutoffs after
  lowest-position tie resolution, and `|E Δ O|` no larger than the oracle's ambiguity band. Any
  violation is a hard failure.
- Systematic bias: `|mean(s_e − R)| ≤ max(|mean(s_o − R)| + 3·s/√n, 0.000965)` over `A`. A
  near-uniform signed shift larger than that is a defect to localize, not noise to tolerate.
- Evidence: Gate C DB421 (raw PyTorch CPU scorer vs greenfield TPU FP32 scorer disagree on 2/2,048
  cutoff members while both satisfy the bounded contract); WS32 8K runs of 2026-08-26/27 (exact
  tokens/state/cache/event 0, seven event-1 selected-position swaps, no aligned error statistics
  recorded). The PP8 run `…_20260808T041407656729112Z` recorded aligned event-1 error
  max/mean/signed `0.27013397/0.18290268/-0.18290268`; a shift of that size would fail the bias rule
  by two orders of magnitude and must be adjudicated on the WS32 arrays before any Gate D claim.
