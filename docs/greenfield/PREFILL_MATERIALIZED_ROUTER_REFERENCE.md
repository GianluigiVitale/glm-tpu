# Layer3 reference realization v2 — completed BF16 MLP input

2026-09-08. Proposed, default-off bounded reference variant; NOT a retrospective
pass for the original layer3 comparison or an amendment to protected decoder §21.

## Evidence and decision

DB585 (`greenfield_fp8_ws32_prefill_router_boundary_diagnostic_l3_20260908T003156848804906Z`,
pin `2bdf7bd574afbcc5927069a4ff76ecf7df95452e`) reproduced BOTH original17x8 route arrays
on all32 owners. Original layer3 refusal remains FAILED. The diagnostic records no
numerical-admission or performance success. Its20 decisive remote archive objects
are generation/size/CRC/SHA verified in `../artifacts/prefill-router-captured-input-fp64-20260908.json`.
All original local HLO/NPZ replay and cross-owner replica checks pass; terminal normal
and root censuses are8/8 clean. Peak HBM330,352,128 B/chip includes the references.

At the failing row4, actual and reference observed BF16 router inputs are identical
over all6144 features. Both standalone M17 and M1 routers on either captured input
produce the candidate order (98 before41), with zero ordered-ID differences between
shapes. Their FP64 captured-input logit errors are at most1.28e-7. The fused scalar
prefix differs from the same captured-input math by up to0.00191164 in logits and
0.00109619 in local partials. FP64 score41-minus98 at row4 is -1.02131983e-5;
standalone paths -1.14440918e-5; fused scalar prefix +3.81469727e-6.
The sole observed normalized-input difference over the whole block is row6,
feature1 offset1443, so it cannot explain row4.

HLO does explicitly contain the BF16 conversion before the scalar router. Its norm
fusion emits both the BF16 captured branch and an F32 branch to scalar projection.
Thus the evidence localizes a fused-prefix versus materialized-boundary discrepancy;
it does NOT prove that HLO omitted a cast or establish the hardware mechanism. The
FP64 calculation starts at captured inputs, not tokens/checkpoint full-forward state.
Do not claim either full-model accuracy or an original uninstrumented logit capture.

Independent reviewer agrees: do not force the batched router to reproduce this
reference behavior or try a succession of speculative compiler barriers. Reuse
the already-measured completed BF16 boundary in a NEW scalar reference realization.

## New reference and unchanged admission requirements

Keep the candidate executable, original fixed inputs, exact ordered-route comparison,
weight bounds, layer-output bounds, cache rules and causal/health interventions
unchanged. Keep the original v1 protocol, artifacts and failed outcome accessible.

For layer3 only, construct an untimed scalar reference with TWO completed device
programs per row:

1. Existing scalar input norm and attention, then existing post-attention fused norm;
   return the completed BF16 MLP input, carried residual, actual pre-attention norm,
   updated KV and health. Do not run or reuse router IDs from this prefix.
2. Existing scalar `ws32_mlp_mapped` with that completed BF16 input as
   `precomputed_normalized_local`, raw retained MoE weights and `add_residual=False`.
   It computes its own router IDs/weights and expert output. No candidate route override.

Only the untimed reference has this extra executable boundary. Candidate remains a
single token-batched layer program. Device arrays pass directly, without a host
numeric reconstruction; the scalar loop is a reference, not production dispatch.
Carry the reference's own KV between rows, preserve shared index/repair caches and
selections, and reconstruct the existing12-field layer result without slot confusion.
The reference's normalization must not be recomputed from a rounded carried residual.

Give the variant a distinct run mode/protocol and DB item. Preserve both prefix and
suffix HLO/StableHLO, selected checkpoint digests, all original arrays, per-chip HBM,
generation-bound archive and cleanup. Declare the reference realization explicitly;
the v1 raw scalar reference remains unchanged. No result from this variant may be
described as identity to the v1 fused reference or the promoted decoder.

## Smallest tests and remaining scope

- CPU: actual20-field geometry, completed BF16 boundary, reference-owned KV carry,
  shared cache/selection identity, suffix source API and distinct classification.
- Use DB585 originals to check observed weights/input hashes, exact canonical routes
  and FP64 arithmetic interpretation; do not substitute synthetic inputs for those.
- Review implementation before launch. One bounded layer3 v2 admission with all three
  fixed cases and original comparator thresholds; first failure stops, originals retained.
- If admitted, proceed to the short full decoder and its OWN §21 evidence, then efficient
  L7/L8 with preregistered prefill/TTFT targets. A reference correction is not a speedup.

No forced-round or optimization-barrier variant is authorized by this design. The
existing forced-BF16 RMS helper is only a possible future cheaper implementation;
its source annotation is not physical rounding proof. No production model changes,
full checkpoint reload or long serial test is required to answer the current question.
