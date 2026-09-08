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

## Wiring readiness

Mode `ws32_prefill_layer_materialized_admission` uses this v2 protocol and a distinct DB
item. Prefix and suffix are separately compiled before execution, including a real CPU
regression of abstract-prefix suffix compilation. Candidate executable/comparator remain
unchanged. Original per-owner BF16 suffix inputs are archived for each of the three cases.
Reference graphs have their own local subgroup payload/call/memory guards; the original
candidate guard is not weakened. Independent implementation review permits one bounded
launch after tests, persistence and fresh fleet guards. Hardware admission remains open.

## v2 outcome — refused, 2026-09-08

Run `greenfield_fp8_ws32_prefill_layer_materialized_admission_l3_20260908T005509603049814Z`
at e7ba4a9e passed all graphs and empty-case checks, then refused the same boundaryrow4
route order. V2's captured normalization differs in20963/104448 elements from DB585
(1230/6144 on row4). The v2 suffix's entire17x8 route array matches FP64 on its OWN
captured inputs exactly on all32owners; row4FP64 margin41-minus98 is +4.11322030658e-6.
The attempted prefix simplification changed the numerical realization. A completed
boundary does not, by itself, reproduce a prior graph's observable BF16 values.

Original HLO contains both BF16 casts in both forms, so a compiler cast-bypass mechanism
is still unproved. V2 remainsFAILED; its guards and numerical thresholds are unchanged.
Evidence in `../artifacts/prefill-materialized-v2-refusal-fp64-replicas-20260908.json`, with56
generation-bound originals, all32 selected hashes, HLO/empty-case replay and captured-input
FP64 analysis. Cleanup8/8 confirmed; no v2 success/DB admission or performance result.

Next discriminator must preserve the exact DB585 scalar prefix and all12 observed
outputs, require original input/signature reproduction, and only then test its completed
BF16 output0 with the existing MLP. Prefix router IDs cannot override the suffix router.
This remains a boundary diagnostic, NOT immediate authorization for another full-layer
trial. DB585 prefix does not expose pre-attention normalization: future full-reference
assembly cannot relabel its post-attention output as that missing observation.

## DB585 prefix → completed MLP discriminator (not v3 layer admission)

Distinct kernel `ws32_prefill_prefix_mlp_diagnostic`, protocol
`ws32-prefill-db585-prefix-completed-mlp-v1`. Only TWO graphs: unchanged DB585
scalar prefix with all12 outputs (campaign graph name `candidate`, explicitly NOT
a batched candidate), and the existing v2 scalar MLP suffix (`reference`). The
unused v2 prefix builder is not compiled or executed. No repeated candidate/M17
router run is needed: their completed-input evidence already exists in DB585.

`prefill_prefix_mlp_protocol.py` derives a compact fixture from all8 original NPZ
and runner files, each checked against the generation/size/CRC/SHA receipts in
the pinned DB585 analysis. It records hashes of ALL12 captured prefix fields per
logical owner, plus the two different complete17x8 route signatures. Original
fused-prefix routes have41-before98 at row4; completed-input scalar replay has
98-before41. These are different expected observations, not route overrides.

`prefill_prefix_mlp_worker.py` retains all17 completed device result tuples and
carries its own KV through output10. After preserving prefix captures, every host
must reproduce all12-field fingerprints, fixed inputs and checkpoint-bound router
weights; a fleet consensus precedes ANY MLP execution. The suffix receives the
same resident output0 and output9 arrays directly, never a host reconstruction.
Archive actual suffix inputs, finite outputs and routes/weights. Refuse changed
input bytes, changed expected route order or replica disagreement. MLP outputs
are not compared against a full-layer oracle, so no bounded output admission is
claimed. Correctness/latency remain NULL in DB, regardless of diagnostic success.

Original v1/v2 failures and thresholds remain unchanged. A successful result
would establish this specific reference boundary only. Full-layer reference
assembly still owes actual PREattention normalization, then all original layer
cases/interventions and the short decoder's independent §21 proof.

DB586 seals this two-program discriminator at `66a2f755` (85s worker/collector,
01:33Z clean8/8): every first12 field matches DB585 across32 owners, completed
MLP route arrays match all17x8 standalone replay routes. Peak334380032B/chip.
Evidence `../artifacts/prefill-db585-prefix-completed-mlp-result-20260908.json`.

## v3 full-layer reference — actual 13th PREnorm observation

Default-off kernel `ws32_prefill_layer_observed_admission`, protocol
`ws32-prefill-layer-db585-observed-reference-v3`. Append the EXISTING `norm`
variable actually supplied to attention as the13th prefix output; no recomputation
or substitute. Default12-output prefix remains unchanged. Compile candidate,
13-output scalar prefix and existing scalar MLP with separate exact guards/budgets.

Before any candidate or MLP execution, evaluate the fixed boundary's17 prefix
rows, carrying ownKV through output10. Archive all13 observed outputs, original
inputs and checkpoint-bound router weights in `boundary_prefix.npz`. Require
DB585 first12-field fingerprints on all32 owners and finite BF16 actual PREnorm;
collective consensus must succeed before any full-layer work. If the added
observation perturbs the prior prefix, refuse and retain original evidence.
No blind barrier variants or relaxed fingerprints are part of this protocol.

Reuse those completed boundary tuples for that case; do not rerun its prefix.
Run the original empty/boundary/tail cases and unchanged causal/cache/health
interventions against the unchanged candidate. Scalar MLP input is prefix0,
carried residual prefix9; output schema contains ownKV10, health11 and actual
PREnorm12. Collector binds boundary suffix-input and full-layer PREnorm captures
to the original prefix observations, in addition to all original numerical checks.
The extra reference programs/observations are untimed; numerical success is only
real-weight/synthetic-state bounded layer admission, not legacy/full-model or
performance proof. No repeated layer0 admission or complete checkpoint load.
