# Corrected 8K refusal — first remaining observed boundary

Status2026-09-09 23:35Z. Own8K is NOT passed. Historical GateD/DB567 and
corrected own2K/DB610 remain valid within their original scope. Owner speed
freeze remains; no new throughput/precision optimization is authorized.

## What actually happened

Run `greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T225859315457683Z`
at `8f9192777bf5e90f17cf458d148ac67a4cc7b42e` completed the full8155-token
prefill in64 B128/B114 calls. Fleet-max request-prefill123.110830421s.
All8hosts then failed exact raw tokens: index11 expected2619, observed576.
All29 generated IDs match the earlier failed wide run, not the live32 control.

All14 ownDSA step contracts and cache/state structure pass. Event1 reproduces
the prospectively registered six-for-six swap. The existing independent sealer
rederivation on this run's own arrays passes allsix FP64 checks, with unchanged
reference/kappa/bounds. Therefore a later-event alarm acknowledgement cannot
turn this into a pass: the rawtoken contract genuinely fails.

Original24 JSON/NPZ/log objects were read by exactgeneration and validated for
size/CRC32C/SHA256. Every NPZ array's shape/dtype/hash matches its runner; all8
array sets and generated IDs are identical. Four authenticated pre/failure
census files, including root checks, are archived and byte-verified;8/8idle.
Original session8907terminalexit1, controller3609003/worker3610624 absent.
No restart/recovery, no protectedDBrow or remoteSUCCESS. No performance promotion.
Receipt: `../artifacts/prefill-canonical8k-token-refusal-20260909.json`.

## What the correction did and did not establish

Against retained live32, first decode step8155:

- Producers0,1,2: selectedpositions AND selectedscores byte-identical.
- First remaining observed difference: event3, producerlayer6. Six positions
  replaced (12 symmetric-set members),1763 orderedposition words and2048 score
  words differ. Order-word count is NOT the number of differing set members.
- All29generated IDs remain equal to the prior failed wide result.

This is improved early observable agreement, not a model fix. Equal producer2
indexer observations do not establish equal layer2 MLP output or residual.
The unresolved interval is after the layer2 indexer through the layer6 indexer;
it does NOT identify MoE3 as the cause. Final cache rows were captured after
generatedtokens diverged and cannot localize the original prefill difference.
Main and independent gpt-6-astra reviewer reached these conclusions separately
from original arrays. There is no new fitted FP64 reference or relaxed bound.

## Next decisive experiment — design pending, not hardware admission

Prepare ONE bounded paired frontier diagnostic through actual layers0..6 over
the original8155-token prompt, using the current canonical physicalB128 path
versus the retained-style32live grouping. Reuse the selected-weight loader,
window/runtime types, compiler journal, collective/memory guards and protected
fleet/collector. Do not create another checkpoint or a second model engine.

1. Reproduce each path's original first-decode event0..3 arrays before using
   new captures to attribute the failure. Instrumentation can change compiled
   arithmetic; if reproduction fails, report that limitation rather than
   treating the capture as the original cause. No generated-token fork is needed.
2. Compare layer2 completed update and carried residual SEPARATELY, then the
   actual layer3 normalized input. Equal sum alone can hide different streams.
3. Only advance through layers3/4/5 to the first unequal boundary needed to
   explain producer6. Distinguish upstream input, routed IDs/weights, completed
   update, carried residual and owncache/selectedstate. Do not canonicalize all
   MoE layers speculatively or start a precision/association search.
4. A first128-only test can prove a local mechanism but cannot explain this
   fullhistory8K witness. Keep the entire8155history for the discriminator.
5. Bound capture storage prospectively: stream comparisons, retain exact
   first-difference operands/locations and the original reproduction witnesses;
   avoid dumping every layer/row/owner tensor. Explicit selected-weight,
   simultaneous HBM, hostmemory, original-output and archive byte budgets are
   required before launch. These budgets and implementation are not yet admitted.

Inspect reusable components before implementation:
`scripts/greenfield/ws32_dense_frontier_{program,prepare,execution,worker,evidence}.py`,
`glm_tpu/greenfield/runtime/ws32_batched_prefill.py`,
`glm_tpu/greenfield/checkpoint/ws32_layer_subset.py` and current layer-window
interfaces. Existing two-layer/first128 profiles cannot simply be relabelled as
this fullhistory diagnostic. Reuse their machinery with a distinct bounded
protocol and independent review. No unchanged full78-layer retry or automatic
promotion of the slower live32 diagnostic is authorized.

After a proven correction: own8K again, then longcapacityHLO/HBM32, allfour128K,
full256K E0, serving/resume/actualTTFT, final DB/archive/8clean. Goal incomplete.
