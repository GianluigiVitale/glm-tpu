# Gate D Compass report adjudication

**Status date:** 2026-08-30
**Input:** `compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`
**Verdict:** useful research lead; **not a Gate D answer or execution authority**

## TL;DR

The report contributes a useful compiler primitive, `stablehlo.reduce_precision`, and correctly
keeps attention on conversion placement and FP32 reduction association. It does not yet solve Gate
D. Its recommended M1 rounds the wrong value: accepted `fused_add_rms_norm` normalizes the
**unrounded FP32 transient sum** and independently returns a BF16-rounded recurrent residual. M1's
sketch rounds the transient before RMS, which changes accepted arithmetic.

The proposed decisive offline Stage 0 is also impossible from the stated evidence. Optimized HLO
contains a program and runtime parameters, not the missing position-8,155 accepted FP32 operand or
its runtime partial values. Those bytes are precisely the sealed observability gap. M2 and M3 then
assume an accepted 32-partial association that has not been recovered and cannot be inferred from a
generic reproducible-summation paper.

Do not implement M1, compile a candidate, or run TPU from this report. Preserve it as research
input and ask for a corrected mechanism that respects all three distinct boundaries below and does
not require unavailable accepted runtime values.

## 1. Authority precedence

The Compass file is an external hypothesis document. It does not supersede:

- accepted source at vLLM commit `a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c`;
- `docs/artifacts/db485-layer1-rms-hlo-causality.json`;
- `docs/artifacts/plan-local-persistent-fp32-shadow-source-rejection.json`;
- `docs/artifacts/gate-d-observability-frontier.json`;
- `docs/artifacts/gate-d-shadow-variant-adjudication.json`;
- the historical mechanism and Pallas tombstones in `docs/greenfield/EVIDENCE_MAP.md`.

Local runtime evidence remains authoritative over a paper, generic compiler description or
plausible hardware explanation.

## 2. The three boundaries the report conflates

Accepted source computes:

```python
summed_fp32 = float32(hidden_update_bf16) + float32(residual_bf16)
carried_residual_bf16 = bfloat16(summed_fp32)
variance_fp32 = mean(square(summed_fp32))
normalized_fp32 = summed_fp32 * rsqrt(variance_fp32 + epsilon)
weighted_output_bf16 = bfloat16(bfloat16(normalized_fp32) * weight_bf16)
return weighted_output_bf16, carried_residual_bf16
```

These are different values and edges:

1. `summed_fp32`: unrounded transient consumed by the current RMSNorm;
2. `carried_residual_bf16`: independently rounded recurrent state consumed at the next boundary;
3. `weighted_output_bf16`: current RMSNorm output consumed by qkv-a.

DB485's explicit `bf16[32,6144]` qkv-a input is edge 3. It does not prove that edge 1 should be
rounded, nor that edge 2 physically materialized in HBM. The sealed HLO certificate explicitly
makes no physical-materialization claim.

## 3. Mechanism-by-mechanism verdict

| Mechanism | Verdict | Decisive reason |
|---|---|---|
| M1 `reduce_precision` before RMS | **Reject as sketched** | Rounds edge 1 before variance/RMS and changes accepted source semantics. |
| M2 fixed accepted reduction tree | **Not constructable** | The accepted runtime partials and exact physical tree are unavailable; LP2/LP4 also use different physical summands/geometry. |
| M3 Pallas/Mosaic fixed tree | **Research only** | Opacity can protect a known algorithm, but the target accepted tree is unknown and prior Pallas/source-fused/layout families are sealed failures. |
| M4 compensated EFT | **Report form not admitted** | No concrete rooted device consumer or all-input proof is supplied; a genuinely distinct auxiliary form remains unadjudicated. |
| M5 auxiliary tuple | **Still unresolved, not rejected by this report** | The declared form has a rooted device auxiliary result; the report incorrectly equates it with the already-rejected unconsumed shadow. |

### 3.1 M1: the proposed primary path is semantically wrong

Compass lines 45--50 apply `reduce_precision(e8m7)` to the “pre-round RMS operand” and feed the
rounded FP32 result into RMS. Accepted source instead squares and normalizes the original unrounded
FP32 sum. This is not a physical pin of accepted semantics; it is a new numerical program.

StableHLO specifies that `reduce_precision` returns the original tensor type after simulated lower-
precision rounding. That makes it a useful semantic rounding primitive, but not proof of BF16
storage or a TPU materialization boundary. `optimization_barrier` is an identity that prevents
motion across the barrier; it does not promise layout, HBM storage or an unchanged executable.

The default `xla_allow_excess_precision=true` is real, but its existence proves only that XLA may
increase an instruction's output precision. It does not prove that this flag acted on the sealed
DB485/DB518 edge, that a required BF16 program result was skipped, or that it caused hidden element
2,795. Calling it the “single most probable root cause” is therefore unsupported by local evidence.

A source-exact variant would have to preserve `summed_fp32` for RMS and preserve the independent
BF16 returned recurrence. Adding `reduce_precision` only before the BF16 returned residual is
semantically redundant with its existing conversion and falls into the already duplicate-closed
rounded-primary family unless a new causal physical distinction is mechanically proved.

### 3.2 Stage 0 is circular

Compass line 185 asks to extract the accepted position-8,155 FP32 operand and 32 partials “from
accepted HLO constants/inputs.” The HLO parameters describe runtime tensors; they do not contain
their position-specific runtime bytes. No preserved artifact contains the accepted unperturbed
`layer1.rms_input_fp32`, and callback reconstruction belongs to a rejected executable class.

Therefore Stage 0 cannot compare accepted and candidate FP32 operands, cannot replay the accepted
runtime partials, and cannot decide M1 versus M2 without first solving the exact observation or
mechanical-proof problem it assumes away.

The suggested hand-set FP32 test merely proves the published semantics of `reduce_precision`.
`0x27bd` is the downstream normalized BF16 output bit at hidden element 2,795, not a known bit
encoding of the missing pre-RMS FP32 operand. Expecting the pre-RMS round itself to equal `0x27bd`
compares different tensors. The test cannot establish that the candidate should apply a round at
that edge.

### 3.3 M2 overclaims equivalence

A reproducible sum produces the same result for its own defined algorithm independent of partition
order. It does not imply equality with an unrelated accepted TPU reduction that did not use that
algorithm. Likewise, a fixed tree matches accepted for all inputs only after proving identical
leaves, rounding points and tree. Those facts are not available here.

Accepted, PP16 and WS32 have different RMS square/reduction/output geometry. An LP2/LP4 program
cannot simply name its local partials “accepted leaf order” and thereby reproduce 32 physical
partials. Without a sealed accepted tree, M2 is another arbitrary/uniform-tree proposal and remains
inside the tombstoned family.

### 3.4 M3 provides containment, not the missing target

Pallas/Mosaic may prevent XLA passes outside a kernel from rewriting its internal algorithm. That
is useful only if the exact algorithm to preserve is known and supported. The report neither
recovers the accepted physical association nor proves that its illustrative remote-DMA,
semaphore, dimension-semantics and `reduce_precision` combination lowers as claimed on TPU v4.

Prior output-only, source-fused, all-live and layout Pallas variants are sealed failures. A future
Pallas proposal must declare a genuinely new association/consumer/reduction/representation/
transport fingerprint and explain why those failures do not apply. “Opaque custom call” alone is
not novelty or numerical proof.

### 3.5 M4 and M5

M4 as written is not admitted. Its error term has no concrete rooted device consumer, and a finite
adversarial interpreter battery is not a symbolic all-input proof. The absence of compensation in
accepted HLO rejects changing primary arithmetic to a compensated algorithm; it does not prove
that every future, source-exact auxiliary physical dependency is impossible. The separately
fingerprinted compensated auxiliary declaration therefore remains unadjudicated, not approved.

M5's rejection was too broad. The declared `auxiliary_device_tuple_dependency` is not the
unconsumed-shadow form. Its auxiliary is a rooted device result and may remain live, although that
can perturb fusion/scheduling. Subsequent work has now bound its exact source, PP16 plan and causal
StableHLO; one candidate-coherent capsule remains missing. Compass itself supplied none of that
authority and still does not prove numerical or optimized-TPU behavior.

## 4. What the report genuinely adds

1. `reduce_precision(e8m7)` is worth retaining as a candidate implementation primitive, not a root-
   cause conclusion.
2. The default excess-precision flag is a concrete environment/codegen variable to record and
   compare in future sealed compile evidence.
3. Optimized-HLO verification remains mandatory because `optimization_barrier` is a scheduling/
   movement boundary, not a physical-layout guarantee.
4. The Compass M4 sketch is not admitted; the distinct compensated auxiliary declaration remains
   unadjudicated pending source-exact causal and coherent authority.
5. The report correctly recommends a bounded layer-0-to-layer-1/event-1 test before a full decoder,
   but only after offline authority and compile-only review exist.

## 5. Historical sequence and current exact next

The original tuple sequence is complete and rejected by coherent replay. The distinct compensated
candidate now has concrete source and candidate-bound PP16 plan/watchpoint authority. Compass M1
remains rejected. The current exact next is a separate source-and-contract batch for compensated
causal StableHLO; no lowering process is authorized by the plan batch. After reviewed persistence
and a separately reviewed forced-CPU lowering, one pinned replayable candidate identity must still
produce all eight coherent watchpoints, including both BF16 RMS operands and their independently
derived FP32 sum. Only successful offline admission and separate review may consider one
compile-only TPU acquisition; only exact optimized TPU HLO may then allow the smallest one-row
layer-0-to-layer-1/event-1 numerical review.

## 6. External claims checked

- XLA currently defaults `xla_allow_excess_precision` to true, but documents it only as permission
  to increase an instruction's output precision:
  <https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc>
- StableHLO `reduce_precision` performs lower-format rounding and returns the original baseline
  tensor type:
  <https://openxla.org/stablehlo/spec#reduce_precision>
- StableHLO `optimization_barrier` is an identity that blocks transformations from moving
  operations across it:
  <https://openxla.org/stablehlo/spec#optimization_barrier>

These generic semantics do not establish what happened in the sealed TPU executable. Local HLO and
runtime evidence must decide that question.

## Bottom line

The Compass report gives us a useful word and primitive for the search, not the missing answer.
Its main proposed implementation violates accepted RMS semantics, and its offline discriminator
requires the exact runtime value that our observability work proved absent. Gate D remains open;
there is no compile or TPU successor from this report.

## Addendum: adjudication of the appended scalar-v2 and A--H reports

The current 611-line Compass artifact, SHA
`d5e4bf8c47fe23d712eb28e1796b4cb1c1569704e1105152cb344826485bc6dd`, appends two later
reports after the M1--M5 analysis. They remain research inputs and authorize no execution.

The corrected scalar-v2 report contributes one admitted result: two finite BF16 RMS operands
determine their single correctly rounded FP32 sum. The separate
`GATE_D_SCALAR_FRONTIER_V2_ADJUDICATION.md` records the limits. Admission v2 now requires every
candidate to seal both operands and byte-match an independently derived 6,144-element FP32 input.
Its T2--T4 tolerance/window proposal is not adopted: the written inversion omits the per-element
norm weight and double rounding at the retained qkv-a boundary, and local replay already rejects
the tested global-scalar successors.

The A--H report contributes useful hardening and diagnostic reminders, not a new Gate-D mechanism:

- exact loaded-library/source hashes, source-to-HLO binding, runtime topology and owner-axis maps
  are valid requirements and are already represented by the fail-closed authority work;
- snapshots capture executable arguments/results, not arbitrary fused intermediates, while an
  isolated submodule can change fusion/layout and cannot be relabelled candidate-coherent state;
- layout, conversion placement and dot algorithms remain possible later diagnostics only after a
  coherent first-divergence capsule exists;
- the proposed dot-precision root cause is downstream of the sealed first mismatch: normalized
  BF16 hidden 2,795 diverges before qkv-a;
- generic CUDA/top-k nondeterminism and margin literature cannot weaken this project's required
  exact DSA selected sets and lowest-position tie order. Local canonical evidence, not an unrelated
  backend's default behavior, defines the oracle;
- formal FP tools may adjudicate a future small, concrete expression, but do not recover missing
  runtime values or an unknown accepted physical tree.

The useful next action remains the smallest one: define and adversarially verify the compensated
causal-StableHLO source/contract without starting JAX. The later capsule must contain all eight
watchpoints, including both BF16 RMS operands and their independently derived FP32 sum, before any
compile-only TPU consideration.
