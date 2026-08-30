# Gate D scalar-frontier v2 adjudication

**Status:** useful observability correction; not mechanism authority.
**Authorization:** no JAX lowering, compilation, model load, cloud workflow or TPU run.

## Verdict

The corrected scalar-frontier report contributes one durable result: the FP32 input to layer-1
RMSNorm is deterministically derivable from the two finite BF16 program operands. BF16-to-FP32
conversion is exact and the single binary32 addition has one correctly rounded result. This is a
derived authority class, not captured accepted bytes.

DB550 independently seals the accepted layer-1 dense update and residual BF16 rows. Therefore the
accepted FP32 sum can be reconstructed offline without reusing the rejected callback executable.
For a new candidate, the same test is decisive only if that candidate preserves both of its own
BF16 operands under the same source/plan/StableHLO/coherence identity. DB518 did not retain those
candidate operands or its RMS FP32 input, so the report cannot retrospectively repair DB518.

## What is adopted

Precompile admission v2 now requires both `layer1.rms_operands_bf16` rows in addition to
`layer1.rms_input_fp32`. The stdlib-only capsule validator derives all 6,144 FP32 sums and requires
byte equality with the declared RMS input. It rejects non-finite operands, extra storage/dead owner
rows, unreferenced arrays, boolean/forged owners and cross-authority reuse. PP16_LP2 and PP8_LP4
have separate exact two-/four-owner schemas.

This converts the report's T1 from an inference into a fail-closed admission invariant:

```text
candidate hidden_update BF16 + candidate residual BF16
    -> independently derived FP32 sum
    -> exact equality with candidate layer1.rms_input_fp32
```

If a future real candidate differs from the independently derived accepted sum, the first causal
frontier is upstream. If the sums match, investigation may proceed to the RMS reduction and
weighted-output consumer using only candidate-coherent state.

## What is not adopted

The report's T2--T4 scale-window formula is incomplete for the retained accepted boundary. The
qkv-a consumer receives the weighted, double-rounded BF16 result:

```text
BF16(BF16(summed_fp32 * inverse) * norm_weight_bf16)
```

A preimage computed as only `BF16(summed_fp32 * s)` omits the per-element norm weight and cannot be
used as the local oracle. Any corrected inversion must include both roundings and every sealed
weight element.

More importantly, existing sealed replay already found no global-scalar solution for accepted SHA
`9936ee1e...d3039` under either tested one-row weighted formula. It reproduces the readable/Pallas
and rejected TPU rows, then identifies weighted-output schedule/geometry—not an ordinary variance
tree—as the unresolved physical distinction. Scalar-tree, gather-before-weight, output-ownership,
direct Pallas and unchanged-run families remain tombstoned. Synthetic Gaussian magnitude estimates
do not supersede this local evidence.

## Exact next

Finish and adversarially review admission v2. A future source-bound candidate must provide the two
BF16 operands, derived FP32 sum, weighted output and complete candidate-coherent query/head/current-
key/cache/scorer history. Only after all offline gates pass may a separate review consider one
compile-only optimized-HLO acquisition. Gate D remains open.
