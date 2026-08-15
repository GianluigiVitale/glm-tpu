# WS32_2D prototype

This document records the first independent GLM implementation of the mandatory all-chip
weight-stationary challenger. It is subordinate to `../glm-tpu-revolution.md` and makes no Gate-D,
TPU-performance, final-checkpoint, or HBM claim.

## Reused evidence

The implementation starts from existing work rather than rebuilding the idea:

- `fce8d6c41` supplies the reciprocal expert/feature layout pattern;
- `dab2db7b3` and `6baf66e2a` supply the FP32-partial rule for 2D reductions;
- `57adb4b99` and `2baf3f0a0` are quantized 2D matmul references.

These pins are design inputs only. No legacy or vLLM execution path is imported.

## Exact prototype contract

The logical mesh is `expert=8 x feature=4`, derived from physical TPU coordinates as
`expert=(x,y)` and `feature=z` on the observed `2x4x4` slice. Repeated feature groups are eight
physical rows of four chips. Repeated expert groups are four physical columns of eight chips.
No repeated group contains 32 chips.

The batch-one residual is globally `[1,6144]`, locally `[1,1536]`, sharded on `feature`, and
replicated on `expert`. It stays in that ownership across a block:

| Operation | Local weight/output ownership | Repeated reduction |
|---|---|---|
| dense gate/up | `[1536,1536]` on expert x feature | FP32 over feature-4 |
| dense down | reciprocal `[1536,1536]` | FP32 over expert-8 |
| routed gate/up | 32 expert identities x `[2048,1536]` | FP32 over feature-4 |
| routed down | 32 expert identities x `[1536,2048]` | FP32 over expert-8 |
| shared expert | feature-sharded and explicitly 8x replicated | FP32 over feature-4 |

The routed body consumes one exact compact top-8 route row. Only the owning expert row computes a
routed expert. The combine returns the same local `[1,1536]` residual shard; it never creates a
physical `[32,6144]` activation or 32 decode rows.

## Capacity accounting

The prototype applies explicit ownership rules to the SHA-pinned source inventory
`a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4`:

| Quantity | Exact value |
|---|---:|
| base source tensors | 117,060 |
| base source bytes | 745,584,507,456 |
| MLP source bytes | 728,700,174,336 |
| shared-expert replication overhead | 19,822,924,800 |
| compact-router replication overhead | 230,400 |
| non-MLP replication overhead | 2,063,457,216 |
| WS32 packed persistent bytes | 767,471,119,872 |
| persistent bytes per chip | 23,983,472,496 |
| FP8-scale bytes per chip | 5,824,752 |

Every chip is exactly balanced in this accounting. The shared expert's 8x replication is explicit,
not silent. The report covers base-model persistent tensors, but it is not Gate B for WS32: it does
not yet define every byte interval, checksum, destination file, or direct-load record. It also does
not include target-context KV/DSA state, compiler overlays, collective buffers, or measured peak
HBM.

## Local proof completed

Forced 32-device CPU tests prove:

- the physical/logical mesh and subgroup families;
- dense and routed/shared MoE results against independent unsharded references;
- output sharding remains `P(None, 'feature')`;
- dense uses one feature-4 and one expert-8 reduction;
- the top-k-4 test MoE uses only subgroup reductions, with no group larger than eight;
- async, wrong-group, wrong-shape, and wrong-reducer HLO mutations refuse;
- all 117,060 base tensors have one explicit capacity rule and exact byte reconciliation.

CPU and synthetic HLO results prove semantics only. The current subgroup linter does not claim the
complete packed-bits-to-live-root arithmetic lineage required for a protected real-layer result.

## Bounded one-layer derivative

`checkpoint/ws32_one_layer.py` reuses the sealed PP8 layer-3 artifact rather than rereading the
753B checkpoint. Its source manifest is independently content-addressed and its four 2.4-GB files
already bind all 1,544 source leaves. The derivative:

- splits each pair of PP8 expert halves into two WS32 expert rows;
- splits every routed hidden input/output dimension over four feature columns;
- reconstructs the shared expert once from its four PP8 pieces and explicitly writes eight
  expert-row replicas;
- shards router weight over expert and feature while replicating only correction bias over feature;
- stores FP8 payloads as exact U8 bits for direct JAX placement;
- writes 32 append-only final-owner files with tensor/file/manifest SHA-256 records;
- commits the manifest last and provides a one-slot direct loader that refuses non-finite FP8 bits,
  non-finite scales, checksum drift and wrong manifest identity.

Tiny sealed-artifact tests reconstruct every routed expert and shared expert from the 32 derivative
files back to its exact source bytes. Protected offline artifact
`greenfield_ws32_one_layer_pack_20260815T070628458699950Z` realizes the plan at code `4c749a6`:
9,971,249,152 packed bytes, 311,601,536 per chip, from 9,706,940,416 unique source bytes plus
explicit shared/bias replication. Manifest `4bf8679d...1f40`, direct loads for slots 0 and 31,
local/remote CRC32C for 39 nonterminal objects, and terminal object-40 `SUCCESS` pass. The temporary
pack was removed and JAX/TPU was never initialized, so this is protected checkpoint evidence—not a
real-layer HLO, HBM, correctness, Gate-D or performance result.

## Bounded next discriminator

Do not run a full decoder or hour-scale 8K workflow. The bounded real MoE layer now has protected
compile evidence. Tag `greenfield_ws32_real_layer_hlo_20260815T081128291986777Z` loaded all 32
final owners and compiled on all eight hosts. Fleet-identical StableHLO/optimized-HLO SHAs are
`dea1384e...c01f` / `6a6acf94...e157`. The graph proves 15/15 live inputs, ten live F32-operand
reductions, nine feature-4 groups, one expert-8 group, maximum group eight and no full-pod hidden
reconstruction. Compiled argument/temp/code/output memory is
311,753,728/34,843,648/23,690,240/6,144 bytes per chip; measured post-load peak is 311,859,200
bytes. XLA schedules BF16 results by fusing post-reduction conversions, but every operand and exact
scalar-add reducer remains F32.

The pinned numerical retry is now sealed as
`greenfield_ws32_real_layer_numerical_20260815T090957477700205Z` at code `d3c3427`. Both normal
and all-eight-on-one-owner cases pass the bounded oracle contract. Their maximum absolute errors
are `0.015625` and `0.03125`; bitwise mismatch totals across 32 replicated shards are 31,632 and
31,728. The graph retains the exact ten local reductions and all 15 live inputs. Maximum measured
peak is 335,805,440 bytes/chip and the smallest largest-free block is 32,648,534,016 bytes/chip.
The 51-object archive was sealed with `SUCCESS` last and authenticated 8/8 cleanup.

This proves the WS32 ownership, direct load, numerical contract, subgroup locality and ample layer
HBM, but rejects the readable whole-matrix-dequant implementation as an execution candidate:
diagnostic-only normal/concentrated p50 is `516.5821635` / `1160.885836` ms, compared with the
protected PP8/PP16 real-layer sub-millisecond results. These timings are not token-throughput or
production performance claims.

The distinct default-off raw-FP8 Pallas body is now protected and selected. Compile acquisition
`greenfield_ws32_pallas_real_layer_hlo_20260815T094203579842098Z` pinned StableHLO/optimized HLO
`fa11961d...6118` / `17ee208a...ff5d` and live closure `0a03b130...6a51`. The protected numerical
run `greenfield_ws32_pallas_real_layer_numerical_20260815T163210555760825Z` at code `12e92cf`
reproduced those pins with all 15 inputs, 27 live Pallas calls and exactly nine feature-4 plus one
expert-8 reduction. Normal and all-eight-on-one-owner cases both have maximum absolute error
`0.03125`; p50 is `1.256975` / `2.303685` ms. Peak measured HBM is 314,218,496 bytes/chip, the
51-object `SUCCESS`-last archive passes, and pre/post censuses are 8/8 clean.

This removes whole-matrix dequantization by roughly 411x/504x at this layer and selects WS32 Pallas
for the complete short decoder. The timing remains a layer diagnostic, not token throughput. Gate D
still requires protected complete-decoder 2K/8K exactness, HLO, HBM, integrity and wall evidence.
An unchanged PP8 8K rerun and another M1/M32 arithmetic arm remain forbidden.
