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

Do not run a full decoder or hour-scale 8K workflow. The next WS32 step is one real layer-0 dense
or one real MoE layer using already packed source leaves, with:

1. load the sealed final-owner derivative directly on its exact physical slots;
2. persist generated StableHLO/optimized HLO before execution;
3. prove exact packed-bit/scale-to-dequant-to-dot-to-live-root lineage and subgroup bijection;
4. retain one live row, no full hidden reconstruction, and only feature-4/expert-8 reductions;
5. compare against the existing sealed layer-3 oracle;
6. record exact topology, HBM, archive, and authenticated cleanup evidence.

Only that bounded result decides whether WS32 advances toward a full decoder. An unchanged PP8 8K
rerun and another M1/M32 arithmetic arm are both forbidden by the Gate-D closure evidence.
