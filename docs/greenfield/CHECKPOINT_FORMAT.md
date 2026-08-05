# Greenfield checkpoint format

The full Gate-B format remains pending. Before the first complete checkpoint,
format version 1 defines a bounded `greenfield_one_layer_moe` artifact for the
mandatory real-layer proof.

The source is never loaded as a model. For sparse layer 3, the packer validates
exactly 1,544 source leaves across safetensor shards 38–40, including shape,
dtype, finite values, source byte count, index SHA-256, and source revision.
No other checkpoint shard is opened.

The artifact contains one file per PP8 stage device. Each owns 64 complete
routed experts, one 512-column/row shard of the shared expert intermediate,
and a replicated copy of the small router weight and correction bias. Routed
weights remain FP8 E4M3 with their FP32 128x128 inverse scales; dequantization
to BF16 occurs only after final device ownership.

`manifest.json` is committed last and records every source leaf, every packed
tensor, ownership interval, source names, shape, dtype, payload bytes, file
bytes, file SHA-256, model/layer/plan/topology/group/code identifiers, and a
canonical manifest SHA-256. The inspector verifies hashes and safetensor
metadata without loading payloads. Destinations are append-only and existing
paths are refused.

This artifact authorizes only the one-layer PP8/PP16 proofs. It is not Gate B
and cannot be reused as evidence that the full 753B checkpoint packs or loads.
