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

The first real artifact is
`greenfield_one_layer_pack_20260805T151828912346032Z`, created at code
`1969d9252a45237f1fa5a1eb0bcd44b6e8e8f3d4` from immutable source revision
`gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658`.
It reconciles 9,706,940,416 unique source bytes to 9,716,380,672 packed payload
bytes; the increase is the declared replication of the router weight and bias.
All four files are 2,429,096,640 bytes. Manifest SHA-256 is
`68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938`.
The approved-bucket artifact has a remote `SUCCESS` marker.

The direct PP8 one-layer loader at protected DB 417 validates every packed file and tensor identity,
maps captured physical slots to the isolated four-chip runtime subcube, transfers 56 final-owner
arrays, and performs 24 FP8 table/scale conversions on device. It records zero host FP8
dequantizations, zero host global concatenations, and peak host/device memory. This validates the
bounded PP8 load mechanism only; the complete plan-aware Gate-B format remains pending.
