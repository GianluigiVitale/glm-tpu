# Greenfield checkpoint format

## Complete PP8 final-layout artifact

Format version 1 now covers the complete `PP8_LP4` checkpoint as well as the
earlier bounded one-layer artifacts. The complete source inventory pins 141
safetensor files, 118,629 leaves, 755,617,140,416 payload bytes, the model
index/config hashes, and immutable source revision
`gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658`.
Inventory SHA-256 is
`a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4`.

The byte-balanced plan maps every leaf to an explicit final owner. Its eight
base stages produce 32 owner files; MTP produces four more. The plan,
execution, and layout hashes are respectively
`c5bfacc3eeaecce59fb92d10d1a03087b70751a96d13ce6dbd7673270995c4ed`,
`23fd23f8513bea9e5f0c53fafd0b978534a75f60010be43d6a105e0d261f5c89`,
and `aca0eb6d4c498271baf3260dacbe55c9bf935539f42a4b7eaa889947627506a0`.
The layout reconciles 760,215,571,712 packed payload bytes, including all
declared splitting/replication, without an unowned or unexplained byte.

Artifact `greenfield_full_pack_pp8_20260805T182222755355852Z` was written at
code `61475d30a708252bc64aaf50fec2c2ea1ea03c63`. Its 36 files contain
760,215,571,712 payload bytes and 760,231,807,280 total file bytes. Packed
manifest SHA-256 is
`0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1`;
the approved checkpoint prefix has `SUCCESS`.

Protected DB 420 / `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z`
loads all 32 base-decoder owners on all 32 chips directly from that artifact.
It validates 122,640 final tensor shards / 750,122,559,744 payload bytes,
per-file and per-tensor identities, finite values, exact stage/device ownership,
and device round trips. Raw FP8 remains U8 E4M3 bits with local FP32 scales;
the loader performs zero host FP8 dequantizations, host global concatenations,
or runtime checkpoint reshards. Maximum weights-only HBM is 24,840,958,464
bytes/chip, leaving at least 8,173,454,848 of the reported 33,014,413,312 bytes.

The local checksum ledger, SQLite integrity/DB linkage, three 8/8 censuses, and
all 47 byte-identical downloaded archive files independently pass. Missing,
corrupt, stale, duplicate, non-finite, or incomplete inputs are covered by
fail-closed tests. These results satisfy Gate B's checkpoint packing/direct-load
criteria. They do **not** prove full-decoder HBM safety: KV, DSA state,
executables, compiler overlays, and decoder temporaries remain unmeasured, and
the plan correctly records `promotion_memory_proven=false`.

## Bounded Gate C derivative

Artifact `greenfield_gate_c_pack_20260805T214609093206269Z` at code `8a50d6a` derives only the 31
real layer-2/3 dense, full-DSA, and IndexShare leaves from the protected complete PP8 layout. It
copies every placement exactly; no independent ownership recipe is allowed. The subset layout
binds parent `aca0eb6d...6a0`, oracle `54262529...4a9f`, and immutable source revision
`gcs-object-set-830f...9658`. It opens only raw source shards 20, 38, and 40 and validates each raw
tensor SHA-256 against the independent oracle while streaming all four owners together.

The 413,810,816 unique source bytes produce 502,446,080 packed payload bytes because the protected
layout intentionally replicates small stage-local leaves. Each owner payload is 125,611,520 bytes;
the four complete files total 502,461,536 bytes. Subset-layout SHA-256 is
`cdbea04f226f2bd93f3d5ae006ea14b5119e091e29a36da52d783e9d2284c678`; packed manifest is
`3c5c48dadb1b42ec6c99667b79196477ed978cb555dc57dec9e4af38583f2a8a`. Local hashes and the sealed
evidence ledger pass, and the approved-bucket objects have verified sizes, generations, CRC32C,
and `SUCCESS`.

This derivative exists only to make the protected Gate C load bounded; it does not replace the
complete Gate B artifact and has no TPU, HBM, decoder, latency, or throughput claim.

## Bounded one-layer precursors

Before the complete artifact, format version 1 defined a bounded
`greenfield_one_layer_moe` artifact for the mandatory real-layer proof.

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

These artifacts authorize only the one-layer PP8/PP16 proofs. They are not the
evidence used to pass Gate B and cannot substitute for DB 420.

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
bounded PP8 load mechanism only; DB 420 is the complete PP8 load proof.

The corresponding final-layout PP16 artifact is
`greenfield_one_layer_pack_pp16_20260805T172003732526347Z`, manifest
`385737230d593d6b2daa46911d1ac31973d9b79a7d43c3353cedf6d9779fe454`. Its two independently
hashed 4,855,045,080-byte files own experts `0:128`/`128:256` and shared-intermediate
width `0:1024`/`1024:2048`. DB 418 directly loads those two owners with 28 transfers,
12 on-device dequantizations, zero host dequant/global concat, and no runtime repartition. This also
remains bounded mechanism evidence; DB 420, not this layer artifact, satisfies Gate B.

## Fused qkv-a feature-runtime derivative

Production pin `0082bac` defines attention layout
`fused_qkv_a_virtual_tp32_n82_v1`. For every live attention slot it replaces
the four separate q-a/kv-a leaves with:

- `qkv_a.weight_bits`: U8/FP8 `[32,6144,82]`, with each virtual shard ordered
  as 64 q-a outputs followed by 18 kv-a outputs;
- `qkv_a.scale_inv`: FP32 `[32,48,82]`, expanding only the source output-scale
  blocks offline before the same shard-major reorder.

The two derived tensors retain their original physical device owner. Feature
expert redistribution and qkv-a fusion are executed in one bounded streaming
pass from the protected base runtime artifact, so no full intermediate artifact
or decode-time repack exists. Source tensor hashes, both transform names,
derived/padding bytes, output tensor/file hashes, and the nondefault attention
layout are bound by the semantic layout, sidecars, control, manifest, and
loader verifier.

Reconstruction against the real protected manifests yields layout hash
`523afb1d...cb4`, semantic manifest `8bd08068...6f9`, 32 files,
`834,369,271,808` payload bytes, and `26,074,039,744` bytes/chip. These are
pre-pack layout facts. Gate B remains reopened until the 32 payloads are
written, independently verified, archived, directly loaded, and protected by
the full failure/cleanup contract.
