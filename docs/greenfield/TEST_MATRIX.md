# Greenfield test and gate matrix

| Level/gate | Current evidence | Status |
|---|---|---|
| L0 immutable geometry/plan | Canonical hashes/round trips/refusals | Pass |
| L0 synthetic topology/groups | Coordinate/order adversaries; PP8/PP16 physical rings | Pass |
| L1 forced CPU collectives/runtime | Seven operation/control variants; sizes 2/4/8/32; deterministic checksums | Pass (mechanism) |
| L2 optimized-HLO contract | Exact physical groups/counts/pairs/shapes, async op handling, tuple fusion, nonidentity device assignment | Pass |
| Gate A physical inventory/groups | Protected topology DB 405 and approved archive | Pass |
| Gate A dependent collective floor | DB 406–414: dominant/bf16/f32/metadata payloads; exact chain75 HLO and full distributions | Pass (mechanism) |
| Reduce-scatter support | Three protected diagnostics prove TPU-v4 XLA rewrites required small decode RS to AR | Evidence-rejected for this benchmark shape |
| Gate A PP8 transport | DB 415 exact 8-hop HLO/distributions; DB 416 8-file/64-core trace | Pass (mechanism) |
| Gate A PP16 transport | DB 415 exact 16-hop HLO/distributions; DB 416 8-file/64-core trace | Pass (mechanism) |
| Gate A inactive-stage/no-host-dispatch proof | One global device program; no model compute; HLO/trace only p2p permutes | Pass (skeleton) |
| Gate A overall | Topology, collective floor, PP8/PP16 transport, trace, provenance, cleanup | Pass |
| L0 MoE numerical contract | Block dequant, sigmoid/noaux_tc bias semantics, ties, normalization | Pass |
| L1 four-device PP8 MoE reference | Distributed routes + all-top-4-on-one-chip; bounded output equivalence | Pass (synthetic) |
| L2 MoE combine HLO | DB 417/418: one `bf16[2,1,6144]` exact four-/two-rank AR; no other collective or dead `[32,6144]` tensor | Pass on TPU (PP8/PP16) |
| One-layer pack format | Exact source leaf set, ownership, byte/hash round trip, corruption refusals | Pass on tiny fixture and real artifact |
| Real layer-3 pack | 1,544 leaves / 9,706,940,416 unique bytes / only shards 38–40; four final PP8 files; manifest `68ef8201...f938`; remote `SUCCESS` | Pass (layout mechanism; not Gate B) |
| Real layer-3 oracle | Raw-source PyTorch; 104 pinned tensors; normal all-slot routes; all-eight-on-slot-2 adversary; manifest `c63ffa19...ebff`; remote `SUCCESS` | Pass (correctness artifact) |
| Exact real PP8 local MoE layer | DB 417: exact routes, bounded normal/concentrated tensors, direct load, HBM, wall, XPlane, DB/archive/cleanup | Pass |
| Real layer-3 PP16 pack | Two final-owner files; 128 experts + shared width 1024/chip; manifest `38573723...e454`; remote `SUCCESS` | Pass (layout mechanism; not Gate B) |
| Exact real PP16 local MoE layer | DB 418: exact routes, bounded tensors, two-rank HLO, HBM, wall, XPlane, DB/archive/cleanup | Pass |
| Complete PP8 plan/layout/pack | 118,629 source leaves; 32 base + 4 MTP owners; exact byte/hash reconciliation; manifest `08694931...78f1` | Pass |
| Complete PP8 direct load | DB 420: all 32 chips / 8 stages, 750,122,559,744 bytes, exact identities, device round trip, peak weights-only HBM, archive/cleanup | Pass |
| Gate B | Complete plan-aware checkpoint manifest/packer/direct loader and fail-closed corruption handling | Pass |
| Gate C reference kernels | Dense/norm/RoPE, FP32 DSA/top-k, compact IndexShare, stage-local KV, sparse MLA and LSE merge; full-width FP32/BF16 legacy comparisons exact | Pass on CPU/reference |
| Real Gate C oracle | 31 raw layer-2/3 tensors; 2,304-token full scorer; 2,048 exact positions; 8,192-byte IndexShare; dense/sparse-attention outputs; manifest `54262529...4a9f` | Pass (correctness artifact) |
| Gate C overall | Sparse PP8/PP16 protected prerequisite, ordered references, and independent oracle exist; protected real dense/full-DSA/IndexShare TPU proof remains | In progress |
| Gates D–H | Await earlier gates | Missing |

Last verified greenfield suite: 173/173 on forced CPU. CPU/HLO reference tests prove
semantics/mechanisms only.
DB 417/418 prove one real sparse layer under both required local plans; DB 420 proves complete
checkpoint integrity/direct loading. None proves full-model correctness, decoder HBM, token latency,
or wall throughput.
