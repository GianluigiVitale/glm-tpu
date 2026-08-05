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
| L2 MoE combine HLO | DB 417: one `bf16[2,1,6144]` four-rank AR; no other collective or dead `[32,6144]` tensor | Pass on TPU (PP8) |
| One-layer pack format | Exact source leaf set, ownership, byte/hash round trip, corruption refusals | Pass on tiny fixture and real artifact |
| Real layer-3 pack | 1,544 leaves / 9,706,940,416 unique bytes / only shards 38–40; four final PP8 files; manifest `68ef8201...f938`; remote `SUCCESS` | Pass (layout mechanism; not Gate B) |
| Real layer-3 oracle | Raw-source PyTorch; 104 pinned tensors; normal all-slot routes; all-eight-on-slot-2 adversary; manifest `c63ffa19...ebff`; remote `SUCCESS` | Pass (correctness artifact) |
| Exact real PP8 local MoE layer | DB 417: exact routes, bounded normal/concentrated tensors, direct load, HBM, wall, XPlane, DB/archive/cleanup | Pass |
| Exact real PP16 local MoE layer | Same oracle; two-chip ownership/HLO/measurement required | In progress |
| Gate C overall | Sparse PP8 passes; dense, DSA, IndexShare, PP16 sparse representatives remain | In progress |
| Gates B/D–H | Prohibited until earlier gates authorize them | Missing |

Last verified greenfield suite: 79/79. CPU/HLO and synthetic TPU chains prove mechanism only. DB
417 proves one real PP8 sparse layer, not full-model correctness, token latency, or wall throughput.
