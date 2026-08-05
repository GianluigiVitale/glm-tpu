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
| Gate C exact real local MoE layer | Not implemented | Missing |
| Gates B/D–H | Prohibited until earlier gates authorize them | Missing |

Last verified greenfield suite: 54/54. CPU/HLO and synthetic TPU chains prove mechanism only; they
do not establish model correctness, token latency, or wall throughput.
