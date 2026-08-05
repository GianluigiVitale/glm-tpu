# Greenfield test and gate matrix

| Level/gate | Current evidence | Status |
|---|---|---|
| L0 immutable geometry/plan | Canonical hashes/round trips/refusals | Pass |
| L0 synthetic topology/groups | Coordinate/order adversaries; PP8/PP16 physical rings | Pass |
| L1 forced CPU collectives/runtime | Seven operation/control variants; sizes 2/4/8/32; deterministic checksums | Pass (mechanism) |
| L2 optimized-HLO contract | Exact physical groups/counts/pairs/shapes, async op handling, tuple fusion, nonidentity device assignment | Pass |
| Gate A physical inventory/groups | Protected topology DB 405 and approved archive | Pass |
| Gate A dependent collective floor | DB 406–412: dominant payload plus `bf16[1,6144]`/`bf16[1,2048]`; exact chain75 HLO and full distributions | Partial: f32/metadata left |
| Reduce-scatter support | Three protected diagnostics prove TPU-v4 XLA rewrites required small decode RS to AR | Evidence-rejected for this benchmark shape |
| Gate A PP8 transport | Not implemented | Missing |
| Gate A PP16 transport | Not implemented | Missing |
| Gate A inactive-stage/no-host-dispatch proof | Not implemented | Missing |
| Gate C exact real local MoE layer | Not implemented | Missing |
| Gates B/D–H | Prohibited until earlier gates authorize them | Missing |

Last verified greenfield suite: 49/49. CPU/HLO and synthetic TPU chains prove mechanism only; they
do not establish model correctness, token latency, or wall throughput.
