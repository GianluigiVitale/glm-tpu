# Greenfield performance and mechanism log

No greenfield model-performance measurement exists yet.

## 2026-08-05 — dependent collective benchmark CPU/HLO mechanism

The greenfield benchmark now compiles genuinely dependent control, all-reduce, reduce-scatter,
all-gather, collective-permute, all-to-all, and fused tuple all-reduce chains. Forced-device CPU
lowering preserves exactly three of each requested operation in the cross-operation test, preserves
tuple-result fusion, and produces deterministic bitwise addressable checksums. An exact-75
all-reduce smoke passes independently for group sizes 2/4/8/32. These are anti-elision and HLO
contract results only; CPU timings are discarded and no TPU or model performance claim exists.
The test uses a non-identity device assignment and proves that logical HLO partition ids are mapped
back to physical device ids before accepting replica groups or collective-permute edges.

The protected runner requires 75 dependent operations, 200+ warmups, 1,000+ measured invocations,
full p50/p90/p95/p99 distributions, exact physical groups/counts/pairs, fleet HLO agreement, and
first/last checksum equality. Protected metal evidence is still pending.

## 2026-08-05 — protected topology/local-group proof

Artifact `greenfield_topology_20260805T125842425591441Z`, DB 405, proved the runtime physical
inventory and explicit PP8/PP16 group manifests at pin `75c8bb14`. All eight hosts independently
computed one contract hash, gathered actual local-device order, and passed pre/post zero-work
census. The approved-bucket archive has `SUCCESS`.

Observed topology is `2x4x4`, 32 JAX-visible TPU-v4 chips, eight JAX processes, four chips/process.
TPU-VM worker suffix is not JAX process order. PP8 uses process ring `[0,2,4,6,7,5,3,1]`; PP16 uses
16 adjacent two-chip stages. Every ring boundary has a distinct neighbor for every transfer lane.

This is a topology mechanism proof only. It contains no transport latency, XPlane, HLO collective,
model token, device ms/token, wall tok/s, or performance claim.

Earlier fail-closed diagnostics were preserved locally: launcher variable expansion before any
TPU work; absent worker harness clones; discovery of TPU-VM/JAX rank remapping; discovery that v4
reports `local_hardware_id=None`; and DB 404's correct inventory but non-neighbor stage ordering.
Each metal diagnostic ended with eight clean hosts. No failed attempt is promotion evidence.
