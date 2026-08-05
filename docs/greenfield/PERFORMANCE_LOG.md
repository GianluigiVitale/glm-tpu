# Greenfield performance and mechanism log

No greenfield model-performance measurement exists yet.

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
