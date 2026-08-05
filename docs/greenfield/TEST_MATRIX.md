# Greenfield test and gate matrix

| Level/gate | Current evidence | Status |
|---|---|---|
| L0 immutable geometry/plan | `tests/greenfield/unit`, exact checked-in config, canonical hash/round trip/refusals | Pass |
| L0 synthetic topology/groups | `tests/greenfield/topology`, coordinate/order adversaries, PP8/PP16 all-lane rings | Pass |
| L1 forced CPU collectives/runtime | Seven operation/control variants preserve exact dependent chains and deterministic checksums; size 2/4/8/32 exact-75 smoke | Pass (mechanism only) |
| L2 StableHLO/HLO contract | Parser/linter plus current-JAX optimized-HLO tests; exact groups/counts/pairs/shapes, tuple fusion, full-pod diagnostic separation | Pass (CPU/HLO) |
| Gate A physical inventory/groups | Protected DB 405 and approved archive | Pass |
| Gate A dependent collective floor | Protected CLI implemented; full TPU distributions/archive/DB run not yet captured | Metal missing |
| Gate A PP8 transport | Not implemented | Missing |
| Gate A PP16 transport | Not implemented | Missing |
| Gate A inactive-stage/no-host-dispatch proof | Not implemented | Missing |
| Gate C exact real local MoE layer | Not implemented | Missing |
| Gates B/D–H | Prohibited until earlier gates authorize them | Missing |

Current CPU command:

```bash
/home/gianl/vllm-env/bin/python -m pytest -q \
  tests/greenfield
```

CPU and HLO prove anti-elision, group semantics, and fail-closed contracts only. They do not establish
TPU latency, model correctness, or model performance.
