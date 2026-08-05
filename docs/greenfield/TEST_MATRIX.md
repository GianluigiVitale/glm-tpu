# Greenfield test and gate matrix

| Level/gate | Current evidence | Status |
|---|---|---|
| L0 immutable geometry/plan | `tests/greenfield/unit`, exact checked-in config, canonical hash/round trip/refusals | Pass |
| L0 synthetic topology/groups | `tests/greenfield/topology`, coordinate/order adversaries, PP8/PP16 all-lane rings | Pass |
| L1 forced CPU collectives/runtime | Not implemented | Missing |
| L2 StableHLO/HLO contract | Not implemented | Missing |
| Gate A physical inventory/groups | Protected DB 405 and approved archive | Pass |
| Gate A dependent collective floor | Not implemented | Missing |
| Gate A PP8 transport | Not implemented | Missing |
| Gate A PP16 transport | Not implemented | Missing |
| Gate A inactive-stage/no-host-dispatch proof | Not implemented | Missing |
| Gate C exact real local MoE layer | Not implemented | Missing |
| Gates B/D–H | Prohibited until earlier gates authorize them | Missing |

Current CPU command:

```bash
/home/gianl/vllm-env/bin/python -m pytest -q \
  tests/greenfield/unit tests/greenfield/topology
```

CPU, synthetic, and topology evidence do not establish model correctness or performance.
