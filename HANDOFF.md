# Handoff — curated private main

## Performance checkpoint (2026-09-19)

The passing research candidate measures 138.85 prefill and 14.04 decode wall
tok/s at 2K, with all 29 DB610 tokens matching on every host. Main retains its
compact evidence and recovery instructions in [docs/perf](docs/perf/README.md).
This checkpoint changes documentation/evidence only; the supported admitted
inference path remains unchanged.

The owner replaced the broad optimization campaign with native MTP/speculative
decoding on the existing GLM-5.2 setup. Follow the active goal and acquisition
status in `/home/gianl/glm-tpu-perf-ref` on
`perf/reference-lowhanging-fruit-20260919`; do not restart old queued experiments
or infer current fleet status from historical cleanup receipts. Latest preserved
research checkpoint: `a7b1ca1b`. Native MTP has not executed and accepted
speculative throughput remains unmeasured. The two GLM-5.3 repositories are
excluded. Originals and research history are preserved.

## Authority and pins

Objective: [CURATION_PLAN](docs/release/CURATION_PLAN.md), completed on
2026-09-15. Read [AGENTS](AGENTS.md), [STATUS](docs/release/STATUS.md) and
[the curation index](docs/curation/README.md) before changing anything.
Self-review is not independent review; no TPU run was made for curation.

- Starting main `b667f00f1ae48c8ff37e92500550c1395d74c66d`: 1,971 files.
- Curated main (fast-forward of `release/curation-20260914`; code pin
  `cc2b36de`, then documentation/receipts): 616 files; 1,380 originals
  removed with exact recovery rows; zero unresolved dispositions.
- Preserved research: `83f0c2728d0d418255a917343cc89d24b815bd0c`
  (`rewrite/topology-first-decode`). Canonical execution checkout
  `/home/gianl/glm-tpu-topology-rewrite` stays detached at `b667f00f`.
- Frozen `MODEL_SOURCE` pin `edecdd94` and every native StableHLO/source
  identity are unchanged; DB616–621 evidence is historical and untouched.

## What main now contains

Supported engine (`glm_tpu/greenfield/{kernels,runtime,sharding,checkpoint,
partitioning}`, `glm_tpu/cli.py`, `user_request.py`), the user controller and
its host operations (`scripts/release/`), the native preparation/validation
modules the worker executes (`scripts/greenfield/ws32_*`, the short-decoder
runner and sealer), the pinned legacy harness modules (`bench/`), the receipts
retained code pins or reads plus the DB616–621 chain (`docs/artifacts/`), the
release/curation documents, and 217 test modules. Every retained file's role,
consumers and review are in `docs/curation/disposition.jsonl`.

## Verification recorded at the end of curation

- Ledger checker complete (`tools/curation_inventory.py`).
- Release check: passed on 2026-09-16 (pytest step 524 passed, 1 skipped; doctor, content audit, frozen-source pin `edecdd94`, compileall and isolated wheel install all clean).
- Whole retained CPU tree: 2,832 passed, 122 skipped, 173 failed, 0 errors in 1 h 30 min on code pin `cc2b36de`.
- Skips and local-evidence dependencies: [TESTING](docs/release/TESTING.md).

## Known limits to carry forward

- Some test replays need sealed originals that live outside Git on this host;
  several DB609/seven-graph text copies are absent locally and those tests skip.
  Three tests read Git history (`git show`), so a shallow clone cannot run them.
- `bench/provenance.py` and `bench/engine.py` are pinned legacy bytes: the
  sealer's enforcement surface and the oracle loader hash them, so editing them
  changes sealed identities. They are dependencies, not entry points.
- Historical documents (`docs/glm-tpu-revolution.md`, `docs/greenfield/*`)
  are retained as the binding specification and gate records; their curation
  notes mark which named files left main.
- Public distribution still needs an owner licensing decision for original
  code and a renewed provenance/privacy review.

## Safety and persistence

No TPU/model/benchmark/tuning runs, environment upgrades, weight copies or
TPU/node/VM/queued-resource management. Tests ALWAYS `JAX_PLATFORMS=cpu`.
Keep private visibility, originals, research refs and Git history. No force-push.
Only `gs://driftbench-dsv4-uc` (US-CENTRAL2), within the live-storage bound;
respect both workload/sync leases and the installed five-minute mirror
([MIRROR_CUTOVER](docs/release/MIRROR_CUTOVER.md)). The final promotion and
mirror verification receipt lives outside Git under
`results/private_release_20260915/` in that bucket.
