# Handoff — curated private main

## Active MTP/speculation research (2026-09-20)

Start with [goal.md](goal.md) for the current state and
[MTP progress](docs/perf/MTP_PROGRESS_20260919.md) for preserved trials.
Native layer-78 acquisition, guarded MTP drafting/acceptance and real comparison
have now executed on all 32 TPU v4 chips. The first completed long comparison
measured ordinary **14.2902**, one-draft **13.0927**, two-draft **12.7682 wall
tok/s**. Both speculative trails diverge at token index 6; all three responses
exhaust 6,144 tokens during reasoning. This is a slower, non-token-exact result,
not an engine promotion. DB610 matches 29/29 in all modes; the short R3 gain
of 9.4% does not establish long-request performance. All eight hosts were
authenticated idle, and the strict fleet summary passed.
[Completed receipt](docs/perf/tpu-real-native-mtp-20260920T015817Z.json).

The representative suite completed both repeats on prose, code and structured
output, with all eight hosts authenticated idle. Ordinary ranges are 14.27–14.44
wall tok/s; two drafts lose 5.9–6.6% on prose, gain 4.3–4.6% on code and gain
10.7–10.9% on structured output. No case reaches the 25% working target, and
speculative output differs from ordinary. Prose needs scoped technical corrections;
code responses are unfinished; structured values are exact but Markdown fences
fail the requested format. [Completed suite receipt](docs/perf/tpu-real-native-suite-20260920T031727Z.json).
Final eligible evidence publication remains outstanding. Keep ordinary as the
DB610-qualified research baseline. The two GLM-5.3 repositories and older queues
remain excluded; decode D5 stays rejected and bug-affected synthetic speeds
are not correctness-qualified baselines.

The prior documentation checkpoint was merged into private main at `5e9ce605`,
with verified regional backups and eight-host cleanup. It publishes ordinary
performance evidence only, not the experimental engine.
[Publication receipt](docs/perf/perf-checkpoint-promotion-20260920.json).
All work remains outside frozen `MODEL_SOURCE=edecdd94`; originals and historical
rejected experiments remain preserved on their branches and in the progress log.

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

TPU runs are allowed since 2026-09-19 (owner decision): the 32 v4 chips of
`db-v4-64-od` may be used for experiments and runs, one workload at a time under
the workload lock; do not create/delete/resize TPU, VM or queued resources.
pytest still runs with `JAX_PLATFORMS=cpu`. Keep private visibility, originals, research refs and Git history. No force-push.
Only `gs://driftbench-dsv4-uc` (US-CENTRAL2), within the live-storage bound;
respect both workload/sync leases and the installed five-minute mirror
([MIRROR_CUTOVER](docs/release/MIRROR_CUTOVER.md)). The final promotion and
mirror verification receipt lives outside Git under
`results/private_release_20260915/` in that bucket.
