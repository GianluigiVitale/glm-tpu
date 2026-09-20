# Handoff — ordinary implementation release

The owner's final acceptance criterion is one correctly completed GSM8K answer.
It passed at executable9469cd733df7fabe7f6f421ab0fad60801cf3138:
expected18, returned18, EOS265tokens, fresh graph/memory checks, all-rank
agreement, eight zero worker exit codes and authenticated idle cleanup.
See [current status](docs/release/STATUS.md), [answer receipt](docs/release/single-answer-20260920.json)
and [integration history](docs/perf/ordinary-release-20260920.md).

Measured decode14.547542tokens/s; prefill0.937823s for92tokens;
cold load/compile1102.028924s. These are separate scopes. No broad accuracy,
ten-answer pass or full128K prompt pass is claimed.

The actual implementation is on release/optimized-ordinary-20260920.
Final private-main and archive identity are authoritative only in the verified
publication receipt:
`/home/gianl/glm-run/gsm8k_acceptance_20260920/publication/final-promotion.json`.
Current operational handoff:
`/home/gianl/glm-run/gsm8k_acceptance_20260920/HANDOFF.md`.

Finish any remaining final checks, backup and promotion from that handoff;
do not launch further question runs or resume the cancelled difficult queue.
The slash goal stays paused. Passing CPU/numerical checks are reused; no source
code changed after the successful one-answer run. Final docs/package checks
are separate. Review is self-review, not independent review.

The following freeze/publication record describes the preceding milestone.

## Performance research frozen by the owner — 2026-09-20

**Do not resume TPU experiments or old queues.** The owner requested stopping,
consolidating the research history, clearing obsolete logs and freezing the
result. [goal.md](goal.md) records this instruction and supersedes historical
pending-work sections. Current closure work is documentation, verified archive
cleanup and eligible publication only.

The final answer suite `perf_real_ordinary_suite_20260920T152417Z` was stopped
by authenticated worker identities; all eight hosts were idle at 15:43:39 UTC.
No answer case completed. The native synchronization control was never launched.
Keep both classified as cancelled/unmeasured, not failed speed trials.

Ordinary research decode remains about 14.3 wall tok/s; the qualified short
request measured 138.85 prompt tok/s and 14.04 decode wall tok/s. Long MTP
outputs diverged. Decode D5, fused reductions and public fused EP remain rejected;
no new serving speed improvement was qualified by the upstream adaptations.

Read the [frozen research record](docs/perf/frozen-20260920/RESULTS_AND_DECISIONS.md)
and [operations record](docs/perf/frozen-20260920/FREEZE_OPERATIONS.md).
The [historical notebooks](docs/perf/frozen-20260920/history/README.md)
preserve the detailed experiments. Earlier pending instructions in them are
historical, not execution authority. Pre-cleanup code/evidence is preserved at
`9dedce4b` on `preserve/glm52-tpu-research-freeze-20260920`.
Frozen `MODEL_SOURCE=edecdd94` and protected DB616–621 remain unchanged.

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

The owner stopped TPU experiments on 2026-09-20. No model run or old queue
may resume under this freeze. pytest stays `JAX_PLATFORMS=cpu`. Preserve
private visibility, originals, research refs and history; no force-push.
Only `gs://driftbench-dsv4-uc` (US-CENTRAL2), within the live-storage bound;
respect both workload/sync leases and the installed five-minute mirror
([MIRROR_CUTOVER](docs/release/MIRROR_CUTOVER.md)). The final promotion and
mirror verification receipt lives outside Git under
`results/private_release_20260915/` in that bucket.
