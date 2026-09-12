# Frozen batched long-context launch — 2026-09-12

Status: integration implemented and locally tested; no batched 128K/256K result
yet. This is the current launch index. Earlier phase/helper “next” entries are
historical, not additional work queues. Authority: §25/§26, current-chat self-review.

## Execute the existing workflow

Use `scripts.greenfield.ws32_batched_launch.numerical_environment` with profile
`ws32_delivery_long_phase_v1` and context `128k_d1_0`. Pass its returned environment
to the existing `scripts/greenfield/run_short_decoder_ws32.sh`, holding
`/home/gianl/.glm-tpu-rsync.lock`; the wrapper holds the workload lease.
Commit/push and verify the regional mirror before launch. Freeze execution and
enforcement source until sealing. Observe the original controller/worker handles;
never restart on a tool/SSH observation timeout. Recovery remains in
`WS32_ORPHAN_RECOVERY.md` and `watch_ws32_run.py`.

The same request builder covers depths1.0,0.0,0.05,0.95 and `256k_e0`. Complete
one original run before the next. No serial teacher forcing, new acquisition-only
campaign, throughput tuning, checkpoint copy or cross-engine prose matching.

## What changed

The actual shell entry, worker and sealer now select the same fixed long request
and all fourteen original-RAW/fresh-optimized pins. Existing short source guards
are unchanged. All five actual expanded commands parse through both real CLIs;
L7 uses20 observed tokens, E0 uses256 timed decode steps, fixed capacities and
existing long ceilings (ceilings are not ETA or speed targets).

The sealer replays all42 original WK calls and both exact materializer calls,
checks per-owner all-live budgets, overlay placement, lifetime peak continuity,
phase/graph/checkpoint identities and WK replica hashes. The existing fleet
validator joins all32 physical owners. Original graph texts are re-inspected,
not authorized by stored PASS fields. E0 remains NO_CORRECTNESS_ORACLE; new DB
items are explicitly batched and do not inherit serial coverage or card parity.
Decode preparation output schemas are metadata, NOT independent tensor replay.

Preparation JSON/HLO has pre-write size/disk guards; HistoryCalls already caps
each original before writing. Transport retains48 bounded originals/rank,
136MiB summed file caps and160MiB rank ceiling; atomic JSON replacement temporarily
needs its previous file plus the new file. No full weight/tensor dump is added.
The wrapper reserves10GiB regional whole-prefix allowance (not expected usage),
including traces, controller/DB and failure publication, and requires6GiB local
free space after creating its seal checkout. Historical2K whole archive2.749GB
plus up to1.25GiB phase originals and larger HLO leaves room in this allowance;
actual final bytes must be reported. Trace publication retains existing bounds.

31 exact archived local copies were evicted,2,139,085,384B; free space became
6,718,005,248B. Cloud originals, primary DB, weights, compact evidence and the
DB615 fleet/rank0 graphs remain. Exact restore generations are in
`docs/artifacts/delivery-resume-local-copy-review-20260912.json`; applied receipt
is the adjacent `delivery-resume-local-copy-eviction-20260912.json`.

## Test evidence and limitations

- Actual long entry/phase originals/composed sealer/transport:54PASS16.11s.
- With producer pre-write guards and deferred decode regressions:76PASS18.03s.
- Final five-context storage/CLI assertions, enforcement inventory and reuse:
  22PASS11.35s. These batches overlap; do not sum them as unique tests.
- Wider precommit batch:190PASS,26FAIL,1existing skip,1deselected,229.57s.
  Two shell-fragment fixture failures were corrected and pass above. Fourteen
  old default B17 recipe cases reject the current model tree as intended: direct
  execution of the unchanged HEAD launcher reproduces the identical source
  refusal. Ten source-enforcement tests require the newly declared source to be
  committed at HEAD; rerun those after commit, without weakening enforcement.
- Earlier actual companion RAW reproduction348.16s and retained ten-graph
  inspection are reused; no new model/compiler trial or numerical proof.
- Postcommit at f36a8a39:62PASS/1existing skip/2FAIL322.49s. Both failures were
  stale shell-string assertions for the old literal4GiB guard. Updated only
  those assertions to check the unchanged4GiB default and dynamic threshold;
  focused rerun2PASS1.59s. Source-enforcement checks pass; no production waiver.

Self-review checked phase order, ownership, full observer/timed cardinality,
original-byte publication, parser compatibility, source guards, code release and
storage scope. It is not independent review. Runtime HBM and task outputs remain
the responsibility of the real long run. All four128K, full256K, official HF-card
scores, request/resume/actual TTFT and final delivery remain open.
