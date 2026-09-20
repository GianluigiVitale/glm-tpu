# Owner-requested stop and research freeze — 2026-09-20

## Scope

The owner explicitly ended further optimization and requested a detailed
Markdown history plus cleanup of failed/superseded logs. This supersedes the
previous benchmark completion objective. No pending benchmark is restarted to
finish the record. Model code, weights, original private outputs and protected
DB616–621 evidence are preserved.

## Authenticated stop

Run: `perf_real_ordinary_suite_20260920T152417Z`.
Immutable worker source: `38a34b4d0e8315cbbb971a6519c5fd19a78e06e0`.
The stop script checked each hostname, worker PID/start tick, command line,
run path and source pin, then signalled that process through a PID file descriptor.
It sent SIGTERM, waited 30 seconds, then SIGKILL to each still-live worker.
No broad process-name kill or infrastructure change was used.

All eight workers stopped. The controller completed its own authenticated
all-host idle checks and receipt collection at **15:43:39 UTC**, then exited.
Its eight exit codes are 255, reflecting owner termination; they are not eight
independent model failures. No answer case completed. The stop receipt is
`owner_cancellation.json` in the preserved private run directory.

The native host/blocking, host/unprofiled and device/unprofiled control matrix
was prepared and CPU-tested but never launched. No subsequent TPU work is queued
by this freeze. Its source remains in the preserved research branch.

## Preservation boundary

Pre-cleanup commit: `9dedce4b`, pushed as
`preserve/glm52-tpu-research-freeze-20260920` in the existing private repository.
Original document and receipt recovery is indexed in [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md).
Eleven historical Markdown notebooks are retained separately from the concise
current result. Their old plans and live-status statements are explicitly historical.

The local archive staging directory is
`/home/gianl/glm-run/perf_freeze_archive_20260920T154624Z`.
It contains 1,397 raw logs (58,835,211 bytes before compression), a per-file
SHA-256 manifest, the original documentation/receipts, and cancellation evidence.
The log tar was read back locally and every member hash checked before cleanup.
No weights, checkpoint copies, prompt text or private output payloads enter Git.
Archive upload, deletion counts, final checks and publication are recorded below
only after verification; this paragraph is not proof that those steps completed.

## Verified regional archive

Location: **US-CENTRAL2**. Live bucket size before upload: **2,078,191,388,749 bytes**, below the 2,500,000,000,000-byte bound including these small archives. Each object was read back at its exact generation and compared byte-for-byte; no weights were copied.

| Object | Generation | Bytes | SHA-256 |
|---|---|---:|---|
| `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/perf_freeze_archive_20260920T154624Z/archive-manifest.json` | `1789919583452690` | 763 | `366ca24169ae2b07eab790f174855df8e4329b4051489db2c13ff1601d37475b` |
| `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/perf_freeze_archive_20260920T154624Z/cancellation-evidence.tar.gz` | `1789919590158221` | 67377 | `595693104665ce8d147a74317f736cc41c1ff6ae4eb47cf3fa6f71bd6057b4cf` |
| `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/perf_freeze_archive_20260920T154624Z/raw-campaign-logs.tar.gz` | `1789919597041089` | 4589959 | `96b6a3a06a87e48c1ec513ea1916b5dac62d90002f7a3858f8a8ab9b94b42a7a` |
| `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/perf_freeze_archive_20260920T154624Z/raw-log-manifest.json` | `1789919603758039` | 273109 | `6e017e49ff321a09bf77f38a15aefedec5a8427e0f87a55ce84d20a7d324df8a` |
| `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/perf_freeze_archive_20260920T154624Z/research-documents-9dedce4b.tar.gz` | `1789919610462360` | 475201 | `05a5ba3c4031db1aee0b55828bd83484bffc95f1943001e42986a9fab8e6a618` |

103 compact JSON receipts were removed from the active documentation tree after archive verification and executable-reference checks. The source-audit JSON read by the acquisition tool remains. Eleven detailed Markdown notebooks moved to the historical folder with freeze notices and repaired links. Four Python documentation references were updated only where the executable AST remained unchanged.

## Completed raw-log cleanup

The cleanup held workload, pod and both sync/cron leases. It authenticated
all eight hosts idle before and after deletion. Only regular `.log` files
whose current size and SHA-256 matched the verified archive manifest were
removed; changed files would have been retained.

**1,397 working log files were removed on host 0.** The same manifest paths
were already absent on hosts 1–7; no mismatched file was deleted. Weights,
raw databases, private prompts/token outputs, source snapshots and HLO evidence
were not deleted. The raw logs remain recoverable from the generation-bound
archive above. No workload was launched by cleanup.

Cleanup evidence SHA-256: `7a157afde23b45e0ee1172e5bb60016246aa01b8c5a9b8b16d8952e6947e4b14`.
Private manifest: `raw-log-cleanup.json` in the archive staging directory.

## Checks and publication boundary

The cleaned perf tree passed `tools/curation_inventory.py`: 751 tracked files,
2,131 dispositions, zero unresolved entries and zero errors. All 20 current
research/navigation Markdown files had no missing local file links. Reference-only
Python edits preserve the executable AST; frozen `MODEL_SOURCE=edecdd94` is unchanged.

`tools/check_release.py` passed: **524 tests passed, one skipped, two warnings**,
plus frozen-source, content, compile and isolated package checks. Review is
assistant self-review, not independent review. No new model quality, hardware
speed or experimental-engine deployment is claimed by these checks.

Main publication is documentation-only: current result, measured history,
archive/recovery metadata and freeze notices. Experimental code remains on the
perf/research branches. The eligible candidate must pass its own release checks
and regional source/Git backup verification before main is updated.

The authoritative final publication receipt is outside Git to avoid embedding
its own future commit hash:
`gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/final-promotion.json`.
Only that completed record establishes the actual final refs and backup state;
a prepared document or archive upload alone does not establish main publication.
