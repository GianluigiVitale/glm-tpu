# Handoff — four-conversation private release

## Active migration to GLM-5.3

The owner requested release versionglm-5.2, retiring old weights and moving to
GLM-5.3. The private release/tag and exact old-weight deletion are complete.
Continue only from `/home/gianl/glm-run/glm53_migration_20260921/HANDOFF.md` and
[migration status](docs/release/GLM53_MIGRATION.md). Inspect acquisition identity
and completion receipts before acting. The latest owner goal authorizes diagnosis,
fixes and retries; preserve verified progress and never duplicate a live job.
The sections below are the preserved GLM-5.2 release history, not a restriction
on the subsequently authorized model migration. The slash goal stays paused.

The owner approved merging the tested implementation and cleaning main's
presentation. Continue from the private operational record:
`/home/gianl/glm-run/four_conversation_release_20260921/HANDOFF.md`.
The final promotion receipt named in [STATUS](docs/release/STATUS.md) determines
whether main and the commit-bound package have actually been published privately.
Leave the slash goal paused.

## What is included

One shared model supports a fixed group of up to four concurrent conversations
with32K total slots each. Sequential queues still accept ten. Main's public
request preparation rejects more than four concurrent requests before launch;
the historical eight-lane numerical code/tests remain as preserved implementation
coverage, not a supported eight-chat hardware configuration.

Real-weight run `optimized_request_20260921T125148906373Z` at executable
`a252eb01dc8e9fcd45311182ba042caa4addc38c` passed fresh graph/memory checks,
all-rank token agreement, concurrent rounds and all-eight zero exits/idle cleanup.
Three GSM8K answers ended correctly; one hit1024output tokens without a final
answer. Per-active-chat decode4.91–5.19tokens/s; cold load/compile1094.574786s;
prefill3.878273s for317total tokens. Full32K input quality was not tested.
The [receipt](docs/release/four-conversations-20260921.json) binds private originals.

The numerical implementation is unchanged after that run. The final release
changes host admission to four, CLI metadata and documentation only. Reuse
passing numerical checks and rerun affected host/interface/package checks.
The completed8K single-request release and its14.55tokens/s evidence remain
available. Eight-chat failures remain in [CONCURRENT](docs/release/CONCURRENT.md).

## Preserved GLM-5.2 operating scope

The following restrictions applied to the completed GLM-5.2 release, before the
owner authorized the GLM-5.3 migration and recovery in current `goal.md`:
No new TPU run, polling, retry, resources or environment changes. Existing jobs
are terminal and all eight hosts cleaned up. Keep the repository private,
all research branches/history/originals, MODEL_SOURCE, DB616–621 and attribution.
Use apply_patch and JAX_PLATFORMS=cpu for tests. Respect both workload/sync locks
and US-CENTRAL2 backup limits. Self-review is not independent review.

No arbitrary file removal is needed: each retained path has a curation purpose.
Earlier handoff text is exactly recoverable with
`git show f249469a5ff53c14366ba0a9e14b29144d34e636:HANDOFF.md`.
