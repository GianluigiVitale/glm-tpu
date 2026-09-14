# Execution boundaries

Start with the [release instructions](../docs/release/INFERENCE.md), not a script
chosen by filename. The site-specific release request path passed TPU admission
in DB621; other historical scripts do not inherit that admission.
Nothing in this directory grants permission to create or manage infrastructure.

| Entry | Role |
|---|---|
| `python -m glm_tpu prepare-request` | Local pinned-tokenizer preparation; zero model execution |
| `python -m scripts.release.launch_ws32_user_request` | Default-off, leased one-user request; original validation, DB and archive sealing |
| `scripts/greenfield/watch_ws32_run.py` | Observe original worker ownership; not restart authority |
| `tools/check_release.py` (repository root) | Offline CPU release checks; not hardware admission |

`release/` is the supported site-specific user-request interface. Its worker, transport,
evidence, database and archive modules are controller components, not independent
ways to bypass leases or source/checkpoint/memory validation.

`greenfield/` also contains historical diagnostics, cold preparation and protected
oracle tools. Some historical modules and receipts are dependencies of current
integrity checks; they cannot be removed by filename or date alone. `analysis/`
contains reusable profiler analysis. Remaining top-level scripts are historical
research/oracle tooling, **not supported release launch
commands**. Some can stop processes or modify environments; do not execute them
as setup or recovery instructions for this engine.

Seven superseded legacy schedulers/provisioning scripts were removed only from
the release tree. Their exact Git recovery locations are in
[the removal ledger](../docs/release/removed-legacy-schedulers.json). Research
continues on preserved branches of this same private repository.

The four external-fork single-chip scripts formerly under `kernel_probe/` were
also removed during curation. They exercised `tpu_inference` kernels rather than
the supported native engine; no current code/config/test entry point consumes
them. Their exact starting-commit/path/blob recovery records are in the
[curation ledger](../docs/curation/disposition.jsonl). Historical review and log
mentions describe those preserved versions, not current runnable commands.

Curation also retires the old benchmark campaign launcher, five legacy Ray
load/cache diagnostic campaigns and the separate upstream PR-submission checker.
Shared ownership/publication checks remain in `release/ws32_host_ops.py`, with
negative/recovery tests. External-fork PR drafts and reconnaissance notes are
research history, not installation instructions; recover them via the same ledger.

The installed repository-mirror cron is outside this directory. It was not
disabled or replaced by cleanup. See [operations](../docs/release/OPERATIONS.md)
for the completed initial release-mirror record and current source-freeze rules.
`release/sync_glm_repositories.sh` is the reviewed installation template for that
cron, not a self-leased command. Its [cutover procedure](../docs/release/MIRROR_CUTOVER.md)
requires the existing locks; do not run it beside an active workflow.
