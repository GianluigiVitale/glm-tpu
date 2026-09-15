# Supported execution boundaries

Start with [inference](../docs/release/INFERENCE.md), not a script selected by
filename. DB621 admitted one site-specific release request; it does not admit
every historical script. Nothing here authorizes infrastructure management.

| Entry | Role |
|---|---|
| `python -m glm_tpu prepare-request` | Local pinned-tokenizer request preparation, no model execution |
| `python -m scripts.release.launch_ws32_user_request` | Default-off, leased single request with original validation/DB/archive sealing |
| `scripts/greenfield/watch_ws32_run.py` | Observe original worker identity; never restart authority |
| `scripts/greenfield/run_ws32_runtime_checkpoint_shm_pack.sh` | Guarded RAM-only reconstruction when genuinely absent; follow the checkpoint guide, do not execute for cleanup |
| `tools/check_release.py` (repository root) | Offline CPU/source/package checks, not hardware admission |

`release/` contains the supported controller, worker, transport, evidence,
database and archive components. They are not alternate entry points for
bypassing leases, source identity, checkpoint integrity or memory admission.
Shared host protections live in `release/ws32_host_ops.py`.

`greenfield/` still includes historical diagnostics and oracle capture tools.
Some code/receipts are current integrity dependencies; their remaining curation
is not complete. `analysis/` contains profiler readers and scoped guard tests.

Three top-level legacy helpers remain: `disk_watchdog.sh`,
`launch_glm_32chip.sh` and `validate_ray_network.sh`. They are SHA-pinned or
required by the accepted-DB485 compile-only acquisition and by the legacy oracle
provenance, not native setup instructions.

Old Ray campaigns, fork-sync/triage helpers, unpinned size-only HF staging,
golden-manifest experiments and the full-bundle backup helper have left the
candidate tree. The installed same-region mirror uses none of those files.
Exact removed originals are recoverable through the [curation ledger](../docs/curation/README.md);
pre-curation removals have a [separate original ledger](../docs/release/removed-legacy-schedulers.json).

The installed five-minute mirror is external to this directory and unchanged.
`release/sync_glm_repositories.sh` is its installation template, not a self-leased
command. Follow [mirror cutover](../docs/release/MIRROR_CUTOVER.md) and the original
locks; do not run it beside an active workflow or disable essential backups.
