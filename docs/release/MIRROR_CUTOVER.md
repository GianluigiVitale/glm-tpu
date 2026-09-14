# Release mirror cutover

**Prepared, not installed or executed.** The active benchmark owns the shared
sync lease. Do not bypass it or change the installed cron/script during its
execution or sealing. Source preparation here performs no cloud writes.

The existing cron calls `/home/gianl/bin/sync-glm.sh` under these nested locks:

```text
/opt/glm-tpu/locks/glm_tpu_rsync.lock
/home/gianl/.glm-tpu-rsync.lock
```

Its four existing source/destination pairs do not explicitly include the release
worktree. The shared Git store may already contain release objects, but that is
not verified backup coverage of the release checkout.

## Exact staged change

`scripts/release/sync_glm_repositories.sh` is a versioned installation template,
not an independently leased launcher. Its only difference from the inspected
installed script is this pair:

```text
/home/gianl/glm-tpu-release → gs://driftbench-dsv4-uc/repos/glm-tpu-release
```

Expected installed predecessor SHA256:
`8303e37d4f3732b66fa896ac0cf8df0903640a00738166838011f1120610cf6b`.

Prepared template SHA256:
`7d8d492593f5d92cfeef0c0e461da26ece9b3d805f659c424d23de5f666510d6`.

A test removes exactly the new pair and verifies the remaining bytes equal the
predecessor hash; Bash syntax also passes. Thus region admission, existing
backups, minimum-file refusal, symlink handling and SQLite sidecar exclusions
are unchanged. No bucket relocation or full checkpoint copy is involved.

## Post-seal installation and verification

1. Confirm original benchmark termination/sealing and authenticated idle owners.
   Obtain both workload/sync leases and the existing cron serialization lock;
   if any is held, wait. Do not replace lock files or kill their owners.
2. Recheck the installed predecessor hash and prepared template hash. If the
   installed script changed meanwhile, reconcile the new diff instead of
   overwriting another change. Install the exact reviewed template at the current
   script path with executable mode. Keep the existing cron schedule/locks.
3. Inspect the source worktrees and exact destinations before mirroring. The
   inherited `rsync -d` removes destination-only files: a wrong/partial source
   is not an acceptable substitute. Keep raw requests/weights out of repo roots.
   Check regional location and live-storage headroom again before publication.
4. Run the installed script under its existing locks while source/DB/Git metadata
   are stable. Do not nest a second independent acquisition of a lock already
   held by the same manual operation. No active SQLite transaction or model
   workflow should overlap this mirror.
5. Verify content, not just a successful log line: use checksum-enabled read-only
   comparison against the exact release prefix and the primary repository's
   shared Git store. Bind the verified local release/main refs, destination
   object generations and content identities in a compact receipt. A skip,
   nonzero sync result, missing prefix or changed file is not mirror completion.
6. After the eligible main merge/push, verify the final refs and mirror again.
   Keep the repository private and preserve all research branches/history.

The mirror retains source and Git history, not the external RAM checkpoint or
live KV state. Recovery of weights uses CHECKPOINTS.md; do not add a hundreds-of-GB
runtime backup to this script. The release worktree is tens of MB, not another
full model copy. No mirror readiness checkbox is closed by this preparation.
