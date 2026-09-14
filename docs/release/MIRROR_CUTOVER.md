# Release mirror cutover

The release pair and checksum comparison were installed on2026-09-14 after
benchmark cancellation and DB621 sealing, under both leases and the cron lock.
The schedule and original backup destinations are unchanged. Final verification
is recorded in `results/private_release_20260914/final-promotion.json` in the
approved bucket; a prepared template or successful sync log is not that proof.

The existing cron calls `/home/gianl/bin/sync-glm.sh` under these nested locks:

```text
/opt/glm-tpu/locks/glm_tpu_rsync.lock
/home/gianl/.glm-tpu-rsync.lock
```

Its five source/destination pairs now explicitly include the release worktree
and the original checkout containing the shared Git store.

## Exact installed change

`scripts/release/sync_glm_repositories.sh` is a versioned installation template,
not an independently leased launcher. Compared with the original script it adds
this pair and `gsutil rsync -c` to compare contents rather than relying on times:

```text
/home/gianl/glm-tpu-release → gs://driftbench-dsv4-uc/repos/glm-tpu-release
```

Original predecessor SHA256:
`8303e37d4f3732b66fa896ac0cf8df0903640a00738166838011f1120610cf6b`.

Installed template SHA256:
`823a84ce84a486ab09eb54018d4da3108618e42e3be4c1668e5c97779c8146ad`.

A test removes exactly the new pair/checksum flag and verifies the remaining
bytes equal the predecessor hash; Bash syntax also passes. Region admission, existing
backups, minimum-file refusal, symlink handling and SQLite sidecar exclusions
are unchanged. No bucket relocation or full checkpoint copy is involved.
The first strict verification refused a changed `FETCH_HEAD`: its local mtime
was newer than the mirrored object. Git/IDE metadata can change independently
of workload locks. Do not ignore a mismatch or claim that `-c` freezes writers;
verify against stable local files and exact remote generations after syncing.

## Post-seal installation and verification

1. Confirm owner-cancelled benchmark worker/publication termination, preserved
   partial originals/cancellation evidence and authenticated idle owners. A full
   benchmark or success seal is NOT required (owner override2026-09-14).
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
full model copy. Retain final checksum/generation receipts outside Git to avoid
a self-referential final-commit hash; the receipt identifies the published pin.
