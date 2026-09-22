# Private reviewer source package

The reviewer archive contains the audited tracked source, tests, configuration,
documentation, compact receipts, attribution and licenses from one exact commit.
It excludes Git internals, weights, credentials, private request inputs, raw
generated answers, token logs and databases. It is a source package, not a model
distribution or deployable runtime image. Keep it private until the owner chooses
to share it; preparing it sends no message and publishes nothing externally.
Historical numerical reference arrays remain in Git and recovery backups but are
excluded from this compact package. The package manifest lists each exclusion.

After the release checks and final commit, from a clean checkout:

```bash
release_commit=$(git rev-parse HEAD)
git archive --format=tar.gz --prefix="glm-tpu-${release_commit}/" \
  --output="/path/outside/git/glm-tpu-${release_commit}.tar.gz" "$release_commit" \
  . ':!docs/artifacts/*.npy'
sha256sum "/path/outside/git/glm-tpu-${release_commit}.tar.gz"
```

Use an existing private output directory. Inspect the member list and apply the
tracked-content audit before sharing. The adjacent publication receipt records
the full commit, archive SHA-256, member count and generation-bound regional
readback in `gs://driftbench-dsv4-uc` (US-CENTRAL2). It stays outside Git to avoid
a self-referential final commit identity. Never interpret an archive without a
passing release-validation receipt as a ready release.

This release's prepared archive is in the private directory
`/home/gianl/glm-run/glm53_resident_release_20260922/publication/`, named
`glm-tpu-<final-commit>.tar.gz`. `latest-package.json` in that directory identifies
the current source package and its verified private-main and regional backup
record. `promotion.json` records the private-main result; `release.json` binds
the private `glm-5.3` tag/release and downloaded archive verification. The
earlier GLM-5.3 publication remains under `glm53_migration_20260921/publication/`.
The completed
GLM-5.2 package and original publication records remain preserved separately;
`/home/gianl/glm-run/four_conversation_release_20260921/promotion.json` identifies
that historical main and its regional recovery objects.
The source archive has no Git database; a separate incremental history
bundle plus the preserved base retain recovery. If the publication receipt is
absent or failed, preparation/promotion is not established.

Start with [the short summary](PROJECT_SUMMARY.md), [README](../../README.md) and
[reviewer guide](REVIEWER_GUIDE.md). The archive supports the portable CPU subset
in [TESTING](TESTING.md). Checks that read `git show` need the private full-history
clone; hardware and sealed-evidence replays additionally need retained external
assets. Research refs and originals remain preserved separately; a compact
reviewer archive does not replace their recovery backups.
