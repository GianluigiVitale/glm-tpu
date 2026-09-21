# Private reviewer source package

The reviewer archive contains the audited tracked source, tests, configuration,
documentation, compact receipts, attribution and licenses from one exact commit.
It excludes Git internals, weights, credentials, private request inputs, raw
generated answers, token logs and databases. It is a source package, not a model
distribution or deployable runtime image. Keep it private until the owner chooses
to share it; preparing it sends no message and publishes nothing externally.

After the release checks and final commit, from a clean checkout:

```bash
release_commit=$(git rev-parse HEAD)
git archive --format=tar.gz --prefix="glm-tpu-${release_commit}/" \
  --output="/path/outside/git/glm-tpu-${release_commit}.tar.gz" "$release_commit"
sha256sum "/path/outside/git/glm-tpu-${release_commit}.tar.gz"
```

Use an existing private output directory. Inspect the member list and apply the
tracked-content audit before sharing. The adjacent publication receipt records
the full commit, archive SHA-256, member count and generation-bound regional
readback in `gs://driftbench-dsv4-uc` (US-CENTRAL2). It stays outside Git to avoid
a self-referential final commit identity. Never interpret an archive without a
passing release-validation receipt as a ready release.

This release's prepared archive is in the private directory
`/home/gianl/glm-run/gsm8k_acceptance_20260920/publication/`, named
`glm-tpu-<final-commit>.tar.gz`. `latest-package.json` in that directory identifies
the current source package and its verified private-main and regional backup
record. The original validated release is preserved in `final-promotion.json`;
later documentation updates retain the same hardware evidence and their own
commit-bound archives. The source archive has no Git database; a separate incremental history
bundle plus the preserved base retain recovery. If the publication receipt is
absent or failed, preparation/promotion is not established.

Start with [the short summary](PROJECT_SUMMARY.md), [README](../../README.md) and
[reviewer guide](REVIEWER_GUIDE.md). The archive supports the portable CPU subset
in [TESTING](TESTING.md). Checks that read `git show` need the private full-history
clone; hardware and sealed-evidence replays additionally need retained external
assets. Research refs and originals remain preserved separately; a compact
reviewer archive does not replace their recovery backups.
