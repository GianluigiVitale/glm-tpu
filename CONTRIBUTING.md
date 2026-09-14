# Development and release policy

This repository is private. Main is reserved for the supported release;
experiments continue on research branches. Readiness: [STATUS](docs/release/STATUS.md).

Use focused changes in isolated worktrees. Preserve original run pins/evidence.
Never change enforcement to relabel a historical failure as a pass. Cosmetic
cleanup does not need new TPU experiments; execution changes need own validation.

Tests require `JAX_PLATFORMS=cpu`. Never run generic tests on TPU, execute
provisioning scripts or launch a second workflow while one is active. Retain
ownership, memory, checkpoint and evidence-collection protections.

Keep weights, credentials, private questions/answers, caches and runtime DBs out
of Git. Cite compact receipts/pins instead of embedding private payloads. Preserve
third-party copyright/license headers. Review actual diff, tests, dependencies,
claims, failures and retention before merge; self-review is not independent review.
Do not publish the repository or rewrite history during cleanup.
