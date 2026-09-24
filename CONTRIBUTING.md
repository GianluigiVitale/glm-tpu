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

## Formatting release-owned code

Use Black 25.1.0; `pyproject.toml` limits it to the local CLI/request interface,
the controller, worker and pack-worker modules, release tools and release tests. From the
repository root:

```bash
black --check .
```

For an intentional formatting change, run `black .` and review the diff. The
formatter is a separate developer tool, not a runtime dependency; do not install
or upgrade it inside an active benchmark environment. Historical model, compiler,
oracle and analysis sources are outside this formatting boundary. Do not expand
the boundary casually: source lines/debug metadata may affect recorded compiler
identities even when executable Python behavior is unchanged.

The initial formatting receipt is
format-ast-check-20260914.json (archived at tag `archive/research-20260922`: `docs/release/format-ast-check-20260914.json`).
All 36 changed Python files have equal syntax trees after normalizing docstring
indentation; only two needed that normalization. The original model-source guard
also passes. This is evidence for a formatting-only change, not TPU admission.
