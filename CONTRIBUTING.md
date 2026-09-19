# Development and release policy

This repository is private. Main is reserved for the supported release;
experiments continue on research branches. Readiness: [STATUS](docs/release/STATUS.md).

Use focused changes in isolated worktrees. Preserve original run pins/evidence.
Never change enforcement to relabel a historical failure as a pass. Cosmetic
cleanup does not need new TPU experiments; execution changes need own validation.

Tests run with `JAX_PLATFORMS=cpu` so pytest never opens the pod's chips. TPU
experiments and runs on the 32-chip pod are allowed through explicit scripts:
one workload at a time, under the workload lock, with the fleet left idle
afterwards. Retain ownership, memory, checkpoint and evidence-collection
protections for protected runs. Performance work follows
[docs/perf](docs/perf/REFERENCE_LOWHANGING_FRUIT_20260919.md).

Keep weights, credentials, private questions/answers, caches and runtime DBs out
of Git. Cite compact receipts/pins instead of embedding private payloads. Preserve
third-party copyright/license headers. Review actual diff, tests, dependencies,
claims, failures and retention before merge; self-review is not independent review.
Do not publish the repository or rewrite history during cleanup.

## Formatting release-owned code

Use Black 25.1.0; `pyproject.toml` limits it to the local CLI/request interface,
`scripts/release/`, release tools and release tests. From the repository root:

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
[format-ast-check-20260914.json](docs/release/format-ast-check-20260914.json).
All 36 changed Python files have equal syntax trees after normalizing docstring
indentation; only two needed that normalization. The original model-source guard
also passes. This is evidence for a formatting-only change, not TPU admission.
