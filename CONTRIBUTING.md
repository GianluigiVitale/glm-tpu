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

## Formatting and lint

Use ruff 0.16.8 (`pyproject.toml` pins it with `required-version`) for both lint and
formatting. It is a separate developer tool, not a runtime dependency; do not install
or upgrade it inside an active benchmark environment. From the repository root:

```bash
ruff check .
ruff format --check .
```

The configuration covers the whole repository with Python 3.12 rules, except the remote
helpers in `glm_tpu/executor/remote/`, which are checked as Python 3.10 programs (they run
under the hosts' system `python3`) and are not formatted: the controller sends their text
to the hosts byte for byte and the equivalence harness records its SHA-256 (G9), so a
change to them is a host change. Until the tree-wide lint cleanup lands, introduce no new
`ruff check` findings.

Keep formatting changes in formatting-only commits (`ruff format` keeps the syntax tree,
up to docstring whitespace) and review the diff. The recorded programs are location-free
(the equivalence harness lowers without source locations), so whitespace does not reach
them; a `jax.named_scope` inside a Pallas kernel body and a Pallas kernel name do
([tools/equivalence/README.md](tools/equivalence/README.md), "Findings at S0"). A
formatting commit never changes them, and a tree-wide one runs G1, G2 and G3.

The Black 25.1.0 boundary that ruff replaces, and its formatting receipt
format-ast-check-20260914.json, are archived at tag `archive/research-20260922`
(`docs/release/format-ast-check-20260914.json`).
