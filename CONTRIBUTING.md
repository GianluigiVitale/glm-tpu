# Contributing

`main` carries the supported release (tag `v1.0.0`); work happens on branches and
reaches `main` only after review. Report security problems privately, as
[SECURITY](SECURITY.md#reporting-a-vulnerability) describes. The layout is
described in the [README](README.md#repository-layout) and in
[ARCHITECTURE](docs/release/ARCHITECTURE.md).

## Changes

- Keep changes focused, one concern per commit, in an isolated worktree. A commit
  message says what changed, why, and which checks ran with their results.
- Code lives in `glm_tpu/`, its tests in the mirrored path under `tests/`
  (`glm_tpu/engine/request.py` is tested by `tests/engine/test_request.py`). A
  change of behaviour comes with a test that fails without it.
- Keep weights, credentials, private questions and answers, caches and runtime
  databases out of Git; cite compact receipts and pins instead of private
  payloads. Preserve third-party copyright and license headers
  ([notices](THIRD_PARTY_NOTICES.md)).
- Review the actual diff, tests, dependencies, claims, failures and retention
  before merge; self-review is not independent review.
- Never change a check to relabel a historical failure as a pass. Do not rewrite
  history or change the repository's visibility during cleanup.
- Cosmetic cleanup needs no new TPU run; an execution change needs its own
  hardware validation, recorded like the release receipts.

## Tests and the equivalence gates

Tests run on the CPU only (`JAX_PLATFORMS=cpu`; the test session refuses any
other platform). Never run tests on a TPU host's accelerator, never run a
provisioning script, and never start a second fleet workflow while one is
active. [TESTING](docs/release/TESTING.md) lists the tiers and commands.

Any change under `glm_tpu/` that is not a declared numerical change must keep the
device programs, CPU execution goldens, checkpoint identities, import closures,
executed functions and wire formats identical to the records in
`tests/golden/data/`: run `python -m tools.equivalence check`,
`check --tier production` and, when `tools/equivalence/` changes,
`python -m tools.equivalence selftest`. A recorded difference is changed only by
a separate re-baseline commit that the maintainer reviews; the
[harness README](tools/equivalence/README.md) describes the rules.

## Formatting and lint

Use ruff 0.16.8 (`pyproject.toml` pins it with `required-version`) for both lint
and formatting, and codespell for spelling. They are developer tools, not runtime
dependencies: install them in a separate tool environment, never inside a
runtime or measurement environment. From the repository root:

```bash
ruff check .
ruff format --check .
codespell --skip='*/hf_config' README.md CONTRIBUTING.md SECURITY.md THIRD_PARTY_NOTICES.md docs glm_tpu tests tools
git diff --check
```

codespell skips the pinned third-party model files in `hf_config/`, which are
kept verbatim. The ruff configuration covers the whole repository with Python 3.12
rules, except the remote helpers in `glm_tpu/executor/remote/`, which are checked
as Python 3.10 programs (they run under the hosts' system `python3`) and are not
formatted: the controller sends their text to the hosts byte for byte and the
equivalence harness records its SHA-256, so a change to them is a host change. Fix
a new finding, or, where the fix would change behaviour, suppress it with a
reasoned `# noqa: CODE (why)` on its line (a reasoned `pyproject.toml` entry for a
whole file).

Keep formatting changes in formatting-only commits (`ruff format` keeps the syntax
tree, up to docstring whitespace) and review the diff. The recorded programs are
location-free (the harness lowers them without source locations), so whitespace
does not reach them; a `jax.named_scope` inside a Pallas kernel body and a Pallas
kernel name do ([harness README](tools/equivalence/README.md), "Findings at S0").
A tree-wide formatting commit runs the full equivalence gates and the tests that
read source text.
