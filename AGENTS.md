# Working in this repository

These rules apply to anyone, person or agent, who changes this repository. Read
the [goal](goal.md), the [handoff](HANDOFF.md) and the
[release status](docs/release/STATUS.md) first; the development policy is in
[CONTRIBUTING](CONTRIBUTING.md) and the test commands are in
[TESTING](docs/release/TESTING.md). Instructions written for earlier campaigns
(the GLM-5.2 and GLM-5.3 releases, the research tree) are history, not
instructions to restart that work; the previous version of this file is
`git show archive/research-20260922:AGENTS.md`.

## Roles and review

- The owner is the integrator. Only the owner approves a re-baseline of the
  equivalence records, a deviation from these rules, a run on the TPU fleet, a
  merge into `main`, a publication and any change of repository visibility or
  upstream.
- An approval given in advance for a class of work (for example "decide
  autonomously and record your decisions") covers the re-baselines, hardware
  runs and decisions of that work; record every decision taken under it where
  the owner will review it. It never covers a merge into `main`, a publication,
  a change of visibility or upstream, or a deviation from a hard rule below:
  each of these needs the owner's explicit approval for that instance.
- Merge into `main` only after the release checks and the review have passed and
  a backup of the work has been verified.
- Work in units, one at a time. For each unit: write down its scope, expected
  record changes and risks before changing the tree; implement it; run every
  gate it requires (below) on the tree you commit; commit it locally; then,
  before anything is pushed, have a separate verifier pass review the commit and
  the raw gate evidence adversarially. Fix every blocking finding by amending
  the unpushed commit and re-running the gates the fix affects, record the other
  findings as follow-ups, then push. The owner reviews each unit afterwards.
- Resolve material findings before a merge. A verifier pass by another assistant
  session is not independent human review. Describe review as it was done.

## Hard rules

- Tests and source audits run on the CPU (`JAX_PLATFORMS=cpu`). Never initialize
  a TPU outside a hardware run the owner authorized.
- Never create, delete, resize or reconfigure compute (TPU VMs, nodes, queued
  resources) or write instance metadata, and never run provisioning scripts. A
  hardware run holds both site workload locks, runs one workload at a time and is
  waited for; staging or repacking a checkpoint also holds the site's sync locks
  ([OPERATIONS](docs/release/OPERATIONS.md),
  [CHECKPOINTS](docs/release/CHECKPOINTS.md)). Never launch beside a live run or
  interrupt a run's collection of its evidence.
- Before stopping or removing a process or a file, prove that it is yours: PID,
  start time, boot and command line for a process; an inventory of what the unit
  created for a path. An elapsed deadline is not permission to stop, relaunch or
  delete anything.
- Storage: only the project's bucket in the fleet's region, as the operator's
  site file names it, within its existing storage bounds. No full-size safety
  copies of the weights.
- Keep out of Git: weights, credentials, private questions and answers, raw run
  directories, token logs, caches, databases, large generated artifacts, and
  private infrastructure literals (home directories, private addresses, host,
  VM, zone and bucket names, timestamped run or pack directory names; the
  committed release receipts and the migration history keep the run names they
  recorded). Cite compact receipts, pins and digests instead. Never print a
  suspected secret.
- No force-push, no history rewriting, no change of visibility or upstream; push
  only this private repository. Preserve other people's changes, every tag
  (among them `glm-5.2`, `glm-5.3`, `archive/research-20260922`,
  `pre-refactor-main-20260922` and `freeze-correct-128k-20260712`), the research
  branches, the receipts and the original evidence.
- Before removing a file, establish what depends on it and that a preserved
  commit (a tag or a research branch) keeps it; static import reachability alone
  is not deletion authority.
- Do not resume the owner-stopped GSM8K evaluation or the research optimization
  campaigns.
- Shell work: absolute paths, `git -C <path>`, `set -euo pipefail`, and a scratch
  directory of your own for each task. Do not edit a checkout that someone else
  or a live run is using. Never run a mutating git command (checkout, switch,
  reset, stash, clean, restore, worktree removal) in a checkout you did not
  create, and never with `--force`.
- Wait for long jobs by completion notification, or check them at least ten
  minutes apart unless you are diagnosing a known failure or answering an
  explicit request for status.

## Changing the code

Code lives in `glm_tpu/` and its tests in the mirrored path under `tests/`
([layout](README.md#repository-layout), [module map](docs/release/ARCHITECTURE.md)).
A change of behaviour comes with a test that fails without it and is declared in
its commit message.

The [graph-equivalence harness](tools/equivalence/README.md) proves that a change
leaves the lowered TPU device programs, the CPU numerics, the checkpoint
identities, the import closures, the executed functions and the wire formats
equal to the records in `tests/golden/data/`. Run the gates of every row that
applies, on the tree you commit, with `JAX_PLATFORMS=cpu`:

| Change | Gates |
|---|---|
| code, tests or data (anything but Markdown) under `glm_tpu/`, `tests/` or `tools/` | `python -m tools.equivalence check` (G1, G1-protocol, G3, G4, G6, G7, G9), `check --tier production` (G2, G2-protocol), `check --gates G6-static`, `pytest tests --ignore=tests/golden`, `GLM_EQUIVALENCE_STRICT=1 pytest tests/golden -m "not cpu32"`, and the helper contract tests (G8) |
| `pyproject.toml`, dependency pins, `examples/` or other configuration | the gates of the first row, and for a dependency or pin change also the `cpu32` golden wrappers (below) |
| anything under `tools/equivalence/` | `python -m tools.equivalence selftest` (G14) |
| anything under `tests/golden/`, and before a release | the `cpu32` golden wrappers, which run the heavy gates through pytest: `GLM_EQUIVALENCE_PRODUCTION=1 GLM_EQUIVALENCE_STRICT=1 pytest tests/golden -m cpu32` |
| Markdown only | `pytest tests/test_documentation.py` and `pytest tests/golden -m "not cpu32"` (the data contracts) |
| every change | `ruff check .`, `ruff format --check .`, codespell and `git diff --check` ([CONTRIBUTING](CONTRIBUTING.md#formatting-and-lint)) |

The exact commands, durations and the helper-test interpreter are in
[TESTING](docs/release/TESTING.md). The heavy gates refuse while a TPU run is
live on the host (`python -m tools.equivalence budget`): run them before or after
a hardware run, never beside it.

A recorded difference is never absorbed silently
([re-baselining](tools/equivalence/README.md#re-baselining)):

- G1-G4 and the fixture (the device programs, CPU numerics and identities) are
  frozen: they are recorded only from production paths equal to the baseline
  `181c013e` and do not change in a restructuring commit. A change meant to
  alter them (a numerical, kernel or model change) needs the owner's decision on
  a new baseline before it starts.
- A move or rename that changes G6 or G7 carries reviewed entries in
  `tools/equivalence/closure_map.toml` in its code commit. A separate commit,
  `[Equivalence] Re-baseline <gates> for <token>`, then records from the clean
  committed code commit (`record --rename-only --reason <token>`) and empties the
  table again.
- A reviewed change of a characterization record (G1-protocol, G2-protocol, G6,
  G7, G9 with G9-http) that is not a pure rename is a separate commit of the same
  title, recorded from the clean committed code commit with exactly one reason
  token: an H number for a sanctioned host change (`H1`, `H11`, ...), a stage or
  a work unit (`WU-E`, ...).
- A normalizer change re-records nothing and re-runs G14.
- A change the CPU records cannot hold (the compiler, libtpu, host behaviour)
  needs a TPU comparison: golden runs of the previous tree and runs of the change
  on the same fleet, compared with `python -m tools.equivalence compare-run`, in
  hardware time the owner authorized.

Commit messages: a title prefix (`[Refactor]`, `[Tests]`, `[Docs]`,
`[Equivalence]`, `[Launch]`, `[Config]`) and the unit's token where there is
one; what changed and why; deviations with their reasons; the gates run on the
committed tree with results and durations, and what was not run and why; the
expected re-baseline, or "none". No tool-attribution trailers.
