# Initial release inventory

> Historical snapshot (2026-09-14, pre-curation). The authoritative file-by-file
> disposition of main is the [curation ledger](../curation/README.md); counts and
> retention statements below describe the tree at the starting pin.

Snapshot: 2026-09-14, starting research pin
`83f0c2728d0d418255a917343cc89d24b815bd0c`. The initial scan reads tracked
working-tree files after the release landing documents were replaced; it is not
an immutable Git-tree byte inventory or the eventual final release manifest.
New, untracked release files were outside this initial scan.

## Findings that determine cleanup order

- 1,912 tracked files, about 35.7 MB in the initial working-tree scan. The large
  storage consumption is outside this checkout, not terabytes of tracked weights.
- The campaign launcher and original worker statically reach **333 Python files**.
  This includes conditional diagnostic imports, not 333 mandatory serving modules.
- **23 reachable files** contain dynamic-dispatch/subprocess sites needing review.
  Static reachability alone cannot prove which files are safe to remove.
- Runtime validators directly refer to historical JSON receipts, configuration,
  benchmark extractors and a chat template. Deleting `docs/artifacts/` wholesale
  would break integrity checks. Source-hash registrations also constrain refactors.
- The launcher hard-codes the research branch, Python path, site and private
  dataset capsule. A release on main needs an actual deployment boundary, not
  instructions that silently run the research checkout.
- There is no root packaging/dependency configuration or root license at the
  starting pin. PyTorch appears in checkpoint/reference utilities even though the
  native model executes in JAX; dependencies must follow actual runtime paths.
- The basic tracked-content secret-pattern scan found no matches. This is **not
  a complete security clearance**: Git history, private datasets and other token
  formats are not covered.
- Three vendored Transformers reference files have package-relative imports that
  cannot resolve as standalone modules here. The scan reports these as ImportError;
  they are reference extracts, not installable native runtime entry points.

Raw path/count results: [initial-inventory.json](initial-inventory.json). No
secret values, question text, gold answers or raw model responses are included.

## Reproduce without initializing JAX or touching TPU

From the release repository root, using Python 3.12:

```bash
python tools/release_inventory.py
JAX_PLATFORMS=cpu python -m pytest tests/release -q
git diff --check
```

The inventory exits nonzero for findings or parse/import-resolution errors; the
reference extracts above currently trigger that diagnostic exit. Its tests pass
8/8, including relative imports, cycles, missing edges, dynamic-call detection,
syntax failure, symlink refusal and token-pattern matching. This is source-audit
coverage only, not model validation or a comprehensive secret-scanner test suite.

## Preservation

The release replacements of README/AGENTS/goal/HANDOFF are recoverable from the
published starting research pin. No model, enforcement, checkpoint, result or
research branch was deleted. No other experiment has been pruned yet. Before
pruning, follow static edges plus dynamic entry points, source registrations,
assets and tests; explicitly retain anything unresolved.

### First release-only removal

Removed the obsolete root `CLAUDE.md`, `KICKOFF.md` and `PLAN.md` instruction
documents. All three were byte-identical to their copies at the published
research pin `83f0c2728d0d418255a917343cc89d24b815bd0c`; that research branch and
Git history are retained. Recover any original with, for example:

```bash
git show 83f0c2728d0d418255a917343cc89d24b815bd0c:PLAN.md
```

Full reads showed stale research queues and superseded performance/gate rules.
Searches of code, tests, configuration and tools found only comments/docstrings
referring to these names, not runtime file reads. Those historical comments do
not override release instructions. No experiment implementation, result, runtime
asset or active-worktree file was deleted. Current release instructions are
`AGENTS.md`, `goal.md`, `HANDOFF.md` and `docs/release/STATUS.md`.

## Next

### Legacy oracle patch removed from release tree

The sole `patches/vllm-fused-indexer-wk-clone.patch` was read in full and removed
only from this release checkout. It is byte-identical to Git blob
`c1b64f0d88b768f4367e2fa1c1b4d07e7013a8fd` at the published starting research
pin. Repository-wide references are historical research prose and the updated
third-party notice; searches found no runtime/test/configuration consumer or
patch-directory loader. It modifies an external legacy vLLM oracle, not this
native-JAX engine. Recover it using `git show` with that commit and original path.
Historical postmortem/log mentions are retained, not rewritten as current steps.
This is a release-tree removal, not deletion from the research branch or history.

### Remaining cleanup

Separate the supported request/deployment path from diagnostic dispatch, audit
its true dependencies/assets, then add tested installation and invocation. Keep
original execution math and protect the active benchmark. Finish license/notice
and privacy review before the main merge. This inventory is a starting point,
not a claim that the release is ready.

## Current supported-boundary review

The scanner now starts at the user controller and local CLI as well as the
benchmark controller/owner worker. At release pin `32e0be38` these roots reached
344 Python files, including conditional historical diagnostic branches. The
registry's explicit dynamic load of `bench/benchmarks.py` and `bench/extract.py`
was inspected separately; both remain required. The three known reference-only
relative-import findings remain visible rather than being called passing imports.
This is a conservative dependency inventory, not proof that every conditional
diagnostic module is used by a user request.

Removed seven isolated legacy scheduling/provisioning scripts after full reads
and exact comparison to the preserved research commit. No incoming executable
references were found in code/scripts/tests/configuration; the installed mirror
cron calls `/home/gianl/bin/sync-glm.sh`, not any removed file. These obsolete
scripts included historical Ray stop/retry chains, auto-merge/push logic and
environment replacement. They do not belong in the supported native workflow.
Exact paths, byte sizes and Git blobs: [removal ledger](removed-legacy-schedulers.json).
Restore an original for research using `git show <preserved_commit>:<path>`.

Kept `launch_glm_32chip.sh` and `validate_ray_network.sh`: protected historical
oracle capture/validation wrappers still refer to their exact source bytes.
The old full-bundle backup helper and the historical sparse gate that invoked it
later left main during curation; the installed mirror is separate and unchanged.
The [scripts index](../../scripts/README.md) marks these historical paths as
unsupported for native deployment. No running source, evidence, weight, DB,
backup process, research branch or Git object was removed.
