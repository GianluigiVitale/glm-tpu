# Initial release inventory

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
JAX_PLATFORMS=cpu python -m unittest discover -s tests/release -v
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

## Next

Separate the supported request/deployment path from diagnostic dispatch, audit
its true dependencies/assets, then add tested installation and invocation. Keep
original execution math and protect the active benchmark. Finish license/notice
and privacy review before the main merge. This inventory is a starting point,
not a claim that the release is ready.
