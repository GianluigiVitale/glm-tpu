# Curation ledger and current review boundary

Status: **in progress; not a curated-main completion claim**.
Full objective: [CURATION_PLAN](../release/CURATION_PLAN.md).

Starting main is `b667f00f1ae48c8ff37e92500550c1395d74c66d`: 1,971 files,
36,166,207 payload bytes. The curation branch inherits presentation commit
`19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4`; neither main nor the canonical
execution checkout has been switched by this work.

## Per-file record

[disposition.jsonl](disposition.jsonl) binds every original path to its Git blob,
mode and size. New tracked files have a null baseline. Each record contains
purpose, consumers, category, action, justification and a review record.
Initialization intentionally marks files unresolved; it does not infer keep or
delete from a filename, import scan, age, size or existing test result.

Retained code/docs need an explicit full-read record bound to their current
SHA-256. Generated evidence needs its actual schema/provenance/consumer review.
The ledger itself uses manifest validation rather than a self-referential hash.
Removed original files require an exact starting-commit/path/blob recovery record.

The checker validates coverage, baseline identity, recorded review hashes and
disposition consistency. **It cannot independently prove that a recorded semantic
review happened**, or discover every dynamic dependency. Those remain review work.

```bash
# Diagnostic only: exits nonzero while curation is incomplete.
JAX_PLATFORMS=cpu python tools/curation_inventory.py

# Prints a refreshed ledger; does not change files or invent reviewed decisions.
JAX_PLATFORMS=cpu python tools/curation_inventory.py --refresh
```

## Findings changing the next action

1. The user controller initially imported the benchmark controller for shared
   SSH, branch admission, original-process observation and publication checks.
   These nine helpers now live in `scripts/release/ws32_host_ops.py`. Function
   ASTs and three constants/remote-program bytes match starting main; a fresh
   CPU import proves the user controller does not load the campaign launcher.
   Shared recovery/archive callers use the same helpers. The historical campaign
   launcher has now been removed; its shared protection tests remain.
   Worker initialization/loading still depends on the historical runner.
2. Eager package exports pulled historical experiments into ordinary imports.
   The benchmarking initializer has been fully inspected in its original form;
   an on-demand facade preserves all 212 original named export targets and the
   original `__all__`. Research aliases are temporary during consumer separation,
   not a final reason to retain research-only files. The runtime initializer is
   inside frozen model-source scope and has NOT been changed.
3. The source inventory now starts only at the CLI, user controller and user
   worker, and resolves the literal lazy-export targets. Their conservative
   closure changed from 345 to 335 Python files. This is a dependency finding,
   **not ten proven deletions or a performance result**. Three known reference
   relative-import errors remain reported.
   After host extraction the count is still 335, but the campaign controller is
   absent: one shared module replaces it. This is an actual dependency boundary,
   not a claim that 335 files are all necessary for a user request.
4. The research log has source-check consumers in two historical PP16 analyzers.
   It has NOT been deleted on the assumption that prose cannot be a dependency.
   These analyzers and their consumers need disposition before removing the log.

## Verified removals so far

- Four `scripts/kernel_probe/` entry points (45,369 bytes): full reads established
  external-fork-only indexer, sparse-MLA, KV-pack and LSE experiments. Their shell
  launcher could switch the external fork's branch. No supported executable,
  configuration or test consumer was found; package discovery excludes scripts.
- Thirty-one `docs/reviews/` historical reviewer transcripts (470,090 bytes):
  scoped content/header inspection and incoming-reference review established old
  external-fork, parity-harness, benchmark and PR-series provenance, not current
  native-engine admission. This is NOT a claim to have read every removed report
  in full. Remaining code mentions are historical comments/docstrings, not file
  reads; current release docs do not link to these transcripts.

- Six obsolete launchers: the benchmark campaign scheduler after shared-host
  extraction and five legacy Ray load/cache experiments. All six were read in
  full. No supported caller uses them. The campaign-only outer orchestration test
  was retired with its scheduler; host admission, original ownership, publication
  and archive negative/recovery checks remain.
- Thirty upstream-workflow files: seven old PR-series drafts, nine external-fork
  reconnaissance notes, twelve upstream reports/audits/reproducer files, the
  later three-PR draft and its submission checker. Document scope/headers and
  incoming references were reviewed, not every removed prose line. The two
  executable files were read in full. The checker was the sole executable reader
  of five removed review documents, so it was retired with them. Remaining old
  research-log/config citations describe preserved historical paths, not runtime
  file reads or current release instructions.

- Fourteen earlier external-fork design/runbook/status documents (464,230 bytes):
  scope/header and incoming-reference review, not full prose reads. These describe
  TorchAX/Ray, old MTP/throughput campaigns or upstream submission, not supported
  native setup. No current release/greenfield-document links or runtime file reads
  were found. The protocol and observability documents remain pending their own
  review; they were not swept away as part of this historical group.

All 85 originals (1,273,782 bytes) are recoverable at starting main and research pin
`83f0c2728d0d418255a917343cc89d24b815bd0c`. Individual blob/size and recovery paths
are in the ledger. Only release-tree files were removed; external evidence,
research refs, model weights and the canonical checkout were untouched. Recover
an original without running it using:

```bash
git show b667f00f1ae48c8ff37e92500550c1395d74c66d:scripts/kernel_probe/run_kernel_gates.sh
```

These are intermediate pruning batches, not final repository curation. Historical
research prose and conditional runtime/benchmark dependencies remain to resolve.

## Checks performed on this intermediate change

- 23 focused inventory/import-boundary tests passed.
- Selected release/host-runtime suite: 446 passed, 1 optional skip, 2 warnings.
- Original frozen numerical-source guard, real user-controller CPU import,
  content scan and isolated wheel checks passed. No TPU execution or hardware
  performance/quality inheritance is claimed.

The host extraction additionally passed 101 targeted helper, user-controller,
deployment and original-publication/archive checks. A stale historical campaign
fixture was corrected in that commit; its outer-orchestration case subsequently
left with the retired campaign scheduler. No production admission check was relaxed.
The expanded selected release check passes 460 tests (1 optional skip, 2 upstream
warnings), with frozen-source, content and isolated-package checks passing.
After scheduler retirement, 62 focused host/user/publication/archive tests passed.
The complete selected release checker also passed after the 36-file retirement
batch: 460 passed, 1 optional skip, 2 warnings, plus frozen-source/content/package
checks. The subsequent 14 removals change only historical prose, not tested code.

Next: continue full reads and per-file decisions; separate shared controller and
loader dependencies; remove verified research-only groups and update their links
and recovery records. No file-count target replaces that work. Do not merge this
intermediate branch merely because its existing selected checks pass.
