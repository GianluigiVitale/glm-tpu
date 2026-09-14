# Curation ledger and review boundary

**In progress; main is not yet curated.**
Objective: [CURATION_PLAN](../release/CURATION_PLAN.md).

Starting main `b667f00f1ae48c8ff37e92500550c1395d74c66d`: 1,971 files,
36,166,207 payload bytes. Candidate `release/curation-20260914` inherits
presentation `19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4`. Main/canonical execution
checkout are unchanged. Candidate: **1,639 files; 342 originals / 7,162,290 bytes
removed; 1,600 unresolved dispositions**. Counts are not completion percentages.

## Every-file accounting

[disposition.jsonl](disposition.jsonl) records original Git blob/mode/size,
purpose, consumers, category, action, justification and review identity.
New paths have a null baseline. Retained code/docs require full reads bound to
current SHA-256; generated evidence requires structure/provenance/consumer review.
Removed originals have exact starting-commit/path/blob recovery.

```bash
# Diagnostic: exits nonzero while curation is incomplete.
JAX_PLATFORMS=cpu python tools/curation_inventory.py
# Print coverage; never invent decisions or change files.
JAX_PLATFORMS=cpu python tools/curation_inventory.py --refresh
```

The checker proves ledger consistency, not that semantic review happened,
absence of dynamic dependencies or model correctness. The ledger itself uses
manifest validation rather than a self-referential hash.

## Supported boundary established so far

- User controller/recovery use `scripts/release/ws32_host_ops.py`, not the
  retired campaign scheduler. Nine helper ASTs and three constants/remote-program
  bytes match starting main; original ownership/source/lease guards remain.
- The lazy benchmarking facade preserves 196 named targets after retiring
  16 unused M2048 aliases; original `__all__` is unchanged. Remaining exports
  still require purpose decisions.
- CLI/user-controller/user-worker roots conservatively reach 335 Python files,
  including conditional diagnostics. This is not proof all are needed. Three
  known reference-extract import findings remain.
- Worker loading still uses the historical short-decoder runner. Frozen model
  source and native source/HLO identities have not changed.
- CLI/request implementation, package entry points, environment asset and tests
  have completed full-read/role review. User-facing installation, inference,
  operations, checkpoint, reviewer, contribution, security and notices docs also
  have reviewed roles. Historical evidence/limitations remain explicitly scoped.

## Removed cohorts and review scope

| Cohort | Files | Basis |
|---|---:|---|
| External-fork kernel entry points | 4 | Full reads; no supported callers |
| Historical external-review transcripts | 31 | Scope/headers/incoming references, not full reads |
| Legacy load experiments and campaign scheduler/test | 6 | Full launcher reads; host helpers extracted first |
| Upstream PR/recon/submission material | 30 | Prose scope; executable files fully read |
| Earlier external-fork design/runbook prose | 14 | Scope/headers; no executable readers |
| PP16 forced-round experiment | 35 | Isolated builders/validators/scripts/tests; scope/consumers |
| Research journal | 1 | Header/tail/provenance; executable readers retired first |
| Projection-contraction orchestration | 24 | Scope/dynamic consumers; shared chain subsequently retired |
| Imported research reports/requests/PR drafts | 14 | Scope/provenance; originals retained in Git |
| RMS/geometry/M2048/DB518 diagnostic chain | 55 | Consumers retired together; shared captured-RMS helpers remain |
| Unconsumed diagnostic JSON receipts | 78 | Parsed objects/kinds, exact preserved bytes and consumers |
| Evidence/reuse journals, catalog and catalog test | 4 | Journal scope/catalog structure; test fully read; genuine no-legacy-import invariant moved unchanged |
| Forced-round PP16 JSON receipts | 18 | Parsed kinds; exact recovery bytes; zero remaining filename references |
| Superseded prefill plans/running journals | 28 | Headers/status/consumers, not full prose reads; no executable readers |

Source-pinned admission documents and actual result receipts remain.
Outside references to the last cohort occur only in historical specification,
lessons or admission prose. Those names resolve in preserved Git, not new files
on main. Three intermediate edits merely redirected prior retired-document
references; those versions remain at `0f1fbbe35e2bc33393f9dd2840781766a21c709b`.

All original removed bytes also match research
`83f0c2728d0d418255a917343cc89d24b815bd0c`. For example:

```bash
git show b667f00f1ae48c8ff37e92500550c1395d74c66d:docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md
```

Only candidate-tree files were removed. Research refs/history, external originals,
DB616–621 receipts, weights and the canonical checkout are untouched.
This is removal provenance, not new hardware replay.
Earlier chronological curation checks remain in this file at `0f1fbbe3`.

## Checks and remaining work

After this documentation/receipt-only batch, selected release checks passed
**462 tests, one optional skip, two upstream warnings**, plus frozen-source,
content and isolated no-deps wheel checks. Not the whole historical test tree,
a full dependency installation or TPU validation.

CLI/request checks additionally passed **26 tests, one optional tokenizer skip**.
Core doctor, metadata-only CLI, controller help and Black passed (45 files
unchanged). Of93 remaining Markdown relative file targets, only the upstream
model-card snapshot's malformed `github.com/...` target is unresolved; its original
bytes remain preserved. This scan does not validate anchors or external URLs.
The metadata-only checkpoint tool/tests are fully read, with its historical
237-weight-entry receipt structure/totals and static metadata pins checked.
No cloud calls were made for that check. Local reference/license
hashes match existing receipts; no new upstream fetch, legal clearance, payload
rehash or recovery reconstruction is claimed.

Finish remaining dependency decisions/full reads/asset provenance, audit all
retained links/anchors/assets and applicable CPU tests, then adversarially
self-review. Tests do not justify retaining an otherwise unused experiment.
Never remove frozen code or user/recovery dependencies by name. Merge only after
all dispositions and final checks close; verify exact regional backup under the
installed locks. No model campaign or independent-review claim.
