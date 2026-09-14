# Curation ledger and review boundary

**In progress; main is not yet curated.**
Objective: [CURATION_PLAN](../release/CURATION_PLAN.md).

Starting main `b667f00f1ae48c8ff37e92500550c1395d74c66d`: 1,971 files,
36,166,207 payload bytes. Candidate `release/curation-20260914` inherits
presentation `19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4`. Main/canonical execution
checkout are unchanged. Candidate: **1,449 files; 535 originals / 11,188,543 bytes
removed; 1,399 unresolved dispositions**. Counts are not completion percentages.

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
- The lazy benchmarking facade preserves 141 named targets after retiring
  16 M2048 and55 PP16 aliases. Retained original target mappings and ordered
  `__all__` entries match baseline; other exports still need purpose decisions.
- Twelve native/delivery/user imports now use the stdlib-only
  `glm_tpu/host_paths.py` guard, with identical function AST. It avoids importing
  experiment preflight just for path checks; old research callers are unchanged.
- CLI/user-controller/user-worker roots conservatively reach 336 Python files,
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
| Standalone legacy Ray/staging/backup/triage workflows and old test module | 18 | Scope/incoming consumers; backup/stager/fork-sync/test fully read; unchanged disk-floor assertion moved to a named test |
| Gate D compensated/tuple/precompile-admission/provisioning and layer-0 capture workflows | 70 | Scope/header/consumer review; exact preserved bytes; launcher-only cases retired from two retained numerical/HLO test modules |
| Associated Gate D contracts and candidate/provisioning receipts | 58 | JSON structure/identity/classification, consumers and exact original recovery; no remaining executable readers |
| PP16 feature2 acquisition, numerical and classifier workflow | 34 | Scope/header/consumer review;55 unused aliases removed; general physical-axis inverse assertion moved unchanged into shared HLO tests |
| Associated PP16 feature2 receipts | 13 | Parsed kinds/scopes/identities; no remaining executable filename/stem/SHA readers; exact baseline/research recovery |

Current native source-pinned admission documents and release result receipts remain.
Outside references to retired prefill journals occur only in historical specification,
lessons or admission prose. Those names resolve in preserved Git, not new files
on main. Three intermediate edits merely redirected prior retired-document
references; those versions remain at `0f1fbbe35e2bc33393f9dd2840781766a21c709b`.

Only four top-level legacy helpers remain: disk watchdog, dump archiver, Ray
launcher and network validator. Retained capture wrappers still consume them;
their final roles remain open. Retired fork-sync/triage mentions in the unchanged
source-pinned launcher are historical help/comments, not calls. The installed
mirror invokes none of the removed scripts and is unchanged. The observability
guide now distills the research methods and links the original toolset to Git.

The latest70-file cohort is outside the335-file native-root closure. A remaining
historical mirror-verifier caller was the layer-0 projection/dense capture
wrapper; that separate launcher and its dedicated tests retired together.
Seven launcher-only test/helper functions in two otherwise-retained modules were
read fully and removed; every remaining function AST is unchanged. Shared dense
layout, projection and RMS/HLO helpers stay because they still have consumers.
No source/HLO identity was re-registered. The installed Git mirror is separate.
The associated58 JSON contracts/receipts were subsequently removed after parsing
their structure/identity/classification and verifying exact baseline/research
bytes. Their remaining references were historical prose or a separate FP32-shadow
rejection record, not runtime readers. Native WS32 admission receipts and the
shared RMS frontier certificate remain because code/tests consume them.

The full long observability playbook was read, then replaced by a compact methods
guide: causal watchpoints, observer effects, coherence, evidence levels, current
tool locations, economical incident packets and exact original recovery. Old
frontiers, commands and the repeated experiment chronology are off the current
page. This document's full-read/role decision is complete; not the whole repo's.

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

After the PP16 code/receipt retirement and native host-guard extraction, selected release checks passed
**477 tests, one optional skip, two upstream warnings**, plus frozen-source,
content and isolated no-deps wheel checks. Not the whole historical test tree,
a full dependency installation or TPU validation.

The host-path/facade/HLO suite passed25 cases. Relative paths, final/ancestor
symlinks (including dangling links), allowed absent targets and import isolation
are covered; this is not race-free filesystem access. All twelve consumer changes
are import-only. The moved mapping assertion and every prior shared HLO test
function have identical ASTs. The final release check includes the13 unconsumed
JSON removals; no missing registered source/asset was found by those checks.

User provenance/database code and its synthetic SQLite tests were fully read and
justified. Shared Transaction/export_rows/PRIMARY_DB still import the historical
benchmark database module. That module was also read in full, but its whole-file
role remains unresolved pending safe shared-helper isolation and adjudication of
benchmark-only archive callers. Passing tests alone does not justify retaining it.

The PP16 straddler classification is not in that removal: a retained oracle
wrapper reads and pins it. Native WK preparation/cold replay also genuinely use
the HistoryCalls writer, load_calls reader and dense-frontier HLO admission.
These specific retained roles were fully read; they do not justify all historical
workflows. Frozen shared compile preparation still has historical branches, and
refactoring source positions can alter raw HLO debug identities. No model source
or admission identity has been changed to make cleanup appear complete.

The focused projection/dense suite also passed **27 tests**;
**11 local historical-artifact cases skipped**. A whole tracked executable-text
scan found no remaining references to any retired module/script stem. Static
closure and syntax checks are supporting evidence, not proof of model behavior.

The retained disk-floor assertion has an identical function AST to its original
and the same repository root. Its focused suite plus release-check orchestration
and retained Ray-network guards passed15 cases. Other old ownership-test cases
only inspected removed campaign scripts and were retired with them.

CLI/request checks additionally passed **26 tests, one optional tokenizer skip**.
Core doctor, metadata-only CLI, controller help and Black passed (45 files
unchanged). Of92 remaining Markdown relative file targets, only the upstream
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
