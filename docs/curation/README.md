# Curation ledger and review boundary

**In progress; main is not yet curated.**
Objective: [CURATION_PLAN](../release/CURATION_PLAN.md).

Starting main `b667f00f1ae48c8ff37e92500550c1395d74c66d`: 1,971 files,
36,166,207 payload bytes. Candidate `release/curation-20260914` inherits
presentation `19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4`. Main/canonical execution
checkout are unchanged. Candidate: **1,419 files;568 originals /11,586,686 bytes
removed;1,361 unresolved dispositions**. Counts are not completion percentages.

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
- The lazy benchmarking facade preserves127 named targets after retiring
  16 M2048,55 PP16 and14 transport aliases. Retained original target mappings and ordered
  `__all__` entries match baseline; other exports still need purpose decisions.
- Twelve native/delivery/user imports now use the stdlib-only
  `glm_tpu/host_paths.py` guard, with identical function AST. It avoids importing
  experiment preflight just for path checks; old research callers are unchanged.
- User provenance now imports the shared SQLite adapter/export/path directly,
  not the benchmark recorder. Ownership replay belongs to shared host operations;
  neither user replay nor publication imports the retired benchmark archive.
- Validation exports are also lazy: all89 originally bound targets resolve to
  their original module attributes. Only the unbound `Ws32LongContextOracle`
  package alias leaves `__all__`; the actual class remains in its module.
- CLI/user-controller/user-worker roots conservatively reach310 Python files,
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
| Benchmark-only DB/archive, dedicated tests, receipt and stale delivery journal | 6 | Current code/tests fully read; shared transaction/export/ownership primitives extracted first; journal scope review; exact original recovery |
| PP transport, PP16 dense and virtual-owner legacy prefill-geometry diagnostics | 21 | Scope/import/consumer review, exact preserved bytes;14 unused exports retired; shared numerical kernels untouched |
| Associated chunk0 diagnostic JSON | 6 | Parsed scope/identity/claims; no remaining filename/stem/SHA readers; original recovery |

## Recovery and remaining boundaries

All removed original bytes match starting main and research
`83f0c2728d0d418255a917343cc89d24b815bd0c`. Recover an original without
restoring an obsolete workflow onto main:

```bash
git show b667f00f1ae48c8ff37e92500550c1395d74c66d:docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md
```

Only candidate-tree copies were removed. Original results, research refs/history,
external evidence, DB616–621, weights and canonical execution are untouched.
Historical specification/inventory mentions are recovery references, not current
commands. Earlier detailed curation chronology is available at251d9d0b.

Lazy validation removes25 legacy modules from the conservative native closure,
including association/integrated-dense/output-geometry diagnostics and capture
validators. Their own callers/tests/assets still need adjudication; they have NOT
been removed simply because the scan is smaller. Remaining facade exports do not
all have final purpose decisions. Full reads are recorded only where performed.

Native WK preparation/cold replay genuinely use HistoryCalls/load_calls and
admission helpers. Shared compiler preparation contains historical branches;
source locations can affect raw HLO debug identities. Do not cosmetically change
frozen numerical code or widen admission hashes to simplify the dependency graph.
The four remaining top-level legacy helpers still have capture consumers. A PP16
straddler receipt is read/pinned by an oracle wrapper. bench/provenance.py supplies
native schema/start_run but also historical helpers and an old sealer pin. These
specific couplings remain open, not blanket permission to retain every experiment.

The observability playbook was read fully and distilled to a current methods
and tools guide; its old commands/frontiers/chronology remain in preserved Git.
User DB/replay/archive implementations and their synthetic tests have fully read,
justified roles. Publication no longer imports the old benchmark archive/recorder.

## Verification and finish criteria

Latest selected release check: **490 passed,1 optional skip,2 upstream warnings**,
plus frozen-source/content/compile and isolated no-deps wheel checks. The five
validation-boundary and nine benchmark-facade tests also passed separately.
They check actual named/star/submodule imports, identity/cache/refusal and
stdlib-only bare package loading. This is not a whole-tree CPU pass, fresh full
dependency installation, TPU validation or new model-performance evidence.

Earlier checks remain scoped and overlap:25 host-path/facade/HLO;19 SQLite/
ownership;27 projection/dense with11 unavailable-local-artifact skips;26 CLI/
request with an optional tokenizer skip. Original helper/moved-test ASTs are
preserved. Checkpoint237-entry metadata, local license/reference hashes and prior
relative-link findings were checked offline; no new cloud payload rehash, legal
clearance or checkpoint reconstruction was claimed. Final links/anchors/assets
and applicable retained CPU coverage remain outstanding.

The ledger has zero consistency errors but1,361 unresolved roles. Finish their
semantic review, remove remaining research-only cohorts, verify tests/docs/assets,
and self-review the actual final diff. Tests do not justify an unused experiment.
Merge only after all dispositions close, then verify exact private main and
regional backup under the installed locks. No independent-review or model-run
claim, and no promotion based on a nicer README alone.
