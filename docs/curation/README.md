# Curation ledger and review boundary

Objective: [CURATION_PLAN](../release/CURATION_PLAN.md). This index publishes
the before/after counts, the per-file disposition ledger, the removal cohorts
with their review basis, exact recovery, and what the verification did and did
not establish. It is self-review by one agent, not independent review.

## Before and after

The table below records the September15–16 curation milestone. The September20
ordinary release adds the justified optimized implementation, tests, instructions
and compact evidence; its final ledger has670tracked files and2050dispositions,
with no unresolved entries. Frozen source and exact recovery remain unchanged.
The September21 four-conversation release has680tracked files and2060dispositions;
its ten added paths are the batching implementation, tests, documentation and
compact hardware receipts. No original research files were removed in this merge.
See [current release status](../release/STATUS.md) and
[integration history](../perf/ordinary-release-20260920.md) for the added scope.

| | Files | Payload bytes |
|---|---:|---:|
| Starting main `b667f00f1ae48c8ff37e92500550c1395d74c66d` | 1,971 | 36,166,207 |
| Removed originals (`action: remove`) | 1,380 | 27,898,971 |
| Retained originals | 591 | 8,267,236 |
| New paths created during release/curation | 23 | see ledger |
| Curated code tree `cc2b36de` (whole-tree CPU run) | 614 | 10,817,830 |
| Final main tip (adds this index, two CPU receipts and the content-audit cap fix) | 616 | `git ls-tree -r -l main` |

Retained originals by category: supported implementation/configuration 148
(2,293,726 B); required dependency or compact evidence 190 (3,344,478 B);
relevant test or documentation 253 (2,629,032 B). The 23 new paths are 6
supported (host operations, SQLite adapter, path guard, extracted native
helpers), 1 dependency (the ledger checker) and 16 tests/documents. Review
kinds recorded: 524 full reads bound to the current SHA-256, 89 generated
receipts validated by structure/provenance/consumer, one manifest validation
(the ledger itself).
Zero unresolved dispositions. Counts describe the ledger, not quality.

## Every-file accounting

[disposition.jsonl](disposition.jsonl) has one row per starting-pin path plus
one per new path: original Git blob/mode/size (`baseline`, null for new files),
purpose, consumers, category, action, justification and `review`
(`kind`, `sha256` of the reviewed bytes, `notes`). Removed rows carry
`recovery = {commit, path, blob}` at the starting pin. Retained `.py/.sh/.md/
.toml/.jinja` files require `full_read`; JSON/NPY/other receipts use
`generated_validation` with the parsed artifact kind and consumers in the notes.

```bash
# Ledger consistency: coverage, baseline identity, removal implementation,
# recovery rows, review kinds and current bytes. Exit nonzero on any error.
JAX_PLATFORMS=cpu python tools/curation_inventory.py
```

The checker proves ledger consistency. It does not prove that the recorded
semantic review was correct, that no dynamically computed path exists, or
anything about model correctness. `tests/release/test_curation_inventory.py`
pins the checker's refusals.

## Supported boundary

The dependency roots are the local CLI (`glm_tpu/cli.py`), the user controller
(`scripts/release/launch_ws32_user_request.py`) and the user worker
(`scripts/release/ws32_user_worker.py`). `tools/release_inventory.py` follows
imports, the literal lazy-export maps of the four package facades, subprocess
sites, literal tracked assets and file reads from those roots. At the final pin
the conservative static closure is 185 Python files (down from 333 at the
starting pin) with 21 files carrying dynamic-dispatch sites, each reviewed:
Git identity checks, `fuser`/`findmnt` ownership probes, controller SSH, the
pinned `bench/` registry loads and the lazy facades.

Separations made without changing numerical execution or admitted identities:

- Host admission/SSH/original-process/publication helpers moved verbatim from
  the retired campaign launcher into `scripts/release/ws32_host_ops.py`;
  transaction/export/path helpers into `scripts/release/ws32_sqlite.py`; the
  stdlib path guard into `glm_tpu/host_paths.py`. AST identity is pinned by
  `tests/release/test_host_ops_boundary.py`, `test_sqlite_boundary.py`,
  `test_host_paths.py` and `test_native_helper_extraction.py`.
- The helpers the native worker executes from the layer-6 prefill campaign
  (BudgetedCalls, memory-owner validation, trace voting, compiler originals,
  WK program builders, the canonical-dense source override) moved verbatim
  into `scripts/greenfield/ws32_budgeted_calls.py`,
  `ws32_compile_originals.py` and `ws32_dense_canonical_source.py`; the
  sealer's enforcement surface names the new homes.
- The benchmarking, checkpoint, partitioning and validation packages are lazy
  facades whose retained names keep their original module targets
  (`tests/release/test_*_import_boundary.py`).

Retained historical coupling, each explained in its ledger row: the six
SHA-pinned legacy harness modules in `bench/` (registry, extractors, schema,
passkey/E0 prompt builders, engine recipe) that the native protocol, the
long-context oracle loader and the sealer pin; the DB485 compile-only family
and its two shell wrappers; `scripts/greenfield/run_short_decoder_ws32.{py,sh}`
and the sealer, which the user worker loads and the sealed runs declare; the
history/dense-frontier/delivery preparation modules the native cold replay
executes; and the 74 receipts that retained code pins or reads or that form
the DB616–621 admission chain. Frozen `MODEL_SOURCE`
(`glm_tpu/greenfield/{kernels,runtime,sharding}`, `types.py`, the model
config) is pinned by the admission registry at `edecdd94` and was not edited.

## Removed cohorts and review basis

Every removed original was compared byte-for-byte with the starting pin before
removal and has a recovery row. "Full read" means the file body was read;
"scope/consumer" means header, role, incoming references and closure were
established without reading every line.

| Cohort | Files | Basis |
|---|---:|---|
| Pre-session removals: fork kernel entry points, review transcripts, PR/recon material, PP16 forced-round experiment, research journal, projection-contraction orchestration, RMS/M2048/DB518 diagnostic chain, unconsumed receipts, superseded plans, legacy Ray/staging/backup workflows, Gate D compensated/tuple/provisioning workflows and receipts, PP16 feature2 workflow and receipts, benchmark-only DB/archive, PP transport diagnostics | 568 | Mixed; recorded per row (executable files fully read, prose by scope/headers, receipts by parsed kind). Chronology at `251d9d0b` |
| PP-era checkpoint pipeline, PP8 decoder compile chain, Gate C proof, Gate-D layer-0 legacy-engine capture diagnostics, DB550 StrategyND/collective replays, GCS reclamation, npz observability, 2K logprob capture (`cf6a92bf`) | 242 | Closure from supported roots, incoming-reference search, header/scope review; dependent facades and tests fully read |
| Layer-6 prefill campaign web: window/completed-window/phase/rolled/router/prefix-MLP/budget/MoE-scaling/dense-frontier/dense-norm/canonical-dense/history-frontier launchers, modules, 107 tests and one fixture (`c7f6d42f`) | 215 | Native helpers first extracted verbatim and pinned; then closure/consumer review of the remaining campaign modules |
| Legacy vLLM bench runner/reporters and tests, engine parity harness, PP-only Pallas microbench proofs, eviction tooling, PP recovery/sealing utilities, layer-0 probes, one-layer WS32 micro-benchmark family (`78fead49`) | 110 | Closure and incoming-reference review; retained facades/tests fully read |
| Unreferenced receipts (202), retired-tooling configs (9), superseded documents (4) (`c1cec5c6`) | 215 | Receipts parsed for kind/consumers with zero remaining filename/SHA readers; seven items restored when tests showed kept code needs them |
| Window-admission optimized-HLO inspection modules and test (`17690311`) | 4 | Native surface kept AST-identical; retired inspection fully read |
| Gate-D evidence mirror trio and tests, FP8 microbench leftovers, destructive dump helper, VM cleanup journal (`a23b1ff0`) | 13 | Full reads; installed mirror and retained receipts verified to not depend on them |
| Campaign tests whose subject modules had already left: layer-6 prefill, dense-frontier, history-frontier, StrategyND-probe and PP-era decoder equivalence tests whose module-level or subprocess-string imports no longer resolved (`cc2b36de`) | 17 | Full reads in the cohort review; found by statically resolving every project import in retained Python and shell files, including code inside subprocess strings, after the whole-tree run reported them |

Recover any original without restoring an obsolete workflow onto main:

```bash
git show b667f00f1ae48c8ff37e92500550c1395d74c66d:<path>
```

Only candidate-tree copies were removed. Original results, research branch
`83f0c2728d0d418255a917343cc89d24b815bd0c`, external evidence, DB616–621,
weights and the canonical detached checkout are untouched. Historical
documents retained on main carry a dated curation note where they name files
that left; those names are recovery references, not commands. The one prose document
whose bytes retained code pins by SHA-256, `docs/greenfield/PREFILL_PERFORMANCE_TARGETS.md`
(a prerequisite of the rolled-short admission registry), stays byte-identical
to the starting pin; its mention of the retired `PREFILL_COST_MODEL.md` is
therefore annotated here rather than in the file.

## Verification

- Ledger checker: complete, zero errors, zero unresolved.
- Release check (`tools/check_release.py`): passed on 2026-09-16 (pytest step 524 passed, 1 skipped; doctor, content audit, frozen-source pin `edecdd94`, compileall and isolated wheel install all clean).
- Whole retained CPU tree (`pytest tests scripts/analysis bench -rfEs`): 2,832 passed, 122 skipped, 173 failed, 0 errors in 1 h 30 min on code pin `cc2b36de`.
  Every failure also fails at the starting pin `b667f00f` (186 there versus 173 here; none introduced by curation): admission tests bound to historical sealing-source pins and a few sealed-identity assertions, left unedited. [TESTING](../release/TESTING.md) lists the modules
  that skip or fail without local sealed evidence, the pin-bound admission
  tests and the three history-dependent tests.
- Static inventory: 185-file closure, zero secret-pattern findings; the three
  vendored Transformers reference extracts are still reported as unresolvable
  relative imports (they are references, not installable modules).
- Links: every relative link in retained Markdown resolves; the remaining
  backticked file names that do not exist on main are inside historical
  documents whose curation notes say so.
- Import resolution: every project import in retained Python and shell files,
  including code inside subprocess strings, resolves on main; every SHA-256 pin
  of a tracked path in retained code was checked against current bytes, and the
  only pinned file curation had touched (`PREFILL_PERFORMANCE_TARGETS.md`) was
  restored to its exact starting-pin bytes.
- Documented offline commands (`python -m glm_tpu info`, `doctor --profile core`,
  `tools/release_inventory.py`, `tools/curation_inventory.py`) run from this checkout.

Not established: independent review, new hardware results, model quality,
portability beyond the documented site, or that every conditional diagnostic
module is exercised by a user request. See [STATUS](../release/STATUS.md).
