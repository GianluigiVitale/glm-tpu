# Release readiness audit

2026-09-14. Source review at `48b4d3b09f1e894b280a8948bc9957e6905d7790`,
plus the subsequent narrow RAM-recovery wrapper change documented below.
This is a requirement-by-requirement audit, **not merge approval**.
The repository remains private; main and the running research source are unchanged.

## Goal requirements and evidence

| Requirement | Evidence inspected | Current conclusion |
|---|---|---|
| 1. Supported engine/imports/assets/entry points | `ARCHITECTURE.md`, `INVENTORY.md`, `scripts/README.md`, static inventory and dynamic-call review below | Mapped; retain conditional diagnostic and source-pinned asset dependencies. Not a standalone wheel-only server. |
| 2. Clean tree, installation, checkpoints, real inference, operations | Preserved-removal ledger; fresh full-install receipt; pinned checkpoint metadata; user controller/executor/replay; `INFERENCE.md`, `OPERATIONS.md`, `CHECKPOINTS.md` | Installation and local checks proved. Canonical release cutover and a real ordinary user response remain unproved. |
| 3. Quality, dependencies, notices, privacy and errors | Original DB616–620 receipts; active campaign registration; 63-package fresh install; vendor/model notices; content/history audit; failure-path tests | No full task-quality/card-parity claim. Known notices and bounded private-content checks are documented; final diff review still required. No public-distribution clearance. |
| 4. Proportionate automated checks | 430-test CPU result before formatting; exact formatting AST receipt; post-format source guard/package install; 40 focused detached/recovery checks | Local evidence is explicit. No CPU fixture substitutes for changed-entry TPU/HLO/HBM evidence. |
| 5. Accurate speeds/scores/limitations | README result table; STATUS evidence scope; pinned native protocol; original scoring audit | Long-context measurements retained. Active campaign has no final seal; its partial scores and scorer issue must be reported separately after collection. |
| 6. Review, push, eligible main merge, regional mirror | Release branch/published pins; unchanged main/research refs; installed mirror configuration | Release commits pushed. Main not merged. Explicit release mirror coverage still needs installation/verification after the active sync lease is released. |

The 128K and 256K evidence need not be regenerated for repository cleanup.
Likewise, do not delete the current RAM weights to manufacture a fresh recovery
test. The metadata audit and historical pack/loader evidence have narrower scopes
than a new reconstruction; state that distinction in the final release notes.

## Dynamic dependency boundary

The current scanner roots are the local CLI, user controller, native benchmark
controller and original owner worker. Their conservative static closure is
**345 Python files**, with dynamic-call hints in **24 files**. These include
conditional historical diagnostics, not 345 modules executing every user request.
All reported dynamic-call expressions were inspected; categories are:

- Git identity/source checks: provenance, original worker and compiler/admission
  helpers use `rev-parse`, `status`, `diff --quiet` or captured-file blob checks.
  They depend on retained Git history; a shallow source export is insufficient.
- Ownership/mount checks: `fuser` and `findmnt` inspect existing owners/mounts.
  `fuser` here is not invoked with `-k`; these checks do not kill processes.
- Controller/watch transport: authenticated SSH to the existing eight-host pod,
  owned worker subprocesses, and CPU-only original publication. The supported
  user controller holds both leases; its upload recovery dispatches publish-only.
- Native registry: explicit file loading of pinned `bench/extract.py` and
  `bench/benchmarks.py`, not the legacy execution launcher. These files must stay.
- Historical long-context oracle: pinned CPU builders/extractors and their sibling
  files are loaded by filename. That separate oracle path includes a legacy
  `engine.py` helper dependency; its model-execution packages are guarded. Do not
  mistake its presence for authorization to run legacy inference in the native
  user path, or remove it without preserving the historical oracle workflow.
- Historical MoE campaign SSH: reachable through diagnostic evidence helpers,
  not a supported user-request command. Keep it classified as historical tooling.
- Scanner false positives: `check_output(result, expected)` and the budget
  overhead helper's `.run` are numerical validation calls, not subprocess calls.

Three reference-extract relative-import errors remain reported by the scanner.
Those files are not standalone installed native modules. Static scanning alone
does not prove absence of every dynamically computed path, or authorize deletion.
The seven removed schedulers had separate full reads, incoming-reference review
and exact preserved Git blobs; that evidence is not generalized to other files.

## Notices and private-content scope

Known Transformers extracts and their Apache license were compared to the local
5.12.0 distribution; the model snapshot notice is pinned to its exact HF revision.
Current license hashes still match the recorded checks:

- Apache-2.0: `77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049`.
- GLM model MIT: `f4a18c6ae40b0a8e7d2b7667f52f6e1994e54a46430d2e172b73cb8c9b5eb0d7`.

A source/header search found the three known reference copyright headers and
internal-code adaptation notes, not a newly identified third-party extract. This
search is not legal clearance or proof that every historical source's provenance
has been established. The unused external patch is absent from this release but
preserved in Git. No blanket license for the owner's original code was invented.

Known credential-format/history scans and private-payload filename checks have
explicit limits in SECURITY.md. Synthetic passkey receipts are retained; raw
questions, responses, weights and DB payloads belong outside Git. Do not turn a
negative pattern scan into an assertion that future public release is approved.

## New finding resolved in this review

The RAM-checkpoint wrapper still required a named branch before calling the
shared exact-pin source guard. That contradicted the reviewed detached deployment
policy and could prevent recovery while the branch stayed in another worktree.
Removed only that redundant check. Canonical path, cleanliness, source bytes,
published owner ref, both leases, absent target, full slot hashes, storage budget
and worker ownership remain enforced. The wrapper is still default-off and was
not executed. Its tests plus actual-Git detached admission tests pass 40/40.

## Outstanding decisive work

1. Let the original benchmark terminate and seal its original evidence. If the
   known single-step parser issue refuses, use recorded original-only recovery;
   do not change executing source or rerun questions.
2. Restore any genuine disk admission shortfall using verified expendable local
   copies, then follow the documented leased canonical source cutover.
3. Execute and seal one bounded ordinary user request with real cold/graph/HBM,
   continuation, first-token/stop, original trace and authenticated cleanup proof.
4. Finish the final supported-diff review and report actual benchmark scope,
   including partial counts, extraction defects and protocol limitations.
5. Verify explicit regional release-mirror coverage; then merge only an eligible
   release to private main and verify the published refs and mirror again.

No speculative tuning, repeated long-context campaign, public upload, history
rewrite or infrastructure management is implied by this checklist.
