# Release readiness audit

2026-09-14. Source review at `48b4d3b09f1e894b280a8948bc9957e6905d7790`,
plus the subsequent narrow RAM-recovery wrapper change documented below.
This is a requirement-by-requirement audit, **not merge approval**.
The repository remains private. Update2026-09-14: DB621 now proves the reviewed
release deployment and ordinary response. The benchmark was owner-cancelled;
the final supported-diff review is complete. Exact main promotion/mirror proof
is in the external final publication record linked from STATUS, not another campaign.
Historical review below retains its original scope and findings.

## Goal requirements and evidence

| Requirement | Evidence inspected | Current conclusion |
|---|---|---|
| 1. Supported engine/imports/assets/entry points | `ARCHITECTURE.md`, `INVENTORY.md`, `scripts/README.md`, static inventory and dynamic-call review below | Mapped; retain conditional diagnostic and source-pinned asset dependencies. Not a standalone wheel-only server. |
| 2. Clean tree, installation, checkpoints, real inference, operations | Preserved-removal ledger; fresh full-install receipt; pinned checkpoint metadata; user controller/executor/replay; DB621; installation/inference/operations/checkpoint docs | Installation, protected canonical release deployment and ordinary response proved. Recovery rebuild was not rerun; retained source metadata and historical pack/loader evidence remain explicit. |
| 3. Quality, dependencies, notices, privacy and errors | Original DB616–620 receipts; campaign registration; 63-package fresh install; vendor/model notices; content/history audit; failure-path tests | No full task-quality/card-parity claim. Known notices, bounded private-content checks and final diff review complete. No public-distribution clearance. |
| 4. Proportionate automated checks | Final431-test CPU pass; source/content/package checks; fresh installation; DB621 protected entry; 7 CLI checks after status-text edit | Local and actual hardware scopes are explicit. No new model test needed for final documentation edits. |
| 5. Accurate speeds/scores/limitations | README table; STATUS; DB621; pinned protocol; original scoring/cancellation evidence | Long-context measurements retained. Cancelled15-question prefix and scorer issue reported without dataset accuracy, parity or benchmark-success claims. |
| 6. Review, push, eligible main merge, regional mirror | Release branch/published pins; retained research ref; installed mirror; pre-promotion checksums | Review and pre-promotion backup pass. Exact final main/private/ref/mirror/idle proof is in the external publication record linked from STATUS. |

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

## Final promotion work

Benchmark cancellation/publication/idle cleanup, canonical source cutover and
bounded ordinary response are complete (DB621). Final supported-diff self-review
covers the actual scope, failure paths and original receipts; no independent
review is claimed. The release/Git-store pre-promotion mirror passed content and
generation checks. The installed template adds only the release pair and checksum
comparison; original guards/locks are preserved and its narrow test passes.
Final published refs/mirror/private/idle proof is linked from STATUS. No new model
run is required; later documentation/status changes do not alter DB621 model math.

No speculative tuning, repeated long-context campaign, public upload, history
rewrite or infrastructure management is implied by this checklist.
