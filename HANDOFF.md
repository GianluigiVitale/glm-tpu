# Release handoff

## 2026-09-14 — isolated preparation started

Owner authorized cleaning and merging private `GianluigiVitale/glm-tpu`, with
production on main and research on other branches in the SAME repository.
No upstream publication or visibility change is authorized.

- Worktree `/home/gianl/glm-tpu-release`, branch `release/production-20260914`.
- Starting research `83f0c2728d0d418255a917343cc89d24b815bd0c` (published).
- Starting main `a4a17ac4e90b15f1994bd8b26917ef62daa52660` (research ancestor).
- GitHub API confirmed private/default main on2026-09-14.
- Original1.64MB research HANDOFF remains at the research pin; this compact
  release handoff does not replace or erase its historical evidence.

Benchmark LIVE at original research pin. Read-only check00:43 UTC authenticated
controller PID2144482/start154030831/boot4ebd122c-7b2a-4388-961f-021fae5f2a52.
Eight workers observed;12 requests completed,9 marked correct; item012 active.
This prefix is NOT full-dataset quality. One earlier scorer false negative is
documented outside frozen source in
`/home/gianl/glm-run/native-benchmark-scoring-audit-20260913.md`.
Do not modify campaign/scoring, interrupt or replace it. Poll >=10 minutes apart
except explicit status requests/failures. Source stays frozen through sealing.

## Next

Complete conservative import/asset inventory before pruning experiments. Current
launcher binds research branch, private capsule and local paths: copying onto
main is not a supported main deployment. Separate configuration while retaining
protections. No packaging/dependency file or root license existed at the start.
Licensing needs provenance/notice review, not an invented blanket license.

See [STATUS](docs/release/STATUS.md) for merge criteria. Original research tree
is untouched. Release checkout ~40MB, no checkpoint/large evidence copy. No main merge.

## First release changes and checks

README/goal/AGENTS/HANDOFF now describe release work rather than old launch
queues. Architecture, operations, branch policy and merge checklist added.
Read-only AST inventory reaches333 Python files from launcher/worker;23 contain
dynamic dispatch. Literal runtime dependencies include historical receipts:
do not remove artifact directories blindly. Full result and limits in
docs/release/INVENTORY.md and initial-inventory.json. Eight stdlib tests pass;
git diff --check passes. No existing Python/model/enforcement source changed.
Root goal is3450 characters. Licensing, dependency installation, true main
deployment, broader tests/privacy review and benchmark seal remain open.

Mirror cron currently names four old worktrees, not this new release worktree.
It also owns the shared Git store via the original checkout, but that does not
prove release-file mirror coverage. Add/verify explicit release-tree coverage
only after the current benchmark releases the sync lease. Do not bypass it.

## Packaging and dependency progress

Added private alpha wheel metadata, explicit dependency profiles, CPU-only torch
index, offline `glm-tpu info/doctor`, and observed 63-package constraints. The
complete pinned closure resolves against public indexes; no active environment
install/upgrade. Initial global CPU-index resolution shadowed requests; fixed
with uv explicit torch-only source, not unsafe-best-match. Offline resolution
alone originally lacked cached metadata; that was not a model/dependency defect.

Isolated wheel build/install (no deps, no network) and console from outside the
checkout pass; missing-runtime refusal passes. Full fresh dependency installation
and real release deployment remain open. Native model-source guard passes
unchanged. 76 CPU checks pass in4.56s: release CLI/inventory, original request
session/runtime, native memory and launch guards. No model/kernel/worker source
changed. Reproduce via tools/check_release_package.py and INSTALLATION.md.

Latest manual benchmark observation 00:59:01 UTC: original controller identity
verified, eight workers observed,12 completed/9 originally correct, item012 at
14,435 tokens. Still live/unsealed; no quality completion claim. Next poll >=10min.
