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

## Initial findings (subsequent progress below)

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

## Deployment branch boundary and stale-instruction cleanup

Release-only launcher now defaults to main with explicit reviewed release or
historical research branch support. Controller validates owner origin/branch/
published pin; workers reject wrong origin or changed FETCH_HEAD before checkout.
Attach binds original tag/pin/branch; missing historical branch means research.
Fixed canonical worktree and original model/source/ownership/storage/lease guards
remain. This is not yet a supported main deployment or generic inference CLI.
Research checkout and currently executing launcher remain untouched.

Removed three stale root instruction files (CLAUDE/KICKOFF/PLAN) only from the
release tree after full reads, dependency search and exact original comparison.
All remain recoverable at the published research pin. See INVENTORY.md.

Manual check01:13 UTC authenticated the original controller, still live;12
completed, item012 at19,802 tokens, watchdog OBSERVED. No restart or launch.

Validation:104 CPU tests pass in3.96s across release checks and original native
request/session/memory/launch tests. Shell guards executed with fake git, including
dirty tree/wrong origin/moved ref refusal; attach and historical branch cases pass.
Model-source guard and git diff --check pass. Research checkout remains clean.

## Third-party provenance progress

All three vendored Transformers Python references exactly match corresponding
files in the installed5.12.0 distribution. Added its byte-identical Apache2.0
license and THIRD_PARTY_NOTICES.md; do not infer a blanket project license or
completed public-release clearance. Model snapshot license/revision and legacy
vLLM patch provenance remain unresolved, as does the full privacy/history audit.
Reference hashes: docs/release/third-party-reference-check-20260914.json.

Wheel now carries the third-party notices/license; isolated no-deps installation
and console/refusal checks pass (240members,1,298,683bytes; SHA256
514c2e91f36787b0fc71b4a938301dd7533fe9e11892a7f9ac0e4ee8b1f7b34b).
No active dependency installs, TPU work or model changes. Full environment install
still pending; controller has~5.1GiB free, so avoid large duplicate environments.

## User request integration (not launch-admitted yet)

Added glm_tpu/user_request.py, prepare-request CLI, and a separate
scripts/release/ws32_user_request.py worker executor. Explicit schema/hashes,
fixed sampled166912capacity/temperature1/top-p.95, bounded full generation budget,
private exclusive0600 output outside source checkout. No benchmark gold/scorer.
Executor reuses native start_request/TokenSink/store/memory/observability/session;
retains failures and never retries donated or partially delivered requests.
Native runtime/model sources unchanged. See docs/release/INFERENCE.md.

Next: protected user controller/worker/publication wiring, actual site/asset
admission and minimal real execution AFTER benchmark termination and sealing.
Preparation and CPU fixtures are not a standalone service or TPU proof.
Checkpoint recovery recipe, full install, privacy/provenance and mirror remain.

Manual observation01:25 UTC: original controller identity authenticated; eight
workers OBSERVED,12completed, item012 at24,395tokens. No interruption/restart.

Checks:144CPU tests passed, one optional tokenizer test skipped in the general
suite; the explicit real-local-tokenizer suite separately passed20tests in8.57s.
Two upstream SWIG deprecation warnings, no failures. Native model-source guard
passes. No tokenizer download, private dataset access or new model execution.
