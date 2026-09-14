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

## Retained checkpoint/recovery audit

Read-only GCS check authenticated sealed manifest/SUCCESS/source ledger and found
all141 canonical weight objects at their original sealed generations/sizes/CRCs:
755,632,050,320bytes.96dense-overlay files present at sealed sizes:2,102,200,128bytes;
overlay payloads NOT rehashed. WS32 RAM runtime files total786,181,673,984bytes.
No weight payload download or repack. Exact object metadata in
docs/release/checkpoint-recovery-metadata-20260914.json; reproduce with
tools/check_checkpoint_recovery.py. See CHECKPOINTS.md for retained paths, direct
canonical-to-tmpfs recipe and the distinction from the old full-GCS pack wrapper.

Found the old RAM recovery wrapper held only the workload lease. Release-only
fix adds sync lease, explicit reviewed main/release ref and owner/published-pin
checks, worker FETCH_HEAD/origin checks, CPU-only pack environment and regional
storage-budget admission. No execution of the enabled wrapper; current weights
and research checkout untouched. It remains default-off pending deployment review.

First focused suite exposed an existing stale assertion expecting TWO DSA records
in unchanged short-runner code that already has THREE. Updated the exact expected
count/set to include the existing canonical8K registration; kept every SHA check.
No worker/sealer/record change.36focused recovery/short-runner CPU checks passed.
Manual01:36 UTC: original controller live,8workers OBSERVED,12completed;
item012 at28,652tokens. Cadence remains>=10min.

Final recovery/release suite:98passed/1optional-local-tokenizer-test skipped in
2.49s; git diff --check passed. Metadata-only cloud check matched all required
source objects. No enabled pack, source payload read, deletion or TPU action.

## Content, history and release-check audit

Read-only scan of all10,281 local Git blobs (1,496,886,163 uncompressed bytes),
including unreachable objects, found no matches for nine known credential rules
and no skipped/error blobs. Tracked snapshot1,940files/35,904,306bytes had no
findings. Exact scope/pins in docs/release/content-audit-20260914.json.
Git fsck passed. An expanded reachable-path filename check also found no matches.
This is heuristic evidence, not universal private-data/security clearance.
Eight historical gold fields were confirmed synthetic passkey evidence and kept.

Added SECURITY.md, private-output ignores, tested location-only content checker
and one offline CPU release command tools/check_release.py. Use pytest, not only
unittest discovery, to cover the actual release tests. No secrets/weights/private
dataset download or history rewrite. Do not repeatedly rescan unchanged1.5GB
history; check new tracked content and retain the original receipt.

Publisher MIT license copied byte-for-byte at model revision
f33c6dc501ee5a2c7e35155653b1b1abbc320951; four configuration/template files match
that revision. Historical reference README differs and is preserved, not silently
replaced or treated as the protocol card. Notice and hash receipt included;
legacy patch provenance and broad public-release clearance remain open.

Removed the unused legacy vLLM patch from the release tree after full read,
repository-wide dependency search and exact blob comparison against preserved
research83f0c272. No code/test/config consumer exists; remaining mentions are
historical prose. Recovery blob/path recorded in INVENTORY.md and notices.
This does not erase it from history or authorize public distribution.

Consolidated release check passed:195CPU tests/1optional tokenizer skip, source
guard, dependency metadata, content scan and offline isolated wheel installation.
Three additional orchestration tests pass for timeout/error refusal and scope
reporting. Current-tree audit passes after legacy-patch removal. No full fresh
dependency install: installed packages occupy4.5GiB and controller has only5GiB
free, so a duplicate environment would exhaust safe disk headroom. Do not modify
the active environment to manufacture an install pass. Real protected user launch,
benchmark terminal/seal and regional mirror remain required before main merge.

## User worker wiring

Added scripts/release/ws32_user_worker.py and original owner-script dispatch for
--user-request. New user namespace is distinct from benchmark registration and
publication. Exact retained site recipe, immutable request-file hash, owner-only
paths, default-off gate, original clean/source checks, pinned tokenizer/template,
topology initialization and native load_runtime all retained. No model math,
original compiler builders, sampling policy or benchmark entry changed. Extended
cold native_root routing ONLY to enforce original byte caps on user evidence;
benchmark _identity/publication still rejects user tags.

88CPU worker/request/original transport tests passed with1optional tokenizer skip;
the final user-worker suite separately passes21tests, including both successful
actual host-executor wiring with fake compiled math and early loader failure.
No hardware execution, admission waiver, generic user controller or user seal yet.
The active research checkout remains untouched; release source is not deployed.

## Fresh full installation proved without filling the root disk

Resolved the disk-only install constraint using a disposable /dev/shm directory:
103GiBtmpfs free and264GiBhost MemAvailable observed before the check. Installed
all63 pinned dependencies plus the project using the explicit CPU torch index.
uv pip check,17doctor metadata entries and actual CPU imports pass; installed
glm_tpu import was verified outside the checkout.78focused tests/1optional skip
pass in that fresh environment. End-of-test scratch delta2,400,190,464bytes;
the entire temporary environment/cache was removed. No active environment
upgrade, checkpoint/model payload read or TPU execution. Reproducible tool:
tools/check_release_install.py; receipt docs/release/fresh-install-20260914.json.

Latest consolidated offline release check:219passed/1skip, original model-source
guard, content scan and isolated wheel installation all pass. Full installation
is no longer a release blocker. Protected user controller/publication/real TPU
admission, final benchmark seal, cleanup and regional mirror remain open.

Manual02:13 UTC observation authenticated original controller PID/start/boot;
watchdog OBSERVED8workers,12completed requests. Still live/unsealed. The install
scratch directory is gone; tmpfs remains103GiBfree and root disk5GiBfree. Next
manual benchmark check no earlier than02:23 UTC absent explicit request/failure.
