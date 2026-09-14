# Release handoff

## Current resume summary (2026-09-14)

OWNER OVERRIDE 08:23 UTC: stop the active benchmark and prioritize merging the
private supported main. Full benchmark completion/success is NOT a merge gate.
Preserve partial originals and scores, record explicit owner cancellation, verify
original worker/publication termination and idle fleet, then finish the smallest
necessary release smoke and final review/mirror/merge. Do not launch a replacement
benchmark or change TPU infrastructure. Older wait-for-benchmark-seal language
below is superseded by this owner instruction. Stop dispatch/cleanup still need
authoritative receipts; no termination or release success is claimed here.

Worktree glm-tpu-release / release/production-20260914; main and research pins
below remain unchanged. Full63-dependency installation is proved and temporary
environment removed. User worker/controller and separate bounded evidence
transport are implemented/default-off, with CPU tests, but NOT TPU-admitted.
User response semantic replay, outer cold/trace/ownership verification and
DB/archive sealing are implemented with CPU tests. Next: final review and real user
admission after owner-cancelled benchmark originals/publication and idle cleanup. Regional mirror
also waits for its sync lease. No main merge or production-readiness claim yet.
Release parser now supports explicit single-step coverage with no cycle/idle
metrics. The frozen benchmark sealer lacks this option; see last section for
prospective original-only recovery, never a reason to interrupt or rerun it.
Sections below are chronological evidence; earlier open-item lists are historical.

Latest cleanup: seven isolated legacy schedulers/provisioners removed from this
release only; exact Git recovery ledger and preservation/oracle tests added.
Consolidated release check: 399 passed, 1 skipped, 2 upstream warnings. Model
source, current research checkout, installed mirror and main remain unchanged.
Detached exact-published-pin controller admission is prepared for the canonical
checkout; no actual source cutover. See OPERATIONS.md for the post-seal order.
Latest check: 430 passed, 1 skipped, 2 upstream warnings. Explicit user upload-only
recovery is now wired/tested; it never reruns model workers or replaces failed
publication markers. No live user request or recovery has been executed.
Release-only formatting is normalized with Black 25.1.0; all 36 Python ASTs are
unchanged except equivalent indentation in two docstrings. Protected model and
compiler paths are excluded. Existing 430-test result is reused for this cosmetic
change; source guard and isolated wheel install were checked again and pass.
READINESS_AUDIT.md now maps each goal requirement to inspected evidence and the
remaining decisive actions. Recovery wrapper uses the shared detached exact-pin
policy; 40 focused checks pass. Current benchmark remains live as of03:59 UTC.
Controller disk floor restored by verified disposable-copy cleanup: observed
6.26 GiB free, with actual installations/weights/evidence preserved. Recheck all
hosts before launch; receipt docs/release/local-headroom-cleanup-20260914.json.
Explicit release mirror coverage is staged in scripts/release/sync_glm_repositories.sh;
installed script/cron unchanged. MIRROR_CUTOVER.md records hashes and the post-seal
locked installation/verification sequence. Do not execute it while the sync lease
is held by the original benchmark.

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

## User controller and evidence transport

Added default-off scripts/release/launch_ws32_user_request.py and
ws32_user_transport.py. Reused source/branch/census/observer/lease and exact
generation/CRC/SHA storage primitives. Explicit user policy uses distinct schemas
and request prefix, item000only; benchmark defaults/registrations unchanged.
Original answers survive a cold publication failure. Recollection is owner-only,
binds the original request and host/boot, and shares HLO bytes without full copies.
Missing cold records remain explicit incomplete ranks. No score or SUCCESS writer.

Controller admission includes both leases,6GiB free, exact published owner source,
retained RAM/overlay, source/tokenizer, US-CENTRAL2/live<2.5TB/soft-delete-off.
These are checks, never infrastructure/bucket-policy management. Ambiguous SSH
dispatch retains leases and observes originals without redispatch. Prelaunch
failure records worker_started=false; unknown wait never invents process exit.
Attach reuses original input/deadline only. Cold/result semantics still need
independent original-evidence replay before any seal or real release claim.

Tests: user/legacy transport plus worker72passed; controller/failure/transport
30passed, then70passed including the soft-delete guard and old transport suites.
These are CPU/fake-GCS orchestration tests, not TPU or external storage evidence.
02:29 UTC manual check authenticated original benchmark controller PID/start/boot;
8workers OBSERVED,12completed. Research checkout clean. Next poll>=02:39 UTC.

Final controller/transport self-review also added explicit worker0-only lease
ownership and tested replacement-PID refusal. Consolidated release command now
includes the original benchmark transport/collector regressions:291passed/1skip
in21.79s, plus source guard, metadata doctor, content audit and isolated no-deps
wheel install all pass. This is not hardware admission or a complete response
seal. No changes to the live benchmark worktree, main branch or regional mirror.

## User original-response replay

Added ws32_user_result.py: bounded original token/stop-policy replay, strict
all-eight-rank input/result identities, original answer decoding, memory/DSA/cache
recomputation and trace byte/path binding. No benchmark registry or score. First
instrumented decode is excluded from ordinary statistics; no ordinary samples
means no decode-rate claim. Executor now records same-live-session continuation.
Missing/failed/corrupted ranks cannot become a complete response. Cold graph,
physical trace coverage, owner authentication and DB/archive sealing are explicit
separate requirements and still need outer wiring. No new model execution.

Adversarial self-review added bool-versus-integer refusal and path traversal/
nonoutput-delivery checks. Seven replay tests exercise actual user host runtime
and NativeObservability with fake compiled math/counters; no TPU/HBM proof.
Consolidated release check:298passed/1skip/2upstream warnings in24.38s; source
guard, metadata, content audit and isolated wheel install pass. First fixture
failures were a trace path str/Path mismatch and wrong expected exception type
for corrupt NPZ, corrected without changing production validators.

Manual02:53 UTC check authenticated controller2144482/start154030831/original
boot and command; watchdog OBSERVED8workers,12completed. Source research checkout
clean, no restart/new launch. Next manual poll>=03:03 UTC except status/failure.

## Single-observer trace sealing defect found before deployment

Reading the outer replay found a concrete contradiction: NativeObservability
captures exactly one observer call, but parse_xplane.aggregate_fleet demanded
at least two step starts to derive cycle_ms. The native sealer then expected
exactly one step/core. Reproduced that refusal with a tiny CPU fixture before
the fix. No protected model run failed or was restarted for this discovery.

Release-only fix: explicit allow_single_step=True on the native observer replay;
default throughput callers retain strict repeated-step validation. Single-step
cycle_ms, idle_pct and category pct_step_cycle are None, not synthetic duration-
based rates. Host/core/module/step consistency checks remain. Tests cover both
the old repeated E0 geometry and actual serialized synthetic XSpace parsing for
8hosts/64cores/1step; mixed counts/nonmonotone starts and wrong modules refuse.
52focused parser/phase checks pass; consolidated311passed/1skip/2upstream warnings
in25.47s plus source/metadata/content/wheel checks. No TPU/performance claim.

IMPORTANT: active research83f0c272 is unchanged and still has the old parser.
Do not hotpatch source during execution/sealing. If its original sealer refuses
at this boundary, preserve failure/originals and authenticate terminal cleanup,
then prepare reviewed original-only recovery recording BOTH execution and
recovery pins. Never rerun questions, edit scores or waive trace coverage.
The new user outer replay must also request single-step parsing explicitly.

## User protected replay and DB/archive wiring completed (CPU scope)

New ws32_user_evidence.py joins exact input/source/tokenizer, original cold
checkpoint/HLO/HBM replay, authenticated PID/start/boot/argv ownership and
successful ended markers. Original pre/post census and two idle observations
are required. Single-step XPlanes are parsed and bound individually to their
original worker hosts with exact8host/64core/1step coverage; cycle/idle remain
unset. Added file_hosts to the generic parser result to avoid parsing2GBtwice.

ws32_user_database.py links only this user's hashes/timings in existing runs/
summary tables plus a separate idempotency table. No fake benchmark accuracy,
model-card value, extra raw prompt/output copy or full DB export. Atomic rollback,
same-tag retry and altered-linked-row refusal are tested with actual SQLite.
ws32_user_archive.py checks original worker/input generation/size/CRC union,
regional/soft-delete and byte-budget boundaries, exports only this run's logical
rows, then conditionally publishes/readbacks the ledger and USER_RESPONSE_SEALED.
The seal is request completion under its stop policy, NOT task quality/project
completion. Archive interruption retains DB/originals and never regenerates.
The user controller now performs all these stages while holding BOTH leases.

Self-reviewed current diff, not independently reviewed. Forty-nine focused
user replay/DB/archive/controller tests passed; the added archive-failure outer
case then passed in the consolidated suite:341passed/1skip/2upstream warnings
in32.90s, plus model-source guard, metadata/content and isolated wheel checks.
Fixtures use actual host-loop/memory/DSA/cache validators, synthetic serialized
XPlanes, actual SQLite and conditional in-memory GCS. Cold hardware/tokenizer
are explicit mocked boundaries in outer orchestration tests; this is NOT real
TPU/source-cutover/admission evidence and does not close those merge items.

Manual03:16 UTC: controller2144482/start154030831/originalboot+command confirmed
LIVE; watchdog OBSERVED8workers,12completed. Research checkout unchanged/clean.
Next manual poll>=03:27 UTC. No remote model/infrastructure/bucket action or
regional mirror override. Real user launch remains forbidden until the benchmark
terminates and its original evidence is sealed (with recorded recovery if the
known single-step parser defect refuses). Next safe parallel-to-benchmark work:
final supported-tree/dependency/privacy review and deployment cutover preparation.

## Supported script boundary and preserved cleanup

Removed seven superseded top-level legacy scripts after full reads, no executable
incoming-reference findings, and byte-identical comparison to preserved research
83f0c272. The 29,166 bytes remain recoverable in Git; exact paths/blobs/sizes are
in docs/release/removed-legacy-schedulers.json. No running file, weight, evidence,
Git object or installed backup process was deleted. scripts/README.md separates
candidate native entry points from historical tools that may modify environments
or stop processes. Dynamic bench registry and source-pinned oracle dependencies
were inspected and retained, including launch_glm_32chip/validate_ray_network.

66 targeted CPU preservation/inventory/oracle tests passed. The consolidated
check now retains those oracle regressions: 399 passed, 1 skipped, 2 upstream
warnings in 43.94s; original model guard, content scan, metadata and isolated
wheel installation passed. Adversarial self-review only, no independent review.
Broader release admission remains open; these checks are not TPU validation.

03:34 UTC manual observation authenticated original controller PID2144482,
start154030831, boot4ebd122c-7b2a-4388-961f-021fae5f2a52 and exact command.
Latest watchdog03:34:40 observed eight workers; 12 completed requests. Research
checkout remains clean. Next manual poll no earlier than03:45 UTC, absent an
explicit status request or known failure. No source cutover or mirror override.

## Detached exact-pin controller deployment

The named-branch-only controller check conflicted with Git worktree ownership:
release/production-20260914 is already checked out in this isolated worktree.
Allow the canonical controller to be detached at the exact clean code pin while
still requiring the original model guard, private owner origin and matching
current published reviewed ref. A different named branch remains a refusal.
No source/path/checkpoint/HLO waiver, branch movement or live deployment occurred.

70 focused deployment/user-launch/recovery tests passed before the additional
real-Git fixture. The final deployment suite passes 38 tests: the fixture creates
two tiny disposable worktrees and calls the actual clean/pin guard, with remote
service/model-site boundaries explicitly mocked. It proves the original branch
stays checked out and unmoved while the canonical fixture is detached. This is
not actual TPU admission. Self-reviewed origin/ref/pin, dirty/source refusal and
worker sync/attach behavior. Cutover sequencing is now in OPERATIONS.md; actual
cutover, user execution/seal, main merge and regional mirror are still pending.

## Original upload recovery gap fixed before TPU deployment

Adversarial review found that a successful user worker followed by failed
publication could never proceed through attach: the original immutable failed
publication marker always triggered refusal, even if original uploads recovered.
Added explicit controller-only `--attach --republish-originals`. Under both leases,
after original owner observation and fresh idle census, it validates successful
worker exits on all ranks before calling publish-only on the failed ranks. The
existing publisher retains conditional creation/generation/CRC/hash checks.

No prepare or model-worker role is dispatched, no request/deadline is replaced,
and original failed markers remain immutable. publication_recovery.json archives
the failures and successful transport receipts separately. An ambiguous upload
leaves original objects in place and no recovery receipt/seal; a later explicit
attach may retry the same immutable uploads. Worker failure, missing ownership,
changed input, wrong upload receipt or missing cold data cannot become success.
Full collection/cold/request/trace/DB/archive replay remains required. Missing
terminal markers and changed-code recovery still need diagnosis, not this flag.

47 targeted recovery/controller/archive tests passed, then four actual-controller
attach tests with isolated real flock leases passed. The consolidated release
check passes 430 tests, 1 optional skip, 2 upstream warnings in 46.88s; unchanged
model-source guard, metadata, content and isolated wheel installation pass.
Fixtures explicitly mock SSH/cloud/model work; this is not live fault injection.
Self-reviewed current diff, not independently reviewed. INFERENCE/OPERATIONS now
describe supported recovery and refusal actions rather than suggesting reruns.

03:47 UTC: original controller2144482/start154030831/originalboot+command remains
LIVE; watchdog03:47:22 observed eight workers, 12 completed requests. Research
checkout still clean. Next manual poll no earlier than03:58 UTC absent an explicit
status request or known failure. No TPU/source cutover/mirror action performed.

## Release code style normalized without a model refactor

Black 25.1.0 now has an explicit pyproject boundary covering only new release
interfaces, tools and tests; historical compiler/model/oracle sources are outside
it. Reformatted 36 files, leaving two already formatted files unchanged. Exact
before/after SHA256s and AST comparison are recorded in
docs/release/format-ast-check-20260914.json against465650fd. Strict AST comparison
first found two docstring indentation differences; inspection confirmed only
whitespace, and inspect.cleandoc-normalized ASTs match for every changed file.
All non-docstring constants/statements are unchanged. Do not confuse AST equality
with identical source/debug/HLO hashes or deployment admission.

Black --check passes for all38 included files. Original model-source admission
passes. Isolated offline no-deps wheel installation/console/missing-runtime
refusal pass; wheel1,303,387bytes,242members, SHA256
e4c7b58c344d644740db65badbc8faec3cbbe53723bd52eea15ce9e3f488997a.
No full dependency install or TPU tests were repeated for formatting; preserve
the previous430-test semantic receipt. CONTRIBUTING records the formatter scope
and cautions against modifying protected source identities. Active environment,
research execution checkout, weights, main and mirror remain untouched.

## Supported-boundary/readiness reconciliation

Inspected all dynamic-call expressions flagged in the 345-file conservative
closure (24 files): Git identity/source checks, ownership/mount inspection,
existing-host controller/watch transport, pinned benchmark registry/oracle
loaders, historical MoE transport and scanner numerical-helper false positives.
No new model execution or deletion authority follows from this static review.
Known reference/notice headers and both included license hashes were checked;
public-distribution clearance is not claimed. See docs/release/READINESS_AUDIT.md
for direct requirement mapping and unresolved deployment/evidence/mirror work.

Found and removed a redundant named-branch assertion in the RAM recovery shell
wrapper. Shared source_preflight still enforces clean exact published owner pin,
canonical path and model bytes, now consistently allowing detached deployment.
All remaining packing/leases/ownership/absent-target/hash guards are unchanged.
40 focused wrapper/deployment tests and formatting check pass. No pack/reload or
new checkpoint payload was generated. CHECKPOINTS.md now describes the same policy.

03:59 UTC authenticated controller2144482/start154030831/originalboot and exact
command still LIVE; watchdog03:58:53 observed8workers,12completed. Next manual
poll>=04:10 UTC except explicit status/failure. Research remains untouched; no
source cutover, infrastructure action, main merge or mirror override. Next
decisive runtime action still waits for original benchmark termination/sealing.

## Controller launch-floor cleanup

Controller available root space was 4,961,935,360 bytes, below the 6 GiB floor.
Removed exactly 1,752,993,313 logical bytes from three resolved targets: an unused
VS Code 1.134.0 server package (commit110a328e,3,031files), and duplicate build-stage
libtpu.so/libjax_common.so files. Full cmp and SHA256 matched retained installed
/opt/glm-tpu copies; hashes were checked again afterward. All-process cmdline,
environment, mappings, executable/cwd and open-FD inspection found no target
references or inaccessible processes, excluding only the audit's own ancestors.
User ownership and no-symlink membership were checked immediately before removal.

Retained active vllm-env and installed capsules, both live VS Code versions,
extensions/settings/conversation history, model weights, benchmark evidence,
and the 11 GB health-log archive (no second recovery copy established). No cloud
write, full-size backup, infrastructure operation or runtime-source change.
The duplicate libraries can be restored from their exact retained installed
copies; the obsolete server can be reinstalled if needed. Staging directories
are deliberately incomplete now, not falsely claimed intact capsules.

After removal:6,721,028,096bytes available (6.259GiB). All17 installed dependency
metadata entries match; research and release source trees were unchanged by the
deletion. This is controller-only dated disk admission, not fleet/HBM proof.
Exact paths, sizes, recovery sources and limits are in the cleanup receipt.
Do not repeat broad cleanup or delete unique health logs to manufacture space.

## Explicit release mirror pair prepared, not installed

Read the installed sync-glm.sh and cron again. The cron still holds the original
/opt serialization lock and shared home sync lease, and the script still lists
only four old worktrees. Staged an exact versioned copy with only the additional
/home/gianl/glm-tpu-release -> repos/glm-tpu-release pair. Expected predecessor
SHA2568303e37d...cf6b; template7d8d4925...10d6 (full values in MIRROR_CUTOVER.md).
One focused test removes exactly that pair and verifies all other bytes against
the predecessor hash; Bash syntax and formatting pass. No new TPU or cloud action.

Installed script/cron are NOT changed. The template requires the existing caller
locks; it is not an independent mirror launcher. After benchmark terminal seal,
recheck the installed predecessor, obtain original locks, install exact reviewed
bytes and verify checksum equality/object generations for release files AND the
primary shared Git store. Repeat final verification after an eligible main merge.
Do not equate a log line, successful skipped run or shared Git objects with
explicit release-file backup coverage. Main and research remain unchanged.
