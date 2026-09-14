# Release status

2026-09-14: **release preparation; main merge not admitted**. Historical receipts
are preserved. This page distinguishes demonstrated behavior from unfinished work.

## Validation

| Capability | Evidence | Limit |
|---|---|---|
| Full short decoder | DB567/§21.6; DB610 corrected batched 2K | Not broad task quality |
| 128K passkey | DB616–619, all four depths | Four protected retrieval prompts |
| Full 256K E0 | DB620 | Capacity/performance; NO_CORRECTNESS_ORACLE |
| Native sampled requests | Active campaign at 83f0c272 | Final seal/quality incomplete |
| Resume | Same live session/cache/RNG | Not crash recovery/persistent KV |
| First token | Local rank0 JSONL write/flush | Not client-network latency |
| Concurrent requests | Not implemented/proven | One active sequence |

Receipts under `docs/artifacts/`:

- `prefill-canonical-short-db610-sealed-20260909.json`
- `prefill-delivery-db616-sealed-20260912.json`
- `prefill-delivery-db617-sealed-20260912.json`
- `prefill-delivery-db618-sealed-20260912.json`
- `prefill-delivery-db619-sealed-20260912.json`
- `prefill-delivery-db620-sealed-20260912.json`

Receipts record code/DB/archive/cleanup bindings. Reading them is not a fresh
external-original replay. Research contracts: `docs/glm-tpu-revolution.md` §25/§26.

## Quality campaign

198 GPQA-Diamond + 30 AIME2026 questions; one draw/item; temperature 1/top-p 0.95;
163,840 maximum generated tokens; capacity 166,912. Twenty-four hours is an
operational tranche, not a full-set ETA. Reasoning can take hours per question.
Incomplete sets are INCONCLUSIVE. GPQA extraction has a full-option versus
letter issue: retain original scores and audit any revised scorer separately
over all eligible saved answers. AIME judging needs specific paid authorization.
Release cleanup grants none. Protocol gaps prevent matched model-card parity.

Prospective sealing issue found during release review: the frozen native
observer captures one step, but its generic fleet parser requires at least two
to calculate a cycle interval. The release branch adds explicit single-step
coverage parsing with cycle/idle metrics left unset; ordinary throughput callers
remain strict. The running research controller/source is NOT changed. If the
original sealer refuses at this boundary, recover the original artifacts after
terminal cleanup with a separately recorded reviewed sealer pin; never rerun
model questions or substitute a device duration for wall throughput.

## Merge checklist

- [ ] Supported code/import/asset closure; safe obsolete-file pruning.
- [x] Fresh pinned dependency installation, CPU imports and focused host tests.
- [ ] Real user inference entry point and protected deployment validation.
- [ ] Main deployment independent of frozen research checkout.
- [ ] Checkpoint preparation/loading and recovery instructions validated.
- [x] Automated CPU and release checks pass in release worktree (not TPU admission).
- [ ] Third-party notices/licensing and secret/private-data audit complete.
- [ ] Native terminal evidence collected; score scope/limitations documented.
- [ ] Request/resume evidence and operational limits documented.
- [ ] Final adversarial self-review resolves material findings.
- [ ] Release pushed, eligible main merge and regional mirror verified.

Do not merge merely because the README is polished or a partial score looks
promising. Do not repeat DB616–620 for this checklist.

Initial findings and reproducible audit commands: [inventory audit](INVENTORY.md).
Requirement-by-requirement evidence and outstanding work:
[readiness audit](READINESS_AUDIT.md). This is not merge approval.

Packaging progress: the private alpha wheel installs and its console works
outside the checkout. A fresh isolated environment installed all 63 observed
dependencies; dependency checks and actual CPU imports passed, then 78 focused
tests passed with 1 optional tokenizer skip. Temporary setup was removed; active
environment unchanged. Real main deployment remains pending. The latest offline
release suite passes 430 tests with 1 skip, including original benchmark transport,
preserved oracle guards and new user controller/transport cases.
See [installation scope](INSTALLATION.md).

Deployment preparation: explicit reviewed main/release branch selection, remote
origin/pin checks and attach branch identity now replace the research-only branch
binding in this worktree. Canonical site/path admission remains; no release launch
has occurred. Three stale root research instruction files removed with preserved
Git originals. See [operations](OPERATIONS.md) and [preservation](INVENTORY.md).
Controller admission now also accepts a detached canonical checkout at the exact
clean published reviewed pin, avoiding Git worktree branch conflicts without
moving research/release refs. Actual Git fixture and rejection tests pass; the
documented post-seal cutover has not been executed.

Provenance progress: all three Transformers reference extracts match installed
5.12.0 files; the matching Apache license and third-party notices are preserved
and included in the wheel. Its isolated install/console checks still pass. Model
snapshot notice is now pinned and included: four configuration/template files
match revision f33c6dc501ee5a2c7e35155653b1b1abbc320951 and its MIT license;
the earlier saved README is preserved unchanged. The unused legacy vLLM patch
was removed from the release with a preserved Git recovery location; historical
public-distribution clearance and broader privacy review remain open. This is not a blanket project license
or public-distribution clearance.

User inference integration: separate bounded prompt format, local pinned-tokenizer
preparation CLI, and single-request worker executor reuse the actual native host
runtime without benchmark validation/scoring. CPU fake-math coverage is not TPU
admission. The default-off user worker now wires the original loader and executor,
with private namespace/source/tokenizer admission and existing cold-write caps.
Protected user controller and separate transport are implemented with CPU
orchestration/byte-roundtrip tests, not actual deployment evidence. User semantic
replay now checks actual tokens, stop policy, all-rank agreement, memory/DSA/cache
originals and separate ordinary/instrumented timings. Outer cold/physical-trace/
ownership verification and idempotent DB/archive sealing are now wired and
CPU-tested, including real SQLite, conditional in-memory GCS and synthetic
serialized XPlane parsing. Real execution admission remains open; see
[inference scope](INFERENCE.md).

Upload recovery: explicit same-tag `--attach --republish-originals` now retries
only failed publication roles after successful original worker exits and idle
ownership checks. Original failed markers remain unchanged; a separate recovery
receipt is archived. Collection/replay/sealing are still mandatory. CPU tests
cover both-lease ownership, ambiguous uploads, refusal and marker preservation;
this has not been fault-injected on the live pod.

Checkpoint audit: all 141 canonical weight objects match original sealed GCS
generations/sizes/CRCs; 96 overlay files have expected sizes. Direct canonical-to-RAM
recovery is documented, and the release recovery wrapper now enforces both leases
and reviewed owner refs. No new pack/load or overlay payload rehash was performed.
See [checkpoint scope and retained paths](CHECKPOINTS.md).

Content audit: all 10,281 local Git blobs (1,496,886,163 uncompressed bytes)
were scanned with nine known credential-format rules, with no matches or scan
errors. Git fsck passed. This bounded heuristic is not comprehensive clearance;
see [security scope](../../SECURITY.md) and the dated content-audit receipt.
Raw request/answer filenames are now ignored and rejected by the current-tree
audit. The single offline `tools/check_release.py` command covers selected CPU
tests, source admission, content checks and isolated no-deps package installation.

Supported-tree cleanup: seven obsolete legacy schedulers/provisioners removed
from this release only, with exact recoverable Git blobs and dependency review.
The script index distinguishes candidate native commands from historical tools.
Dynamic benchmark registry and source-pinned oracle dependencies remain intact;
static reachability alone was not used to authorize deletion. See the
[removal ledger](removed-legacy-schedulers.json) and [inventory](INVENTORY.md).

Style cleanup: Black 25.1.0 now has an explicit release-only boundary. All 36
reformatted Python files retain equal ASTs, allowing only docstring indentation
normalization in two files. Model/compiler/historical source bytes are excluded;
their original source guard passes. The existing 430-test semantic result is
retained, not presented as a new hardware result. See
[the exact formatting receipt](format-ast-check-20260914.json).

Controller disk headroom: removing one unused old VS Code server and two verified
duplicate build-stage libraries raised available root-disk space to 6,721,028,096
bytes (6.26 GiB), above the 6 GiB launch floor. Installed copies, live IDE versions,
weights, benchmark evidence and health-log recovery originals remain intact.
This is a dated controller-only observation; recheck all hosts before launch.
Exact targets and recovery sources: [cleanup receipt](local-headroom-cleanup-20260914.json).

Mirror preparation: a versioned installation template adds only the release
worktree pair to the inspected installed script. Exact predecessor/template
hashes and the post-seal verification procedure are recorded in
[mirror cutover](MIRROR_CUTOVER.md). One focused byte-diff/syntax test passes;
the installed script and cron are unchanged, and no mirror completion is claimed.
