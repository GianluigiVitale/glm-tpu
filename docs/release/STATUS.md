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

## Merge checklist

- [ ] Supported code/import/asset closure; safe obsolete-file pruning.
- [x] Fresh pinned dependency installation, CPU imports and focused host tests.
- [ ] Real user inference entry point and protected deployment validation.
- [ ] Main deployment independent of frozen research checkout.
- [ ] Checkpoint preparation/loading and recovery instructions validated.
- [ ] Automated CPU and release checks pass in release worktree.
- [ ] Third-party notices/licensing and secret/private-data audit complete.
- [ ] Native terminal evidence collected; score scope/limitations documented.
- [ ] Request/resume evidence and operational limits documented.
- [ ] Final adversarial self-review resolves material findings.
- [ ] Release pushed, eligible main merge and regional mirror verified.

Do not merge merely because the README is polished or a partial score looks
promising. Do not repeat DB616–620 for this checklist.

Initial findings and reproducible audit commands: [inventory audit](INVENTORY.md).

Packaging progress: the private alpha wheel installs and its console works
outside the checkout. A fresh isolated environment installed all63 observed
dependencies; dependency checks and actual CPU imports passed, then78focused
tests passed with1optional tokenizer skip. Temporary setup was removed; active
environment unchanged. Real main deployment remains pending. The latest offline
release suite passes291tests/1skip, including both original benchmark transport
suites and new user controller/transport cases. See [installation scope](INSTALLATION.md).

Deployment preparation: explicit reviewed main/release branch selection, remote
origin/pin checks and attach branch identity now replace the research-only branch
binding in this worktree. Canonical site/path admission remains; no release launch
has occurred. Three stale root research instruction files removed with preserved
Git originals. See [operations](OPERATIONS.md) and [preservation](INVENTORY.md).

Provenance progress: all three Transformers reference extracts match installed
5.12.0 files; the matching Apache license and third-party notices are preserved
and included in the wheel. Its isolated install/console checks still pass. Model
snapshot notice is now pinned and included: four configuration/template files
match revision f33c6dc501ee5a2c7e35155653b1b1abbc320951 and its MIT license;
the earlier saved README is preserved unchanged. Legacy vLLM patch provenance
and broader privacy review remain open. This is not a blanket project license
or public-distribution clearance.

User inference integration: separate bounded prompt format, local pinned-tokenizer
preparation CLI, and single-request worker executor reuse the actual native host
runtime without benchmark validation/scoring. CPU fake-math coverage is not TPU
admission. The default-off user worker now wires the original loader and executor,
with private namespace/source/tokenizer admission and existing cold-write caps.
Protected user controller and separate transport are implemented with CPU
orchestration/byte-roundtrip tests, not actual deployment evidence. User semantic
replay/sealing and real execution admission remain open; see [inference scope](INFERENCE.md).

Checkpoint audit: all141 canonical weight objects match original sealed GCS
generations/sizes/CRCs;96overlay files have expected sizes. Direct canonical-to-RAM
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
