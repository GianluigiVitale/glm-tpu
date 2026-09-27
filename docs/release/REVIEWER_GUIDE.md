# Reviewer guide

1. Read the [summary](PROJECT_SUMMARY.md) and the [README](../../README.md).
2. Follow the [architecture](ARCHITECTURE.md) into the request path: the
   controller (`glm_tpu/executor/`), the worker (`glm_tpu/worker/`), the model
   runner and its admission (`glm_tpu/runner/`), the engine and the per-request
   loop (`glm_tpu/engine/`), then the model, layers and kernels
   (`glm_tpu/models/glm_moe_dsa/`, `glm_tpu/layers/`, `glm_tpu/kernels/`). Cache
   ownership and independent stopping are in `glm_tpu/layers/attention/kv_cache.py`
   and `glm_tpu/engine/request_session.py`.
3. Inspect the [resident receipt](glm53-resident-results-20260922.json) and the
   [scoring and timing definitions](STATUS.md). The 740/770 result is a partial,
   owner-stopped GSM8K evaluation. The separate
   [four-chat receipt](glm53-four-answers-20260921.json) establishes concurrency.
   All-host token agreement is separate from correctness.
4. Follow [checkpoint verification](CHECKPOINTS.md), then the source-bound launch,
   memory admission, resident submission and cleanup in [OPERATIONS](OPERATIONS.md).
5. Run the offline subset of [TESTING](TESTING.md); it needs no model weights or
   cloud access. The equivalence harness ([README](../../tools/equivalence/README.md))
   shows how restructuring commits were proved to leave the device programs
   unchanged. The tables and checkers that proved the restructuring commits'
   moves, renames, splits and text edits are in the history, last at commit
   `a6f1b0e0` (`git show a6f1b0e0:tools/migration/move_map.toml`).
6. Read the [migration history](GLM53_MIGRATION.md) for failures, superseded work
   and recovery. The research history and the curation ledger are at the tag
   `archive/research-20260922`
   (`git show archive/research-20260922:docs/curation/README.md`).

The ordered prefix and public benchmark familiarity limit accuracy comparisons.
Short prompts and allocated 32K caches do not establish full-32K-input quality.
Resident mode serves sequential questions through a private inbox; four-chat
batching is separate. Raw prompts, outputs and references stay outside the
repository.

The GLM-5.2 release, the research branches and the original evidence remain
preserved. The trained weights, the model architecture and the compiler and
runtime are reused; the implementation and systems integration are the
contribution. [Notices](../../THIRD_PARTY_NOTICES.md). Review is assistant
self-review, not an independent assessment.

## Source package

A reviewer package is the tracked tree of one commit, without Git internals,
weights, credentials, private requests, raw answers, token logs or databases
(none of which are tracked). From a clean checkout, into a private directory
outside the checkout:

```bash
commit=$(git rev-parse HEAD)
git archive --format=tar.gz --prefix="glm-tpu-${commit}/" --output="/path/outside/git/glm-tpu-${commit}.tar.gz" "$commit"
sha256sum "/path/outside/git/glm-tpu-${commit}.tar.gz"
```

Inspect the member list before sharing it, and record the commit and the archive's
SHA-256 with it. The package supports the offline CPU checks; it is a source
package, not a model distribution or a deployable image. The tests that read Git
history (`git show`) need a clone, and the hardware steps need the site's assets.
