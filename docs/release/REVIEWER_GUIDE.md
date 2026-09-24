# Reviewer guide

1. Read the [summary](PROJECT_SUMMARY.md) and [README](../../README.md).
2. Follow [architecture](ARCHITECTURE.md) into the ordinary runtime, batched decode,
   cache ownership and independent stopping in `glm_tpu/optimized/`.
3. Inspect the [resident receipt](glm53-resident-results-20260922.json) and
   [scoring/timing definitions](STATUS.md). The 740/770 result is a partial,
   owner-stopped GSM8K evaluation. The separate [four-chat receipt](glm53-four-answers-20260921.json)
   establishes concurrency. All-host token agreement is separate from correctness.
4. Follow [checkpoint verification](CHECKPOINTS.md), then source-bound launch,
   memory admission, resident submission and cleanup in `scripts/release/`.
5. Run the offline subset in [TESTING](TESTING.md). It needs no model weights or
   cloud access. Full-history tests have separate requirements.
6. Read [migration history](GLM53_MIGRATION.md) and the curation ledger (archived at tag `archive/research-20260922`: `docs/curation/README.md`)
   for failures, superseded work and exact recovery.

The ordered prefix and public benchmark familiarity limit accuracy comparisons.
Short prompts and allocated 32K caches do not establish full-32K-input quality.
Resident mode serves sequential questions through a private inbox; four-chat
batching is separate. Raw prompts, outputs and references stay outside the archive.

GLM-5.2, research branches, originals and DB616–621 remain preserved. Trained
weights, model architecture and compiler/runtime are reused; implementation and
systems integration are the contribution. [Attribution](../../THIRD_PARTY_NOTICES.md).
Review is assistant self-review, not an independent assessment.
