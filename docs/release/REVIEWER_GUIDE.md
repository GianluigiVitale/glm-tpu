# Reviewer guide

1. Read the [project summary](PROJECT_SUMMARY.md) and [README](../../README.md).
2. Follow [architecture](ARCHITECTURE.md) into the ordinary runtime, batched decode,
   cache ownership and independent stopping in `glm_tpu/optimized/`.
3. Inspect the [GLM-5.3 receipt](glm53-four-answers-20260921.json): executable/model
   pins, four completed answers, arithmetic, token hashes, timings and limits.
   All-host token agreement and answer correctness are distinct checks.
4. Follow checkpoint placement and [owner verification](CHECKPOINTS.md), then
   source-bound fleet launch, memory admission and cleanup in `scripts/release/`.
5. Run the offline CPU subset in the README. Full-history and external-evidence
   tests have separate requirements in [TESTING](TESTING.md).
6. Read [migration history](GLM53_MIGRATION.md) and the [curation ledger](../curation/README.md)
   for failures, superseded work, file purposes and exact recovery.

Current evidence is four familiar short GSM8K examples, normal EOS and4x32K cache
allocation. It does not establish broad accuracy, full32K-input quality, online
serving or independent review. One example was recognized by the model; these are
not clean held-out accuracy measurements. Raw prompts, outputs and gold references
remain private and separate from the compact reviewer archive.

Historical GLM-5.2 evidence, DB616–621 and sampling/long-context work remain
preserved under tagglm-5.2 and research branches. The model weights, tokenizer and
compiler/runtime are reused; implementation and integration are the project's
contribution. See [attribution](../../THIRD_PARTY_NOTICES.md). Review was assistant
self-review, not an independent assessment. Earlier guide:
`git show glm-5.2:docs/release/REVIEWER_GUIDE.md`.
