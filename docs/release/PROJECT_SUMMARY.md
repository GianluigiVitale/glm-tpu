# Project summary for reviewers

**Problem.** Run the existing GLM-5.2-FP8 model on a fixed eight-host, 32-chip
TPU v4 installation, with inspectable numerical behavior, explicit device
ownership, usable request delivery and reproducible evidence.

**Approach.** Native JAX/Pallas execution partitions work across feature-four
and expert-eight groups. The ordinary greedy path combines grouped routed
experts, the empty-slot correction, resident BF16 non-routed weights, DSA
selection, batched prefill and a packed decode/delivery loop. Requests share
8,192 prompt/output slots and run one at a time. A protected controller binds
the published source, checkpoint, topology and environment before execution.

**Engineering contributions.** This repository implements distributed execution
and checkpoint placement, cache/frontier invariants, grouped expert and
sparse-attention integration, prefill/decode composition, graph and live memory
admission, failure-aware token delivery, authenticated fleet cleanup and
evidence/recovery tooling. The release extracts the retained ordinary numerical
path into `glm_tpu/optimized` and connects it to the private request interface.
The [architecture](ARCHITECTURE.md), [promotion provenance](OPTIMIZED_PROMOTION.md)
and [curation ledger](../curation/README.md) make these boundaries inspectable.

**Reuse and attribution.** GLM architecture, trained weights, tokenizer and chat
template originate with Zhipu AI. Hugging Face Transformers extracts supply
reference material; JAX, Pallas, XLA and libtpu supply compiler/runtime facilities.
This work does not claim a new foundation model, training result, compiler or
independent invention of grouped matrix multiplication or sparse attention.
[Third-party notices](../../THIRD_PARTY_NOTICES.md) retain source and license
scope. Original project code has no blanket open-source license.

**Results.** Candidate `4f551e6b` passed 571 CPU release tests with one skip,
plus source, content and isolated package checks. Trained release validation and
main promotion remain pending in [STATUS](STATUS.md). Historical ordinary
research throughput is preserved in [measurements](../perf/frozen-20260920/MEASUREMENTS.md);
it must not be substituted for validation of the release command.

**Reproducibility.** The [README](../../README.md) leads to inference,
installation and offline CPU inspection. Compact receipts identify source,
graph, output and archive hashes. [Source packaging](SHAREABLE_PACKAGE.md) binds
reviewer files to a commit. Hardware reproduction additionally requires the
private site's retained checkpoint, topology and runtime assets; source-only
reviewers can inspect code and run the portable CPU subset in [TESTING](TESTING.md).

**Limitations.** One greedy request, 8K combined capacity, cold startup on every
invocation, no HTTP service, no concurrent batching or durable KV recovery.
Thinking consumes the output budget and may end before a final answer. Short
token agreement does not establish answer correctness or model-card accuracy.
Legacy sampled/long-context evidence is separate. Research comparisons include
failed prose/format checks and capped answers; speculative decoding was rejected.
Review was performed by the assistant in this chat, not an independent reviewer.
