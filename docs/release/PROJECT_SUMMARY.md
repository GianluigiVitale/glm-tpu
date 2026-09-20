# Project summary for reviewers

**Problem.** Run the existing GLM-5.2-FP8 model on a fixed eight-host, 32-chip
TPU v4 installation, with inspectable numerical behavior, explicit device
ownership, usable request delivery and reproducible evidence.

**Approach.** Native JAX/Pallas execution partitions work across feature-four
and expert-eight groups. The ordinary greedy path combines grouped routed
experts, the empty-slot correction, resident BF16 non-routed weights, DSA
selection, batched prefill and a packed decode/delivery loop. Up to ten questions
queue behind one loaded model, with fresh state per question. The larger profile
accepts 131,072 input tokens and 166,912 combined prompt/output slots; the earlier
profile has 8,192 combined slots. A protected controller binds
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

**Results.** The [ordinary8K release](single-answer-20260920.json) completed one
GSM8K test example correctly: expected18, returned18, normal EOS after265tokens.
It delivered **14.55 decode tokens/s**, prefilled92tokens in0.94s
(**98.10 prompt tokens/s**), and required **1,102.03s cold load/compile**.
All eight hosts agreed and cleaned up; fresh graphs/memory passed.
The earlier integration matched29reference tokens. The final release reuses
571passing CPU checks with one skip and the affected extension checks listed
in [STATUS](STATUS.md); these are not an independent quality assessment.

**Reproducibility.** The [README](../../README.md) leads to inference,
installation and offline CPU inspection. Compact receipts identify source,
graph, output and archive hashes. [Source packaging](SHAREABLE_PACKAGE.md) binds
reviewer files to a commit. Hardware reproduction additionally requires the
private site's retained checkpoint, topology and runtime assets; source-only
reviewers can inspect code and run the portable CPU subset in [TESTING](TESTING.md).

**Limitations.** One greedy request generates at a time; queued questions share
startup but wait for earlier answers. Cold startup on every invocation, no HTTP
service, no simultaneous batching or durable KV recovery.
Thinking consumes the output budget and may end before a final answer. Short
token agreement does not establish answer correctness or model-card accuracy.
Legacy sampled/long-context evidence is separate. Research comparisons include
failed prose/format checks and capped answers; speculative decoding was rejected.
Review was performed by the assistant in this chat, not an independent reviewer.
