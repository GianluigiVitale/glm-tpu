# Project summary

**Problem.** Run GLM-5.3-FP8 on eight hosts and32TPUv4chips with correct checkpoint
placement, independent concurrent conversation state and inspectable measurements.

**Approach.** Native JAX/Pallas execution uses feature-four/expert-eight sharding,
grouped experts, sparse attention, resident BF16 non-routed weights, sequential
prompt prefill and batched decoding. Four conversations share one loaded model
while retaining separate32K caches, output streams and stopping decisions.

**Contributions.** Distributed checkpoint packing and verified owner loading;
model/operator and compiler integration; cache ownership and request delivery;
fresh graph/memory admission; protected fleet operation and reproducible evidence.
The GLM-5.3 migration reuses the working engine while independently pinning new
weights, configuration, template and license. It removes the old default output
caps from this path. [Architecture](ARCHITECTURE.md) maps these contributions to code.

**Reuse.** Zhipu AI supplies the trained model, architecture and tokenizer.
Hugging Face reference material and JAX/Pallas/XLA/libtpu retain their attribution.
This work claims neither a new foundation model nor a new compiler.
[Licenses and attribution](../../THIRD_PARTY_NOTICES.md).

**Results.** Four fixed GSM8K examples completed correctly at normal EOS:
18,3,70000,540. Active-chat decode4.89–5.12tokens/s; aggregate9.27tokens/s;
prefill317tokens in3.885s; cold verification/loading/compilation1178.09s;
batch after startup/warmup66.30s. PeakHBM28.79GB/chip. All-host tokens/graphs,
fresh memory admission and eight-host cleanup passed.
[Receipt and timing boundaries](glm53-four-answers-20260921.json).

**Reproducibility.** The [README](../../README.md) gives one recommended command
and offline CPU review path. Source, model, template, checkpoint and result hashes
bind the execution. The [reviewer archive](SHAREABLE_PACKAGE.md) binds the final
source/docs to a commit. Hardware reproduction requires the retained site's
external weights, topology and environment; the wheel alone is not a deployment.

**Limitations.** Four familiar short questions are a functional check, not broad
accuracy; the model recognized one example. Allocated32K caches do not establish
full32K-input quality. Generation is greedy, prompts prefill sequentially, and
cold startup occurs on every invocation. No persistent HTTP service, online
admission or durable KV recovery. Long-context and sampled history is separate.
Review is assistant self-review, not independent review. GLM-5.2 is preserved
at tagglm-5.2, with retired weight payloads and separate historical measurements.
