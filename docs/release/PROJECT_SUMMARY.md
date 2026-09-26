# Project summary

**Problem.** Make GLM-5.3-FP8 usable on eight hosts and 32 TPU v4 chips, with
correct weight placement, manageable compiler memory, independent conversation
state and reproducible answer and speed evidence.

**Approach.** Native JAX and Pallas execution combines an expert-8 x feature-4
mesh, grouped routed experts, DSA sparse attention and resident BF16 non-routed
weights. A resident session reuses the weights and the compiled programs for
sequential questions. A separate batch mode shares the weights across four
independent 32K caches, prefilling the prompts one after another and decoding
them together.

**Contributions.** Distributed checkpoint packing and verified owner loading;
model and operator integration; cache ownership and request delivery; fresh graph
and memory admission; protected fleet operation; private resident submission and
commit-bound evidence; a graph-equivalence harness that proves restructuring
commits leave the device programs unchanged. GLM-5.3 provenance is pinned
independently while the prior GLM-5.2 release stays preserved.
[Architecture and code map](ARCHITECTURE.md).

**Reuse.** Z.AI supplies the trained model, the architecture and the tokenizer.
Hugging Face reference material and JAX, Pallas, XLA and libtpu keep their
attribution. This project implements the inference system, not a new foundation
model or compiler. [Notices](../../THIRD_PARTY_NOTICES.md).

**Results.** One chat decoded at 13.57 tokens/s. A resident sequential GSM8K run
scored 740/770 correct (96.1%), counting three context-exhausted attempts in the
denominator, and averaged 13.50 decode tokens/s. The owner stopped after the first
770 of 1,319 test rows; this is a partial evaluation. Solo prefill was 0.965 s for
85 input tokens; cold verification, loading and compilation took 1,219.89 s, paid
once per session. Peak observed HBM was 28.23 GB per chip. A separate four-chat
test completed four correct answers at 4.89–5.12 tokens/s per active chat.
[Receipts and timing boundaries](STATUS.md).

**Reproducibility.** The [README](../../README.md) gives the recommended command
and the offline CPU checks. Model, template, checkpoint, code and result hashes
bind each execution. A reviewer package is a `git archive` of one commit
([reviewer guide](REVIEWER_GUIDE.md#source-package)). Hardware reproduction needs
the site's external assets and environment.

**Limitations.** The ordered, owner-stopped benchmark prefix is not the full
GSM8K test score or proof of uncontaminated held-out performance. Numeric scoring
checks final answers, not every reasoning step. Short prompts do not establish
full-32K-input quality. Resident mode is sequential and uses a private file inbox;
the [local chat UI](../UI.md) and its [local API](../API.md) offer saved histories,
streamed answers and tool calling over that one sequential queue, without parallel
decoding, online batch admission or durable KV recovery. Model processes stay
loaded until an explicit stop or a failure. Review is assistant self-review, not
independent review. The GLM-5.2 history and its licenses remain preserved.

**License.** The project's own work is under the [Apache License 2.0](../../LICENSE);
third-party material keeps its own license.
