# Project summary

**Problem.** Make GLM-5.3-FP8 usable on eight hosts and 32 TPU v4 chips, with
correct weight placement, manageable compiler memory, independent conversation
state and reproducible answer/speed evidence.

**Approach.** Native JAX/Pallas execution combines feature-four/expert-eight
sharding, grouped experts, sparse attention and resident BF16 non-routed weights.
A resident ordinary session reuses weights and compiled graphs for sequential
questions. A separate batch mode shares weights across four independent 32K
caches, prefilling prompts sequentially and decoding concurrently.

**Contributions.** Distributed checkpoint packing and verified owner loading;
model/operator integration; cache ownership and request delivery; fresh graph
and memory admission; protected fleet operation; private resident submission and
commit-bound evidence. GLM-5.3 provenance is independently pinned while preserving
the prior GLM-5.2 release. [Architecture and code map](ARCHITECTURE.md).

**Reuse.** Z.AI supplies the trained model, architecture and tokenizer. Hugging
Face reference material and JAX/Pallas/XLA/libtpu retain their attribution. This
project implements inference systems rather than a new foundation model or
compiler. [Third-party notices](../../THIRD_PARTY_NOTICES.md).

**Results.** One chat decoded at 13.57 tokens/s. A resident sequential GSM8K run
scored 740/770 correct (96.1%), including three context-exhausted attempts in the
denominator, and averaged 13.50 decode tokens/s. The owner stopped after the first
770 of 1,319 test rows; this is a partial evaluation. Solo prefill was 0.965 s for
85 input tokens; cold verification/loading/compilation was 1,219.89 s, paid once
per session. Peak observed HBM was 28.23 GB/chip. A separate four-chat test
completed four correct answers at 4.89–5.12 tokens/s per active chat.
[Receipts and timing boundaries](STATUS.md).

**Reproducibility.** The [README](../../README.md) gives a recommended command and
offline CPU path. Model, template, checkpoint, code and result hashes bind each
execution. The [archive](SHAREABLE_PACKAGE.md) contains source/docs from one
commit, with an incremental Git recovery bundle and verified regional backup.
Hardware reproduction needs the retained site's external assets and environment.

**Limitations.** The ordered, owner-stopped benchmark prefix is not the full
GSM8K test score or proof of uncontaminated held-out performance. Numeric scoring
checks final answers, not every reasoning step. Short prompts do not establish
full-32K-input quality. Resident mode is sequential and uses a private file inbox;
the [local chat UI](../UI.md) and its [local API](../API.md) offer saved histories,
streamed answers and tool calling over that one sequential queue, without parallel
decoding, online batch admission or durable KV recovery. Model processes remain loaded
until explicit stop or failure. Review is assistant self-review, not independent
review. GLM-5.2 history and licenses remain preserved.
