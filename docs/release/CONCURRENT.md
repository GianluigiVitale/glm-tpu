# Four concurrent GLM-5.3 conversations

One shared model, four independent32768-slot caches, separate output streams and
EOS decisions. Prefill is sequential; decode steps execute the active batch.
This supports a fixed submitted group, without online admission or a HTTP server.
[Recommended command](OPTIMIZED_INFERENCE.md) · [Measured result](STATUS.md).

All four fixed GSM8K examples completed correctly at normal EOS:18,3,70000,540.
Active-chat decode4.89–5.12tokens/s; aggregate9.27tokens/s for the mixed-length
batch. Full remaining output budgets were available, with thinking enabled/max.
The [receipt](glm53-four-answers-20260921.json) records tokens, timing boundaries,
fresh graph/memory checks, host agreement and cleanup. Short prompts validate
allocated capacity and functionality, not full32K input quality or broad accuracy.

Independent cache/stopping CPU tests and actual differing EOS lengths112,57,298,
100tokens support independent state. Token agreement is checked separately from
answer arithmetic. No claim of eight-conversation capacity is made: historical
GLM-5.2 eight-chat attempts exceeded available memory and remain in the
[failure receipt](concurrent-failure-20260921.json). Earlier four-chat5.2results
were3correct/1capped; preserved at tagglm-5.2 and in
[four-conversations-20260921.json](four-conversations-20260921.json).
Full previous explanation is recoverable with
`git show glm-5.2:docs/release/CONCURRENT.md`.
