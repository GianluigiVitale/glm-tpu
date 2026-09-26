# Four concurrent GLM-5.3 conversations

One shared model, four independent 32,768-slot caches, separate output streams
and separate EOS decisions. Prefill runs one conversation after another; each
decode step advances the whole active batch. This serves a fixed submitted
group: there is no online admission and no HTTP server for this mode.
[Command](OPTIMIZED_INFERENCE.md#four-conversations-at-once) ·
[Measured result](STATUS.md) · [Design](ARCHITECTURE.md#resident-and-concurrent-modes).

All four fixed GSM8K examples completed correctly at normal EOS: 18, 3, 70000
and 540. Active-chat decode was 4.89–5.12 tokens/s; the aggregate was
9.27 tokens/s for the mixed-length batch. The full remaining output budgets were
available, with thinking on at maximum effort. The
[receipt](glm53-four-answers-20260921.json) records the tokens, the timing
boundaries, the fresh graph and memory checks, the host agreement and the
cleanup. Short prompts validate the allocated capacity and the function, not
full-32K-input quality or broad accuracy.

The CPU tests of independent caches and stopping, and the actually differing EOS
lengths of 112, 57, 298 and 100 tokens, support independent state. Token
agreement across hosts is checked separately from the answers' arithmetic. No
eight-conversation capacity is claimed: the historical GLM-5.2 eight-chat
attempts exceeded the available memory. Their failure receipt and the earlier
GLM-5.2 four-chat results (three correct, one capped) are preserved at the tag
`archive/research-20260922`
(`git show archive/research-20260922:docs/release/concurrent-failure-20260921.json`,
`git show archive/research-20260922:docs/release/four-conversations-20260921.json`)
and the GLM-5.2 page at `git show glm-5.2:docs/release/CONCURRENT.md`.
