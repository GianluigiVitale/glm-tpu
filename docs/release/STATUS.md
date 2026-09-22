# GLM-5.3 release status

Resident ordinary inference was executed at
`5c3c1d6bee18817fdd4d665923747e233e76ee4a`, run
`optimized_request_20260921T233911390408Z`. It produced a correct solo answer,
then reused the same loaded model for 770 independent GSM8K requests.
[Resident result receipt](glm53-resident-results-20260922.json).

## Single-chat measurement

| Measurement | Slowest-host result |
|---|---:|
| Decode | 13.570447 tokens/s |
| Prompt prefill | 85 tokens / 0.965029 s |
| Decode time | 239 timed tokens / 17.611801 s |
| Total output including thinking and prefill-produced first token | 240 tokens |
| Cold verification/loading/compilation | 1,219.894875 s |
| Worker including initialization, cold startup, warmup and answer | 1,249.783933 s |
| Peak HBM per chip | 28,228,733,440 bytes |

Decode includes fleet votes and local token writes. It excludes prefill, startup,
queue overhead and final text decoding. Worker wall excludes controller staging
and SSH. Compilation is included in cold time; overlapping intervals must not
be added. The completed final answer was 70,000, checked against reference and
arithmetic. All eight hosts agreed on tokens and graphs, and memory checks
passed. Authenticated workers remained live with libtpu after answering.

## Partial GSM8K evaluation

The owner stopped at the ordered prefix of test rows 0–769 from
`openai/gsm8k`, configuration `main`, revision
`740312add88f781978c0658806c59bc2815b9866`. All 770 were freshly executed in
independent conversations; prior demonstration answers were not reused.

| Outcome | Count |
|---|---:|
| Correct completed final answers | 740 |
| Incorrect scored final answers with normal EOS | 27 |
| Context exhausted, counted incorrect | 3 |
| Total processed | 770 |
| Unrun after owner cancellation | 549 |
| Accuracy on the processed subset | 96.103896% |

Prompts were original questions plus a boxed-answer format instruction, without
brevity instructions or examples. References were separate from model inputs.
Decoding was greedy, thinking enabled/max, with every remaining slot after full
tokenization in a 32,768-slot cache. The four-hour operational deadline applied
per question. No answer-driven retries, resampling or hidden output caps.

Scoring considered only the final channel after `</think>` at normal EOS. It
used the last numeric boxed answer, then a `####` numeric marker, then the last
number in the final channel, comparing normalized Decimal values exactly.
There were 763 boxed extractions and four last-number fallbacks; all four
fallbacks were incorrect. Three unfinished context-exhausted outputs were never
scored from numbers in their reasoning. Scoring verifies final numeric results,
not the validity of every intermediate step.

The run emitted 298,578 tokens including thinking. Of these, 297,808 were timed
decode tokens; summed slowest-host decode time was 22,067.408499 s, giving
13.495377 tokens/s. Prefill totaled 60,764 input tokens in 756.988795 s.
These sums exclude queue/SSH collection overhead and the prior cold startup.
Peak HBM stayed at 28,228,733,440 bytes/chip. All-host token/graph identities and
memory evidence were checked per result. The remaining queue was withdrawn;
the scorer/notifier exited, while the model and its workload leases were retained.

This is **not a full-test-set GSM8K result**. Public benchmark familiarity,
ordered-prefix selection and stopping after observed progress limit comparison
with independently chosen complete evaluations. No full-32K-input quality claim.

## Concurrent behavior and release verification

The separate four-chat run at `c2f60efe` completed four correct normal-EOS answers,
with 4.89–5.12 tokens/s per active chat and 9.269431 aggregate tokens/s.
Its 317-token sequential prefill took 3.884926 s; cold startup took 1,178.091731 s;
peak HBM was 28,789,189,632 bytes/chip. That invocation verified eight-host cleanup.
[Original receipt](glm53-four-answers-20260921.json) preserves all boundaries.
Its concise-explanation suffix differs from the partial benchmark prompt.

The preceding main passed 629 CPU tests with one skip. The resident change passed
65 affected tests; unchanged numerical code reuses those and actual TPU evidence.
Final affected/source/content/package/archive checks and assistant self-review
are bound to the final commit in
`/home/gianl/glm-run/glm53_resident_release_20260922/publication/promotion.json`.
Final main, private release/tag and backup are established by that receipt and
its release publication receipt, not by this status text alone.

Source: 141 verified shards / 755,632,050,320 bytes; owner pack: 32 files /
786,181,673,984 bytes. [Checkpoint bindings](../../configs/glm53-site.json).
No new hardware run or model interruption is needed for source promotion.
The resident code is staged in its original run directory; merging Git does
not redeploy it. No HTTP endpoint, dynamic cache capacity, online batch admission
or durable restart recovery. Review is self-review, not independent review.
[History](GLM53_MIGRATION.md) preserves earlier results and failures.
