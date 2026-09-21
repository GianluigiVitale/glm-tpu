# Release status

**This branch is migrating to GLM-5.3.** GLM-5.2 is frozen at private tag
`glm-5.2`; its weight payloads were retired at the owner's request. Read
[migration status](GLM53_MIGRATION.md). All measurements below describe GLM-5.2;
GLM-5.3 has not yet passed inference validation.

The owner approved the four-conversation implementation for private main with
its measured answer limitations disclosed. The release supports a fixed group
of up to **four concurrent conversations**, with **32,768 combined input/history/
reasoning/output slots per conversation**, or a sequential queue of ten.

## Four-conversation result

Executable `a252eb01dc8e9fcd45311182ba042caa4addc38c` ran four short GSM8K
questions concurrently with one shared model. Fresh graph/memory admission,
shared decode rounds, all-rank token/graph agreement and all-eight zero exits
and authenticated cleanup passed.

**Three answers completed correctly; one reached its1,024-token output limit
without a final answer.** This is a bounded functional check, not a dataset
accuracy estimate. The allocated32K caches do not establish full32K input quality.
[Receipt](four-conversations-20260921.json) · [Operation and history](CONCURRENT.md).

| Slowest-host measurement | Result |
|---|---:|
| Per-active-chat decode | 4.91–5.19tokens/s |
| Aggregate over the mixed-length batch | 7.984149tokens/s |
| Sequential prefill,317total input tokens | 3.878273s |
| Cold load/compile | 1094.574786s |
| Maximum observed HBM per chip | 28,789,189,632bytes |

The first eight-chat attempt required34.09GiB per chip; its memory repair
required31.17GiB against30.75GiB available. Both failed before answers, with
all-eight cleanup. A3.05GiB temporary cache copy remained. The public interface
therefore caps concurrent submissions at four. [Failure receipts](concurrent-failure-20260921.json).

## Single-request release

The ordinary8K engine at `9469cd733df7fabe7f6f421ab0fad60801cf3138` completed
GSM8K test row0 correctly:18, normal EOS after265tokens. Decode14.547542tokens/s;
prefill0.937823s for92tokens; cold load/compile1102.028924s; maximum observed HBM
28,228,733,440bytes/chip. All-rank agreement, graph/memory and eight-host cleanup
passed. [Answer receipt](single-answer-20260920.json).

The preceding8K integration at4f551e6b matched29/29reference-prefix tokens.
Token agreement is separate from answer correctness.
[Integration receipt](ordinary-integration-20260920.json).

## Checks and private promotion

The [original release checks](ordinary-release-checks-20260920.json), including
571tests with one skip and affected extension checks, are reused for unchanged
code. The batching implementation's three affected numerical CPU checks passed,
including exact token agreement, inactive state and cross-lane isolation. Their
CPU interpreter and floating-point limits are detailed in [CONCURRENT](CONCURRENT.md).
Test sets overlap and must not be added into one total.

Final host admission, CLI, request/controller, content, curation, source and
isolated package checks, self-review and the exact final commit/archive identities
are recorded outside Git in:
`/home/gianl/glm-run/four_conversation_release_20260921/promotion.json`.
That receipt must show a successful private-main push and generation-bound
US-CENTRAL2 backup/readback; an absent receipt does not establish promotion.
[Reviewer package](SHAREABLE_PACKAGE.md).

No numerical graph code changed after the four-chat hardware test; final source
changes restrict public admission to four and update CLI metadata. No new TPU
run was performed for promotion. Review is assistant self-review, not independent.

## Scope and history

This is retained-site inference on eight hosts/32TPUv4chips. It requires private
checkpoint/topology assets and the pinned environment. Each invocation incurs
cold startup; there is no HTTP server, online batch admission or durable KV
recovery. Reasoning can exhaust the output limit. General answer accuracy and
full32K input correctness remain unknown. Larger sequential/legacy sampled
profiles have separate [inference limits](INFERENCE.md).

[Research history](../perf/README.md), [ordinary integration history](../perf/ordinary-release-20260920.md)
and [curation](../curation/README.md) preserve successes, failures, superseded work,
research branches, originals and DB616–621. Attribution and licenses are retained.
Older detailed status is exactly recoverable with
`git show f249469a5ff53c14366ba0a9e14b29144d34e636:docs/release/STATUS.md`.
