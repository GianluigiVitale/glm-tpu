# Ordinary release integration: results and superseded attempts

This record preserves the engineering decisions behind the ordinary release.
It is history, not instructions to resume workloads. The owner ended optimization
research and ultimately reduced answer acceptance to one correctly completed
GSM8K example. The implementation, source archive, honest measurements and private
main requirements were retained.

## What shipped and what was reused

The ordinary engine was extracted from trained research pin
`5ff7b01e7a6520c652e7ce6dcc4a7012363e0ff8`, following the dependency-first
curation precedent at `c7f6d42f` / `493b67de`. It combines D1 grouped routed
experts and the empty-slot fix, D8 BF16 non-routed weights, D10 sparse selection,
D4 packed delivery, and D8/P1/P2 prefill. Model weights, architecture, tokenizer,
Transformers reference material and the JAX/Pallas/XLA runtime are reused, with
their attribution and licenses retained. This is execution/integration engineering,
not new model training or independent invention of those algorithms.

The release adds private request preparation, a protected source-bound launch,
fresh graph/memory admission, token delivery and authenticated cleanup. The
`ask` interface accepts one to ten strings, loads once and generates sequentially
with fresh state per question. The 8K profile has8192 combined slots; the optional
128K profile has166912 slots and an input limit of131072. Queue capacity is not
simultaneous model batching or proof of ten successful answers.

## Preserved attempts

Private originals live in `/home/gianl/glm-run/<run-id>`; raw questions, reasoning
and token streams stay outside Git and the reviewer source archive. These paths
are recovery references, not commands to launch another run.

| Source / run | Outcome and disposition |
|---|---|
| `4f551e6b`; `optimized_request_20260920T170223095076Z` | Completed original8K integration.2034prompt tokens,256output tokens,29/29reference-prefix agreement, all-rank agreement, fresh graphs/memory and eight-host cleanup. Output cap stopped reasoning; no completed answer. Collected once and never duplicated. |
| `bae824a0`; `optimized_request_20260920T173446591702Z` | First queued128K attempt. B128 prefill compilation requested31.45GiB against30.75GiB, exceeding HBM by720.25MiB. No answer; authenticated eight-host cleanup. |
| `b3ced443`; `optimized_request_20260920T175700282937Z` | Refused the busy staging lock before SSH, source staging or model execution. Empty run root preserved; not a model failure or measured attempt. |
| `4acb873a`; `optimized_request_20260920T180158114672Z` | Corrected-cache profile compiled/admitted all six graphs. First difficult question exhausted32768output tokens without final answer. Second interrupted during reasoning; eight not run. Deliberately stopped to correct the avoidable default cutoff; eight-host cleanup verified. |
| `9469cd73`; `optimized_request_20260920T193412487216Z` | Same ten selected questions with available output capacity. First produced over39000tokens without a completed answer at the recorded status check. Owner explicitly cancelled the difficult queue and requested one easy question. Zero completed request reports; eight worker exit codes255 after authenticated stop, idle cleanup and lease release verified. |
| `9469cd73`; `optimized_request_20260920T205942191301Z` | One cached GSM8K test example on the8K path. Correct18, EOS265tokens; all eight workers exit0 and authenticated idle. This satisfies the revised owner criterion. [Receipt](../release/single-answer-20260920.json). |

The difficult examples were five GPQA-Diamond and five AIME2026 cases selected
once with seed14666840778903218970. One GPQA input had131072tokens using irrelevant
filler. It was intended as a capacity check, not long-document reasoning quality;
it did not complete. No resampling replaced an incorrect result. The switch to
GSM8K was an explicit change of the owner's acceptance scope and is reported as
such, not a successful ten-question benchmark.

## Necessary fixes and checks

The long-cache OOM was fixed by reusing the legacy path's exclusive state
donation: prefill donates argument2 and decode argument1. This avoids a redundant
cache copy without changing numerical programs. Twenty affected prefill/runtime
checks and two decoder checks passed, including CPU32 comparisons.

The source mirror's staging locks now wait once while both workload locks remain
nonblocking refusals. Thirteen affected checks passed. This avoids treating a
temporary source-backup lock as a failed model invocation; no retry controller
was added.

The avoidable explicit32768-token cutoff was removed from the default128K
question path. After tokenization it allocates the remaining combined capacity,
capped at the retained163840maximum; a full131072-token input leaves35840output
slots. Explicit caps and the8K default2048 remain unchanged. Thirty affected
CPU checks passed, and a prepare-only tokenizer check proved all ten prompt arrays
unchanged. This did not solve the difficult question's prolonged reasoning.

The original integration passed571tests with one skip plus source/content/package
checks. The queue extension passed114affected checks with one optional skip;
40portable checks passed from an extracted archive without Git. These sets overlap.
No fresh research campaign or new engine was started, and passing checks were
reused when code was unchanged. See the [check index](../release/ordinary-release-checks-20260920.json).

## Final evidence and limitations

The completed GSM8K example is test row0 at dataset revision
`740312add88f781978c0658806c59bc2815b9866`, selected before output from the existing
cache. Gold stayed separate from the prompt. The final numeric answer18 matches
the reference and the arithmetic `(16 - 3 - 4) * 2`. It ended at EOS; token-prefix
agreement was not used as a substitute for this answer check.

Slowest-host measurements:1102.028924seconds cold loading/compilation,
0.937823seconds prefill for92tokens, and18.147396seconds for264timed decode tokens
(14.547542tokens/s). The output contained265tokens including reasoning.
Maximum observed HBM was28228733440bytes per chip across32owners. Startup,
prefill and decode are distinct; no network latency or persistent-service claim.

Reference evidence remains the earlier ordinary integration:29/29DB610prefix
agreement,142.038211prompt tokens/s at2034tokens,14.354231decode tokens/s and
1095.333484seconds cold startup. These historical timing values are not a second
headline for the final release. The unchanged ordinary numerical path and its
CPU checks support reuse; neither run proves broad task accuracy.

The larger profile compiled, but no full128K-input answer was completed. Difficult
questions can consume their whole budget without useful final output. There is
no HTTP service, simultaneous batching, durable KV recovery or qualified
speculative decoding. Assistant self-review is not independent review.

## Recovery and preservation

No Git history was rewritten. Research refs, the pre-cleanup preservation ref,
originals and DB616–621 remain intact; frozen `MODEL_SOURCE=edecdd94` is unchanged.
The [curation ledger](../curation/README.md) retains exact recovery for removed
files. Earlier failed and rejected kernel/MTP/global-max/fused-EP work remains in
the [frozen research index](README.md) and its preserved branches.

The previously verified full-history base is
`gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/publication/frozen-history.bundle`,
generation1789920747317967, SHA256
`8a9a0a11a0ccb32ddc2539c6296006f234d20351ba9b8d3e1d63a7bb9740758d`.
The final source archive and incremental history are separate, generation-bound
objects in the same US-CENTRAL2 bucket. The private final publication receipt
named in [STATUS](../release/STATUS.md) records exact commit, hashes and readbacks.
No weights, raw answer logs, private prompts or databases enter the source archive.
