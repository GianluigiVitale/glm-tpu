# Concurrent conversations — four run; one answer unfinished

The owner subsequently requested a four-conversation test at the same32K total
slots per conversation. At `a252eb01`, **four conversations decoded concurrently
and passed TPU graph/memory admission**. All eight hosts agreed on output tokens
and graph identities, exited0 and passed authenticated idle cleanup.
Shared token-round records verify concurrent advancement and independent stopping.

Three GSM8K examples ended at EOS with correct final answers:18,3and540.
The third submitted question reached its1024output-token limit while still
reasoning, without a final answer. Its reference answer is70000; mentioning it
among alternative calculations does not count as a correct completed response.
Thus this is a successful hardware concurrency test, **not four completed correct
answers**. No automatic retry or main promotion followed.
[Four-conversation receipt](four-conversations-20260921.json).

| Scope, slowest-host measurement | Result |
|---|---|
| Cold load/compile | 1094.574786 s |
| Sequential prefill,317 total prompt tokens | 3.878273 s |
| Per-chat decode while active | 4.91–5.19 tokens/s |
| Aggregate decode over the entire mixed-length batch | 7.984149 tokens/s |
| Batch decode wall time, including the truncated answer | 208.287699 s |
| Maximum observed HBM per chip | 28,789,189,632 bytes |

This proves allocation and execution with four32K caches and short prompts,
not answer quality on full32K inputs. Finished lanes stay allocated until the
group ends; the long unfinished answer dominates aggregate batch time.

## Preserved eight-conversation failures

The owner-requested repair at `18dda892` failed the same compile gate:
**31.17GiB required versus30.75GiB available per chip**,437.87MiB over.
Program memory fell from6.31GiB to3.40GiB and temporary fragmentation from
2.87GiB to297.53MiB, but the3.05GiB whole-cache copy remains. This was a memory
reduction, not successful copy elimination. All eight workers exited1 and
authenticated cleanup passed. No answers, serving speed or startup-to-ready
measurement resulted. The later explicitly requested four-chat test is separate;
the eight-chat workload was not retried or promoted.

The revised decoder masks
finished conversations at each layer's cache update, replacing the final
whole-cache selection that retained the pre-decode bank. Small position and
selection arrays are still masked at the final boundary. The unmasked single
decoder path retains its behavior. CPU correctness and fresh TPU compilation,
live-memory admission and actual answers are required before calling this fixed.
The three affected CPU checks passed; that did not establish hardware fit.

The eight-conversation,32K-per-conversation test failed before generating any
answers. At executable `211743748388e58c68ed956885920d73f0f1eb8d`, TPU compilation
of `batch_decode` reported **34.09GiB required versus30.75GiB available per chip**,
an excess of3.34GiB. All eight workers exited with code1 and authenticated idle
cleanup passed on all eight hosts. No retry was launched and this candidate was
not promoted to main. [Failure receipt](concurrent-failure-20260921.json).

The compiler reported a3.05GiB temporary copy of the KV cache and6.31GiB of
program memory, including2.87GiB fragmentation. Cache allocation alone had passed;
that did not establish decoder fit. These allocations explain this implementation's
failed gate, not a proof that every possible batching implementation must fail.
Neither two- nor three-conversation operation nor a smaller context was tested.

This addition accepts one fixed group of one to eight conversations, with one
shared copy of the model weights. Each conversation has its own cache, position,
token stream and EOS/output limit. After prompt preparation, one compiled
decoder call advances every active conversation in the group by one token.

```bash
python -m glm_tpu ask --questions /private/questions.json \
  --context 32k --concurrent --max-new-tokens 1024 --wall-seconds 3600
```

`questions.json` is a private JSON array of one to eight question strings. The
32,768-slot budget applies **per conversation** and includes the input, template,
reasoning and final output. Preparation rejects any input plus requested output
budget that exceeds it. `--prepare-only` tokenizes without starting a workload.
Use the installation and retained-site requirements in the main README.

The existing protected controller still authenticates source/checkpoint identity,
holds both workload leases and sync locks, checks all eight hosts, and cleans up
its owned processes. There is no automatic retry. The original sequential queue
remains available when `--concurrent` is absent.

## Scheduling and memory

Prompt prefill is sequential within the group. Decode is batched, with weights
and the rotary table shared across conversations. Finished lanes retain their
state and emit no further tokens. Some expert kernels use internal loops under
JAX's batching transformation; batching does not promise an eightfold speedup.

A single donated batch cache bank plus one temporary prefill cache avoids
keeping eight separate prefill allocations and another stacked copy. Eight
32K KV/index caches require 3,447,717,888 bytes per chip, excluding weights,
metadata and working memory. Actual compiler allocations and live HBM must pass
admission with the entire group resident before execution.

This interface processes the submitted group and exits. It does not provide
HTTP transport, add new requests to a running batch, or retain chat caches after
the process exits. Re-submit conversation history within the per-chat budget.

## CPU evidence and failed hardware gate

Host tests cover separate histories, EOS/length stopping, simultaneous round
delivery, ambiguous-delivery refusal, deadline handling, request identity and
the original fleet controls. CPU32 numerical tests compare eight different
prefilled histories against eight independent retained decoder executions and
check that an inactive lane's state is unchanged. Pallas kernels run in their
CPU interpreter. Unsupported CPU mixed BF16 dots use exact operand widening to
FP32 in the isolated test process; this is interpreter evidence, not a TPU graph
or performance result.

Batch and separate execution need not produce bit-identical floating-point
buffers: matrix operation shapes change accumulation order. The fixture requires
exact emitted tokens, unchanged existing history and inactive state, identical
selected-position sets and exact positions/counts. A direct isolation check
changes the eighth conversation's token and requires every byte of the other
seven conversations' computed state and outputs to remain identical.
Scalar-versus-batch cache differences are recorded as diagnostics. Initial
absolute0.0625 and relative0.015625 cache bounds failed (maximum absolute0.15625;
one relative row error0.02778). These bounds were replaced by the direct
perturbation check, rather than increased until passing. Near-zero scores also
made a relative score criterion unsuitable. Diagnostic failures remain preserved
privately; no bit-identical scalar cache or hardware success is claimed.

The final CPU32 check passed:15active decode selections agreed with separate
execution, the inactive lane and pre-existing histories remained unchanged,
and perturbing lane8 left the other seven lanes bit-identical. Maximum observed
scalar/batch cache difference was0.15625 absolute and0.047255 relative L2 for an
affected row. This distinction is a limitation of the scalar numerical comparison,
not a claim of identical internal results.

The original real-weight attempt used a fixed batch of the first eight cached GSM8K test rows,
selected before outputs, with private golds kept out of model inputs and Git.
Checkpoint verification/loading and prefill compilation passed, but batch decode
compilation failed before warmup or question execution. There are zero completed
answers to grade and no batched startup-to-ready, prefill or decode speed result.
The working single-request main release and its evidence remain unchanged.
