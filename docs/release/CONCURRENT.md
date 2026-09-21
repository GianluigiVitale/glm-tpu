# Concurrent conversations — candidate, hardware result pending

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

## Evidence and remaining gate

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

The real-weight gate is a fixed batch of the first eight cached GSM8K test rows,
selected before outputs, with private golds kept out of model inputs and Git.
At this candidate stage no batched hardware answer, speed or full32K input pass
is claimed. A completed short-question check would establish those examples
and the allocated capacity, not general accuracy or full-context answer quality.
