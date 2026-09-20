# Ordinary greedy questions

For the short release example, explicitly select the 8K profile:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Your question" \
  --context 8k --max-new-tokens 2048
```

The owner reduced answer acceptance to one correctly completed GSM8K example.
That check passed at executable9469cd73: expected18, returned18, EOS265tokens;
see the [receipt](single-answer-20260920.json). The larger profile and queue
remain implemented, with the evidence limits below.

The owner accepted queued generation for ten simultaneous submissions. From
the clean published checkout on rank0, use `python -m glm_tpu ask "question"`,
or `python -m glm_tpu ask --questions /private/questions.json` for a JSON array
of one to ten strings. One model load serves the whole queue. Each question has
fresh state and separate `itemNNN/tokens.jsonl` and `itemNNN/answer.txt` outputs.
The default `--context 128k` accepts at most 131,072 input tokens and uses
166,912 combined slots. By default, each question receives the remaining output
space up to the existing 163,840-token maximum: 35,840 output tokens remain for
a full 131,072-token input. An explicit `--max-new-tokens` cap is preserved.
The full budget must fit; no input is silently truncated. `--context 8k` selects the
prior 8,192 combined-slot profile and a default 2,048-token output budget.

The queued 128K question run was cancelled during its first difficult answer.
Its graph compilation passed; ten answers and a full 128K input did not complete.
The original
single-request 8K run at `4f551e6b` completed, matched the reference prefix and
cleaned up, but stopped at its 256-token cap during reasoning. See [STATUS](STATUS.md).
The API does not guarantee a correct answer to every question. Partial outputs
remain private after a failure; an incomplete answer must not be scored correct.

Use `--prepare-only` to create private inputs without a model run. The direct
`prepare-request`/controller interface below remains available for the 8K profile;
`--profile ordinary-greedy-128k` explicitly prepares the larger profile.

The [release status](STATUS.md) records trained validation and the location of
the commit-bound private publication record.

Both ordinary profiles use GLM-5.2-FP8 on the existing eight-host TPU v4
site. The explicit preparation example below selects the smaller 8K profile,
with 8,192 slots shared by the prompt and entire output budget.
Thinking is on/max; reasoning consumes that budget.
Checkpoint shards and the pinned tokenizer must already be present. No download,
resource creation, quantization change or speculative draft is performed.

From the clean, published source checkout and the existing Python 3.12 runtime,
prepare a private text-only message array outside Git:

```json
[{"role":"user","content":"Explain why the sky is blue in one sentence."}]
```

```bash
JAX_PLATFORMS=cpu python -m glm_tpu prepare-request \
  --profile ordinary-greedy-8k \
  --repo /home/gianl/glm-tpu-release \
  --tokenizer-root /home/gianl/gcs-models/models/GLM-5.2-FP8 \
  --messages /path/outside/git/messages.json \
  --output /path/outside/git/request.json \
  --request-id example-001 --max-new-tokens 2048

JAX_PLATFORMS=cpu python -m scripts.release.launch_ws32_optimized_request \
  --request /path/outside/git/request.json --wall-seconds 7200
```

Use existing private parent directories. Preparation validates the retained
tokenizer/template hashes and creates the request file exclusively with mode
0600. It does not run the model. The controller must run on authenticated rank0;
it acquires both workload leases and both staging locks, checks the idle fleet,
stages exact published source, authenticates all eight environments, then starts
one worker per host. It rejects dirty/unpublished source and unsupported requests.

Each invocation cold-loads and compiles the model, warms disposable state, then
executes the request from a fresh cache. It is not a persistent server. The
controller prints a private run directory containing `summary.json`.
For `ask`, each `itemNNN` subdirectory contains `tokens.jsonl` and `answer.txt`;
a directly prepared single request writes these files in the run directory.
Tokens are written and flushed locally as generated; answer
text is decoded at completion. Partial delivery remains available after failure;
generation is never automatically retried. An unresolved cleanup retains leases
and requires diagnosis of the recorded process identities before another run.

Decode throughput counts tokens after the first prefill-produced token, divided
by decode wall time, including model execution, required fleet votes and local
token writes/flushes. Cold load/compile, disposable warmup, prefill and final text
decoding are separate. This is not network streaming latency. Receipt hashes and
agreement establish which output was produced, not whether its answer is correct.

The earlier sampled long-context controller remains available via
[legacy inference](INFERENCE.md), with explicit `--profile legacy-sampled` during
preparation. Its different sampling, capacity and historical timings do not apply
to this greedy profile. Detailed experiments remain in the
[frozen research record](../perf/frozen-20260920/RESULTS_AND_DECISIONS.md).
