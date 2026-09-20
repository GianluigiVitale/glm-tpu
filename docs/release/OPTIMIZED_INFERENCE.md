# Ordinary greedy inference

This entry is a release candidate until the integration receipt in
[release status](STATUS.md) records trained validation and promotion.

The retained ordinary profile uses GLM-5.2-FP8 on the existing eight-host TPU v4
site. It supports one greedy request, with 8,192 slots shared by the prompt and
the entire output budget. Thinking is on/max; reasoning consumes that budget.
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
controller prints a private run directory containing `tokens.jsonl`, `answer.txt`
and `summary.json`. Tokens are written and flushed locally as generated; answer
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
