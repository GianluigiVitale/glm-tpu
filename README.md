# GLM-5.2-FP8 on 32 TPU v4 chips

A native JAX/Pallas implementation of GLM-5.2-FP8 for eight hosts with four TPU
v4 chips each. The project makes distributed inference, checkpoint placement,
request delivery and validation inspectable on a fixed private installation.

The ordinary engine combines grouped routed experts, resident BF16 non-routed
weights, optimized sparse-attention selection, batched prefill and a packed
decode loop. The question interface accepts **up to ten queued questions**, with
one generating at a time and a separate cache for each. The 128K input profile
has 166,912 combined prompt/output slots; its hardware check is pending. The
previous 8K profile has a completed integration run. The model, trained weights
and tokenizer are reused; the contribution
is their TPU execution and the surrounding correctness and recovery engineering.
[Short project summary](docs/release/PROJECT_SUMMARY.md) ·
[Architecture](docs/release/ARCHITECTURE.md).

## Release results

Candidate `4f551e6b` passed **571 CPU tests, one skipped**, plus source,
content and isolated package checks. Its 8K integration run delivered
**14.35 decode tokens/s**, with **142.04 prompt tokens/s** at 2,034 prompt tokens
and **1,095.33 seconds** cold load/compile. All eight hosts agreed with the
29-token reference prefix and cleaned up. The 256-token cap stopped during
reasoning; this was not a completed answer. The queued 128K extension and ten
randomly selected difficult questions remain pending. Main is not yet promoted.
[Validation and receipts](docs/release/STATUS.md).

Decode throughput counts output tokens after the first prefill-produced token,
including required fleet votes and rank0 local token writes/flushes. Startup,
warmup and prefill are measured separately; no network streaming rate is claimed.
Token agreement checks a numerical regression, not answer correctness.

## Requirements and installation

Actual inference requires the existing `db-v4-64-od` site in `us-central2-b`,
all eight hosts / 32 chips, retained checkpoint shards and topology assets,
the pinned tokenizer, full Git history, Linux and the documented Python 3.12
runtime. The runtime uses JAX/jaxlib 0.10.1 and libtpu 0.0.41. Checkpoints must
already be available; the inference command neither downloads weights nor
provisions hardware.

Use the retained environment on the site. For a separate review environment,
follow the pinned [installation recipe](docs/release/INSTALLATION.md), including
the explicit CPU PyTorch source. The wheel contains Python components; hardware
execution additionally needs the full source checkout and private site assets.
[Checkpoint requirements and recovery](docs/release/CHECKPOINTS.md).

## Recommended inference path

From the clean, published checkout on authenticated rank0:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Explain why the sky is blue in one sentence."
```

To submit ten questions together, put their strings in a private JSON array
outside Git, then run:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask \
  --questions /path/outside/git/questions.json --context 128k
```

The model loads once and processes the queue in order, writing separate outputs
and printing the answers when the run finishes. This is queued generation, not
ten simultaneous model executions. The default output budget is 32,768 tokens
per question; input plus the whole output budget must fit 166,912 slots.

For explicit preparation, create a text-only messages file, for example:

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

Preparation is local and opens no TPU devices. The controller acquires both
workload leases, authenticates all eight hosts, verifies source and weights,
compiles and admits fresh graphs, warms disposable state and executes from a
fresh cache. It prints a private run directory with streamed `tokens.jsonl`,
decoded `answer.txt` and timing/identity receipts. A file called `answer.txt`
can contain capped reasoning; check the stop reason and final answer.

Each invocation cold-loads and compiles. Thinking is on/max and consumes the
output budget. Generation never automatically retries, and unresolved cleanup
retains the workload leases. [Full ordinary instructions](docs/release/OPTIMIZED_INFERENCE.md).
The separately documented [legacy sampled path](docs/release/INFERENCE.md)
has different sampling and historical evidence.

## Offline review

With an already installed CPU environment, these commands need no weights,
cloud access or TPU devices:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu info
JAX_PLATFORMS=cpu python -m glm_tpu doctor --profile core
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/release/test_cli.py tests/release/test_optimized_request.py \
  tests/release/test_optimized_launch.py tests/release/test_optimized_runtime.py
```

Doctor checks installed metadata. The small test subset checks request, failure
and delivery contracts with synthetic CPU results. The full site release check
is `JAX_PLATFORMS=cpu python tools/check_release.py`; it also needs documented
local assets and Git history. [Testing scope](docs/release/TESTING.md) records
historical whole-tree failures rather than hiding them.
[Reviewer guide](docs/release/REVIEWER_GUIDE.md) ·
[Commit-bound source package](docs/release/SHAREABLE_PACKAGE.md).

## Code and history

| Path | Purpose |
|---|---|
| `glm_tpu/optimized/` | Ordinary greedy engine and request runtime |
| `glm_tpu/greenfield/` | Shared kernels, checkpoint loading and frozen reference bodies |
| `scripts/release/` | Protected controllers, workers and evidence handling |
| `tests/` | CPU checks and historical evidence replays |
| `docs/release/` | Instructions, architecture, receipts and limitations |
| `docs/perf/` | Detailed successes, failures, superseded experiments and recovery |
| `docs/artifacts/` | Protected original receipts, including DB616–621 |
| `docs/curation/` | File dispositions and exact recovery information |

Research branches and originals are preserved. Start with the
[historical results and decisions](docs/perf/frozen-20260920/RESULTS_AND_DECISIONS.md)
or [recovery index](docs/perf/README.md). Optimization research is stopped.

## Limits and attribution

This is a private, site-specific engine: no HTTP endpoint, simultaneous model
batching, durable KV recovery or qualified speculative decoding. Queued questions
share startup but wait for earlier answers. The greedy extension does not inherit
the legacy engine's long-context validation.
Short token agreement and completed examples do not establish model-card
accuracy; prior research answers include factual/format failures and unfinished
reasoning. Review is assistant self-review, not independent review.

Maintained by Gianluigi Vitale. GLM weights/architecture/tokenizer originate with
Zhipu AI; Transformers reference extracts and JAX/Pallas/XLA/libtpu are reused
with their own attribution and licenses. [Third-party notices](THIRD_PARTY_NOTICES.md).
Original code has no blanket open-source license; keep this repository and its
review package private. Weights, secrets, private prompts and raw outputs are
excluded from the package.

[Operations](docs/release/OPERATIONS.md) · [Development](CONTRIBUTING.md) ·
[Security](SECURITY.md).
