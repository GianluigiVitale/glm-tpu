# GLM-5.2-FP8 on 32 TPU v4 chips

A native JAX/Pallas inference implementation for the existing GLM-5.2-FP8 model
on eight hosts with four TPU v4 chips each. It makes checkpoint placement,
distributed execution, question answering and recovery inspectable on a fixed
private installation.

The ordinary greedy engine combines grouped routed experts, resident BF16
non-routed weights, sparse-attention selection, batched prefill and a packed
decode loop. The trained model and tokenizer are reused; the contribution is
their TPU implementation and its execution, correctness and recovery engineering.
[Project summary](docs/release/PROJECT_SUMMARY.md) ·
[Architecture](docs/release/ARCHITECTURE.md).

## Release results

The documented 8K path answered one GSM8K test question correctly and ended
normally. Expected and returned answer: **18**; the checkable arithmetic is
`(16 − 3 − 4) × 2 = 18`. This is one completed functional example, not a dataset
accuracy score or a guarantee about other questions.

| Measurement | Result |
|---|---|
| Decode, including fleet votes and local token writes | **14.55 tokens/s** |
| Prefill, 92 input tokens | **0.94 s; 98.10 tokens/s** |
| Cold checkpoint load and compilation | **1,102.03 s (18.4 min)** |
| Output through EOS | 265 tokens, including reasoning; 264 timed decode tokens |
| Prefill plus decode, excluding startup and warmup | 19.09 s |
| Maximum observed HBM per chip | 28,228,733,440 bytes |
| Hardware and capacity | Eight hosts / 32 TPU v4 chips; 8,192 combined prompt/output slots |

The [answer receipt](docs/release/single-answer-20260920.json) identifies tested
source `9469cd73`, the pinned GSM8K test row, output hashes, fresh graph/memory
checks, all-rank agreement and successful eight-host cleanup. A preceding
integration matched the 29-token reference prefix; that numerical check is
separate from answer correctness. [Checks and evidence](docs/release/STATUS.md).
Timings use the slowest host; prefill throughput varies with prompt length.

## Requirements and installation

Inference requires the existing `db-v4-64-od` site in `us-central2-b`, all eight
hosts, retained checkpoint shards and topology assets, the pinned tokenizer,
full Git history, Linux and the documented Python 3.12 environment. The runtime
uses JAX/jaxlib 0.10.1 and libtpu 0.0.41. Weights and site assets are not included.

Use the retained environment on the site. To install a separate review
environment, follow the pinned [installation recipe](docs/release/INSTALLATION.md),
including its explicit CPU PyTorch source. The wheel contains Python components;
hardware execution additionally requires the full source checkout and private
assets. [Checkpoint requirements and recovery](docs/release/CHECKPOINTS.md).

## Recommended inference path

From the clean, published checkout on authenticated rank0:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask \
  "Explain why the sky is blue in one sentence." \
  --context 8k --max-new-tokens 2048
```

Use the installed site interpreter described above. The command verifies source,
weights and all eight hosts, loads and compiles the model, generates from fresh
state, then prints the answer after cleanup. It prints a private run directory
containing streamed `item000/tokens.jsonl`, decoded `item000/answer.txt`, and
timing/identity receipts. Preparation opens no TPU devices; the protected worker
explicitly selects TPU execution.

The prompt and full output allowance must fit in 8,192 slots. Reasoning consumes
the 2,048-token allowance too, so a difficult question can stop without a final
answer. Check the recorded finish reason as well as the text. Every invocation
cold-loads and compiles: **this is not an interactive persistent server**.
No automatic retries occur; unresolved cleanup retains the workload leases.

[Full inference instructions](docs/release/OPTIMIZED_INFERENCE.md) describe
optional queues of up to ten questions and the larger profile. Generation is
sequential. Omitting `--context` selects that larger profile, so keep the explicit
8K option for this recommended path. [Legacy sampled inference](docs/release/INFERENCE.md)
has separate sampling, capacity and historical evidence.

## Offline CPU review

With an already installed CPU environment, these commands need no weights,
cloud access or TPU devices:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu info
JAX_PLATFORMS=cpu python -m glm_tpu doctor --profile core
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/release/test_cli.py tests/release/test_optimized_request.py \
  tests/release/test_optimized_launch.py tests/release/test_optimized_runtime.py \
  tests/release/test_optimized_ask.py
```

The subset checks request, failure and delivery contracts with synthetic CPU
results. The full site release command, `JAX_PLATFORMS=cpu python tools/check_release.py`,
also needs documented local assets and Git history.
[Testing scope and historical failures](docs/release/TESTING.md) ·
[Reviewer guide](docs/release/REVIEWER_GUIDE.md) ·
[Commit-bound source archive](docs/release/SHAREABLE_PACKAGE.md).

## Code and history

| Path | Purpose |
|---|---|
| `glm_tpu/optimized/` | Ordinary greedy engine and request runtime |
| `glm_tpu/greenfield/` | Shared kernels, checkpoint loading and frozen reference bodies |
| `scripts/release/` | Protected controllers, workers and evidence handling |
| `tests/` | CPU checks and historical evidence replays |
| `docs/release/` | Instructions, receipts and limitations |
| `docs/perf/` | Preserved experiments, failures and recovery |
| `docs/artifacts/` | Protected original receipts, including DB616–621 |
| `docs/curation/` | File dispositions and exact recovery information |

The [release integration history](docs/perf/ordinary-release-20260920.md) records
the failed and superseded attempts. The [research index](docs/perf/README.md)
preserves earlier results and branches separately. Optimization research is stopped.

## Limits and attribution

Only one easy answer is qualified here. Difficult questions produced prolonged,
unfinished reasoning. The larger profile's graphs compiled, but a full
131,072-token question and ten completed answers were not demonstrated. The
greedy extension does not inherit the legacy engine's long-context validation.
There is no HTTP endpoint, simultaneous model batching, durable KV recovery or
qualified speculative decoding. Review is assistant self-review, not independent
review.

Maintained by Gianluigi Vitale. GLM weights, architecture and tokenizer originate
with Zhipu AI; Transformers reference extracts and JAX/Pallas/XLA/libtpu are reused
with their own attribution and licenses. [Third-party notices](THIRD_PARTY_NOTICES.md).
Original code has no blanket open-source license. Keep the repository and review
archive private; the archive excludes weights, secrets, private prompts, raw
outputs and databases.

[Operations](docs/release/OPERATIONS.md) · [Development](CONTRIBUTING.md) ·
[Security](SECURITY.md).
