# GLM TPU

### GLM-5.3 FP8 inference in native JAX on 32 TPU v4 chips

A systems engineering project that takes a trained mixture-of-experts model
from checkpoint shards to completed answers: distributed weight placement,
sparse attention, expert routing, cache ownership and concurrent decoding.

**13.57 tokens/s for one chat · Keep the model loaded · Four-chat batching**

[Results](#measured-results) · [Layout](#repository-layout) ·
[Check without TPU hardware](#check-without-tpu-hardware) ·
[Run inference](#run-inference) · [Documentation](#documentation) ·
[License](#license)

## Engineering contribution

The central problem is making checkpoint placement, communication, compiler
memory use and independent conversation state agree across eight hosts and
32 TPU v4 chips.

| Area | Implementation |
|---|---|
| Distributed execution | An expert-8 x feature-4 device mesh keeps the hidden state sharded; every collective runs over one mesh axis, and full-pod exchanges are limited to small consensus values. |
| Model computation | Grouped routed experts over FP8 weights, resident BF16 non-routed weights, DSA sparse attention, native JAX and Pallas kernels. |
| Conversation state | Shared weights with independent caches and stopping; four-chat batching prefills prompts one after another and decodes them together. |
| Resident operation | Sequential requests reuse the loaded weights and compiled programs, with fresh state for each request. |
| Reproducibility | Verified checkpoint packing, source-bound launches, graph and memory admission, per-host token receipts, and a graph-equivalence harness that proves refactors leave the device programs unchanged. |

GLM's architecture, trained weights and tokenizer are reused. JAX, Pallas, XLA
and libtpu provide the compiler and runtime. The contribution is the TPU
implementation and its integration. [Architecture](docs/release/ARCHITECTURE.md) ·
[Notices](THIRD_PARTY_NOTICES.md).

```mermaid
flowchart LR
    W[Verified shared weights] --> M[Resident model on 32 TPU v4 chips]
    Q[Private question queue] --> P[Fresh conversation state]
    P --> M
    M --> A[Answer and timing receipts]
```

## Measured results

**740 of 770 GSM8K questions scored correct: 96.1% on the evaluated subset.**
The owner stopped after test rows 0–769; 549 of the 1,319 test questions were not
run. The 30 unsuccessful cases include three that exhausted the context. This is an
ordered partial evaluation, not a full-test-set accuracy claim.
[Result receipt](docs/release/glm53-resident-results-20260922.json).

| Measurement | Result |
|---|---:|
| Single-chat decode, one completed answer | **13.57 tokens/s** |
| Decode across the 770-question sequential evaluation | **13.50 tokens/s** |
| Solo prompt prefill, 85 tokens | **0.965 s** |
| Solo decode, 239 timed tokens | **17.612 s** |
| Solo cold verification, loading and compilation | **1,219.89 s · 20.33 min** |
| Maximum observed HBM per chip, resident evaluation | **28.23 GB** |
| Four simultaneous chats, per active chat | **4.89–5.12 tokens/s** |

Output counts include thinking. Timings use the slowest host; the evaluation
rate is total timed decode tokens divided by summed per-question decode time.
It excludes prompt prefill, queue overhead and startup. The first output token
comes from prefill. Weights and compiled programs were reused across 770 questions;
every question started with independent conversation state.

The evaluation used greedy decoding, maximum thinking, full remaining 32K
allowances and a boxed-answer format instruction without a brevity instruction.
References stayed out of the model inputs. Scoring compares final-channel numbers
exactly; unfinished reasoning never counts as a correct answer. All-host token
agreement is checked separately. A checkable solo example is house profit:
`80,000 × 2.5 − (80,000 + 50,000) = 70,000`.

The separate [four-chat test](docs/release/glm53-four-answers-20260921.json)
completed four correct answers at normal EOS. Its mixed-length aggregate decode
was 9.27 tokens/s, distinct from each chat's speed.
[Timing boundaries and limitations](docs/release/STATUS.md).

## Repository layout

| Path | Contents |
|---|---|
| `glm_tpu/` | The engine package (`python -m glm_tpu`, console command `glm-tpu`). |
| `glm_tpu/entrypoints/` | The command line, the loopback chat UI, the OpenAI-compatible `/v1` API and their server. |
| `glm_tpu/executor/`, `glm_tpu/worker/` | The rank-0 controller that stages and launches a run on the fleet, and the per-host worker. |
| `glm_tpu/engine/`, `glm_tpu/runner/` | Requests, the per-request host loop and the resident protocol; program building, compilation and admission. |
| `glm_tpu/models/`, `glm_tpu/layers/`, `glm_tpu/kernels/` | GLM-5.3 (`glm_moe_dsa`), its per-shard layer bodies and the Pallas TPU kernels. |
| `glm_tpu/model_loader/`, `glm_tpu/distributed/`, `glm_tpu/config/` | The checkpoint pipeline, the multi-host runtime and mesh, the model and site configuration. |
| `tests/` | CPU tests, laid out like `glm_tpu/`; `tests/golden/` holds the equivalence records, `tests/reference/` an unsharded reference model. |
| `tools/equivalence/` | The graph-equivalence harness ([README](tools/equivalence/README.md)). |
| `examples/site.example.toml` | The template of the untracked site file that names the fleet, paths and checkpoint pins. |
| `docs/` | The documentation listed below. |

A map of every module is in [ARCHITECTURE](docs/release/ARCHITECTURE.md).

## Check without TPU hardware

Python 3.12 on Linux. Install into a new virtual environment (never into a
working TPU environment); [INSTALLATION](docs/release/INSTALLATION.md) has the
details and the other extras:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python '.[runtime,dev]'
```

No weights, cloud credentials or TPU devices are needed for:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m glm_tpu info
JAX_PLATFORMS=cpu .venv/bin/python -m glm_tpu collect-env --profile runtime
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/cli/test_main.py tests/entrypoints/cli/test_ask.py tests/entrypoints/cli/test_prepare.py tests/entrypoints/cli/test_collect_env.py tests/entrypoints/cli/test_checkpoint.py tests/engine/test_request.py tests/engine/test_llm_engine.py tests/executor/test_multihost_executor.py tests/executor/test_fleet.py tests/worker/test_tpu_worker.py tests/utils_/test_io_utils.py tests/config/test_site.py tests/runner/test_tpu_runner.py
```

These check the command line, request integrity and capacity refusals, the
controller's identity, SSH and failure gates, the checkpoint commands and the
delivery and deadline behaviour with synthetic CPU results, in about half a
minute. The full suite and the equivalence gates are described in
[TESTING](docs/release/TESTING.md). Tests establish software contracts; model
speed and answer evidence come from real weights on TPU.

## Run inference

Hardware inference needs a TPU v4 slice of **eight hosts with 32 chips**, the
verified GLM-5.3 owner checkpoint on every host, the topology binding assets, a
clean published checkout on an allowed branch and the pinned Python 3.12
environment (JAX/jaxlib 0.10.1, libtpu 0.0.41). An untracked site file names all
of it ([`examples/site.example.toml`](examples/site.example.toml)).
[Installation](docs/release/INSTALLATION.md) · [Checkpoint](docs/release/CHECKPOINTS.md) ·
[Operations](docs/release/OPERATIONS.md).

To start a session from rank 0 when the fleet is idle:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Your question" --keep-loaded --wall-seconds 14400
```

The default is **32,768 combined input/history/thinking/output slots**
(`--context 8k|32k|128k|256k`). The runtime reserves the cache and compiles fixed
shapes at startup; shorter questions do not have to fill that capacity. Omitting
an output cap gives the answer every remaining slot. Thinking is enabled at
maximum effort.

The command prints a private run directory and leaves the model loaded after
answering. Later prepared requests go to that session's inbox; they do not start
another model. Full history must be supplied to continue a conversation.
[Submission, stopping and four-chat usage](docs/release/OPTIMIZED_INFERENCE.md).

For a browser workspace, the [chat UI](docs/UI.md) attaches to an existing
resident session: saved conversations, streamed answers, thinking, light/dark
themes and a mobile layout. Forward its loopback port 8011 to open it locally.
The model stays loaded when the UI closes. The same server also exposes a
stateless [OpenAI-compatible `/v1` API](docs/API.md) with tool calling and
streaming, for local development tools.

This is a single-site engine with a private file queue, a local chat UI and a
key-authenticated loopback API. There is no automatic recovery of live model or
KV state after a process failure. Resident mode serves sequential requests;
four-chat batching is a separate invocation. Short prompts do not establish
full-32K-input quality. Public benchmark familiarity and the owner-selected
stopping point limit the interpretation of the partial score.

## Documentation

| Page | For |
|---|---|
| [Project summary](docs/release/PROJECT_SUMMARY.md) | the problem, approach, results and limits on one page |
| [Installation](docs/release/INSTALLATION.md) | environments, extras, `collect-env`, the wheel, the site file |
| [Ordinary inference](docs/release/OPTIMIZED_INFERENCE.md) · [Four conversations](docs/release/CONCURRENT.md) | `ask`, resident sessions, the inbox, stopping, batching |
| [Chat UI](docs/UI.md) · [Local API](docs/API.md) | the browser workspace and the `/v1` API |
| [Checkpoint](docs/release/CHECKPOINTS.md) | the pinned model, the packed owner files, `checkpoint inventory` and `checkpoint verify` |
| [Operations](docs/release/OPERATIONS.md) | the controller, launch policy, locks, failure handling and diagnosis |
| [Architecture](docs/release/ARCHITECTURE.md) | the module map, the request path, parallelism and numerical conventions |
| [Testing](docs/release/TESTING.md) · [Equivalence harness](tools/equivalence/README.md) | the test layout, tiers and gates |
| [Reviewer guide](docs/release/REVIEWER_GUIDE.md) | a reading order and the source package |
| [Release status](docs/release/STATUS.md) · [Migration history](docs/release/GLM53_MIGRATION.md) | the measured GLM-5.3 release and how it was reached |
| [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) | the development policy and the release checks |

## History

The GLM-5.2 implementation and measurements are preserved at the tag `glm-5.2`;
its weight payloads were retired. The research tree this release was cut from
(research history, the curation ledger, the legacy sampled and long-context
interfaces, benchmarks and their evidence) is preserved at the tag
`archive/research-20260922`; for example
`git show archive/research-20260922:docs/perf/README.md`.

## License

Copyright 2026 Gianluigi Vitale. The project's own work is licensed under the
[Apache License 2.0](LICENSE). Third-party material keeps its own license: the
GLM-5.3 configuration files under `glm_tpu/models/glm_moe_dsa/hf_config/` are
under Z.AI's GLM-5.3 license ([notices](THIRD_PARTY_NOTICES.md)).

Maintained by **Gianluigi Vitale**. The repository is private. Review is
assistant self-review, not independent review.
