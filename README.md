# GLM-5.2 on 32 TPU v4 chips

Native JAX inference for GLM-5.2-FP8 on eight hosts, four TPU v4 chips per host.
Performance research was frozen on **September 20, 2026**.

## Latest retained results

**Optimized ordinary research decode: approximately 14.3 output tokens/s.**
**Prefill: 138.85 prompt tokens/s at a 2,034-token prompt.**

| Measurement | Latest retained ordinary research result |
|---|---:|
| Decode across the fixed prose/code/structured suite, two repeats each | **14.3175 wall tok/s** |
| Short 2,034-token request: prefill | **138.85 prompt tok/s** |
| Same short request: decode | **14.04 wall tok/s** |
| Short-request reference token agreement | **29/29 on all eight hosts** |

The two decode rates come from different workloads. The suite rate is total
timed delivered tokens divided by total decode wall time across all six cases.
Cold loading/compilation, prefill and network transport are excluded from decode.
The short request uses in-memory delivery; the suite includes token-file
write/flush and used component profiling.

**What main actually runs:** the supported inference command still uses the
older release engine, whose historical 2K measurement was **7.66 decode tok/s**.
The September 20 merge published the research documentation only.
The approximately 14.3 tok/s implementation remains on
[`perf/reference-lowhanging-fruit-20260919`](https://github.com/GianluigiVitale/glm-tpu/tree/perf/reference-lowhanging-fruit-20260919),
frozen at [`e3290fd8`](https://github.com/GianluigiVitale/glm-tpu/tree/e3290fd8).
Checking out main does not enable that faster engine.

The retained research path uses grouped experts, resident BF16 non-routed
weights, optimized DSA, a packed decode loop and optimized prefill.
MTP/speculative decoding was **not qualified**: long outputs diverged from
ordinary decode. Later upstream-inspired adaptations established no qualified
end-to-end improvement. The cancelled final comparison remains unmeasured.

Short reference-token agreement does not establish general answer quality.
In the completed suite, prose needed technical corrections, code exhausted its
token budget, and structured values were correct but failed the requested format.

[Measured rows and answer checks](docs/perf/frozen-20260920/MEASUREMENTS.md) ·
[What worked and what failed](docs/perf/frozen-20260920/RESULTS_AND_DECISIONS.md) ·
[Research index and recovery](docs/perf/README.md)

## Historical validation

The older release completed four protected 128K retrieval runs, a 262,144-token
capacity run, and one ordinary user-response smoke test. Those establish
specific historical capabilities, not validation of the optimized research
implementation at those context lengths.

[Older release measurements](docs/perf/frozen-20260920/MEASUREMENTS.md#historical-release-engine-baselines) ·
[Release validation and limits](docs/release/STATUS.md) ·
[Detailed research notebooks](docs/perf/frozen-20260920/history/README.md)

## Use this repository

This is a private, site-specific, single-request engine. Actual inference needs
retained checkpoint shards, full Git history, Python 3.12 and the documented
site configuration. The supported controller cold-loads and compiles for each
invocation; its historical smoke test took approximately 38 minutes to start.
Research is frozen; historical launch instructions do not authorize new runs.

Offline inspection:

```bash
python -m glm_tpu info
python -m glm_tpu doctor --profile core
```

These commands inspect metadata without initializing TPU devices or fetching
weights. Follow [installation](docs/release/INSTALLATION.md),
[checkpoint recovery](docs/release/CHECKPOINTS.md) and
[inference](docs/release/INFERENCE.md) for the supported release path.

CPU release checks:

```bash
JAX_PLATFORMS=cpu python tools/check_release.py
```

The September 20 freeze passed **524 tests, one skipped**, plus source, content
and isolated package checks. [Testing scope and historical whole-tree failures](docs/release/TESTING.md)
remain documented. CPU checks do not replace trained-model validation.

## Repository map

| Directory | Purpose |
|---|---|
| [`glm_tpu/`](glm_tpu/) | Supported release engine, kernels, checkpoint loading and CLI |
| [`scripts/`](scripts/) | Release controller, host operations and retained validation tools |
| [`tests/`](tests/) | CPU tests and historical evidence checks |
| [`docs/perf/`](docs/perf/README.md) | Latest research results, measured comparisons and archived history |
| [`docs/release/`](docs/release/STATUS.md) | Installation, architecture, inference and release validation |
| [`docs/artifacts/`](docs/artifacts/) | Protected original benchmark receipts |
| [`docs/curation/`](docs/curation/README.md) | File dispositions and exact recovery information |

[Architecture](docs/release/ARCHITECTURE.md) ·
[Reviewer guide](docs/release/REVIEWER_GUIDE.md) ·
[Observability](docs/greenfield/GATE_D_OBSERVABILITY_PLAYBOOK.md)

## Project status

Maintained by Gianluigi Vitale. Optimization is stopped. Research code and
original evidence remain on preserved branches; failed and superseded logs
have verified recovery archives. Weights, credentials, private prompts and raw
databases stay outside Git.

The release has no HTTP endpoint, concurrent batching, durable KV recovery or
qualified speculative decoding. Full model-card quality is unestablished;
256K capacity evidence had no answer-correctness oracle. The sampled release
request graph has a separate 166,912-token prompt-plus-output capacity.

[Development](CONTRIBUTING.md) · [Operations](docs/release/OPERATIONS.md) ·
[Security](SECURITY.md) · [Third-party notices](THIRD_PARTY_NOTICES.md).
Public distribution still requires a licensing decision and provenance/privacy review.
