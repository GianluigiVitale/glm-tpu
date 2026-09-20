# GLM TPU

### Native-JAX long-context inference on 32 TPU v4 chips

A systems research implementation of **GLM-5.2-FP8**: a complete 78-layer
decoder, topology-aware sharding, batched prefill, and an evidence trail linking
execution to checkpoint bytes, compiler graphs, memory and wall time.

**Measured milestones:** four 128K retrieval runs · a 262,144-token capacity run ·
a protected ordinary user response · 2,832 passing CPU tests on the curated tree.
These are distinct validation results, not a claim of full model-quality parity.

[Architecture](docs/release/ARCHITECTURE.md) ·
[Results](#measured-results) · [Reviewer guide](docs/release/REVIEWER_GUIDE.md) ·
[Installation](docs/release/INSTALLATION.md) · [Inference](docs/release/INFERENCE.md)

> **Scope:** a private, site-specific, single-request research engine—not a
> portable hosted service or a state-of-the-art throughput claim. The supported
> controller cold-loads and compiles on each invocation; the user smoke measured
> approximately 38 minutes for that startup phase.

## The systems problem

Making a large mixture-of-experts model execute is only part of accelerator
portability. Weight placement, sparse selection, cache state, compiler behavior
and physical communication must agree, while leaving memory for long contexts.
A fast device kernel alone does not establish a correct or fast request.

This project implements the native execution path and the machinery to inspect
those boundaries together. It combines DeepSeek Sparse Attention (DSA) operations
used by GLM, IndexShare, routed/shared experts, and explicit distributed ownership.
The attention mechanism's name does **not** mean this runs the DeepSeek model:
the target is GLM-5.2-FP8, and inference does not execute through legacy
`tpu-inference`.

### Engineering focus

- **Topology-aware execution.** WS32_2D uses all 32 chips with explicit feature-four
  and expert-eight groups. Hidden state stays sharded rather than being
  reconstructed across the full pod repeatedly inside each layer.
- **Separate prefill and decode.** Layer-major prefill processes prompt rows in
  B128/B114 shapes; decode has one live row. These are not concurrent request batches.
- **Weight integrity and placement.** Packed payloads, scales, manifests and
  direct local-shard loading connect retained source weights to device layout.
- **Inspectable behavior.** DSA selection/tie checks, cache checks, compiler-graph
  inspection and per-chip memory measurements accompany original tokens and wall
  time. Each check has a specific scope; none proves universal bitwise equivalence.
- **Failure-aware operation.** Exact source pins, ownership checks and serialized
  launches protect execution. Upload/collection recovery preserves originals
  rather than rerunning a response to replace missing evidence.

These are implementation areas, not claims that each technique is novel.
PP8/PP16 exploration and its evidence remain in research history; the supported
release path is WS32_2D.

## Execution and evidence

```mermaid
flowchart TB
    input["Private messages + pinned tokenizer"] --> request["Validated request"]
    request --> admission["Controller: source / ownership / resource checks"]
    weights["Verified checkpoint shards + scales"] --> engine
    admission --> engine["Native JAX · 8 hosts / 32 TPU v4 chips"]
    engine --> prefill["Batched prefill → causal cache"]
    prefill --> decode["Single-row decode + live request state"]
    decode --> output["Local token stream + terminal response"]
    engine -.-> evidence["Compiler graphs / memory / traces / integrity"]
    decode -.-> evidence
    output --> replay["Validate original request evidence"]
    evidence --> replay
    replay --> archive["Result DB + generation-bound archive + idle census"]
```

Conceptual dataflow, not physical wiring or a latency timeline. Device traces and
profiler-free timing have separate scopes. A locally flushed token is not a
network-delivered latency measurement.

## Measured results

Historical protected runs on the same 8-host, 32-chip TPU v4 installation. Each
receipt records its own source/recovery pins and validation boundary. These are
**not** a new benchmark of the documentation release, an optimized scaling study,
or an apples-to-apples comparison against another serving engine.

| Workload | Prefill tokens/s | Decode wall tokens/s | Evidence and scope |
|---|---:|---:|---|
| 2,034-token prompt | 62.761 | 7.660 | [DB610](docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json): scoped short-context numerical checks |
| 127,363-token passkey, depth 0.95 | 45.459 | 6.935 | [DB619](docs/artifacts/prefill-delivery-db619-sealed-20260912.json): retrieval on this protected prompt |
| 262,144-token E0 | 32.157 | 6.148 | [DB620](docs/artifacts/prefill-delivery-db620-sealed-20260912.json): capacity and structural checks; **no correctness oracle** |

Prefill excludes cold loading/compilation and later decode preparation. Decode
rates are steady wall measurements, not aggregate multi-request throughput.
All four 128K passkey depths completed (DB616–619). The 256K run measured
**29.930 GB peak HBM per chip** and **3.084 GB minimum headroom** (decimal GB).

### Opt-in performance research

Native MTP-assisted speculation has completed paired real-weight repeats.

| Request | Ordinary wall tok/s | One draft | Two drafts |
|---|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 |
| Structured output | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 |

Two drafts gain 4.3–4.6% on code and 10.7–10.9% on structured output, but lose
on prose; the 25% working target is unmet. Speculative trails differ from ordinary.
Prose needs scoped corrections; code remains unfinished at the cap; structured
values are correct but Markdown fences fail the requested format. All eight hosts
were authenticated idle. This is research evidence, not an engine deployment or
broad quality claim. [Completed comparison](docs/perf/tpu-real-native-suite-20260920T031727Z.json)
and [full progress](docs/perf/MTP_PROGRESS_20260919.md). The evidence-only
checkpoint is merged into private main at `c142d284`, with verified regional
backup; [publication receipt](docs/perf/mtp-evidence-promotion-20260920.json).

A [subsequent trained replay](docs/perf/tpu-real-prefix-replay-20260920T090655Z.json)
reproduces code R2/R3 and prose R3 disagreement from identical ordinary starting
states, before draft acceptance. Zero-token rollback is bitwise equal on all
eight hosts; multi-row cache differences reach layer 0. The completed
[layer/head trace](docs/perf/tpu-real-prefix-trace-20260920T100227Z.json) confirms
first-layer R2/R3 differences on all eight hosts with stable instrumentation;
R1 instrumentation itself changes the computation and cannot establish its cause.
An unrolled-attention candidate is undergoing trained correctness and paired
target-window latency checks. These diagnostics add no accepted-throughput or
serving qualification to the table above.

Synthetic weights at real GLM geometry, all 78 layers on the same 32 TPU v4 chips:

| Measurement | Frozen | Challenger | Evidence |
|---|---:|---:|---|
| Complete 2K prefill, capacity 2,560 | 64.14 prompt tok/s | 123.32 prompt tok/s | [Eight-rank receipt](docs/perf/tpu-microbench-prefill-model-20260919T150628Z.json) |
| Complete 128K prefill, capacity 131,584 | Not paired | 85.44 prompt tok/s | [Eight-rank receipt](docs/perf/tpu-microbench-prefill-128k-20260919T152806Z.json) |
| Decode, capacity 8,192 | 8.24 tok/s | 15.57–15.60 tok/s (bug affected) | [Frozen comparator](docs/perf/tpu-microbench-replica-correct-20260919T143305Z.json), [100-step repeat](docs/perf/tpu-microbench-fused-reductions-20260919T145930Z.json) |

The sampled challenger also measures **14.11 → 14.71 wall tok/s** with the compact
host loop, including eight-host votes and an in-memory delivery sink. Tokens and
final state/residual agree bitwise; this is separate from the greedy step above.
[Paired receipt](docs/perf/tpu-microbench-request-loop-20260919T153427Z.json). Both loop variants used the affected
routed kernel; this isolates host overhead but does not establish correct serving speed.

A real-weight 2K DB610 comparison measured **138.6 prompt tok/s** and
**15.18–15.28 model decode tok/s**, but **failed token parity**: only the prefill
token matched; all 28 decode outputs were zero. Fleet health, graph/memory checks
and final cleanup passed. These timings are a rejected numerical candidate, not
an accepted serving speedup. [Real-weight receipt](docs/perf/tpu-real-db610-20260919T163936Z.json).

An empty-owner routed-kernel bug has since been reproduced on TPU and fixed
in an explicit candidate. Earlier grouped-MoE timings remain affected; corrected
real-model validation resolves the explosion but still matches only 17/29 DB610
tokens (first mismatch index 11). It measures 138.8 prompt tok/s and 14.77–15.05
model decode tok/s and remains unadmitted. [Corrected receipt](docs/perf/tpu-real-empty-fixed-20260919T173656Z.json). [Diagnosis and fix evidence](docs/perf/REAL_WEIGHT_VALIDATION_20260919.md).

Full 29-token ablations now isolate the divergence to **decode D5**:
**D1+D8 and D1+D8+D10 match all 29 DB610 tokens on all eight hosts** from
the same D8/P1/P2 prefill state. D1+D8+D5 reproduces 17/29 matches.
The shared prefill measured **139.04 prompt tok/s**; these diagnostic decode
variants make no timing claim.
[Ablation receipt](docs/perf/tpu-real-ablation-20260919T180952Z.json).

A clean real-weight D1/D8/D10 run also passes **29/29 tokens on all eight hosts**:

| Research candidate at 2,034 prompt tokens | Prompt tok/s | Model decode tok/s | Decode p50 |
| --- | ---: | ---: | ---: |
| D8/P1/P2 prefill + D1/D8/D10 decode | 138.95 | 14.55–14.69 | 66.98–67.70 ms |

Prefill includes block health votes and receipts; decode excludes host checks
and delivery. This is about 2.21× DB610 prefill throughput; decode is not directly
comparable with DB610's wall timing. All graph/memory checks and authenticated
fleet cleanup passed. [Clean real-weight receipt](docs/perf/tpu-real-no-d5-20260919T184958Z.json).


A paired trained-weight host-loop trial also passes 29/29 tokens and bitwise
final state/residual: **13.32 -> 14.04 wall decode tok/s (+5.44%)** with D4.
This includes host checks and an in-memory delivery sink, excluding prefill,
compilation and five warm steps (23 timed steps); it is not network latency.
[Real request-loop receipt](docs/perf/tpu-real-request-loop-20260919T192804Z.json).

A fresh long question measured **14.41 wall decode tok/s** over 6,143 decode
steps (6,144 generated tokens including prefill's first token), with host votes
and rank0 token-file write/flush included. All eight hosts agreed; the preceding
29-token DB610 check passed. The 338-token prompt took 2.96 s to prefill;
warmed first-token latency was 3.20 s. Cold loading/compilation and network
transport are excluded. Generation reached its 6,144-token cap during reasoning,
so this run does **not** establish a correct completed answer or an MTP speedup.
[Long-question receipt](docs/perf/tpu-real-long-question-20260920T005757Z.json).

These opt-in programs are outside the frozen release. Synthetic token outputs
differ; the passing DB610 trails are a narrow real-weight check, not general
model-quality validation. The fused feature-
reduction experiment failed and remains disabled. [Scope and numerical boundaries](docs/perf/D4_D8_PROGRESS_20260919.md).

### Ordinary user-response validation

[DB621](docs/release/user-response-db621-sealed-20260914.json) exercised the release
request path: a 19-token prompt, 71 generated tokens, the requested `READY` answer,
EOS termination, original evidence validation and authenticated eight-host cleanup.
Generation includes reasoning tokens; this is a smoke test, not a quality score.

Startup loading/compilation took 2,283.431 seconds. After that separate phase,
local first-token delivery took 9.368 seconds; the 50.241-second request wall was
**instrumented**, not a clean steady-service latency measurement.
[Full qualifications and evidence linkage](docs/release/STATUS.md).

### What is not established

- Full GPQA/AIME accuracy or agreement with the original model card. The stopped
  campaign's incomplete prefix is not a dataset accuracy estimate.
- Blanket bit-exact equivalence across builds, contexts or outputs. The 2K
  numerical result does not prove the current path at 8K.
- General task quality at 256K: E0 had no correctness oracle.
- HTTP serving, concurrent batching, persistent warm serving, durable KV recovery
  or speculative decoding. Pause/resume is within the same live session.
- A sampled 256K user endpoint: the current sampled graph has capacity 166,912
  tokens shared by prompt and generation; the 256K E0 graph is separate.

## Start here

For an offline first look, from this checkout with Python 3.12:

```bash
python -m glm_tpu info
python -m glm_tpu doctor --profile core
```

These commands inspect metadata without initializing TPU devices, downloading
weights or starting inference. `doctor` reports missing/mismatched dependencies;
it does not grant launch authority.

With the documented development environment installed:

```bash
JAX_PLATFORMS=cpu python tools/check_release.py
```

The release check on the curated tree passed with 524 tests passed and 1 skipped, plus
source, content and isolated package checks; the whole retained CPU tree is
recorded in [TESTING](docs/release/TESTING.md): 2,832 passed, 122 skipped with stated reasons and 173 failed, every failure being a historical admission test bound to a sealing-source pin or a sealed-identity assertion that fails identically at the starting pin (none introduced by curation).
[Release-check receipt](docs/release/curation-cpu-check-20260915.json) ·
[Whole-tree receipt](docs/release/curation-whole-tree-cpu-20260915.json) ·
[Fresh-install receipt](docs/release/fresh-install-20260914.json).
CPU tests do not replace hardware evidence.

Actual inference needs retained weights, full Git history, a reviewed source pin
and the existing site configuration. Follow [installation](docs/release/INSTALLATION.md),
[checkpoint recovery](docs/release/CHECKPOINTS.md) and the
[user-request example](docs/release/INFERENCE.md); the wheel alone is not a server.
Do not use a historical campaign script as a generic installer.

## Explore the implementation

| Area | Entry point |
|---|---|
| Layer-major prefill | [ws32_batched_prefill.py](glm_tpu/greenfield/runtime/ws32_batched_prefill.py) |
| Decoder construction | [ws32_decoder.py](glm_tpu/greenfield/runtime/ws32_decoder.py) |
| Live request state, delivery and stop policy | [ws32_request_session.py](glm_tpu/greenfield/runtime/ws32_request_session.py) |
| JAX and Pallas operations | [kernels/](glm_tpu/greenfield/kernels/) |
| Weight layout and direct loading | [checkpoint/](glm_tpu/greenfield/checkpoint/) |
| WS32 mesh, sharding and HLO contracts | [sharding/](glm_tpu/greenfield/sharding/) |
| Protected user controller and recovery | [scripts/release/](scripts/release/) |
| Request/failure-path checks | [tests/release/](tests/release/) |
| Per-file curation ledger and recovery | [docs/curation/](docs/curation/README.md) |
| Performance analysis vs. the Kaggle TPU reference engines, opt-in challengers | [docs/perf/](docs/perf/REFERENCE_LOWHANGING_FRUIT_20260919.md) · [glm_tpu/perf/](glm_tpu/perf/) |

For a focused technical review, use the [reviewer guide](docs/release/REVIEWER_GUIDE.md).
The [observability guide](docs/greenfield/GATE_D_OBSERVABILITY_PLAYBOOK.md)
distills the causal-debugging lessons, evidence limits and current tool locations;
the original campaign instructions remain recoverable in Git.

## Project policy

Maintained by Gianluigi Vitale. `main` is the curated private release: every
tracked file has a recorded role in the [curation ledger](docs/curation/README.md),
and research-only material is recoverable at the starting pin. Research continues
on `rewrite/topology-first-decode` and other preserved branches.
Original results and research history remain intact. Weights, credentials,
private questions and raw runtime databases stay outside Git.

[Development](CONTRIBUTING.md) · [Operations](docs/release/OPERATIONS.md) ·
[Security and audit limits](SECURITY.md) · [Third-party notices](THIRD_PARTY_NOTICES.md).
Public distribution still requires an original-code licensing decision and
historical privacy/provenance review; no public-release clearance is implied.
